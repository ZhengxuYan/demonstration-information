"""
Export scored RoboMimic demos as review videos plus metadata.

Example:

python scripts/quality/export_scored_demos.py \
    --scores /Users/jasonyan/Desktop/demonstration-information/square_mh_wrist_ksg_seed1/square_mh.pkl \
    --hdf5 /path/to/image.hdf5 \
    --output /Users/jasonyan/Desktop/demonstration-information/square_mh_wrist_review
"""

from __future__ import annotations

import argparse
import csv
import html
import pickle
import shutil
import subprocess
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np


CAMERA_KEY_BY_NAME = {
    "wrist": "robot0_eye_in_hand_image",
    "agent": "agentview_image",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True, help="Path to a quality-estimation .pkl file.")
    parser.add_argument("--hdf5", type=Path, required=True, help="Path to the RoboMimic image.hdf5 file.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for videos and metadata.")
    parser.add_argument(
        "--camera",
        choices=sorted(CAMERA_KEY_BY_NAME),
        default="wrist",
        help="Which RoboMimic camera stream to export.",
    )
    parser.add_argument("--fps", type=int, default=20, help="Frames per second for exported MP4s.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing MP4 files instead of reusing them.",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Skip generating the HTML review index.",
    )
    return parser.parse_args()


def load_scores(path: Path) -> tuple[dict[int, float], dict[int, float]]:
    with path.open("rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a dict in {path}, found {type(data)!r}")
    if "ep_idx" not in data or "quality_by_ep_idx" not in data:
        raise ValueError(f"{path} is missing expected keys 'ep_idx' and 'quality_by_ep_idx'")

    pred_by_ep_idx = data["ep_idx"]
    label_by_ep_idx = data["quality_by_ep_idx"]
    if not isinstance(pred_by_ep_idx, dict) or not isinstance(label_by_ep_idx, dict):
        raise ValueError("Expected both 'ep_idx' and 'quality_by_ep_idx' to be dicts")
    return pred_by_ep_idx, label_by_ep_idx


def format_score(score: float) -> str:
    return f"{score:+0.4f}".replace("+", "")


def output_name(ep_idx: int, pred_score: float, human_label: float) -> str:
    return f"demo_{ep_idx:04d}_score_{format_score(pred_score)}_label_{int(human_label)}.mp4"


def write_video(video_path: Path, frames, fps: int) -> int:
    num_frames = len(frames)
    if num_frames == 0:
        raise ValueError(f"No frames found for {video_path.name}")

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is not None:
        sample = np.asarray(frames[0])
        if sample.ndim != 3 or sample.shape[2] != 3:
            raise ValueError(f"Expected RGB frames shaped (H, W, 3), got {sample.shape}")
        height, width, _ = sample.shape
        cmd = [
            ffmpeg_path,
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            assert proc.stdin is not None
            for frame in frames:
                arr = np.asarray(frame, dtype=np.uint8)
                if arr.shape != sample.shape:
                    raise ValueError(f"Inconsistent frame shape for {video_path.name}: {arr.shape} vs {sample.shape}")
                proc.stdin.write(arr.tobytes())
            proc.stdin.close()
            return_code = proc.wait()
        except Exception:
            proc.kill()
            proc.wait()
            raise
        if return_code != 0:
            stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr is not None else ""
            raise RuntimeError(f"ffmpeg failed for {video_path}: {stderr}")
        return num_frames

    with imageio.get_writer(video_path, fps=fps, codec="libx264") as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame))
    return num_frames


def write_manifest(manifest_path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["ep_idx", "pred_score", "human_label", "video_path", "num_frames"]
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_html(output_path: Path, rows: list[dict[str, object]], camera_name: str) -> None:
    body = []
    for row in rows:
        video_path = html.escape(str(row["video_path"]))
        ep_idx = int(row["ep_idx"])
        pred_score = float(row["pred_score"])
        human_label = row["human_label"]
        num_frames = row["num_frames"]
        body.append(
            f"""
            <article class="card" data-score="{pred_score:.6f}" data-ep="{ep_idx}">
              <video controls preload="metadata" src="{video_path}"></video>
              <div class="meta">
                <div><strong>demo_{ep_idx}</strong></div>
                <div>pred score: {pred_score:.6f}</div>
                <div>human label: {human_label}</div>
                <div>frames: {num_frames}</div>
                <div class="path">{video_path}</div>
              </div>
            </article>
            """
        )

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>square/mh {camera_name} review</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f2efe8;
      --panel: #fffdf7;
      --ink: #1f1f1f;
      --muted: #6c6a66;
      --border: #d9d2c3;
      --accent: #245b4a;
    }}
    body {{
      margin: 0;
      background: linear-gradient(180deg, #f6f2e8 0%, #ece7d9 100%);
      color: var(--ink);
      font-family: "Iowan Old Style", "Palatino Linotype", serif;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 1;
      padding: 16px 20px;
      background: rgba(246, 242, 232, 0.95);
      border-bottom: 1px solid var(--border);
      backdrop-filter: blur(10px);
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 24px;
    }}
    .sub {{
      color: var(--muted);
      font-size: 14px;
    }}
    main {{
      padding: 18px;
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
      gap: 16px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      overflow: hidden;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
    }}
    video {{
      display: block;
      width: 100%;
      background: #000;
    }}
    .meta {{
      padding: 12px 14px 14px;
      display: grid;
      gap: 4px;
      font-size: 14px;
    }}
    .path {{
      color: var(--muted);
      font-size: 12px;
      word-break: break-all;
    }}
    .accent {{
      color: var(--accent);
    }}
  </style>
</head>
<body>
  <header>
    <h1>square/mh review <span class="accent">({camera_name})</span></h1>
    <div class="sub">Sorted by predicted score descending. Each card links a demo video with its score and human label.</div>
  </header>
  <main>
    {"".join(body)}
  </main>
</body>
</html>
"""
    output_path.write_text(html_doc)


def main() -> None:
    args = parse_args()
    pred_by_ep_idx, label_by_ep_idx = load_scores(args.scores)

    output_dir = args.output
    videos_dir = output_dir / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    camera_key = CAMERA_KEY_BY_NAME[args.camera]
    rows = []

    with h5py.File(args.hdf5, "r") as f:
        for ep_idx in sorted(pred_by_ep_idx):
            demo_key = f"demo_{ep_idx}"
            if demo_key not in f["data"]:
                raise KeyError(f"{demo_key} not found in {args.hdf5}")

            pred_score = float(pred_by_ep_idx[ep_idx])
            human_label = float(label_by_ep_idx.get(ep_idx, float("nan")))
            file_name = output_name(ep_idx, pred_score, human_label)
            video_path = videos_dir / file_name

            frames = f["data"][demo_key]["obs"][camera_key]
            if args.overwrite or not video_path.exists():
                num_frames = write_video(video_path, frames, args.fps)
            else:
                num_frames = len(frames)

            rows.append(
                {
                    "ep_idx": ep_idx,
                    "pred_score": pred_score,
                    "human_label": human_label,
                    "video_path": str(video_path.relative_to(output_dir)),
                    "num_frames": num_frames,
                }
            )

    rows.sort(key=lambda row: (-float(row["pred_score"]), int(row["ep_idx"])))
    write_manifest(output_dir / "manifest.csv", rows)
    if not args.no_html:
        write_html(output_dir / "index.html", rows, args.camera)

    print(f"Exported {len(rows)} demos to {output_dir}")


if __name__ == "__main__":
    main()
