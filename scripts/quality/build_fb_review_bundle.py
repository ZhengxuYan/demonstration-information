"""
Build a static review bundle for forward/backward grab demos.

This copies demo mp4s into a share folder and creates an index.html with the
same card layout and synchronized score traces used for the square wrist review.

Example:

python scripts/quality/build_fb_review_bundle.py \
  --source-root /Users/jasonyan/Desktop/demonstration-information/fb_demos \
  --scores-root /Users/jasonyan/Desktop/demonstration-information/fb_demos_scores \
  --output-root /Users/jasonyan/Desktop/demonstration-information/fb_demos_manual_share
"""

from __future__ import annotations

import argparse
import html
import json
import pickle
import shutil
import statistics as st
import subprocess
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--scores-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_scores(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def smooth_scores(scores: np.ndarray, window: int = 9) -> np.ndarray:
    if scores.size <= 2 or window <= 1:
        return scores
    window = min(window, scores.size if scores.size % 2 == 1 else scores.size - 1)
    window = max(window, 3)
    pad = window // 2
    padded = np.pad(scores, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def trace_map(data: dict) -> dict[int, dict[str, object]]:
    sample_score = np.asarray(data["sample_score"])
    sample_ep_idx = np.asarray(data["sample_ep_idx"])
    sample_step_idx = np.asarray(data["sample_step_idx"])

    traces = {}
    for ep_idx in np.unique(sample_ep_idx):
        mask = sample_ep_idx == ep_idx
        steps = sample_step_idx[mask]
        scores = sample_score[mask]
        order = np.argsort(steps)
        steps = steps[order]
        scores = scores[order]
        unique_steps = np.unique(steps)
        mean_scores = np.array([scores[steps == step].mean() for step in unique_steps], dtype=float)
        smooth_mean_scores = smooth_scores(mean_scores)
        traces[int(ep_idx)] = {
            "steps": unique_steps.astype(int).tolist(),
            "scores": smooth_mean_scores.tolist(),
            "min_score": float(smooth_mean_scores.min()),
            "max_score": float(smooth_mean_scores.max()),
            "num_frames": int(unique_steps.max()) + 1,
        }
    return traces


def format_score(score: float) -> str:
    return f"{score:.4f}"


def write_video(video_path: Path, frames: np.ndarray, fps: int = 20) -> int:
    num_frames = len(frames)
    if num_frames == 0:
        raise ValueError(f"No frames found for {video_path.name}")

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg is required to build the FB wrist-view review bundle")

    sample = np.asarray(frames[0], dtype=np.uint8)
    height, width, channels = sample.shape
    if channels != 3:
        raise ValueError(f"Expected RGB frames, got shape {sample.shape}")

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
                raise ValueError(f"Inconsistent frame shape {arr.shape} for {video_path}")
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


def summarize(rows: list[dict[str, object]]) -> dict[str, float]:
    scores = [float(row["pred_score"]) for row in rows]
    frames = [int(row["num_frames"]) for row in rows]
    return {
        "count": len(rows),
        "score_mean": st.mean(scores),
        "score_std": st.stdev(scores) if len(scores) > 1 else 0.0,
        "score_min": min(scores),
        "score_median": st.median(scores),
        "score_max": max(scores),
        "frames_mean": st.mean(frames),
    }


def summary_table(summary: dict[str, float]) -> str:
    return f"""
    <div class="stat-grid">
      <div class="stat"><span>count</span><strong>{summary['count']}</strong></div>
      <div class="stat"><span>mean score</span><strong>{summary['score_mean']:.4f}</strong></div>
      <div class="stat"><span>std</span><strong>{summary['score_std']:.4f}</strong></div>
      <div class="stat"><span>median</span><strong>{summary['score_median']:.4f}</strong></div>
      <div class="stat"><span>min</span><strong>{summary['score_min']:.4f}</strong></div>
      <div class="stat"><span>max</span><strong>{summary['score_max']:.4f}</strong></div>
      <div class="stat"><span>mean frames</span><strong>{summary['frames_mean']:.1f}</strong></div>
    </div>
    """


def card_html(row: dict[str, object]) -> str:
    ep_idx = int(row["ep_idx"])
    pred_score = float(row["pred_score"])
    label = float(row["category_label"])
    num_frames = int(row["num_frames"])
    video_path = html.escape(str(row["video_path"]))
    video_url = html.escape(str(row["video_url"]))
    trace = row["trace"]
    trace_html = f"""
      <div class="trace-wrap">
        <svg class="trace" viewBox="0 0 300 96" preserveAspectRatio="none"
             data-steps='{html.escape(json.dumps(trace["steps"]))}'
             data-scores='{html.escape(json.dumps(trace["scores"]))}'
             data-min-score="{trace["min_score"]:.6f}"
             data-max-score="{trace["max_score"]:.6f}"
             data-num-frames="{num_frames}">
          <line class="trace-axis" x1="42" y1="10" x2="42" y2="86"></line>
          <line class="trace-grid" x1="42" y1="10" x2="292" y2="10"></line>
          <line class="trace-grid" x1="42" y1="48" x2="292" y2="48"></line>
          <line class="trace-grid" x1="42" y1="86" x2="292" y2="86"></line>
          <text class="trace-tick trace-tick-top" x="38" y="14"></text>
          <text class="trace-tick trace-tick-mid" x="38" y="52"></text>
          <text class="trace-tick trace-tick-bottom" x="38" y="90"></text>
          <path class="trace-line"></path>
          <circle class="trace-dot" r="4"></circle>
        </svg>
        <div class="trace-caption">
          <span>step-wise score trace</span>
          <span class="trace-readout">step 0</span>
        </div>
      </div>
    """
    return f"""
    <article class="card" data-score="{pred_score:.6f}" data-ep="{ep_idx}">
      <video controls preload="metadata" src="{video_url}"></video>
      {trace_html}
      <div class="meta">
        <div class="title"><strong>demo_{ep_idx}</strong></div>
        <div>pred score: <span class="accent">{pred_score:.6f}</span></div>
        <div>category label: {label:.1f}</div>
        <div>frames: {num_frames}</div>
        <div class="path">{video_path}</div>
      </div>
    </article>
    """


def section_html(title: str, description: str, rows: list[dict[str, object]]) -> str:
    return f"""
    <section class="section">
      <div class="section-head">
        <div>
          <h2>{html.escape(title)}</h2>
          <p>{html.escape(description)}</p>
        </div>
        {summary_table(summarize(rows))}
      </div>
      <div class="cards">
        {''.join(card_html(row) for row in rows)}
      </div>
    </section>
    """


def build_page(output: Path, forward_rows: list[dict[str, object]], backward_rows: list[dict[str, object]]) -> None:
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>forward/backward grab review</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #efe9dd;
      --panel: #fffaf0;
      --ink: #1f1b17;
      --muted: #6f6a63;
      --border: #d9d0c0;
      --accent: #0f5b5c;
      --accent-soft: #d9ece7;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15, 91, 92, 0.08), transparent 30%),
        linear-gradient(180deg, #f7f1e6 0%, var(--bg) 100%);
      font-family: "Iowan Old Style", "Palatino Linotype", serif;
    }}
    header {{
      padding: 28px 24px 20px;
      border-bottom: 1px solid var(--border);
      background: rgba(255, 250, 240, 0.92);
      position: sticky;
      top: 0;
      z-index: 2;
      backdrop-filter: blur(10px);
    }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    .lede {{
      margin: 0;
      max-width: 900px;
      color: var(--muted);
      line-height: 1.5;
      font-size: 15px;
    }}
    main {{ padding: 24px; display: grid; gap: 28px; }}
    .section {{ display: grid; gap: 16px; }}
    .section-head {{ display: grid; gap: 12px; align-items: start; }}
    h2 {{ margin: 0 0 4px; font-size: 24px; }}
    .section-head p {{ margin: 0; color: var(--muted); max-width: 800px; line-height: 1.5; }}
    .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(110px, 1fr)); gap: 10px; }}
    .stat {{ background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 10px 12px; box-shadow: 0 8px 24px rgba(0, 0, 0, 0.04); }}
    .stat span {{ display: block; font-size: 12px; color: var(--muted); margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.04em; }}
    .stat strong {{ font-size: 18px; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 16px; }}
    .card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 16px; overflow: hidden; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05); }}
    video {{ display: block; width: 100%; background: #000; }}
    .trace-wrap {{ padding: 12px 14px 6px; display: grid; gap: 6px; border-top: 1px solid var(--border); border-bottom: 1px solid rgba(0, 0, 0, 0.04); background: linear-gradient(180deg, rgba(15, 91, 92, 0.03), rgba(15, 91, 92, 0.01)); }}
    .trace {{ width: 100%; height: 96px; overflow: visible; }}
    .trace-axis {{ stroke: rgba(31, 27, 23, 0.45); stroke-width: 1.2; }}
    .trace-grid {{ stroke: rgba(15, 91, 92, 0.12); stroke-width: 1; stroke-dasharray: 3 3; }}
    .trace-tick {{ fill: var(--muted); font-size: 10px; text-anchor: end; font-family: "Iowan Old Style", "Palatino Linotype", serif; }}
    .trace-line {{ fill: none; stroke: var(--accent); stroke-width: 2.5; stroke-linecap: round; stroke-linejoin: round; }}
    .trace-dot {{ fill: #b9472c; stroke: white; stroke-width: 1.5; }}
    .trace-caption {{ display: flex; justify-content: space-between; gap: 8px; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }}
    .trace-readout {{ color: var(--accent); font-weight: 700; }}
    .meta {{ display: grid; gap: 5px; padding: 12px 14px 14px; font-size: 14px; }}
    .title {{ font-size: 16px; }}
    .path {{ color: var(--muted); font-size: 12px; word-break: break-all; }}
    .accent {{ color: var(--accent); font-weight: 700; }}
    .summary-banner {{ background: var(--accent-soft); border: 1px solid rgba(15, 91, 92, 0.15); border-radius: 14px; padding: 14px 16px; line-height: 1.5; }}
  </style>
</head>
<body>
  <header>
    <h1>forward/backward grab review</h1>
    <p class="lede">
      Category-level DemInf scores for the custom forward-grab and backward-grab robosuite demos. Each card shows the demo video, its trajectory score, and a synchronized step-wise score trace.
    </p>
  </header>
  <main>
    <div class="summary-banner">
      <strong>Setup:</strong> scores were computed with the existing square wrist observation VAE and action VAE, then grouped into <code>forward_grab</code> and <code>backward_grab</code>.
    </div>
    {section_html("forward_grab", "Demos where the handle is grasped with the opening facing forward relative to the wrist camera.", forward_rows)}
    {section_html("backward_grab", "Demos where the handle is grasped with the opening facing backward relative to the wrist camera.", backward_rows)}
  </main>
  <script>
    function clamp(value, low, high) {{
      return Math.max(low, Math.min(high, value));
    }}
    function buildTrace(svg) {{
      const steps = JSON.parse(svg.dataset.steps);
      const scores = JSON.parse(svg.dataset.scores);
      const numFrames = Number(svg.dataset.numFrames);
      if (!steps.length || !scores.length) return null;
      const width = 300;
      const height = 96;
      const axisX = 42;
      const padX = axisX;
      const padY = 10;
      const maxStep = Math.max(numFrames - 1, steps[steps.length - 1], 1);
      const minScore = Math.min(...scores);
      const maxScore = Math.max(...scores);
      const midScore = (minScore + maxScore) / 2;
      const scoreSpan = Math.max(maxScore - minScore, 1e-6);
      const xFor = (step) => padX + (step / maxStep) * (width - 2 * padX);
      const yFor = (score) => height - padY - ((score - minScore) / scoreSpan) * (height - 2 * padY);
      svg.querySelector(".trace-tick-top").textContent = maxScore.toFixed(2);
      svg.querySelector(".trace-tick-mid").textContent = midScore.toFixed(2);
      svg.querySelector(".trace-tick-bottom").textContent = minScore.toFixed(2);
      const points = steps.map((step, idx) => [xFor(step), yFor(scores[idx])]);
      svg.querySelector(".trace-line").setAttribute(
        "d",
        points.map((point, idx) => (idx === 0 ? "M " : "L ") + point[0].toFixed(2) + " " + point[1].toFixed(2)).join(" ")
      );
      return {{ steps, scores, xFor, yFor, maxStep }};
    }}
    function interpolateTrace(trace, targetStep) {{
      const steps = trace.steps;
      const scores = trace.scores;
      if (targetStep <= steps[0]) return {{ step: targetStep, score: scores[0] }};
      if (targetStep >= steps[steps.length - 1]) return {{ step: targetStep, score: scores[scores.length - 1] }};
      for (let i = 1; i < steps.length; i++) {{
        if (targetStep <= steps[i]) {{
          const leftStep = steps[i - 1];
          const rightStep = steps[i];
          const alpha = (targetStep - leftStep) / Math.max(rightStep - leftStep, 1e-6);
          const score = scores[i - 1] + alpha * (scores[i] - scores[i - 1]);
          return {{ step: targetStep, score }};
        }}
      }}
      return {{ step: targetStep, score: scores[scores.length - 1] }};
    }}
    document.querySelectorAll(".card").forEach((card) => {{
      const svg = card.querySelector(".trace");
      const video = card.querySelector("video");
      if (!svg || !video) return;
      const trace = buildTrace(svg);
      if (!trace) return;
      const dot = svg.querySelector(".trace-dot");
      const readout = card.querySelector(".trace-readout");
      const update = () => {{
        const frac = video.duration ? clamp(video.currentTime / video.duration, 0, 1) : 0;
        const targetStep = frac * trace.maxStep;
        const point = interpolateTrace(trace, targetStep);
        dot.setAttribute("cx", trace.xFor(point.step).toFixed(2));
        dot.setAttribute("cy", trace.yFor(point.score).toFixed(2));
        readout.textContent = "step " + Math.round(point.step) + " | score " + point.score.toFixed(3);
      }};
      video.addEventListener("loadedmetadata", update);
      video.addEventListener("timeupdate", update);
      video.addEventListener("seeked", update);
      update();
    }});
  </script>
</body>
</html>
"""
    output.write_text(html_doc)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    def first_run_dir(category_root: Path) -> Path:
        run_dirs = sorted(path for path in category_root.iterdir() if path.is_dir())
        if not run_dirs:
            raise ValueError(f"No run directory found under {category_root}")
        return run_dirs[0]

    categories = {
        "forward_grab": {
            "video_dir": first_run_dir(args.source_root / "forward_grab"),
            "hdf5_path": first_run_dir(args.source_root / "forward_grab") / "image.hdf5",
            "score_path": args.scores_root / "forward_grab.pkl",
            "label": 1.0,
        },
        "backward_grab": {
            "video_dir": first_run_dir(args.source_root / "backward_grab"),
            "hdf5_path": first_run_dir(args.source_root / "backward_grab") / "image.hdf5",
            "score_path": args.scores_root / "backward_grab.pkl",
            "label": 0.0,
        },
    }

    rows_by_category: dict[str, list[dict[str, object]]] = {}
    for category, info in categories.items():
        category_out = args.output_root / category
        category_out.mkdir(parents=True, exist_ok=True)

        data = load_scores(info["score_path"])
        traces = trace_map(data)
        rows = []
        with h5py.File(info["hdf5_path"], "r") as f:
            for ep_idx, pred_score in sorted(data["ep_idx"].items(), key=lambda kv: (-kv[1], kv[0])):
                demo_key = f"demo_{int(ep_idx)}"
                if demo_key not in f["data"]:
                    continue
                trace = traces[int(ep_idx)]
                out_name = f"demo_{int(ep_idx):04d}_score_{format_score(float(pred_score))}_label_{int(info['label'])}.mp4"
                dest_video = category_out / out_name
                if args.overwrite or not dest_video.exists():
                    frames = f["data"][demo_key]["obs"]["robot0_eye_in_hand_image"][:]
                    write_video(dest_video, frames)
                rows.append(
                    {
                        "ep_idx": int(ep_idx),
                        "pred_score": float(pred_score),
                        "category_label": float(info["label"]),
                        "num_frames": int(trace["num_frames"]),
                        "video_path": str(dest_video.relative_to(args.output_root)),
                        "video_url": f"{dest_video.relative_to(args.output_root)}?v={int(dest_video.stat().st_mtime_ns)}",
                        "trace": trace,
                    }
                )
        rows_by_category[category] = rows

    build_page(args.output_root / "index.html", rows_by_category["forward_grab"], rows_by_category["backward_grab"])
    print(args.output_root / "index.html")


if __name__ == "__main__":
    main()
