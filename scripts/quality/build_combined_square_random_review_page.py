"""
Build a local HTML review page for the combined Square MH + random collected scores.

The page compares wrist and third-person score traces for:
- manually grouped Square MH wrist-visibility videos in obs_full
- manually grouped Square MH wrist-visibility videos in obs_partial
- all locally collected random Square/Post demos

Example:

python scripts/quality/build_combined_square_random_review_page.py \
  --share-root /Users/jasonyan/Desktop/demonstration-information/square_mh_wrist_manual_share \
  --scores-root /Users/jasonyan/Desktop/demonstration-information/combined_square_random_scores \
  --manual-video-root /Users/jasonyan/Desktop/demonstration-information/merged
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import pickle
import re
import shutil
import statistics as st
import subprocess
from pathlib import Path

import h5py
import numpy as np


CAMERAS = ("wrist", "agent")
DATASETS = ("square_mh", "random_square_post")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--share-root", type=Path, required=True, help="Folder containing obs_full and obs_partial.")
    parser.add_argument("--scores-root", type=Path, required=True, help="Folder containing wrist/ and agent/ score pkl files.")
    parser.add_argument(
        "--mh-only-both-scores-root",
        type=Path,
        default=None,
        help=(
            "Optional Square-MH-only fused multi-view score root. "
            "Expected to contain both/square_mh.pkl."
        ),
    )
    parser.add_argument("--manual-video-root", type=Path, required=True, help="Folder containing collected demo_*.mp4 files.")
    parser.add_argument(
        "--manual-hdf5",
        type=Path,
        default=None,
        help="Optional collected image.hdf5. Defaults to <manual-video-root>/image.hdf5 when present.",
    )
    parser.add_argument(
        "--manual-camera",
        choices=("wrist", "agent"),
        default="wrist",
        help="Camera stream to export for collected HDF5 videos.",
    )
    parser.add_argument("--manual-fps", type=int, default=20, help="FPS for collected videos exported from HDF5.")
    parser.add_argument(
        "--mh-hdf5",
        type=Path,
        default=None,
        help="Optional Square MH image.hdf5 for exporting paired agent-view videos. Defaults to <share-root>/../image.hdf5 when present.",
    )
    parser.add_argument("--mh-fps", type=int, default=20, help="FPS for Square MH videos exported from HDF5.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML path. Defaults to <share-root>/combined_scores.html.",
    )
    parser.add_argument("--max-trace-points", type=int, default=240, help="Maximum points to embed per trace.")
    return parser.parse_args()


CAMERA_KEY_BY_NAME = {
    "wrist": "robot0_eye_in_hand_image",
    "agent": "agentview_image",
}


def downsample(steps: np.ndarray, scores: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if steps.size <= max_points:
        return steps, scores
    idx = np.linspace(0, steps.size - 1, max_points).round().astype(int)
    return steps[idx], scores[idx]


def load_score_bundle(path: Path, max_trace_points: int) -> dict[str, object]:
    with path.open("rb") as f:
        data = pickle.load(f)

    ep_scores = {int(k): float(v) for k, v in data["ep_idx"].items()}
    sample_score = np.asarray(data["sample_score"], dtype=float)
    sample_ep_idx = np.asarray(data["sample_ep_idx"], dtype=int)
    sample_step_idx = np.asarray(data["sample_step_idx"], dtype=int)

    traces: dict[int, dict[str, object]] = {}
    for ep_idx in sorted(np.unique(sample_ep_idx).tolist()):
        mask = sample_ep_idx == ep_idx
        steps = sample_step_idx[mask]
        scores = sample_score[mask]
        order = np.argsort(steps)
        steps = steps[order]
        scores = scores[order]
        unique_steps = np.unique(steps)
        mean_scores = np.array([scores[steps == step].mean() for step in unique_steps], dtype=float)
        ds_steps, ds_scores = downsample(unique_steps, mean_scores, max_trace_points)
        traces[int(ep_idx)] = {
            "steps": ds_steps.astype(int).tolist(),
            "scores": [round(float(v), 5) for v in ds_scores],
        }

    return {"ep_scores": ep_scores, "traces": traces}


def load_scores(scores_root: Path, max_trace_points: int) -> dict[str, dict[str, dict[str, object]]]:
    scores = {
        camera: {
            dataset: load_score_bundle(scores_root / camera / f"{dataset}.pkl", max_trace_points)
            for dataset in DATASETS
        }
        for camera in CAMERAS
    }
    both_root = scores_root / "both"
    if both_root.exists():
        scores["both"] = {
            dataset: load_score_bundle(both_root / f"{dataset}.pkl", max_trace_points)
            for dataset in DATASETS
            if (both_root / f"{dataset}.pkl").exists()
        }
    return scores


def parse_mh_video(path: Path, category: str, share_root: Path) -> dict[str, object]:
    match = re.search(r"demo_(\d+)_score_([-+0-9.]+)_label_([-+0-9.]+)", path.stem)
    if match is None:
        raise ValueError(f"Unexpected Square MH video filename: {path}")
    ep_idx = int(match.group(1))
    old_score = float(match.group(2))
    label = float(match.group(3))
    return {
        "category": category,
        "dataset": "square_mh",
        "ep_idx": ep_idx,
        "title": f"MH demo_{ep_idx:04d}",
        "video": path.relative_to(share_root).as_posix(),
        "wrist_video": path.relative_to(share_root).as_posix(),
        "agent_video": None,
        "old_score": old_score,
        "label": label,
    }


def parse_manual_video(path: Path, share_root: Path) -> dict[str, object]:
    match = re.fullmatch(r"demo_(\d+)", path.stem)
    if match is None:
        raise ValueError(f"Unexpected collected video filename: {path}")
    ep_idx = int(match.group(1))
    return {
        "category": "manual_collected",
        "dataset": "random_square_post",
        "ep_idx": ep_idx,
        "title": f"collected demo_{ep_idx:03d}",
        "video": os.path.relpath(path, share_root),
        "wrist_video": os.path.relpath(path, share_root),
        "agent_video": None,
        "old_score": None,
        "label": None,
    }


def write_video(video_path: Path, frames: np.ndarray, fps: int) -> int:
    num_frames = len(frames)
    if num_frames == 0:
        raise ValueError(f"No frames found for {video_path}")

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg is required to export exact-match videos from HDF5")

    sample = np.asarray(frames[0], dtype=np.uint8)
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


def export_hdf5_camera_videos(
    hdf5_path: Path,
    share_root: Path,
    output_dir_name: str,
    camera: str,
    fps: int,
    ep_indices: list[int] | None = None,
    name_width: int = 3,
) -> Path:
    camera_key = CAMERA_KEY_BY_NAME[camera]
    output_dir = share_root / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    keep = None if ep_indices is None else set(ep_indices)

    with h5py.File(hdf5_path, "r") as f:
        demo_keys = sorted(f["data"].keys(), key=lambda key: int(key.split("_")[-1]))
        for demo_key in demo_keys:
            ep_idx = int(demo_key.split("_")[-1])
            if keep is not None and ep_idx not in keep:
                continue
            video_path = output_dir / f"demo_{ep_idx:0{name_width}d}.mp4"
            if video_path.exists():
                continue
            frames = f["data"][demo_key]["obs"][camera_key][:]
            write_video(video_path, frames, fps)

    return output_dir


def export_hdf5_paired_videos(
    hdf5_path: Path,
    share_root: Path,
    output_dir_name: str,
    fps: int,
    ep_indices: list[int] | None = None,
    name_width: int = 3,
) -> Path:
    output_dir = share_root / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    keep = None if ep_indices is None else set(ep_indices)

    with h5py.File(hdf5_path, "r") as f:
        demo_keys = sorted(f["data"].keys(), key=lambda key: int(key.split("_")[-1]))
        for demo_key in demo_keys:
            ep_idx = int(demo_key.split("_")[-1])
            if keep is not None and ep_idx not in keep:
                continue
            video_path = output_dir / f"demo_{ep_idx:0{name_width}d}.mp4"
            if video_path.exists():
                continue
            obs = f["data"][demo_key]["obs"]
            wrist = obs[CAMERA_KEY_BY_NAME["wrist"]][:]
            agent = obs[CAMERA_KEY_BY_NAME["agent"]][:]
            n = min(len(wrist), len(agent))
            frames = np.concatenate([wrist[:n], agent[:n]], axis=2)
            write_video(video_path, frames, fps)

    return output_dir


def attach_scores(
    row: dict[str, object],
    scores: dict[str, dict[str, dict[str, object]]],
    mh_only_both_scores: dict[str, object] | None = None,
) -> dict[str, object]:
    dataset = str(row["dataset"])
    ep_idx = int(row["ep_idx"])
    row = dict(row)

    wrist_bundle = scores["wrist"][dataset]
    agent_bundle = scores["agent"][dataset]
    wrist_score = wrist_bundle["ep_scores"].get(ep_idx)  # type: ignore[index]
    agent_score = agent_bundle["ep_scores"].get(ep_idx)  # type: ignore[index]
    wrist_trace = wrist_bundle["traces"].get(ep_idx)  # type: ignore[index]
    agent_trace = agent_bundle["traces"].get(ep_idx)  # type: ignore[index]
    both_bundle = scores.get("both", {}).get(dataset)
    both_score = None if both_bundle is None else both_bundle["ep_scores"].get(ep_idx)  # type: ignore[index]
    both_trace = None if both_bundle is None else both_bundle["traces"].get(ep_idx)  # type: ignore[index]

    mh_only_both_score = None
    mh_only_both_trace = None
    if dataset == "square_mh" and mh_only_both_scores is not None:
        mh_only_both_score = mh_only_both_scores["ep_scores"].get(ep_idx)  # type: ignore[index]
        mh_only_both_trace = mh_only_both_scores["traces"].get(ep_idx)  # type: ignore[index]

    row["wrist_score"] = wrist_score
    row["agent_score"] = agent_score
    row["delta_score"] = None if wrist_score is None or agent_score is None else agent_score - wrist_score
    row["both_score"] = both_score
    row["mh_only_both_score"] = mh_only_both_score
    row["wrist_trace"] = wrist_trace
    row["agent_trace"] = agent_trace
    row["both_trace"] = both_trace
    row["mh_only_both_trace"] = mh_only_both_trace
    row["num_frames"] = max(
        [0]
        + [
            int(max(trace["steps"])) + 1
            for trace in (wrist_trace, agent_trace, both_trace, mh_only_both_trace)
            if trace is not None and trace.get("steps")
        ]
    )
    return row


def collect_rows(
    share_root: Path,
    manual_video_root: Path,
    scores: dict[str, dict[str, dict[str, object]]],
    mh_only_both_scores: dict[str, object] | None = None,
    manual_agent_video_root: Path | None = None,
    manual_paired_video_root: Path | None = None,
    mh_agent_video_root: Path | None = None,
    mh_paired_video_root: Path | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for category in ("obs_full", "obs_partial"):
        for path in sorted((share_root / category).glob("*.mp4")):
            row = parse_mh_video(path, category, share_root)
            if mh_agent_video_root is not None:
                agent_video = mh_agent_video_root / f"demo_{int(row['ep_idx']):04d}.mp4"
                if agent_video.exists():
                    row["agent_video"] = agent_video.relative_to(share_root).as_posix()
            if mh_paired_video_root is not None:
                paired_video = mh_paired_video_root / f"demo_{int(row['ep_idx']):04d}.mp4"
                if paired_video.exists():
                    row["paired_video"] = paired_video.relative_to(share_root).as_posix()
            rows.append(attach_scores(row, scores, mh_only_both_scores=mh_only_both_scores))

    for path in sorted(manual_video_root.glob("demo_*.mp4"), key=lambda p: int(p.stem.split("_")[1])):
        row = parse_manual_video(path, share_root)
        if manual_agent_video_root is not None:
            agent_video = manual_agent_video_root / f"demo_{int(row['ep_idx']):03d}.mp4"
            if agent_video.exists():
                row["agent_video"] = os.path.relpath(agent_video, share_root)
        if manual_paired_video_root is not None:
            paired_video = manual_paired_video_root / f"demo_{int(row['ep_idx']):03d}.mp4"
            if paired_video.exists():
                row["paired_video"] = os.path.relpath(paired_video, share_root)
        rows.append(attach_scores(row, scores, mh_only_both_scores=mh_only_both_scores))

    rows.sort(key=lambda row: (str(row["category"]), -float(row["wrist_score"] or -math.inf), int(row["ep_idx"])))
    return rows


def summarize(rows: list[dict[str, object]], category: str) -> dict[str, object]:
    subset = [row for row in rows if row["category"] == category]
    wrist = [float(row["wrist_score"]) for row in subset if row["wrist_score"] is not None]
    agent = [float(row["agent_score"]) for row in subset if row["agent_score"] is not None]
    both = [float(row["both_score"]) for row in subset if row.get("both_score") is not None]
    mh_only_both = [
        float(row["mh_only_both_score"]) for row in subset if row.get("mh_only_both_score") is not None
    ]
    delta = [float(row["delta_score"]) for row in subset if row["delta_score"] is not None]

    def metric(vals: list[float]) -> dict[str, float | None]:
        if not vals:
            return {"mean": None, "std": None, "min": None, "max": None}
        return {
            "mean": float(st.mean(vals)),
            "std": float(st.stdev(vals)) if len(vals) > 1 else 0.0,
            "min": float(min(vals)),
            "max": float(max(vals)),
        }

    return {
        "count": len(subset),
        "wrist": metric(wrist),
        "agent": metric(agent),
        "both": metric(both),
        "mh_only_both": metric(mh_only_both),
        "delta": metric(delta),
    }


def build_html(rows: list[dict[str, object]], summaries: dict[str, dict[str, object]]) -> str:
    payload = json.dumps({"rows": rows, "summaries": summaries}, separators=(",", ":"))
    escaped_payload = html.escape(payload, quote=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Combined Square DemInf Review</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #ebe5d5;
      --panel: #fffaf0;
      --ink: #1d1a16;
      --muted: #746d63;
      --border: #d7cdbc;
      --wrist: #0f6d67;
      --agent: #b54a2a;
      --both: #315f9c;
      --mh-only-both: #b47a12;
      --soft: #ddeee8;
      --shadow: rgba(32, 25, 18, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at 8% 0%, rgba(15, 109, 103, 0.12), transparent 28%),
        radial-gradient(circle at 92% 8%, rgba(181, 74, 42, 0.10), transparent 24%),
        linear-gradient(180deg, #f8f1e4 0%, var(--bg) 100%);
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 5;
      padding: 22px 24px 18px;
      border-bottom: 1px solid var(--border);
      background: rgba(255, 250, 240, 0.93);
      backdrop-filter: blur(10px);
    }}
    h1 {{ margin: 0 0 8px; font-size: clamp(25px, 3vw, 36px); letter-spacing: -0.03em; }}
    .lede {{ max-width: 1120px; margin: 0; color: var(--muted); line-height: 1.45; }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      margin-top: 16px;
    }}
    button, select, input {{
      border: 1px solid var(--border);
      background: var(--panel);
      color: var(--ink);
      border-radius: 999px;
      padding: 8px 12px;
      font: inherit;
      box-shadow: 0 4px 14px var(--shadow);
    }}
    button.active {{ background: var(--ink); color: var(--panel); }}
    input {{ min-width: 120px; }}
    main {{ padding: 22px 24px 34px; display: grid; gap: 24px; }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 12px;
    }}
    .summary-card, .card {{
      background: rgba(255, 250, 240, 0.95);
      border: 1px solid var(--border);
      border-radius: 18px;
      box-shadow: 0 12px 32px var(--shadow);
    }}
    .summary-card {{ padding: 14px 16px; }}
    .summary-card h2 {{ margin: 0 0 8px; font-size: 18px; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(96px, 1fr)); gap: 8px; }}
    .metric span, .meta span {{ display: block; color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .055em; }}
    .metric strong {{ font-size: 18px; }}
    .section-title {{ display: flex; align-items: baseline; justify-content: space-between; gap: 14px; }}
    .section-title h2 {{ margin: 0; font-size: 24px; }}
    .section-title p {{ margin: 0; color: var(--muted); }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 16px; }}
    .card {{ overflow: hidden; }}
    .videos {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1px; background: #050403; }}
    .videos.paired {{ grid-template-columns: 1fr; }}
    .video-panel {{ position: relative; background: #050403; }}
    .video-panel span {{
      position: absolute;
      left: 8px;
      top: 8px;
      z-index: 1;
      border-radius: 999px;
      padding: 3px 7px;
      background: rgba(5, 4, 3, .68);
      color: #fffaf0;
      font-size: 12px;
      letter-spacing: .03em;
    }}
    video {{ display: block; width: 100%; aspect-ratio: 4 / 3; object-fit: contain; background: #050403; }}
    .plot {{ padding: 12px 14px 8px; border-top: 1px solid var(--border); background: linear-gradient(180deg, rgba(15,109,103,.04), transparent); }}
    svg {{ width: 100%; height: 126px; overflow: visible; display: block; }}
    .grid {{ stroke: rgba(29,26,22,.12); stroke-width: 1; stroke-dasharray: 4 4; }}
    .zero {{ stroke: rgba(29,26,22,.35); stroke-width: 1.2; }}
    .axis-label {{ fill: var(--muted); font-size: 10px; }}
    .line {{ fill: none; stroke-width: 2.4; stroke-linecap: round; stroke-linejoin: round; }}
    .wrist-line {{ stroke: var(--wrist); }}
    .agent-line {{ stroke: var(--agent); }}
    .both-line {{ stroke: var(--both); }}
    .mh-only-both-line {{ stroke: var(--mh-only-both); stroke-dasharray: 5 4; }}
    .playhead {{ stroke: #16120e; stroke-width: 2; opacity: .78; }}
    .legend {{ display: flex; flex-wrap: wrap; justify-content: space-between; gap: 8px 12px; color: var(--muted); font-size: 12px; }}
    .swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 999px; margin-right: 5px; }}
    .swatch.wrist {{ background: var(--wrist); }}
    .swatch.agent {{ background: var(--agent); }}
    .swatch.both {{ background: var(--both); }}
    .swatch.mh-only-both {{ background: var(--mh-only-both); }}
    .body {{ padding: 12px 14px 15px; display: grid; gap: 9px; }}
    .title {{ display: flex; justify-content: space-between; gap: 10px; align-items: baseline; }}
    .title strong {{ font-size: 17px; }}
    .pill {{ border-radius: 999px; background: var(--soft); padding: 4px 8px; color: #17433f; font-size: 12px; white-space: nowrap; }}
    .meta {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }}
    .meta strong {{ font-size: 17px; }}
    .wrist {{ color: var(--wrist); }}
    .agent {{ color: var(--agent); }}
    .both {{ color: var(--both); }}
    .mh-only-both {{ color: var(--mh-only-both); }}
    .path {{ color: var(--muted); font-size: 12px; word-break: break-all; }}
    .hidden {{ display: none !important; }}
    @media (max-width: 700px) {{
      header, main {{ padding-left: 14px; padding-right: 14px; }}
      .cards {{ grid-template-columns: 1fr; }}
      .summary-grid, .meta {{ grid-template-columns: repeat(2, 1fr); }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Combined Square DemInf Review</h1>
    <p class="lede">
      Three categories are shown: Square MH demos where the wrist view fully observes the opening,
      Square MH demos where it is partially occluded, and the manually collected random-init demos.
      Each card shows transition-level DemInf score traces for wrist-only, third-person-only,
      and fused wrist+third-person checkpoints. Square MH cards also include the MH-only fused checkpoint.
    </p>
    <div class="toolbar">
      <button class="filter active" data-category="all">All</button>
      <button class="filter" data-category="obs_full">MH obs_full</button>
      <button class="filter" data-category="obs_partial">MH obs_partial</button>
      <button class="filter" data-category="manual_collected">Collected</button>
      <select id="sort">
        <option value="category">Sort by category</option>
        <option value="wrist_desc">Wrist score high to low</option>
        <option value="agent_desc">Agent score high to low</option>
        <option value="both_desc">Both-view score high to low</option>
        <option value="mh_only_both_desc">MH-only both high to low</option>
        <option value="delta_desc">Agent - wrist high to low</option>
        <option value="demo">Demo index</option>
      </select>
      <select id="manual-speed">
        <option value="1">Collected speed: exact HDF5 20 fps</option>
        <option value="0.5">Collected speed: 0.5x</option>
        <option value="0.25">Collected speed: 0.25x</option>
      </select>
      <label>smooth window <input id="smooth-window" type="number" min="1" max="101" step="2" value="9"></label>
      <input id="search" placeholder="Filter demo id or category">
    </div>
  </header>
  <main>
    <section class="summary" id="summary"></section>
    <section>
      <div class="section-title">
        <h2 id="result-title">All samples</h2>
        <p id="result-count"></p>
      </div>
      <div class="cards" id="cards"></div>
    </section>
  </main>
  <script id="payload" type="application/json">{escaped_payload}</script>
  <script>
    const payload = JSON.parse(document.getElementById('payload').textContent);
    let currentCategory = 'all';

    const categoryNames = {{
      obs_full: 'MH obs_full',
      obs_partial: 'MH obs_partial',
      manual_collected: 'Collected random-init'
    }};

    function fmt(v, digits = 4) {{
      return v === null || v === undefined || Number.isNaN(v) ? 'n/a' : Number(v).toFixed(digits);
    }}

    function statBlock(label, stats) {{
      return `<div class="metric"><span>${{label}}</span><strong>${{fmt(stats.mean)}}</strong></div>`;
    }}

    function renderSummary() {{
      const el = document.getElementById('summary');
      el.innerHTML = Object.entries(payload.summaries).map(([key, s]) => `
        <article class="summary-card">
          <h2>${{categoryNames[key]}}</h2>
          <div class="summary-grid">
            <div class="metric"><span>count</span><strong>${{s.count}}</strong></div>
            ${{statBlock('wrist mean', s.wrist)}}
            ${{statBlock('agent mean', s.agent)}}
            ${{statBlock('both mean', s.both)}}
            ${{statBlock('MH-only both', s.mh_only_both)}}
            ${{statBlock('delta mean', s.delta)}}
          </div>
        </article>
      `).join('');
    }}

    function pathFromTrace(trace, xMin, xMax, yMin, yMax) {{
      if (!trace || !trace.steps || trace.steps.length === 0) return '';
      const left = 36, right = 344, top = 12, bottom = 108;
      const xSpan = Math.max(1, xMax - xMin);
      const ySpan = Math.max(1e-6, yMax - yMin);
      return trace.steps.map((step, i) => {{
        const x = left + ((step - xMin) / xSpan) * (right - left);
        const y = bottom - ((trace.scores[i] - yMin) / ySpan) * (bottom - top);
        return `${{i === 0 ? 'M' : 'L'}} ${{x.toFixed(2)}} ${{y.toFixed(2)}}`;
      }}).join(' ');
    }}

    function smoothingWindow() {{
      const raw = Number(document.getElementById('smooth-window').value);
      if (!Number.isFinite(raw) || raw <= 1) return 1;
      return Math.max(1, Math.floor(raw));
    }}

    function smoothTrace(trace, window) {{
      if (!trace || !trace.scores || trace.scores.length <= 2 || window <= 1) return trace;
      const half = Math.floor(window / 2);
      const scores = trace.scores.map((_, idx) => {{
        const start = Math.max(0, idx - half);
        const end = Math.min(trace.scores.length, idx + half + 1);
        let total = 0;
        for (let i = start; i < end; i++) total += trace.scores[i];
        return total / (end - start);
      }});
      return {{steps: trace.steps, scores}};
    }}

    function slopePer100(trace) {{
      if (!trace || trace.steps.length < 2) return null;
      const n = trace.steps.length;
      const meanX = trace.steps.reduce((a, b) => a + b, 0) / n;
      const meanY = trace.scores.reduce((a, b) => a + b, 0) / n;
      let num = 0, den = 0;
      for (let i = 0; i < n; i++) {{
        const dx = trace.steps[i] - meanX;
        num += dx * (trace.scores[i] - meanY);
        den += dx * dx;
      }}
      return den <= 0 ? null : (num / den) * 100;
    }}

    function traceSvg(row) {{
      const window = smoothingWindow();
      const wristTrace = smoothTrace(row.wrist_trace, window);
      const agentTrace = smoothTrace(row.agent_trace, window);
      const bothTrace = smoothTrace(row.both_trace, window);
      const mhOnlyBothTrace = smoothTrace(row.mh_only_both_trace, window);
      const traces = [wristTrace, agentTrace, bothTrace, mhOnlyBothTrace].filter(Boolean);
      if (traces.length === 0) return '<div class="plot">No transition trace available.</div>';
      const allSteps = traces.flatMap(t => t.steps);
      const allScores = traces.flatMap(t => t.scores);
      const xMin = 0, xMax = Math.max(row.num_frames - 1, ...allSteps, 1);
      let yMin = Math.min(...allScores), yMax = Math.max(...allScores);
      if (yMin === yMax) {{ yMin -= 0.1; yMax += 0.1; }}
      const zeroY = 108 - ((0 - yMin) / Math.max(1e-6, yMax - yMin)) * (108 - 12);
      const showZero = zeroY >= 12 && zeroY <= 108;
      return `
        <div class="plot">
          <svg viewBox="0 0 380 126" preserveAspectRatio="none" aria-label="transition score traces">
            <line class="grid" x1="36" y1="12" x2="344" y2="12"></line>
            <line class="grid" x1="36" y1="60" x2="344" y2="60"></line>
            <line class="grid" x1="36" y1="108" x2="344" y2="108"></line>
            ${{showZero ? `<line class="zero" x1="36" y1="${{zeroY.toFixed(2)}}" x2="344" y2="${{zeroY.toFixed(2)}}"></line>` : ''}}
            <text class="axis-label" x="4" y="16">${{fmt(yMax, 2)}}</text>
            <text class="axis-label" x="4" y="112">${{fmt(yMin, 2)}}</text>
            <path class="line wrist-line" d="${{pathFromTrace(wristTrace, xMin, xMax, yMin, yMax)}}"></path>
            <path class="line agent-line" d="${{pathFromTrace(agentTrace, xMin, xMax, yMin, yMax)}}"></path>
            <path class="line both-line" d="${{pathFromTrace(bothTrace, xMin, xMax, yMin, yMax)}}"></path>
            <path class="line mh-only-both-line" d="${{pathFromTrace(mhOnlyBothTrace, xMin, xMax, yMin, yMax)}}"></path>
            <line class="playhead" x1="36" y1="8" x2="36" y2="112"></line>
          </svg>
          <div class="legend">
            <span><span class="swatch wrist"></span>wrist trace, slope/100: <strong>${{fmt(slopePer100(wristTrace), 3)}}</strong></span>
            <span><span class="swatch agent"></span>agent trace, slope/100: <strong>${{fmt(slopePer100(agentTrace), 3)}}</strong></span>
            <span><span class="swatch both"></span>both-view, slope/100: <strong>${{fmt(slopePer100(bothTrace), 3)}}</strong></span>
            ${{mhOnlyBothTrace ? `<span><span class="swatch mh-only-both"></span>MH-only both, slope/100: <strong>${{fmt(slopePer100(mhOnlyBothTrace), 3)}}</strong></span>` : ''}}
          </div>
        </div>`;
    }}

    function card(row) {{
      const oldScore = row.old_score === null ? '' : `<div class="metric"><span>old filename score</span><strong>${{fmt(row.old_score)}}</strong></div>`;
      const agentVideo = row.agent_video ? `
            <div class="video-panel"><span>agent view</span><video controls preload="metadata" src="${{row.agent_video}}" data-category="${{row.category}}" data-role="agent"></video></div>
          ` : `
            <div class="video-panel"><span>agent view unavailable</span><video controls preload="metadata" data-category="${{row.category}}" data-role="agent"></video></div>
          `;
      const videoHtml = row.paired_video ? `
          <div class="videos paired">
            <div class="video-panel"><span>wrist view | agent view</span><video controls preload="metadata" src="${{row.paired_video}}" data-category="${{row.category}}" data-role="wrist"></video></div>
          </div>
        ` : `
          <div class="videos">
            <div class="video-panel"><span>wrist view</span><video controls preload="metadata" src="${{row.wrist_video || row.video}}" data-category="${{row.category}}" data-role="wrist"></video></div>
            ${{agentVideo}}
          </div>
        `;
      return `
        <article class="card" data-category="${{row.category}}" data-demo="${{row.ep_idx}}">
          ${{videoHtml}}
          ${{traceSvg(row)}}
          <div class="body">
            <div class="title"><strong>${{row.title}}</strong><span class="pill">${{categoryNames[row.category]}}</span></div>
            <div class="meta">
              <div class="metric"><span>wrist score</span><strong class="wrist">${{fmt(row.wrist_score)}}</strong></div>
              <div class="metric"><span>agent score</span><strong class="agent">${{fmt(row.agent_score)}}</strong></div>
              <div class="metric"><span>both-view score</span><strong class="both">${{fmt(row.both_score)}}</strong></div>
              <div class="metric"><span>MH-only both</span><strong class="mh-only-both">${{fmt(row.mh_only_both_score)}}</strong></div>
              <div class="metric"><span>agent - wrist</span><strong>${{fmt(row.delta_score)}}</strong></div>
              ${{oldScore}}
              <div class="metric"><span>label</span><strong>${{row.label === null ? 'n/a' : fmt(row.label, 1)}}</strong></div>
              <div class="metric"><span>dataset ep</span><strong>${{row.ep_idx}}</strong></div>
            </div>
            <div class="path">wrist: ${{row.wrist_video || row.video}}</div>
            <div class="path">agent: ${{row.agent_video || 'not exported'}}</div>
            <div class="path">paired: ${{row.paired_video || 'not exported'}}</div>
          </div>
        </article>`;
    }}

    function sortedRows(rows) {{
      const mode = document.getElementById('sort').value;
      const copy = [...rows];
      const num = (row, key) => row[key] ?? -Infinity;
      if (mode === 'wrist_desc') copy.sort((a, b) => num(b, 'wrist_score') - num(a, 'wrist_score'));
      else if (mode === 'agent_desc') copy.sort((a, b) => num(b, 'agent_score') - num(a, 'agent_score'));
      else if (mode === 'both_desc') copy.sort((a, b) => num(b, 'both_score') - num(a, 'both_score'));
      else if (mode === 'mh_only_both_desc') copy.sort((a, b) => num(b, 'mh_only_both_score') - num(a, 'mh_only_both_score'));
      else if (mode === 'delta_desc') copy.sort((a, b) => num(b, 'delta_score') - num(a, 'delta_score'));
      else if (mode === 'demo') copy.sort((a, b) => a.ep_idx - b.ep_idx || a.category.localeCompare(b.category));
      else copy.sort((a, b) => a.category.localeCompare(b.category) || num(b, 'wrist_score') - num(a, 'wrist_score'));
      return copy;
    }}

    function renderCards() {{
      const query = document.getElementById('search').value.toLowerCase().trim();
      let rows = payload.rows.filter(row => currentCategory === 'all' || row.category === currentCategory);
      if (query) {{
        rows = rows.filter(row => `${{row.title}} ${{row.category}} ${{row.ep_idx}}`.toLowerCase().includes(query));
      }}
      rows = sortedRows(rows);
      document.getElementById('cards').innerHTML = rows.map(card).join('');
      document.getElementById('result-title').textContent = currentCategory === 'all' ? 'All samples' : categoryNames[currentCategory];
      document.getElementById('result-count').textContent = `${{rows.length}} visible cards`;
      attachVideoSync();
    }}

    function manualPlaybackRate() {{
      return Number(document.getElementById('manual-speed').value);
    }}

    function updatePlayhead(cardEl) {{
      const video = cardEl.querySelector('video[data-role="wrist"]') || cardEl.querySelector('video');
      const playhead = cardEl.querySelector('.playhead');
      if (!video || !playhead) return;
      const duration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : 0;
      const progress = duration ? Math.min(1, Math.max(0, video.currentTime / duration)) : 0;
      const x = 36 + progress * (344 - 36);
      playhead.setAttribute('x1', x.toFixed(2));
      playhead.setAttribute('x2', x.toFixed(2));
    }}

    function syncLoop(video, cardEl) {{
      updatePlayhead(cardEl);
      if (!video.paused && !video.ended) {{
        if (video.requestVideoFrameCallback) {{
          video.requestVideoFrameCallback(() => syncLoop(video, cardEl));
        }} else {{
          requestAnimationFrame(() => syncLoop(video, cardEl));
        }}
      }}
    }}

    function applyPlaybackRate(video) {{
      video.playbackRate = video.dataset.category === 'manual_collected' ? manualPlaybackRate() : 1;
    }}

    function attachVideoSync() {{
      document.querySelectorAll('.card').forEach(cardEl => {{
        const videos = Array.from(cardEl.querySelectorAll('video')).filter(video => video.currentSrc || video.getAttribute('src'));
        if (!videos.length) return;
        videos.forEach(applyPlaybackRate);
        let syncing = false;

        const syncPeers = source => {{
          if (syncing) return;
          syncing = true;
          videos.forEach(peer => {{
            if (peer === source) return;
            if (Number.isFinite(source.currentTime) && Math.abs(peer.currentTime - source.currentTime) > 0.08) {{
              peer.currentTime = Math.min(source.currentTime, Number.isFinite(peer.duration) ? peer.duration : source.currentTime);
            }}
          }});
          syncing = false;
        }};

        videos.forEach(video => {{
          applyPlaybackRate(video);
          video.addEventListener('loadedmetadata', () => {{
            applyPlaybackRate(video);
            updatePlayhead(cardEl);
          }});
          video.addEventListener('play', () => {{
            syncPeers(video);
            videos.forEach(peer => {{
              if (peer !== video && peer.paused) peer.play().catch(() => {{}});
            }});
            syncLoop(video, cardEl);
          }});
          video.addEventListener('pause', () => {{
            videos.forEach(peer => {{
              if (peer !== video && !peer.paused) peer.pause();
            }});
            updatePlayhead(cardEl);
          }});
          video.addEventListener('seeked', () => {{
            syncPeers(video);
            updatePlayhead(cardEl);
          }});
          video.addEventListener('timeupdate', () => {{
            syncPeers(video);
            updatePlayhead(cardEl);
          }});
        }});
        updatePlayhead(cardEl);
      }});
    }}

    document.querySelectorAll('.filter').forEach(button => {{
      button.addEventListener('click', () => {{
        currentCategory = button.dataset.category;
        document.querySelectorAll('.filter').forEach(b => b.classList.toggle('active', b === button));
        renderCards();
      }});
    }});
    document.getElementById('sort').addEventListener('change', renderCards);
    document.getElementById('search').addEventListener('input', renderCards);
    document.getElementById('smooth-window').addEventListener('change', renderCards);
    document.getElementById('manual-speed').addEventListener('change', () => {{
      document.querySelectorAll('video[data-category="manual_collected"]').forEach(applyPlaybackRate);
    }});

    renderSummary();
    renderCards();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    output = args.output or args.share_root / "combined_scores.html"
    manual_video_root = args.manual_video_root
    manual_agent_video_root = None
    manual_paired_video_root = None
    manual_hdf5 = args.manual_hdf5
    if manual_hdf5 is None:
        candidate = args.manual_video_root / "image.hdf5"
        manual_hdf5 = candidate if candidate.exists() else None
    if manual_hdf5 is not None:
        manual_video_root = export_hdf5_camera_videos(
            hdf5_path=manual_hdf5,
            share_root=args.share_root,
            output_dir_name="manual_collected_hdf5_wrist",
            camera="wrist",
            fps=args.manual_fps,
            name_width=3,
        )
        manual_agent_video_root = export_hdf5_camera_videos(
            hdf5_path=manual_hdf5,
            share_root=args.share_root,
            output_dir_name="manual_collected_hdf5_agent",
            camera="agent",
            fps=args.manual_fps,
            name_width=3,
        )
        manual_paired_video_root = export_hdf5_paired_videos(
            hdf5_path=manual_hdf5,
            share_root=args.share_root,
            output_dir_name="manual_collected_hdf5_paired",
            fps=args.manual_fps,
            name_width=3,
        )
    else:
        existing_wrist = args.share_root / "manual_collected_hdf5_wrist"
        existing_agent = args.share_root / "manual_collected_hdf5_agent"
        existing_paired = args.share_root / "manual_collected_hdf5_paired"
        if existing_wrist.exists():
            manual_video_root = existing_wrist
        if existing_agent.exists():
            manual_agent_video_root = existing_agent
        if existing_paired.exists():
            manual_paired_video_root = existing_paired

    mh_agent_video_root = None
    mh_paired_video_root = None
    mh_hdf5 = args.mh_hdf5
    if mh_hdf5 is None:
        candidate = args.share_root.parent / "image.hdf5"
        mh_hdf5 = candidate if candidate.exists() else None
    if mh_hdf5 is not None:
        selected_mh_eps = []
        for category in ("obs_full", "obs_partial"):
            for path in sorted((args.share_root / category).glob("*.mp4")):
                selected_mh_eps.append(int(path.stem.split("_")[1]))
        mh_agent_video_root = export_hdf5_camera_videos(
            hdf5_path=mh_hdf5,
            share_root=args.share_root,
            output_dir_name="square_mh_hdf5_agent",
            camera="agent",
            fps=args.mh_fps,
            ep_indices=selected_mh_eps,
            name_width=4,
        )
        mh_paired_video_root = export_hdf5_paired_videos(
            hdf5_path=mh_hdf5,
            share_root=args.share_root,
            output_dir_name="square_mh_hdf5_paired",
            fps=args.mh_fps,
            ep_indices=selected_mh_eps,
            name_width=4,
        )
    else:
        existing_agent = args.share_root / "square_mh_hdf5_agent"
        existing_paired = args.share_root / "square_mh_hdf5_paired"
        if existing_agent.exists():
            mh_agent_video_root = existing_agent
        if existing_paired.exists():
            mh_paired_video_root = existing_paired

    scores = load_scores(args.scores_root, args.max_trace_points)
    mh_only_both_scores = None
    if args.mh_only_both_scores_root is not None:
        mh_only_path = args.mh_only_both_scores_root / "both" / "square_mh.pkl"
        if mh_only_path.exists():
            mh_only_both_scores = load_score_bundle(mh_only_path, args.max_trace_points)
    rows = collect_rows(
        args.share_root,
        manual_video_root,
        scores,
        mh_only_both_scores=mh_only_both_scores,
        manual_agent_video_root=manual_agent_video_root,
        manual_paired_video_root=manual_paired_video_root,
        mh_agent_video_root=mh_agent_video_root,
        mh_paired_video_root=mh_paired_video_root,
    )
    summaries = {category: summarize(rows, category) for category in ("obs_full", "obs_partial", "manual_collected")}
    output.write_text(build_html(rows, summaries), encoding="utf-8")
    print(f"Wrote {output}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
