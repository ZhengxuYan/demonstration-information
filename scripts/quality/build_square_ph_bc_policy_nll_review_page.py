#!/usr/bin/env python3
"""Build a static review page for Square PH plain-BC policy NLL scores.

Example:

python scripts/quality/build_square_ph_bc_policy_nll_review_page.py \
  --scores-root /Users/jasonyan/Desktop/demonstration-information/robomimic_policy_scores/square_ph_bc_wrist_proprio \
  --hdf5 /Users/jasonyan/Desktop/demonstration-information/robomimic_square_ph/image.hdf5 \
  --output-dir /Users/jasonyan/Desktop/demonstration-information/square_ph_bc_policy_nll_review
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import pickle
import shutil
import statistics as st
import subprocess
from pathlib import Path

import h5py
import numpy as np


WRIST_KEY = "robot0_eye_in_hand_image"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores-root",
        type=Path,
        default=Path("/Users/jasonyan/Desktop/demonstration-information/robomimic_policy_scores/square_ph_bc_wrist_proprio"),
    )
    parser.add_argument(
        "--hdf5",
        type=Path,
        default=Path("/Users/jasonyan/Desktop/demonstration-information/robomimic_square_ph/image.hdf5"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/jasonyan/Desktop/demonstration-information/square_ph_bc_policy_nll_review"),
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-demos", type=int, default=60)
    parser.add_argument("--max-trace-points", type=int, default=220)
    return parser.parse_args()


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


def select_demo_indices(gmm: dict[str, object], discrete: dict[str, object], max_demos: int) -> list[int]:
    gmm_scores = gmm["ep_scores"]
    discrete_scores = discrete["ep_scores"]
    common = sorted(set(gmm_scores) & set(discrete_scores))
    if len(common) <= max_demos:
        return common

    def quantile_pick(values: list[tuple[int, float]], count: int) -> list[int]:
        values = sorted(values, key=lambda item: item[1])
        idxs = np.linspace(0, len(values) - 1, count).round().astype(int)
        return [values[int(idx)][0] for idx in idxs]

    per_bucket = max(5, max_demos // 4)
    picked: list[int] = []
    picked.extend(quantile_pick([(i, float(gmm_scores[i])) for i in common], per_bucket))
    picked.extend(quantile_pick([(i, float(discrete_scores[i])) for i in common], per_bucket))
    picked.extend(quantile_pick([(i, float(discrete_scores[i]) - float(gmm_scores[i])) for i in common], per_bucket))
    picked.extend(quantile_pick([(i, abs(float(discrete_scores[i]) - float(gmm_scores[i]))) for i in common], per_bucket))

    deduped = []
    seen = set()
    for idx in picked:
        if idx not in seen:
            deduped.append(idx)
            seen.add(idx)
    return deduped[:max_demos]


def write_video(video_path: Path, frames: np.ndarray, fps: int) -> None:
    video_path.parent.mkdir(parents=True, exist_ok=True)
    if video_path.exists():
        return

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to export HDF5 videos")

    frames = np.asarray(frames, dtype=np.uint8)
    height, width = frames.shape[1], frames.shape[2]
    cmd = [
        ffmpeg,
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
        "-vf",
        "scale=336:336:flags=neighbor",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(video_path),
    ]
    proc = subprocess.run(cmd, input=frames.tobytes(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="replace"))


def export_videos(hdf5_path: Path, output_dir: Path, ep_indices: list[int], fps: int) -> dict[int, str]:
    rel_paths = {}
    with h5py.File(hdf5_path, "r") as f:
        for ep_idx in ep_indices:
            demo = f"demo_{ep_idx}"
            frames = f["data"][demo]["obs"][WRIST_KEY][:]
            rel = Path("videos") / f"demo_{ep_idx:04d}.mp4"
            write_video(output_dir / rel, frames, fps)
            rel_paths[ep_idx] = rel.as_posix()
    return rel_paths


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    def metric(key: str) -> dict[str, float | None]:
        vals = [float(row[key]) for row in rows]
        if not vals:
            return {"mean": None, "std": None, "min": None, "max": None}
        return {
            "mean": float(st.mean(vals)),
            "std": float(st.stdev(vals)) if len(vals) > 1 else 0.0,
            "min": float(min(vals)),
            "max": float(max(vals)),
        }

    return {
        "count": len(rows),
        "gmm": metric("gmm_score"),
        "discrete": metric("discrete_score"),
    }


def write_review_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["ep_idx", "gmm_score", "discrete_score", "delta_score", "video"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "ep_idx": row["ep_idx"],
                    "gmm_score": row["gmm_score"],
                    "discrete_score": row["discrete_score"],
                    "delta_score": row["delta_score"],
                    "video": row["video"],
                }
            )


def build_html(rows: list[dict[str, object]], summary: dict[str, object]) -> str:
    payload = html.escape(json.dumps({"rows": rows, "summary": summary}, separators=(",", ":")), quote=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Square PH BC Policy NLL Review</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #e7ece7;
      --panel: #fbfcf7;
      --ink: #171918;
      --muted: #67706b;
      --border: #ccd7cf;
      --gmm: #0f6d67;
      --disc: #b54a2a;
      --soft: #dcece7;
      --shadow: rgba(17, 24, 20, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: linear-gradient(180deg, #f7faf5 0%, var(--bg) 100%);
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 5;
      padding: 22px 24px 18px;
      border-bottom: 1px solid var(--border);
      background: rgba(251, 252, 247, 0.94);
      backdrop-filter: blur(10px);
    }}
    h1 {{ margin: 0 0 8px; font-size: clamp(25px, 3vw, 36px); letter-spacing: 0; }}
    .lede {{ max-width: 1120px; margin: 0; color: var(--muted); line-height: 1.45; }}
    .toolbar {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-top: 16px; }}
    button, select, input {{
      border: 1px solid var(--border);
      background: var(--panel);
      color: var(--ink);
      border-radius: 8px;
      padding: 8px 12px;
      font: inherit;
      box-shadow: 0 4px 14px var(--shadow);
    }}
    main {{ padding: 22px 24px 34px; display: grid; gap: 24px; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; }}
    .summary-card, .card {{
      background: rgba(251, 252, 247, 0.96);
      border: 1px solid var(--border);
      border-radius: 8px;
      box-shadow: 0 12px 32px var(--shadow);
    }}
    .summary-card {{ padding: 14px 16px; }}
    .summary-card h2 {{ margin: 0 0 8px; font-size: 18px; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }}
    .metric span, .meta span {{ display: block; color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .055em; }}
    .metric strong, .meta strong {{ font-size: 18px; }}
    .section-title {{ display: flex; align-items: baseline; justify-content: space-between; gap: 14px; }}
    .section-title h2 {{ margin: 0; font-size: 24px; }}
    .section-title p {{ margin: 0; color: var(--muted); }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(390px, 1fr)); gap: 16px; }}
    .card {{ overflow: hidden; }}
    .video-panel {{ position: relative; background: #050403; }}
    .video-panel span {{
      position: absolute;
      left: 8px;
      top: 8px;
      z-index: 1;
      border-radius: 8px;
      padding: 3px 7px;
      background: rgba(5, 4, 3, .68);
      color: #fffaf0;
      font-size: 12px;
      letter-spacing: .03em;
    }}
    video {{ display: block; width: 100%; aspect-ratio: 1 / 1; object-fit: contain; background: #050403; }}
    .plot {{ padding: 12px 14px 8px; border-top: 1px solid var(--border); }}
    .plot-title {{ display: flex; justify-content: space-between; gap: 12px; color: var(--muted); font-size: 12px; margin-bottom: 4px; }}
    svg {{ width: 100%; height: 116px; overflow: visible; display: block; }}
    .grid {{ stroke: rgba(23,25,24,.13); stroke-width: 1; stroke-dasharray: 4 4; }}
    .axis-label {{ fill: var(--muted); font-size: 10px; }}
    .line {{ fill: none; stroke-width: 2.4; stroke-linecap: round; stroke-linejoin: round; }}
    .gmm-line {{ stroke: var(--gmm); }}
    .disc-line {{ stroke: var(--disc); }}
    .playhead {{ stroke: #16120e; stroke-width: 2; opacity: .78; }}
    .legend {{ display: flex; justify-content: space-between; gap: 10px; color: var(--muted); font-size: 12px; }}
    .body {{ padding: 12px 14px 15px; display: grid; gap: 9px; }}
    .title {{ display: flex; justify-content: space-between; gap: 10px; align-items: baseline; }}
    .title strong {{ font-size: 17px; }}
    .pill {{ border-radius: 8px; background: var(--soft); padding: 4px 8px; color: #17433f; font-size: 12px; white-space: nowrap; }}
    .meta {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }}
    .gmm {{ color: var(--gmm); }}
    .disc {{ color: var(--disc); }}
    .hidden {{ display: none !important; }}
    @media (max-width: 760px) {{
      header, main {{ padding-left: 14px; padding-right: 14px; }}
      .cards {{ grid-template-columns: 1fr; }}
      .summary-grid, .meta {{ grid-template-columns: repeat(2, 1fr); }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Square PH BC Policy NLL Review</h1>
    <p class="lede">
      Plain BC GMM and discrete-binned policy scores on Square PH with wrist camera and robot proprioception.
      Higher values indicate transitions assigned lower probability by the learned policy.
    </p>
    <div class="toolbar">
      <label>Sort
        <select id="sort">
          <option value="disc_desc">Discrete high to low</option>
          <option value="disc_asc">Discrete low to high</option>
          <option value="gmm_desc">GMM high to low</option>
          <option value="gmm_asc">GMM low to high</option>
          <option value="delta_desc">Delta high to low</option>
          <option value="ep_idx">Demo index</option>
        </select>
      </label>
      <label>Smooth
        <input id="smooth" type="range" min="1" max="31" step="2" value="9">
      </label>
      <span id="count"></span>
    </div>
  </header>
  <main>
    <section class="summary" id="summary"></section>
    <section>
      <div class="section-title">
        <h2>Selected Demonstrations</h2>
        <p>Cards are sampled across score quantiles; all trajectory scores are in the CSV files.</p>
      </div>
      <div class="cards" id="cards"></div>
    </section>
  </main>
  <script id="payload" type="application/json">{payload}</script>
  <script>
    const payload = JSON.parse(document.getElementById('payload').textContent);
    let rows = payload.rows.slice();
    let smoothWindow = 9;

    const fmt = (v) => Number.isFinite(v) ? v.toFixed(3) : 'n/a';
    const count = document.getElementById('count');
    const cards = document.getElementById('cards');
    const sort = document.getElementById('sort');
    const smooth = document.getElementById('smooth');
    const summary = document.getElementById('summary');

    function smoothScores(values, win) {{
      if (win <= 1 || values.length < 3) return values.slice();
      const half = Math.floor(win / 2);
      return values.map((_, i) => {{
        const start = Math.max(0, i - half);
        const end = Math.min(values.length, i + half + 1);
        const slice = values.slice(start, end);
        return slice.reduce((a, b) => a + b, 0) / slice.length;
      }});
    }}

    function pathFor(trace) {{
      if (!trace || !trace.scores || trace.scores.length === 0) return '';
      const scores = smoothScores(trace.scores, smoothWindow);
      const w = 360, h = 86, left = 38, right = 8, top = 8, bottom = 18;
      const min = Math.min(...scores), max = Math.max(...scores);
      const span = Math.max(1e-6, max - min);
      return scores.map((s, i) => {{
        const x = left + (scores.length === 1 ? 0 : i * (w - left - right) / (scores.length - 1));
        const y = top + (max - s) * (h - top - bottom) / span;
        return `${{i === 0 ? 'M' : 'L'}}${{x.toFixed(2)}} ${{y.toFixed(2)}}`;
      }}).join(' ');
    }}

    function plot(trace, cls, label, score) {{
      const scores = trace && trace.scores ? smoothScores(trace.scores, smoothWindow) : [];
      const min = scores.length ? Math.min(...scores) : 0;
      const max = scores.length ? Math.max(...scores) : 1;
      return `<div class="plot">
        <div class="plot-title"><span>${{label}}</span><strong>${{fmt(score)}}</strong></div>
        <svg viewBox="0 0 360 96" preserveAspectRatio="none">
          <line class="grid" x1="38" x2="352" y1="8" y2="8"></line>
          <line class="grid" x1="38" x2="352" y1="68" y2="68"></line>
          <text class="axis-label" x="2" y="12">${{fmt(max)}}</text>
          <text class="axis-label" x="2" y="72">${{fmt(min)}}</text>
          <path class="line ${{cls}}" d="${{pathFor(trace)}}"></path>
          <line class="playhead" x1="38" x2="38" y1="5" y2="72"></line>
        </svg>
        <div class="legend"><span>start</span><span>end</span></div>
      </div>`;
    }}

    function renderSummary() {{
      const s = payload.summary;
      summary.innerHTML = `
        <div class="summary-card">
          <h2>Selected demos</h2>
          <div class="summary-grid">
            <div class="metric"><span>count</span><strong>${{s.count}}</strong></div>
            <div class="metric"><span>GMM mean</span><strong class="gmm">${{fmt(s.gmm.mean)}}</strong></div>
            <div class="metric"><span>Disc mean</span><strong class="disc">${{fmt(s.discrete.mean)}}</strong></div>
          </div>
        </div>
        <div class="summary-card">
          <h2>GMM NLL</h2>
          <div class="summary-grid">
            <div class="metric"><span>min</span><strong>${{fmt(s.gmm.min)}}</strong></div>
            <div class="metric"><span>std</span><strong>${{fmt(s.gmm.std)}}</strong></div>
            <div class="metric"><span>max</span><strong>${{fmt(s.gmm.max)}}</strong></div>
          </div>
        </div>
        <div class="summary-card">
          <h2>Discrete NLL</h2>
          <div class="summary-grid">
            <div class="metric"><span>min</span><strong>${{fmt(s.discrete.min)}}</strong></div>
            <div class="metric"><span>std</span><strong>${{fmt(s.discrete.std)}}</strong></div>
            <div class="metric"><span>max</span><strong>${{fmt(s.discrete.max)}}</strong></div>
          </div>
        </div>`;
    }}

    function sortRows() {{
      const mode = sort.value;
      rows.sort((a, b) => {{
        if (mode === 'disc_desc') return b.discrete_score - a.discrete_score || a.ep_idx - b.ep_idx;
        if (mode === 'disc_asc') return a.discrete_score - b.discrete_score || a.ep_idx - b.ep_idx;
        if (mode === 'gmm_desc') return b.gmm_score - a.gmm_score || a.ep_idx - b.ep_idx;
        if (mode === 'gmm_asc') return a.gmm_score - b.gmm_score || a.ep_idx - b.ep_idx;
        if (mode === 'delta_desc') return b.delta_score - a.delta_score || a.ep_idx - b.ep_idx;
        return a.ep_idx - b.ep_idx;
      }});
    }}

    function renderCards() {{
      sortRows();
      count.textContent = `${{rows.length}} demos`;
      cards.innerHTML = rows.map(row => `
        <article class="card">
          <div class="video-panel">
            <span>demo_${{String(row.ep_idx).padStart(4, '0')}}</span>
            <video src="${{row.video}}" controls preload="metadata"></video>
          </div>
          ${{plot(row.gmm_trace, 'gmm-line', 'GMM transition NLL', row.gmm_score)}}
          ${{plot(row.discrete_trace, 'disc-line', 'Discrete transition NLL', row.discrete_score)}}
          <div class="body">
            <div class="title">
              <strong>Square PH demo_${{String(row.ep_idx).padStart(4, '0')}}</strong>
              <span class="pill">plain BC</span>
            </div>
            <div class="meta">
              <div><span>GMM mean</span><strong class="gmm">${{fmt(row.gmm_score)}}</strong></div>
              <div><span>Disc mean</span><strong class="disc">${{fmt(row.discrete_score)}}</strong></div>
              <div><span>Delta</span><strong>${{fmt(row.delta_score)}}</strong></div>
            </div>
          </div>
        </article>`).join('');
      syncVideoPlayheads();
    }}

    function syncVideoPlayheads() {{
      document.querySelectorAll('.card').forEach(card => {{
        const video = card.querySelector('video');
        const heads = card.querySelectorAll('.playhead');
        video.addEventListener('timeupdate', () => {{
          const ratio = video.duration ? video.currentTime / video.duration : 0;
          const x = 38 + ratio * (352 - 38);
          heads.forEach(head => {{
            head.setAttribute('x1', x);
            head.setAttribute('x2', x);
          }});
        }});
      }});
    }}

    sort.addEventListener('change', renderCards);
    smooth.addEventListener('input', () => {{
      smoothWindow = Number(smooth.value);
      renderCards();
    }});
    renderSummary();
    renderCards();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gmm = load_score_bundle(args.scores_root / "gmm_bc_epoch_200.pkl", args.max_trace_points)
    discrete = load_score_bundle(args.scores_root / "discrete_bc_epoch_200.pkl", args.max_trace_points)
    ep_indices = select_demo_indices(gmm, discrete, args.max_demos)
    videos = export_videos(args.hdf5, args.output_dir, ep_indices, args.fps)

    rows = []
    for ep_idx in ep_indices:
        gmm_score = float(gmm["ep_scores"][ep_idx])
        discrete_score = float(discrete["ep_scores"][ep_idx])
        rows.append(
            {
                "ep_idx": int(ep_idx),
                "video": videos[ep_idx],
                "gmm_score": gmm_score,
                "discrete_score": discrete_score,
                "delta_score": discrete_score - gmm_score,
                "gmm_trace": gmm["traces"].get(ep_idx, {}),
                "discrete_trace": discrete["traces"].get(ep_idx, {}),
            }
        )
    rows.sort(key=lambda row: (-float(row["discrete_score"]), int(row["ep_idx"])))

    summary = summarize(rows)
    write_review_csv(rows, args.output_dir / "robomimic_policy_nll_reviews.csv")
    (args.output_dir / "index.html").write_text(build_html(rows, summary))
    print(f"wrote {args.output_dir / 'index.html'}")
    print(f"wrote {args.output_dir / 'robomimic_policy_nll_reviews.csv'}")


if __name__ == "__main__":
    main()
