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
    parser.add_argument(
        "--annotations-csv",
        type=Path,
        default=Path("/Users/jasonyan/Desktop/demonstration-information/square_ph_observability_annotations.csv"),
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-demos", type=int, default=0)
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


def load_observability_annotations(path: Path | None) -> dict[int, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    by_ep: dict[int, dict[str, str]] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep_idx = int(row["ep_idx"])
            by_ep[ep_idx] = {
                "observability": (row.get("label") or "").strip() or "unlabeled",
                "annotation_note": (row.get("note") or "").strip(),
            }
    return by_ep


def select_demo_indices(gmm: dict[str, object], discrete: dict[str, object], max_demos: int) -> list[int]:
    gmm_scores = gmm["ep_scores"]
    discrete_scores = discrete["ep_scores"]
    common = sorted(set(gmm_scores) & set(discrete_scores))
    if max_demos <= 0 or len(common) <= max_demos:
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


def export_videos(hdf5_path: Path, output_dir: Path, ep_indices: list[int], fps: int) -> dict[int, dict[str, object]]:
    assets = {}
    with h5py.File(hdf5_path, "r") as f:
        for ep_idx in ep_indices:
            demo = f"demo_{ep_idx}"
            frames = f["data"][demo]["obs"][WRIST_KEY][:]
            rel = Path("videos") / f"demo_{ep_idx:04d}.mp4"
            write_video(output_dir / rel, frames, fps)
            assets[ep_idx] = {
                "video": rel.as_posix(),
                "num_frames": int(len(frames)),
                "fps": fps,
            }
    return assets


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

    def bucket(observability: str | None = None) -> dict[str, object]:
        bucket_rows = rows if observability is None else [row for row in rows if row["observability"] == observability]
        return {
            "count": len(bucket_rows),
            "gmm": metric_from_rows(bucket_rows, "gmm_score"),
            "discrete": metric_from_rows(bucket_rows, "discrete_score"),
        }

    def metric_from_rows(bucket_rows: list[dict[str, object]], key: str) -> dict[str, float | None]:
        vals = [float(row[key]) for row in bucket_rows]
        if not vals:
            return {"mean": None, "std": None, "min": None, "max": None}
        return {
            "mean": float(st.mean(vals)),
            "std": float(st.stdev(vals)) if len(vals) > 1 else 0.0,
            "min": float(min(vals)),
            "max": float(max(vals)),
        }

    return {
        "all": bucket(),
        "full": bucket("full"),
        "partial": bucket("partial"),
        "unlabeled": bucket("unlabeled"),
    }


def write_review_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "ep_idx",
                "observability",
                "annotation_note",
                "gmm_score",
                "discrete_score",
                "video",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "ep_idx": row["ep_idx"],
                    "observability": row["observability"],
                    "annotation_note": row["annotation_note"],
                    "gmm_score": row["gmm_score"],
                    "discrete_score": row["discrete_score"],
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
      --bg: #ece6d8;
      --panel: #fffaf0;
      --ink: #1f1b16;
      --muted: #746b5f;
      --border: #d7cbbb;
      --gmm: #0f6d67;
      --disc: #b54a2a;
      --shadow: rgba(34, 26, 18, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at 8% 0%, rgba(15, 109, 103, .12), transparent 28%),
        radial-gradient(circle at 90% 6%, rgba(181, 74, 42, .12), transparent 24%),
        linear-gradient(180deg, #f8f1e4 0%, var(--bg) 100%);
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 5;
      padding: 22px 24px 18px;
      border-bottom: 1px solid var(--border);
      background: rgba(255, 250, 240, .93);
      backdrop-filter: blur(10px);
    }}
    h1 {{ margin: 0 0 8px; font-size: clamp(25px, 3vw, 36px); letter-spacing: -.03em; }}
    .lede {{ max-width: 1120px; margin: 0; color: var(--muted); line-height: 1.45; }}
    .toolbar {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 16px; }}
    button, select, input {{
      border: 1px solid var(--border);
      background: var(--panel);
      color: var(--ink);
      border-radius: 999px;
      padding: 8px 12px;
      font: inherit;
      box-shadow: 0 4px 14px var(--shadow);
    }}
    main {{ padding: 22px 24px 34px; display: grid; gap: 24px; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; }}
    .summary-card, .card {{
      background: rgba(255, 250, 240, .95);
      border: 1px solid var(--border);
      border-radius: 18px;
      box-shadow: 0 12px 32px var(--shadow);
    }}
    .summary-card {{ padding: 14px 16px; }}
    .summary-card h2 {{ margin: 0 0 8px; font-size: 18px; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(96px, 1fr)); gap: 8px; }}
    .summary-card.full {{ background: rgba(225, 241, 232, .96); }}
    .summary-card.partial {{ background: rgba(245, 231, 218, .96); }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 16px; }}
    .card {{ overflow: hidden; }}
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
    video {{ display: block; width: 100%; aspect-ratio: 4 / 3; object-fit: contain; background: #050403; image-rendering: pixelated; }}
    .plot {{ padding: 12px 14px 8px; border-top: 1px solid var(--border); background: linear-gradient(180deg, rgba(15,109,103,.04), transparent); }}
    .plot-title {{ display: flex; justify-content: space-between; gap: 12px; color: var(--muted); font-size: 12px; margin-bottom: 4px; }}
    svg {{ width: 100%; height: 126px; overflow: visible; display: block; }}
    .gridline {{ stroke: rgba(29,26,22,.12); stroke-width: 1; stroke-dasharray: 4 4; }}
    .zero {{ stroke: rgba(29,26,22,.35); stroke-width: 1.2; }}
    .axis-label {{ fill: var(--muted); font-size: 10px; }}
    .line {{ fill: none; stroke-width: 2.4; stroke-linecap: round; stroke-linejoin: round; }}
    .gmm-line {{ stroke: var(--gmm); }}
    .disc-line {{ stroke: var(--disc); }}
    .playhead {{ stroke: #16120e; stroke-width: 2; opacity: .78; }}
    .legend {{ display: flex; flex-wrap: wrap; justify-content: space-between; gap: 8px 12px; color: var(--muted); font-size: 12px; }}
    .swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 999px; margin-right: 5px; }}
    .swatch.gmm {{ background: var(--gmm); }}
    .swatch.disc {{ background: var(--disc); }}
    .body {{ padding: 12px 14px 15px; display: grid; gap: 9px; }}
    .title {{ display: flex; justify-content: space-between; gap: 10px; align-items: baseline; }}
    .title strong {{ font-size: 17px; }}
    .pill {{ border-radius: 999px; background: #ddeee8; padding: 4px 8px; color: #17433f; font-size: 12px; white-space: nowrap; }}
    .pill.partial {{ background: #f3e1d6; color: #7b3b22; }}
    .pill.unlabeled {{ background: #ece8de; color: #60584e; }}
    .metric {{ border: 1px solid var(--border); border-radius: 12px; padding: 8px; background: #fffdf6; }}
    .metric span, .meta span {{ display: block; color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .055em; }}
    .metric strong, .meta strong {{ font-size: 18px; }}
    .meta {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }}
    .path {{ color: var(--muted); font-size: 12px; word-break: break-all; }}
    .gmm {{ color: var(--gmm); }}
    .disc {{ color: var(--disc); }}
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
    <h1>Square PH BC Policy NLL Review</h1>
    <p class="lede">
      Compares plain BC GMM and discrete-binned policy scores on Square PH with wrist camera and robot proprioception.
      Videos are exported from the Square PH HDF5 wrist camera.
    </p>
    <div class="toolbar">
      <select id="sort">
        <option value="disc_desc">Sort discrete</option>
        <option value="gmm_desc">Sort GMM</option>
        <option value="demo">Sort demo id</option>
      </select>
      <select id="observability-filter">
        <option value="all">All observability labels</option>
        <option value="full">Full only</option>
        <option value="partial">Partial only</option>
        <option value="unlabeled">Unlabeled only</option>
      </select>
      <label>smooth window <input id="smooth-window" type="number" min="1" max="101" step="2" value="9"></label>
      <input id="search" placeholder="Filter demo id">
    </div>
  </header>
  <main>
    <section class="summary" id="summary"></section>
    <section>
      <div class="cards" id="cards"></div>
    </section>
  </main>
  <script id="payload" type="application/json">{payload}</script>
  <script>
    const payload = JSON.parse(document.getElementById('payload').textContent);
    const DATA = payload;
    const fmt = (v) => Number.isFinite(v) ? v.toFixed(3) : 'n/a';
    const cards = document.getElementById('cards');
    const summary = document.getElementById('summary');

    function statBlock(label, stats) {{
      const mean = stats && Number.isFinite(stats.mean) ? stats.mean : NaN;
      return `<div class="metric"><span>${{label}}</span><strong>${{fmt(mean)}}</strong></div>`;
    }}

    function categorySummaryCard(title, cls, stats) {{
      return `
        <article class="summary-card ${{cls}}">
          <h2>${{title}}</h2>
          <div class="summary-grid">
            <div class="metric"><span>count</span><strong>${{stats.count}}</strong></div>
            ${{statBlock('GMM mean', stats.gmm)}}
            ${{statBlock('Discrete mean', stats.discrete)}}
          </div>
        </article>`;
    }}

    function renderSummary(rows) {{
      const all = {{
        count: rows.length,
        gmm: summariseMetric(rows.map(r => r.gmm_score)),
        discrete: summariseMetric(rows.map(r => r.discrete_score)),
      }};
      const fullRows = rows.filter(r => r.observability === 'full');
      const partialRows = rows.filter(r => r.observability === 'partial');
      const unlabeledRows = rows.filter(r => r.observability === 'unlabeled');
      const blocks = [
        categorySummaryCard('Visible demos', '', all),
        categorySummaryCard('Full observability', 'full', {{
          count: fullRows.length,
          gmm: summariseMetric(fullRows.map(r => r.gmm_score)),
          discrete: summariseMetric(fullRows.map(r => r.discrete_score)),
        }}),
        categorySummaryCard('Partial observability', 'partial', {{
          count: partialRows.length,
          gmm: summariseMetric(partialRows.map(r => r.gmm_score)),
          discrete: summariseMetric(partialRows.map(r => r.discrete_score)),
        }}),
      ];
      if (unlabeledRows.length) {{
        blocks.push(categorySummaryCard('Unlabeled', 'unlabeled', {{
          count: unlabeledRows.length,
          gmm: summariseMetric(unlabeledRows.map(r => r.gmm_score)),
          discrete: summariseMetric(unlabeledRows.map(r => r.discrete_score)),
        }}));
      }}
      summary.innerHTML = blocks.join('');
    }}

    function summariseMetric(values) {{
      if (!values.length) return {{mean: NaN}};
      return {{mean: values.reduce((a, b) => a + b, 0) / values.length}};
    }}

    function smoothingWindow() {{
      const raw = Number(document.getElementById('smooth-window').value);
      if (!Number.isFinite(raw) || raw <= 1) return 1;
      return Math.max(1, Math.floor(raw));
    }}

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

    function scoreAtStep(trace, step) {{
      if (!trace || !trace.steps || !trace.scores || trace.steps.length === 0) return NaN;
      const steps = trace.steps;
      const scores = smoothScores(trace.scores, smoothingWindow());
      let bestIdx = 0;
      let bestDist = Math.abs(steps[0] - step);
      for (let i = 1; i < steps.length; i++) {{
        const dist = Math.abs(steps[i] - step);
        if (dist < bestDist) {{
          bestDist = dist;
          bestIdx = i;
        }}
      }}
      return scores[bestIdx];
    }}

    function pathFor(trace, numFrames) {{
      if (!trace || !trace.scores || trace.scores.length === 0) return '';
      const scores = smoothScores(trace.scores, smoothingWindow());
      const steps = trace.steps;
      const left = 36, right = 344, top = 12, bottom = 108;
      const xMin = 0;
      const xMax = Math.max(numFrames - 1, ...steps, 1);
      const min = Math.min(...scores), max = Math.max(...scores);
      const xSpan = Math.max(1, xMax - xMin);
      const ySpan = Math.max(1e-6, max - min);
      return scores.map((s, i) => {{
        const x = left + ((steps[i] - xMin) / xSpan) * (right - left);
        const y = bottom - ((s - min) / ySpan) * (bottom - top);
        return `${{i === 0 ? 'M' : 'L'}} ${{x.toFixed(2)}} ${{y.toFixed(2)}}`;
      }}).join(' ');
    }}

    function plot(trace, cls, label, row, readoutClass, readoutId, initialValue) {{
      if (!trace || !trace.scores || trace.scores.length === 0) return '<div class="plot">No transition trace available.</div>';
      const scores = smoothScores(trace.scores, smoothingWindow());
      const steps = trace.steps || [];
      const min = Math.min(...scores);
      const max = Math.max(...scores);
      const zeroY = 108 - ((0 - min) / Math.max(1e-6, max - min)) * (108 - 12);
      const showZero = zeroY >= 12 && zeroY <= 108;
      const xMax = Math.max(row.num_frames - 1, ...steps, 1);
      return `<div class="plot">
        <div class="plot-title"><span>${{label}}</span><strong id="${{readoutId}}" class="${{readoutClass}}">${{fmt(initialValue)}}</strong></div>
        <svg viewBox="0 0 380 126" preserveAspectRatio="none" aria-label="transition score traces">
          <line class="gridline" x1="36" y1="12" x2="344" y2="12"></line>
          <line class="gridline" x1="36" y1="60" x2="344" y2="60"></line>
          <line class="gridline" x1="36" y1="108" x2="344" y2="108"></line>
          ${{showZero ? `<line class="zero" x1="36" y1="${{zeroY.toFixed(2)}}" x2="344" y2="${{zeroY.toFixed(2)}}"></line>` : ''}}
          <text class="axis-label" x="4" y="16">${{fmt(max)}}</text>
          <text class="axis-label" x="4" y="112">${{fmt(min)}}</text>
          <text class="axis-label" x="36" y="121">0</text>
          <text class="axis-label" x="326" y="121">${{xMax}}</text>
          <path class="line ${{cls}}" d="${{pathFor(trace, row.num_frames)}}"></path>
          <line class="playhead" x1="36" x2="36" y1="12" y2="108"></line>
        </svg>
        <div class="legend">
          <span><span class="swatch ${{cls === 'gmm-line' ? 'gmm' : 'disc'}}"></span>${{label}}</span>
          <span>frame 0 → ${{xMax}}</span>
        </div>
      </div>`;
    }}

    function sortedRows(rows) {{
      const mode = document.getElementById('sort').value;
      const observabilityFilter = document.getElementById('observability-filter').value;
      const search = document.getElementById('search').value.trim().toLowerCase();
      let copy = rows.slice();
      if (observabilityFilter !== 'all') copy = copy.filter(row => row.observability === observabilityFilter);
      if (search) copy = copy.filter(row => String(row.ep_idx).includes(search));
      copy.sort((a, b) => {{
        if (mode === 'disc_desc') return b.discrete_score - a.discrete_score || a.ep_idx - b.ep_idx;
        if (mode === 'gmm_desc') return b.gmm_score - a.gmm_score || a.ep_idx - b.ep_idx;
        return a.ep_idx - b.ep_idx;
      }});
      return copy;
    }}

    function render() {{
      const rows = sortedRows(DATA.rows);
      renderSummary(rows);
      cards.innerHTML = rows.map(row => {{
        const pillClass = row.observability === 'partial' ? 'partial' : (row.observability === 'unlabeled' ? 'unlabeled' : '');
        const noteHtml = row.annotation_note ? `<div class="path">note: ${{row.annotation_note}}</div>` : '';
        return `
        <article class="card" data-ep-idx="${{row.ep_idx}}">
          <div class="video-panel">
            <span>demo_${{String(row.ep_idx).padStart(4, '0')}}</span>
            <video src="${{row.video}}" controls preload="metadata"></video>
          </div>
          ${{plot(row.gmm_trace, 'gmm-line', 'GMM transition NLL', row, 'gmm', `gmm-readout-${{row.ep_idx}}`, row.gmm_score)}}
          ${{plot(row.discrete_trace, 'disc-line', 'Discrete transition NLL', row, 'disc', `disc-readout-${{row.ep_idx}}`, row.discrete_score)}}
          <div class="body">
            <div class="title">
              <strong>demo_${{String(row.ep_idx).padStart(4, '0')}}</strong>
              <span class="pill ${{pillClass}}">${{row.observability}}</span>
            </div>
            <div class="meta">
              <div><span>frames</span><strong>${{row.num_frames}}</strong></div>
              <div><span>GMM mean</span><strong class="gmm">${{fmt(row.gmm_score)}}</strong></div>
              <div><span>Disc mean</span><strong class="disc">${{fmt(row.discrete_score)}}</strong></div>
              <div><span>observability</span><strong>${{row.observability}}</strong></div>
            </div>
            ${{noteHtml}}
          </div>
        </article>`;
      }}).join('');
      syncVideoPlayheads();
    }}

    function frameIndexForVideo(video, row, mediaTime) {{
      const fps = row.fps || 20;
      const sourceTime = Number.isFinite(mediaTime) ? mediaTime : video.currentTime;
      return Math.min(row.num_frames - 1, Math.max(0, Math.round(sourceTime * fps)));
    }}

    function updatePlayhead(card) {{
      const video = card.querySelector('video');
      const row = DATA.rows.find(item => item.ep_idx === Number(card.dataset.epIdx));
      if (!video || !row) return;
      const frameIdx = frameIndexForVideo(video, row);
      const x = 36 + ((frameIdx / Math.max(1, row.num_frames - 1)) * (344 - 36));
      card.querySelectorAll('.playhead').forEach(head => {{
        head.setAttribute('x1', x);
        head.setAttribute('x2', x);
      }});
      const gmmReadout = card.querySelector(`#gmm-readout-${{row.ep_idx}}`);
      const discReadout = card.querySelector(`#disc-readout-${{row.ep_idx}}`);
      if (gmmReadout) gmmReadout.textContent = fmt(scoreAtStep(row.gmm_trace, frameIdx));
      if (discReadout) discReadout.textContent = fmt(scoreAtStep(row.discrete_trace, frameIdx));
    }}

    function syncVideoPlayheads() {{
      document.querySelectorAll('.card').forEach(card => {{
        const video = card.querySelector('video');
        if (!video) return;
        const update = () => updatePlayhead(card);
        video.addEventListener('loadedmetadata', update);
        video.addEventListener('timeupdate', update);
        video.addEventListener('seeked', update);
        update();
      }});
    }}

    document.getElementById('sort').addEventListener('change', render);
    document.getElementById('observability-filter').addEventListener('change', render);
    document.getElementById('search').addEventListener('input', render);
    document.getElementById('smooth-window').addEventListener('change', render);
    render();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gmm = load_score_bundle(args.scores_root / "gmm_bc_epoch_2000_v2.pkl", args.max_trace_points)
    discrete = load_score_bundle(args.scores_root / "discrete_bc_epoch_2000_v2.pkl", args.max_trace_points)
    annotations = load_observability_annotations(args.annotations_csv)
    ep_indices = select_demo_indices(gmm, discrete, args.max_demos)
    videos = export_videos(args.hdf5, args.output_dir, ep_indices, args.fps)

    rows = []
    for ep_idx in ep_indices:
        gmm_score = float(gmm["ep_scores"][ep_idx])
        discrete_score = float(discrete["ep_scores"][ep_idx])
        annotation = annotations.get(ep_idx, {"observability": "unlabeled", "annotation_note": ""})
        rows.append(
            {
                "ep_idx": int(ep_idx),
                "video": videos[ep_idx]["video"],
                "num_frames": videos[ep_idx]["num_frames"],
                "fps": videos[ep_idx]["fps"],
                "observability": annotation["observability"],
                "annotation_note": annotation["annotation_note"],
                "gmm_score": gmm_score,
                "discrete_score": discrete_score,
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
