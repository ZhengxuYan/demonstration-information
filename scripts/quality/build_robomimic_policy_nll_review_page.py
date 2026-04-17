#!/usr/bin/env python3
"""Build a static review page for robomimic policy NLL scores.

The page mirrors the Square MH review layout: each card contains a paired
wrist/agent video and synchronized transition-wise score traces. Here the two
traces are the Transformer-GMM continuous-action NLL and the discrete-binned
policy NLL.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import pickle
import re
import shutil
import statistics as st
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores-root",
        type=Path,
        default=Path("/Users/jasonyan/Desktop/demonstration-information/robomimic_policy_scores/square_mh_no_object"),
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=Path("/Users/jasonyan/Desktop/demonstration-information/square_mh_wrist_agent_paired_deploy"),
    )
    parser.add_argument(
        "--manual-share-root",
        type=Path,
        default=Path("/Users/jasonyan/Desktop/demonstration-information/square_mh_wrist_manual_share_deploy"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/Users/jasonyan/Desktop/demonstration-information/robomimic_policy_nll_review"),
    )
    parser.add_argument("--max-trace-points", type=int, default=260)
    parser.add_argument("--copy-videos", action="store_true", default=True)
    return parser.parse_args()


def load_score_bundle(path: Path, max_trace_points: int) -> dict[str, object]:
    with path.open("rb") as f:
        data = pickle.load(f)
    ep_scores = {int(k): float(v) for k, v in data["ep_idx"].items()}
    sample_score = np.asarray(data["sample_score"], dtype=float)
    sample_ep_idx = np.asarray(data["sample_ep_idx"], dtype=int)
    sample_step_idx = np.asarray(data["sample_step_idx"], dtype=int)

    traces = {}
    for ep_idx in sorted(np.unique(sample_ep_idx).tolist()):
        mask = sample_ep_idx == ep_idx
        steps = sample_step_idx[mask]
        scores = sample_score[mask]
        order = np.argsort(steps)
        steps = steps[order]
        scores = scores[order]
        unique_steps = np.unique(steps)
        mean_scores = np.array([scores[steps == step].mean() for step in unique_steps], dtype=float)
        if unique_steps.size > max_trace_points:
            keep = np.linspace(0, unique_steps.size - 1, max_trace_points).round().astype(int)
            unique_steps = unique_steps[keep]
            mean_scores = mean_scores[keep]
        traces[int(ep_idx)] = {
            "steps": unique_steps.astype(int).tolist(),
            "scores": [round(float(v), 5) for v in mean_scores],
        }
    return {"ep_scores": ep_scores, "traces": traces}


def category_map(manual_share_root: Path) -> dict[int, str]:
    mapping = {}
    for category in ("obs_full", "obs_partial"):
        for path in (manual_share_root / category).glob("*.mp4"):
            match = re.search(r"demo_(\d+)_", path.name)
            if match:
                mapping[int(match.group(1))] = category
    return mapping


def collect_rows(args, gmm, discrete) -> list[dict[str, object]]:
    video_dir = args.video_root / "square_mh_hdf5_paired"
    out_video_dir = args.output_root / "square_mh_hdf5_paired"
    out_video_dir.mkdir(parents=True, exist_ok=True)
    cats = category_map(args.manual_share_root)

    rows = []
    for src in sorted(video_dir.glob("demo_*.mp4"), key=lambda p: int(p.stem.split("_")[-1])):
        ep_idx = int(src.stem.split("_")[-1])
        if ep_idx not in gmm["ep_scores"] or ep_idx not in discrete["ep_scores"]:
            continue
        dst = out_video_dir / src.name
        if args.copy_videos and not dst.exists():
            shutil.copy2(src, dst)
        video_path = dst.relative_to(args.output_root).as_posix() if args.copy_videos else src.as_posix()
        gmm_score = gmm["ep_scores"][ep_idx]
        disc_score = discrete["ep_scores"][ep_idx]
        rows.append(
            {
                "category": cats.get(ep_idx, "unlabeled"),
                "ep_idx": ep_idx,
                "title": f"MH demo_{ep_idx:04d}",
                "video": video_path,
                "gmm_score": gmm_score,
                "discrete_score": disc_score,
                "delta_score": disc_score - gmm_score,
                "gmm_trace": gmm["traces"].get(ep_idx),
                "discrete_trace": discrete["traces"].get(ep_idx),
            }
        )
    rows.sort(key=lambda row: (-float(row["discrete_score"]), int(row["ep_idx"])))
    return rows


def summarize(rows: list[dict[str, object]], category: str) -> dict[str, object]:
    subset = [row for row in rows if category == "all" or row["category"] == category]

    def metric(key: str):
        vals = [float(row[key]) for row in subset if row.get(key) is not None]
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
        "gmm": metric("gmm_score"),
        "discrete": metric("discrete_score"),
    }


def build_html(rows: list[dict[str, object]], summaries: dict[str, object]) -> str:
    payload = html.escape(json.dumps({"rows": rows, "summaries": summaries}, separators=(",", ":")), quote=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Square MH Policy NLL Review</title>
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
    button.active {{ background: var(--ink); color: var(--panel); }}
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
    video {{ display: block; width: 100%; aspect-ratio: 8 / 3; object-fit: contain; background: #050403; }}
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
    .path {{ color: var(--muted); font-size: 12px; word-break: break-all; }}
    .hidden {{ display: none !important; }}
    @media (max-width: 760px) {{
      header, main {{ padding-left: 14px; padding-right: 14px; }}
      .cards {{ grid-template-columns: 1fr; }}
      .summary-grid, .meta {{ grid-template-columns: repeat(2, 1fr); }}
      video {{ aspect-ratio: 4 / 3; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Square MH Policy NLL Review</h1>
    <p class="lede">
      Transformer-GMM and discrete-binned policy scores on Square MH with multi-view images and robot proprioception only.
      Higher values indicate transitions that the learned policy assigns lower probability.
    </p>
    <div class="toolbar">
      <button class="filter active" data-category="all">All</button>
      <button class="filter" data-category="obs_full">MH obs_full</button>
      <button class="filter" data-category="obs_partial">MH obs_partial</button>
      <button class="filter" data-category="unlabeled">Unlabeled</button>
      <select id="sort">
        <option value="disc_desc">Discrete NLL high to low</option>
        <option value="gmm_desc">GMM NLL high to low</option>
        <option value="disc_asc">Discrete NLL low to high</option>
        <option value="gmm_asc">GMM NLL low to high</option>
        <option value="demo">Demo index</option>
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
  <script id="payload" type="application/json">{payload}</script>
  <script>
    const payload = JSON.parse(document.getElementById('payload').textContent);
    let currentCategory = 'all';
    const categoryNames = {{all: 'All', obs_full: 'MH obs_full', obs_partial: 'MH obs_partial', unlabeled: 'Unlabeled'}};
    function fmt(v, digits = 4) {{ return v === null || v === undefined || Number.isNaN(v) ? 'n/a' : Number(v).toFixed(digits); }}
    function statBlock(label, stats) {{ return `<div class="metric"><span>${{label}}</span><strong>${{fmt(stats.mean)}}</strong></div>`; }}
    function renderSummary() {{
      const el = document.getElementById('summary');
      el.innerHTML = Object.entries(payload.summaries).map(([key, s]) => `
        <article class="summary-card">
          <h2>${{categoryNames[key]}}</h2>
          <div class="summary-grid">
            <div class="metric"><span>count</span><strong>${{s.count}}</strong></div>
            ${{statBlock('GMM mean', s.gmm)}}
            ${{statBlock('Discrete mean', s.discrete)}}
          </div>
        </article>
      `).join('');
    }}
    function smoothingWindow() {{
      const raw = Number(document.getElementById('smooth-window').value);
      if (!Number.isFinite(raw) || raw <= 1) return 1;
      return Math.max(1, Math.floor(raw));
    }}
    function smoothTrace(trace, window) {{
      if (!trace || !trace.scores || trace.scores.length <= 2 || window <= 1) return trace;
      const half = Math.floor(window / 2);
      const scores = trace.scores.map((_, i) => {{
        let total = 0, count = 0;
        for (let j = Math.max(0, i - half); j <= Math.min(trace.scores.length - 1, i + half); j++) {{
          total += trace.scores[j]; count += 1;
        }}
        return total / count;
      }});
      return {{steps: trace.steps, scores}};
    }}
    function pathFromTrace(trace, xMin, xMax, yMin, yMax) {{
      if (!trace || !trace.steps || trace.steps.length === 0) return '';
      const left = 40, right = 344, top = 12, bottom = 96;
      const xSpan = Math.max(1, xMax - xMin);
      const ySpan = Math.max(1e-6, yMax - yMin);
      return trace.steps.map((step, i) => {{
        const x = left + ((step - xMin) / xSpan) * (right - left);
        const y = bottom - ((trace.scores[i] - yMin) / ySpan) * (bottom - top);
        return `${{i === 0 ? 'M' : 'L'}} ${{x.toFixed(2)}} ${{y.toFixed(2)}}`;
      }}).join(' ');
    }}
    function plotHtml(trace, kind, label, score) {{
      const smoothed = smoothTrace(trace, smoothingWindow());
      const vals = smoothed?.scores || [];
      const steps = smoothed?.steps || [];
      const yMin = vals.length ? Math.min(...vals) : 0;
      const yMax = vals.length ? Math.max(...vals) : 1;
      const pad = Math.max(1e-6, (yMax - yMin) * 0.08);
      const min = yMin - pad;
      const max = yMax + pad;
      const xMax = steps.length ? Math.max(...steps) : 1;
      const d = pathFromTrace(smoothed, 0, xMax, min, max);
      return `
        <div class="plot">
          <div class="plot-title"><span>${{label}} transition NLL</span><span>trajectory mean ${{fmt(score)}}</span></div>
          <svg viewBox="0 0 380 112" preserveAspectRatio="none">
            <line class="grid" x1="40" y1="12" x2="344" y2="12"></line>
            <line class="grid" x1="40" y1="54" x2="344" y2="54"></line>
            <line class="grid" x1="40" y1="96" x2="344" y2="96"></line>
            <text class="axis-label" x="4" y="16">${{fmt(max, 2)}}</text>
            <text class="axis-label" x="4" y="58">${{fmt((max + min) / 2, 2)}}</text>
            <text class="axis-label" x="4" y="100">${{fmt(min, 2)}}</text>
            <path class="line ${{kind}}-line" d="${{d}}"></path>
            <line class="playhead" x1="40" y1="10" x2="40" y2="100"></line>
          </svg>
        </div>
      `;
    }}
    function cardHtml(row) {{
      return `
        <article class="card" data-category="${{row.category}}" data-demo="${{row.ep_idx}}">
          <div class="video-panel"><span>wrist view | agent view</span><video controls preload="metadata" src="${{row.video}}"></video></div>
          ${{plotHtml(row.gmm_trace, 'gmm', 'GMM', row.gmm_score)}}
          ${{plotHtml(row.discrete_trace, 'disc', 'Discrete', row.discrete_score)}}
          <div class="body">
            <div class="title"><strong>${{row.title}}</strong><span class="pill">${{categoryNames[row.category] || row.category}}</span></div>
            <div class="meta">
              <div><span>GMM NLL</span><strong class="gmm">${{fmt(row.gmm_score)}}</strong></div>
              <div><span>Discrete NLL</span><strong class="disc">${{fmt(row.discrete_score)}}</strong></div>
              <div><span>demo</span><strong>${{row.ep_idx}}</strong></div>
            </div>
            <div class="path">${{row.video}}</div>
          </div>
        </article>
      `;
    }}
    function sortRows(rows) {{
      const mode = document.getElementById('sort').value;
      const sorted = [...rows];
      const num = v => Number.isFinite(Number(v)) ? Number(v) : -Infinity;
      if (mode === 'disc_desc') sorted.sort((a, b) => num(b.discrete_score) - num(a.discrete_score));
      if (mode === 'gmm_desc') sorted.sort((a, b) => num(b.gmm_score) - num(a.gmm_score));
      if (mode === 'disc_asc') sorted.sort((a, b) => num(a.discrete_score) - num(b.discrete_score));
      if (mode === 'gmm_asc') sorted.sort((a, b) => num(a.gmm_score) - num(b.gmm_score));
      if (mode === 'demo') sorted.sort((a, b) => a.ep_idx - b.ep_idx);
      return sorted;
    }}
    function filteredRows() {{
      const search = document.getElementById('search').value.toLowerCase().trim();
      return payload.rows.filter(row => {{
        const catOk = currentCategory === 'all' || row.category === currentCategory;
        const hay = `${{row.title}} ${{row.category}} ${{row.ep_idx}}`.toLowerCase();
        return catOk && (!search || hay.includes(search));
      }});
    }}
    function updatePlayhead(video, card) {{
      const lines = card.querySelectorAll('.playhead');
      const duration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : 0;
      const progress = duration ? Math.min(1, Math.max(0, video.currentTime / duration)) : 0;
      const x = 40 + progress * (344 - 40);
      lines.forEach(line => {{ line.setAttribute('x1', x); line.setAttribute('x2', x); }});
    }}
    function attachSync(card) {{
      const video = card.querySelector('video');
      if (!video) return;
      const update = () => updatePlayhead(video, card);
      video.addEventListener('loadedmetadata', update);
      video.addEventListener('timeupdate', update);
      video.addEventListener('seeked', update);
      update();
    }}
    function renderCards() {{
      const rows = sortRows(filteredRows());
      document.getElementById('result-title').textContent = categoryNames[currentCategory] || currentCategory;
      document.getElementById('result-count').textContent = `${{rows.length}} demos`;
      const cards = document.getElementById('cards');
      cards.innerHTML = rows.map(cardHtml).join('');
      cards.querySelectorAll('.card').forEach(attachSync);
    }}
    document.querySelectorAll('.filter').forEach(button => {{
      button.addEventListener('click', () => {{
        document.querySelectorAll('.filter').forEach(btn => btn.classList.remove('active'));
        button.classList.add('active');
        currentCategory = button.dataset.category;
        renderCards();
      }});
    }});
    document.getElementById('sort').addEventListener('change', renderCards);
    document.getElementById('search').addEventListener('input', renderCards);
    document.getElementById('smooth-window').addEventListener('input', renderCards);
    renderSummary();
    renderCards();
  </script>
</body>
</html>
"""


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    gmm = load_score_bundle(args.scores_root / "gmm_epoch_200.pkl", args.max_trace_points)
    discrete = load_score_bundle(args.scores_root / "discrete_epoch_200.pkl", args.max_trace_points)
    rows = collect_rows(args, gmm, discrete)
    summaries = {
        "all": summarize(rows, "all"),
        "obs_full": summarize(rows, "obs_full"),
        "obs_partial": summarize(rows, "obs_partial"),
        "unlabeled": summarize(rows, "unlabeled"),
    }
    out = args.output_root / "index.html"
    out.write_text(build_html(rows, summaries))
    print(f"wrote {out}")
    print(f"rows: {len(rows)}")


if __name__ == "__main__":
    main()
