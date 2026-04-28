#!/usr/bin/env python3
"""Build a static report for Square PH policy-view BC NLL experiments."""

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


SPECS = [
    {
        "key": "gmm_agent_wrist",
        "algo": "GMM",
        "view": "agent_wrist",
        "label": "GMM / agent+wrist",
        "score_file": "gmm_agent_wrist_epoch_2000.pkl",
        "primary_image": "agentview_image",
        "dataset": "square_ph_agent_wrist_image.hdf5",
    },
    {
        "key": "gmm_left_close_low_wrist",
        "algo": "GMM",
        "view": "left_close_low_wrist",
        "label": "GMM / left-close-low+wrist",
        "score_file": "gmm_left_close_low_wrist_epoch_2000.pkl",
        "primary_image": "left_close_low_image",
        "dataset": "square_ph_left_close_low_wrist_image.hdf5",
    },
    {
        "key": "discrete_agent_wrist",
        "algo": "Discrete",
        "view": "agent_wrist",
        "label": "Discrete / agent+wrist",
        "score_file": "discrete_agent_wrist_epoch_2000.pkl",
        "primary_image": "agentview_image",
        "dataset": "square_ph_agent_wrist_image.hdf5",
    },
    {
        "key": "discrete_left_close_low_wrist",
        "algo": "Discrete",
        "view": "left_close_low_wrist",
        "label": "Discrete / left-close-low+wrist",
        "score_file": "discrete_left_close_low_wrist_epoch_2000.pkl",
        "primary_image": "left_close_low_image",
        "dataset": "square_ph_left_close_low_wrist_image.hdf5",
    },
]

COLORS = {
    "gmm_agent_wrist": "#0f6d67",
    "gmm_left_close_low_wrist": "#56a38f",
    "discrete_agent_wrist": "#b54a2a",
    "discrete_left_close_low_wrist": "#d78a40",
}

WRIST_KEY = "robot0_eye_in_hand_image"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores-root",
        type=Path,
        default=Path("/iris/u/jasonyan/data/robomimic_policy_scores/square_ph_policy_view_bc"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/iris/u/jasonyan/data/policy_view_experiments/square_ph"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("square_ph_policy_view_bc_report"),
    )
    parser.add_argument(
        "--annotations-csv",
        type=Path,
        default=Path("square_ph_observability_annotations.csv"),
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
        means = np.array([scores[steps == step].mean() for step in unique_steps], dtype=float)
        ds_steps, ds_scores = downsample(unique_steps, means, max_trace_points)
        traces[int(ep_idx)] = {
            "steps": ds_steps.astype(int).tolist(),
            "scores": [round(float(v), 5) for v in ds_scores],
        }
    return {"ep_scores": ep_scores, "traces": traces}


def load_annotations(path: Path) -> dict[int, dict[str, str]]:
    if not path.exists():
        return {}
    out: dict[int, dict[str, str]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            ep_idx = int(row["ep_idx"])
            out[ep_idx] = {
                "observability": (row.get("label") or "").strip() or "unlabeled",
                "annotation_note": (row.get("note") or "").strip(),
            }
    return out


def metric(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(st.mean(values)),
        "std": float(st.stdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def select_ep_indices(bundles: dict[str, dict[str, object]], max_demos: int) -> list[int]:
    common = set.intersection(*(set(bundle["ep_scores"].keys()) for bundle in bundles.values()))
    ep_indices = sorted(common)
    if max_demos <= 0 or len(ep_indices) <= max_demos:
        return ep_indices
    score_items = [(ep, max(float(bundle["ep_scores"][ep]) for bundle in bundles.values())) for ep in ep_indices]
    ordered = sorted(score_items, key=lambda item: item[1])
    picks = np.linspace(0, len(ordered) - 1, max_demos).round().astype(int)
    return sorted({ordered[int(i)][0] for i in picks})


def write_video(path: Path, frames: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to export report videos")
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
        "scale=672:336:flags=neighbor",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    proc = subprocess.run(cmd, input=frames.tobytes(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="replace"))


def export_view_videos(data_root: Path, output_dir: Path, ep_indices: list[int], fps: int) -> dict[str, dict[int, dict[str, object]]]:
    assets: dict[str, dict[int, dict[str, object]]] = {}
    for spec in (SPECS[0], SPECS[1]):
        view = spec["view"]
        primary = spec["primary_image"]
        dataset_path = data_root / spec["dataset"]
        assets[view] = {}
        with h5py.File(dataset_path, "r") as f:
            for ep_idx in ep_indices:
                demo = f["data"][f"demo_{ep_idx}"]
                left = demo["obs"][primary][:]
                right = demo["obs"][WRIST_KEY][:]
                frames = np.concatenate([left, right], axis=2)
                rel = Path("videos") / view / f"demo_{ep_idx:04d}.mp4"
                write_video(output_dir / rel, frames, fps)
                assets[view][ep_idx] = {
                    "video": rel.as_posix(),
                    "num_frames": int(len(frames)),
                    "fps": fps,
                }
    return assets


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, object]]:
    bundles = {spec["key"]: load_score_bundle(args.scores_root / spec["score_file"], args.max_trace_points) for spec in SPECS}
    ep_indices = select_ep_indices(bundles, args.max_demos)
    videos = export_view_videos(args.data_root, args.output_dir, ep_indices, args.fps)
    annotations = load_annotations(args.annotations_csv)
    rows: list[dict[str, object]] = []
    for ep_idx in ep_indices:
        annotation = annotations.get(ep_idx, {"observability": "unlabeled", "annotation_note": ""})
        scores = {spec["key"]: float(bundles[spec["key"]]["ep_scores"][ep_idx]) for spec in SPECS}
        traces = {spec["key"]: bundles[spec["key"]]["traces"].get(ep_idx, {}) for spec in SPECS}
        rows.append(
            {
                "ep_idx": int(ep_idx),
                "observability": annotation["observability"],
                "annotation_note": annotation["annotation_note"],
                "videos": {
                    "agent_wrist": videos["agent_wrist"][ep_idx],
                    "left_close_low_wrist": videos["left_close_low_wrist"][ep_idx],
                },
                "scores": scores,
                "traces": traces,
                "gmm_gap": scores["gmm_left_close_low_wrist"] - scores["gmm_agent_wrist"],
                "discrete_gap": scores["discrete_left_close_low_wrist"] - scores["discrete_agent_wrist"],
                "agent_policy_gap": scores["discrete_agent_wrist"] - scores["gmm_agent_wrist"],
                "left_policy_gap": scores["discrete_left_close_low_wrist"] - scores["gmm_left_close_low_wrist"],
            }
        )
    summary = {
        spec["key"]: {
            "label": spec["label"],
            "algo": spec["algo"],
            "view": spec["view"],
            "color": COLORS[spec["key"]],
            "dataset": str(args.data_root / spec["dataset"]),
            "score_file": str(args.scores_root / spec["score_file"]),
            "nll": metric([float(row["scores"][spec["key"]]) for row in rows]),
        }
        for spec in SPECS
    }
    return rows, summary


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="") as f:
        fieldnames = [
            "ep_idx",
            "observability",
            "gmm_agent_wrist",
            "gmm_left_close_low_wrist",
            "discrete_agent_wrist",
            "discrete_left_close_low_wrist",
            "gmm_view_gap",
            "discrete_view_gap",
            "agent_policy_gap",
            "left_policy_gap",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "ep_idx": row["ep_idx"],
                    "observability": row["observability"],
                    "gmm_agent_wrist": row["scores"]["gmm_agent_wrist"],
                    "gmm_left_close_low_wrist": row["scores"]["gmm_left_close_low_wrist"],
                    "discrete_agent_wrist": row["scores"]["discrete_agent_wrist"],
                    "discrete_left_close_low_wrist": row["scores"]["discrete_left_close_low_wrist"],
                    "gmm_view_gap": row["gmm_gap"],
                    "discrete_view_gap": row["discrete_gap"],
                    "agent_policy_gap": row["agent_policy_gap"],
                    "left_policy_gap": row["left_policy_gap"],
                }
            )


def build_html(rows: list[dict[str, object]], summary: dict[str, object]) -> str:
    payload = html.escape(
        json.dumps({"rows": rows, "summary": summary, "specs": SPECS, "colors": COLORS}, separators=(",", ":")),
        quote=False,
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Square PH Policy-View BC Report</title>
  <style>
    :root {{ --bg:#efe7d8; --panel:#fffaf0; --ink:#211b15; --muted:#746b5f; --border:#d8cbbb; --shadow:rgba(33,27,21,.09); }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; color:var(--ink); background:radial-gradient(circle at 10% 0%,rgba(15,109,103,.14),transparent 28%),radial-gradient(circle at 92% 8%,rgba(181,74,42,.13),transparent 25%),linear-gradient(180deg,#fbf3e5,var(--bg)); font-family:"Iowan Old Style","Palatino Linotype",serif; }}
    header {{ position:sticky; top:0; z-index:5; padding:22px 24px 18px; border-bottom:1px solid var(--border); background:rgba(255,250,240,.94); backdrop-filter:blur(10px); }}
    h1 {{ margin:0 0 8px; font-size:clamp(26px,3vw,40px); letter-spacing:-.035em; }}
    .lede {{ margin:0; max-width:1100px; color:var(--muted); line-height:1.45; }}
    .toolbar {{ display:flex; flex-wrap:wrap; gap:10px; margin-top:16px; }}
    button,select,input {{ border:1px solid var(--border); background:var(--panel); color:var(--ink); border-radius:999px; padding:8px 12px; font:inherit; box-shadow:0 4px 14px var(--shadow); }}
    main {{ padding:22px 24px 38px; display:grid; gap:22px; }}
    .summary {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(250px,1fr)); gap:12px; }}
    .summary-card,.card {{ background:rgba(255,250,240,.96); border:1px solid var(--border); border-radius:18px; box-shadow:0 12px 32px var(--shadow); }}
    .summary-card {{ padding:15px; border-top:5px solid var(--accent); }}
    .summary-card h2 {{ margin:0 0 9px; font-size:18px; }}
    .summary-grid,.score-grid {{ display:grid; grid-template-columns:repeat(2,1fr); gap:8px; }}
    .metric {{ border:1px solid var(--border); border-radius:12px; padding:8px; background:#fffdf7; }}
    .metric span,.meta span {{ display:block; color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.055em; }}
    .metric strong,.meta strong {{ font-size:18px; }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(520px,1fr)); gap:16px; }}
    .card {{ overflow:hidden; }}
    .videos {{ display:grid; grid-template-columns:1fr 1fr; background:#050403; gap:1px; }}
    .video-panel {{ position:relative; background:#050403; }}
    .video-panel span {{ position:absolute; left:8px; top:8px; z-index:1; border-radius:999px; padding:3px 7px; background:rgba(5,4,3,.7); color:#fffaf0; font-size:12px; }}
    video {{ display:block; width:100%; aspect-ratio:2/1; object-fit:contain; background:#050403; }}
    .body {{ padding:13px 14px 16px; display:grid; gap:11px; }}
    .title {{ display:flex; justify-content:space-between; gap:10px; align-items:baseline; }}
    .pill {{ border-radius:999px; background:#ddeee8; padding:4px 8px; color:#17433f; font-size:12px; white-space:nowrap; }}
    .pill.partial {{ background:#f3e1d6; color:#7b3b22; }}
    .pill.unlabeled {{ background:#ece8de; color:#60584e; }}
    .plot {{ border:1px solid var(--border); border-radius:14px; padding:9px; background:#fffdf7; }}
    .plot-title {{ display:flex; justify-content:space-between; color:var(--muted); font-size:12px; margin-bottom:4px; }}
    svg {{ width:100%; height:118px; display:block; overflow:visible; }}
    .gridline {{ stroke:rgba(29,26,22,.12); stroke-width:1; stroke-dasharray:4 4; }}
    .line {{ fill:none; stroke-width:2.25; stroke-linecap:round; stroke-linejoin:round; }}
    .playhead {{ stroke:#16120e; stroke-width:2; opacity:.75; }}
    .axis-label {{ fill:var(--muted); font-size:10px; }}
    .plots {{ display:grid; grid-template-columns:repeat(2,1fr); gap:8px; }}
    .path {{ color:var(--muted); font-size:12px; word-break:break-all; }}
    .hidden {{ display:none!important; }}
    @media (max-width:760px) {{ header,main {{ padding-left:14px; padding-right:14px; }} .cards,.plots,.videos {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Square PH Policy-View BC Report</h1>
    <p class="lede">Four finished BC runs scored by transition negative log likelihood. Videos show primary view and wrist camera side by side; traces are synchronized to video playback.</p>
    <div class="toolbar">
      <select id="sort">
        <option value="worst">Sort highest NLL</option>
        <option value="view_gap">Sort largest view gap</option>
        <option value="policy_gap">Sort largest GMM/discrete gap</option>
        <option value="demo">Sort demo id</option>
      </select>
      <select id="view-filter"><option value="all">All views</option><option value="agent_wrist">Agent+wrist</option><option value="left_close_low_wrist">Left-close-low+wrist</option></select>
      <select id="policy-filter"><option value="all">All policies</option><option value="GMM">GMM</option><option value="Discrete">Discrete</option></select>
      <select id="observability-filter"><option value="all">All observability labels</option><option value="full">Full</option><option value="partial">Partial</option><option value="unlabeled">Unlabeled</option></select>
      <label>smooth <input id="smooth-window" type="number" min="1" max="101" step="2" value="9"></label>
      <input id="search" placeholder="Filter demo id">
    </div>
  </header>
  <main>
    <section id="summary" class="summary"></section>
    <section id="cards" class="cards"></section>
  </main>
  <script id="payload" type="application/json">{payload}</script>
  <script>
    const DATA = JSON.parse(document.getElementById('payload').textContent);
    const fmt = v => Number.isFinite(v) ? v.toFixed(3) : 'n/a';
    const specs = DATA.specs;
    const cards = document.getElementById('cards');
    const summary = document.getElementById('summary');
    function smooth(values) {{
      const win = Math.max(1, Math.floor(Number(document.getElementById('smooth-window').value) || 1));
      if (win <= 1 || values.length < 3) return values.slice();
      const half = Math.floor(win / 2);
      return values.map((_, i) => {{
        const s = Math.max(0, i - half), e = Math.min(values.length, i + half + 1);
        return values.slice(s, e).reduce((a,b)=>a+b,0) / (e - s);
      }});
    }}
    function tracePath(trace, frames) {{
      if (!trace || !trace.scores || !trace.scores.length) return '';
      const scores = smooth(trace.scores), steps = trace.steps;
      const left=35,right=345,top=10,bottom=100,xMax=Math.max(frames-1,...steps,1);
      const min=Math.min(...scores), max=Math.max(...scores), span=Math.max(1e-6,max-min);
      return scores.map((score,i)=>`${{i?'L':'M'}} ${{(left + steps[i]/xMax*(right-left)).toFixed(2)}} ${{(bottom - (score-min)/span*(bottom-top)).toFixed(2)}}`).join(' ');
    }}
    function traceExtent(trace) {{
      if (!trace || !trace.scores || !trace.scores.length) return {{min: NaN, max: NaN}};
      const scores = smooth(trace.scores);
      return {{min: Math.min(...scores), max: Math.max(...scores)}};
    }}
    function scoreAt(trace, frame) {{
      if (!trace || !trace.steps || !trace.steps.length) return NaN;
      const scores = smooth(trace.scores);
      let best=0, dist=Math.abs(trace.steps[0]-frame);
      for (let i=1;i<trace.steps.length;i++) {{
        const d=Math.abs(trace.steps[i]-frame);
        if (d < dist) {{ best=i; dist=d; }}
      }}
      return scores[best];
    }}
    function visibleSpecs() {{
      const view = document.getElementById('view-filter').value;
      const policy = document.getElementById('policy-filter').value;
      return specs.filter(s => (view === 'all' || s.view === view) && (policy === 'all' || s.algo === policy));
    }}
    function renderSummary() {{
      const active = visibleSpecs();
      summary.innerHTML = active.map(s => {{
        const m = DATA.summary[s.key].nll;
        return `<article class="summary-card" style="--accent:${{DATA.colors[s.key]}}">
          <h2>${{s.label}}</h2>
          <div class="summary-grid">
            <div class="metric"><span>mean NLL</span><strong>${{fmt(m.mean)}}</strong></div>
            <div class="metric"><span>std</span><strong>${{fmt(m.std)}}</strong></div>
            <div class="metric"><span>min</span><strong>${{fmt(m.min)}}</strong></div>
            <div class="metric"><span>max</span><strong>${{fmt(m.max)}}</strong></div>
          </div>
          <p class="path">${{DATA.summary[s.key].dataset}}</p>
        </article>`;
      }}).join('');
    }}
    function sortedRows() {{
      const obs = document.getElementById('observability-filter').value;
      const search = document.getElementById('search').value.trim();
      const mode = document.getElementById('sort').value;
      let rows = DATA.rows.slice();
      if (obs !== 'all') rows = rows.filter(r => r.observability === obs);
      if (search) rows = rows.filter(r => String(r.ep_idx).includes(search));
      const active = visibleSpecs().map(s => s.key);
      rows.sort((a,b) => {{
        if (mode === 'demo') return a.ep_idx - b.ep_idx;
        if (mode === 'view_gap') return Math.max(Math.abs(b.gmm_gap), Math.abs(b.discrete_gap)) - Math.max(Math.abs(a.gmm_gap), Math.abs(a.discrete_gap));
        if (mode === 'policy_gap') return Math.max(Math.abs(b.agent_policy_gap), Math.abs(b.left_policy_gap)) - Math.max(Math.abs(a.agent_policy_gap), Math.abs(a.left_policy_gap));
        return Math.max(...active.map(k => b.scores[k])) - Math.max(...active.map(k => a.scores[k]));
      }});
      return rows;
    }}
    function plot(row, spec) {{
      const videoMeta = row.videos[spec.view];
      const id = `${{spec.key}}-${{row.ep_idx}}`;
      const extent = traceExtent(row.traces[spec.key]);
      return `<div class="plot" data-spec="${{spec.key}}">
        <div class="plot-title"><span>${{spec.label}}</span><strong id="${{id}}">${{fmt(row.scores[spec.key])}}</strong></div>
        <svg viewBox="0 0 380 112" preserveAspectRatio="none">
          <line class="gridline" x1="35" y1="10" x2="345" y2="10"></line><line class="gridline" x1="35" y1="55" x2="345" y2="55"></line><line class="gridline" x1="35" y1="100" x2="345" y2="100"></line>
          <text class="axis-label" x="3" y="14">${{fmt(extent.max)}}</text>
          <text class="axis-label" x="3" y="103">${{fmt(extent.min)}}</text>
          <path class="line" stroke="${{DATA.colors[spec.key]}}" d="${{tracePath(row.traces[spec.key], videoMeta.num_frames)}}"></path>
          <line class="playhead" x1="35" x2="35" y1="10" y2="100"></line>
          <text class="axis-label" x="35" y="110">0</text><text class="axis-label" x="320" y="110">${{videoMeta.num_frames - 1}}</text>
        </svg>
      </div>`;
    }}
    function render() {{
      const active = visibleSpecs();
      renderSummary();
      cards.innerHTML = sortedRows().map(row => {{
        const pill = row.observability === 'partial' ? 'partial' : row.observability === 'unlabeled' ? 'unlabeled' : '';
        return `<article class="card" data-ep-idx="${{row.ep_idx}}">
          <div class="videos">
            <div class="video-panel"><span>agent + wrist</span><video data-view="agent_wrist" src="${{row.videos.agent_wrist.video}}" controls preload="metadata"></video></div>
            <div class="video-panel"><span>left-close-low + wrist</span><video data-view="left_close_low_wrist" src="${{row.videos.left_close_low_wrist.video}}" controls preload="metadata"></video></div>
          </div>
          <div class="body">
            <div class="title"><strong>demo_${{String(row.ep_idx).padStart(4,'0')}}</strong><span class="pill ${{pill}}">${{row.observability}}</span></div>
            <div class="score-grid">${{active.map(s => `<div class="metric"><span>${{s.label}}</span><strong style="color:${{DATA.colors[s.key]}}">${{fmt(row.scores[s.key])}}</strong></div>`).join('')}}</div>
            <div class="plots">${{active.map(s => plot(row, s)).join('')}}</div>
            ${{row.annotation_note ? `<div class="path">note: ${{row.annotation_note}}</div>` : ''}}
          </div>
        </article>`;
      }}).join('');
      sync();
    }}
    function update(card, view) {{
      const row = DATA.rows.find(r => r.ep_idx === Number(card.dataset.epIdx));
      const video = card.querySelector(`video[data-view="${{view}}"]`);
      if (!row || !video) return;
      const meta = row.videos[view], frame = Math.min(meta.num_frames - 1, Math.max(0, Math.round(video.currentTime * (meta.fps || 20))));
      const x = 35 + frame / Math.max(1, meta.num_frames - 1) * (345 - 35);
      specs.filter(s => s.view === view).forEach(s => {{
        const plotEl = card.querySelector(`.plot[data-spec="${{s.key}}"]`);
        if (!plotEl) return;
        plotEl.querySelectorAll('.playhead').forEach(head => {{ head.setAttribute('x1', x); head.setAttribute('x2', x); }});
        const readout = card.querySelector(`#${{s.key}}-${{row.ep_idx}}`);
        if (readout) readout.textContent = fmt(scoreAt(row.traces[s.key], frame));
      }});
    }}
    function sync() {{
      document.querySelectorAll('.card').forEach(card => {{
        card.querySelectorAll('video').forEach(video => {{
          const fn = () => update(card, video.dataset.view);
          video.addEventListener('loadedmetadata', fn); video.addEventListener('timeupdate', fn); video.addEventListener('seeked', fn); fn();
        }});
      }});
    }}
    ['sort','view-filter','policy-filter','observability-filter','smooth-window'].forEach(id => document.getElementById(id).addEventListener('change', render));
    document.getElementById('search').addEventListener('input', render);
    render();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, summary = build_rows(args)
    write_csv(rows, args.output_dir / "square_ph_policy_view_bc_report.csv")
    (args.output_dir / "index.html").write_text(build_html(rows, summary))
    print(f"wrote {args.output_dir / 'index.html'}")
    print(f"wrote {args.output_dir / 'square_ph_policy_view_bc_report.csv'}")


if __name__ == "__main__":
    main()
