"""
Build a local HTML review page for Square PH wrist DemInf scores.

Example:

python scripts/quality/build_square_ph_wrist_review_page.py \
  --scores-root /Users/jasonyan/Desktop/demonstration-information/square_ph_wrist_scores \
  --hdf5 /Users/jasonyan/Desktop/demonstration-information/robomimic_square_ph/image.hdf5 \
  --output-dir /Users/jasonyan/Desktop/demonstration-information/square_ph_wrist_review_deploy
"""

from __future__ import annotations

import argparse
import html
import json
import pickle
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np


WRIST_KEY = "robot0_eye_in_hand_image"
METHODS = {
    "image_only": "Wrist image only",
    "image_proprio": "Wrist image + proprio",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-root", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-demos", type=int, default=36)
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
    quality_by_ep_idx = {int(k): float(v) for k, v in data.get("quality_by_ep_idx", {}).items()}

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
        ds_steps, ds_scores = downsample(unique_steps, mean_scores, max_trace_points)
        traces[int(ep_idx)] = {
            "steps": ds_steps.astype(int).tolist(),
            "scores": [round(float(v), 5) for v in ds_scores],
        }

    return {
        "ep_scores": ep_scores,
        "traces": traces,
        "quality_by_ep_idx": quality_by_ep_idx,
    }


def select_demo_indices(scores: dict[str, dict[str, object]], max_demos: int) -> list[int]:
    image_scores = scores["image_only"]["ep_scores"]
    proprio_scores = scores["image_proprio"]["ep_scores"]
    common = sorted(set(image_scores) & set(proprio_scores))
    if len(common) <= max_demos:
        return common

    def quantile_pick(values: list[tuple[int, float]], count: int) -> list[int]:
        values = sorted(values, key=lambda item: item[1])
        if not values:
            return []
        idxs = np.linspace(0, len(values) - 1, count).round().astype(int)
        return [values[int(idx)][0] for idx in idxs]

    per_bucket = max(4, max_demos // 4)
    picked: list[int] = []
    picked.extend(quantile_pick([(i, image_scores[i]) for i in common], per_bucket))
    picked.extend(quantile_pick([(i, proprio_scores[i]) for i in common], per_bucket))
    picked.extend(quantile_pick([(i, proprio_scores[i] - image_scores[i]) for i in common], per_bucket))
    picked.extend(quantile_pick([(i, abs(proprio_scores[i] - image_scores[i])) for i in common], per_bucket))

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
            }
    return assets


def maybe_copy_plot(src: Path, dst: Path) -> str | None:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst.name


def build_html(
    output_dir: Path,
    scores: dict[str, dict[str, object]],
    videos: dict[int, dict[str, object]],
) -> None:
    rows = []
    image_scores = scores["image_only"]["ep_scores"]
    proprio_scores = scores["image_proprio"]["ep_scores"]
    labels = scores["image_only"].get("quality_by_ep_idx", {})

    for ep_idx in videos:
        image_score = image_scores[ep_idx]
        proprio_score = proprio_scores[ep_idx]
        gap = proprio_score - image_score
        label = labels.get(ep_idx)
        rows.append(
            {
                "ep_idx": ep_idx,
                "video": videos[ep_idx]["video"],
                "num_frames": videos[ep_idx]["num_frames"],
                "image_score": image_score,
                "proprio_score": proprio_score,
                "gap": gap,
                "label": label,
                "image_trace": scores["image_only"]["traces"].get(ep_idx, {}),
                "proprio_trace": scores["image_proprio"]["traces"].get(ep_idx, {}),
            }
        )

    payload = {"rows": rows}
    payload_json = json.dumps(payload)

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Square PH Wrist DemInf Review</title>
  <style>
    :root {{
      --bg: #ece6d8;
      --panel: #fffaf0;
      --ink: #1f1b16;
      --muted: #746b5f;
      --border: #d7cbbb;
      --image: #0f6d67;
      --proprio: #b54a2a;
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
    .lede {{ margin: 0; max-width: 1100px; color: var(--muted); line-height: 1.45; }}
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
    button.active {{ background: var(--ink); color: var(--panel); }}
    main {{ padding: 22px 24px 34px; display: grid; gap: 24px; }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 12px;
    }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 16px; }}
    .card {{
      background: rgba(255, 250, 240, .95);
      border: 1px solid var(--border);
      border-radius: 18px;
      box-shadow: 0 12px 32px var(--shadow);
      overflow: hidden;
    }}
    .summary-card {{ padding: 14px 16px; background: rgba(255, 250, 240, .95); border: 1px solid var(--border); border-radius: 18px; box-shadow: 0 12px 32px var(--shadow); }}
    .summary-card h2 {{ margin: 0 0 8px; font-size: 18px; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(96px, 1fr)); gap: 8px; }}
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
    svg {{ width: 100%; height: 126px; overflow: visible; display: block; }}
    .gridline {{ stroke: rgba(29,26,22,.12); stroke-width: 1; stroke-dasharray: 4 4; }}
    .zero {{ stroke: rgba(29,26,22,.35); stroke-width: 1.2; }}
    .axis-label {{ fill: var(--muted); font-size: 10px; }}
    .line {{ fill: none; stroke-width: 2.4; stroke-linecap: round; stroke-linejoin: round; }}
    .image-line {{ stroke: var(--image); }}
    .proprio-line {{ stroke: var(--proprio); }}
    .playhead {{ stroke: #16120e; stroke-width: 2; opacity: .78; }}
    .legend {{ display: flex; flex-wrap: wrap; justify-content: space-between; gap: 8px 12px; color: var(--muted); font-size: 12px; }}
    .swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 999px; margin-right: 5px; }}
    .swatch.image {{ background: var(--image); }}
    .swatch.proprio {{ background: var(--proprio); }}
    .body {{ padding: 12px 14px 15px; display: grid; gap: 9px; }}
    .title {{ display: flex; justify-content: space-between; gap: 10px; align-items: baseline; }}
    .title strong {{ font-size: 17px; }}
    .pill {{ border-radius: 999px; background: #ddeee8; padding: 4px 8px; color: #17433f; font-size: 12px; white-space: nowrap; }}
    h2, h3 {{ margin: 0; }}
    .meta {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }}
    .metric {{
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 8px;
      background: #fffdf6;
    }}
    .metric span {{ display: block; color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .055em; }}
    .metric strong {{ font-size: 18px; }}
    .image-score {{ color: var(--image); }}
    .proprio-score {{ color: var(--proprio); }}
    .path {{ color: var(--muted); font-size: 12px; word-break: break-all; }}
    @media (max-width: 700px) {{
      header, main {{ padding-left: 14px; padding-right: 14px; }}
      .cards {{ grid-template-columns: 1fr; }}
      .summary-grid, .meta {{ grid-template-columns: repeat(2, 1fr); }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Square PH Wrist DemInf Review</h1>
    <p class="lede">Compares MI scores from a wrist-image-only observation VAE against a wrist-image + robot-proprio observation VAE. Videos are exported from the Square PH HDF5 wrist camera.</p>
    <div class="toolbar">
      <select id="sort">
        <option value="gap_abs">Sort by |proprio - image|</option>
        <option value="image_score">Sort image-only</option>
        <option value="proprio_score">Sort image+proprio</option>
        <option value="gap">Sort gap</option>
        <option value="demo">Sort demo id</option>
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
  <script>
    const DATA = {payload_json};
    const cards = document.getElementById('cards');

    function fmt(x) {{
      return Number.isFinite(x) ? x.toFixed(3) : 'n/a';
    }}

    function statBlock(label, values) {{
      const mean = values.length ? values.reduce((a, b) => a + b, 0) / values.length : NaN;
      return `<div class="metric"><span>${{label}}</span><strong>${{fmt(mean)}}</strong></div>`;
    }}

    function renderSummary(rows) {{
      document.getElementById('summary').innerHTML = `
        <article class="summary-card">
          <h2>Visible demos</h2>
          <div class="summary-grid">
            <div class="metric"><span>count</span><strong>${{rows.length}}</strong></div>
            ${{statBlock('image mean', rows.map(r => r.image_score))}}
            ${{statBlock('proprio mean', rows.map(r => r.proprio_score))}}
            ${{statBlock('gap mean', rows.map(r => r.gap))}}
          </div>
        </article>`;
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
      const imageTrace = smoothTrace(row.image_trace, window);
      const proprioTrace = smoothTrace(row.proprio_trace, window);
      const traces = [imageTrace, proprioTrace].filter(Boolean);
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
            <line class="gridline" x1="36" y1="12" x2="344" y2="12"></line>
            <line class="gridline" x1="36" y1="60" x2="344" y2="60"></line>
            <line class="gridline" x1="36" y1="108" x2="344" y2="108"></line>
            ${{showZero ? `<line class="zero" x1="36" y1="${{zeroY.toFixed(2)}}" x2="344" y2="${{zeroY.toFixed(2)}}"></line>` : ''}}
            <text class="axis-label" x="4" y="16">${{fmt(yMax, 2)}}</text>
            <text class="axis-label" x="4" y="112">${{fmt(yMin, 2)}}</text>
            <path class="line image-line" d="${{pathFromTrace(imageTrace, xMin, xMax, yMin, yMax)}}"></path>
            <path class="line proprio-line" d="${{pathFromTrace(proprioTrace, xMin, xMax, yMin, yMax)}}"></path>
            <line class="playhead" x1="36" y1="8" x2="36" y2="112"></line>
          </svg>
          <div class="legend">
            <span><span class="swatch image"></span>image only, slope/100: <strong>${{fmt(slopePer100(imageTrace), 3)}}</strong></span>
            <span><span class="swatch proprio"></span>image + proprio, slope/100: <strong>${{fmt(slopePer100(proprioTrace), 3)}}</strong></span>
          </div>
        </div>`;
    }}

    function card(row) {{
      return `
        <article class="card" data-demo="${{row.ep_idx}}">
          <div class="video-panel">
            <span>wrist view</span>
            <video controls preload="metadata" src="${{row.video}}" data-role="wrist"></video>
          </div>
          ${{traceSvg(row)}}
          <div class="body">
            <div class="title"><strong>demo_${{String(row.ep_idx).padStart(4, '0')}}</strong><span class="pill">Square PH</span></div>
            <div class="meta">
              <div class="metric"><span>image only</span><strong class="image-score">${{fmt(row.image_score)}}</strong></div>
              <div class="metric"><span>image + proprio</span><strong class="proprio-score">${{fmt(row.proprio_score)}}</strong></div>
              <div class="metric"><span>gap</span><strong>${{fmt(row.gap)}}</strong></div>
              <div class="metric"><span>label</span><strong>${{row.label ?? 'n/a'}}</strong></div>
              <div class="metric"><span>frames</span><strong>${{row.num_frames}}</strong></div>
              <div class="metric"><span>dataset ep</span><strong>${{row.ep_idx}}</strong></div>
            </div>
            <div class="path">wrist: ${{row.video}}</div>
          </div>
        </article>`;
    }}

    function sortedRows(rows) {{
      const rows = [...DATA.rows];
      const mode = document.getElementById('sort').value;
      const query = document.getElementById('search').value.toLowerCase().trim();
      let copy = [...rows];
      if (query) copy = copy.filter(row => `${{row.ep_idx}}`.includes(query));
      if (mode === 'image_score') copy.sort((a, b) => b.image_score - a.image_score);
      else if (mode === 'proprio_score') copy.sort((a, b) => b.proprio_score - a.proprio_score);
      else if (mode === 'gap') copy.sort((a, b) => b.gap - a.gap);
      else if (mode === 'demo') copy.sort((a, b) => a.ep_idx - b.ep_idx);
      else copy.sort((a, b) => Math.abs(b.gap) - Math.abs(a.gap));
      return copy;
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

    function attachVideoSync() {{
      document.querySelectorAll('.card').forEach(cardEl => {{
        const video = cardEl.querySelector('video[data-role="wrist"]') || cardEl.querySelector('video');
        if (!video) return;
        video.addEventListener('loadedmetadata', () => updatePlayhead(cardEl));
        video.addEventListener('play', () => syncLoop(video, cardEl));
        video.addEventListener('pause', () => updatePlayhead(cardEl));
        video.addEventListener('seeked', () => updatePlayhead(cardEl));
        video.addEventListener('timeupdate', () => updatePlayhead(cardEl));
        updatePlayhead(cardEl);
      }});
    }}

    function render() {{
      const rows = sortedRows(DATA.rows);
      renderSummary(rows);
      cards.innerHTML = rows.map(card).join('');
      attachVideoSync();
    }}

    document.getElementById('sort').addEventListener('change', render);
    document.getElementById('search').addEventListener('input', render);
    document.getElementById('smooth-window').addEventListener('change', render);
    render();
  </script>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_doc)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scores = {
        method: load_score_bundle(args.scores_root / method / "square_ph.pkl", args.max_trace_points)
        for method in METHODS
    }
    ep_indices = select_demo_indices(scores, args.max_demos)
    videos = export_videos(args.hdf5, args.output_dir, ep_indices, args.fps)
    build_html(args.output_dir, scores, videos)
    print(args.output_dir / "index.html")


if __name__ == "__main__":
    main()
