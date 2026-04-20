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


def maybe_copy_plot(src: Path, dst: Path) -> str | None:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst.name


def build_html(
    output_dir: Path,
    scores: dict[str, dict[str, object]],
    videos: dict[int, str],
    plots: dict[str, dict[str, str | None]],
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
                "video": videos[ep_idx],
                "image_score": image_score,
                "proprio_score": proprio_score,
                "gap": gap,
                "label": label,
                "image_trace": scores["image_only"]["traces"].get(ep_idx, {}),
                "proprio_trace": scores["image_proprio"]["traces"].get(ep_idx, {}),
            }
        )

    payload = {
        "rows": rows,
        "plots": plots,
    }
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
    button, select {{
      border: 1px solid var(--border);
      background: var(--panel);
      color: var(--ink);
      border-radius: 999px;
      padding: 8px 12px;
      font: inherit;
      box-shadow: 0 4px 14px var(--shadow);
    }}
    button.active {{ background: var(--ink); color: var(--panel); }}
    main {{ padding: 22px 24px 34px; display: grid; gap: 22px; }}
    .plots, .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(310px, 1fr)); gap: 16px; }}
    .card {{
      background: rgba(255, 250, 240, .95);
      border: 1px solid var(--border);
      border-radius: 18px;
      box-shadow: 0 12px 32px var(--shadow);
      overflow: hidden;
    }}
    .card-body {{ padding: 14px 16px 16px; }}
    .plot-card {{ padding: 14px; }}
    .plot-card img {{ width: 100%; display: block; border-radius: 12px; border: 1px solid var(--border); }}
    video {{ width: 100%; display: block; background: #111; image-rendering: pixelated; }}
    h2, h3 {{ margin: 0; }}
    .meta {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; margin: 12px 0; }}
    .metric {{
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 8px;
      background: #fffdf6;
    }}
    .metric span {{ display: block; color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .055em; }}
    .metric strong {{ font-size: 18px; }}
    canvas {{ width: 100%; height: 120px; border: 1px solid var(--border); border-radius: 12px; background: #fffdf6; }}
    .legend {{ display: flex; gap: 12px; color: var(--muted); font-size: 13px; margin-top: 8px; }}
    .swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 999px; margin-right: 4px; }}
    .image {{ background: var(--image); }}
    .proprio {{ background: var(--proprio); }}
  </style>
</head>
<body>
  <header>
    <h1>Square PH Wrist DemInf Review</h1>
    <p class="lede">Compares MI scores from a wrist-image-only observation VAE against a wrist-image + robot-proprio observation VAE. Videos are exported from the Square PH HDF5 wrist camera.</p>
    <div class="toolbar">
      <button class="active" data-sort="gap_abs">Sort by |proprio - image|</button>
      <button data-sort="image_score">Sort image-only</button>
      <button data-sort="proprio_score">Sort image+proprio</button>
      <button data-sort="ep_idx">Sort demo id</button>
    </div>
  </header>
  <main>
    <section class="plots" id="plots"></section>
    <section class="grid" id="grid"></section>
  </main>
  <script>
    const DATA = {payload_json};
    const grid = document.getElementById('grid');
    const plots = document.getElementById('plots');
    let currentSort = 'gap_abs';

    function fmt(x) {{
      return Number.isFinite(x) ? x.toFixed(3) : 'n/a';
    }}

    function renderPlots() {{
      const labels = {{
        image_only: 'Wrist image only',
        image_proprio: 'Wrist image + proprio'
      }};
      plots.innerHTML = Object.entries(DATA.plots).map(([method, files]) => `
        <article class="card plot-card">
          <h2>${{labels[method]}}</h2>
          ${{files.curve ? `<img src="${{method}}/${{files.curve}}" alt="${{method}} curve">` : ''}}
          ${{files.hist ? `<img src="${{method}}/${{files.hist}}" alt="${{method}} histogram">` : ''}}
        </article>
      `).join('');
    }}

    function drawTrace(canvas, imageTrace, proprioTrace) {{
      const ctx = canvas.getContext('2d');
      const w = canvas.width = canvas.clientWidth * devicePixelRatio;
      const h = canvas.height = canvas.clientHeight * devicePixelRatio;
      ctx.scale(devicePixelRatio, devicePixelRatio);
      const cw = canvas.clientWidth;
      const ch = canvas.clientHeight;
      ctx.clearRect(0, 0, cw, ch);
      const traces = [imageTrace, proprioTrace].filter(t => t && t.steps && t.steps.length);
      if (!traces.length) return;
      const xs = traces.flatMap(t => t.steps);
      const ys = traces.flatMap(t => t.scores);
      const minX = Math.min(...xs), maxX = Math.max(...xs);
      const minY = Math.min(...ys), maxY = Math.max(...ys);
      const pad = 12;
      function px(x) {{ return pad + (x - minX) / Math.max(1, maxX - minX) * (cw - 2 * pad); }}
      function py(y) {{ return ch - pad - (y - minY) / Math.max(1e-6, maxY - minY) * (ch - 2 * pad); }}
      ctx.strokeStyle = '#d7cbbb';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pad, ch - pad);
      ctx.lineTo(cw - pad, ch - pad);
      ctx.stroke();
      for (const [trace, color] of [[imageTrace, '#0f6d67'], [proprioTrace, '#b54a2a']]) {{
        if (!trace || !trace.steps || !trace.steps.length) continue;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        trace.steps.forEach((x, i) => {{
          const y = trace.scores[i];
          if (i === 0) ctx.moveTo(px(x), py(y));
          else ctx.lineTo(px(x), py(y));
        }});
        ctx.stroke();
      }}
    }}

    function sortedRows() {{
      const rows = [...DATA.rows];
      rows.sort((a, b) => {{
        if (currentSort === 'gap_abs') return Math.abs(b.gap) - Math.abs(a.gap);
        return a[currentSort] - b[currentSort];
      }});
      return rows;
    }}

    function render() {{
      grid.innerHTML = sortedRows().map(row => `
        <article class="card">
          <video controls muted preload="metadata" src="${{row.video}}"></video>
          <div class="card-body">
            <h3>demo_${{String(row.ep_idx).padStart(4, '0')}}</h3>
            <div class="meta">
              <div class="metric"><span>image only</span><strong>${{fmt(row.image_score)}}</strong></div>
              <div class="metric"><span>image+prop</span><strong>${{fmt(row.proprio_score)}}</strong></div>
              <div class="metric"><span>gap</span><strong>${{fmt(row.gap)}}</strong></div>
              <div class="metric"><span>label</span><strong>${{row.label ?? 'n/a'}}</strong></div>
            </div>
            <canvas data-ep="${{row.ep_idx}}"></canvas>
            <div class="legend">
              <span><i class="swatch image"></i>image only</span>
              <span><i class="swatch proprio"></i>image + proprio</span>
            </div>
          </div>
        </article>
      `).join('');
      for (const canvas of grid.querySelectorAll('canvas')) {{
        const row = DATA.rows.find(r => String(r.ep_idx) === canvas.dataset.ep);
        drawTrace(canvas, row.image_trace, row.proprio_trace);
      }}
    }}

    document.querySelectorAll('button[data-sort]').forEach(button => {{
      button.addEventListener('click', () => {{
        document.querySelectorAll('button[data-sort]').forEach(b => b.classList.remove('active'));
        button.classList.add('active');
        currentSort = button.dataset.sort;
        render();
      }});
    }});
    renderPlots();
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

    plots = {}
    for method in METHODS:
        method_dir = args.output_dir / method
        plots[method] = {
            "curve": maybe_copy_plot(args.scores_root / method / "square_ph_curve.png", method_dir / "square_ph_curve.png"),
            "hist": maybe_copy_plot(args.scores_root / method / "square_ph_hist.png", method_dir / "square_ph_hist.png"),
        }

    build_html(args.output_dir, scores, videos, plots)
    print(args.output_dir / "index.html")


if __name__ == "__main__":
    main()
