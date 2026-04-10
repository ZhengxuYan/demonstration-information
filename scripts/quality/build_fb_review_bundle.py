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

DEFAULT_SMOOTHING_WINDOW = 9
DEFAULT_STAGE_NAMES = [
    "approach",
    "pregrasp_align",
    "grasp",
    "transport",
    "insert_align",
    "insert",
    "release",
    "retreat",
]


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
        traces[int(ep_idx)] = {
            "steps": unique_steps.astype(int).tolist(),
            "scores": mean_scores.tolist(),
            "min_score": float(mean_scores.min()),
            "max_score": float(mean_scores.max()),
            "num_frames": int(unique_steps.max()) + 1,
        }
    return traces


def smooth_scores(scores: list[float], requested_window: int = DEFAULT_SMOOTHING_WINDOW) -> np.ndarray:
    arr = np.asarray(scores, dtype=float)
    if arr.size <= 2 or requested_window <= 1:
        return arr
    window = min(requested_window, arr.size if arr.size % 2 == 1 else arr.size - 1)
    window = max(window, 1)
    if window <= 1:
        return arr
    half = window // 2
    out = []
    for idx in range(arr.size):
        low = max(0, idx - half)
        high = min(arr.size, idx + half + 1)
        out.append(float(np.mean(arr[low:high])))
    return np.asarray(out, dtype=float)


def initial_trace_view(trace: dict[str, object]) -> dict[str, object]:
    steps = np.asarray(trace["steps"], dtype=float)
    scores = smooth_scores(trace["scores"], DEFAULT_SMOOTHING_WINDOW)
    width, height = 300.0, 96.0
    axis_x, pad_y = 42.0, 10.0
    max_step = max(float(trace["num_frames"]) - 1.0, float(steps[-1]) if len(steps) else 1.0, 1.0)
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))
    score_span = max(max_score - min_score, 1e-6)

    def x_for(step: float) -> float:
        return axis_x + (step / max_step) * (width - 2 * axis_x)

    def y_for(score: float) -> float:
        return height - pad_y - ((score - min_score) / score_span) * (height - 2 * pad_y)

    points = [(x_for(float(step)), y_for(float(score))) for step, score in zip(steps, scores)]
    path_d = " ".join(
        ("M " if idx == 0 else "L ") + f"{x:.2f} {y:.2f}" for idx, (x, y) in enumerate(points)
    )
    return {
        "path_d": path_d,
        "tick_top": f"{max_score:.2f}",
        "tick_mid": f"{((min_score + max_score) / 2):.2f}",
        "tick_bottom": f"{min_score:.2f}",
    }


def build_annotation_payload(forward_rows: list[dict[str, object]], backward_rows: list[dict[str, object]]) -> dict:
    payload = {"stages": DEFAULT_STAGE_NAMES, "demos": {}}
    for category, rows in (("forward_grab", forward_rows), ("backward_grab", backward_rows)):
        for row in rows:
            key = f"{category}/demo_{int(row['ep_idx'])}"
            payload["demos"][key] = {
                "category": category,
                "ep_idx": int(row["ep_idx"]),
                "num_frames": int(row["num_frames"]),
                "stages": [{"name": name, "start": None, "end": None} for name in DEFAULT_STAGE_NAMES],
            }
    return payload


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
    annotation_key = html.escape(str(row["annotation_key"]))
    trace = row["trace"]
    initial_view = initial_trace_view(trace)
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
          <g class="trace-stage-bands"></g>
          <text class="trace-tick trace-tick-top" x="38" y="14">{initial_view["tick_top"]}</text>
          <text class="trace-tick trace-tick-mid" x="38" y="52">{initial_view["tick_mid"]}</text>
          <text class="trace-tick trace-tick-bottom" x="38" y="90">{initial_view["tick_bottom"]}</text>
          <path class="trace-line" d="{initial_view["path_d"]}"></path>
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
      <div class="annotation-panel" data-annotation-key="{annotation_key}">
        <div class="annotation-head">
          <strong>stage boundaries</strong>
        </div>
        <div class="annotation-toolbar">
          <label>
            active stage
            <select class="annotation-stage-select"></select>
          </label>
          <button type="button" class="annotation-mark-start">mark start at current frame</button>
          <button type="button" class="annotation-mark-end">mark end at current frame</button>
          <span class="annotation-current-frame">frame 0</span>
        </div>
        <div class="annotation-rows"></div>
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


def build_page(
    output: Path, forward_rows: list[dict[str, object]], backward_rows: list[dict[str, object]], annotations_payload: dict
) -> None:
    annotations_json = html.escape(json.dumps(annotations_payload))
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
    .trace-stage-band {{ opacity: 0.16; }}
    .trace-stage-label {{ fill: var(--muted); font-size: 9px; text-anchor: middle; }}
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
    .controls {{ display: flex; flex-wrap: wrap; align-items: center; gap: 12px; margin-top: 12px; font-size: 14px; color: var(--muted); }}
    .controls label {{ display: flex; align-items: center; gap: 10px; }}
    .controls input[type="range"] {{ width: 220px; accent-color: var(--accent); }}
    .window-readout {{ color: var(--accent); font-weight: 700; min-width: 84px; }}
    .annotation-actions {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px; align-items: center; }}
    .annotation-actions button, .annotation-row button {{
      border: 1px solid var(--border);
      background: white;
      border-radius: 8px;
      padding: 6px 10px;
      font: inherit;
      cursor: pointer;
    }}
    .annotation-status {{ color: var(--muted); font-size: 13px; }}
    .annotation-panel {{ border-top: 1px solid rgba(0, 0, 0, 0.06); padding: 12px 14px 14px; display: grid; gap: 8px; }}
    .annotation-head {{ font-size: 14px; }}
    .annotation-toolbar {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
    .annotation-toolbar label {{ display: flex; align-items: center; gap: 8px; font-size: 13px; color: var(--muted); }}
    .annotation-toolbar select {{
      border: 1px solid var(--border);
      background: white;
      border-radius: 8px;
      padding: 6px 8px;
      font: inherit;
      color: var(--ink);
    }}
    .annotation-current-frame {{ color: var(--accent); font-size: 13px; font-weight: 700; }}
    .annotation-rows {{ display: grid; gap: 8px; }}
    .annotation-row {{ display: grid; grid-template-columns: minmax(92px, 1fr) 62px 62px auto auto; gap: 8px; align-items: center; font-size: 13px; }}
    .annotation-row input {{ width: 100%; border: 1px solid var(--border); border-radius: 8px; padding: 6px 8px; font: inherit; }}
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
      <div class="controls">
        <label for="smoothing-window">
          smoothing window
          <input id="smoothing-window" type="range" min="1" max="31" step="2" value="9">
        </label>
        <span class="window-readout" id="smoothing-window-readout">window 9</span>
      </div>
      <div class="annotation-actions">
        <button type="button" id="export-annotations">export annotations json</button>
        <label>
          <button type="button" id="import-annotations-trigger">import annotations json</button>
          <input type="file" id="import-annotations" accept="application/json" style="display:none">
        </label>
        <span class="annotation-status" id="annotation-status">local autosave on</span>
      </div>
    </div>
    {section_html("forward_grab", "Demos where the handle is grasped with the opening facing forward relative to the wrist camera.", forward_rows)}
    {section_html("backward_grab", "Demos where the handle is grasped with the opening facing backward relative to the wrist camera.", backward_rows)}
  </main>
  <script>
    const DEFAULT_ANNOTATIONS = JSON.parse('{annotations_json}');
    const STAGE_COLORS = ["#0f5b5c", "#c96c33", "#7f8c1f", "#7f4fa3", "#bf3b73", "#2377b8", "#5f5f5f", "#18966e"];
    function clamp(value, low, high) {{
      return Math.max(low, Math.min(high, value));
    }}
    function smoothScores(scores, requestedWindow) {{
      if (!scores.length || requestedWindow <= 1 || scores.length <= 2) return scores.slice();
      let window = Math.min(requestedWindow, scores.length % 2 === 1 ? scores.length : scores.length - 1);
      window = Math.max(window, 1);
      if (window <= 1) return scores.slice();
      const half = Math.floor(window / 2);
      const smoothed = [];
      for (let i = 0; i < scores.length; i++) {{
        let total = 0;
        let count = 0;
        for (let j = Math.max(0, i - half); j <= Math.min(scores.length - 1, i + half); j++) {{
          total += scores[j];
          count += 1;
        }}
        smoothed.push(total / count);
      }}
      return smoothed;
    }}
    function buildTrace(svg) {{
      const steps = JSON.parse(svg.dataset.steps);
      const rawScores = JSON.parse(svg.dataset.scores);
      const numFrames = Number(svg.dataset.numFrames);
      if (!steps.length || !rawScores.length) return null;
      const width = 300;
      const height = 96;
      const axisX = 42;
      const padX = axisX;
      const padY = 10;
      const maxStep = Math.max(numFrames - 1, steps[steps.length - 1], 1);
      const xFor = (step) => padX + (step / maxStep) * (width - 2 * padX);
      return {{ steps, rawScores, numFrames, width, height, padY, xFor, maxStep }};
    }}
    function renderTrace(trace, svg, window) {{
      const scores = smoothScores(trace.rawScores, window);
      const minScore = Math.min(...scores);
      const maxScore = Math.max(...scores);
      const midScore = (minScore + maxScore) / 2;
      const scoreSpan = Math.max(maxScore - minScore, 1e-6);
      const yFor = (score) => trace.height - trace.padY - ((score - minScore) / scoreSpan) * (trace.height - 2 * trace.padY);
      svg.querySelector(".trace-tick-top").textContent = maxScore.toFixed(2);
      svg.querySelector(".trace-tick-mid").textContent = midScore.toFixed(2);
      svg.querySelector(".trace-tick-bottom").textContent = minScore.toFixed(2);
      const points = trace.steps.map((step, idx) => [trace.xFor(step), yFor(scores[idx])]);
      svg.querySelector(".trace-line").setAttribute(
        "d",
        points.map((point, idx) => (idx === 0 ? "M " : "L ") + point[0].toFixed(2) + " " + point[1].toFixed(2)).join(" ")
      );
      trace.scores = scores;
      trace.yFor = yFor;
    }}
    function renderStageBands(trace, svg, stages) {{
      const group = svg.querySelector(".trace-stage-bands");
      group.innerHTML = "";
      stages.forEach((stage, idx) => {{
        if (stage.start == null || stage.end == null) return;
        const start = clamp(Number(stage.start), 0, trace.maxStep);
        const end = clamp(Number(stage.end), start, trace.maxStep);
        const x1 = trace.xFor(start);
        const x2 = trace.xFor(end);
        const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        rect.setAttribute("class", "trace-stage-band");
        rect.setAttribute("x", x1.toFixed(2));
        rect.setAttribute("y", "10");
        rect.setAttribute("width", Math.max(x2 - x1, 1).toFixed(2));
        rect.setAttribute("height", "76");
        rect.setAttribute("fill", STAGE_COLORS[idx % STAGE_COLORS.length]);
        group.appendChild(rect);
        const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
        text.setAttribute("class", "trace-stage-label");
        text.setAttribute("x", ((x1 + x2) / 2).toFixed(2));
        text.setAttribute("y", "20");
        text.textContent = stage.name;
        group.appendChild(text);
      }});
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
    function loadAnnotations() {{
      return JSON.parse(JSON.stringify(DEFAULT_ANNOTATIONS));
    }}
    function normalizeStages(rawStages) {{
      const stageNames = DEFAULT_ANNOTATIONS.stages || [];
      const byName = new Map(Array.isArray(rawStages) ? rawStages.map((stage) => [stage && stage.name, stage]) : []);
      return stageNames.map((name) => {{
        const existing = byName.get(name) || {{}};
        return {{
          name,
          start: Number.isFinite(existing.start) ? existing.start : null,
          end: Number.isFinite(existing.end) ? existing.end : null,
        }};
      }});
    }}
    let annotations = loadAnnotations();
    function saveAnnotations() {{
      document.getElementById("annotation-status").textContent = "edited locally, export json to save";
    }}
    const traces = [];
    document.querySelectorAll(".card").forEach((card) => {{
      const svg = card.querySelector(".trace");
      const video = card.querySelector("video");
      if (!svg || !video) return;
      const trace = buildTrace(svg);
      if (!trace) return;
      const panel = card.querySelector(".annotation-panel");
      const key = panel.dataset.annotationKey;
      const demoAnnotations = (annotations.demos && annotations.demos[key]) || {{}};
      const stages = normalizeStages(demoAnnotations.stages);
      if (!annotations.demos) annotations.demos = {{}};
      annotations.demos[key] = {{
        category: demoAnnotations.category || key.split("/")[0],
        ep_idx: demoAnnotations.ep_idx ?? Number(card.dataset.ep),
        num_frames: demoAnnotations.num_frames ?? trace.numFrames,
        stages,
      }};
      traces.push({{ trace, svg, card, video, panel, key, stages }});
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
      trace.update = update;
    }});
    const smoothingInput = document.getElementById("smoothing-window");
    const smoothingReadout = document.getElementById("smoothing-window-readout");
    function rerenderAll() {{
      const window = Number(smoothingInput.value);
      smoothingReadout.textContent = "window " + window;
      traces.forEach(({{ trace, svg, stages }}) => {{
        renderTrace(trace, svg, window);
        renderStageBands(trace, svg, stages);
        trace.update();
      }});
    }}
    function currentFrameFor(video, trace) {{
      const frac = video.duration ? clamp(video.currentTime / video.duration, 0, 1) : 0;
      return Math.round(frac * trace.maxStep);
    }}
    function refreshCurrentFrame(item) {{
      const readout = item.panel.querySelector(".annotation-current-frame");
      if (!readout) return;
      readout.textContent = "frame " + currentFrameFor(item.video, item.trace);
    }}
    function buildAnnotationRows(item) {{
      const rowsRoot = item.panel.querySelector(".annotation-rows");
      rowsRoot.innerHTML = "";
      item.stages.forEach((stage) => {{
        const row = document.createElement("div");
        row.className = "annotation-row";
        row.innerHTML = `
          <strong>${{stage.name}}</strong>
          <input type="number" min="0" value="${{stage.start ?? ""}}" placeholder="start">
          <input type="number" min="0" value="${{stage.end ?? ""}}" placeholder="end">
          <button type="button">set start</button>
          <button type="button">set end</button>
        `;
        const [startInput, endInput, startBtn, endBtn] = row.querySelectorAll("input, button");
        startInput.addEventListener("change", () => {{
          stage.start = startInput.value === "" ? null : Number(startInput.value);
          saveAnnotations();
          rerenderAll();
        }});
        endInput.addEventListener("change", () => {{
          stage.end = endInput.value === "" ? null : Number(endInput.value);
          saveAnnotations();
          rerenderAll();
        }});
        startBtn.addEventListener("click", () => {{
          stage.start = currentFrameFor(item.video, item.trace);
          startInput.value = stage.start;
          saveAnnotations();
          rerenderAll();
        }});
        endBtn.addEventListener("click", () => {{
          stage.end = currentFrameFor(item.video, item.trace);
          endInput.value = stage.end;
          saveAnnotations();
          rerenderAll();
        }});
        rowsRoot.appendChild(row);
      }});
    }}
    function bindAnnotationToolbar(item) {{
      const select = item.panel.querySelector(".annotation-stage-select");
      const startButton = item.panel.querySelector(".annotation-mark-start");
      const endButton = item.panel.querySelector(".annotation-mark-end");
      if (!select || !startButton || !endButton) return;
      select.innerHTML = item.stages
        .map((stage, idx) => `<option value="${{idx}}">${{stage.name}}</option>`)
        .join("");
      const applyBoundary = (boundaryKey) => {{
        const stageIndex = Number(select.value);
        const stage = item.stages[stageIndex];
        if (!stage) return;
        stage[boundaryKey] = currentFrameFor(item.video, item.trace);
        saveAnnotations();
        buildAnnotationRows(item);
        rerenderAll();
        refreshCurrentFrame(item);
      }};
      startButton.addEventListener("click", () => applyBoundary("start"));
      endButton.addEventListener("click", () => applyBoundary("end"));
      const updateFrame = () => refreshCurrentFrame(item);
      item.video.addEventListener("loadedmetadata", updateFrame);
      item.video.addEventListener("timeupdate", updateFrame);
      item.video.addEventListener("seeked", updateFrame);
      updateFrame();
    }}
    smoothingInput.addEventListener("input", rerenderAll);
    traces.forEach(({{ trace, svg, stages }}) => {{
      renderTrace(trace, svg, Number(smoothingInput.value));
      renderStageBands(trace, svg, stages);
    }});
    traces.forEach(({{ trace }}) => trace.update());
    traces.forEach((item) => {{
      try {{
        buildAnnotationRows(item);
        bindAnnotationToolbar(item);
      }} catch (err) {{
        console.error("annotation init failed", item.key, err);
      }}
    }});
    document.getElementById("export-annotations").addEventListener("click", () => {{
      const blob = new Blob([JSON.stringify(annotations, null, 2)], {{ type: "application/json" }});
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "stage_annotations.json";
      link.click();
      URL.revokeObjectURL(url);
    }});
    document.getElementById("import-annotations-trigger").addEventListener("click", () => {{
      document.getElementById("import-annotations").click();
    }});
    document.getElementById("import-annotations").addEventListener("change", async (event) => {{
      const file = event.target.files && event.target.files[0];
      if (!file) return;
      annotations = JSON.parse(await file.text());
      traces.forEach((item) => {{
        const demoAnnotations = (annotations.demos && annotations.demos[item.key]) || {{}};
        item.stages.splice(0, item.stages.length, ...normalizeStages(demoAnnotations.stages));
        if (!annotations.demos) annotations.demos = {{}};
        annotations.demos[item.key] = {{
          category: demoAnnotations.category || item.key.split("/")[0],
          ep_idx: demoAnnotations.ep_idx ?? Number(item.card.dataset.ep),
          num_frames: demoAnnotations.num_frames ?? item.trace.numFrames,
          stages: item.stages,
        }};
        buildAnnotationRows(item);
        bindAnnotationToolbar(item);
      }});
      rerenderAll();
      document.getElementById("annotation-status").textContent = "imported annotations";
    }});
    if (!traces.length) {{
      smoothingInput.disabled = true;
    }}
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
                        "annotation_key": f"{category}/demo_{int(ep_idx)}",
                        "video_path": str(dest_video.relative_to(args.output_root)),
                        "video_url": f"{dest_video.relative_to(args.output_root)}?v={int(dest_video.stat().st_mtime_ns)}",
                        "trace": trace,
                    }
                )
        rows_by_category[category] = rows

    annotations_payload = build_annotation_payload(rows_by_category["forward_grab"], rows_by_category["backward_grab"])
    build_page(
        args.output_root / "index.html",
        rows_by_category["forward_grab"],
        rows_by_category["backward_grab"],
        annotations_payload,
    )
    print(args.output_root / "index.html")


if __name__ == "__main__":
    main()
