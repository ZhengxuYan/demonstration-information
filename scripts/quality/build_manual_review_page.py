"""
Build a static review page for manually categorized demo videos.

Example:

python scripts/quality/build_manual_review_page.py \
    --review-root /Users/jasonyan/Desktop/demonstration-information/square_mh_wrist_review
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import pickle
import statistics as st
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-root", type=Path, required=True, help="Directory containing manifest.csv and category folders.")
    parser.add_argument("--output", type=Path, default=None, help="Output HTML path. Defaults to <review-root>/manual_index.html.")
    parser.add_argument(
        "--detailed-scores",
        type=Path,
        default=None,
        help="Optional path to a detailed quality-estimation pickle with sample_score, sample_ep_idx, and sample_step_idx.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> dict[int, dict[str, object]]:
    by_ep = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep_idx = int(row["ep_idx"])
            by_ep[ep_idx] = {
                "ep_idx": ep_idx,
                "pred_score": float(row["pred_score"]),
                "human_label": float(row["human_label"]),
                "video_path": row["video_path"],
                "num_frames": int(row["num_frames"]),
            }
    return by_ep


def parse_category(root: Path, name: str, by_ep: dict[int, dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for path in sorted((root / name).glob("*.mp4")):
        ep_idx = int(path.stem.split("_")[1])
        row = dict(by_ep[ep_idx])
        row["category"] = name
        row["manual_video_path"] = str(path.relative_to(root))
        rows.append(row)
    rows.sort(key=lambda row: (-float(row["pred_score"]), int(row["ep_idx"])))
    return rows


def load_detailed_traces(path: Path | None) -> dict[int, dict[str, object]]:
    if path is None or not path.exists():
        return {}

    with path.open("rb") as f:
        data = pickle.load(f)

    required = {"sample_score", "sample_ep_idx", "sample_step_idx"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"{path} is missing {sorted(missing)}")

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
        }
    return traces


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
    human_label = float(row["human_label"])
    num_frames = int(row["num_frames"])
    video_path = html.escape(str(row["manual_video_path"]))
    trace = row.get("trace")
    trace_html = ""
    if trace is not None:
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
      <video controls preload="metadata" src="{video_path}"></video>
      {trace_html}
      <div class="meta">
        <div class="title"><strong>demo_{ep_idx}</strong></div>
        <div>pred score: <span class="accent">{pred_score:.6f}</span></div>
        <div>human label: {human_label:.1f}</div>
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


def build_page(root: Path, output: Path, detailed_scores: Path | None = None) -> None:
    by_ep = load_manifest(root / "manifest.csv")
    traces = load_detailed_traces(detailed_scores)
    obs_full = parse_category(root, "obs_full", by_ep)
    obs_partial = parse_category(root, "obs_partial", by_ep)
    for rows in (obs_full, obs_partial):
        for row in rows:
            row["trace"] = traces.get(int(row["ep_idx"]))

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>square/mh wrist visibility review</title>
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
      --warn-soft: #efe2d0;
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
    h1 {{
      margin: 0 0 8px;
      font-size: 30px;
    }}
    .lede {{
      margin: 0;
      max-width: 900px;
      color: var(--muted);
      line-height: 1.5;
      font-size: 15px;
    }}
    main {{
      padding: 24px;
      display: grid;
      gap: 28px;
    }}
    .section {{
      display: grid;
      gap: 16px;
    }}
    .section-head {{
      display: grid;
      gap: 12px;
      align-items: start;
    }}
    h2 {{
      margin: 0 0 4px;
      font-size: 24px;
    }}
    .section-head p {{
      margin: 0;
      color: var(--muted);
      max-width: 800px;
      line-height: 1.5;
    }}
    .stat-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
      gap: 10px;
    }}
    .stat {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 10px 12px;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.04);
    }}
    .stat span {{
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 4px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .stat strong {{
      font-size: 18px;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
      gap: 16px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 16px;
      overflow: hidden;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
    }}
    video {{
      display: block;
      width: 100%;
      background: #000;
    }}
    .trace-wrap {{
      padding: 12px 14px 6px;
      display: grid;
      gap: 6px;
      border-top: 1px solid var(--border);
      border-bottom: 1px solid rgba(0, 0, 0, 0.04);
      background: linear-gradient(180deg, rgba(15, 91, 92, 0.03), rgba(15, 91, 92, 0.01));
    }}
    .trace {{
      width: 100%;
      height: 96px;
      overflow: visible;
    }}
    .trace-axis {{
      stroke: rgba(31, 27, 23, 0.45);
      stroke-width: 1.2;
    }}
    .trace-grid {{
      stroke: rgba(15, 91, 92, 0.12);
      stroke-width: 1;
      stroke-dasharray: 3 3;
    }}
    .trace-tick {{
      fill: var(--muted);
      font-size: 10px;
      text-anchor: end;
      font-family: "Iowan Old Style", "Palatino Linotype", serif;
    }}
    .trace-line {{
      fill: none;
      stroke: var(--accent);
      stroke-width: 2.5;
      stroke-linecap: round;
      stroke-linejoin: round;
    }}
    .trace-dot {{
      fill: #b9472c;
      stroke: white;
      stroke-width: 1.5;
    }}
    .trace-caption {{
      display: flex;
      justify-content: space-between;
      gap: 8px;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .trace-readout {{
      color: var(--accent);
      font-weight: 700;
    }}
    .meta {{
      display: grid;
      gap: 5px;
      padding: 12px 14px 14px;
      font-size: 14px;
    }}
    .title {{
      font-size: 16px;
    }}
    .path {{
      color: var(--muted);
      font-size: 12px;
      word-break: break-all;
    }}
    .accent {{
      color: var(--accent);
      font-weight: 700;
    }}
    .summary-banner {{
      background: var(--accent-soft);
      border: 1px solid rgba(15, 91, 92, 0.15);
      border-radius: 14px;
      padding: 14px 16px;
      line-height: 1.5;
    }}
  </style>
</head>
<body>
  <header>
    <h1>square/mh wrist visibility review</h1>
    <p class="lede">
      Manually categorized human-label-3 demonstrations from the wrist-only DemInf analysis. Videos are grouped by whether the wrist camera fully sees the square opening during the key manipulation phase (<strong>obs_full</strong>) or whether the opening is partially occluded by the handle (<strong>obs_partial</strong>).
    </p>
  </header>
  <main>
    <div class="summary-banner">
      <strong>Key result:</strong> the manually labeled <code>obs_full</code> demos have higher DemInf scores on average than <code>obs_partial</code> demos in this wrist-only setup.
    </div>
    {section_html("obs_full", "The square opening remains visible to the wrist camera after grasp, so the task-relevant geometry stays observable.", obs_full)}
    {section_html("obs_partial", "The opening sits behind the handle from the wrist viewpoint, so the task-relevant geometry is partially hidden during execution.", obs_partial)}
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
      const minStep = 0;
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

      return {{ steps, scores, points, xFor, yFor, maxStep }};
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
    output = args.output or (args.review_root / "manual_index.html")
    build_page(args.review_root, output, detailed_scores=args.detailed_scores)
    print(output)


if __name__ == "__main__":
    main()
