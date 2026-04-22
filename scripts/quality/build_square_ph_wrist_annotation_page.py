"""
Build a local HTML annotation page for Square PH wrist-view observability labels.

Example:

python scripts/quality/build_square_ph_wrist_annotation_page.py \
  --scores-root /Users/jasonyan/Desktop/demonstration-information/square_ph_wrist_scores \
  --hdf5 /Users/jasonyan/Desktop/demonstration-information/robomimic_square_ph/image.hdf5 \
  --output-dir /Users/jasonyan/Desktop/demonstration-information/square_ph_wrist_annotation_deploy
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REVIEW_BUILDER = REPO_ROOT / "scripts" / "quality" / "build_square_ph_wrist_review_page.py"
spec = importlib.util.spec_from_file_location("square_ph_wrist_review_builder", REVIEW_BUILDER)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load review builder from {REVIEW_BUILDER}")
review_builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review_builder)
export_videos = review_builder.export_videos
load_score_bundle = review_builder.load_score_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-root", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-trace-points", type=int, default=0)
    parser.add_argument("--max-demos", type=int, default=0, help="0 means all scored trajectories.")
    return parser.parse_args()


def common_episode_indices(scores: dict[str, dict[str, object]], max_demos: int) -> list[int]:
    image_scores = scores["image_only"]["ep_scores"]
    proprio_scores = scores["image_proprio"]["ep_scores"]
    common = sorted(set(image_scores) & set(proprio_scores))
    if max_demos > 0:
        return common[:max_demos]
    return common


def build_html(
    output_dir: Path,
    scores: dict[str, dict[str, object]],
    videos: dict[int, dict[str, object]],
) -> None:
    rows = []
    image_scores = scores["image_only"]["ep_scores"]
    proprio_scores = scores["image_proprio"]["ep_scores"]
    labels = scores["image_only"].get("quality_by_ep_idx", {})

    for ep_idx in sorted(videos):
        rows.append(
            {
                "ep_idx": ep_idx,
                "video": videos[ep_idx]["video"],
                "num_frames": videos[ep_idx]["num_frames"],
                "fps": videos[ep_idx]["fps"],
                "label": labels.get(ep_idx),
                "image_score": image_scores[ep_idx],
                "proprio_score": proprio_scores[ep_idx],
                "gap": proprio_scores[ep_idx] - image_scores[ep_idx],
                "image_trace": scores["image_only"]["traces"].get(ep_idx, {}),
                "proprio_trace": scores["image_proprio"]["traces"].get(ep_idx, {}),
            }
        )

    payload_json = json.dumps({"rows": rows})

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Square PH Observability Annotation</title>
  <style>
    :root {{
      --bg: #ece6d8;
      --panel: #fffaf0;
      --ink: #1d1814;
      --muted: #6e655d;
      --border: #d9cfbf;
      --accent: #0f6d67;
      --accent-2: #b8572a;
      --shadow: 0 16px 40px rgba(36, 26, 19, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15, 109, 103, 0.08), transparent 28%),
        linear-gradient(180deg, #f8f1e4 0%, var(--bg) 100%);
      font-family: "Iowan Old Style", "Palatino Linotype", serif;
    }}
    header {{
      padding: 32px 28px 18px;
      border-bottom: 1px solid rgba(0, 0, 0, 0.06);
    }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    p.sub {{
      margin: 0;
      color: var(--muted);
      max-width: 900px;
      line-height: 1.45;
    }}
    .toolbar {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      padding: 18px 28px 12px;
      align-items: end;
    }}
    .toolbar label {{
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 13px;
    }}
    .toolbar input, .toolbar select, .toolbar button, .toolbar textarea {{
      width: 100%;
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.95);
      color: var(--ink);
      border-radius: 10px;
      padding: 10px 12px;
      font: inherit;
    }}
    .toolbar button {{
      cursor: pointer;
      font-weight: 600;
      background: #f4ecde;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: 10px;
      padding: 0 28px 18px;
    }}
    .metric {{
      background: rgba(255,255,255,0.66);
      border: 1px solid rgba(0,0,0,0.06);
      border-radius: 14px;
      padding: 12px 14px;
      box-shadow: var(--shadow);
    }}
    .metric span {{
      display: block;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 4px;
    }}
    .metric strong {{ font-size: 20px; }}
    .status {{
      padding: 0 28px 16px;
      color: var(--muted);
      font-size: 13px;
    }}
    .cards {{
      padding: 0 22px 28px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 18px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid rgba(0,0,0,0.07);
      border-radius: 18px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }}
    .video-panel {{
      padding: 14px;
      background: linear-gradient(180deg, rgba(15,109,103,.08), rgba(255,255,255,0));
    }}
    .video-panel span {{
      display: inline-block;
      margin-bottom: 8px;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    video {{
      width: 100%;
      aspect-ratio: 1 / 1;
      display: block;
      border-radius: 14px;
      background: #0f0c0a;
    }}
    .plot {{ padding: 12px 14px 8px; border-top: 1px solid var(--border); background: linear-gradient(180deg, rgba(15,109,103,.04), transparent); }}
    .line {{ fill: none; stroke-width: 2.3; stroke-linecap: round; stroke-linejoin: round; }}
    .image-line {{ stroke: var(--accent); }}
    .proprio-line {{ stroke: var(--accent-2); }}
    .gridline {{ stroke: rgba(0,0,0,0.08); stroke-width: 1; }}
    .zero {{ stroke: rgba(0,0,0,0.18); stroke-dasharray: 4 4; stroke-width: 1; }}
    .axis-label {{ fill: var(--muted); font-size: 10px; }}
    .playhead {{ stroke: #16120e; stroke-width: 2; opacity: .78; }}
    .legend {{
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      font-size: 12px;
      color: var(--muted);
      margin-top: 8px;
    }}
    .swatch {{
      width: 11px;
      height: 11px;
      display: inline-block;
      border-radius: 999px;
      margin-right: 6px;
      vertical-align: middle;
    }}
    .swatch.image {{ background: var(--accent); }}
    .swatch.proprio {{ background: var(--accent-2); }}
    .body {{ padding: 14px; display: grid; gap: 12px; }}
    .title {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
    }}
    .pill {{
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 12px;
      color: var(--muted);
      background: rgba(255,255,255,0.65);
    }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
    }}
    .meta .metric {{
      box-shadow: none;
      background: rgba(255,255,255,0.56);
      padding: 10px 12px;
    }}
    .annotation {{
      display: grid;
      gap: 10px;
      border-top: 1px solid var(--border);
      padding-top: 12px;
    }}
    .annotation-group {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .choice {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 6px 10px;
      background: rgba(255,255,255,0.7);
      font-size: 13px;
    }}
    .choice input {{ width: auto; margin: 0; }}
    .path {{
      font-size: 12px;
      color: var(--muted);
      word-break: break-all;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Square PH Wrist Observability Annotation</h1>
    <p class="sub">
      Local annotation tool for all scored Square PH wrist-view demos. Label each trajectory as full observability,
      partial observability, or leave it unlabeled. Labels are stored in browser local storage and can be exported/imported as JSON or CSV.
    </p>
  </header>

  <section class="toolbar">
    <label>Sort
      <select id="sort">
        <option value="demo">demo id</option>
        <option value="image_score">image only score</option>
        <option value="proprio_score">image + proprio score</option>
        <option value="gap">score gap</option>
        <option value="annotation">annotation status</option>
      </select>
    </label>
    <label>Filter
      <select id="filter">
        <option value="all">all</option>
        <option value="unlabeled">unlabeled</option>
        <option value="full">full observability</option>
        <option value="partial">partial observability</option>
      </select>
    </label>
    <label>Search demo id
      <input id="search" placeholder="e.g. 42">
    </label>
    <label>Export JSON
      <button id="export-json" type="button">download json</button>
    </label>
    <label>Export CSV
      <button id="export-csv" type="button">download csv</button>
    </label>
    <label>Import JSON
      <button id="import-trigger" type="button">import json</button>
      <input id="import-json" type="file" accept=".json,application/json" style="display:none">
    </label>
  </section>

  <section class="summary" id="summary"></section>
  <div class="status" id="status">no local edits yet</div>
  <main class="cards" id="cards"></main>

  <script>
    const DATA = {payload_json};
    const STORAGE_KEY = "square_ph_observability_annotations_v1";
    const cards = document.getElementById("cards");

    function fmt(x, digits = 4) {{
      return Number.isFinite(x) ? x.toFixed(digits) : "n/a";
    }}

    function loadAnnotations() {{
      try {{
        return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{{}}");
      }} catch (err) {{
        console.warn("failed to parse local annotations", err);
        return {{}};
      }}
    }}

    let annotations = loadAnnotations();

    function annotationFor(epIdx) {{
      return annotations[String(epIdx)] || {{ label: "unlabeled", note: "" }};
    }}

    function saveAnnotations(statusText) {{
      localStorage.setItem(STORAGE_KEY, JSON.stringify(annotations));
      document.getElementById("status").textContent = statusText || "saved locally";
      renderSummary(DATA.rows);
    }}

    function escapeHtml(text) {{
      return String(text)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }}

    function pathFromTrace(trace, xMin, xMax, yMin, yMax) {{
      if (!trace || !trace.steps || trace.steps.length === 0) return "";
      const left = 36, right = 344, top = 12, bottom = 108;
      const xSpan = Math.max(1e-6, xMax - xMin);
      const ySpan = Math.max(1e-6, yMax - yMin);
      return trace.steps.map((step, i) => {{
        const x = left + ((step - xMin) / xSpan) * (right - left);
        const y = bottom - ((trace.scores[i] - yMin) / ySpan) * (bottom - top);
        return `${{i === 0 ? "M" : "L"}}${{x.toFixed(2)}},${{y.toFixed(2)}}`;
      }}).join(" ");
    }}

    function traceSvg(row) {{
      const imageTrace = row.image_trace;
      const proprioTrace = row.proprio_trace;
      const traces = [imageTrace, proprioTrace].filter(t => t && t.steps && t.steps.length);
      if (!traces.length) return '<div class="plot">No transition trace available.</div>';
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
            <span><span class="swatch image"></span>image only</span>
            <span><span class="swatch proprio"></span>image + proprio</span>
          </div>
        </div>`;
    }}

    function card(row) {{
      const ann = annotationFor(row.ep_idx);
      return `
        <article class="card" data-demo="${{row.ep_idx}}" data-annotation="${{ann.label}}">
          <div class="video-panel">
            <span>wrist view</span>
            <video controls preload="metadata" src="${{row.video}}" data-role="wrist"></video>
          </div>
          ${{traceSvg(row)}}
          <div class="body">
            <div class="title"><strong>demo_${{String(row.ep_idx).padStart(4, "0")}}</strong><span class="pill">Square PH</span></div>
            <div class="meta">
              <div class="metric"><span>image only</span><strong>${{fmt(row.image_score)}}</strong></div>
              <div class="metric"><span>image + proprio</span><strong>${{fmt(row.proprio_score)}}</strong></div>
              <div class="metric"><span>gap</span><strong>${{fmt(row.gap)}}</strong></div>
              <div class="metric"><span>label</span><strong>${{row.label ?? "n/a"}}</strong></div>
              <div class="metric"><span>frames</span><strong>${{row.num_frames}}</strong></div>
              <div class="metric"><span>current</span><strong class="current-frame">0</strong></div>
            </div>
            <div class="annotation">
              <div class="annotation-group">
                <label class="choice"><input type="radio" name="obs_${{row.ep_idx}}" value="unlabeled" ${{ann.label === "unlabeled" ? "checked" : ""}}>unlabeled</label>
                <label class="choice"><input type="radio" name="obs_${{row.ep_idx}}" value="full" ${{ann.label === "full" ? "checked" : ""}}>full observability</label>
                <label class="choice"><input type="radio" name="obs_${{row.ep_idx}}" value="partial" ${{ann.label === "partial" ? "checked" : ""}}>partial observability</label>
              </div>
              <textarea rows="2" placeholder="optional note">${{escapeHtml(ann.note || "")}}</textarea>
            </div>
            <div class="path">wrist: ${{row.video}}</div>
          </div>
        </article>`;
    }}

    function renderSummary(rows) {{
      const values = Object.values(annotations);
      const full = values.filter(v => v.label === "full").length;
      const partial = values.filter(v => v.label === "partial").length;
      const unlabeled = rows.length - full - partial;
      document.getElementById("summary").innerHTML = `
        <div class="metric"><span>total demos</span><strong>${{rows.length}}</strong></div>
        <div class="metric"><span>full</span><strong>${{full}}</strong></div>
        <div class="metric"><span>partial</span><strong>${{partial}}</strong></div>
        <div class="metric"><span>unlabeled</span><strong>${{unlabeled}}</strong></div>
      `;
    }}

    function rowMatchesFilter(row) {{
      const query = document.getElementById("search").value.toLowerCase().trim();
      const filter = document.getElementById("filter").value;
      const ann = annotationFor(row.ep_idx);
      if (query && !String(row.ep_idx).includes(query)) return false;
      if (filter !== "all" && ann.label !== filter) return false;
      return true;
    }}

    function sortedRows(rows) {{
      const mode = document.getElementById("sort").value;
      const copy = rows.filter(rowMatchesFilter);
      if (mode === "image_score") copy.sort((a, b) => b.image_score - a.image_score);
      else if (mode === "proprio_score") copy.sort((a, b) => b.proprio_score - a.proprio_score);
      else if (mode === "gap") copy.sort((a, b) => Math.abs(b.gap) - Math.abs(a.gap));
      else if (mode === "annotation") copy.sort((a, b) => annotationFor(a.ep_idx).label.localeCompare(annotationFor(b.ep_idx).label) || a.ep_idx - b.ep_idx);
      else copy.sort((a, b) => a.ep_idx - b.ep_idx);
      return copy;
    }}

    function frameIndexForVideo(video, row, mediaTime) {{
      const fps = row.fps || 20;
      const sourceTime = Number.isFinite(mediaTime) ? mediaTime : video.currentTime;
      return Math.min(row.num_frames - 1, Math.max(0, Math.round(sourceTime * fps)));
    }}

    function updatePlayhead(cardEl, mediaTime) {{
      const row = DATA.rows.find(r => String(r.ep_idx) === cardEl.dataset.demo);
      const video = cardEl.querySelector("video[data-role='wrist']");
      const playhead = cardEl.querySelector(".playhead");
      const currentFrame = cardEl.querySelector(".current-frame");
      if (!row || !video || !playhead) return;
      const frameIdx = frameIndexForVideo(video, row, mediaTime);
      const x = 36 + ((frameIdx / Math.max(1, row.num_frames - 1)) * (344 - 36));
      playhead.setAttribute("x1", x.toFixed(2));
      playhead.setAttribute("x2", x.toFixed(2));
      if (currentFrame) currentFrame.textContent = String(frameIdx);
    }}

    function syncLoop(video, cardEl) {{
      if (!video.paused && !video.ended) {{
        if (video.requestVideoFrameCallback) {{
          video.requestVideoFrameCallback((now, metadata) => {{
            updatePlayhead(cardEl, metadata.mediaTime);
            syncLoop(video, cardEl);
          }});
        }} else {{
          requestAnimationFrame(() => {{
            updatePlayhead(cardEl);
            syncLoop(video, cardEl);
          }});
        }}
      }} else {{
        updatePlayhead(cardEl);
      }}
    }}

    function bindCard(cardEl) {{
      const epIdx = cardEl.dataset.demo;
      const radioButtons = cardEl.querySelectorAll(`input[name="obs_${{epIdx}}"]`);
      const textarea = cardEl.querySelector("textarea");
      const video = cardEl.querySelector("video[data-role='wrist']");

      radioButtons.forEach(input => {{
        input.addEventListener("change", () => {{
          annotations[epIdx] = annotations[epIdx] || {{ label: "unlabeled", note: "" }};
          annotations[epIdx].label = input.value;
          cardEl.dataset.annotation = input.value;
          saveAnnotations(`saved local label for demo_${{String(epIdx).padStart(4, "0")}}`);
          render();
        }});
      }});

      textarea.addEventListener("change", () => {{
        annotations[epIdx] = annotations[epIdx] || {{ label: "unlabeled", note: "" }};
        annotations[epIdx].note = textarea.value;
        saveAnnotations(`saved local note for demo_${{String(epIdx).padStart(4, "0")}}`);
      }});

      if (video) {{
        video.addEventListener("loadedmetadata", () => updatePlayhead(cardEl));
        video.addEventListener("play", () => syncLoop(video, cardEl));
        video.addEventListener("pause", () => updatePlayhead(cardEl));
        video.addEventListener("seeked", () => updatePlayhead(cardEl));
        video.addEventListener("timeupdate", () => updatePlayhead(cardEl));
        updatePlayhead(cardEl);
      }}
    }}

    function exportJson() {{
      const blob = new Blob([JSON.stringify(annotations, null, 2)], {{ type: "application/json" }});
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "square_ph_observability_annotations.json";
      link.click();
      URL.revokeObjectURL(url);
    }}

    function exportCsv() {{
      const lines = ["ep_idx,label,note"];
      DATA.rows.forEach(row => {{
        const ann = annotationFor(row.ep_idx);
        const note = (ann.note || "").replaceAll('"', '""');
        lines.push(`${{row.ep_idx}},${{ann.label}},"${{note}}"`);
      }});
      const blob = new Blob([lines.join("\\n")], {{ type: "text/csv" }});
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "square_ph_observability_annotations.csv";
      link.click();
      URL.revokeObjectURL(url);
    }}

    async function importJson(file) {{
      const imported = JSON.parse(await file.text());
      annotations = imported;
      saveAnnotations("imported annotations");
      render();
    }}

    function render() {{
      const rows = sortedRows(DATA.rows);
      renderSummary(DATA.rows);
      cards.innerHTML = rows.map(card).join("");
      document.querySelectorAll(".card").forEach(bindCard);
    }}

    document.getElementById("sort").addEventListener("change", render);
    document.getElementById("filter").addEventListener("change", render);
    document.getElementById("search").addEventListener("input", render);
    document.getElementById("export-json").addEventListener("click", exportJson);
    document.getElementById("export-csv").addEventListener("click", exportCsv);
    document.getElementById("import-trigger").addEventListener("click", () => document.getElementById("import-json").click());
    document.getElementById("import-json").addEventListener("change", async (event) => {{
      const file = event.target.files && event.target.files[0];
      if (file) await importJson(file);
    }});
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
        for method in ("image_only", "image_proprio")
    }
    ep_indices = common_episode_indices(scores, args.max_demos)
    videos = export_videos(args.hdf5, args.output_dir, ep_indices, args.fps)
    build_html(args.output_dir, scores, videos)
    print(args.output_dir / "index.html")


if __name__ == "__main__":
    main()
