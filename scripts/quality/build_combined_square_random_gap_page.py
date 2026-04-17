"""Build a gap-focused DemInf review page from an existing combined review page.

The source page is expected to contain the embedded JSON payload produced by
build_combined_square_random_review_page.py. This page keeps the paired videos
but visualizes differences between camera-conditioned scores:

- agent - wrist
- both-view - wrist
- MH-only both-view - wrist, for Square MH demos only
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-html", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_payload(source_html: Path) -> dict[str, object]:
    text = source_html.read_text(encoding="utf-8")
    match = re.search(r'<script id="payload" type="application/json">(.*?)</script>', text, re.S)
    if match is None:
        raise ValueError(f"Could not find embedded payload in {source_html}")
    return json.loads(html.unescape(match.group(1)))


def build_html(payload: dict[str, object]) -> str:
    escaped_payload = html.escape(json.dumps(payload, separators=(",", ":")), quote=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DemInf View Gap Review</title>
  <style>
    :root {{
      --bg: #efe9db;
      --panel: #fffaf0;
      --ink: #1d1a16;
      --muted: #746d63;
      --border: #d7cdbc;
      --agent-gap: #b54a2a;
      --both-gap: #315f9c;
      --mh-gap: #b47a12;
      --shadow: rgba(32, 25, 18, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at 12% 0%, rgba(49, 95, 156, .12), transparent 30%),
        radial-gradient(circle at 90% 10%, rgba(180, 122, 18, .13), transparent 26%),
        linear-gradient(180deg, #fbf4e8 0%, var(--bg) 100%);
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 4;
      padding: 20px 24px 16px;
      border-bottom: 1px solid var(--border);
      background: rgba(255, 250, 240, .94);
      backdrop-filter: blur(10px);
    }}
    h1 {{ margin: 0 0 8px; font-size: clamp(25px, 3vw, 36px); letter-spacing: -.03em; }}
    .lede {{ max-width: 1120px; margin: 0; color: var(--muted); line-height: 1.45; }}
    .toolbar {{ display: flex; flex-wrap: wrap; align-items: center; gap: 10px; margin-top: 15px; }}
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
    main {{ padding: 22px 24px 34px; display: grid; gap: 22px; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 12px; }}
    .summary-card, .card {{
      background: rgba(255, 250, 240, .95);
      border: 1px solid var(--border);
      border-radius: 18px;
      box-shadow: 0 12px 32px var(--shadow);
    }}
    .summary-card {{ padding: 14px 16px; }}
    .summary-card h2 {{ margin: 0 0 8px; font-size: 18px; }}
    .summary-grid, .meta {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(108px, 1fr)); gap: 8px; }}
    .metric span {{ display: block; color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .055em; }}
    .metric strong {{ font-size: 18px; }}
    .section-title {{ display: flex; align-items: baseline; justify-content: space-between; gap: 14px; }}
    .section-title h2 {{ margin: 0; font-size: 24px; }}
    .section-title p {{ margin: 0; color: var(--muted); }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(390px, 1fr)); gap: 16px; }}
    .card {{ overflow: hidden; }}
    .videos {{ background: #050403; }}
    .video-panel {{ position: relative; background: #050403; }}
    .video-panel span {{
      position: absolute;
      top: 8px;
      left: 8px;
      z-index: 1;
      padding: 3px 7px;
      border-radius: 999px;
      background: rgba(5, 4, 3, .68);
      color: #fffaf0;
      font-size: 12px;
    }}
    video {{
      display: block;
      width: min(100%, 336px);
      aspect-ratio: 2 / 1;
      object-fit: contain;
      background: #050403;
      margin: 0 auto;
      image-rendering: pixelated;
      image-rendering: crisp-edges;
    }}
    .plot {{ padding: 12px 14px 8px; border-top: 1px solid var(--border); background: linear-gradient(180deg, rgba(49,95,156,.05), transparent); }}
    svg {{ width: 100%; height: 150px; display: block; overflow: visible; }}
    .grid {{ stroke: rgba(29,26,22,.12); stroke-width: 1; stroke-dasharray: 4 4; }}
    .zero {{ stroke: rgba(29,26,22,.5); stroke-width: 1.4; }}
    .axis-label {{ fill: var(--muted); font-size: 10px; }}
    .line {{ fill: none; stroke-width: 2.5; stroke-linecap: round; stroke-linejoin: round; }}
    .agent-gap-line {{ stroke: var(--agent-gap); }}
    .both-gap-line {{ stroke: var(--both-gap); }}
    .mh-gap-line {{ stroke: var(--mh-gap); stroke-dasharray: 5 4; }}
    .playhead {{ stroke: #16120e; stroke-width: 2; opacity: .76; }}
    .legend {{ display: flex; flex-wrap: wrap; justify-content: space-between; gap: 8px 12px; color: var(--muted); font-size: 12px; }}
    .swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 999px; margin-right: 5px; }}
    .agent-gap {{ color: var(--agent-gap); }}
    .both-gap {{ color: var(--both-gap); }}
    .mh-gap {{ color: var(--mh-gap); }}
    .swatch.agent-gap {{ background: var(--agent-gap); }}
    .swatch.both-gap {{ background: var(--both-gap); }}
    .swatch.mh-gap {{ background: var(--mh-gap); }}
    .body {{ padding: 12px 14px 15px; display: grid; gap: 9px; }}
    .title {{ display: flex; justify-content: space-between; gap: 10px; align-items: baseline; }}
    .title strong {{ font-size: 17px; }}
    .pill {{ border-radius: 999px; background: #ddeee8; padding: 4px 8px; color: #17433f; font-size: 12px; white-space: nowrap; }}
    .path {{ color: var(--muted); font-size: 12px; word-break: break-all; }}
    @media (max-width: 700px) {{
      header, main {{ padding-left: 14px; padding-right: 14px; }}
      .cards {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>DemInf View Gap Review</h1>
    <p class="lede">
      This page focuses on disagreement between observation settings. Positive gaps mean the fuller view scored
      the transition/demo higher than wrist-only. The most useful curves for partial observability are
      <strong>both-view - wrist</strong> and, for Square MH, <strong>MH-only both - wrist</strong>.
    </p>
    <div class="toolbar">
      <button class="filter active" data-category="all">All</button>
      <button class="filter" data-category="obs_full">MH obs_full</button>
      <button class="filter" data-category="obs_partial">MH obs_partial</button>
      <button class="filter" data-category="manual_collected">Collected</button>
      <select id="sort">
        <option value="gap_abs">Sort by max |both - wrist|</option>
        <option value="both_gap_desc">Both - wrist high to low</option>
        <option value="agent_gap_desc">Agent - wrist high to low</option>
        <option value="mh_gap_desc">MH-only both - wrist high to low</option>
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

    function smoothingWindow() {{
      const raw = Number(document.getElementById('smooth-window').value);
      if (!Number.isFinite(raw) || raw <= 1) return 1;
      return Math.max(1, Math.floor(raw));
    }}

    function smoothTrace(trace, window) {{
      if (!trace || !trace.scores || trace.scores.length <= 2 || window <= 1) return trace;
      const half = Math.floor(window / 2);
      return {{
        steps: trace.steps,
        scores: trace.scores.map((_, idx) => {{
          const start = Math.max(0, idx - half);
          const end = Math.min(trace.scores.length, idx + half + 1);
          let total = 0;
          for (let i = start; i < end; i++) total += trace.scores[i];
          return total / (end - start);
        }})
      }};
    }}

    function diffTrace(a, b) {{
      if (!a || !b || !a.steps || !b.steps) return null;
      const bByStep = new Map(b.steps.map((step, i) => [step, b.scores[i]]));
      const steps = [];
      const scores = [];
      a.steps.forEach((step, i) => {{
        if (!bByStep.has(step)) return;
        steps.push(step);
        scores.push(a.scores[i] - bByStep.get(step));
      }});
      return steps.length ? {{steps, scores}} : null;
    }}

    function pathFromTrace(trace, xMin, xMax, yMin, yMax) {{
      if (!trace || !trace.steps || trace.steps.length === 0) return '';
      const left = 38, right = 350, top = 12, bottom = 130;
      const xSpan = Math.max(1, xMax - xMin);
      const ySpan = Math.max(1e-6, yMax - yMin);
      return trace.steps.map((step, i) => {{
        const x = left + ((step - xMin) / xSpan) * (right - left);
        const y = bottom - ((trace.scores[i] - yMin) / ySpan) * (bottom - top);
        return `${{i === 0 ? 'M' : 'L'}} ${{x.toFixed(2)}} ${{y.toFixed(2)}}`;
      }}).join(' ');
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

    function maxAbs(trace) {{
      return trace && trace.scores.length ? Math.max(...trace.scores.map(v => Math.abs(v))) : null;
    }}

    function meanVal(trace) {{
      return trace && trace.scores.length ? trace.scores.reduce((a, b) => a + b, 0) / trace.scores.length : null;
    }}

    function gapTraces(row) {{
      const window = smoothingWindow();
      const wrist = smoothTrace(row.wrist_trace, window);
      return {{
        agentGap: diffTrace(smoothTrace(row.agent_trace, window), wrist),
        bothGap: diffTrace(smoothTrace(row.both_trace, window), wrist),
        mhGap: diffTrace(smoothTrace(row.mh_only_both_trace, window), wrist)
      }};
    }}

    function traceSvg(row) {{
      const {{agentGap, bothGap, mhGap}} = gapTraces(row);
      const traces = [agentGap, bothGap, mhGap].filter(Boolean);
      if (!traces.length) return '<div class="plot">No gap trace available.</div>';
      const allSteps = traces.flatMap(t => t.steps);
      const allScores = traces.flatMap(t => t.scores);
      const xMin = 0, xMax = Math.max(row.num_frames - 1, ...allSteps, 1);
      let yMin = Math.min(...allScores, 0), yMax = Math.max(...allScores, 0);
      if (yMin === yMax) {{ yMin -= .1; yMax += .1; }}
      const zeroY = 130 - ((0 - yMin) / Math.max(1e-6, yMax - yMin)) * (130 - 12);
      return `
        <div class="plot">
          <svg viewBox="0 0 386 150" preserveAspectRatio="none" aria-label="view gap traces">
            <line class="grid" x1="38" y1="12" x2="350" y2="12"></line>
            <line class="grid" x1="38" y1="71" x2="350" y2="71"></line>
            <line class="grid" x1="38" y1="130" x2="350" y2="130"></line>
            <line class="zero" x1="38" y1="${{zeroY.toFixed(2)}}" x2="350" y2="${{zeroY.toFixed(2)}}"></line>
            <text class="axis-label" x="4" y="16">${{fmt(yMax, 2)}}</text>
            <text class="axis-label" x="4" y="${{zeroY.toFixed(2)}}">0</text>
            <text class="axis-label" x="4" y="134">${{fmt(yMin, 2)}}</text>
            <path class="line agent-gap-line" d="${{pathFromTrace(agentGap, xMin, xMax, yMin, yMax)}}"></path>
            <path class="line both-gap-line" d="${{pathFromTrace(bothGap, xMin, xMax, yMin, yMax)}}"></path>
            <path class="line mh-gap-line" d="${{pathFromTrace(mhGap, xMin, xMax, yMin, yMax)}}"></path>
            <line class="playhead" x1="38" y1="8" x2="38" y2="134"></line>
          </svg>
          <div class="legend">
            <span><span class="swatch agent-gap"></span>agent - wrist, slope/100: <strong>${{fmt(slopePer100(agentGap), 3)}}</strong></span>
            <span><span class="swatch both-gap"></span>both - wrist, slope/100: <strong>${{fmt(slopePer100(bothGap), 3)}}</strong></span>
            ${{mhGap ? `<span><span class="swatch mh-gap"></span>MH-only both - wrist, slope/100: <strong>${{fmt(slopePer100(mhGap), 3)}}</strong></span>` : ''}}
          </div>
        </div>`;
    }}

    function scalarGaps(row) {{
      return {{
        agentGap: row.agent_score == null || row.wrist_score == null ? null : row.agent_score - row.wrist_score,
        bothGap: row.both_score == null || row.wrist_score == null ? null : row.both_score - row.wrist_score,
        mhGap: row.mh_only_both_score == null || row.wrist_score == null ? null : row.mh_only_both_score - row.wrist_score
      }};
    }}

    function card(row) {{
      const gaps = scalarGaps(row);
      return `
        <article class="card" data-category="${{row.category}}" data-demo="${{row.ep_idx}}">
          <div class="videos">
            <div class="video-panel"><span>wrist view | agent view</span><video controls preload="metadata" src="${{row.paired_video}}" data-category="${{row.category}}" data-role="wrist"></video></div>
          </div>
          ${{traceSvg(row)}}
          <div class="body">
            <div class="title"><strong>${{row.title}}</strong><span class="pill">${{categoryNames[row.category]}}</span></div>
            <div class="meta">
              <div class="metric"><span>agent - wrist</span><strong class="agent-gap">${{fmt(gaps.agentGap)}}</strong></div>
              <div class="metric"><span>both - wrist</span><strong class="both-gap">${{fmt(gaps.bothGap)}}</strong></div>
              <div class="metric"><span>MH-only both - wrist</span><strong class="mh-gap">${{fmt(gaps.mhGap)}}</strong></div>
              <div class="metric"><span>max |both - wrist|</span><strong>${{fmt(maxAbs(gapTraces(row).bothGap))}}</strong></div>
              <div class="metric"><span>mean both gap</span><strong>${{fmt(meanVal(gapTraces(row).bothGap))}}</strong></div>
              <div class="metric"><span>dataset ep</span><strong>${{row.ep_idx}}</strong></div>
            </div>
            <div class="path">video: ${{row.paired_video}}</div>
          </div>
        </article>`;
    }}

    function collectSummary(category) {{
      const rows = payload.rows.filter(row => row.category === category);
      const vals = rows.map(row => scalarGaps(row).bothGap).filter(v => v !== null && !Number.isNaN(v));
      const mhVals = rows.map(row => scalarGaps(row).mhGap).filter(v => v !== null && !Number.isNaN(v));
      const mean = arr => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : null;
      const max = arr => arr.length ? Math.max(...arr) : null;
      return {{count: rows.length, bothMean: mean(vals), bothMax: max(vals), mhMean: mean(mhVals), mhMax: max(mhVals)}};
    }}

    function renderSummary() {{
      const el = document.getElementById('summary');
      el.innerHTML = ['obs_full', 'obs_partial', 'manual_collected'].map(category => {{
        const s = collectSummary(category);
        return `
          <article class="summary-card">
            <h2>${{categoryNames[category]}}</h2>
            <div class="summary-grid">
              <div class="metric"><span>count</span><strong>${{s.count}}</strong></div>
              <div class="metric"><span>mean both - wrist</span><strong class="both-gap">${{fmt(s.bothMean)}}</strong></div>
              <div class="metric"><span>max both - wrist</span><strong class="both-gap">${{fmt(s.bothMax)}}</strong></div>
              <div class="metric"><span>mean MH-only gap</span><strong class="mh-gap">${{fmt(s.mhMean)}}</strong></div>
            </div>
          </article>`;
      }}).join('');
    }}

    function sortedRows(rows) {{
      const mode = document.getElementById('sort').value;
      const copy = [...rows];
      const gap = row => scalarGaps(row);
      if (mode === 'both_gap_desc') copy.sort((a, b) => (gap(b).bothGap ?? -Infinity) - (gap(a).bothGap ?? -Infinity));
      else if (mode === 'agent_gap_desc') copy.sort((a, b) => (gap(b).agentGap ?? -Infinity) - (gap(a).agentGap ?? -Infinity));
      else if (mode === 'mh_gap_desc') copy.sort((a, b) => (gap(b).mhGap ?? -Infinity) - (gap(a).mhGap ?? -Infinity));
      else if (mode === 'demo') copy.sort((a, b) => a.ep_idx - b.ep_idx || a.category.localeCompare(b.category));
      else copy.sort((a, b) => (maxAbs(gapTraces(b).bothGap) ?? -Infinity) - (maxAbs(gapTraces(a).bothGap) ?? -Infinity));
      return copy;
    }}

    function renderCards() {{
      const query = document.getElementById('search').value.toLowerCase().trim();
      let rows = payload.rows.filter(row => currentCategory === 'all' || row.category === currentCategory);
      if (query) rows = rows.filter(row => `${{row.title}} ${{row.category}} ${{row.ep_idx}}`.toLowerCase().includes(query));
      rows = sortedRows(rows);
      document.getElementById('cards').innerHTML = rows.map(card).join('');
      document.getElementById('result-title').textContent = currentCategory === 'all' ? 'All samples' : categoryNames[currentCategory];
      document.getElementById('result-count').textContent = `${{rows.length}} visible cards`;
      attachVideoSync();
    }}

    function manualPlaybackRate() {{
      return Number(document.getElementById('manual-speed').value);
    }}

    function applyPlaybackRate(video) {{
      video.playbackRate = video.dataset.category === 'manual_collected' ? manualPlaybackRate() : 1;
    }}

    function updatePlayhead(cardEl) {{
      const video = cardEl.querySelector('video');
      const playhead = cardEl.querySelector('.playhead');
      if (!video || !playhead) return;
      const duration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : 0;
      const progress = duration ? Math.min(1, Math.max(0, video.currentTime / duration)) : 0;
      const x = 38 + progress * (350 - 38);
      playhead.setAttribute('x1', x.toFixed(2));
      playhead.setAttribute('x2', x.toFixed(2));
    }}

    function syncLoop(video, cardEl) {{
      updatePlayhead(cardEl);
      if (!video.paused && !video.ended) {{
        if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(() => syncLoop(video, cardEl));
        else requestAnimationFrame(() => syncLoop(video, cardEl));
      }}
    }}

    function attachVideoSync() {{
      document.querySelectorAll('.card').forEach(cardEl => {{
        const video = cardEl.querySelector('video');
        if (!video) return;
        applyPlaybackRate(video);
        video.addEventListener('loadedmetadata', () => {{ applyPlaybackRate(video); updatePlayhead(cardEl); }});
        video.addEventListener('play', () => syncLoop(video, cardEl));
        video.addEventListener('pause', () => updatePlayhead(cardEl));
        video.addEventListener('seeked', () => updatePlayhead(cardEl));
        video.addEventListener('timeupdate', () => updatePlayhead(cardEl));
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
    payload = load_payload(args.source_html)
    args.output.write_text(build_html(payload), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Rows: {len(payload['rows'])}")  # type: ignore[index]


if __name__ == "__main__":
    main()
