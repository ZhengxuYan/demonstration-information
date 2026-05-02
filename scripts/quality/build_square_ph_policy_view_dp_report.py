#!/usr/bin/env python3
"""Build a static report for Diffusion Policy policy-view experiments."""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
from pathlib import Path


RUN_SETS = {
    "square_ph": {
        "title": "Square PH Diffusion Policy View Report",
        "lede": (
            "Two finished DP runs compared by rollout success, training curves, checkpoints, "
            "and generated rollout media. This page reports policy performance rather than likelihood."
        ),
        "output_dir": "square_ph_policy_view_dp_report",
        "json_name": "square_ph_policy_view_dp_report.json",
        "runs": [
            {
                "key": "agent_wrist",
                "label": "DP / agent+wrist",
                "dir": "square_ph_dp_agent_wrist_abs_50_seed42",
                "color": "#0f6d67",
            },
            {
                "key": "left_close_low_wrist",
                "label": "DP / left-close-low+wrist",
                "dir": "square_ph_dp_left_close_low_wrist_abs_50_seed42",
                "color": "#b54a2a",
            },
        ],
    },
    "expert200": {
        "title": "Expert200 Diffusion Policy View Report",
        "lede": (
            "Two expert200 DP runs compared by rollout success, training curves, checkpoints, "
            "and generated rollout media after converting actions to the robomimic absolute-action convention."
        ),
        "output_dir": "expert200_policy_view_dp_report",
        "json_name": "expert200_policy_view_dp_report.json",
        "runs": [
            {
                "key": "agent_wrist",
                "label": "Expert200 DP / agent+wrist",
                "dir": "expert200_dp_agent_wrist_abs_212",
                "color": "#0f6d67",
            },
            {
                "key": "left_close_low_wrist",
                "label": "Expert200 DP / left-close-low+wrist",
                "dir": "expert200_dp_left_close_low_wrist_abs_212",
                "color": "#b54a2a",
            },
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-set", choices=sorted(RUN_SETS), default="square_ph")
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("/iris/u/jasonyan/data/diffusion_policy_outputs/policy_view_experiments"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-videos", type=int, default=28)
    return parser.parse_args()


def read_json_lines(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def checkpoint_score(path: Path) -> float | None:
    match = re.search(r"test_mean_score=([0-9]+(?:\.[0-9]+)?)", path.name)
    return float(match.group(1)) if match else None


def trace(rows: list[dict[str, object]], key: str) -> list[dict[str, float]]:
    out = []
    for row in rows:
        if key in row:
            x = row.get("epoch", row.get("global_step", len(out)))
            out.append({"x": float(x), "y": float(row[key])})
    return out


def final_with_key(rows: list[dict[str, object]], key: str) -> float | None:
    for row in reversed(rows):
        if key in row:
            return float(row[key])
    return None


def final_test_rewards(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    final = rows[-1] if rows else {}
    rewards = []
    for key, value in final.items():
        if str(key).startswith("test/sim_max_reward_"):
            seed = int(str(key).rsplit("_", 1)[-1])
            rewards.append({"seed": seed, "reward": float(value)})
    rewards.sort(key=lambda item: item["seed"])
    return rewards


def copy_videos(run_dir: Path, output_dir: Path, run_key: str, max_videos: int) -> list[dict[str, object]]:
    media_dir = run_dir / "media"
    if not media_dir.exists():
        return []
    videos = sorted(media_dir.glob("*.mp4"))[:max_videos]
    copied = []
    for src in videos:
        rel = Path("media") / run_key / src.name
        dst = output_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
        copied.append({"path": rel.as_posix(), "name": src.name})
    return copied


def load_run(args: argparse.Namespace, spec: dict[str, str]) -> dict[str, object]:
    run_dir = args.outputs_root / spec["dir"]
    logs_path = run_dir / "logs.json.txt"
    if not logs_path.exists():
        raise FileNotFoundError(f"missing DP log file: {logs_path}")
    logs = read_json_lines(logs_path)
    checkpoints = sorted((run_dir / "checkpoints").glob("*.ckpt"))
    best = sorted(
        [ckpt for ckpt in checkpoints if checkpoint_score(ckpt) is not None],
        key=lambda ckpt: checkpoint_score(ckpt) or -1,
        reverse=True,
    )
    return {
        **spec,
        "run_dir": str(run_dir),
        "num_log_rows": len(logs),
        "final_epoch": int(logs[-1].get("epoch", -1)) if logs else None,
        "final_global_step": int(logs[-1].get("global_step", -1)) if logs else None,
        "final_train_loss": final_with_key(logs, "train_loss"),
        "final_val_loss": final_with_key(logs, "val_loss"),
        "final_train_mean_score": final_with_key(logs, "train/mean_score"),
        "final_test_mean_score": final_with_key(logs, "test/mean_score"),
        "best_checkpoint": str(best[0]) if best else None,
        "latest_checkpoint": str(run_dir / "checkpoints" / "latest.ckpt"),
        "checkpoints": [{"path": str(ckpt), "score": checkpoint_score(ckpt)} for ckpt in checkpoints],
        "test_rewards": final_test_rewards(logs),
        "traces": {
            "train_loss": trace(logs, "train_loss"),
            "val_loss": trace(logs, "val_loss"),
            "train_mean_score": trace(logs, "train/mean_score"),
            "test_mean_score": trace(logs, "test/mean_score"),
        },
        "videos": copy_videos(run_dir, args.output_dir, spec["key"], args.max_videos),
    }


def build_html(runs: list[dict[str, object]], title: str, lede: str) -> str:
    payload = html.escape(json.dumps({"runs": runs}, separators=(",", ":")), quote=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{ --bg:#e7ece5; --panel:#fbfbf2; --ink:#18201b; --muted:#657064; --border:#ccd5c8; --shadow:rgba(24,32,27,.1); }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; color:var(--ink); background:radial-gradient(circle at 10% 0%,rgba(15,109,103,.16),transparent 28%),radial-gradient(circle at 88% 8%,rgba(181,74,42,.13),transparent 25%),linear-gradient(180deg,#f8f8ed,var(--bg)); font-family:"Avenir Next","Trebuchet MS",sans-serif; }}
    header {{ position:sticky; top:0; z-index:5; padding:22px 24px 18px; border-bottom:1px solid var(--border); background:rgba(251,251,242,.94); backdrop-filter:blur(10px); }}
    h1 {{ margin:0 0 8px; font-size:clamp(26px,3vw,40px); letter-spacing:-.04em; }}
    .lede {{ margin:0; max-width:1050px; color:var(--muted); line-height:1.45; }}
    main {{ padding:22px 24px 38px; display:grid; gap:22px; }}
    .summary,.charts,.runs {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:14px; }}
    .card {{ background:rgba(251,251,242,.96); border:1px solid var(--border); border-radius:20px; box-shadow:0 14px 34px var(--shadow); overflow:hidden; }}
    .card-body {{ padding:15px; display:grid; gap:12px; }}
    .run-card {{ border-top:6px solid var(--accent); }}
    h2,h3 {{ margin:0; }}
    .metric-grid {{ display:grid; grid-template-columns:repeat(2,1fr); gap:8px; }}
    .metric {{ border:1px solid var(--border); border-radius:13px; padding:9px; background:#fffef8; }}
    .metric span,.path {{ display:block; color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.055em; }}
    .metric strong {{ font-size:22px; }}
    .path {{ text-transform:none; letter-spacing:0; word-break:break-all; line-height:1.35; }}
    svg {{ width:100%; height:180px; display:block; overflow:visible; }}
    .gridline {{ stroke:rgba(24,32,27,.12); stroke-width:1; stroke-dasharray:4 4; }}
    .line {{ fill:none; stroke-width:2.6; stroke-linecap:round; stroke-linejoin:round; }}
    .axis-label {{ fill:var(--muted); font-size:10px; }}
    .strip {{ display:grid; grid-template-columns:repeat(25,1fr); gap:3px; }}
    .tick {{ height:18px; border-radius:4px; background:#d6ddd1; border:1px solid rgba(24,32,27,.12); }}
    .tick.success {{ background:#0f6d67; }}
    .tick.fail {{ background:#ead6ca; }}
    .videos {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(220px,1fr)); gap:10px; }}
    video {{ display:block; width:100%; aspect-ratio:4/3; object-fit:contain; background:#050403; border-radius:14px; }}
    .video-name {{ color:var(--muted); font-size:11px; margin-top:4px; word-break:break-all; }}
    @media (max-width:720px) {{ header,main {{ padding-left:14px; padding-right:14px; }} .metric-grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <p class="lede">{html.escape(lede)}</p>
  </header>
  <main>
    <section class="summary" id="summary"></section>
    <section class="charts" id="charts"></section>
    <section class="runs" id="runs"></section>
  </main>
  <script id="payload" type="application/json">{payload}</script>
  <script>
    const DATA = JSON.parse(document.getElementById('payload').textContent);
    const fmt = v => Number.isFinite(v) ? v.toFixed(3) : 'n/a';
    function path(points) {{
      if (!points || points.length === 0) return '';
      const left=36,right=356,top=18,bottom=148;
      const xs = points.map(p => p.x), ys = points.map(p => p.y);
      const xMin=Math.min(...xs), xMax=Math.max(...xs), yMin=Math.min(...ys), yMax=Math.max(...ys);
      return points.map((p,i) => {{
        const x = left + ((p.x - xMin) / Math.max(1e-6, xMax - xMin)) * (right - left);
        const y = bottom - ((p.y - yMin) / Math.max(1e-6, yMax - yMin)) * (bottom - top);
        return `${{i ? 'L' : 'M'}} ${{x.toFixed(2)}} ${{y.toFixed(2)}}`;
      }}).join(' ');
    }}
    function chart(title, traces, valueKey) {{
      const all = traces.flatMap(t => t.points || []);
      const ys = all.map(p => p.y);
      const yMin = ys.length ? Math.min(...ys) : NaN, yMax = ys.length ? Math.max(...ys) : NaN;
      return `<article class="card"><div class="card-body"><h3>${{title}}</h3>
        <svg viewBox="0 0 390 170" preserveAspectRatio="none">
          <line class="gridline" x1="36" y1="18" x2="356" y2="18"></line><line class="gridline" x1="36" y1="83" x2="356" y2="83"></line><line class="gridline" x1="36" y1="148" x2="356" y2="148"></line>
          <text class="axis-label" x="4" y="22">${{fmt(yMax)}}</text><text class="axis-label" x="4" y="152">${{fmt(yMin)}}</text>
          ${{traces.map(t => `<path class="line" stroke="${{t.color}}" d="${{path(t.points)}}"></path>`).join('')}}
        </svg>
        <div class="path">${{traces.map(t => `${{t.label}}: ${{fmt(t[valueKey])}}`).join(' | ')}}</div>
      </div></article>`;
    }}
    function render() {{
      document.getElementById('summary').innerHTML = DATA.runs.map(run => `<article class="card run-card" style="--accent:${{run.color}}"><div class="card-body">
        <h2>${{run.label}}</h2>
        <div class="metric-grid">
          <div class="metric"><span>test mean score</span><strong>${{fmt(run.final_test_mean_score)}}</strong></div>
          <div class="metric"><span>train mean score</span><strong>${{fmt(run.final_train_mean_score)}}</strong></div>
          <div class="metric"><span>val loss</span><strong>${{fmt(run.final_val_loss)}}</strong></div>
          <div class="metric"><span>epoch</span><strong>${{run.final_epoch}}</strong></div>
        </div>
        <div class="path">best checkpoint: ${{run.best_checkpoint || 'n/a'}}</div>
        <div class="strip">${{run.test_rewards.map(r => `<span class="tick ${{r.reward > 0 ? 'success' : 'fail'}}" title="seed ${{r.seed}} reward ${{r.reward}}"></span>`).join('')}}</div>
      </div></article>`).join('');
      const tracesFor = key => DATA.runs.map(run => ({{label: run.label, color: run.color, points: run.traces[key], final: run[`final_${{key}}`]}}));
      document.getElementById('charts').innerHTML = [
        chart('Train loss', DATA.runs.map(r => ({{label:r.label,color:r.color,points:r.traces.train_loss, final_train_loss:r.final_train_loss}})), 'final_train_loss'),
        chart('Validation loss', DATA.runs.map(r => ({{label:r.label,color:r.color,points:r.traces.val_loss, final_val_loss:r.final_val_loss}})), 'final_val_loss'),
        chart('Train mean score', DATA.runs.map(r => ({{label:r.label,color:r.color,points:r.traces.train_mean_score, final_train_mean_score:r.final_train_mean_score}})), 'final_train_mean_score'),
        chart('Test mean score', DATA.runs.map(r => ({{label:r.label,color:r.color,points:r.traces.test_mean_score, final_test_mean_score:r.final_test_mean_score}})), 'final_test_mean_score')
      ].join('');
      document.getElementById('runs').innerHTML = DATA.runs.map(run => `<article class="card run-card" style="--accent:${{run.color}}"><div class="card-body">
        <h2>${{run.label}} rollout videos</h2>
        <p class="path">${{run.run_dir}}</p>
        <div class="videos">${{run.videos.map(v => `<div><video src="${{v.path}}" controls preload="metadata"></video><div class="video-name">${{v.name}}</div></div>`).join('')}}</div>
      </div></article>`).join('');
    }}
    render();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    run_set = RUN_SETS[args.run_set]
    if args.output_dir is None:
        args.output_dir = Path(str(run_set["output_dir"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = [load_run(args, spec) for spec in run_set["runs"]]
    json_path = args.output_dir / str(run_set["json_name"])
    (args.output_dir / "index.html").write_text(build_html(runs, str(run_set["title"]), str(run_set["lede"])))
    json_path.write_text(json.dumps({"runs": runs}, indent=2) + "\n")
    print(f"wrote {args.output_dir / 'index.html'}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
