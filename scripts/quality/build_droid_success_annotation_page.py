#!/usr/bin/env python3
"""Build a static annotation page for DROID success trajectories."""

from __future__ import annotations

import argparse
import html
import json
import os
import shutil
from pathlib import Path

import h5py


DEFAULT_STAGES = [
    "approach_pen",
    "grasp_pen",
    "lift_pen",
    "move_to_cup",
    "insert_or_release",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/Users/jasonyan/Desktop/droid-main/data/success"),
        help="Root containing date/session/trajectory.h5 directories.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/Users/jasonyan/Desktop/demonstration-information/droid_success_annotation_share"),
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of trajectories.")
    parser.add_argument("--copy-videos", action="store_true", help="Copy MP4s instead of symlinking them.")
    return parser.parse_args()


def h5_attr(value) -> str:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def trajectory_frames(path: Path) -> tuple[int, dict[str, str]]:
    with h5py.File(path, "r") as f:
        attrs = {key: h5_attr(value) for key, value in f.attrs.items()}
        if "action/cartesian_position" in f:
            frames = int(f["action/cartesian_position"].shape[0])
        elif "observation/robot_state/cartesian_position" in f:
            frames = int(f["observation/robot_state/cartesian_position"].shape[0])
        else:
            frames = 0
    return frames, attrs


def mp4_paths(session_dir: Path) -> list[Path]:
    preferred = sorted((session_dir / "recordings" / "MP4").glob("*.mp4"))
    if preferred:
        return preferred
    return sorted((session_dir / "recordings" / "H264").glob("*.mp4"))


def safe_id(path: Path, root: Path) -> str:
    rel = path.relative_to(root).parent
    return "__".join(part.replace(":", "_") for part in rel.parts)


def materialize_video(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def collect_rows(input_root: Path, output_root: Path, limit: int | None, copy_videos: bool) -> list[dict[str, object]]:
    rows = []
    videos_root = output_root / "videos"
    for traj_path in sorted(input_root.glob("*/*/trajectory.h5")):
        if limit is not None and len(rows) >= limit:
            break
        session_dir = traj_path.parent
        videos = mp4_paths(session_dir)
        if not videos:
            continue
        try:
            num_frames, attrs = trajectory_frames(traj_path)
        except OSError as exc:
            print(f"WARNING: skipping unreadable HDF5 {traj_path}: {exc}")
            continue
        row_id = safe_id(traj_path, input_root)
        video_rows = []
        for src in videos:
            dst = videos_root / row_id / src.name
            materialize_video(src, dst, copy=copy_videos)
            video_rows.append(
                {
                    "camera": src.stem,
                    "src": str(dst.relative_to(output_root)),
                    "source_path": str(src),
                }
            )
        rows.append(
            {
                "id": row_id,
                "date": traj_path.parents[1].name,
                "session": session_dir.name,
                "task": attrs.get("current_task", ""),
                "time": attrs.get("time", session_dir.name),
                "success": attrs.get("success", ""),
                "num_frames": num_frames,
                "trajectory_path": str(traj_path),
                "videos": video_rows,
            }
        )
    return rows


def annotations_payload(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "stages": DEFAULT_STAGES,
        "trajectories": {
            str(row["id"]): {
                "task": row["task"],
                "date": row["date"],
                "session": row["session"],
                "num_frames": row["num_frames"],
                "stages": [{"name": name, "start": None, "end": None} for name in DEFAULT_STAGES],
            }
            for row in rows
        },
    }


def card_html(row: dict[str, object], index: int) -> str:
    videos = row["videos"]
    videos_html = "\n".join(
        f"""
        <div class="video-pane">
          <video controls preload="metadata" src="{html.escape(video['src'])}" data-camera="{html.escape(video['camera'])}"></video>
          <div class="video-label">{html.escape(video['camera'])}</div>
        </div>
        """
        for video in videos
    )
    return f"""
    <article class="card" data-key="{html.escape(str(row['id']))}" data-index="{index}" data-date="{html.escape(str(row['date']))}">
      <div class="card-head">
        <div>
          <strong>{index:03d}</strong>
          <span>{html.escape(str(row['task']))}</span>
        </div>
        <button type="button" class="sync-button">sync play</button>
      </div>
      <div class="video-grid">{videos_html}</div>
      <div class="meta">
        <span>{html.escape(str(row['date']))}</span>
        <span>{html.escape(str(row['session']))}</span>
        <span>{html.escape(str(row['num_frames']))} frames</span>
      </div>
      <div class="annotation-panel">
        <div class="annotation-toolbar">
          <label>active stage <select class="stage-select"></select></label>
          <button type="button" class="mark-start">mark start</button>
          <button type="button" class="mark-end">mark end</button>
          <span class="frame-readout">frame 0</span>
        </div>
        <div class="stage-rows"></div>
      </div>
      <details>
        <summary>source</summary>
        <code>{html.escape(str(row['trajectory_path']))}</code>
      </details>
    </article>
    """


def build_html(rows: list[dict[str, object]], payload: dict[str, object]) -> str:
    payload_json = json.dumps(payload).replace("</", "<\\/")
    cards = "\n".join(card_html(row, idx + 1) for idx, row in enumerate(rows))
    dates = sorted({str(row["date"]) for row in rows})
    date_options = '<option value="all">all dates</option>' + "".join(
        f'<option value="{html.escape(date)}">{html.escape(date)}</option>' for date in dates
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DROID success trajectory annotations</title>
  <style>
    :root {{
      --bg: #f6f7f4;
      --panel: #ffffff;
      --ink: #151914;
      --muted: #637066;
      --border: #d9dfd8;
      --accent: #256f61;
      --accent-soft: #e3f0eb;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    header {{ position: sticky; top: 0; z-index: 3; padding: 18px 22px; background: rgba(255,255,255,0.94); border-bottom: 1px solid var(--border); backdrop-filter: blur(10px); }}
    h1 {{ margin: 0 0 8px; font-size: 22px; letter-spacing: 0; }}
    .topbar {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }}
    button, select, input {{ font: inherit; }}
    button, select {{ border: 1px solid var(--border); background: white; border-radius: 7px; padding: 7px 10px; }}
    button {{ cursor: pointer; }}
    .status {{ color: var(--muted); font-size: 13px; }}
    main {{ padding: 18px; display: grid; grid-template-columns: repeat(auto-fill, minmax(390px, 1fr)); gap: 14px; }}
    .card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; box-shadow: 0 8px 24px rgba(20, 30, 22, 0.05); }}
    .card[hidden] {{ display: none; }}
    .card-head {{ display: flex; justify-content: space-between; gap: 12px; align-items: center; padding: 10px 12px; border-bottom: 1px solid var(--border); }}
    .card-head div {{ display: grid; gap: 2px; }}
    .card-head span {{ color: var(--muted); font-size: 13px; }}
    .sync-button {{ padding: 5px 8px; font-size: 12px; }}
    .video-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1px; background: var(--border); }}
    .video-pane {{ background: #111; position: relative; }}
    video {{ display: block; width: 100%; aspect-ratio: 4 / 3; background: #111; }}
    .video-label {{ position: absolute; left: 8px; bottom: 8px; padding: 3px 7px; background: rgba(0,0,0,0.62); color: white; border-radius: 6px; font-size: 12px; }}
    .meta {{ display: flex; flex-wrap: wrap; gap: 8px; padding: 9px 12px; color: var(--muted); font-size: 12px; border-bottom: 1px solid var(--border); }}
    .annotation-panel {{ padding: 10px 12px 12px; display: grid; gap: 9px; }}
    .annotation-toolbar {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
    .annotation-toolbar label {{ display: flex; gap: 7px; align-items: center; color: var(--muted); font-size: 13px; }}
    .frame-readout {{ color: var(--accent); font-weight: 700; font-size: 13px; }}
    .stage-rows {{ display: grid; gap: 7px; }}
    .stage-row {{ display: grid; grid-template-columns: minmax(110px, 1fr) 64px 64px auto auto; gap: 7px; align-items: center; font-size: 13px; }}
    .stage-row input {{ width: 100%; border: 1px solid var(--border); border-radius: 7px; padding: 6px 7px; }}
    .stage-row button {{ padding: 5px 7px; font-size: 12px; }}
    details {{ border-top: 1px solid var(--border); padding: 8px 12px 12px; color: var(--muted); font-size: 12px; }}
    code {{ word-break: break-all; }}
  </style>
</head>
<body>
  <header>
    <h1>DROID success trajectory annotations</h1>
    <div class="topbar">
      <select id="date-filter">{date_options}</select>
      <input id="search" type="search" placeholder="search task/session" size="28">
      <button type="button" id="export-json">export annotations json</button>
      <button type="button" id="import-trigger">import annotations json</button>
      <input id="import-json" type="file" accept="application/json" hidden>
      <span class="status" id="status">{len(rows)} trajectories · local autosave on</span>
    </div>
  </header>
  <main>{cards}</main>
  <script>
    const DEFAULT_ANNOTATIONS = {payload_json};
    const STORAGE_KEY = "droid_success_stage_annotations_v1";
    const clone = (obj) => JSON.parse(JSON.stringify(obj));
    let annotations = loadAnnotations();

    function loadAnnotations() {{
      const saved = localStorage.getItem(STORAGE_KEY);
      if (!saved) return clone(DEFAULT_ANNOTATIONS);
      try {{
        const parsed = JSON.parse(saved);
        return mergeAnnotations(parsed);
      }} catch {{
        return clone(DEFAULT_ANNOTATIONS);
      }}
    }}
    function mergeAnnotations(raw) {{
      const merged = clone(DEFAULT_ANNOTATIONS);
      if (!raw || !raw.trajectories) return merged;
      for (const [key, value] of Object.entries(raw.trajectories)) {{
        if (!merged.trajectories[key]) continue;
        const byName = new Map((value.stages || []).map((stage) => [stage.name, stage]));
        merged.trajectories[key].stages = merged.stages.map((name) => {{
          const old = byName.get(name) || {{}};
          return {{ name, start: Number.isFinite(old.start) ? old.start : null, end: Number.isFinite(old.end) ? old.end : null }};
        }});
      }}
      return merged;
    }}
    function saveAnnotations() {{
      localStorage.setItem(STORAGE_KEY, JSON.stringify(annotations));
      document.getElementById("status").textContent = "edited locally, export json to save";
    }}
    function maxFrame(card) {{
      const key = card.dataset.key;
      return Math.max(0, Number(annotations.trajectories[key].num_frames || 1) - 1);
    }}
    function activeVideo(card) {{
      return card._activeVideo || card.querySelector("video");
    }}
    function currentFrame(card) {{
      const video = activeVideo(card);
      const frac = video && video.duration ? Math.max(0, Math.min(1, video.currentTime / video.duration)) : 0;
      return Math.round(frac * maxFrame(card));
    }}
    function updateReadout(card) {{
      card.querySelector(".frame-readout").textContent = "frame " + currentFrame(card);
    }}
    function renderRows(card) {{
      const key = card.dataset.key;
      const rows = card.querySelector(".stage-rows");
      rows.innerHTML = "";
      annotations.trajectories[key].stages.forEach((stage) => {{
        const row = document.createElement("div");
        row.className = "stage-row";
        row.innerHTML = `
          <strong>${{stage.name}}</strong>
          <input type="number" min="0" placeholder="start" value="${{stage.start ?? ""}}">
          <input type="number" min="0" placeholder="end" value="${{stage.end ?? ""}}">
          <button type="button">set start</button>
          <button type="button">set end</button>
        `;
        const [startInput, endInput, startButton, endButton] = row.querySelectorAll("input, button");
        startInput.addEventListener("change", () => {{
          stage.start = startInput.value === "" ? null : Number(startInput.value);
          saveAnnotations();
        }});
        endInput.addEventListener("change", () => {{
          stage.end = endInput.value === "" ? null : Number(endInput.value);
          saveAnnotations();
        }});
        startButton.addEventListener("click", () => {{
          stage.start = currentFrame(card);
          startInput.value = stage.start;
          saveAnnotations();
        }});
        endButton.addEventListener("click", () => {{
          stage.end = currentFrame(card);
          endInput.value = stage.end;
          saveAnnotations();
        }});
        rows.appendChild(row);
      }});
    }}
    function bindCard(card) {{
      const key = card.dataset.key;
      const select = card.querySelector(".stage-select");
      select.innerHTML = annotations.stages.map((stage, idx) => `<option value="${{idx}}">${{stage}}</option>`).join("");
      card.querySelectorAll("video").forEach((video) => {{
        video.addEventListener("play", () => {{ card._activeVideo = video; }});
        video.addEventListener("seeking", () => {{ card._activeVideo = video; }});
        video.addEventListener("timeupdate", () => updateReadout(card));
        video.addEventListener("loadedmetadata", () => updateReadout(card));
      }});
      card.querySelector(".mark-start").addEventListener("click", () => {{
        const stage = annotations.trajectories[key].stages[Number(select.value)];
        stage.start = currentFrame(card);
        saveAnnotations();
        renderRows(card);
      }});
      card.querySelector(".mark-end").addEventListener("click", () => {{
        const stage = annotations.trajectories[key].stages[Number(select.value)];
        stage.end = currentFrame(card);
        saveAnnotations();
        renderRows(card);
      }});
      card.querySelector(".sync-button").addEventListener("click", () => {{
        const videos = Array.from(card.querySelectorAll("video"));
        const base = activeVideo(card);
        videos.forEach((video) => {{
          video.currentTime = base.currentTime;
          video.play();
        }});
      }});
      renderRows(card);
      updateReadout(card);
    }}
    document.querySelectorAll(".card").forEach(bindCard);

    function applyFilters() {{
      const date = document.getElementById("date-filter").value;
      const query = document.getElementById("search").value.toLowerCase();
      let shown = 0;
      document.querySelectorAll(".card").forEach((card) => {{
        const text = card.textContent.toLowerCase();
        const okDate = date === "all" || card.dataset.date === date;
        const okQuery = !query || text.includes(query);
        card.hidden = !(okDate && okQuery);
        if (!card.hidden) shown += 1;
      }});
      document.getElementById("status").textContent = shown + " visible · local autosave on";
    }}
    document.getElementById("date-filter").addEventListener("change", applyFilters);
    document.getElementById("search").addEventListener("input", applyFilters);

    document.getElementById("export-json").addEventListener("click", () => {{
      const blob = new Blob([JSON.stringify(annotations, null, 2)], {{ type: "application/json" }});
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "droid_success_stage_annotations.json";
      link.click();
      URL.revokeObjectURL(url);
    }});
    document.getElementById("import-trigger").addEventListener("click", () => document.getElementById("import-json").click());
    document.getElementById("import-json").addEventListener("change", async (event) => {{
      const file = event.target.files && event.target.files[0];
      if (!file) return;
      annotations = mergeAnnotations(JSON.parse(await file.text()));
      localStorage.setItem(STORAGE_KEY, JSON.stringify(annotations));
      document.querySelectorAll(".card").forEach(renderRows);
      document.getElementById("status").textContent = "imported annotations";
    }});
  </script>
</body>
</html>
"""


def write_manifest(output_root: Path, rows: list[dict[str, object]]) -> None:
    manifest = output_root / "manifest.json"
    manifest.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(args.input_root, args.output_root, args.limit, args.copy_videos)
    payload = annotations_payload(rows)
    (args.output_root / "index.html").write_text(build_html(rows, payload), encoding="utf-8")
    (args.output_root / "default_annotations.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_manifest(args.output_root, rows)
    print(f"wrote {args.output_root / 'index.html'} ({len(rows)} trajectories)")
    print(f"wrote {args.output_root / 'default_annotations.json'}")
    print(f"wrote {args.output_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
