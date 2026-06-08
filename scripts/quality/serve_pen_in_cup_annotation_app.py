#!/usr/bin/env python3
"""Serve a local annotation app for pen-in-cup DROID videos.

The app scans either a local copy of raw DROID episode folders or a flat
directory of annotation proxy MP4 files, serves the videos, and writes one CSV
with two labels per episode:

- observability: full / partial
- optimality: better / okay / worse

Example:

python scripts/quality/serve_pen_in_cup_annotation_app.py --port 8766
"""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import re
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO_ROOT = REPO_ROOT / "pen_in_cup_annotation_app" / "videos" / "06-07-2026-total-102"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "pen_in_cup_annotations.csv"

OBS_LABELS = {"unlabeled", "full", "partial"}
OPT_LABELS = {"unlabeled", "better", "okay", "worse"}
PREFERRED_CAMERA_ORDER = {
    "17471093": "wrist",
    "23404442": "exterior",
}


@dataclass(frozen=True)
class VideoRef:
    label: str
    path: Path


@dataclass(frozen=True)
class EpisodeRow:
    dataset: str
    ep_idx: int
    episode: str
    title: str
    videos: tuple[VideoRef, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def camera_label(path: Path) -> str:
    return PREFERRED_CAMERA_ORDER.get(path.stem, path.stem)


def sort_video(path: Path) -> tuple[int, str]:
    order = {"wrist": 0, "exterior": 1}
    label = camera_label(path)
    return order.get(label, 10), path.name


def episode_dirs(video_root: Path) -> list[Path]:
    return sorted(
        path
        for path in video_root.iterdir()
        if path.is_dir() and ((path / "recordings" / "MP4").is_dir() or (path / "recordings" / "H264").is_dir())
    )


def proxy_video_rows(video_root: Path) -> list[EpisodeRow]:
    rows: list[EpisodeRow] = []
    for fallback_idx, path in enumerate(sorted(video_root.glob("*.mp4"))):
        match = re.match(r"0*(\d+)[_-](.+)\.mp4$", path.name)
        if match:
            ep_idx = int(match.group(1))
            episode = match.group(2)
        else:
            ep_idx = fallback_idx
            episode = path.stem
        title = f"Pen-in-cup {ep_idx:03d} · {episode}"
        rows.append(EpisodeRow("pen_in_cup", ep_idx, episode, title, (VideoRef("proxy", path),)))
    return sorted(rows, key=lambda row: row.ep_idx)


def episode_videos(episode_dir: Path) -> tuple[VideoRef, ...]:
    mp4_dir = episode_dir / "recordings" / "MP4"
    h264_dir = episode_dir / "recordings" / "H264"
    paths = sorted(mp4_dir.glob("*.mp4"), key=sort_video) if mp4_dir.is_dir() else []
    if not paths and h264_dir.is_dir():
        paths = sorted(h264_dir.glob("*.mp4"), key=sort_video)
    return tuple(VideoRef(camera_label(path), path) for path in paths if "stereo" not in path.stem.lower())


def build_rows(video_root: Path) -> list[EpisodeRow]:
    proxy_rows = proxy_video_rows(video_root)
    if proxy_rows:
        return proxy_rows

    rows: list[EpisodeRow] = []
    for ep_idx, episode_dir in enumerate(episode_dirs(video_root)):
        videos = episode_videos(episode_dir)
        title = f"Pen-in-cup {ep_idx:03d} · {episode_dir.name}"
        rows.append(EpisodeRow("pen_in_cup", ep_idx, episode_dir.name, title, videos))
    return rows


def annotation_key(dataset: str, ep_idx: int) -> str:
    return f"{dataset}:{ep_idx}"


def read_annotations(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    out: dict[tuple[str, int], dict[str, str]] = {}
    if not path.exists():
        return out
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            dataset = row.get("dataset", "pen_in_cup")
            ep_idx = int(row["ep_idx"])
            observability = (row.get("observability") or "unlabeled").strip() or "unlabeled"
            optimality = (row.get("optimality") or "unlabeled").strip() or "unlabeled"
            out[(dataset, ep_idx)] = {
                "observability": observability if observability in OBS_LABELS else "unlabeled",
                "optimality": optimality if optimality in OPT_LABELS else "unlabeled",
                "note": row.get("note") or "",
                "updated_at": row.get("updated_at") or "",
            }
    return out


def write_annotations(path: Path, rows: list[EpisodeRow], annotations: dict[tuple[str, int], dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row_by_key = {(row.dataset, row.ep_idx): row for row in rows}
    with path.open("w", newline="") as f:
        fieldnames = [
            "dataset",
            "ep_idx",
            "episode",
            "observability",
            "optimality",
            "note",
            "updated_at",
            "videos",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key, ann in sorted(annotations.items(), key=lambda item: item[0]):
            row = row_by_key.get(key)
            if row is None:
                continue
            observability = ann.get("observability", "unlabeled")
            optimality = ann.get("optimality", "unlabeled")
            note = ann.get("note", "")
            if observability == "unlabeled" and optimality == "unlabeled" and not note:
                continue
            writer.writerow(
                {
                    "dataset": row.dataset,
                    "ep_idx": row.ep_idx,
                    "episode": row.episode,
                    "observability": observability,
                    "optimality": optimality,
                    "note": note,
                    "updated_at": ann.get("updated_at", ""),
                    "videos": json.dumps([str(video.path) for video in row.videos]),
                }
            )


def row_json(row: EpisodeRow) -> dict[str, object]:
    return {
        "dataset": row.dataset,
        "ep_idx": row.ep_idx,
        "episode": row.episode,
        "title": row.title,
        "videos": [
            {
                "label": video.label,
                "path": str(video.path),
                "url": f"/media?path={quote(str(video.path))}",
                "ready": video.path.exists(),
            }
            for video in row.videos
        ],
    }


def make_handler(
    rows: list[EpisodeRow],
    annotations: dict[tuple[str, int], dict[str, str]],
    output_csv: Path,
    media_root: Path,
):
    row_by_key = {(row.dataset, row.ep_idx): row for row in rows}

    class Handler(BaseHTTPRequestHandler):
        server_version = "PenInCupAnnotation/1.0"

        def log_message(self, fmt: str, *args: object) -> None:
            print(f"{self.address_string()} - {fmt % args}")

        def send_json(self, payload: object, status: int = 200) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self.serve_index()
            elif parsed.path == "/api/catalog":
                self.serve_catalog()
            elif parsed.path == "/media":
                self.serve_media(parsed.query)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/api/annotation":
                self.send_error(HTTPStatus.NOT_FOUND)
                return

            length = int(self.headers.get("Content-Length", "0"))
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                dataset = str(body["dataset"])
                ep_idx = int(body["ep_idx"])
                observability = str(body.get("observability") or "unlabeled")
                optimality = str(body.get("optimality") or "unlabeled")
                note = str(body.get("note") or "")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                self.send_json({"error": f"bad annotation payload: {exc}"}, status=400)
                return

            key = (dataset, ep_idx)
            if key not in row_by_key:
                self.send_json({"error": "unknown episode"}, status=404)
                return
            if observability not in OBS_LABELS:
                self.send_json({"error": f"invalid observability: {observability}"}, status=400)
                return
            if optimality not in OPT_LABELS:
                self.send_json({"error": f"invalid optimality: {optimality}"}, status=400)
                return

            annotations[key] = {
                "observability": observability,
                "optimality": optimality,
                "note": note,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
            write_annotations(output_csv, rows, annotations)
            self.send_json({"ok": True, "annotation": annotations[key], "output_csv": str(output_csv)})

        def serve_index(self) -> None:
            data = INDEX_HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def serve_catalog(self) -> None:
            by_key = {
                annotation_key(dataset, ep_idx): value
                for (dataset, ep_idx), value in annotations.items()
            }
            self.send_json(
                {
                    "rows": [row_json(row) for row in rows],
                    "annotations": by_key,
                    "output_csv": str(output_csv),
                    "video_root": str(media_root),
                }
            )

        def serve_media(self, query: str) -> None:
            values = parse_qs(query).get("path", [])
            if not values:
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            path = Path(values[0]).resolve()
            allowed = media_root.resolve()
            if not (path == allowed or allowed in path.parents):
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            if not path.exists() or not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return

            file_size = path.stat().st_size
            content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            start = 0
            end = file_size - 1
            status = HTTPStatus.OK
            range_header = self.headers.get("Range")
            if range_header:
                match = re.match(r"bytes=(\d*)-(\d*)", range_header)
                if match:
                    if match.group(1):
                        start = int(match.group(1))
                    if match.group(2):
                        end = int(match.group(2))
                    end = min(end, file_size - 1)
                    status = HTTPStatus.PARTIAL_CONTENT

            if start > end or start >= file_size:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return

            length = end - start + 1
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.end_headers()

            with path.open("rb") as f:
                f.seek(start)
                remaining = length
                while remaining:
                    chunk = f.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

    return Handler


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Pen-in-Cup Annotation</title>
  <style>
    :root {
      --bg: #f6f7f4;
      --panel: #ffffff;
      --ink: #1b1f21;
      --muted: #66706d;
      --border: #d8ded8;
      --accent: #176d60;
      --accent-soft: #e4f1ee;
      --warn: #9b6425;
      --bad: #9d3030;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    button, input, select, textarea { font: inherit; }
    .app { min-height: 100vh; display: grid; grid-template-columns: 300px minmax(0, 1fr); }
    aside {
      border-right: 1px solid var(--border);
      background: #fbfcf9;
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      position: sticky;
      top: 0;
      height: 100vh;
      overflow: auto;
    }
    h1 { margin: 0; font-size: 20px; line-height: 1.15; letter-spacing: 0; }
    .small { color: var(--muted); font-size: 12px; line-height: 1.45; }
    .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 12px; }
    label.field { display: grid; gap: 6px; color: var(--muted); font-size: 12px; }
    input, select, textarea {
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 9px 10px;
      background: #fff;
      color: var(--ink);
    }
    main { min-width: 0; display: grid; grid-template-rows: auto minmax(0, 1fr); }
    header {
      background: rgba(246, 247, 244, 0.94);
      backdrop-filter: blur(8px);
      border-bottom: 1px solid var(--border);
      padding: 16px 22px;
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 16px;
      align-items: center;
      position: sticky;
      top: 0;
      z-index: 5;
    }
    .title-row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
    .title-row h2 { margin: 0; font-size: 22px; line-height: 1.2; letter-spacing: 0; }
    .badge {
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 4px 9px;
      font-size: 12px;
      color: var(--muted);
      background: #fff;
    }
    .top-actions { display: flex; gap: 8px; align-items: center; }
    .top-actions button, .choice {
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #fff;
      color: var(--ink);
      cursor: pointer;
    }
    .top-actions button { min-height: 38px; padding: 8px 12px; }
    .primary { background: var(--accent) !important; color: #fff !important; border-color: var(--accent) !important; }
    .workspace { padding: 20px 22px 28px; display: grid; gap: 18px; align-content: start; }
    .video-grid {
      --video-max: 560px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, min(100%, var(--video-max))));
      gap: 14px;
      justify-content: start;
    }
    .video-grid.large { --video-max: 760px; }
    .video-card {
      background: #101313;
      border-radius: 8px;
      overflow: hidden;
      border: 1px solid #202929;
      max-width: var(--video-max);
    }
    .view-label {
      color: #dce8e4;
      font-size: 12px;
      padding: 8px 10px;
      display: flex;
      justify-content: space-between;
      gap: 12px;
    }
    video {
      display: block;
      width: 100%;
      background: #050606;
      aspect-ratio: 16 / 9;
      object-fit: contain;
    }
    .annotation-area { display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: 16px; align-items: start; }
    .label-section { display: grid; gap: 10px; margin-bottom: 14px; }
    .section-title { font-size: 13px; font-weight: 700; color: var(--muted); }
    .choice-grid { display: grid; grid-template-columns: repeat(3, minmax(100px, 1fr)); gap: 10px; }
    .choice {
      min-height: 58px;
      display: grid;
      align-content: center;
      justify-items: start;
      padding: 10px 12px;
      text-align: left;
    }
    .choice strong { font-size: 15px; }
    .choice span { color: var(--muted); font-size: 12px; margin-top: 2px; }
    .choice.active { border-color: var(--accent); background: var(--accent-soft); }
    .choice[data-value="worse"].active { border-color: var(--bad); background: #f7e8e8; }
    .choice[data-value="partial"].active, .choice[data-value="okay"].active { border-color: var(--warn); background: #f8eee6; }
    textarea { min-height: 94px; resize: vertical; }
    .progress { display: grid; gap: 8px; }
    .bar { height: 10px; background: #e5e9e4; border-radius: 999px; overflow: hidden; border: 1px solid var(--border); }
    .bar div { height: 100%; background: var(--accent); width: 0; }
    .metrics { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; }
    .metric { padding: 10px; border: 1px solid var(--border); border-radius: 8px; background: #fff; }
    .metric span { display: block; color: var(--muted); font-size: 11px; margin-bottom: 3px; }
    .metric strong { font-size: 18px; }
    .queue { max-height: 330px; overflow: auto; display: grid; gap: 6px; }
    .queue button {
      border: 1px solid var(--border);
      background: #fff;
      border-radius: 8px;
      padding: 8px 10px;
      text-align: left;
      cursor: pointer;
      display: flex;
      justify-content: space-between;
      gap: 12px;
    }
    .queue button.active { border-color: var(--accent); background: var(--accent-soft); }
    .dot {
      width: 9px;
      height: 9px;
      border-radius: 999px;
      display: inline-block;
      background: #c1c8c2;
      margin-right: 6px;
    }
    .dot.full, .dot.better { background: #176d60; }
    .dot.partial, .dot.okay { background: #9b6425; }
    .dot.worse { background: #9d3030; }
    .path { color: var(--muted); font-size: 12px; word-break: break-all; line-height: 1.45; }
    @media (max-width: 980px) {
      .app { grid-template-columns: 1fr; }
      aside { position: static; height: auto; border-right: 0; border-bottom: 1px solid var(--border); }
      header { position: static; grid-template-columns: 1fr; }
      .annotation-area { grid-template-columns: 1fr; }
      .choice-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <div>
        <h1>Pen-in-Cup Annotation</h1>
        <div class="small" id="save-target">Loading catalog...</div>
      </div>
      <section class="panel">
        <div class="metrics" id="metrics"></div>
      </section>
      <section class="panel" style="display:grid;gap:10px">
        <label class="field">Filter
          <select id="filter">
            <option value="incomplete_ready">incomplete with video</option>
            <option value="incomplete">incomplete all</option>
            <option value="all">all</option>
            <option value="obs_full">observability full</option>
            <option value="obs_partial">observability partial</option>
            <option value="opt_better">optimality better</option>
            <option value="opt_okay">optimality okay</option>
            <option value="opt_worse">optimality worse</option>
          </select>
        </label>
        <label class="field">Search episode id/name
          <input id="search" placeholder="e.g. 42 or Sat_Jun">
        </label>
        <label class="field">Auto advance
          <select id="auto-advance">
            <option value="on">on</option>
            <option value="off">off</option>
          </select>
        </label>
        <label class="field">Video size
          <select id="video-size">
            <option value="medium">medium</option>
            <option value="large">large</option>
          </select>
        </label>
        <label class="field">Playback speed
          <select id="playback-speed">
            <option value="1">1x</option>
            <option value="1.5">1.5x</option>
            <option value="2">2x</option>
            <option value="3" selected>3x</option>
          </select>
        </label>
      </section>
      <section class="panel">
        <div class="progress">
          <div class="small" id="progress-text"></div>
          <div class="bar"><div id="progress-bar"></div></div>
        </div>
      </section>
      <section class="panel">
        <div class="small">Shortcuts: F full, P partial, 1 better, 2 okay, 3 worse, S save, N next, B back, Space play/pause.</div>
      </section>
    </aside>
    <main>
      <header>
        <div>
          <div class="title-row">
            <h2 id="title">Loading...</h2>
            <span class="badge" id="obs-badge"></span>
            <span class="badge" id="opt-badge"></span>
          </div>
          <div class="small" id="status">Starting local app.</div>
        </div>
        <div class="top-actions">
          <button id="prev">Back</button>
          <button class="primary" id="save">Save</button>
          <button class="primary" id="next">Next</button>
        </div>
      </header>
      <section class="workspace">
        <div class="video-grid" id="video-grid"></div>
        <div class="annotation-area">
          <section class="panel">
            <div class="label-section">
              <div class="section-title">Observability</div>
              <div class="choice-grid">
                <button class="choice" data-kind="observability" data-value="full"><strong>Full</strong><span>F</span></button>
                <button class="choice" data-kind="observability" data-value="partial"><strong>Partial</strong><span>P</span></button>
                <button class="choice" data-kind="observability" data-value="unlabeled"><strong>Clear</strong><span></span></button>
              </div>
            </div>
            <div class="label-section">
              <div class="section-title">Optimality</div>
              <div class="choice-grid">
                <button class="choice" data-kind="optimality" data-value="better"><strong>Better</strong><span>1</span></button>
                <button class="choice" data-kind="optimality" data-value="okay"><strong>Okay</strong><span>2</span></button>
                <button class="choice" data-kind="optimality" data-value="worse"><strong>Worse</strong><span>3</span></button>
              </div>
            </div>
            <label class="field">Note
              <textarea id="note" placeholder="optional note"></textarea>
            </label>
          </section>
          <section class="panel">
            <div class="queue" id="queue"></div>
          </section>
        </div>
        <div class="path" id="paths"></div>
      </section>
    </main>
  </div>
  <script>
    const EMPTY = { observability: "unlabeled", optimality: "unlabeled", note: "" };
    const SHORTCUTS = {
      f: ["observability", "full"],
      p: ["observability", "partial"],
      "1": ["optimality", "better"],
      "2": ["optimality", "okay"],
      "3": ["optimality", "worse"],
    };
    let rows = [];
    let annotations = {};
    let filtered = [];
    let activeIndex = 0;

    const keyFor = row => `${row.dataset}:${row.ep_idx}`;
    const annFor = row => annotations[keyFor(row)] || EMPTY;
    const complete = ann => ann.observability !== "unlabeled" && ann.optimality !== "unlabeled";
    const hasReadyVideo = row => row.videos.some(video => video.ready);

    async function loadCatalog() {
      const res = await fetch("/api/catalog");
      const data = await res.json();
      rows = data.rows || [];
      annotations = data.annotations || {};
      document.getElementById("save-target").textContent = `Saving to ${data.output_csv}`;
      applyFilters();
    }

    function applyFilters() {
      const filter = document.getElementById("filter").value;
      const search = document.getElementById("search").value.trim().toLowerCase();
      filtered = rows.filter(row => {
        const ann = annFor(row);
        if (filter === "incomplete_ready" && (complete(ann) || !hasReadyVideo(row))) return false;
        if (filter === "incomplete" && complete(ann)) return false;
        if (filter === "obs_full" && ann.observability !== "full") return false;
        if (filter === "obs_partial" && ann.observability !== "partial") return false;
        if (filter === "opt_better" && ann.optimality !== "better") return false;
        if (filter === "opt_okay" && ann.optimality !== "okay") return false;
        if (filter === "opt_worse" && ann.optimality !== "worse") return false;
        if (search && !(`${row.ep_idx} ${row.episode} ${row.title}`.toLowerCase().includes(search))) return false;
        return true;
      });
      activeIndex = Math.min(activeIndex, Math.max(filtered.length - 1, 0));
      render();
    }

    function renderMetrics() {
      const total = rows.length;
      const ready = rows.filter(hasReadyVideo).length;
      const done = rows.filter(row => complete(annFor(row))).length;
      const obsFull = rows.filter(row => annFor(row).observability === "full").length;
      const optBetter = rows.filter(row => annFor(row).optimality === "better").length;
      document.getElementById("metrics").innerHTML = `
        <div class="metric"><span>Episodes</span><strong>${total}</strong></div>
        <div class="metric"><span>With video</span><strong>${ready}</strong></div>
        <div class="metric"><span>Complete</span><strong>${done}</strong></div>
        <div class="metric"><span>Full / Better</span><strong>${obsFull}/${optBetter}</strong></div>
      `;
      document.getElementById("progress-text").textContent = `complete: ${done}/${total}`;
      document.getElementById("progress-bar").style.width = `${total ? 100 * done / total : 0}%`;
    }

    function renderQueue() {
      document.getElementById("queue").innerHTML = filtered.map((row, index) => {
        const ann = annFor(row);
        return `
          <button class="${index === activeIndex ? "active" : ""}" data-index="${index}">
            <span><span class="dot ${ann.observability}"></span>${row.ep_idx.toString().padStart(3, "0")}</span>
            <span>${ann.observability}/${ann.optimality}</span>
          </button>
        `;
      }).join("");
    }

    function render() {
      renderMetrics();
      renderQueue();
      const row = filtered[activeIndex];
      if (!row) {
        document.getElementById("title").textContent = "No episodes";
        document.getElementById("status").textContent = "No rows match the current filter.";
        document.getElementById("video-grid").innerHTML = "";
        document.getElementById("paths").textContent = "";
        return;
      }

      const ann = annFor(row);
      document.getElementById("title").textContent = row.title;
      document.getElementById("obs-badge").textContent = `observability: ${ann.observability}`;
      document.getElementById("opt-badge").textContent = `optimality: ${ann.optimality}`;
      document.getElementById("status").textContent = `${activeIndex + 1}/${filtered.length}`;
      document.getElementById("note").value = ann.note || "";

      document.getElementById("video-grid").className = `video-grid ${document.getElementById("video-size").value}`;
      document.getElementById("video-grid").innerHTML = row.videos.map(video => video.ready ? `
        <article class="video-card">
          <div class="view-label"><span>${video.label}</span><span>${row.ep_idx.toString().padStart(3, "0")}</span></div>
          <video controls muted loop playsinline preload="metadata" src="${video.url}"></video>
        </article>
      ` : `
        <article class="video-card">
          <div class="view-label"><span>${video.label}</span><span>${row.ep_idx.toString().padStart(3, "0")}</span></div>
          <div style="color:#dce8e4;padding:24px">Video missing: ${video.path}</div>
        </article>
      `).join("");
      document.getElementById("paths").innerHTML = row.videos.map(video => `${video.label}: ${video.path}`).join("<br>");
      document.querySelectorAll(".choice").forEach(button => {
        const kind = button.dataset.kind;
        button.classList.toggle("active", button.dataset.value === ann[kind]);
      });
      applyPlaybackRate();
      autoplayVideos();
    }

    function applyPlaybackRate() {
      const speed = Number(document.getElementById("playback-speed").value);
      document.querySelectorAll("video").forEach(video => {
        video.playbackRate = speed;
        video.onloadedmetadata = () => video.playbackRate = speed;
      });
    }

    function autoplayVideos() {
      document.querySelectorAll("video").forEach(video => {
        video.muted = true;
        video.play().catch(() => {});
      });
    }

    async function saveActive(advance = false) {
      const row = filtered[activeIndex];
      if (!row) return;
      const current = annFor(row);
      const next = {
        observability: current.observability || "unlabeled",
        optimality: current.optimality || "unlabeled",
        note: document.getElementById("note").value || "",
      };
      annotations[keyFor(row)] = next;
      const res = await fetch("/api/annotation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset: row.dataset, ep_idx: row.ep_idx, ...next }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "save failed");
      annotations[keyFor(row)] = data.annotation;
      if (advance && document.getElementById("auto-advance").value === "on" && complete(data.annotation)) {
        const filter = document.getElementById("filter").value;
        if (!filter.startsWith("incomplete")) {
          activeIndex = Math.min(activeIndex + 1, filtered.length - 1);
        }
      }
      applyFilters();
    }

    function setLabel(kind, value) {
      const row = filtered[activeIndex];
      if (!row) return;
      const current = { ...annFor(row), note: document.getElementById("note").value || "" };
      current[kind] = value;
      annotations[keyFor(row)] = current;
      render();
      if (complete(current)) saveActive(true);
    }

    function move(delta) {
      activeIndex = Math.min(Math.max(activeIndex + delta, 0), Math.max(filtered.length - 1, 0));
      render();
    }

    function togglePlay() {
      const videos = [...document.querySelectorAll("video")];
      const anyPlaying = videos.some(video => !video.paused);
      videos.forEach(video => anyPlaying ? video.pause() : video.play());
    }

    document.getElementById("filter").addEventListener("change", applyFilters);
    document.getElementById("search").addEventListener("input", applyFilters);
    document.getElementById("video-size").addEventListener("change", render);
    document.getElementById("playback-speed").addEventListener("change", applyPlaybackRate);
    document.getElementById("prev").addEventListener("click", () => move(-1));
    document.getElementById("next").addEventListener("click", () => move(1));
    document.getElementById("save").addEventListener("click", () => saveActive(false));
    document.getElementById("queue").addEventListener("click", event => {
      const button = event.target.closest("button[data-index]");
      if (!button) return;
      activeIndex = Number(button.dataset.index);
      render();
    });
    document.body.addEventListener("click", event => {
      const button = event.target.closest("button[data-kind]");
      if (button) setLabel(button.dataset.kind, button.dataset.value);
    });
    window.addEventListener("keydown", event => {
      if (event.target.matches("input, textarea, select")) return;
      const key = event.key.toLowerCase();
      if (SHORTCUTS[key]) {
        event.preventDefault();
        const [kind, value] = SHORTCUTS[key];
        setLabel(kind, value);
      } else if (key === "s") {
        event.preventDefault();
        saveActive(false);
      } else if (key === "n" || event.key === "ArrowRight") {
        event.preventDefault();
        move(1);
      } else if (key === "b" || event.key === "ArrowLeft") {
        event.preventDefault();
        move(-1);
      } else if (event.key === " ") {
        event.preventDefault();
        togglePlay();
      }
    });
    loadCatalog();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    video_root = args.video_root.expanduser().resolve()
    rows = build_rows(video_root)
    annotations = read_annotations(args.output_csv)
    write_annotations(args.output_csv, rows, annotations)

    print(f"video_root={video_root}")
    print(f"episodes={len(rows)}")
    print(f"with_video={sum(1 for row in rows if row.videos)}")
    print(f"output_csv={args.output_csv.resolve()}")
    print(f"url=http://{args.host}:{args.port}")

    handler = make_handler(rows, annotations, args.output_csv.resolve(), video_root)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping annotation server.")


if __name__ == "__main__":
    main()
