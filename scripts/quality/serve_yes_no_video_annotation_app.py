#!/usr/bin/env python3
"""Serve a local yes/no video annotation app.

Example:

python -u scripts/quality/serve_yes_no_video_annotation_app.py \
  --video-root /Users/jasonyan/Desktop/pomdp_vla_square_rollouts/videos \
  --output-csv /Users/jasonyan/Desktop/pomdp_vla_square_rollouts/quality_annotations.csv \
  --port 8768
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import mimetypes
import shutil
import re
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse


VALID_LABELS = {"yes", "no", "unlabeled"}


@dataclass(frozen=True)
class Item:
    index: int
    source: str
    demo_key: str
    title: str
    video: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def demo_sort_key(path: Path) -> tuple[str, int, str]:
    match = re.search(r"demo_(\d+)", path.stem)
    demo_idx = int(match.group(1)) if match else 10**12
    return str(path.parent.relative_to(path.parents[1])), demo_idx, path.name


def discover_items(video_root: Path) -> list[Item]:
    videos = sorted(video_root.rglob("*.mp4"), key=demo_sort_key)
    items = []
    for idx, path in enumerate(videos, start=1):
        source = path.parent.name
        match = re.search(r"(demo_\d+)", path.stem)
        demo_key = match.group(1) if match else path.stem
        items.append(
            Item(
                index=idx,
                source=source,
                demo_key=demo_key,
                title=f"{source} / {demo_key}",
                video=path.resolve(),
            )
        )
    return items


def read_annotations(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path.exists():
        return {}
    annotations = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (row.get("source", ""), row.get("demo_key", ""))
            annotations[key] = row
    return annotations


def encode_annotations(annotations: dict[tuple[str, str], dict[str, str]]) -> dict[str, dict[str, str]]:
    return {f"{source}::{demo_key}": row for (source, demo_key), row in annotations.items()}


def write_annotations(path: Path, items: list[Item], annotations: dict[tuple[str, str], dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["index", "source", "demo_key", "label", "note", "updated_at", "video"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in items:
            row = annotations.get((item.source, item.demo_key), {})
            writer.writerow(
                {
                    "index": item.index,
                    "source": item.source,
                    "demo_key": item.demo_key,
                    "label": row.get("label", "unlabeled"),
                    "note": row.get("note", ""),
                    "updated_at": row.get("updated_at", ""),
                    "video": str(item.video),
                }
            )


def build_page() -> bytes:
    doc = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Quality Annotation</title>
  <style>
    :root { color-scheme: light; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; background: #f7f8f5; color: #171917; }
    header { position: sticky; top: 0; z-index: 2; display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 14px 18px; background: rgba(247,248,245,.96); border-bottom: 1px solid #d9ddd4; }
    h1 { margin: 0; font-size: 18px; font-weight: 650; }
    .meta { color: #5f665f; font-size: 13px; }
    main { display: grid; grid-template-columns: minmax(0, 1fr) 280px; gap: 18px; padding: 18px; }
    .viewer { min-width: 0; }
    video { width: 100%; max-height: calc(100vh - 190px); background: #111; border-radius: 6px; }
    .title { margin: 12px 0 6px; font-size: 15px; font-weight: 650; }
    .path { color: #5f665f; font-size: 12px; overflow-wrap: anywhere; }
    .controls { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 14px 0; }
    button { border: 1px solid #b8beb5; background: #fff; color: #171917; border-radius: 6px; padding: 12px 14px; font-size: 15px; font-weight: 650; cursor: pointer; }
    button:hover { background: #eef1eb; }
    button.active.yes { background: #146c43; border-color: #146c43; color: #fff; }
    button.active.no { background: #9f2d20; border-color: #9f2d20; color: #fff; }
    .wide { grid-column: 1 / -1; }
    textarea { width: 100%; min-height: 68px; resize: vertical; box-sizing: border-box; border: 1px solid #b8beb5; border-radius: 6px; padding: 10px; font: inherit; }
    aside { border-left: 1px solid #d9ddd4; padding-left: 18px; }
    .list { display: grid; gap: 6px; max-height: calc(100vh - 120px); overflow: auto; }
    .row { display: grid; grid-template-columns: 42px 1fr 34px; gap: 8px; align-items: center; border: 1px solid #d9ddd4; border-radius: 6px; background: #fff; padding: 7px; cursor: pointer; }
    .row.current { outline: 2px solid #283f74; }
    .row-title { font-size: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .pill { justify-self: end; min-width: 26px; border-radius: 999px; padding: 2px 6px; text-align: center; font-size: 11px; background: #e6e9e2; color: #4f584f; }
    .pill.yes { background: #d7ecdf; color: #0f5a35; }
    .pill.no { background: #f3d7d3; color: #842419; }
    @media (max-width: 860px) { main { grid-template-columns: 1fr; } aside { border-left: 0; padding-left: 0; } }
  </style>
</head>
<body>
  <header>
    <h1>Quality Annotation</h1>
    <div class="meta" id="summary"></div>
  </header>
  <main>
    <section class="viewer">
      <video id="video" controls autoplay muted preload="metadata"></video>
      <div class="title" id="title"></div>
      <div class="path" id="path"></div>
      <div class="controls">
        <button id="yes" class="yes">Yes</button>
        <button id="no" class="no">No</button>
        <button id="speed1">1x</button>
        <button id="speed15">1.5x</button>
        <button id="speed2">2x</button>
        <button id="speed4">4x</button>
        <button id="clear" class="wide">Clear</button>
      </div>
      <textarea id="note" placeholder="Note"></textarea>
    </section>
    <aside>
      <div class="list" id="list"></div>
    </aside>
  </main>
  <script>
    let items = [];
    let annotations = {};
    let current = 0;
    const video = document.getElementById('video');
    const title = document.getElementById('title');
    const pathEl = document.getElementById('path');
    const note = document.getElementById('note');
    const summary = document.getElementById('summary');
    const list = document.getElementById('list');
    const yesBtn = document.getElementById('yes');
    const noBtn = document.getElementById('no');
    const clearBtn = document.getElementById('clear');
    const speedBtns = [
      [document.getElementById('speed1'), 1.0],
      [document.getElementById('speed15'), 1.5],
      [document.getElementById('speed2'), 2.0],
      [document.getElementById('speed4'), 4.0],
    ];
    let playbackRate = 4.0;

    function key(item) { return item.source + '::' + item.demo_key; }
    function labelFor(item) { return (annotations[key(item)] || {}).label || 'unlabeled'; }

    async function load() {
      const res = await fetch('/api/catalog');
      const data = await res.json();
      items = data.items;
      annotations = data.annotations;
      renderList();
      show(0);
    }

    function renderList() {
      list.innerHTML = '';
      items.forEach((item, idx) => {
        const label = labelFor(item);
        const row = document.createElement('div');
        row.className = 'row' + (idx === current ? ' current' : '');
        row.onclick = () => show(idx);
        row.innerHTML = `<div>${idx + 1}</div><div class="row-title">${item.title}</div><div class="pill ${label}">${label === 'unlabeled' ? '-' : label}</div>`;
        list.appendChild(row);
      });
      const counts = {yes: 0, no: 0, unlabeled: 0};
      items.forEach(item => counts[labelFor(item)] = (counts[labelFor(item)] || 0) + 1);
      summary.textContent = `${items.length} videos | yes ${counts.yes || 0} | no ${counts.no || 0} | unlabeled ${counts.unlabeled || 0}`;
    }

    function show(idx) {
      if (!items.length) return;
      current = Math.max(0, Math.min(items.length - 1, idx));
      const item = items[current];
      const ann = annotations[key(item)] || {};
      video.src = '/media?path=' + encodeURIComponent(item.video);
      video.playbackRate = playbackRate;
      title.textContent = `${current + 1} / ${items.length}  ${item.title}`;
      pathEl.textContent = item.video;
      note.value = ann.note || '';
      yesBtn.classList.toggle('active', labelFor(item) === 'yes');
      noBtn.classList.toggle('active', labelFor(item) === 'no');
      renderList();
    }

    async function save(label) {
      const item = items[current];
      const body = {source: item.source, demo_key: item.demo_key, label, note: note.value};
      const res = await fetch('/api/annotate', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
      const data = await res.json();
      annotations = data.annotations;
      if (label !== 'unlabeled' && current + 1 < items.length) show(current + 1);
      else show(current);
    }

    yesBtn.onclick = () => save('yes');
    noBtn.onclick = () => save('no');
    clearBtn.onclick = () => save('unlabeled');
    speedBtns.forEach(([button, rate]) => {
      button.onclick = () => {
        playbackRate = rate;
        video.playbackRate = playbackRate;
      };
    });
    video.onloadedmetadata = () => { video.playbackRate = playbackRate; };
    note.onchange = () => save(labelFor(items[current]));
    document.addEventListener('keydown', (e) => {
      if (e.target === note) return;
      if (e.key === 'ArrowRight') show(current + 1);
      if (e.key === 'ArrowLeft') show(current - 1);
      if (e.key.toLowerCase() === 'y') save('yes');
      if (e.key.toLowerCase() === 'n') save('no');
      if (e.key === 'Backspace') save('unlabeled');
    });
    load();
  </script>
</body>
</html>
"""
    return doc.encode("utf-8")


class App:
    def __init__(self, items: list[Item], output_csv: Path):
        self.items = items
        self.output_csv = output_csv
        self.annotations = read_annotations(output_csv)
        write_annotations(output_csv, items, self.annotations)


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:
            return

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send(HTTPStatus.OK, build_page(), "text/html; charset=utf-8")
                return
            if parsed.path == "/api/catalog":
                payload = {
                    "items": [
                        {
                            "index": item.index,
                            "source": item.source,
                            "demo_key": item.demo_key,
                            "title": item.title,
                            "video": str(item.video),
                        }
                        for item in app.items
                    ],
                    "annotations": encode_annotations(app.annotations),
                }
                self._send_json(payload)
                return
            if parsed.path == "/media":
                params = parse_qs(parsed.query)
                path = Path(params.get("path", [""])[0]).resolve()
                allowed = {item.video for item in app.items}
                if path not in allowed or not path.exists():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                size = path.stat().st_size
                range_header = self.headers.get("Range")
                if range_header and range_header.startswith("bytes="):
                    start_text, _, end_text = range_header.removeprefix("bytes=").partition("-")
                    start = int(start_text) if start_text else 0
                    end = int(end_text) if end_text else size - 1
                    start = max(0, min(start, size - 1))
                    end = max(start, min(end, size - 1))
                    self.send_response(HTTPStatus.PARTIAL_CONTENT)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                    self.send_header("Content-Length", str(end - start + 1))
                    self.end_headers()
                    with path.open("rb") as f:
                        f.seek(start)
                        self.wfile.write(f.read(end - start + 1))
                else:
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Length", str(size))
                    self.end_headers()
                    with path.open("rb") as f:
                        shutil.copyfileobj(f, self.wfile)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/api/annotate":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length) or b"{}")
            label = data.get("label", "unlabeled")
            if label not in VALID_LABELS:
                self.send_error(HTTPStatus.BAD_REQUEST, f"invalid label {label}")
                return
            source = str(data.get("source", ""))
            demo_key = str(data.get("demo_key", ""))
            key = (source, demo_key)
            if key not in {(item.source, item.demo_key) for item in app.items}:
                self.send_error(HTTPStatus.BAD_REQUEST, "unknown item")
                return
            app.annotations[key] = {
                "label": label,
                "note": str(data.get("note", "")),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
            write_annotations(app.output_csv, app.items, app.annotations)
            self._send_json({"ok": True, "annotations": encode_annotations(app.annotations)})

        def _send_json(self, payload: object) -> None:
            self._send(HTTPStatus.OK, json.dumps(payload).encode("utf-8"), "application/json")

        def _send(self, status: HTTPStatus, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main() -> None:
    args = parse_args()
    items = discover_items(args.video_root)
    if not items:
        raise SystemExit(f"No MP4 files found under {args.video_root}")
    app = App(items, args.output_csv)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"Catalog: {len(items)} videos")
    print(f"Saving annotations to: {args.output_csv}")
    print(f"Open: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
