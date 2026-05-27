#!/usr/bin/env python3
"""Serve a local annotation app for POMDP-VLA rollout quality labels.

The app expects a manifest from render_pomdp_vla_annotation_videos.py and
autosaves labels to CSV.
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


@dataclass(frozen=True)
class Row:
    demo_key: str
    ep_idx: int
    video: Path
    source_hdf5: str
    num_states: int
    rendered_frames: int


VALID_SCORES = {"", "1", "2", "3"}
FIELDNAMES = [
    "demo_key",
    "ep_idx",
    "quality_score",
    "observability_score",
    "note",
    "video",
    "source_hdf5",
    "updated_at",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def read_manifest(path: Path, video_root: Path | None = None) -> list[Row]:
    rows: list[Row] = []
    with path.open(newline="") as f:
        for item in csv.DictReader(f):
            video = Path(item["video"])
            if video_root is not None:
                video = video_root / video.name
            rows.append(
                Row(
                    demo_key=item["demo_key"],
                    ep_idx=int(item["ep_idx"]),
                    video=video,
                    source_hdf5=item.get("source_hdf5", ""),
                    num_states=int(item.get("num_states") or 0),
                    rendered_frames=int(item.get("rendered_frames") or 0),
                )
            )
    return sorted(rows, key=lambda row: row.ep_idx)


def read_annotations(path: Path) -> dict[str, dict[str, str]]:
    annotations: dict[str, dict[str, str]] = {}
    if not path.exists():
        return annotations
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            annotations[row["demo_key"]] = {
                "quality_score": row.get("quality_score", ""),
                "observability_score": row.get("observability_score", ""),
                "note": row.get("note", ""),
                "updated_at": row.get("updated_at", ""),
            }
    return annotations


def write_annotations(path: Path, rows: list[Row], annotations: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            ann = annotations.get(row.demo_key, {})
            quality = ann.get("quality_score", "")
            observability = ann.get("observability_score", "")
            note = ann.get("note", "")
            if not quality and not observability and not note:
                continue
            writer.writerow(
                {
                    "demo_key": row.demo_key,
                    "ep_idx": row.ep_idx,
                    "quality_score": quality,
                    "observability_score": observability,
                    "note": note,
                    "video": str(row.video),
                    "source_hdf5": row.source_hdf5,
                    "updated_at": ann.get("updated_at", ""),
                }
            )


def row_to_json(row: Row) -> dict[str, object]:
    return {
        "demo_key": row.demo_key,
        "ep_idx": row.ep_idx,
        "video_url": f"/media?path={quote(str(row.video.resolve()))}",
        "video": str(row.video),
        "source_hdf5": row.source_hdf5,
        "num_states": row.num_states,
        "rendered_frames": row.rendered_frames,
    }


def make_handler(
    rows: list[Row],
    annotations: dict[str, dict[str, str]],
    output_csv: Path,
    allowed_roots: list[Path],
):
    row_by_key = {row.demo_key: row for row in rows}

    class Handler(BaseHTTPRequestHandler):
        server_version = "PomdpVlaAnnotation/1.0"

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
                self.send_json(
                    {
                        "rows": [row_to_json(row) for row in rows],
                        "annotations": annotations,
                        "output_csv": str(output_csv),
                    }
                )
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
                demo_key = str(body["demo_key"])
                quality_score = str(body.get("quality_score", ""))
                observability_score = str(body.get("observability_score", ""))
                note = str(body.get("note", ""))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                self.send_json({"error": f"bad annotation payload: {exc}"}, status=400)
                return
            if demo_key not in row_by_key:
                self.send_json({"error": f"unknown demo_key {demo_key}"}, status=404)
                return
            if quality_score not in VALID_SCORES:
                self.send_json({"error": f"bad quality_score {quality_score}"}, status=400)
                return
            if observability_score not in VALID_SCORES:
                self.send_json({"error": f"bad observability_score {observability_score}"}, status=400)
                return

            annotations[demo_key] = {
                "quality_score": quality_score,
                "observability_score": observability_score,
                "note": note,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
            write_annotations(output_csv, rows, annotations)
            self.send_json({"ok": True, "annotation": annotations[demo_key], "output_csv": str(output_csv)})

        def serve_index(self) -> None:
            data = INDEX_HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def serve_media(self, query: str) -> None:
            values = parse_qs(query).get("path", [])
            if not values:
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            path = Path(values[0]).resolve()
            if not any(path == root or root in path.parents for root in allowed_roots):
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            if not path.exists() or not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return

            file_size = path.stat().st_size
            range_header = self.headers.get("Range")
            start = 0
            end = file_size - 1
            status = HTTPStatus.OK
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

            self.send_response(status)
            self.send_header("Content-Type", mimetypes.guess_type(str(path))[0] or "application/octet-stream")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.end_headers()

            with path.open("rb") as f:
                f.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = f.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

    return Handler


INDEX_HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>POMDP-VLA Annotation</title>
  <style>
    :root { color-scheme: light; --border:#d8dee8; --ink:#18202b; --muted:#5e6a7a; --accent:#1f6feb; }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--ink); background: #f6f8fb; }
    header { position: sticky; top: 0; z-index: 5; background: white; border-bottom: 1px solid var(--border); padding: 12px 20px; display: flex; gap: 16px; align-items: center; justify-content: space-between; }
    header h1 { font-size: 17px; margin: 0; }
    .stats { color: var(--muted); font-size: 13px; }
    main { max-width: 1220px; margin: 0 auto; padding: 18px; }
    .toolbar { display: flex; gap: 10px; align-items: center; margin-bottom: 14px; flex-wrap: wrap; }
    button, select, input, textarea { font: inherit; }
    button { border: 1px solid var(--border); background: white; padding: 7px 10px; border-radius: 6px; cursor: pointer; }
    button.primary { background: var(--accent); color: white; border-color: var(--accent); }
    button.score { min-width: 42px; }
    button.score.active { background: var(--accent); color: white; border-color: var(--accent); }
    .row { background: white; border: 1px solid var(--border); border-radius: 8px; padding: 14px; margin-bottom: 14px; }
    .topline { display: flex; justify-content: space-between; gap: 10px; align-items: baseline; margin-bottom: 10px; }
    .title { font-weight: 650; }
    .meta { color: var(--muted); font-size: 12px; }
    .grid { display: grid; grid-template-columns: minmax(420px, 2fr) minmax(320px, 1fr); gap: 14px; align-items: start; }
    video { width: 100%; background: #0b0d10; border-radius: 6px; display: block; }
    .hint { color: var(--muted); font-size: 12px; margin-top: 6px; }
    .panel { border: 1px solid var(--border); border-radius: 8px; padding: 12px; }
    .field { margin-bottom: 12px; }
    .field label { display: block; font-weight: 650; margin-bottom: 6px; }
    .scores { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    textarea { width: 100%; min-height: 62px; resize: vertical; border: 1px solid var(--border); border-radius: 6px; padding: 8px; }
    .saved { color: #22733b; font-size: 12px; min-height: 18px; }
    .legend { font-size: 12px; color: var(--muted); line-height: 1.45; }
    @media (max-width: 860px) { .grid { grid-template-columns: 1fr; } main { padding: 10px; } }
  </style>
</head>
<body>
  <header>
    <h1>POMDP-VLA seed1 annotation</h1>
    <div class="stats" id="stats"></div>
  </header>
  <main>
    <div class="toolbar">
      <button id="prev">Prev</button>
      <button id="next" class="primary">Next</button>
      <button id="toggle-filter">Show unlabeled only</button>
      <span class="stats" id="output"></span>
    </div>
    <div class="legend">
      Quality: 3 = successful and efficient, 2 = successful but suboptimal or messy, 1 = failed or clearly bad.
      Observability: 3 = task-relevant state is clear from left_close_low + wrist, 2 = partially visible, 1 = hard to infer.
    </div>
    <div id="rows"></div>
  </main>
  <script>
    let catalog = [];
    let annotations = {};
    let unlabeledOnly = false;
    let currentIndex = 0;

    function isComplete(row) {
      const ann = annotations[row.demo_key] || {};
      return ann.quality_score && ann.observability_score;
    }

    function visibleRows() {
      return catalog.filter(row => !unlabeledOnly || !isComplete(row));
    }

    function scoreButtons(row, field) {
      const ann = annotations[row.demo_key] || {};
      const current = ann[field] || "";
      return [1, 2, 3].map(score => {
        const active = String(score) === current ? " active" : "";
        return `<button class="score${active}" data-demo="${row.demo_key}" data-field="${field}" data-score="${score}">${score}</button>`;
      }).join("");
    }

    function render() {
      const rows = visibleRows();
      currentIndex = Math.max(0, Math.min(currentIndex, Math.max(rows.length - 1, 0)));
      const done = catalog.filter(isComplete).length;
      document.getElementById("stats").textContent = `${done}/${catalog.length} complete`;
      const output = window.outputCsv || "";
      document.getElementById("output").textContent = output ? `saving to ${output}` : "";
      document.getElementById("toggle-filter").textContent = unlabeledOnly ? "Show all" : "Show unlabeled only";
      if (!rows.length) {
        document.getElementById("rows").innerHTML = "<p>All rows are labeled.</p>";
        return;
      }
      const row = rows[currentIndex];
      const ann = annotations[row.demo_key] || {};
      document.getElementById("rows").innerHTML = `
        <section class="row">
          <div class="topline">
            <div class="title">${row.demo_key} · episode ${row.ep_idx}</div>
            <div class="meta">${currentIndex + 1}/${rows.length} · ${row.rendered_frames} frames</div>
          </div>
          <div class="grid">
            <div>
              <video src="${row.video_url}" controls autoplay muted loop playsinline></video>
              <div class="hint">Left: left_close_low third-person. Right: wrist camera.</div>
            </div>
            <div class="panel">
              <div class="field">
                <label>Quality score</label>
                <div class="scores">${scoreButtons(row, "quality_score")}</div>
              </div>
              <div class="field">
                <label>Observability score</label>
                <div class="scores">${scoreButtons(row, "observability_score")}</div>
              </div>
              <div class="field">
                <label>Note</label>
                <textarea id="note" data-demo="${row.demo_key}">${ann.note || ""}</textarea>
              </div>
              <div class="saved" id="saved">${ann.updated_at ? "Saved " + ann.updated_at : ""}</div>
            </div>
          </div>
        </section>`;
      document.querySelectorAll("button.score").forEach(btn => {
        btn.addEventListener("click", () => {
          const demo = btn.dataset.demo;
          const field = btn.dataset.field;
          annotations[demo] = annotations[demo] || {};
          annotations[demo][field] = btn.dataset.score;
          save(demo);
        });
      });
      document.getElementById("note").addEventListener("change", (event) => {
        const demo = event.target.dataset.demo;
        annotations[demo] = annotations[demo] || {};
        annotations[demo].note = event.target.value;
        save(demo);
      });
    }

    async function save(demo) {
      const ann = annotations[demo] || {};
      const res = await fetch("/api/annotation", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          demo_key: demo,
          quality_score: ann.quality_score || "",
          observability_score: ann.observability_score || "",
          note: ann.note || ""
        })
      });
      if (!res.ok) {
        alert(await res.text());
        return;
      }
      const payload = await res.json();
      annotations[demo] = payload.annotation;
      render();
    }

    document.getElementById("prev").onclick = () => { currentIndex -= 1; render(); };
    document.getElementById("next").onclick = () => { currentIndex += 1; render(); };
    document.getElementById("toggle-filter").onclick = () => { unlabeledOnly = !unlabeledOnly; currentIndex = 0; render(); };
    document.addEventListener("keydown", (event) => {
      if (event.key === "ArrowRight") { currentIndex += 1; render(); }
      if (event.key === "ArrowLeft") { currentIndex -= 1; render(); }
    });

    fetch("/api/catalog").then(res => res.json()).then(data => {
      catalog = data.rows;
      annotations = data.annotations || {};
      window.outputCsv = data.output_csv;
      render();
    });
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    video_root = args.video_root.resolve() if args.video_root else None
    rows = read_manifest(args.manifest, video_root=video_root)
    annotations = read_annotations(args.output_csv)
    media_root = video_root or args.manifest.parent.resolve()
    allowed_roots = [media_root, args.manifest.parent.resolve(), args.output_csv.parent.resolve()]
    handler = make_handler(rows, annotations, args.output_csv, allowed_roots)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"serving http://{args.host}:{args.port}")
    print(f"manifest={args.manifest}")
    print(f"output_csv={args.output_csv}")
    server.serve_forever()


if __name__ == "__main__":
    main()
