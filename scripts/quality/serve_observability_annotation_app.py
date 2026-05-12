"""Serve a local video annotation app for Square observability labels.

The app preloads existing Square PH labels, exports any missing PH wrist-view
videos from HDF5, and saves new labels to a single CSV as soon as a tag is
chosen in the browser.

Example:

python scripts/quality/serve_observability_annotation_app.py --port 8765
"""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PH_HDF5 = REPO_ROOT / "robomimic_square_ph" / "image.hdf5"
PH_EXISTING_VIDEOS = REPO_ROOT / "square_ph_wrist_annotation_deploy" / "videos"
PH_LEGACY_CSV = REPO_ROOT / "square_ph_observability_annotations.csv"
MH_HDF5 = REPO_ROOT / "image.hdf5"
MH_EXISTING_VIDEOS = REPO_ROOT / "square_mh_wrist_review" / "videos"
EXPERT_VIDEO_ROOT = REPO_ROOT / "expert200"
APP_ROOT = REPO_ROOT / "observability_annotation_app"
EXPERT_WRIST_VIDEOS = APP_ROOT / "videos" / "expert200_wrist"
OUTPUT_CSV = REPO_ROOT / "observability_annotations.csv"

PH_WRIST_KEY = "robot0_eye_in_hand_image"
MH_WRIST_KEY = "robot0_eye_in_hand_image"
VALID_LABELS = {"unlabeled", "full", "partial", "unsure", "unusable"}
LABEL_ORDER = {"full": 0, "partial": 1, "unsure": 2, "unusable": 3, "unlabeled": 4}


@dataclass(frozen=True)
class VideoRef:
    label: str
    path: Path


@dataclass(frozen=True)
class Row:
    dataset: str
    ep_idx: int
    title: str
    videos: tuple[VideoRef, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--app-root", type=Path, default=APP_ROOT)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--no-export-missing-ph",
        action="store_true",
        help="Do not export the missing Square PH videos from the HDF5 file.",
    )
    return parser.parse_args()


def demo_index(path: Path) -> int:
    match = re.search(r"demo_0*(\d+)\.mp4$", path.name)
    if not match:
        raise ValueError(f"Could not parse demo index from {path}")
    return int(match.group(1))


def write_video(video_path: Path, frames: np.ndarray, fps: int) -> None:
    video_path.parent.mkdir(parents=True, exist_ok=True)
    if video_path.exists():
        return

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to export missing PH videos")

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
        "-movflags",
        "+faststart",
        "-pix_fmt",
        "yuv420p",
        str(video_path),
    ]
    proc = subprocess.run(cmd, input=frames.tobytes(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="replace"))


def export_missing_ph_videos(app_root: Path, fps: int) -> None:
    generated_dir = app_root / "videos" / "square_ph"
    missing = [
        ep_idx
        for ep_idx in range(200)
        if not (PH_EXISTING_VIDEOS / f"demo_{ep_idx:04d}.mp4").exists()
        and not (generated_dir / f"demo_{ep_idx:04d}.mp4").exists()
    ]
    if not missing:
        return

    with h5py.File(PH_HDF5, "r") as f:
        for ep_idx in missing:
            generated = generated_dir / f"demo_{ep_idx:04d}.mp4"
            frames = f["data"][f"demo_{ep_idx}"]["obs"][PH_WRIST_KEY][:]
            write_video(generated, frames, fps)


def mh_review_videos_by_ep() -> dict[int, Path]:
    videos: dict[int, Path] = {}
    for path in sorted(MH_EXISTING_VIDEOS.glob("demo_*.mp4")):
        match = re.match(r"demo_0*(\d+)_", path.name)
        if match:
            videos[int(match.group(1))] = path
    return videos


def export_hdf5_wrist_video(hdf5_path: Path, ep_idx: int, output_path: Path, fps: int) -> None:
    if output_path.exists():
        return
    with h5py.File(hdf5_path, "r") as f:
        frames = f["data"][f"demo_{ep_idx}"]["obs"][MH_WRIST_KEY][:]
    write_video(output_path, frames, fps)


def build_rows(app_root: Path, export_ph: bool, fps: int) -> list[Row]:
    if export_ph:
        export_missing_ph_videos(app_root, fps)

    rows: list[Row] = []
    generated_ph = app_root / "videos" / "square_ph"
    for ep_idx in range(200):
        existing = PH_EXISTING_VIDEOS / f"demo_{ep_idx:04d}.mp4"
        generated = generated_ph / f"demo_{ep_idx:04d}.mp4"
        video = existing if existing.exists() else generated
        if video.exists():
            rows.append(Row("square_ph", ep_idx, f"PH demo_{ep_idx:04d}", (VideoRef("wrist", video),)))

    mh_existing = mh_review_videos_by_ep()
    mh_generated_dir = app_root / "videos" / "square_mh"
    for ep_idx in range(300):
        generated = mh_generated_dir / f"demo_{ep_idx:04d}.mp4"
        video = generated if generated.exists() else (mh_existing.get(ep_idx) or generated)
        rows.append(Row("square_mh", ep_idx, f"MH demo_{ep_idx:04d}", (VideoRef("wrist", video),)))

    expert_paths = sorted(EXPERT_VIDEO_ROOT.glob("demo_*.mp4"), key=demo_index)
    for path in expert_paths:
        ep_idx = demo_index(path)
        wrist = EXPERT_WRIST_VIDEOS / f"demo_{ep_idx:04d}.mp4"
        if wrist.exists():
            rows.append(Row("expert200", ep_idx, f"Expert demo_{ep_idx:03d}", (VideoRef("wrist", wrist),)))
        else:
            rows.append(Row("expert200", ep_idx, f"Expert demo_{ep_idx:03d}", (VideoRef("video", path),)))

    return rows


def read_annotations(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    annotations: dict[tuple[str, int], dict[str, str]] = {}

    if PH_LEGACY_CSV.exists():
        with PH_LEGACY_CSV.open(newline="") as f:
            for row in csv.DictReader(f):
                ep_idx = int(row["ep_idx"])
                label = (row.get("label") or "unlabeled").strip() or "unlabeled"
                annotations[("square_ph", ep_idx)] = {
                    "label": label if label in VALID_LABELS else "unlabeled",
                    "note": row.get("note") or "",
                    "updated_at": "",
                }

    if path.exists():
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                dataset = row["dataset"]
                ep_idx = int(row["ep_idx"])
                if dataset == "square_mh" and "manual_collected_hdf5_wrist" in (row.get("video") or ""):
                    continue
                label = (row.get("label") or "unlabeled").strip() or "unlabeled"
                annotations[(dataset, ep_idx)] = {
                    "label": label if label in VALID_LABELS else "unlabeled",
                    "note": row.get("note") or "",
                    "updated_at": row.get("updated_at") or "",
                }

    return annotations


def write_annotations(path: Path, rows: list[Row], annotations: dict[tuple[str, int], dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "ep_idx", "label", "note", "updated_at", "video"])
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item.dataset, item.ep_idx)):
            ann = annotations.get((row.dataset, row.ep_idx), {})
            label = ann.get("label", "unlabeled")
            if label == "unlabeled" and not ann.get("note"):
                continue
            writer.writerow(
                {
                    "dataset": row.dataset,
                    "ep_idx": row.ep_idx,
                    "label": label,
                    "note": ann.get("note", ""),
                    "updated_at": ann.get("updated_at", ""),
                    "video": str(row.videos[0].path),
                }
            )


def row_to_json(row: Row) -> dict[str, object]:
    return {
        "dataset": row.dataset,
        "ep_idx": row.ep_idx,
        "title": row.title,
        "videos": [
            {
                "label": video.label,
                "url": f"/media?path={quote(str(video.path))}",
                "path": str(video.path),
                "ready": video.path.exists(),
            }
            for video in row.videos
        ],
    }


def annotation_key(dataset: str, ep_idx: int) -> str:
    return f"{dataset}:{ep_idx}"


def make_handler(rows: list[Row], annotations: dict[tuple[str, int], dict[str, str]], output_csv: Path):
    row_by_key = {(row.dataset, row.ep_idx): row for row in rows}

    class Handler(BaseHTTPRequestHandler):
        server_version = "ObservabilityAnnotation/1.0"

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
                label = str(body.get("label") or "unlabeled")
                note = str(body.get("note") or "")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                self.send_json({"error": f"bad annotation payload: {exc}"}, status=400)
                return

            key = (dataset, ep_idx)
            if key not in row_by_key:
                self.send_json({"error": "unknown dataset/demo"}, status=404)
                return
            if label not in VALID_LABELS:
                self.send_json({"error": f"invalid label: {label}"}, status=400)
                return

            annotations[key] = {
                "label": label,
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
                    "rows": [row_to_json(row) for row in rows],
                    "annotations": by_key,
                    "output_csv": str(output_csv),
                    "counts": {
                        dataset: sum(1 for row in rows if row.dataset == dataset)
                        for dataset in sorted({row.dataset for row in rows})
                    },
                }
            )

        def serve_media(self, query: str) -> None:
            values = parse_qs(query).get("path", [])
            if not values:
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            path = Path(values[0]).resolve()
            allowed_roots = [REPO_ROOT.resolve(), APP_ROOT.resolve()]
            if not any(path == root or root in path.parents for root in allowed_roots):
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            if not path.exists() or not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            else:
                file_size = path.stat().st_size
            range_header = self.headers.get("Range")
            content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"

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

            length = end - start + 1
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "public, max-age=31536000, immutable")
            self.send_header("Content-Length", str(length))
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.end_headers()

            with path.open("rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        def maybe_create_lazy_media(self, path: Path) -> bool:
            mh_generated_root = (APP_ROOT / "videos" / "square_mh").resolve()
            if not (path == mh_generated_root or mh_generated_root in path.parents):
                return False
            match = re.match(r"demo_0*(\d+)\.mp4$", path.name)
            if not match:
                return False
            ep_idx = int(match.group(1))
            try:
                export_hdf5_wrist_video(MH_HDF5, ep_idx, path, fps=20)
            except Exception as exc:
                print(f"Failed to lazily export {path}: {exc}")
                return False
            return path.exists()

    return Handler


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Square Observability Annotation</title>
  <style>
    :root {
      --bg: #f5f6f1;
      --panel: #ffffff;
      --ink: #171b1c;
      --muted: #66706d;
      --border: #d9dfd9;
      --accent: #136f63;
      --accent-soft: #e3f1ee;
      --warn: #a8521f;
      --bad: #9d3030;
      --shadow: 0 14px 38px rgba(27, 39, 36, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      background: var(--bg);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    button, input, select, textarea { font: inherit; }
    .app {
      min-height: 100vh;
      display: grid;
      grid-template-columns: 276px minmax(0, 1fr);
    }
    aside {
      border-right: 1px solid var(--border);
      background: #fbfcf8;
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 16px;
      position: sticky;
      top: 0;
      height: 100vh;
      overflow: auto;
    }
    h1 {
      margin: 0;
      font-size: 20px;
      line-height: 1.15;
      letter-spacing: 0;
    }
    .small { color: var(--muted); font-size: 12px; line-height: 1.45; }
    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
    }
    .dataset-tabs, .filter-grid { display: grid; gap: 8px; }
    .tab {
      border: 1px solid var(--border);
      background: #fff;
      border-radius: 8px;
      padding: 9px 10px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      cursor: pointer;
      color: var(--ink);
    }
    .tab.active {
      border-color: var(--accent);
      background: var(--accent-soft);
      color: #0d403a;
      font-weight: 650;
    }
    .tab span:last-child { color: var(--muted); font-size: 12px; }
    label.field {
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 12px;
    }
    input, select, textarea {
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 9px 10px;
      background: #fff;
      color: var(--ink);
    }
    main {
      min-width: 0;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
    }
    header {
      background: rgba(245, 246, 241, 0.94);
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
    .title-row { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
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
    .icon-btn, .primary-btn {
      min-height: 38px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: #fff;
      color: var(--ink);
      cursor: pointer;
      padding: 8px 12px;
    }
    .primary-btn { background: var(--accent); color: #fff; border-color: var(--accent); }
    .workspace {
      padding: 20px 22px 28px;
      display: grid;
      gap: 18px;
      align-content: start;
    }
    .video-grid {
      --video-max: 420px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, min(100%, var(--video-max))));
      gap: 14px;
      justify-content: start;
    }
    .video-grid.size-medium { --video-max: 560px; }
    .video-grid.size-large { --video-max: 760px; }
    .video-card {
      background: #101313;
      border-radius: 8px;
      overflow: hidden;
      box-shadow: var(--shadow);
      border: 1px solid #202929;
      max-width: var(--video-max);
    }
    .video-card .view-label {
      color: #dce8e4;
      font-size: 12px;
      padding: 8px 10px;
      display: flex;
      justify-content: space-between;
    }
    video {
      display: block;
      width: 100%;
      background: #050606;
      aspect-ratio: 1 / 1;
      object-fit: contain;
    }
    .annotation-area {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 300px;
      gap: 16px;
      align-items: start;
    }
    .choice-grid {
      display: grid;
      grid-template-columns: repeat(5, minmax(118px, 1fr));
      gap: 10px;
    }
    .choice {
      min-height: 58px;
      border: 1px solid var(--border);
      background: #fff;
      border-radius: 8px;
      cursor: pointer;
      display: grid;
      align-content: center;
      justify-items: start;
      padding: 10px 12px;
      color: var(--ink);
      text-align: left;
    }
    .choice strong { font-size: 15px; }
    .choice span { color: var(--muted); font-size: 12px; margin-top: 2px; }
    .choice.active { border-color: var(--accent); background: var(--accent-soft); }
    .choice[data-label="unusable"].active { border-color: var(--bad); background: #f7e8e8; }
    .choice[data-label="unsure"].active { border-color: var(--warn); background: #f8eee6; }
    textarea { min-height: 98px; resize: vertical; }
    .progress {
      display: grid;
      gap: 8px;
    }
    .bar {
      height: 10px;
      background: #e5e9e4;
      border-radius: 999px;
      overflow: hidden;
      border: 1px solid var(--border);
    }
    .bar div { height: 100%; background: var(--accent); width: 0; }
    .metrics { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; }
    .metric { padding: 10px; border: 1px solid var(--border); border-radius: 8px; background: #fff; }
    .metric span { display: block; color: var(--muted); font-size: 11px; margin-bottom: 3px; }
    .metric strong { font-size: 18px; }
    .queue {
      max-height: 260px;
      overflow: auto;
      display: grid;
      gap: 6px;
    }
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
    .label-dot {
      width: 9px;
      height: 9px;
      border-radius: 999px;
      display: inline-block;
      background: #c1c8c2;
      margin-right: 7px;
    }
    .label-dot.full { background: #136f63; }
    .label-dot.partial { background: #b9822c; }
    .label-dot.unsure { background: #a8521f; }
    .label-dot.unusable { background: #9d3030; }
    .path {
      color: var(--muted);
      font-size: 12px;
      word-break: break-all;
      line-height: 1.45;
    }
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
        <h1>Observability Annotation</h1>
        <div class="small" id="save-target">Loading catalog...</div>
      </div>
      <section class="panel">
        <div class="dataset-tabs" id="dataset-tabs"></div>
      </section>
      <section class="panel filter-grid">
        <label class="field">Filter
          <select id="filter">
            <option value="unlabeled_ready">unlabeled with video</option>
            <option value="unlabeled">unlabeled all</option>
            <option value="all">all</option>
            <option value="full">full</option>
            <option value="partial">partial</option>
            <option value="unsure">unsure</option>
            <option value="unusable">unusable</option>
          </select>
        </label>
        <label class="field">Search demo id
          <input id="search" placeholder="e.g. 42">
        </label>
        <label class="field">Auto advance
          <select id="auto-advance">
            <option value="on">on</option>
            <option value="off">off</option>
          </select>
        </label>
        <label class="field">Video size
          <select id="video-size">
            <option value="compact">compact</option>
            <option value="medium">medium</option>
            <option value="large">large</option>
          </select>
        </label>
        <label class="field">Auto play
          <select id="auto-play">
            <option value="on">on</option>
            <option value="off">off</option>
          </select>
        </label>
        <label class="field">Playback speed
          <select id="playback-speed">
            <option value="1">1x</option>
            <option value="1.5">1.5x</option>
            <option value="2" selected>2x</option>
            <option value="3">3x</option>
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
        <div class="small">Shortcuts: F full, P partial, U unsure, X unusable, E clear, N next, B back, Space play/pause.</div>
      </section>
    </aside>
    <main>
      <header>
        <div>
          <div class="title-row">
            <h2 id="title">Loading...</h2>
            <span class="badge" id="dataset-badge"></span>
            <span class="badge" id="label-badge"></span>
          </div>
          <div class="small" id="status">Starting local app.</div>
        </div>
        <div class="top-actions">
          <button class="icon-btn" id="prev">Back</button>
          <button class="primary-btn" id="next">Next</button>
        </div>
      </header>
      <section class="workspace">
        <div class="video-grid" id="video-grid"></div>
        <div class="annotation-area">
          <section class="panel">
            <div class="choice-grid" id="choice-grid">
              <button class="choice" data-label="full"><strong>Full</strong><span>F</span></button>
              <button class="choice" data-label="partial"><strong>Partial</strong><span>P</span></button>
              <button class="choice" data-label="unsure"><strong>Unsure</strong><span>U</span></button>
              <button class="choice" data-label="unusable"><strong>Unusable</strong><span>X</span></button>
              <button class="choice" data-label="unlabeled"><strong>Clear</strong><span>E</span></button>
            </div>
            <div style="height:12px"></div>
            <label class="field">Note
              <textarea id="note" placeholder="optional note"></textarea>
            </label>
            <div style="height:10px"></div>
            <div class="path" id="paths"></div>
          </section>
          <aside class="panel" style="position:static;height:auto;overflow:visible;border:1px solid var(--border);">
            <div class="metrics" id="metrics"></div>
            <div style="height:12px"></div>
            <div class="queue" id="queue"></div>
          </aside>
        </div>
      </section>
    </main>
  </div>
  <script>
    const DATASET_LABELS = {
      square_ph: "Square PH",
      square_mh: "Square MH",
      expert200: "Expert200",
    };
    const LABELS = ["full", "partial", "unsure", "unusable", "unlabeled"];
    const SHORTCUTS = { f: "full", p: "partial", u: "unsure", x: "unusable", e: "unlabeled" };
    let rows = [];
    let annotations = {};
    let activeDataset = "square_ph";
    let activeIndex = 0;
    let visibleRows = [];
    let noteTimer = null;

    const keyFor = row => `${row.dataset}:${row.ep_idx}`;
    const annFor = row => annotations[keyFor(row)] || { label: "unlabeled", note: "" };
    const isLabeled = label => label && label !== "unlabeled";
    const hasReadyVideo = row => row.videos.some(video => video.ready);

    async function loadCatalog() {
      const res = await fetch("/api/catalog");
      const data = await res.json();
      rows = data.rows;
      annotations = data.annotations || {};
      document.getElementById("save-target").textContent = `Saving to ${data.output_csv}`;
      renderTabs();
      render();
    }

    function datasetRows(dataset = activeDataset) {
      return rows.filter(row => row.dataset === dataset);
    }

    function rowMatches(row) {
      const filter = document.getElementById("filter").value;
      const search = document.getElementById("search").value.trim();
      const ann = annFor(row);
      if (filter === "unlabeled_ready") {
        if (ann.label !== "unlabeled" || !hasReadyVideo(row)) return false;
      } else if (filter !== "all" && ann.label !== filter) {
        return false;
      }
      if (search && !String(row.ep_idx).includes(search)) return false;
      return true;
    }

    function filteredRows() {
      return datasetRows().filter(rowMatches).sort((a, b) => a.ep_idx - b.ep_idx);
    }

    function renderTabs() {
      const tabs = document.getElementById("dataset-tabs");
      const datasets = [...new Set(rows.map(row => row.dataset))];
      tabs.innerHTML = datasets.map(dataset => {
        const all = datasetRows(dataset);
        const labeled = all.filter(row => isLabeled(annFor(row).label)).length;
        return `<button class="tab ${dataset === activeDataset ? "active" : ""}" data-dataset="${dataset}">
          <span>${DATASET_LABELS[dataset] || dataset}</span><span>${labeled}/${all.length}</span>
        </button>`;
      }).join("");
      tabs.querySelectorAll("button").forEach(button => {
        button.addEventListener("click", () => {
          activeDataset = button.dataset.dataset;
          activeIndex = 0;
          renderTabs();
          render();
        });
      });
    }

    function renderMetrics() {
      const all = datasetRows();
      const counts = Object.fromEntries(LABELS.map(label => [label, 0]));
      all.forEach(row => counts[annFor(row).label || "unlabeled"] += 1);
      const labeled = all.length - counts.unlabeled;
      const ready = all.filter(hasReadyVideo).length;
      document.getElementById("metrics").innerHTML = `
        <div class="metric"><span>Total</span><strong>${all.length}</strong></div>
        <div class="metric"><span>With video</span><strong>${ready}</strong></div>
        <div class="metric"><span>Labeled</span><strong>${labeled}</strong></div>
        <div class="metric"><span>Full</span><strong>${counts.full}</strong></div>
        <div class="metric"><span>Partial</span><strong>${counts.partial}</strong></div>
        <div class="metric"><span>Unsure</span><strong>${counts.unsure}</strong></div>
        <div class="metric"><span>Unusable</span><strong>${counts.unusable}</strong></div>`;
      document.getElementById("progress-text").textContent = `${DATASET_LABELS[activeDataset]} progress: ${labeled}/${all.length}`;
      document.getElementById("progress-bar").style.width = `${all.length ? (100 * labeled / all.length) : 0}%`;
    }

    function renderQueue() {
      const queue = document.getElementById("queue");
      const around = visibleRows.slice(Math.max(0, activeIndex - 10), activeIndex + 31);
      queue.innerHTML = around.map(row => {
        const index = visibleRows.indexOf(row);
        const ann = annFor(row);
        return `<button class="${index === activeIndex ? "active" : ""}" data-index="${index}">
          <span><span class="label-dot ${ann.label}"></span>${row.title}</span><span>${ann.label}</span>
        </button>`;
      }).join("");
      queue.querySelectorAll("button").forEach(button => {
        button.addEventListener("click", () => {
          activeIndex = Number(button.dataset.index);
          render();
        });
      });
    }

    function render() {
      visibleRows = filteredRows();
      if (!visibleRows.length) {
        document.getElementById("title").textContent = "No demos match this filter";
        document.getElementById("dataset-badge").textContent = DATASET_LABELS[activeDataset] || activeDataset;
        document.getElementById("label-badge").textContent = "";
        document.getElementById("video-grid").innerHTML = "";
        document.getElementById("paths").textContent = "";
        document.getElementById("note").value = "";
        renderMetrics();
        renderQueue();
        return;
      }
      activeIndex = Math.max(0, Math.min(activeIndex, visibleRows.length - 1));
      const row = visibleRows[activeIndex];
      const ann = annFor(row);
      document.getElementById("title").textContent = row.title;
      document.getElementById("dataset-badge").textContent = `${DATASET_LABELS[row.dataset] || row.dataset} ${activeIndex + 1}/${visibleRows.length}`;
      document.getElementById("label-badge").textContent = ann.label;
      document.getElementById("note").value = ann.note || "";
      const singleVideo = row.videos.length === 1;
      document.getElementById("video-grid").innerHTML = row.videos.map(video => video.ready ? `
        <article class="video-card">
          <div class="view-label"><span>${singleVideo ? "video" : video.label}</span><span>${row.title}</span></div>
          <video controls muted loop playsinline preload="metadata" src="${video.url}"></video>
        </article>` : `
        <article class="video-card">
          <div class="view-label"><span>${singleVideo ? "video" : video.label}</span><span>${row.title}</span></div>
          <div style="aspect-ratio:1/1;display:grid;place-items:center;color:#dce8e4;padding:18px;text-align:center;">
            No pre-rendered wrist video for this demo.
          </div>
        </article>`).join("");
      applyVideoSize();
      applyPlaybackSpeed();
      maybeAutoPlay();
      document.getElementById("paths").innerHTML = row.videos.map(video => `${video.label}: ${video.path}`).join("<br>");
      document.querySelectorAll(".choice").forEach(button => {
        button.classList.toggle("active", button.dataset.label === ann.label);
      });
      renderMetrics();
      renderTabs();
      renderQueue();
    }

    function applyVideoSize() {
      const grid = document.getElementById("video-grid");
      const size = document.getElementById("video-size").value;
      grid.classList.toggle("size-medium", size === "medium");
      grid.classList.toggle("size-large", size === "large");
    }

    function maybeAutoPlay() {
      if (document.getElementById("auto-play").value !== "on") return;
      document.querySelectorAll("video").forEach(video => {
        video.playbackRate = Number(document.getElementById("playback-speed").value);
        video.currentTime = 0;
        video.play().catch(() => {
          document.getElementById("status").textContent = "Browser blocked autoplay; press Space once to start playback.";
        });
      });
    }

    function applyPlaybackSpeed() {
      const speed = Number(document.getElementById("playback-speed").value);
      document.querySelectorAll("video").forEach(video => {
        video.playbackRate = speed;
        video.onloadedmetadata = () => {
          video.playbackRate = speed;
        };
        video.onratechange = () => {
          if (Math.abs(video.playbackRate - speed) > 0.01) video.playbackRate = speed;
        };
      });
    }

    async function saveActive(label = null, note = null, advance = false) {
      if (!visibleRows.length) return;
      const row = visibleRows[activeIndex];
      const current = annFor(row);
      const next = {
        label: label || current.label || "unlabeled",
        note: note ?? document.getElementById("note").value,
      };
      annotations[keyFor(row)] = next;
      document.getElementById("status").textContent = `Saving ${row.title}...`;
      const res = await fetch("/api/annotation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset: row.dataset, ep_idx: row.ep_idx, label: next.label, note: next.note }),
      });
      const data = await res.json();
      if (!res.ok) {
        document.getElementById("status").textContent = data.error || "Save failed";
        return;
      }
      annotations[keyFor(row)] = data.annotation;
      document.getElementById("status").textContent = `Saved ${row.title} to ${data.output_csv}`;
      if (advance && document.getElementById("auto-advance").value === "on") {
        if (rowMatches(row)) activeIndex += 1;
        render();
      } else {
        render();
      }
    }

    function nextItem() {
      activeIndex = Math.min(activeIndex + 1, Math.max(0, visibleRows.length - 1));
      render();
    }

    function prevItem() {
      activeIndex = Math.max(activeIndex - 1, 0);
      render();
    }

    function togglePlay() {
      const videos = [...document.querySelectorAll("video")];
      const anyPlaying = videos.some(video => !video.paused);
      videos.forEach(video => {
        if (anyPlaying) video.pause();
        else video.play();
      });
    }

    document.getElementById("choice-grid").addEventListener("click", event => {
      const button = event.target.closest("button[data-label]");
      if (button) saveActive(button.dataset.label, null, true);
    });
    document.getElementById("note").addEventListener("input", event => {
      clearTimeout(noteTimer);
      noteTimer = setTimeout(() => saveActive(null, event.target.value, false), 450);
    });
    document.getElementById("next").addEventListener("click", nextItem);
    document.getElementById("prev").addEventListener("click", prevItem);
    document.getElementById("filter").addEventListener("change", () => { activeIndex = 0; render(); });
    document.getElementById("search").addEventListener("input", () => { activeIndex = 0; render(); });
    document.getElementById("video-size").addEventListener("change", applyVideoSize);
    document.getElementById("auto-play").addEventListener("change", () => {
      if (document.getElementById("auto-play").value === "on") maybeAutoPlay();
      else document.querySelectorAll("video").forEach(video => video.pause());
    });
    document.getElementById("playback-speed").addEventListener("change", applyPlaybackSpeed);
    document.addEventListener("keydown", event => {
      if (event.target.matches("textarea,input,select")) return;
      const key = event.key.toLowerCase();
      if (SHORTCUTS[key]) saveActive(SHORTCUTS[key], null, true);
      else if (key === "n" || event.key === "ArrowRight") nextItem();
      else if (key === "b" || event.key === "ArrowLeft") prevItem();
      else if (event.code === "Space") { event.preventDefault(); togglePlay(); }
    });

    loadCatalog().catch(err => {
      document.getElementById("status").textContent = `Failed to load catalog: ${err}`;
    });
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    args.app_root.mkdir(parents=True, exist_ok=True)
    rows = build_rows(args.app_root, export_ph=not args.no_export_missing_ph, fps=args.fps)
    annotations = read_annotations(args.output_csv)
    write_annotations(args.output_csv, rows, annotations)

    counts = {dataset: sum(1 for row in rows if row.dataset == dataset) for dataset in sorted({row.dataset for row in rows})}
    print(f"Catalog: {counts}")
    print(f"Saving annotations to: {args.output_csv}")
    print(f"Open: http://{args.host}:{args.port}")

    handler = make_handler(rows, annotations, args.output_csv)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
