#!/usr/bin/env python3
"""Validate timestamp-based MP4 frame alignment for DROID pen-in-cup raw data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-mean-delta-ms", type=float, default=20.0)
    parser.add_argument("--max-max-delta-ms", type=float, default=40.0)
    return parser.parse_args()


def load_metadata(episode_dir: Path) -> dict:
    candidates = sorted(episode_dir.glob("metadata_*.json"))
    if not candidates:
        return {}
    with candidates[0].open("r", encoding="utf-8") as f:
        return json.load(f)


def camera_ids(h5: h5py.File, mp4_dir: Path, metadata: dict) -> tuple[str, str]:
    camera_types = h5["observation"]["camera_type"]
    available = {path.stem for path in mp4_dir.glob("*.mp4") if "stereo" not in path.stem.lower()}
    wrist_hint = str(metadata.get("wrist_cam_serial", ""))
    ext1_hint = str(metadata.get("ext1_cam_serial", ""))
    wrist_ids = [serial for serial in camera_types if int(camera_types[serial][0]) == 0 and serial in available]
    exterior_ids = [serial for serial in camera_types if int(camera_types[serial][0]) != 0 and serial in available]
    wrist = wrist_hint if wrist_hint in available else wrist_ids[0]
    ext1 = ext1_hint if ext1_hint in available else exterior_ids[0]
    return wrist, ext1


def camera_timestamps(h5: h5py.File, serial: str) -> np.ndarray:
    cameras = h5["observation"]["timestamp"]["cameras"]
    for suffix in ("estimated_capture", "frame_received"):
        key = f"{serial}_{suffix}"
        if key in cameras:
            return np.asarray(cameras[key][:], dtype=np.float64)
    raise ValueError(f"missing timestamp for {serial}")


def frame_count(mp4_path: Path) -> int:
    cap = cv2.VideoCapture(str(mp4_path))
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if count <= 0:
        raise ValueError(f"bad frame count for {mp4_path}")
    return count


def aligned_frame_indices(frame_count: int, target_timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(target_timestamps) == frame_count:
        indices = np.arange(frame_count, dtype=np.int64)
        return indices, target_timestamps, np.zeros(frame_count, dtype=np.float64)
    frame_timestamps = np.linspace(target_timestamps[0], target_timestamps[-1], frame_count)
    right = np.searchsorted(frame_timestamps, target_timestamps, side="left")
    right = np.clip(right, 1, frame_count - 1)
    left = right - 1
    choose_left = np.abs(target_timestamps - frame_timestamps[left]) <= np.abs(frame_timestamps[right] - target_timestamps)
    indices = np.where(choose_left, left, right).astype(np.int64)
    deltas = np.abs(target_timestamps - frame_timestamps[indices])
    return indices, frame_timestamps, deltas


def main() -> None:
    args = parse_args()
    episodes = sorted(
        path
        for path in args.raw_root.iterdir()
        if path.is_dir() and (path / "trajectory.h5").is_file() and (path / "recordings" / "MP4").is_dir()
    )
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    rows: list[dict[str, object]] = []
    errors: list[str] = []
    for ep_idx, episode_dir in enumerate(episodes):
        with h5py.File(episode_dir / "trajectory.h5", "r") as h5:
            length = int(h5["action"]["cartesian_position"].shape[0])
            wrist, ext1 = camera_ids(h5, episode_dir / "recordings" / "MP4", load_metadata(episode_dir))
            for serial, role in ((wrist, "wrist"), (ext1, "ext1")):
                count = frame_count(episode_dir / "recordings" / "MP4" / f"{serial}.mp4")
                target_ts = camera_timestamps(h5, serial)
                indices, _, deltas = aligned_frame_indices(count, target_ts)
                row = {
                    "ep_idx": ep_idx,
                    "episode": episode_dir.name,
                    "role": role,
                    "serial": serial,
                    "trajectory_len": length,
                    "mp4_frames": count,
                    "first_index": int(indices[0]),
                    "last_index": int(indices[-1]),
                    "unique_indices": int(len(np.unique(indices))),
                    "duplicate_indices": int(len(indices) - len(np.unique(indices))),
                    "mean_delta_ms": float(np.mean(deltas)),
                    "max_delta_ms": float(np.max(deltas)),
                }
                rows.append(row)
                if row["trajectory_len"] != len(indices):
                    errors.append(f"{episode_dir.name} {role}: length mismatch")
                if np.any(np.diff(indices) < 0):
                    errors.append(f"{episode_dir.name} {role}: non-monotonic indices")
                if row["mean_delta_ms"] > args.max_mean_delta_ms:
                    errors.append(f"{episode_dir.name} {role}: mean delta {row['mean_delta_ms']:.2f} ms")
                if row["max_delta_ms"] > args.max_max_delta_ms:
                    errors.append(f"{episode_dir.name} {role}: max delta {row['max_delta_ms']:.2f} ms")

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)

    print(f"episodes={len(episodes)}")
    if rows:
        print(f"trajectory_len_min={min(int(row['trajectory_len']) for row in rows)}")
        print(f"trajectory_len_max={max(int(row['trajectory_len']) for row in rows)}")
        print(f"mp4_frames_min={min(int(row['mp4_frames']) for row in rows)}")
        print(f"mp4_frames_max={max(int(row['mp4_frames']) for row in rows)}")
        print(f"mean_delta_ms_max={max(float(row['mean_delta_ms']) for row in rows):.2f}")
        print(f"max_delta_ms_max={max(float(row['max_delta_ms']) for row in rows):.2f}")
        print(f"duplicate_indices_total={sum(int(row['duplicate_indices']) for row in rows)}")
    if args.output_csv is not None:
        print(f"output_csv={args.output_csv}")
    if errors:
        print("ALIGNMENT_VALIDATION_ERRORS")
        for error in errors[:100]:
            print(error)
        raise SystemExit(1)
    print("ALIGNMENT_VALIDATION_OK")


if __name__ == "__main__":
    main()
