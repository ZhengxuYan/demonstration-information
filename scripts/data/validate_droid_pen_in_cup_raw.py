#!/usr/bin/env python3
"""Validate raw DROID pen-in-cup trajectories before RLDS conversion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py


REQUIRED_DATASETS = (
    "action/cartesian_position",
    "action/cartesian_velocity",
    "action/gripper_position",
    "action/gripper_velocity",
    "action/joint_position",
    "action/joint_velocity",
    "observation/robot_state/cartesian_position",
    "observation/robot_state/gripper_position",
    "observation/robot_state/joint_positions",
    "observation/camera_type",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=102)
    return parser.parse_args()


def metadata_path(episode_dir: Path) -> Path | None:
    matches = sorted(episode_dir.glob("metadata_*.json"))
    return matches[0] if matches else None


def h5_has_path(h5: h5py.File, path: str) -> bool:
    node = h5
    for part in path.split("/"):
        if part not in node:
            return False
        node = node[part]
    return True


def main() -> None:
    args = parse_args()
    episodes = sorted(
        path
        for path in args.raw_root.iterdir()
        if path.is_dir() and (path / "trajectory.h5").is_file() and (path / "recordings" / "MP4").is_dir()
    )
    errors: list[str] = []
    lengths: list[int] = []
    missing_ext2 = 0

    for idx, episode_dir in enumerate(episodes):
        h5_path = episode_dir / "trajectory.h5"
        meta_path = metadata_path(episode_dir)
        if meta_path is None:
            errors.append(f"{episode_dir}: missing metadata_*.json")
            continue
        with meta_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        if metadata.get("success") is not True:
            errors.append(f"{episode_dir}: metadata success is not True")
        if str(metadata.get("ext2_cam_serial", "N/A")).upper() == "N/A":
            missing_ext2 += 1

        with h5py.File(h5_path, "r") as h5:
            for required in REQUIRED_DATASETS:
                if not h5_has_path(h5, required):
                    errors.append(f"{episode_dir}: missing {required}")
            if h5_has_path(h5, "action/cartesian_position"):
                length = int(h5["action"]["cartesian_position"].shape[0])
                lengths.append(length)
                if length < 18:
                    errors.append(f"{episode_dir}: too short for n_obs=2,n_action=16 ({length})")
            camera_types = h5.get("observation/camera_type")
            if camera_types is None:
                continue
            serials = list(camera_types.keys())
            mp4_serials = {path.stem for path in (episode_dir / "recordings" / "MP4").glob("*.mp4")}
            wrist = [s for s in serials if int(camera_types[s][0]) == 0 and s in mp4_serials]
            exterior = [s for s in serials if int(camera_types[s][0]) != 0 and s in mp4_serials]
            if not wrist:
                errors.append(f"{episode_dir}: no wrist camera MP4")
            if not exterior:
                errors.append(f"{episode_dir}: no exterior camera MP4")
        if idx < 5:
            print(f"checked {idx:03d}: {episode_dir.name}")

    print(f"raw_root={args.raw_root}")
    print(f"episodes={len(episodes)}")
    print(f"missing_ext2={missing_ext2}")
    if lengths:
        print(f"length_min={min(lengths)} length_max={max(lengths)} length_mean={sum(lengths) / len(lengths):.1f}")

    if len(episodes) != args.expected_episodes:
        errors.append(f"expected {args.expected_episodes} episodes, found {len(episodes)}")

    if errors:
        print("VALIDATION_ERRORS")
        for error in errors[:100]:
            print(error)
        raise SystemExit(1)
    print("RAW_VALIDATION_OK")


if __name__ == "__main__":
    main()
