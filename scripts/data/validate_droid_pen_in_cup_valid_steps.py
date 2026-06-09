#!/usr/bin/env python3
"""Validate DROID pen-in-cup transition filtering masks."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--min-valid-steps", type=int, default=18)
    return parser.parse_args()


def dataset_at(h5: h5py.File, path: str) -> h5py.Dataset | None:
    node = h5
    for part in path.split("/"):
        if part not in node:
            return None
        node = node[part]
    return node if isinstance(node, h5py.Dataset) else None


def skip_action(h5: h5py.File, length: int) -> np.ndarray:
    dataset = dataset_at(h5, "observation/timestamp/skip_action")
    if dataset is None:
        dataset = dataset_at(h5, "observation/skip_action")
    if dataset is None:
        return np.zeros(length, dtype=bool)
    values = np.asarray(dataset[:], dtype=bool)
    if len(values) != length:
        raise ValueError(f"skip_action length {len(values)} != {length}")
    return values


def movement_enabled(h5: h5py.File, length: int) -> tuple[np.ndarray, str]:
    for path in ("observation/movement_enabled", "observation/timestamp/movement_enabled", "movement_enabled"):
        dataset = dataset_at(h5, path)
        if dataset is None:
            continue
        values = np.asarray(dataset[:], dtype=bool)
        if values.ndim > 0 and len(values) == length:
            return values, path
    return np.ones(length, dtype=bool), "default_true_no_per_step_dataset"


def main() -> None:
    args = parse_args()
    episodes = sorted(
        path
        for path in args.raw_root.iterdir()
        if path.is_dir() and (path / "trajectory.h5").is_file()
    )
    rows = []
    errors = []
    for ep_idx, episode_dir in enumerate(episodes):
        with h5py.File(episode_dir / "trajectory.h5", "r") as h5:
            length = int(h5["action"]["cartesian_position"].shape[0])
            skip = skip_action(h5, length)
            movement, movement_source = movement_enabled(h5, length)
            valid = movement & ~skip
            row = {
                "ep_idx": ep_idx,
                "episode": episode_dir.name,
                "length": length,
                "skip_true": int(skip.sum()),
                "skip_false": int((~skip).sum()),
                "movement_true": int(movement.sum()),
                "movement_false": int((~movement).sum()),
                "movement_source": movement_source,
                "valid_steps": int(valid.sum()),
            }
            rows.append(row)
            if row["valid_steps"] < args.min_valid_steps:
                errors.append(f"{episode_dir.name}: valid_steps={row['valid_steps']}")

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)

    print(f"episodes={len(rows)}")
    print(f"total_steps={sum(row['length'] for row in rows)}")
    print(f"skip_true={sum(row['skip_true'] for row in rows)}")
    print(f"skip_false={sum(row['skip_false'] for row in rows)}")
    print(f"movement_false={sum(row['movement_false'] for row in rows)}")
    print(f"valid_steps={sum(row['valid_steps'] for row in rows)}")
    print(f"valid_min={min(row['valid_steps'] for row in rows)}")
    print(f"valid_max={max(row['valid_steps'] for row in rows)}")
    print(f"movement_sources={sorted(set(row['movement_source'] for row in rows))}")
    if args.output_csv is not None:
        print(f"output_csv={args.output_csv}")
    if errors:
        print("VALID_STEP_VALIDATION_ERRORS")
        for error in errors[:100]:
            print(error)
        raise SystemExit(1)
    print("VALID_STEP_VALIDATION_OK")


if __name__ == "__main__":
    main()
