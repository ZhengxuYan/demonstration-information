#!/usr/bin/env python3
"""Ensure a robomimic HDF5 file has non-empty train / valid masks."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite train / valid masks even if both already exist and are non-empty.",
    )
    return parser.parse_args()


def sorted_demo_keys(data_group) -> list[str]:
    return sorted(data_group.keys(), key=lambda key: int(key.split("_")[-1]))


def read_mask(mask_group, key: str) -> list[str]:
    if mask_group is None or key not in mask_group:
        return []
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in mask_group[key][:]]


def write_mask(f: h5py.File, train: list[str], valid: list[str], valid_ratio: float, seed: int) -> None:
    if "mask" in f:
        del f["mask"]
    mask = f.create_group("mask")
    mask.create_dataset("train", data=np.asarray([x.encode("utf-8") for x in train], dtype="S"))
    mask.create_dataset("valid", data=np.asarray([x.encode("utf-8") for x in valid], dtype="S"))
    f.attrs["mask_valid_ratio"] = float(valid_ratio)
    f.attrs["mask_split_seed"] = int(seed)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.valid_ratio < 1.0:
        raise ValueError(f"--valid-ratio must be in (0, 1), got {args.valid_ratio}")

    with h5py.File(args.dataset, "r+") as f:
        demos = sorted_demo_keys(f["data"])
        if len(demos) < 2:
            raise ValueError(f"{args.dataset} needs at least 2 demos for train / valid split")

        mask_group = f.get("mask")
        existing_train = read_mask(mask_group, "train")
        existing_valid = read_mask(mask_group, "valid")
        if existing_train and existing_valid and not args.overwrite:
            print(
                f"existing split ok {args.dataset}: "
                f"train={len(existing_train)} valid={len(existing_valid)}"
            )
            return

        rng = np.random.default_rng(args.seed)
        num_valid = max(1, int(round(args.valid_ratio * len(demos))))
        num_valid = min(num_valid, len(demos) - 1)
        valid_idx = set(rng.choice(np.arange(len(demos)), size=num_valid, replace=False).astype(int).tolist())
        train = [demo for idx, demo in enumerate(demos) if idx not in valid_idx]
        valid = [demo for idx, demo in enumerate(demos) if idx in valid_idx]
        if not train or not valid:
            raise AssertionError(f"empty train or valid split for {args.dataset}")
        write_mask(f, train, valid, args.valid_ratio, args.seed)
        print(f"wrote split {args.dataset}: train={len(train)} valid={len(valid)} seed={args.seed}")


if __name__ == "__main__":
    main()
