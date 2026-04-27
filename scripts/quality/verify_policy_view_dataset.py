#!/usr/bin/env python3
"""Verify a policy-view robomimic HDF5 dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--expected-demos", type=int, required=True)
    parser.add_argument("--expected-action-dim", type=int, required=True)
    parser.add_argument("--required-obs-key", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with h5py.File(args.dataset, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda key: int(key.split("_")[-1]))
        expected = [f"demo_{i}" for i in range(args.expected_demos)]
        if demos != expected:
            raise AssertionError(f"Expected contiguous demos {expected[:3]}...{expected[-3:]}, got {demos[:3]}...{demos[-3:]}")
        first = f["data"]["demo_0"]
        action_dim = int(first["actions"].shape[-1])
        if action_dim != args.expected_action_dim:
            raise AssertionError(f"Expected action dim {args.expected_action_dim}, got {action_dim}")
        for key in args.required_obs_key:
            if key not in first["obs"]:
                raise AssertionError(f"Missing obs key {key}")
            if key not in first["next_obs"]:
                raise AssertionError(f"Missing next_obs key {key}")
        print(f"ok {args.dataset}: demos={len(demos)} action_dim={action_dim} obs_keys={args.required_obs_key}")


if __name__ == "__main__":
    main()
