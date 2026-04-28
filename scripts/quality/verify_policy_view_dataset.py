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
    parser.add_argument(
        "--expected-obs-shape",
        action="append",
        default=[],
        help="Expected trailing obs shape as KEY=DIM[,DIM...], for example robot0_gripper_qpos=2",
    )
    return parser.parse_args()


def parse_shape_specs(specs: list[str]) -> dict[str, tuple[int, ...]]:
    parsed = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Expected KEY=DIM[,DIM...] shape spec, got {spec!r}")
        key, shape_text = spec.split("=", 1)
        parsed[key] = tuple(int(part) for part in shape_text.split(",") if part)
    return parsed


def main() -> None:
    args = parse_args()
    expected_obs_shapes = parse_shape_specs(args.expected_obs_shape)
    with h5py.File(args.dataset, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda key: int(key.split("_")[-1]))
        expected = [f"demo_{i}" for i in range(args.expected_demos)]
        if demos != expected:
            raise AssertionError(f"Expected contiguous demos {expected[:3]}...{expected[-3:]}, got {demos[:3]}...{demos[-3:]}")
        action_dim = None
        for demo_key in demos:
            demo = f["data"][demo_key]
            horizon = int(demo["actions"].shape[0])
            this_action_dim = int(demo["actions"].shape[-1])
            if this_action_dim != args.expected_action_dim:
                raise AssertionError(f"{demo_key}: expected action dim {args.expected_action_dim}, got {this_action_dim}")
            action_dim = this_action_dim
            if "states" in demo and int(demo["states"].shape[0]) != horizon:
                raise AssertionError(f"{demo_key}: states length {demo['states'].shape[0]} != actions length {horizon}")
            for key in args.required_obs_key:
                if key not in demo["obs"]:
                    raise AssertionError(f"{demo_key}: missing obs key {key}")
                if key not in demo["next_obs"]:
                    raise AssertionError(f"{demo_key}: missing next_obs key {key}")
                if int(demo["obs"][key].shape[0]) != horizon:
                    raise AssertionError(f"{demo_key}: obs/{key} length {demo['obs'][key].shape[0]} != actions length {horizon}")
                if int(demo["next_obs"][key].shape[0]) != horizon:
                    raise AssertionError(
                        f"{demo_key}: next_obs/{key} length {demo['next_obs'][key].shape[0]} != actions length {horizon}"
                    )
                if key in expected_obs_shapes:
                    expected_shape = (horizon,) + expected_obs_shapes[key]
                    obs_shape = tuple(demo["obs"][key].shape)
                    next_obs_shape = tuple(demo["next_obs"][key].shape)
                    if obs_shape != expected_shape:
                        raise AssertionError(f"{demo_key}: obs/{key} shape {obs_shape} != expected {expected_shape}")
                    if next_obs_shape != expected_shape:
                        raise AssertionError(
                            f"{demo_key}: next_obs/{key} shape {next_obs_shape} != expected {expected_shape}"
                        )
        print(f"ok {args.dataset}: demos={len(demos)} action_dim={action_dim} obs_keys={args.required_obs_key}")


if __name__ == "__main__":
    main()
