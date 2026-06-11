#!/usr/bin/env python3
"""Validate a bimanual RoboMimic transport image.hdf5 for DemInf."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


REQUIRED_OBS_KEYS = (
    "shouldercamera0_image",
    "shouldercamera1_image",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "robot0_joint_pos",
    "robot0_joint_vel",
    "robot1_eef_pos",
    "robot1_eef_quat",
    "robot1_gripper_qpos",
    "robot1_joint_pos",
    "robot1_joint_vel",
    "object",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", required=True, help="Path to transport image.hdf5.")
    parser.add_argument("--expected-demos", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    path = Path(args.hdf5)
    if not path.exists():
        raise FileNotFoundError(path)

    lengths = []
    with h5py.File(path, "r") as f:
        if "data" not in f:
            raise KeyError("Missing /data group")
        demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]))
        if args.expected_demos is not None and len(demos) != args.expected_demos:
            raise ValueError(f"Expected {args.expected_demos} demos, found {len(demos)}")
        if not demos:
            raise ValueError("No demos found")
        for demo in demos:
            group = f["data"][demo]
            if "actions" not in group:
                raise KeyError(f"{demo} missing actions")
            actions = group["actions"]
            if actions.shape[-1] != 14:
                raise ValueError(f"{demo} action dim expected 14, got {actions.shape[-1]}")
            if "obs" not in group or "next_obs" not in group:
                raise KeyError(f"{demo} missing obs or next_obs")
            obs = group["obs"]
            next_obs = group["next_obs"]
            for key in REQUIRED_OBS_KEYS:
                if key not in obs:
                    raise KeyError(f"{demo}/obs missing {key}; available={sorted(obs.keys())}")
                if key not in next_obs:
                    raise KeyError(f"{demo}/next_obs missing {key}; available={sorted(next_obs.keys())}")
                if obs[key].shape[0] != actions.shape[0]:
                    raise ValueError(f"{demo}/obs/{key} length mismatch: {obs[key].shape[0]} vs {actions.shape[0]}")
            if obs["shouldercamera0_image"].shape[1:] != (84, 84, 3):
                raise ValueError(f"{demo} shouldercamera0_image shape={obs['shouldercamera0_image'].shape}")
            if obs["shouldercamera1_image"].shape[1:] != (84, 84, 3):
                raise ValueError(f"{demo} shouldercamera1_image shape={obs['shouldercamera1_image'].shape}")
            state_dim = 0
            for robot in (0, 1):
                state_dim += obs[f"robot{robot}_eef_pos"].shape[-1]
                state_dim += obs[f"robot{robot}_eef_quat"].shape[-1]
                state_dim += obs[f"robot{robot}_gripper_qpos"].shape[-1]
                state_dim += obs[f"robot{robot}_joint_pos"].shape[-1]
                state_dim += obs[f"robot{robot}_joint_vel"].shape[-1]
            state_dim += obs["object"].shape[-1]
            if state_dim != 87:
                raise ValueError(f"{demo} transport state dim expected 87, got {state_dim}")
            lengths.append(actions.shape[0])

    arr = np.asarray(lengths)
    print(f"hdf5={path}")
    print(f"demos={len(lengths)}")
    print(f"length_min={arr.min()} length_max={arr.max()} length_mean={arr.mean():.1f}")
    print("ROBOMIMIC_TRANSPORT_HDF5_OK")


if __name__ == "__main__":
    main()
