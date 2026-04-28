#!/usr/bin/env python3
"""Repair expert200 gripper qpos observations without re-rendering images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from prepare_policy_view_datasets import create_env, select_gripper_joint_indexes, sorted_demo_keys, upsert_obs_pair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent-dataset",
        type=Path,
        default=Path("/iris/u/jasonyan/data/policy_view_experiments/expert200/expert200_agent_wrist_image_abs.hdf5"),
    )
    parser.add_argument(
        "--left-dataset",
        type=Path,
        default=Path(
            "/iris/u/jasonyan/data/policy_view_experiments/expert200/expert200_left_close_low_wrist_image_abs.hdf5"
        ),
    )
    parser.add_argument(
        "--env-meta-source",
        type=Path,
        default=Path("/iris/u/jasonyan/data/diffusion_policy/robomimic/datasets/square/ph/image_abs.hdf5"),
    )
    parser.add_argument("--render-height", type=int, default=84)
    parser.add_argument("--render-width", type=int, default=84)
    return parser.parse_args()


def load_env_meta(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        return json.loads(f["data"].attrs["env_args"])


def collect_gripper_qpos(env, states: np.ndarray) -> np.ndarray:
    robot = env.env.robots[0]
    arm = robot.arms[0] if getattr(robot, "arms", None) else "right"
    indexes = select_gripper_joint_indexes(robot._ref_gripper_joint_pos_indexes, arm)
    values = []
    for state in states:
        env.env.sim.set_state_from_flattened(state)
        env.env.sim.forward()
        values.append([env.env.sim.data.qpos[index] for index in indexes])
    return np.asarray(values, dtype=np.float32)


def repair_dataset(path: Path, env) -> None:
    with h5py.File(path, "r+") as f:
        for demo_key in tqdm(sorted_demo_keys(f["data"]), desc=f"repairing {path.name}"):
            demo = f["data"][demo_key]
            values = collect_gripper_qpos(env, demo["states"][:])
            upsert_obs_pair(demo, "robot0_gripper_qpos", values)


def main() -> None:
    args = parse_args()
    env = create_env(load_env_meta(args.env_meta_source), args.render_height, args.render_width, ["agentview"])
    env.reset()
    repair_dataset(args.agent_dataset, env)
    repair_dataset(args.left_dataset, env)


if __name__ == "__main__":
    main()
