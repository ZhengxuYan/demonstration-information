#!/usr/bin/env python3
"""Repair expert200 DP datasets without re-rendering images.

This script fixes the generated expert200 policy-view HDF5 files in-place:
- backfills robomimic env metadata
- converts raw delta actions to robomimic absolute actions
- repairs robot0_gripper_qpos to the expected 2D shape
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from prepare_policy_view_datasets import (
    create_env,
    load_env_meta,
    select_gripper_joint_indexes,
    sorted_demo_keys,
    upgrade_controller_config,
    upsert_obs_pair,
)


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
        "--delta-env-meta-source",
        type=Path,
        default=Path("/iris/u/jasonyan/data/diffusion_policy/robomimic/datasets/square/ph/image.hdf5"),
    )
    parser.add_argument(
        "--output-env-meta-source",
        type=Path,
        default=Path("/iris/u/jasonyan/data/diffusion_policy/robomimic/datasets/square/ph/image_abs.hdf5"),
    )
    parser.add_argument("--render-height", type=int, default=84)
    parser.add_argument("--render-width", type=int, default=84)
    return parser.parse_args()


def load_env_meta(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        return json.loads(f["data"].attrs["env_args"])


def load_env_args(path: Path) -> str:
    with h5py.File(path, "r") as f:
        return f["data"].attrs["env_args"]


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


def controller_goal(robot):
    if hasattr(robot, "part_controllers"):
        controller = robot.part_controllers["right"]
    else:
        controller = robot.controller
    return (
        np.asarray(controller.goal_pos, dtype=np.float32),
        Rotation.from_matrix(np.asarray(controller.goal_ori)).as_rotvec().astype(np.float32),
    )


def convert_delta_actions_to_abs(env, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    d_a = len(env.env.robots[0].action_limits[0])
    stacked_actions = actions.reshape(*actions.shape[:-1], -1, d_a)
    action_goal_pos = np.zeros(stacked_actions.shape[:-1] + (3,), dtype=np.float32)
    action_goal_ori = np.zeros(stacked_actions.shape[:-1] + (3,), dtype=np.float32)
    action_remainder = stacked_actions[..., 6:].astype(np.float32)

    for i, state in enumerate(states):
        env.reset_to({"states": state})
        for robot_idx, robot in enumerate(env.env.robots):
            robot.control(stacked_actions[i, robot_idx], policy_step=True)
            action_goal_pos[i, robot_idx], action_goal_ori[i, robot_idx] = controller_goal(robot)

    stacked_abs_actions = np.concatenate([action_goal_pos, action_goal_ori, action_remainder], axis=-1)
    return stacked_abs_actions.reshape(actions.shape).astype(np.float32)


def repair_dataset(path: Path, qpos_env, action_env, env_args: str) -> None:
    with h5py.File(path, "r+") as f:
        f["data"].attrs["env_args"] = env_args
        for demo_key in tqdm(sorted_demo_keys(f["data"]), desc=f"repairing {path.name}"):
            demo = f["data"][demo_key]
            states = demo["states"][:]
            if "actions_delta" not in demo:
                demo.create_dataset("actions_delta", data=demo["actions"][:])
            delta_actions = demo["actions_delta"][:]
            abs_actions = convert_delta_actions_to_abs(action_env, states, delta_actions)
            demo["actions"][:] = abs_actions
            values = collect_gripper_qpos(qpos_env, states)
            upsert_obs_pair(demo, "robot0_gripper_qpos", values)


def main() -> None:
    args = parse_args()
    env_args = load_env_args(args.output_env_meta_source)
    qpos_env = create_env(json.loads(env_args), args.render_height, args.render_width, ["agentview"])
    qpos_env.reset()

    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils

    delta_meta = upgrade_controller_config(load_env_meta(args.delta_env_meta_source))
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": ["robot0_eef_pos"], "rgb": []}})
    action_env = EnvUtils.create_env_from_metadata(
        env_meta=delta_meta,
        render=False,
        render_offscreen=False,
        use_image_obs=False,
    )
    repair_dataset(args.agent_dataset, qpos_env, action_env, env_args)
    repair_dataset(args.left_dataset, qpos_env, action_env, env_args)


if __name__ == "__main__":
    main()
