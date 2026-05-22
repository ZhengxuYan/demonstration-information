#!/usr/bin/env python3
"""Evaluate robomimic BC checkpoints with rollout success sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import types
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch


LEFT_CLOSE_LOW_POS = np.asarray([0.42205740, -0.23999999, 1.15230719], dtype=np.float64)
LEFT_CLOSE_LOW_QUAT_WXYZ = np.asarray([0.81392215, 0.36066498, 0.18452251, 0.41641680], dtype=np.float64)


def install_lang_utils_stub() -> None:
    """Avoid loading optional CLIP / transformers code during env construction."""
    module_name = "robomimic.utils.lang_utils"
    if module_name in sys.modules:
        return
    stub = types.ModuleType(module_name)
    stub.LANG_EMB_OBS_KEY = "lang_emb"
    stub.get_lang_emb = lambda lang: None
    stub.get_lang_emb_shape = lambda: []
    sys.modules[module_name] = stub


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--policy", type=str, default="")
    parser.add_argument("--view", type=str, default="")
    parser.add_argument("--n-rollouts", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--left-close-low", action="store_true")
    parser.add_argument(
        "--left-close-low-as-agentview",
        action="store_true",
        help="Render left-close-low and store it in obs['agentview_image']; use for policies trained on left images saved under agentview_image.",
    )
    parser.add_argument("--image-height", type=int, default=84)
    parser.add_argument("--image-width", type=int, default=84)
    parser.add_argument("--device", type=str, default=None, help="torch device override, e.g. cuda:0 or cpu")
    return parser.parse_args()


def checkpoint_epoch(path: Path) -> int:
    match = re.search(r"model_epoch_(\d+)", path.name)
    if match is None:
        raise ValueError(f"Cannot parse epoch from checkpoint name: {path}")
    return int(match.group(1))


def checkpoint_label(path: Path) -> str:
    if "best_validation" in path.name or "valid_best" in path.name:
        return "best_validation"
    return f"epoch_{checkpoint_epoch(path)}"


def render_left_close_low_image(env, height: int, width: int) -> np.ndarray:
    raw_env = env.unwrapped if hasattr(env, "unwrapped") else env
    cam_id = raw_env.env.sim.model.camera_name2id("agentview")
    old_pos = np.array(raw_env.env.sim.model.cam_pos[cam_id], copy=True)
    old_quat = np.array(raw_env.env.sim.model.cam_quat[cam_id], copy=True)
    try:
        raw_env.env.sim.model.cam_pos[cam_id] = LEFT_CLOSE_LOW_POS
        raw_env.env.sim.model.cam_quat[cam_id] = LEFT_CLOSE_LOW_QUAT_WXYZ
        raw_env.env.sim.forward()
        return raw_env.render(mode="rgb_array", height=height, width=width, camera_name="agentview").copy()
    finally:
        raw_env.env.sim.model.cam_pos[cam_id] = old_pos
        raw_env.env.sim.model.cam_quat[cam_id] = old_quat
        raw_env.env.sim.forward()


def maybe_add_left_close_low(obs: dict, env, enabled: bool, height: int, width: int) -> dict:
    if not enabled:
        return obs
    obs = deepcopy(obs)
    obs["left_close_low_image"] = render_left_close_low_image(env, height=height, width=width)
    return obs


def maybe_replace_agentview_with_left_close_low(obs: dict, env, enabled: bool, height: int, width: int) -> dict:
    if not enabled:
        return obs
    obs = deepcopy(obs)
    obs["agentview_image"] = render_left_close_low_image(env, height=height, width=width)
    return obs


def prepare_obs(
    obs: dict,
    env,
    add_left_close_low: bool,
    left_close_low_as_agentview: bool,
    image_height: int,
    image_width: int,
) -> dict:
    obs = maybe_add_left_close_low(obs, env, add_left_close_low, image_height, image_width)
    obs = maybe_replace_agentview_with_left_close_low(
        obs, env, left_close_low_as_agentview, image_height, image_width
    )
    return obs


def rollout(
    policy,
    env,
    horizon: int,
    add_left_close_low: bool,
    left_close_low_as_agentview: bool,
    image_height: int,
    image_width: int,
) -> dict:
    policy.start_episode()
    obs = env.reset()
    state_dict = env.get_state()
    obs = env.reset_to(state_dict)
    obs = prepare_obs(obs, env, add_left_close_low, left_close_low_as_agentview, image_height, image_width)

    total_reward = 0.0
    success = False
    step_i = -1
    try:
        for step_i in range(horizon):
            action = policy(ob=obs)
            next_obs, reward, done, _ = env.step(action)
            next_obs = prepare_obs(
                next_obs, env, add_left_close_low, left_close_low_as_agentview, image_height, image_width
            )
            total_reward += reward
            success = bool(env.is_success()["task"])
            if done or success:
                break
            obs = next_obs
    except env.rollout_exceptions as exc:
        print(f"WARNING: rollout exception: {exc}", file=sys.stderr)

    return {
        "Return": float(total_reward),
        "Horizon": int(step_i + 1),
        "Success_Rate": float(success),
    }


def evaluate_checkpoint(args: argparse.Namespace, ckpt_path: Path) -> dict:
    install_lang_utils_stub()

    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.tensor_utils as TensorUtils
    import robomimic.utils.torch_utils as TorchUtils

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=str(ckpt_path), device=device, verbose=False)
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict,
        render=False,
        render_offscreen=True,
        verbose=False,
    )

    per_rollout = []
    for rollout_idx in range(args.n_rollouts):
        seed = int(args.seed + checkpoint_epoch(ckpt_path) * 1000 + rollout_idx)
        np.random.seed(seed)
        torch.manual_seed(seed)
        stats = rollout(
            policy=policy,
            env=env,
            horizon=args.horizon,
            add_left_close_low=args.left_close_low,
            left_close_low_as_agentview=args.left_close_low_as_agentview,
            image_height=args.image_height,
            image_width=args.image_width,
        )
        per_rollout.append(stats)
        print(
            json.dumps(
                {
                    "dataset": args.dataset,
                    "policy": args.policy,
                    "view": args.view,
                    "run_name": args.run_name,
                    "checkpoint_label": checkpoint_label(ckpt_path),
                    "epoch": checkpoint_epoch(ckpt_path),
                    "rollout": rollout_idx,
                    **stats,
                }
            ),
            flush=True,
        )

    rollout_stats = TensorUtils.list_of_flat_dict_to_dict_of_list(per_rollout)
    summary = {key: float(np.mean(rollout_stats[key])) for key in rollout_stats}
    summary["Dataset"] = args.dataset
    summary["Policy"] = args.policy
    summary["View"] = args.view
    summary["Run_Name"] = args.run_name
    summary["Checkpoint_Label"] = checkpoint_label(ckpt_path)
    summary["Epoch"] = checkpoint_epoch(ckpt_path)
    summary["Checkpoint"] = str(ckpt_path)
    summary["Num_Success"] = int(np.sum(rollout_stats["Success_Rate"]))
    summary["Num_Rollouts"] = int(args.n_rollouts)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / f"{args.run_name}_epoch_{summary['Epoch']:04d}_rollouts.json"
    with detail_path.open("w") as f:
        json.dump({"summary": summary, "rollouts": per_rollout}, f, indent=2)

    return summary


def main() -> None:
    args = parse_args()
    checkpoints = sorted(args.checkpoints, key=checkpoint_epoch)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for ckpt_path in checkpoints:
        print(f"evaluating {args.run_name} epoch {checkpoint_epoch(ckpt_path)}: {ckpt_path}", flush=True)
        summaries.append(evaluate_checkpoint(args, ckpt_path))

        csv_path = args.output_dir / f"{args.run_name}_summary.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "Dataset",
                    "Policy",
                    "View",
                    "Run_Name",
                    "Checkpoint_Label",
                    "Epoch",
                    "Checkpoint",
                    "Num_Rollouts",
                    "Num_Success",
                    "Success_Rate",
                    "Return",
                    "Horizon",
                ],
            )
            writer.writeheader()
            for row in summaries:
                writer.writerow(row)

    print(json.dumps({"output_dir": str(args.output_dir), "num_checkpoints": len(summaries)}, indent=2))


if __name__ == "__main__":
    main()
