"""Replay Robomimic HDF5 demo actions in the installed robosuite env.

This is a dataset / environment compatibility diagnostic. It answers whether
recorded open-loop demo actions still solve the task under the current local
robosuite stack after loading each demo's initial simulator state.
"""

import argparse
import json
import os
from copy import deepcopy

import h5py
import numpy as np
from robomimic.utils import env_utils

from openx.envs.robomimic import _sanitize_env_metadata_for_installed_robosuite


def _load_env_meta(dataset_path: str, use_image_obs: bool) -> dict:
    with h5py.File(os.path.expanduser(dataset_path), "r") as f:
        env_meta = json.loads(f["data"].attrs["env_args"])
    env_meta = _sanitize_env_metadata_for_installed_robosuite(env_meta)
    env_meta = deepcopy(env_meta)
    if use_image_obs:
        env_kwargs = env_meta["env_kwargs"]
        env_kwargs["use_camera_obs"] = True
        env_kwargs["camera_names"] = ["agentview", "robot0_eye_in_hand"]
        env_kwargs.setdefault("camera_heights", 84)
        env_kwargs.setdefault("camera_widths", 84)
    return env_meta


def _make_env(dataset_path: str, use_image_obs: bool):
    env_meta = _load_env_meta(dataset_path, use_image_obs=use_image_obs)
    env = env_utils.create_env_from_metadata(
        env_meta=env_meta,
        env_name=env_meta["env_name"],
        render=False,
        render_offscreen=use_image_obs,
        use_image_obs=use_image_obs,
    )
    env.env.ignore_done = False
    return env


def _success_from_env(env) -> bool:
    success = env.is_success()
    if isinstance(success, dict):
        if "task" in success:
            return bool(success["task"])
        return bool(any(success.values()))
    return bool(success)


def _demo_names(f, max_demos: int | None):
    demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]))
    if max_demos is not None:
        demos = demos[:max_demos]
    return demos


def replay_dataset(dataset_path: str, max_demos: int | None, horizon: int | None, use_image_obs: bool):
    env = _make_env(dataset_path, use_image_obs=use_image_obs)
    rows = []

    with h5py.File(os.path.expanduser(dataset_path), "r") as f:
        demos = _demo_names(f, max_demos=max_demos)
        for demo in demos:
            group = f["data"][demo]
            states = group["states"][:]
            actions = group["actions"][:]
            model_file = group.attrs.get("model_file", None)
            initial_state = {"states": states[0]}
            if model_file is not None:
                initial_state["model"] = model_file

            env.reset_to(initial_state)
            ep_reward = 0.0
            success = _success_from_env(env)
            steps = min(actions.shape[0], horizon) if horizon is not None else actions.shape[0]
            first_success_step = 0 if success else None

            for i in range(steps):
                _, reward, _, _ = env.step(actions[i])
                ep_reward += float(reward)
                step_success = _success_from_env(env)
                success = success or step_success
                if step_success and first_success_step is None:
                    first_success_step = i + 1

            rows.append(
                {
                    "demo": demo,
                    "len": int(actions.shape[0]),
                    "played_steps": int(steps),
                    "reward": ep_reward,
                    "success": bool(success),
                    "first_success_step": first_success_step,
                }
            )
            print(
                f"{demo}: success={int(success)} reward={ep_reward:.3f} "
                f"len={actions.shape[0]} played={steps} first_success_step={first_success_step}",
                flush=True,
            )

    success_rate = np.mean([row["success"] for row in rows]) if rows else np.nan
    mean_reward = np.mean([row["reward"] for row in rows]) if rows else np.nan
    print("\nsummary")
    print(f"dataset={dataset_path}")
    print(f"num_demos={len(rows)}")
    print(f"success_rate={success_rate:.4f}")
    print(f"mean_reward={mean_reward:.4f}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--max_demos", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--use_image_obs", action="store_true")
    args = parser.parse_args()
    replay_dataset(
        dataset_path=args.dataset,
        max_demos=args.max_demos,
        horizon=args.horizon,
        use_image_obs=args.use_image_obs,
    )


if __name__ == "__main__":
    main()
