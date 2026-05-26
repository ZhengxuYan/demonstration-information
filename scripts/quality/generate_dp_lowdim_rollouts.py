#!/usr/bin/env python3
"""Generate robomimic HDF5 rollouts from a Diffusion Policy lowdim checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import types
from collections import deque
from pathlib import Path

import dill
import h5py
import hydra
import numpy as np
import torch
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from omegaconf import OmegaConf
from tqdm import tqdm


OmegaConf.register_new_resolver("eval", eval, replace=True)


def install_lang_utils_stub() -> None:
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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-rollouts", type=int, default=200)
    parser.add_argument("--seed-start", type=int, default=100000)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_policy(checkpoint: Path, output_dir: Path, device: str):
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=str(output_dir))
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(torch.device(device))
    policy.eval()
    return cfg, policy


def create_env(dataset_path: str, obs_keys: list[str], abs_action: bool):
    install_lang_utils_stub()
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.obs_utils as ObsUtils

    ObsUtils.initialize_obs_modality_mapping_from_dict({"low_dim": obs_keys})
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    if abs_action:
        env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=False,
        use_image_obs=False,
    )
    return env, env_meta


def obs_vector(raw_obs: dict, obs_keys: list[str]) -> np.ndarray:
    return np.concatenate([raw_obs[key] for key in obs_keys], axis=0).astype(np.float32)


def undo_abs_action(action: np.ndarray) -> np.ndarray:
    transformer = RotationTransformer("axis_angle", "rotation_6d")
    raw_shape = action.shape
    if raw_shape[-1] == 20:
        action = action.reshape(-1, 2, 10)
    d_rot = action.shape[-1] - 4
    pos = action[..., :3]
    rot = action[..., 3 : 3 + d_rot]
    gripper = action[..., [-1]]
    rot = transformer.inverse(rot)
    out = np.concatenate([pos, rot, gripper], axis=-1)
    if raw_shape[-1] == 20:
        out = out.reshape(*raw_shape[:-1], 14)
    return out


def rollout_one(policy, env, cfg, seed: int, horizon: int) -> dict:
    obs_keys = list(cfg.task.obs_keys)
    n_obs_steps = int(cfg.n_obs_steps)
    n_latency_steps = int(cfg.n_latency_steps)
    abs_action = bool(cfg.task.abs_action)

    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset()
    raw_obs = env.get_observation()
    obs_hist = deque([obs_vector(raw_obs, obs_keys)] * n_obs_steps, maxlen=n_obs_steps)
    policy.reset()

    rows = {
        "states": [],
        "actions": [],
        "rewards": [],
        "dones": [],
        "obs": {key: [] for key in obs_keys},
    }
    success = False
    total_reward = 0.0

    while len(rows["actions"]) < horizon:
        np_obs = np.stack(list(obs_hist), axis=0)[None].astype(np.float32)
        obs_dict = {"obs": torch.from_numpy(np_obs).to(policy.device)}
        with torch.no_grad():
            action_dict = policy.predict_action(obs_dict)
        action_seq = dict_apply(action_dict, lambda x: x.detach().cpu().numpy())["action"][0]
        action_seq = action_seq[n_latency_steps:]
        if abs_action:
            env_action_seq = undo_abs_action(action_seq)
        else:
            env_action_seq = action_seq

        for action in env_action_seq:
            if len(rows["actions"]) >= horizon:
                break
            rows["states"].append(np.asarray(env.get_state()["states"]))
            for key in obs_keys:
                rows["obs"][key].append(np.asarray(raw_obs[key]))
            next_obs, reward, done, _ = env.step(action)
            rows["actions"].append(np.asarray(action, dtype=np.float32))
            rows["rewards"].append(float(reward))
            success = bool(env.is_success()["task"])
            rows["dones"].append(bool(done or success))
            total_reward += float(reward)
            raw_obs = next_obs
            obs_hist.append(obs_vector(raw_obs, obs_keys))
            if done or success:
                return {
                    **rows,
                    "success": success,
                    "return": total_reward,
                    "horizon": len(rows["actions"]),
                }

    return {
        **rows,
        "success": success,
        "return": total_reward,
        "horizon": len(rows["actions"]),
    }


def write_demo(data_group: h5py.Group, demo_key: str, result: dict, model_file: str | None) -> None:
    demo = data_group.create_group(demo_key)
    obs = demo.create_group("obs")
    for key, values in result["obs"].items():
        obs.create_dataset(key, data=np.asarray(values))
    demo.create_dataset("states", data=np.asarray(result["states"]))
    demo.create_dataset("actions", data=np.asarray(result["actions"]))
    demo.create_dataset("rewards", data=np.asarray(result["rewards"], dtype=np.float32))
    demo.create_dataset("dones", data=np.asarray(result["dones"], dtype=np.bool_))
    demo.attrs["num_samples"] = int(result["horizon"])
    demo.attrs["success"] = int(result["success"])
    demo.attrs["return"] = float(result["return"])
    demo.attrs["horizon"] = int(result["horizon"])
    if model_file is not None:
        demo.attrs["model_file"] = model_file


def main() -> None:
    args = parse_args()
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(args.output)
        args.output.unlink()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    cfg, policy = load_policy(args.checkpoint, args.output.parent / "_workspace", args.device)
    dataset_path = os.path.expanduser(str(cfg.task.dataset_path))
    env, env_meta = create_env(dataset_path, list(cfg.task.obs_keys), bool(cfg.task.abs_action))
    model_file = None

    total = 0
    successes = 0
    with h5py.File(args.output, "w") as f:
        f.attrs["checkpoint"] = str(args.checkpoint)
        f.attrs["source_dataset_path"] = dataset_path
        data = f.create_group("data")
        data.attrs["env_args"] = json.dumps(env_meta, indent=4)
        for rollout_idx in tqdm(range(args.n_rollouts), desc=args.checkpoint.name):
            seed = args.seed_start + rollout_idx
            result = rollout_one(policy, env, cfg, seed=seed, horizon=args.horizon)
            demo_key = f"demo_{rollout_idx}"
            write_demo(data, demo_key, result, model_file=model_file)
            total += int(result["horizon"])
            successes += int(result["success"])
            print(
                json.dumps(
                    {
                        "rollout": rollout_idx,
                        "seed": seed,
                        "success": int(result["success"]),
                        "return": float(result["return"]),
                        "horizon": int(result["horizon"]),
                    }
                ),
                flush=True,
            )
        data.attrs["num_demos"] = args.n_rollouts
        data.attrs["total"] = total
        f.create_group("mask")
        f["mask"].create_dataset(
            "train",
            data=np.asarray([f"demo_{i}".encode("utf-8") for i in range(args.n_rollouts)]),
        )
    print(args.output)
    print(f"num_demos={args.n_rollouts} successes={successes} success_rate={successes / args.n_rollouts:.3f}")


if __name__ == "__main__":
    main()
