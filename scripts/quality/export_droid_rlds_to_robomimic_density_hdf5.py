#!/usr/bin/env python3
"""Export a DROID RLDS dataset to robomimic HDF5 for action-density scoring.

The exported file contains all observation keys needed for the conditioning
sweep:

* agentview_image: third-person image
* robot0_eye_in_hand_image: wrist image
* robot_state: low-dimensional robot state
* action_prior_dummy: a constant feature used for unconditional p(a)

The action target can be either a single action or a flattened future action
chunk. Actions can optionally be normalized in the exported HDF5 so the same
robomimic likelihood code can score Gaussian, GMM, and discrete density heads.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import tensorflow_datasets as tfds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rlds-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-target", choices=["single", "chunk"], default="single")
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument(
        "--action-normalization",
        choices=["none", "zscore", "minmax", "bounded_minmax", "percentile_minmax"],
        default="none",
    )
    parser.add_argument("--action-bound-low-percentile", type=float, default=1.0)
    parser.add_argument("--action-bound-high-percentile", type=float, default=99.0)
    parser.add_argument("--env-name", default="droid_density")
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def as_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return as_text(value.item())
    return str(value)


def step_array(steps: dict[str, Any], path: tuple[str, ...]) -> np.ndarray:
    node: Any = steps
    for key in path:
        node = node[key]
    return np.asarray(node)


def stack_step_values(values: list[Any]) -> Any:
    first = values[0]
    if isinstance(first, dict):
        return {key: stack_step_values([value[key] for value in values]) for key in first}
    return np.asarray(values)


def materialize_steps(steps: Any) -> dict[str, Any]:
    if isinstance(steps, dict):
        return steps
    rows = list(steps)
    if not rows:
        raise ValueError("Episode has no steps")
    return stack_step_values(rows)


def build_robot_state(steps: dict[str, Any]) -> np.ndarray:
    obs = steps["observation"]
    parts = [
        np.asarray(obs["cartesian_position"], dtype=np.float32),
        np.asarray(obs["gripper_position"], dtype=np.float32),
        np.asarray(obs["joint_position"], dtype=np.float32),
    ]
    return np.concatenate(parts, axis=-1)


def build_action_targets(actions: np.ndarray, target: str, chunk_size: int) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if target == "single":
        return actions
    if chunk_size <= 0:
        raise ValueError(f"--chunk-size must be positive, got {chunk_size}")
    chunks = []
    for i in range(len(actions)):
        idx = np.clip(np.arange(i, i + chunk_size), 0, len(actions) - 1)
        chunks.append(actions[idx].reshape(-1))
    return np.asarray(chunks, dtype=np.float32)


def normalization_stats(actions: np.ndarray, mode: str, low_percentile: float, high_percentile: float) -> dict[str, np.ndarray]:
    if mode == "none":
        return {}
    if mode == "zscore":
        mean = actions.mean(axis=0)
        std = actions.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}
    if mode == "minmax":
        amin = actions.min(axis=0)
        amax = actions.max(axis=0)
        scale = np.where((amax - amin) < 1e-6, 1.0, amax - amin)
        return {"min": amin.astype(np.float32), "max": amax.astype(np.float32), "scale": scale.astype(np.float32)}
    if mode in ("bounded_minmax", "percentile_minmax"):
        if not 0.0 <= low_percentile < high_percentile <= 100.0:
            raise ValueError(
                "Expected 0 <= --action-bound-low-percentile < "
                "--action-bound-high-percentile <= 100"
            )
        low = np.percentile(actions, low_percentile, axis=0)
        high = np.percentile(actions, high_percentile, axis=0)
        scale = np.where((high - low) < 1e-6, 1.0, high - low)
        return {
            "low": low.astype(np.float32),
            "high": high.astype(np.float32),
            "scale": scale.astype(np.float32),
        }
    raise ValueError(mode)


def apply_normalization(actions: np.ndarray, mode: str, stats: dict[str, np.ndarray]) -> np.ndarray:
    if mode == "none":
        return actions.astype(np.float32)
    if mode == "zscore":
        return ((actions - stats["mean"]) / stats["std"]).astype(np.float32)
    if mode == "minmax":
        return (2.0 * (actions - stats["min"]) / stats["scale"] - 1.0).astype(np.float32)
    if mode in ("bounded_minmax", "percentile_minmax"):
        clipped = np.clip(actions, stats["low"], stats["high"])
        return (2.0 * (clipped - stats["low"]) / stats["scale"] - 1.0).astype(np.float32)
    raise ValueError(mode)


def action_clip_stats(actions: np.ndarray, mode: str, stats: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if mode not in ("bounded_minmax", "percentile_minmax"):
        zeros = np.zeros(actions.shape[-1], dtype=np.int64)
        return {"low_count": zeros, "high_count": zeros, "clipped_count": zeros, "clipped_fraction": zeros.astype(np.float32)}
    low_count = np.sum(actions < stats["low"], axis=0).astype(np.int64)
    high_count = np.sum(actions > stats["high"], axis=0).astype(np.int64)
    clipped_count = low_count + high_count
    clipped_fraction = (clipped_count / max(len(actions), 1)).astype(np.float32)
    return {
        "low_count": low_count,
        "high_count": high_count,
        "clipped_count": clipped_count,
        "clipped_fraction": clipped_fraction,
    }


def sorted_episode_rows(builder) -> list[dict[str, Any]]:
    ds = builder.as_dataset(split="train", shuffle_files=False)
    rows = list(tfds.as_numpy(ds))
    rows.sort(key=lambda ep: int(ep["episode_metadata"]["ep_idx"]))
    return rows


def write_dataset(args: argparse.Namespace) -> None:
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite to replace it")
    if not 0.0 < args.valid_ratio < 1.0:
        raise ValueError("--valid-ratio must be in (0, 1)")

    builder = tfds.builder_from_directory(builder_dir=str(args.rlds_path))
    episodes = sorted_episode_rows(builder)
    if len(episodes) < 2:
        raise ValueError(f"Need at least two episodes, got {len(episodes)}")

    raw_action_targets = []
    parsed = []
    for ep in episodes:
        steps = materialize_steps(ep["steps"])
        ep_idx = int(ep["episode_metadata"]["ep_idx"])
        episode_name = Path(as_text(ep["episode_metadata"]["file_path"])).parent.name
        actions = build_action_targets(np.asarray(steps["action"], dtype=np.float32), args.action_target, args.chunk_size)
        row = {
            "ep_idx": ep_idx,
            "episode": episode_name,
            "agentview_image": np.asarray(steps["observation"]["exterior_image_1_left"], dtype=np.uint8),
            "robot0_eye_in_hand_image": np.asarray(steps["observation"]["wrist_image_left"], dtype=np.uint8),
            "robot_state": build_robot_state(steps),
            "actions": actions,
        }
        lengths = {key: len(value) for key, value in row.items() if isinstance(value, np.ndarray)}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Length mismatch for ep_idx={ep_idx}: {lengths}")
        parsed.append(row)
        raw_action_targets.append(actions)

    rng = np.random.default_rng(args.seed)
    valid_count = max(1, int(round(args.valid_ratio * len(parsed))))
    valid_count = min(valid_count, len(parsed) - 1)
    valid_indices = set(rng.choice(np.arange(len(parsed)), size=valid_count, replace=False).astype(int).tolist())
    train_actions = np.concatenate([row["actions"] for idx, row in enumerate(parsed) if idx not in valid_indices], axis=0)
    all_actions = np.concatenate(raw_action_targets, axis=0)
    stats = normalization_stats(
        train_actions,
        args.action_normalization,
        args.action_bound_low_percentile,
        args.action_bound_high_percentile,
    )
    clip_stats = action_clip_stats(all_actions, args.action_normalization, stats)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()

    with h5py.File(args.output, "w") as f:
        data = f.create_group("data")
        env_args = json.dumps({"env_name": args.env_name, "type": 2, "env_kwargs": {}})
        data.attrs["env_args"] = env_args
        train_keys = []
        valid_keys = []
        total_samples = 0
        for out_idx, row in enumerate(parsed):
            demo_key = f"demo_{out_idx}"
            if out_idx in valid_indices:
                valid_keys.append(demo_key)
            else:
                train_keys.append(demo_key)

            grp = data.create_group(demo_key)
            obs = grp.create_group("obs")
            obs.create_dataset("agentview_image", data=row["agentview_image"], compression="gzip", compression_opts=1)
            obs.create_dataset(
                "robot0_eye_in_hand_image",
                data=row["robot0_eye_in_hand_image"],
                compression="gzip",
                compression_opts=1,
            )
            obs.create_dataset("robot_state", data=row["robot_state"].astype(np.float32))
            obs.create_dataset("action_prior_dummy", data=np.zeros((len(row["actions"]), 1), dtype=np.float32))
            normalized_actions = apply_normalization(row["actions"], args.action_normalization, stats)
            if not np.all(np.isfinite(normalized_actions)):
                raise ValueError(f"Non-finite normalized actions for ep_idx={row['ep_idx']}")
            if args.action_normalization in ("bounded_minmax", "percentile_minmax"):
                tolerance = 1e-5
                if normalized_actions.min() < -1.0 - tolerance or normalized_actions.max() > 1.0 + tolerance:
                    raise ValueError(
                        "Percentile-bounded minmax produced actions outside [-1, 1]: "
                        f"min={normalized_actions.min()} max={normalized_actions.max()}"
                    )
            grp.create_dataset("actions", data=normalized_actions)
            grp.create_dataset("actions_raw", data=row["actions"].astype(np.float32))
            grp.attrs["num_samples"] = int(len(row["actions"]))
            grp.attrs["ep_idx"] = int(row["ep_idx"])
            grp.attrs["episode"] = str(row["episode"])
            total_samples += int(len(row["actions"]))

        mask = f.create_group("mask")
        mask.create_dataset("train", data=np.asarray([x.encode("utf-8") for x in train_keys], dtype="S"))
        mask.create_dataset("valid", data=np.asarray([x.encode("utf-8") for x in valid_keys], dtype="S"))

        f.attrs["total"] = int(total_samples)
        f.attrs["env_args"] = env_args
        f.attrs["rlds_path"] = str(args.rlds_path)
        f.attrs["action_target"] = args.action_target
        f.attrs["chunk_size"] = int(args.chunk_size)
        f.attrs["action_normalization"] = args.action_normalization
        f.attrs["action_bound_low_percentile"] = float(args.action_bound_low_percentile)
        f.attrs["action_bound_high_percentile"] = float(args.action_bound_high_percentile)
        f.attrs["valid_ratio"] = float(args.valid_ratio)
        f.attrs["split_seed"] = int(args.seed)
        for key, value in stats.items():
            f.attrs[f"action_norm_{key}"] = value
        for key, value in clip_stats.items():
            f.attrs[f"action_norm_{key}"] = value

    print(f"wrote {args.output}")
    print(f"episodes={len(parsed)} train={len(train_keys)} valid={len(valid_keys)} transitions={total_samples}")
    print(f"action_dim={all_actions.shape[-1]} action_target={args.action_target} normalization={args.action_normalization}")


def main() -> None:
    write_dataset(parse_args())


if __name__ == "__main__":
    main()
