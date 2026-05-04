#!/usr/bin/env python3
"""Export state-action VAE latents for a robomimic image.hdf5 dataset.

The output .npz is compatible with visualize_random_post_knn_entropy.py:
arrays are latent, ep_idx, step_idx, and demo_key.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import jax
import numpy as np

from openx.utils.evaluate import load_checkpoint
from score_robomimic_hdf5 import IMAGE_KEY_TO_HDF5_DATASET, concatenate_ordered, normalize_tree, stats_subtree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained sa VAE checkpoint directory.")
    parser.add_argument("--dataset", type=Path, required=True, help="Robomimic image.hdf5 dataset.")
    parser.add_argument("--output", type=Path, required=True, help="Output .npz path.")
    parser.add_argument("--filter-key", type=str, default=None, help="Optional HDF5 mask key, e.g. train or valid.")
    parser.add_argument("--camera", choices=["both"], default="both", help="Currently exports agent+wrist fused latents.")
    parser.add_argument("--batch-size", type=int, default=1024)
    return parser.parse_args()


def mask_demos(hdf5_file: h5py.File, filter_key: str | None) -> set[str] | None:
    if filter_key is None:
        return None
    path = f"mask/{filter_key}"
    if path not in hdf5_file:
        raise KeyError(f"{hdf5_file.filename} has no {path}")
    demos = hdf5_file[path][:]
    return {x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in demos}


def load_episodes(
    dataset: Path,
    obs_structure: dict,
    action_structure: dict,
    stats: dict,
    image_keys: list[str],
    filter_key: str | None,
) -> list[dict]:
    episodes = []
    with h5py.File(dataset, "r") as f:
        keep = mask_demos(f, filter_key)
        demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]))
        for demo in demos:
            if keep is not None and demo not in keep:
                continue
            grp = f["data"][demo]
            obs_grp = grp["obs"]
            for image_key in image_keys:
                hdf5_key = IMAGE_KEY_TO_HDF5_DATASET[image_key]
                if hdf5_key not in obs_grp:
                    raise KeyError(
                        f"{dataset}:{demo}/obs is missing {hdf5_key}; available keys: {sorted(obs_grp.keys())}"
                    )

            raw_obs = {
                "state": {
                    "EE_POS": obs_grp["robot0_eef_pos"][:].astype(np.float32),
                    "EE_QUAT": obs_grp["robot0_eef_quat"][:].astype(np.float32),
                    "GRIPPER": obs_grp["robot0_gripper_qpos"][:, :1].astype(np.float32),
                },
                "image": {
                    image_key: obs_grp[IMAGE_KEY_TO_HDF5_DATASET[image_key]][:].astype(np.float32) / 255.0
                    for image_key in image_keys
                },
            }
            raw_action = {
                "desired_delta": {
                    "EE_POS": grp["actions"][:, :3].astype(np.float32),
                    "EE_EULER": grp["actions"][:, 3:6].astype(np.float32),
                },
                "desired_absolute": {
                    "GRIPPER": grp["actions"][:, -1:].astype(np.float32),
                },
            }
            normalized_obs = {
                "state": normalize_tree(raw_obs["state"], obs_structure["state"], stats_subtree(stats, "state")),
                "image": raw_obs["image"],
            }
            normalized_action = normalize_tree(raw_action, action_structure, stats_subtree(stats, "action"))
            action = concatenate_ordered(normalized_action)
            ep_len = action.shape[0]
            step_idx = np.arange(ep_len, dtype=np.int32)[:-1]
            episodes.append(
                {
                    "observation": {
                        "state": concatenate_ordered(normalized_obs["state"])[:-1, None, :],
                        "image": {
                            image_key: normalized_obs["image"][image_key][:-1, None, ...]
                            for image_key in image_keys
                        },
                    },
                    "action": action[:-1, None, :],
                    "mask": np.ones((ep_len - 1, 1), dtype=bool),
                    "ep_idx": np.full(ep_len - 1, int(demo.split("_")[-1]), dtype=np.int32),
                    "quality_score": np.zeros(ep_len - 1, dtype=np.float32),
                    "dataset_id": np.zeros(ep_len - 1, dtype=np.int32),
                    "step_idx": step_idx,
                    "demo_key": np.full(ep_len - 1, demo, dtype=object),
                }
            )
    return episodes


def stack_latent_batches(episodes: list[dict], batch_size: int, image_keys: list[str]):
    merged = {
        "observation": {
            "state": np.concatenate([ep["observation"]["state"] for ep in episodes], axis=0),
            "image": {
                image_key: np.concatenate([ep["observation"]["image"][image_key] for ep in episodes], axis=0)
                for image_key in image_keys
            },
        },
        "action": np.concatenate([ep["action"] for ep in episodes], axis=0),
        "mask": np.concatenate([ep["mask"] for ep in episodes], axis=0),
        "ep_idx": np.concatenate([ep["ep_idx"] for ep in episodes], axis=0),
        "step_idx": np.concatenate([ep["step_idx"] for ep in episodes], axis=0),
        "demo_key": np.concatenate([ep["demo_key"] for ep in episodes], axis=0),
    }

    total = merged["action"].shape[0]
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        yield {
            "observation": {
                "state": merged["observation"]["state"][start:end],
                "image": {
                    image_key: merged["observation"]["image"][image_key][start:end] for image_key in image_keys
                },
            },
            "action": merged["action"][start:end],
            "mask": merged["mask"][start:end],
            "ep_idx": merged["ep_idx"][start:end],
            "step_idx": merged["step_idx"][start:end],
            "demo_key": merged["demo_key"][start:end],
        }


def main() -> None:
    args = parse_args()
    alg, state, dataset_statistics, config = load_checkpoint(str(args.checkpoint))

    obs_structure = config.structure["observation"].to_dict()
    action_structure = config.structure["action"].to_dict()
    image_keys = sorted(obs_structure.get("image", {}).keys())
    if image_keys != ["agent", "wrist"]:
        raise ValueError(f"Expected checkpoint with both agent+wrist image keys; got {image_keys}")

    episodes = load_episodes(
        dataset=args.dataset,
        obs_structure=obs_structure,
        action_structure=action_structure,
        stats=dataset_statistics,
        image_keys=image_keys,
        filter_key=args.filter_key,
    )
    if not episodes:
        raise ValueError(f"No episodes loaded from {args.dataset} with filter_key={args.filter_key}")

    predict = jax.jit(lambda batch, rng: alg.predict(state, batch, rng))
    latents = []
    ep_idxs = []
    step_idxs = []
    demo_keys = []
    base_rng = jax.random.key(0)
    for batch_idx, batch in enumerate(stack_latent_batches(episodes, args.batch_size, image_keys)):
        rng = jax.random.fold_in(base_rng, batch_idx)
        model_batch = {
            "observation": {
                "state": jax.device_put(batch["observation"]["state"]),
                "image": {key: jax.device_put(value) for key, value in batch["observation"]["image"].items()},
            },
            "action": jax.device_put(batch["action"]),
            "mask": jax.device_put(batch["mask"]),
        }
        latents.append(np.asarray(predict(model_batch, rng), dtype=np.float32))
        ep_idxs.append(np.asarray(batch["ep_idx"], dtype=np.int64))
        step_idxs.append(np.asarray(batch["step_idx"], dtype=np.int64))
        demo_keys.append(np.asarray(batch["demo_key"], dtype=object))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        latent=np.concatenate(latents, axis=0),
        ep_idx=np.concatenate(ep_idxs, axis=0),
        step_idx=np.concatenate(step_idxs, axis=0),
        demo_key=np.concatenate(demo_keys, axis=0),
    )
    print(args.output)


if __name__ == "__main__":
    main()
