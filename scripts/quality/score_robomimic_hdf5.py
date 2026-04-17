"""
Score robomimic image.hdf5 files directly with existing DemInf checkpoints.

This bypasses the TFDS/RLDS conversion step and is intended for small custom
datasets exported with robomimic's dataset_states_to_obs.py script.

Example:

python scripts/quality/score_robomimic_hdf5.py \
    --obs_ckpt /iris/u/jasonyan/data/deminf_outputs/robomimic_image/square_mh_wrist_obs_vae_seed1 \
    --action_ckpt /iris/u/jasonyan/data/deminf_outputs/robomimic_image/square_mh_action_vae_seed1 \
    --dataset forward_grab=1=/iris/u/jasonyan/data/fb_demos/forward_grab_image.hdf5 \
    --dataset backward_grab=0=/iris/u/jasonyan/data/fb_demos/backward_grab_image.hdf5 \
    --batch_size 1024 \
    --output /iris/u/jasonyan/data/deminf_outputs/fb_demos_scores
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import dataclass
from typing import Dict, Iterable

import h5py
import jax
import numpy as np
import tensorflow as tf
from jax import numpy as jnp
from jax.scipy.special import digamma
from matplotlib import pyplot as plt
from scipy import stats

from openx.data.utils import NormalizationType
from openx.utils.evaluate import load_checkpoint


CAMERA_DATASETS = {
    "wrist": "robot0_eye_in_hand_image",
    "agent": "agentview_image",
}
IMAGE_KEY_TO_HDF5_DATASET = {
    "wrist": "robot0_eye_in_hand_image",
    "agent": "agentview_image",
}


def _l2_dists(z):
    return jnp.linalg.norm(z[:, None, :] - z[None, :, :], axis=-1)


def ksg_estimator(batch, rng, ks, obs_alg, obs_state, action_alg, action_state):
    obs_rng, action_rng = jax.random.split(rng)
    z_obs = obs_alg.predict(obs_state, batch, obs_rng)
    z_action = action_alg.predict(action_state, batch, action_rng)

    obs_dist = _l2_dists(z_obs)
    action_dist = _l2_dists(z_action)

    joint_dist = jnp.maximum(obs_dist, action_dist)
    joint_knn_dists = jnp.sort(joint_dist, axis=-1)[:, ks]

    obs_count = jnp.sum(obs_dist[:, :, None] < joint_knn_dists[:, None, :], axis=1)
    action_count = jnp.sum(action_dist[:, :, None] < joint_knn_dists[:, None, :], axis=1)

    return -jnp.mean(digamma(obs_count) + digamma(action_count), axis=-1)


def _distance_diagnostic_stats(obs_dist, action_dist, joint_dist, ks):
    """Summarize which marginal dominates the L-infinity joint distance."""
    batch_size = obs_dist.shape[0]
    off_diagonal = ~jnp.eye(batch_size, dtype=bool)

    all_obs = obs_dist[off_diagonal]
    all_action = action_dist[off_diagonal]
    all_joint = joint_dist[off_diagonal]

    joint_order = jnp.argsort(joint_dist, axis=-1)[:, ks]
    knn_obs = jnp.take_along_axis(obs_dist, joint_order, axis=1).reshape(-1)
    knn_action = jnp.take_along_axis(action_dist, joint_order, axis=1).reshape(-1)
    knn_joint = jnp.take_along_axis(joint_dist, joint_order, axis=1).reshape(-1)

    def summarize(prefix, obs_values, action_values, joint_values):
        return {
            f"{prefix}_count": jnp.asarray(obs_values.size, dtype=jnp.float32),
            f"{prefix}_action_gt_obs_count": jnp.sum(action_values > obs_values, dtype=jnp.float32),
            f"{prefix}_obs_gt_action_count": jnp.sum(obs_values > action_values, dtype=jnp.float32),
            f"{prefix}_equal_count": jnp.sum(action_values == obs_values, dtype=jnp.float32),
            f"{prefix}_obs_l2_sum": jnp.sum(obs_values),
            f"{prefix}_action_l2_sum": jnp.sum(action_values),
            f"{prefix}_joint_linf_sum": jnp.sum(joint_values),
            f"{prefix}_joint_linf_max": jnp.max(joint_values),
            f"{prefix}_action_minus_obs_sum": jnp.sum(action_values - obs_values),
        }

    return {
        **summarize("all_pairs", all_obs, all_action, all_joint),
        **summarize("joint_knn", knn_obs, knn_action, knn_joint),
    }


def ksg_estimator_with_distance_diagnostics(batch, rng, ks, obs_alg, obs_state, action_alg, action_state):
    obs_rng, action_rng = jax.random.split(rng)
    z_obs = obs_alg.predict(obs_state, batch, obs_rng)
    z_action = action_alg.predict(action_state, batch, action_rng)

    obs_dist = _l2_dists(z_obs)
    action_dist = _l2_dists(z_action)

    joint_dist = jnp.maximum(obs_dist, action_dist)
    joint_knn_dists = jnp.sort(joint_dist, axis=-1)[:, ks]

    obs_count = jnp.sum(obs_dist[:, :, None] < joint_knn_dists[:, None, :], axis=1)
    action_count = jnp.sum(action_dist[:, :, None] < joint_knn_dists[:, None, :], axis=1)
    score = -jnp.mean(digamma(obs_count) + digamma(action_count), axis=-1)

    return score, _distance_diagnostic_stats(obs_dist, action_dist, joint_dist, ks)


@dataclass
class DatasetSpec:
    name: str
    label: float
    path: str


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obs_ckpt", required=True, help="Observation VAE checkpoint directory.")
    parser.add_argument("--action_ckpt", required=True, help="Action VAE checkpoint directory.")
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Dataset spec in the form name=label=/path/to/image.hdf5. Repeat for multiple datasets.",
    )
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--output", required=True, help="Output directory for pickle and plots.")
    parser.add_argument(
        "--camera",
        default=None,
        choices=sorted(CAMERA_DATASETS) + ["both"],
        help=(
            "Camera stream to load from HDF5. Use 'both' for fused agent+wrist checkpoints. "
            "Defaults to the image keys in the observation "
            "checkpoint config, if present."
        ),
    )
    parser.add_argument(
        "--shared-normalization",
        action="store_true",
        help="Normalize all provided datasets together instead of each dataset independently.",
    )
    parser.add_argument(
        "--reference-score-pkl",
        default=None,
        help="Optional pickle with sample_score used only to define the clipping and normalization scale.",
    )
    parser.add_argument(
        "--distance-diagnostics",
        action="store_true",
        help=(
            "Save diagnostics for the L-infinity joint distance used by KSG, including how often "
            "action-latent L2 distance is larger than observation-latent L2 distance."
        ),
    )
    return parser.parse_args()


def parse_dataset_spec(spec: str) -> DatasetSpec:
    parts = spec.split("=", 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid --dataset spec: {spec}. Expected name=label=/path/to/image.hdf5")
    name, label, path = parts
    return DatasetSpec(name=name, label=float(label), path=path)


def _normalize(x, mode: str, mean, std, low, high):
    if mode == NormalizationType.NONE:
        return x
    if mode == NormalizationType.GAUSSIAN:
        return np.divide(x - mean, std, out=np.zeros_like(x), where=std != 0)
    if mode == NormalizationType.BOUNDS:
        x = np.clip(x, low, high)
        denom = high - low
        return 2 * np.divide(x - low, denom, out=np.zeros_like(x), where=denom != 0) - 1
    if mode == NormalizationType.BOUNDS_5STDV:
        low = np.maximum(low, mean - 5 * std)
        high = np.minimum(high, mean + 5 * std)
        x = np.clip(x, low, high)
        denom = high - low
        return 2 * np.divide(x - low, denom, out=np.zeros_like(x), where=denom != 0) - 1
    raise ValueError(f"Invalid normalization mode: {mode}")


def normalize_tree(tree: Dict, structure: Dict, stats: Dict) -> Dict:
    def _child_stats(current_stats: Dict, current_key: str) -> Dict:
        if all(name in current_stats for name in ("mean", "std", "min", "max")):
            return {name: current_stats[name][current_key] for name in ("mean", "std", "min", "max")}
        if current_key in current_stats:
            return current_stats[current_key]
        raise KeyError(f"Could not find stats for key {current_key}")

    def _leaf_stat(current_stats: Dict, stat_name: str, current_key: str):
        if stat_name in current_stats:
            return current_stats[stat_name][current_key]
        if current_key in current_stats and stat_name in current_stats[current_key]:
            return current_stats[current_key][stat_name]
        raise KeyError(f"Could not find {stat_name} stats for key {current_key}")

    out = {}
    for key, substructure in structure.items():
        if isinstance(substructure, dict):
            out[key] = normalize_tree(tree[key], substructure, _child_stats(stats, key))
        else:
            out[key] = _normalize(
                np.asarray(tree[key], dtype=np.float32),
                str(substructure),
                np.asarray(_leaf_stat(stats, "mean", key), dtype=np.float32),
                np.asarray(_leaf_stat(stats, "std", key), dtype=np.float32),
                np.asarray(_leaf_stat(stats, "min", key), dtype=np.float32),
                np.asarray(_leaf_stat(stats, "max", key), dtype=np.float32),
            )
    return out


def stats_subtree(stats: Dict, key: str) -> Dict:
    return {name: value[key] for name, value in stats.items() if name in ("mean", "std", "min", "max")}


def concatenate_ordered(tree: Dict) -> np.ndarray:
    flat = []
    for value in tree.values():
        if isinstance(value, dict):
            flat.append(concatenate_ordered(value))
        else:
            flat.append(np.asarray(value, dtype=np.float32))
    return np.concatenate(flat, axis=-1)


def gather_time(x: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return x[idx]


def chunk_episode(
    observation: Dict,
    action: np.ndarray,
    ep_idx: int,
    quality_score: float,
    dataset_id: int,
    image_keys: list[str],
) -> Dict:
    ep_len = action.shape[0]
    idx = np.arange(ep_len)
    obs_idx = np.maximum(idx[:, None], 0)
    action_idx = idx[:, None]
    mask = action_idx < (ep_len - 1)
    action_idx = np.minimum(action_idx, ep_len - 1)

    obs = {
        "state": gather_time(observation["state"], obs_idx),
        "image": {image_key: gather_time(observation["image"][image_key], obs_idx) for image_key in image_keys},
    }
    act = gather_time(action, action_idx)
    act = np.where(mask[..., None], act, 0).astype(np.float32)

    # Match dataloader behavior: cut the last terminal transition.
    obs = {
        "state": obs["state"][:-1],
        "image": {image_key: obs["image"][image_key][:-1] for image_key in image_keys},
    }
    act = act[:-1]
    mask = mask[:-1]
    step_idx = np.arange(ep_len, dtype=np.int32)[:-1]

    length = step_idx.shape[0]
    return {
        "observation": obs,
        "action": act,
        "mask": mask,
        "ep_idx": np.full(length, ep_idx, dtype=np.int32),
        "quality_score": np.full(length, quality_score, dtype=np.float32),
        "dataset_id": np.full(length, dataset_id, dtype=np.int32),
        "step_idx": step_idx,
    }


def load_hdf5_dataset(
    spec: DatasetSpec,
    dataset_id: int,
    obs_structure: Dict,
    action_structure: Dict,
    stats: Dict,
    image_keys: list[str],
):
    episodes = []
    with h5py.File(spec.path, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]))
        for demo in demos:
            grp = f["data"][demo]
            if "obs" not in grp:
                raise KeyError(
                    f"{spec.path}:{demo} has no 'obs' group. DemInf image scoring expects an image.hdf5 "
                    "with RoboMimic observations, not a states-only demo.hdf5."
                )
            obs_grp = grp["obs"]
            for key in image_keys:
                hdf5_image_dataset = IMAGE_KEY_TO_HDF5_DATASET[key]
                if hdf5_image_dataset not in obs_grp:
                    raise KeyError(
                        f"{spec.path}:{demo}/obs is missing '{hdf5_image_dataset}'. "
                        f"Available observation keys: {sorted(obs_grp.keys())}"
                    )
            raw_obs = {
                "state": {
                    "EE_POS": obs_grp["robot0_eef_pos"][:].astype(np.float32),
                    "EE_QUAT": obs_grp["robot0_eef_quat"][:].astype(np.float32),
                    "GRIPPER": obs_grp["robot0_gripper_qpos"][:, :1].astype(np.float32),
                },
                "image": {
                    key: obs_grp[IMAGE_KEY_TO_HDF5_DATASET[key]][:].astype(np.float32) / 255.0
                    for key in image_keys
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

            ep = chunk_episode(
                observation={
                    "state": concatenate_ordered(normalized_obs["state"]),
                    "image": normalized_obs["image"],
                },
                action=concatenate_ordered(normalized_action),
                ep_idx=int(demo.split("_")[-1]),
                quality_score=spec.label,
                dataset_id=dataset_id,
                image_keys=image_keys,
            )
            episodes.append(ep)
    return episodes


def stack_batches(episodes: Iterable[Dict], batch_size: int, image_keys: list[str]):
    merged = {}
    for ep in episodes:
        for key, value in ep.items():
            if key not in merged:
                merged[key] = []
            merged[key].append(value)

    merged = {
        "observation": {
            "state": np.concatenate([x["state"] for x in merged["observation"]], axis=0),
            "image": {
                image_key: np.concatenate([x["image"][image_key] for x in merged["observation"]], axis=0)
                for image_key in image_keys
            },
        },
        "action": np.concatenate(merged["action"], axis=0),
        "mask": np.concatenate(merged["mask"], axis=0),
        "ep_idx": np.concatenate(merged["ep_idx"], axis=0),
        "quality_score": np.concatenate(merged["quality_score"], axis=0),
        "dataset_id": np.concatenate(merged["dataset_id"], axis=0),
        "step_idx": np.concatenate(merged["step_idx"], axis=0),
    }

    total = merged["action"].shape[0]
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        yield {
            "observation": {
                "state": merged["observation"]["state"][start:end],
                "image": {
                    image_key: merged["observation"]["image"][image_key][start:end]
                    for image_key in image_keys
                },
            },
            "action": merged["action"][start:end],
            "mask": merged["mask"][start:end],
            "ep_idx": merged["ep_idx"][start:end],
            "quality_score": merged["quality_score"][start:end],
            "dataset_id": merged["dataset_id"][start:end],
            "step_idx": merged["step_idx"][start:end],
        }


def aggregate_scores(
    stats_array: np.ndarray, attrs: Dict[str, np.ndarray], raw_stats_array: np.ndarray | None = None
) -> Dict:
    scores = {}
    for attr_name, attr in attrs.items():
        if attr_name == "step_idx":
            continue
        scores[attr_name] = {}
        for attr_val in np.unique(attr).tolist():
            scores[attr_name][int(attr_val) if np.issubdtype(type(attr_val), np.integer) else attr_val] = float(
                np.mean(stats_array[attr == attr_val])
            )

    scores["sample_score"] = stats_array
    if raw_stats_array is not None:
        scores["raw_sample_score"] = raw_stats_array
    for attr_name, attr in attrs.items():
        scores[f"sample_{attr_name}"] = attr

    if "ep_idx" in attrs and "quality_score" in attrs:
        scores["quality_by_ep_idx"] = {}
        for ep_idx in np.unique(attrs["ep_idx"]):
            scores["quality_by_ep_idx"][int(ep_idx)] = float(np.mean(attrs["quality_score"][attrs["ep_idx"] == ep_idx]))

    return scores


def normalize_scores(stats_array: np.ndarray, reference_scores: np.ndarray | None = None) -> np.ndarray:
    reference = stats_array if reference_scores is None else reference_scores
    clipped = np.clip(stats_array, a_min=np.percentile(reference, 1), a_max=np.percentile(reference, 99))
    std = np.std(reference)
    if std == 0:
        std = 1.0
    return (clipped - np.mean(reference)) / std


def merge_distance_diagnostics(total: Dict[str, float], batch_diag: Dict[str, np.ndarray]) -> Dict[str, float]:
    for key, value in batch_diag.items():
        value = float(np.asarray(value))
        if key.endswith("_max"):
            total[key] = max(total.get(key, float("-inf")), value)
        else:
            total[key] = total.get(key, 0.0) + value
    return total


def finalize_distance_diagnostics(total: Dict[str, float], ks: np.ndarray) -> Dict:
    out = {
        "ks": [int(k) for k in ks.tolist()],
        "note": (
            "KSG uses L-infinity over marginal latent L2 distances: "
            "joint_distance(i,j)=max(obs_l2(i,j), action_l2(i,j)). "
            "action_gt_obs_frac is the fraction where action_l2 > obs_l2, so action dominates the joint radius."
        ),
    }
    for prefix in ("all_pairs", "joint_knn"):
        count = total.get(f"{prefix}_count", 0.0)
        if count <= 0:
            out[prefix] = {"count": 0}
            continue
        action_gt = total.get(f"{prefix}_action_gt_obs_count", 0.0)
        obs_gt = total.get(f"{prefix}_obs_gt_action_count", 0.0)
        equal = total.get(f"{prefix}_equal_count", 0.0)
        out[prefix] = {
            "count": int(count),
            "action_gt_obs_count": int(action_gt),
            "obs_gt_action_count": int(obs_gt),
            "equal_count": int(equal),
            "action_gt_obs_frac": action_gt / count,
            "obs_gt_action_frac": obs_gt / count,
            "equal_frac": equal / count,
            "mean_obs_l2": total.get(f"{prefix}_obs_l2_sum", 0.0) / count,
            "mean_action_l2": total.get(f"{prefix}_action_l2_sum", 0.0) / count,
            "mean_joint_linf": total.get(f"{prefix}_joint_linf_sum", 0.0) / count,
            "max_joint_linf": total.get(f"{prefix}_joint_linf_max", float("nan")),
            "mean_action_minus_obs_l2": total.get(f"{prefix}_action_minus_obs_sum", 0.0) / count,
        }
    return out


def save_plots(ds_scores: Dict, out_dir: str, ds_name: str):
    if "quality_by_ep_idx" not in ds_scores:
        return
    idxs = list(ds_scores["ep_idx"].keys())
    x = np.array([ds_scores["quality_by_ep_idx"][idx] for idx in idxs])
    y = np.array([ds_scores["ep_idx"][idx] for idx in idxs])
    r = stats.pearsonr(x, y) if len(np.unique(x)) > 1 and len(y) > 1 else None
    title = f"r={r.statistic:.2f}" if r is not None else "r=n/a"

    _, ax = plt.subplots(1, 1, figsize=(3, 3))
    for v in np.sort(np.unique(x)):
        ax.hist(y[x == v], bins=20, alpha=0.5, label=f"Quality {v:g}")
    ax.legend()
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, ds_name + "_hist.png"))
    plt.close()

    plt.figure(figsize=(3, 3))
    sort_idx = np.argsort(y)
    rev_sorted_quality_labels = x[sort_idx][::-1]
    total_quality_labels = np.cumsum(rev_sorted_quality_labels)
    num_data_points = 1 + np.arange(total_quality_labels.shape[0])
    avg_quality_label = total_quality_labels / num_data_points
    plt.plot(np.arange(avg_quality_label.shape[0]), avg_quality_label[::-1], label="method")
    oracle_labels = np.cumsum(np.sort(x)[::-1]) / num_data_points
    plt.plot(np.arange(oracle_labels.shape[0]), oracle_labels[::-1], color="gray", label="oracle")
    plt.gca().hlines(np.mean(x), xmin=0, xmax=oracle_labels.shape[0], color="red", linestyles="dashed")
    plt.xlabel("Episodes Removed")
    plt.ylabel("Average Quality Label")
    plt.legend(frameon=False)
    plt.ylim(np.min(x), np.max(x))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, ds_name + "_curve.png"))
    plt.close()


def main():
    args = parse_args()
    dataset_specs = [parse_dataset_spec(spec) for spec in args.dataset]
    os.makedirs(args.output, exist_ok=True)

    obs_alg, obs_state, dataset_statistics, obs_config = load_checkpoint(args.obs_ckpt)
    if "mean" not in dataset_statistics and len(dataset_statistics) == 1:
        dataset_statistics = next(iter(dataset_statistics.values()))
    action_alg, action_state, _, _ = load_checkpoint(args.action_ckpt)

    obs_structure = obs_config.structure["observation"].to_dict()
    action_structure = obs_config.structure["action"].to_dict()
    checkpoint_image_keys = list(obs_structure.get("image", {}).keys())
    if args.camera is None:
        image_keys = checkpoint_image_keys
    elif args.camera == "both":
        image_keys = ["agent", "wrist"]
    else:
        image_keys = [args.camera]
    missing_image_keys = [key for key in image_keys if key not in obs_structure.get("image", {})]
    if missing_image_keys:
        raise ValueError(
            f"Checkpoint observation config expects image keys {checkpoint_image_keys}, "
            f"but requested image keys {image_keys}."
        )

    ks = np.arange(5, 8)
    if args.distance_diagnostics:
        pred_fn = jax.jit(
            lambda batch, rng: ksg_estimator_with_distance_diagnostics(
                batch,
                rng,
                ks=ks,
                obs_alg=obs_alg,
                obs_state=obs_state,
                action_alg=action_alg,
                action_state=action_state,
            )
        )
    else:
        pred_fn = jax.jit(
            lambda batch, rng: ksg_estimator(
                batch,
                rng,
                ks=ks,
                obs_alg=obs_alg,
                obs_state=obs_state,
                action_alg=action_alg,
                action_state=action_state,
            )
        )

    reference_scores = None
    if args.reference_score_pkl is not None:
        with open(args.reference_score_pkl, "rb") as f:
            reference_data = pickle.load(f)
        if "raw_sample_score" not in reference_data:
            raise ValueError(
                f"{args.reference_score_pkl} is missing raw_sample_score. "
                "Regenerate the reference scores with the updated quality estimation code."
            )
        reference_scores = np.asarray(reference_data["raw_sample_score"], dtype=np.float32)

    dataset_results = []
    for dataset_id, spec in enumerate(dataset_specs):
        episodes = load_hdf5_dataset(
            spec,
            dataset_id=dataset_id,
            obs_structure=obs_structure,
            action_structure=action_structure,
            stats=dataset_statistics,
            image_keys=image_keys,
        )

        stats_list = []
        attrs = {k: [] for k in ("ep_idx", "step_idx", "quality_score", "dataset_id")}
        distance_diag = {}
        rng = jax.random.key(0)

        for batch_index, batch in enumerate(stack_batches(episodes, args.batch_size, image_keys=image_keys)):
            rng = jax.random.fold_in(rng, batch_index)
            batch = jax.tree.map(jnp.asarray, batch)
            if args.distance_diagnostics:
                pred, batch_distance_diag = pred_fn(batch, rng)
                distance_diag = merge_distance_diagnostics(distance_diag, batch_distance_diag)
                pred = np.asarray(pred)
            else:
                pred = np.asarray(pred_fn(batch, rng))
            stats_list.append(pred)
            for key in attrs:
                attrs[key].append(np.asarray(batch[key]))

        stats_array = np.concatenate(stats_list, axis=0)
        attrs = {k: np.concatenate(v, axis=0) for k, v in attrs.items()}
        finalized_distance_diag = (
            finalize_distance_diagnostics(distance_diag, ks) if args.distance_diagnostics else None
        )
        dataset_results.append((spec, stats_array, attrs, finalized_distance_diag))

    if args.shared_normalization:
        combined_stats = np.concatenate([stats_array for _, stats_array, _, _ in dataset_results], axis=0)
        normalized_by_name = {
            spec.name: normalize_scores(stats_array, reference_scores if reference_scores is not None else combined_stats)
            for spec, stats_array, _, _ in dataset_results
        }
    else:
        normalized_by_name = {
            spec.name: normalize_scores(stats_array, reference_scores)
            for spec, stats_array, _, _ in dataset_results
        }

    for spec, raw_stats_array, attrs, distance_diag in dataset_results:
        stats_array = normalized_by_name[spec.name]
        scores = aggregate_scores(stats_array, attrs, raw_stats_array=raw_stats_array)
        if distance_diag is not None:
            scores["distance_diagnostics"] = distance_diag
            diag_path = os.path.join(args.output, spec.name + "_distance_diagnostics.json")
            with open(diag_path, "w") as f:
                json.dump(distance_diag, f, indent=2, sort_keys=True)
            print(f"Wrote {diag_path}")
        with open(os.path.join(args.output, spec.name + ".pkl"), "wb") as f:
            pickle.dump(scores, f)
        save_plots(scores, args.output, spec.name)
        print(f"Wrote {os.path.join(args.output, spec.name + '.pkl')}")


if __name__ == "__main__":
    main()
