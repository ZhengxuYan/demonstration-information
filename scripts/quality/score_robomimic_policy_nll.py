#!/usr/bin/env python3
"""Score robomimic demonstrations with a trained BC policy checkpoint.

The score is transition-level negative log likelihood under the policy:

* GMM: -log p_continuous(a_t | s_t)
* Discrete: -sum_j log p(bin(a_t[j]) | s_t)
* Soft discrete: -sum_j sum_i q_i(a_t[j]) log p(bin_i | s_t)

It also reports transition-level policy entropy:

* GMM: sample estimate E[-log pi(a | s)], with actions sampled from pi
* Discrete / soft discrete: closed-form categorical entropy over bins

Outputs a pickle with sample-level scores and per-trajectory means.
"""

from __future__ import annotations

import argparse
import csv
import pickle
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.distributions as D
from torch.utils.data import DataLoader
from tqdm import tqdm

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.algo import algo_factory
from robomimic.utils.train_utils import dataset_factory

from discrete_nll_utils import (
    discrete_entropy_from_logits,
    hard_discrete_nll_from_logits,
    soft_discrete_nll_from_logits,
)
from policy_entropy_utils import sample_entropy_from_distribution


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to robomimic .pth checkpoint.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to robomimic image.hdf5 dataset.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory.")
    parser.add_argument("--name", type=str, required=True, help="Output stem, e.g. gmm or discrete.")
    parser.add_argument("--filter-key", type=str, default=None, help="Optional HDF5 mask key, e.g. train or valid.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cuda or cpu.")
    parser.add_argument(
        "--discrete-loss-type",
        choices=["checkpoint", "hard_ce", "soft_ce"],
        default="checkpoint",
        help="Override discrete NLL target type. By default, use checkpoint config.",
    )
    parser.add_argument("--soft-sigma-bins", type=float, default=None, help="Override soft_ce Gaussian sigma in bins.")
    parser.add_argument("--soft-truncate-bins", type=int, default=None, help="Override soft_ce truncation radius in bins.")
    parser.add_argument("--gmm-entropy-samples", type=int, default=128, help="Number of action samples per state for GMM entropy.")
    parser.add_argument("--entropy-seed", type=int, default=0, help="Random seed for sample-based entropy estimates.")
    return parser.parse_args()


def load_algo(checkpoint: Path, dataset: Path, device: torch.device):
    ckpt_dict = FileUtils.load_dict_from_checkpoint(str(checkpoint))
    algo_name, _ = FileUtils.algo_name_from_checkpoint(ckpt_dict=ckpt_dict)
    config, _ = FileUtils.config_from_checkpoint(algo_name=algo_name, ckpt_dict=ckpt_dict, verbose=False)

    # Make dataset path explicit so scoring does not depend on the path saved in the checkpoint.
    config.unlock()
    config.train.data = [{"path": str(dataset)}]
    config.train.hdf5_cache_mode = "low_dim"
    config.train.num_data_workers = 0
    config.lock()

    ObsUtils.initialize_obs_utils_with_config(config)
    shape_meta = ckpt_dict["shape_metadata"]
    algo = algo_factory(
        algo_name,
        config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )
    algo.deserialize(ckpt_dict["model"])
    algo.set_eval()
    # For GMM policies, robomimic's eval path can force tiny std (1e-4) when
    # low_noise_eval=True, which is useful for action sampling but distorts
    # likelihood scoring. Disable it here so NLL uses the learned variance.
    policy = algo.nets["policy"] if "policy" in algo.nets else None
    if policy is not None and hasattr(policy, "low_noise_eval"):
        policy.low_noise_eval = False
    return algo, config


def index_metadata(dataset, indices: np.ndarray):
    ep_idxs = []
    step_idxs = []
    demo_keys = []
    for global_index in indices.astype(int):
        demo_key = dataset._index_to_demo_id[global_index]
        demo_start = dataset._demo_id_to_start_indices[demo_key]
        demo_index_offset = 0 if dataset.pad_frame_stack else (dataset.n_frame_stack - 1)
        step_idx = global_index - demo_start + demo_index_offset
        ep_idxs.append(int(demo_key.split("_")[-1]))
        step_idxs.append(int(step_idx))
        demo_keys.append(demo_key)
    return np.asarray(ep_idxs), np.asarray(step_idxs), np.asarray(demo_keys)


def make_loader(config, batch_size: int, num_workers: int, filter_key: str | None = None):
    dataset = dataset_factory(config, obs_keys=list(config.all_obs_keys), filter_by_attribute=filter_key)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    return dataset, loader


def discrete_score_settings(config, args) -> dict[str, object] | None:
    if "discrete" not in config.algo or not config.algo.discrete.enabled:
        return None
    loss_type = getattr(config.algo.discrete, "loss_type", "hard_ce")
    if args.discrete_loss_type != "checkpoint":
        loss_type = args.discrete_loss_type
    return {
        "loss_type": loss_type,
        "num_bins": int(config.algo.discrete.num_bins),
        "soft_sigma_bins": float(
            args.soft_sigma_bins
            if args.soft_sigma_bins is not None
            else getattr(config.algo.discrete, "soft_sigma_bins", 1.5)
        ),
        "soft_truncate_bins": int(
            args.soft_truncate_bins
            if args.soft_truncate_bins is not None
            else getattr(config.algo.discrete, "soft_truncate_bins", 6)
        ),
    }


def _final_timestep_mixture(dist):
    """Match BC_Transformer_GMM's final-timestep distribution slicing."""
    if len(dist.batch_shape) != 2:
        return dist
    component_distribution = D.Normal(
        loc=dist.component_distribution.base_dist.loc[:, -1],
        scale=dist.component_distribution.base_dist.scale[:, -1],
    )
    component_distribution = D.Independent(component_distribution, 1)
    mixture_distribution = D.Categorical(logits=dist.mixture_distribution.logits[:, -1])
    return D.MixtureSameFamily(
        mixture_distribution=mixture_distribution,
        component_distribution=component_distribution,
    )


def policy_distribution_for_batch(algo, input_batch):
    policy = algo.nets["policy"]
    kwargs = {
        "obs_dict": input_batch["obs"],
        "goal_dict": input_batch["goal_obs"],
    }
    if "Transformer" in type(policy).__name__:
        kwargs["actions"] = None
        kwargs["low_noise_eval"] = False
    try:
        dist = policy.forward_train(**kwargs)
    except TypeError:
        kwargs.pop("actions", None)
        kwargs.pop("low_noise_eval", None)
        dist = policy.forward_train(**kwargs)
    if hasattr(algo, "supervise_all_steps") and not algo.supervise_all_steps and hasattr(dist, "batch_shape"):
        dist = _final_timestep_mixture(dist)
    return dist


def score(algo, config, dataset, loader, args):
    discrete_settings = discrete_score_settings(config, args)
    sample_score = []
    sample_log_prob = []
    sample_entropy = []
    sample_ep_idx = []
    sample_step_idx = []
    sample_demo_key = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="scoring")):
            indices = np.asarray(TensorUtils.to_numpy(batch["index"]))
            input_batch = algo.process_batch_for_training(batch)
            input_batch = algo.postprocess_batch_for_training(
                input_batch,
                obs_normalization_stats=None,
            )
            predictions = algo._forward_training(input_batch)
            if discrete_settings is not None and "logits" in predictions:
                loss_type = str(discrete_settings["loss_type"])
                if loss_type == "soft_ce":
                    nll_t = soft_discrete_nll_from_logits(
                        predictions["logits"],
                        input_batch["actions"],
                        num_bins=int(discrete_settings["num_bins"]),
                        sigma_bins=float(discrete_settings["soft_sigma_bins"]),
                        truncate_bins=int(discrete_settings["soft_truncate_bins"]),
                    )
                elif loss_type == "hard_ce":
                    nll_t = hard_discrete_nll_from_logits(
                        predictions["logits"],
                        input_batch["actions"],
                        num_bins=int(discrete_settings["num_bins"]),
                    )
                else:
                    raise ValueError(f"Unsupported discrete loss_type: {loss_type}")
                log_probs = -TensorUtils.to_numpy(nll_t).astype(np.float64)
                entropy_t = discrete_entropy_from_logits(predictions["logits"], input_batch["actions"])
            else:
                log_probs = TensorUtils.to_numpy(predictions["log_probs"]).astype(np.float64)
                dist = policy_distribution_for_batch(algo, input_batch)
                entropy_t = sample_entropy_from_distribution(
                    dist,
                    num_samples=int(args.gmm_entropy_samples),
                    seed=int(args.entropy_seed) + batch_idx,
                    device=input_batch["actions"].device,
                )
            nll = -log_probs
            entropy = TensorUtils.to_numpy(entropy_t).astype(np.float64)

            ep_idxs, step_idxs, demo_keys = index_metadata(dataset, indices)
            sample_score.append(nll)
            sample_log_prob.append(log_probs)
            sample_entropy.append(entropy)
            sample_ep_idx.append(ep_idxs)
            sample_step_idx.append(step_idxs)
            sample_demo_key.append(demo_keys)

    sample_score = np.concatenate(sample_score, axis=0)
    sample_log_prob = np.concatenate(sample_log_prob, axis=0)
    sample_entropy = np.concatenate(sample_entropy, axis=0)
    sample_ep_idx = np.concatenate(sample_ep_idx, axis=0)
    sample_step_idx = np.concatenate(sample_step_idx, axis=0)
    sample_demo_key = np.concatenate(sample_demo_key, axis=0)

    ep_idx_scores = OrderedDict()
    ep_idx_entropy = OrderedDict()
    for ep_idx in sorted(np.unique(sample_ep_idx).tolist()):
        mask = sample_ep_idx == ep_idx
        ep_idx_scores[int(ep_idx)] = float(sample_score[mask].mean())
        ep_idx_entropy[int(ep_idx)] = float(sample_entropy[mask].mean())

    return {
        "ep_idx": ep_idx_scores,
        "ep_idx_nll": ep_idx_scores,
        "ep_idx_entropy": ep_idx_entropy,
        "sample_score": sample_score.astype(np.float32),
        "sample_nll": sample_score.astype(np.float32),
        "sample_log_prob": sample_log_prob.astype(np.float32),
        "sample_entropy": sample_entropy.astype(np.float32),
        "sample_ep_idx": sample_ep_idx.astype(np.int64),
        "sample_step_idx": sample_step_idx.astype(np.int64),
        "sample_demo_key": sample_demo_key,
        "score_name": (
            "soft_discrete_negative_log_likelihood"
            if discrete_settings is not None and discrete_settings["loss_type"] == "soft_ce"
            else "negative_log_likelihood"
        ),
        "score_parameters": discrete_settings or {},
        "entropy_name": (
            "closed_form_categorical_entropy"
            if discrete_settings is not None
            else "sampled_policy_entropy"
        ),
        "entropy_parameters": (
            {}
            if discrete_settings is not None
            else {
                "num_samples": int(args.gmm_entropy_samples),
                "seed": int(args.entropy_seed),
            }
        ),
        "checkpoint_config": {
            "algo_name": config.algo_name,
            "frame_stack": int(config.train.frame_stack),
            "seq_length": int(config.train.seq_length),
            "all_obs_keys": list(config.all_obs_keys),
        },
        "filter_key": getattr(dataset, "filter_by_attribute", None),
    }


def write_csv(scores: dict, path: Path):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ep_idx", "mean_nll", "mean_entropy"])
        for ep_idx, mean_nll in scores["ep_idx"].items():
            writer.writerow([ep_idx, mean_nll, scores["ep_idx_entropy"][ep_idx]])


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if args.device is None:
        device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    else:
        device = torch.device(args.device)

    algo, config = load_algo(args.checkpoint, args.dataset, device)
    dataset, loader = make_loader(config, args.batch_size, args.num_workers, args.filter_key)
    scores = score(algo, config, dataset, loader, args)

    pkl_path = args.output / f"{args.name}.pkl"
    csv_path = args.output / f"{args.name}_trajectory_scores.csv"
    with pkl_path.open("wb") as f:
        pickle.dump(scores, f)
    write_csv(scores, csv_path)

    values = np.asarray(list(scores["ep_idx"].values()), dtype=np.float64)
    print(f"wrote {pkl_path}")
    print(f"wrote {csv_path}")
    print(
        "trajectory mean NLL: "
        f"n={len(values)} mean={values.mean():.6f} std={values.std():.6f} "
        f"min={values.min():.6f} max={values.max():.6f}"
    )
    ent_values = np.asarray(list(scores["ep_idx_entropy"].values()), dtype=np.float64)
    print(
        "trajectory mean entropy: "
        f"n={len(ent_values)} mean={ent_values.mean():.6f} std={ent_values.std():.6f} "
        f"min={ent_values.min():.6f} max={ent_values.max():.6f}"
    )


if __name__ == "__main__":
    main()
