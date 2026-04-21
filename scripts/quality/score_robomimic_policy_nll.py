#!/usr/bin/env python3
"""Score robomimic demonstrations with a trained BC policy checkpoint.

The score is transition-level negative log likelihood under the policy:

* GMM: -log p_continuous(a_t | s_t)
* Discrete: -sum_j log p(bin(a_t[j]) | s_t)

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
from torch.utils.data import DataLoader
from tqdm import tqdm

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.algo import algo_factory
from robomimic.utils.train_utils import dataset_factory


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to robomimic .pth checkpoint.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to robomimic image.hdf5 dataset.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory.")
    parser.add_argument("--name", type=str, required=True, help="Output stem, e.g. gmm or discrete.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cuda or cpu.")
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


def make_loader(config, batch_size: int, num_workers: int):
    dataset = dataset_factory(config, obs_keys=list(config.all_obs_keys), filter_by_attribute=None)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    return dataset, loader


def score(algo, config, dataset, loader):
    sample_score = []
    sample_log_prob = []
    sample_ep_idx = []
    sample_step_idx = []
    sample_demo_key = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="scoring"):
            indices = np.asarray(TensorUtils.to_numpy(batch["index"]))
            input_batch = algo.process_batch_for_training(batch)
            input_batch = algo.postprocess_batch_for_training(
                input_batch,
                obs_normalization_stats=None,
            )
            predictions = algo._forward_training(input_batch)
            log_probs = TensorUtils.to_numpy(predictions["log_probs"]).astype(np.float64)
            nll = -log_probs

            ep_idxs, step_idxs, demo_keys = index_metadata(dataset, indices)
            sample_score.append(nll)
            sample_log_prob.append(log_probs)
            sample_ep_idx.append(ep_idxs)
            sample_step_idx.append(step_idxs)
            sample_demo_key.append(demo_keys)

    sample_score = np.concatenate(sample_score, axis=0)
    sample_log_prob = np.concatenate(sample_log_prob, axis=0)
    sample_ep_idx = np.concatenate(sample_ep_idx, axis=0)
    sample_step_idx = np.concatenate(sample_step_idx, axis=0)
    sample_demo_key = np.concatenate(sample_demo_key, axis=0)

    ep_idx_scores = OrderedDict()
    for ep_idx in sorted(np.unique(sample_ep_idx).tolist()):
        mask = sample_ep_idx == ep_idx
        ep_idx_scores[int(ep_idx)] = float(sample_score[mask].mean())

    return {
        "ep_idx": ep_idx_scores,
        "sample_score": sample_score.astype(np.float32),
        "sample_log_prob": sample_log_prob.astype(np.float32),
        "sample_ep_idx": sample_ep_idx.astype(np.int64),
        "sample_step_idx": sample_step_idx.astype(np.int64),
        "sample_demo_key": sample_demo_key,
        "score_name": "negative_log_likelihood",
        "checkpoint_config": {
            "algo_name": config.algo_name,
            "frame_stack": int(config.train.frame_stack),
            "seq_length": int(config.train.seq_length),
            "all_obs_keys": list(config.all_obs_keys),
        },
    }


def write_csv(scores: dict, path: Path):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ep_idx", "mean_nll"])
        for ep_idx, mean_nll in scores["ep_idx"].items():
            writer.writerow([ep_idx, mean_nll])


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if args.device is None:
        device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    else:
        device = torch.device(args.device)

    algo, config = load_algo(args.checkpoint, args.dataset, device)
    dataset, loader = make_loader(config, args.batch_size, args.num_workers)
    scores = score(algo, config, dataset, loader)

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


if __name__ == "__main__":
    main()
