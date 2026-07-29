#!/usr/bin/env python3
"""Measure conditional action likelihood after controlled image perturbations."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils

from score_threading_pomdp_6 import (
    as_device_obs,
    conditional_obs_keys,
    flatten_actions,
    index_metadata,
    load_algo,
    make_loader,
    policy_dist,
)


EXTERIOR = "agentview_image"
WRIST = "robot0_eye_in_hand_image"
VARIANTS = ("correct", "shuffle_exterior", "shuffle_wrist", "shuffle_both", "temporal_shift")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--filter-key", default="score_all")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--progress-bins", type=int, default=10)
    parser.add_argument("--temporal-shift-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--device")
    return parser.parse_args()


def metadata(dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(dataset), dtype=np.int64)
    ep, step = index_metadata(dataset, indices)
    lengths: dict[int, int] = defaultdict(int)
    for ep_idx in ep:
        lengths[int(ep_idx)] += 1
    progress = np.asarray(
        [step_idx / max(1, lengths[int(ep_idx)] - 1) for ep_idx, step_idx in zip(ep, step)],
        dtype=np.float64,
    )
    return ep, step, progress


def shuffled_donors(
    ep: np.ndarray,
    days: np.ndarray,
    progress: np.ndarray,
    progress_bins: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bins = np.minimum(progress_bins - 1, (progress * progress_bins).astype(int))
    donors = np.empty(len(ep), dtype=np.int64)
    all_indices = np.arange(len(ep), dtype=np.int64)
    for index in all_indices:
        candidates = all_indices[
            (days == days[index]) & (bins == bins[index]) & (ep != ep[index])
        ]
        if not len(candidates):
            candidates = all_indices[(bins == bins[index]) & (ep != ep[index])]
        if not len(candidates):
            raise ValueError(f"No cross-episode donor for local index {index}")
        donors[index] = int(rng.choice(candidates))
    return donors


def temporal_donors(
    ep: np.ndarray,
    step: np.ndarray,
    fraction: float,
) -> np.ndarray:
    lookup = {(int(e), int(s)): index for index, (e, s) in enumerate(zip(ep, step))}
    lengths = {int(e): int(step[ep == e].max()) + 1 for e in np.unique(ep)}
    result = np.empty(len(ep), dtype=np.int64)
    for index, (ep_idx, step_idx) in enumerate(zip(ep, step)):
        length = lengths[int(ep_idx)]
        offset = max(1, int(round(fraction * length)))
        shifted = (int(step_idx) + offset) % length
        result[index] = lookup[(int(ep_idx), shifted)]
    return result


def replace_images(
    target: dict[str, torch.Tensor],
    shuffled: dict[str, torch.Tensor],
    temporal: dict[str, torch.Tensor],
    variant: str,
) -> dict[str, torch.Tensor]:
    result = dict(target)
    if variant in {"shuffle_exterior", "shuffle_both"}:
        result[EXTERIOR] = shuffled[EXTERIOR]
    if variant in {"shuffle_wrist", "shuffle_both"}:
        result[WRIST] = shuffled[WRIST]
    if variant == "temporal_shift":
        result[EXTERIOR] = temporal[EXTERIOR]
        result[WRIST] = temporal[WRIST]
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    device = (
        torch.device(args.device)
        if args.device
        else TorchUtils.get_torch_device(try_to_use_cuda=True)
    )
    algo, config = load_algo(args.checkpoint, args.dataset, device)
    obs_keys = conditional_obs_keys(config)
    if EXTERIOR not in obs_keys or WRIST not in obs_keys:
        raise ValueError(f"Counterfactual checkpoint must use both RGB keys, got {obs_keys}")
    dataset, target_loader = make_loader(config, args.batch_size, args.filter_key)
    ep, step, progress = metadata(dataset)
    manifest = pd.read_csv(args.manifest).set_index("ep_idx")
    if set(ep.tolist()) - set(manifest.index.astype(int)):
        raise ValueError("Manifest does not cover every scored episode")
    days = np.asarray(
        [str(manifest.loc[int(ep_idx), "episode"]).split("_", 1)[0] for ep_idx in ep]
    )
    shuffle_indices = shuffled_donors(
        ep, days, progress, args.progress_bins, args.seed
    )
    temporal_indices = temporal_donors(ep, step, args.temporal_shift_fraction)
    shuffled_loader = DataLoader(
        Subset(dataset, shuffle_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    temporal_loader = DataLoader(
        Subset(dataset, temporal_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for target_batch, shuffled_batch, temporal_batch in tqdm(
            zip(target_loader, shuffled_loader, temporal_loader),
            total=len(target_loader),
            desc="image counterfactuals",
        ):
            local_indices = np.asarray(TensorUtils.to_numpy(target_batch["index"])).astype(int)
            batch_ep, batch_step = index_metadata(dataset, local_indices)
            actions = flatten_actions(
                TensorUtils.to_device(target_batch["actions"], device).float()
            )
            target_obs = as_device_obs(target_batch["obs"], obs_keys, device)
            shuffled_obs = as_device_obs(shuffled_batch["obs"], obs_keys, device)
            temporal_obs = as_device_obs(temporal_batch["obs"], obs_keys, device)
            values: dict[str, np.ndarray] = {}
            for variant in VARIANTS:
                obs = replace_images(target_obs, shuffled_obs, temporal_obs, variant)
                log_prob = policy_dist(algo, obs, tuple(range(actions.shape[-1]))).log_prob(actions)
                values[variant] = TensorUtils.to_numpy(log_prob).astype(np.float64)
            for row_index, ep_idx in enumerate(batch_ep):
                row: dict[str, object] = {
                    "ep_idx": int(ep_idx),
                    "step_idx": int(batch_step[row_index]),
                    "progress": float(progress[local_indices[row_index]]),
                }
                for variant in VARIANTS:
                    value = float(values[variant][row_index])
                    row[f"log_prob_{variant}"] = value
                    if variant != "correct":
                        row[f"delta_{variant}"] = float(values["correct"][row_index] - value)
                rows.append(row)

    numeric = np.asarray(
        [[float(row[f"log_prob_{variant}"]) for variant in VARIANTS] for row in rows]
    )
    if not np.isfinite(numeric).all():
        raise ValueError("Non-finite counterfactual log probability")
    write_csv(args.output / "transition_image_counterfactuals.csv", rows)
    episode_rows = []
    for ep_idx in sorted({int(row["ep_idx"]) for row in rows}):
        selected = [row for row in rows if int(row["ep_idx"]) == ep_idx]
        episode_row: dict[str, object] = {"ep_idx": ep_idx, "num_steps": len(selected)}
        for key in rows[0]:
            if key.startswith(("log_prob_", "delta_")):
                episode_row[key] = float(np.mean([float(row[key]) for row in selected]))
        episode_rows.append(episode_row)
    write_csv(args.output / "episode_image_counterfactuals.csv", episode_rows)
    metadata_out = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "manifest": str(args.manifest),
        "filter_key": args.filter_key,
        "seed": args.seed,
        "progress_bins": args.progress_bins,
        "temporal_shift_fraction": args.temporal_shift_fraction,
        "obs_keys": obs_keys,
        "variants": list(VARIANTS),
        "episodes": len(episode_rows),
        "transitions": len(rows),
        "finite": True,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metadata.json").write_text(json.dumps(metadata_out, indent=2) + "\n")
    print(json.dumps(metadata_out, indent=2))


if __name__ == "__main__":
    main()
