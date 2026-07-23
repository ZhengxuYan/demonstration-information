#!/usr/bin/env python3
"""Compute the six POMDP action-information scores from trained density models."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.distributions as D
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.algo import algo_factory
from robomimic.utils.train_utils import dataset_factory


SCORE_NAMES = (
    "neg_h_data_cond",
    "neg_h_model_cond",
    "mi_data_direct",
    "mi_data_mc_marginal",
    "mi_model_direct",
    "mi_model_mc_marginal",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditional-checkpoint", type=Path, required=True)
    parser.add_argument("--prior-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--filter-key", default="score_all")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--mc-action-samples", type=int, default=16)
    parser.add_argument("--mc-marginal-states", type=int, default=512)
    parser.add_argument("--marginal-state-chunk", type=int, default=128)
    parser.add_argument("--eval-action-chunk", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--action-dims",
        default=None,
        help="Comma-separated action dimensions to score, e.g. 0,1,2,3,4,5. Defaults to all dimensions.",
    )
    return parser.parse_args()


def parse_action_dims(value: str | None, action_dim: int) -> tuple[int, ...]:
    if value is None:
        return tuple(range(action_dim))
    dims = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not dims:
        raise ValueError("--action-dims must contain at least one dimension")
    if len(set(dims)) != len(dims):
        raise ValueError(f"--action-dims contains duplicates: {dims}")
    if min(dims) < 0 or max(dims) >= action_dim:
        raise ValueError(f"--action-dims {dims} outside action dimension {action_dim}")
    return dims


def load_algo(checkpoint: Path, dataset: Path, device: torch.device):
    ckpt_dict = FileUtils.load_dict_from_checkpoint(str(checkpoint))
    algo_name, _ = FileUtils.algo_name_from_checkpoint(ckpt_dict=ckpt_dict)
    config, _ = FileUtils.config_from_checkpoint(algo_name=algo_name, ckpt_dict=ckpt_dict, verbose=False)
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
    policy = algo.nets["policy"] if "policy" in algo.nets else None
    if policy is not None and hasattr(policy, "low_noise_eval"):
        policy.low_noise_eval = False
    return algo, config


def make_loader(config, batch_size: int, filter_key: str):
    dataset = dataset_factory(config, obs_keys=list(config.all_obs_keys), filter_by_attribute=filter_key)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    return dataset, loader


def index_metadata(dataset, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ep_idxs = []
    step_idxs = []
    for global_index in indices.astype(int):
        demo_key = dataset._index_to_demo_id[global_index]
        demo_start = dataset._demo_id_to_start_indices[demo_key]
        demo_index_offset = 0 if dataset.pad_frame_stack else (dataset.n_frame_stack - 1)
        ep_idxs.append(int(demo_key.split("_")[-1]))
        step_idxs.append(int(global_index - demo_start + demo_index_offset))
    return np.asarray(ep_idxs, dtype=np.int64), np.asarray(step_idxs, dtype=np.int64)


def marginalize_action_dist(dist: D.Distribution, action_dims: tuple[int, ...]) -> D.Distribution:
    """Return the marginal distribution over selected action dimensions."""
    index = torch.as_tensor(action_dims, dtype=torch.long, device=dist.mean.device)
    if isinstance(dist, D.Independent) and isinstance(dist.base_dist, D.Normal):
        base = dist.base_dist
        return D.Independent(
            D.Normal(
                loc=torch.index_select(base.loc, -1, index),
                scale=torch.index_select(base.scale, -1, index),
            ),
            1,
        )
    if isinstance(dist, D.MixtureSameFamily):
        component = dist.component_distribution
        if not isinstance(component, D.Independent) or not isinstance(component.base_dist, D.Normal):
            raise TypeError(f"Unsupported mixture component distribution: {type(component)}")
        base = component.base_dist
        projected_component = D.Independent(
            D.Normal(
                loc=torch.index_select(base.loc, -1, index),
                scale=torch.index_select(base.scale, -1, index),
            ),
            1,
        )
        return D.MixtureSameFamily(dist.mixture_distribution, projected_component)
    raise TypeError(f"Unsupported policy distribution for action marginal: {type(dist)}")


def policy_dist(algo, obs: dict[str, torch.Tensor], action_dims: tuple[int, ...]):
    dist = algo.nets["policy"].forward_train(obs_dict=obs, goal_dict=OrderedDict())
    return marginalize_action_dist(dist, action_dims)


def log_prob(
    algo,
    obs: dict[str, torch.Tensor],
    actions: torch.Tensor,
    action_dims: tuple[int, ...],
) -> torch.Tensor:
    return policy_dist(algo, obs, action_dims).log_prob(actions)


def flatten_actions(actions: torch.Tensor) -> torch.Tensor:
    """Convert robomimic action tensors with horizon=1 to plain [..., ac_dim]."""
    if actions.ndim >= 3 and actions.shape[-2] == 1:
        actions = actions.squeeze(-2)
    if actions.ndim < 2:
        raise ValueError(f"Expected action tensor with at least 2 dims, got {tuple(actions.shape)}")
    return actions


def sample_actions(dist: D.Distribution, num_samples: int, seed: int, device: torch.device) -> torch.Tensor:
    devices = [device.index] if device.type == "cuda" and device.index is not None else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        return dist.sample(sample_shape=torch.Size([num_samples]))


def prior_obs(batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    return {"action_prior_dummy": torch.zeros((batch_size, 1), dtype=torch.float32, device=device)}


def flatten_obs(value: torch.Tensor) -> torch.Tensor:
    """Remove robomimic's sequence axis while preserving image dimensions."""
    if value.ndim >= 3 and value.shape[1] == 1:
        value = value[:, 0]
    elif value.ndim == 3 and value.shape[-2] == 1:
        value = value.squeeze(-2)
    return value


def as_device_obs(obs: dict, keys: list[str], device: torch.device) -> dict[str, torch.Tensor]:
    processed = ObsUtils.process_obs_dict({key: obs[key] for key in keys})
    return {
        key: flatten_obs(TensorUtils.to_device(processed[key], device).float())
        for key in keys
    }


def conditional_obs_keys(config) -> list[str]:
    keys = [key for key in config.all_obs_keys if key != "action_prior_dummy"]
    if not keys:
        raise ValueError(f"Conditional checkpoint has no observation keys: {list(config.all_obs_keys)}")
    return keys


def collect_state_pool(
    dataset,
    obs_keys: list[str],
    k: int,
    seed: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if len(dataset) == 0:
        raise ValueError("No observations available for the MC marginal pool")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=k, replace=len(dataset) < k)
    loader = DataLoader(
        Subset(dataset, indices.astype(int).tolist()),
        batch_size=min(64, k),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    observations: dict[str, list[np.ndarray]] = {key: [] for key in obs_keys}
    for batch in loader:
        processed = ObsUtils.process_obs_dict({key: batch["obs"][key] for key in obs_keys})
        for key in obs_keys:
            value = flatten_obs(processed[key].float())
            observations[key].append(TensorUtils.to_numpy(value).astype(np.float32))
    all_observations = {key: np.concatenate(values, axis=0) for key, values in observations.items()}
    return {
        key: torch.as_tensor(values, dtype=torch.float32, device=device)
        for key, values in all_observations.items()
    }


def concatenate_distributions(parts: list[D.Distribution]) -> D.Distribution:
    first = parts[0]
    if isinstance(first, D.Independent) and isinstance(first.base_dist, D.Normal):
        return D.Independent(
            D.Normal(
                torch.cat([part.base_dist.loc for part in parts], dim=0),
                torch.cat([part.base_dist.scale for part in parts], dim=0),
            ),
            1,
        )
    if isinstance(first, D.MixtureSameFamily):
        mixture = D.Categorical(logits=torch.cat([part.mixture_distribution.logits for part in parts], dim=0))
        component = D.Independent(
            D.Normal(
                torch.cat([part.component_distribution.base_dist.loc for part in parts], dim=0),
                torch.cat([part.component_distribution.base_dist.scale for part in parts], dim=0),
            ),
            1,
        )
        return D.MixtureSameFamily(mixture, component)
    raise TypeError(f"Cannot concatenate distribution type {type(first)}")


def reference_distribution(
    algo,
    observation_pool: dict[str, torch.Tensor],
    action_dims: tuple[int, ...],
    chunk_size: int,
) -> D.Distribution:
    total = len(next(iter(observation_pool.values())))
    parts = []
    for start in range(0, total, chunk_size):
        obs = {key: value[start : start + chunk_size] for key, value in observation_pool.items()}
        parts.append(policy_dist(algo, obs, action_dims))
    return concatenate_distributions(parts)


def mc_log_marginal(
    cond_algo,
    reference_dist: D.Distribution,
    actions: torch.Tensor,
    action_dims: tuple[int, ...],
    action_chunk: int,
) -> torch.Tensor:
    """Return log mean_k q_theta(actions_b | state_k) for each action_b."""
    device = actions.device
    outputs = []
    k_total = int(reference_dist.batch_shape[0])
    for action_start in range(0, actions.shape[0], action_chunk):
        action_block = actions[action_start : action_start + action_chunk]
        log_q = reference_dist.log_prob(action_block[:, None, :])
        outputs.append(torch.logsumexp(log_q, dim=1) - np.log(k_total))
    return torch.cat(outputs, dim=0).to(device)


def add_episode_values(grouped: dict[int, dict[str, list[float]]], ep: np.ndarray, values: dict[str, np.ndarray]) -> None:
    for idx, ep_idx in enumerate(ep.astype(int).tolist()):
        bucket = grouped.setdefault(int(ep_idx), {name: [] for name in SCORE_NAMES})
        for name in SCORE_NAMES:
            bucket[name].append(float(values[name][idx]))


def main() -> None:
    args = parse_args()
    if args.mc_action_samples <= 0 or args.mc_marginal_states <= 0:
        raise ValueError("MC sample counts must be positive")
    device = torch.device(args.device) if args.device else TorchUtils.get_torch_device(try_to_use_cuda=True)
    cond_algo, cond_config = load_algo(args.conditional_checkpoint, args.dataset, device)
    prior_algo, _ = load_algo(args.prior_checkpoint, args.dataset, device)
    action_dim = int(cond_algo.ac_dim)
    action_dims = parse_action_dims(args.action_dims, action_dim)
    action_index = torch.as_tensor(action_dims, dtype=torch.long, device=device)
    dataset, loader = make_loader(cond_config, args.batch_size, args.filter_key)
    obs_keys = conditional_obs_keys(cond_config)
    state_pool = collect_state_pool(dataset, obs_keys, args.mc_marginal_states, args.seed, device)
    with torch.no_grad():
        reference_dist = reference_distribution(
            cond_algo,
            state_pool,
            action_dims,
            max(1, args.marginal_state_chunk),
        )

    grouped: dict[int, dict[str, list[float]]] = {}
    sample_rows = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="pomdp scores")):
            indices = np.asarray(TensorUtils.to_numpy(batch["index"]))
            ep_idxs, step_idxs = index_metadata(dataset, indices)
            full_actions = flatten_actions(TensorUtils.to_device(batch["actions"], device).float())
            actions = torch.index_select(full_actions, -1, action_index)
            cond_obs = as_device_obs(batch["obs"], obs_keys, device)
            b = int(actions.shape[0])

            cond_dist = policy_dist(cond_algo, cond_obs, action_dims)
            log_cond_data = cond_dist.log_prob(actions)
            log_prior_data = log_prob(prior_algo, prior_obs(b, device), actions, action_dims)
            log_mc_data = mc_log_marginal(
                cond_algo,
                reference_dist,
                actions,
                action_dims,
                args.eval_action_chunk,
            )

            sampled = sample_actions(
                cond_dist,
                args.mc_action_samples,
                args.seed + 1009 * batch_idx,
                device,
            )
            flat_sampled = sampled.reshape(args.mc_action_samples * b, -1)
            log_cond_model = cond_dist.log_prob(sampled)
            log_prior_model = log_prob(
                prior_algo,
                prior_obs(args.mc_action_samples * b, device),
                flat_sampled,
                action_dims,
            ).reshape(args.mc_action_samples, b)
            log_mc_model = mc_log_marginal(
                cond_algo,
                reference_dist,
                flat_sampled,
                action_dims,
                args.eval_action_chunk,
            ).reshape(args.mc_action_samples, b)

            values_t = {
                "neg_h_data_cond": log_cond_data,
                "neg_h_model_cond": log_cond_model.mean(dim=0),
                "mi_data_direct": log_cond_data - log_prior_data,
                "mi_data_mc_marginal": log_cond_data - log_mc_data,
                "mi_model_direct": (log_cond_model - log_prior_model).mean(dim=0),
                "mi_model_mc_marginal": (log_cond_model - log_mc_model).mean(dim=0),
            }
            values = {name: TensorUtils.to_numpy(tensor).astype(np.float64) for name, tensor in values_t.items()}
            add_episode_values(grouped, ep_idxs, values)
            for row_idx, ep_idx in enumerate(ep_idxs.astype(int).tolist()):
                row = {"ep_idx": ep_idx, "step_idx": int(step_idxs[row_idx])}
                row.update({name: float(values[name][row_idx]) for name in SCORE_NAMES})
                sample_rows.append(row)

    episode_scores = OrderedDict()
    for ep_idx in sorted(grouped):
        episode_scores[ep_idx] = OrderedDict(
            (name, float(np.mean(grouped[ep_idx][name]))) for name in SCORE_NAMES
        )

    out = {
        "ep_idx": {name: OrderedDict((ep, vals[name]) for ep, vals in episode_scores.items()) for name in SCORE_NAMES},
        "episode_scores": episode_scores,
        "sample_rows": sample_rows,
        "score_names": SCORE_NAMES,
        "metadata": {
            "conditional_checkpoint": str(args.conditional_checkpoint),
            "prior_checkpoint": str(args.prior_checkpoint),
            "dataset": str(args.dataset),
            "filter_key": args.filter_key,
            "mc_action_samples": int(args.mc_action_samples),
            "mc_marginal_states": int(args.mc_marginal_states),
            "seed": int(args.seed),
            "action_dims": list(action_dims),
            "source_action_dim": action_dim,
            "conditional_obs_keys": obs_keys,
            "higher_is_better": True,
        },
    }

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "threading_pomdp_6_scores.pkl").open("wb") as f:
        pickle.dump(out, f)
    with (args.output / "threading_pomdp_6_scores.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ep_idx", *SCORE_NAMES])
        for ep_idx, vals in episode_scores.items():
            writer.writerow([ep_idx, *[vals[name] for name in SCORE_NAMES]])
    with (args.output / "threading_pomdp_6_sample_scores.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, ["ep_idx", "step_idx", *SCORE_NAMES])
        writer.writeheader()
        writer.writerows(sample_rows)
    (args.output / "metadata.json").write_text(json.dumps(out["metadata"], indent=2) + "\n")
    print(args.output / "threading_pomdp_6_scores.csv")
    print(f"episodes={len(episode_scores)} samples={len(sample_rows)}")


if __name__ == "__main__":
    main()
