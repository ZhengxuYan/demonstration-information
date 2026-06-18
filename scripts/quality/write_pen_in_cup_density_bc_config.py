#!/usr/bin/env python3
"""Generate robomimic BC density-model configs for pen-in-cup score sweeps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LOW_DIM_BY_CONDITION = {
    "image_state": ["robot_state"],
    "image": [],
    "state": ["robot_state"],
    "action_prior": ["action_prior_dummy"],
}

RGB_BY_CONDITION = {
    "image_state": ["agentview_image", "robot0_eye_in_hand_image"],
    "image": ["agentview_image", "robot0_eye_in_hand_image"],
    "state": [],
    "action_prior": [],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", choices=["gaussian", "gmm", "discrete"], required=True)
    parser.add_argument("--condition", choices=sorted(LOW_DIM_BY_CONDITION), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--repo", type=Path, default=Path("/iris/u/jasonyan/repos/demonstration-information"))
    parser.add_argument("--num-epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epoch-steps", type=int, default=100)
    parser.add_argument("--validation-steps", type=int, default=25)
    parser.add_argument("--save-every-n-epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--l2-regularization", type=float, default=0.0)
    parser.add_argument("--actor-layer-dims", type=str, default="1024,1024")
    parser.add_argument("--hdf5-normalize-obs", action="store_true")
    parser.add_argument("--gmm-modes", type=int, default=5)
    parser.add_argument("--gaussian-min-std", type=float, default=1e-4)
    parser.add_argument("--gaussian-fixed-std", action="store_true")
    parser.add_argument("--discrete-bins", type=int, default=256)
    parser.add_argument("--discrete-loss-type", choices=["hard_ce", "soft_ce"], default="hard_ce")
    parser.add_argument("--soft-sigma-bins", type=float, default=1.5)
    parser.add_argument("--soft-truncate-bins", type=int, default=6)
    parser.add_argument("--log-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="pen-in-cup-density")
    return parser.parse_args()


def load_base(repo: Path, algo: str) -> dict:
    if algo == "discrete":
        path = repo / "configs" / "robomimic" / "square_ph_bc_discrete_wrist.json"
    else:
        path = repo / "configs" / "robomimic" / "square_ph_bc_gmm_wrist.json"
    with path.open() as f:
        return json.load(f)


def parse_dims(value: str) -> list[int]:
    dims = [int(part) for part in value.split(",") if part.strip()]
    if not dims:
        raise ValueError("--actor-layer-dims cannot be empty")
    return dims


def main() -> None:
    args = parse_args()
    cfg = load_base(args.repo, "discrete" if args.algo == "discrete" else "gmm")

    cfg["experiment"]["name"] = args.run_name
    cfg["experiment"]["validate"] = True
    cfg["experiment"]["render"] = False
    cfg["experiment"]["render_video"] = False
    cfg["experiment"]["rollout"]["enabled"] = False
    cfg["experiment"]["save"]["enabled"] = True
    cfg["experiment"]["save"]["every_n_epochs"] = int(args.save_every_n_epochs)
    cfg["experiment"]["save"]["on_best_validation"] = True
    cfg["experiment"]["logging"]["log_wandb"] = bool(args.log_wandb)
    cfg["experiment"]["logging"]["wandb_proj_name"] = args.wandb_project
    cfg["experiment"]["epoch_every_n_steps"] = int(args.epoch_steps)
    cfg["experiment"]["validation_epoch_every_n_steps"] = int(args.validation_steps)

    cfg["train"]["data"] = str(args.dataset)
    cfg["train"]["output_dir"] = args.output_dir
    cfg["train"]["num_data_workers"] = 0
    cfg["train"]["hdf5_cache_mode"] = "low_dim"
    cfg["train"]["hdf5_filter_key"] = "train"
    cfg["train"]["hdf5_validation_filter_key"] = "valid"
    cfg["train"]["hdf5_load_next_obs"] = False
    cfg["train"]["hdf5_normalize_obs"] = bool(args.hdf5_normalize_obs)
    cfg["train"]["seq_length"] = 1
    cfg["train"]["frame_stack"] = 1
    cfg["train"]["batch_size"] = int(args.batch_size)
    cfg["train"]["num_epochs"] = int(args.num_epochs)
    cfg["train"]["seed"] = 1

    cfg["algo"]["actor_layer_dims"] = parse_dims(args.actor_layer_dims)
    cfg["algo"]["optim_params"]["policy"]["learning_rate"]["initial"] = float(args.learning_rate)
    cfg["algo"]["optim_params"]["policy"]["regularization"]["L2"] = float(args.l2_regularization)

    cfg["algo"]["gaussian"] = cfg["algo"].get("gaussian", {})
    cfg["algo"]["gmm"] = cfg["algo"].get("gmm", {})
    cfg["algo"]["discrete"] = cfg["algo"].get("discrete", {})
    cfg["algo"]["gaussian"]["enabled"] = args.algo == "gaussian"
    cfg["algo"]["gmm"]["enabled"] = args.algo == "gmm"
    cfg["algo"]["discrete"]["enabled"] = args.algo == "discrete"
    cfg["algo"]["vae"]["enabled"] = False
    cfg["algo"]["rnn"]["enabled"] = False
    cfg["algo"]["transformer"]["enabled"] = False

    cfg["algo"]["gaussian"]["fixed_std"] = bool(args.gaussian_fixed_std)
    cfg["algo"]["gaussian"]["init_std"] = 0.1
    cfg["algo"]["gaussian"]["min_std"] = float(args.gaussian_min_std)
    cfg["algo"]["gaussian"]["std_activation"] = "softplus"
    cfg["algo"]["gaussian"]["low_noise_eval"] = True

    cfg["algo"]["gmm"]["num_modes"] = int(args.gmm_modes)
    cfg["algo"]["gmm"]["min_std"] = float(args.gaussian_min_std)
    cfg["algo"]["gmm"]["std_activation"] = "softplus"
    cfg["algo"]["gmm"]["low_noise_eval"] = True

    cfg["algo"]["discrete"]["num_bins"] = int(args.discrete_bins)
    cfg["algo"]["discrete"]["bin_type"] = "uniform"
    cfg["algo"]["discrete"]["loss_type"] = args.discrete_loss_type
    cfg["algo"]["discrete"]["soft_sigma_bins"] = float(args.soft_sigma_bins)
    cfg["algo"]["discrete"]["soft_truncate_bins"] = int(args.soft_truncate_bins)

    cfg["observation"]["modalities"]["obs"]["low_dim"] = LOW_DIM_BY_CONDITION[args.condition]
    cfg["observation"]["modalities"]["obs"]["rgb"] = RGB_BY_CONDITION[args.condition]
    cfg["observation"]["modalities"]["obs"]["depth"] = []
    cfg["observation"]["modalities"]["obs"]["scan"] = []
    cfg["observation"]["modalities"]["goal"]["low_dim"] = []
    cfg["observation"]["modalities"]["goal"]["rgb"] = []

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cfg, indent=4) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
