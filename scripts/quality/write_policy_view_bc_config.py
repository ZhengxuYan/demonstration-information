#!/usr/bin/env python3
"""Generate robomimic BC configs for policy-view experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


RGB_KEYS = {
    "agent_wrist": ["agentview_image", "robot0_eye_in_hand_image"],
    "left_close_low_wrist": ["left_close_low_image", "robot0_eye_in_hand_image"],
}

LOW_DIM_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", choices=["gmm", "discrete"], required=True)
    parser.add_argument("--view", choices=["agent_wrist", "left_close_low_wrist"], required=True)
    parser.add_argument("--repo", type=Path, default=Path("/iris/u/jasonyan/repos/demonstration-information"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-dir", type=str, default="/iris/u/jasonyan/data/robomimic_outputs/policy_view_experiments")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=Path("/iris/u/jasonyan/data/policy_view_experiments/square_ph"))
    parser.add_argument("--dataset-prefix", type=str, default="square_ph")
    parser.add_argument("--run-prefix", type=str, default="square_ph_bc")
    parser.add_argument("--run-name", type=str, default=None, help="Explicit experiment name. Overrides prefix/algo/view/suffix.")
    parser.add_argument("--suffix", type=str, default="_200_seed1")
    parser.add_argument("--obs-mode", choices=["image_state", "state_only"], default="image_state")
    parser.add_argument("--num-epochs", type=int, default=2000)
    parser.add_argument("--enable-validation", action="store_true")
    parser.add_argument("--log-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="policy-view-bc-random-post")
    parser.add_argument("--l2-regularization", type=float, default=0.0)
    parser.add_argument("--discrete-loss-type", choices=["hard_ce", "soft_ce"], default="hard_ce")
    parser.add_argument("--soft-sigma-bins", type=float, default=1.5)
    parser.add_argument("--soft-truncate-bins", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_name = "square_ph_bc_gmm_wrist.json" if args.algo == "gmm" else "square_ph_bc_discrete_wrist.json"
    base_path = args.repo / "configs" / "robomimic" / base_name
    with base_path.open() as f:
        cfg = json.load(f)

    dataset_path = args.dataset or args.dataset_root / f"{args.dataset_prefix}_{args.view}_image.hdf5"
    exp_name = args.run_name or f"{args.run_prefix}_{args.algo}_{args.view}{args.suffix}"
    cfg["experiment"]["name"] = exp_name
    cfg["train"]["data"] = str(dataset_path)
    cfg["train"]["output_dir"] = args.output_dir
    cfg["train"]["num_epochs"] = args.num_epochs
    cfg["train"]["hdf5_filter_key"] = "train" if args.enable_validation else None
    cfg["train"]["hdf5_validation_filter_key"] = "valid" if args.enable_validation else None
    cfg["experiment"]["validate"] = bool(args.enable_validation)
    cfg["experiment"]["save"]["on_best_validation"] = bool(args.enable_validation)
    cfg["experiment"]["logging"]["log_wandb"] = bool(args.log_wandb)
    if args.log_wandb:
        cfg["experiment"]["logging"]["wandb_proj_name"] = args.wandb_project
    cfg["algo"]["optim_params"]["policy"]["regularization"]["L2"] = args.l2_regularization
    if args.algo == "discrete":
        cfg["algo"]["discrete"]["loss_type"] = args.discrete_loss_type
        cfg["algo"]["discrete"]["soft_sigma_bins"] = args.soft_sigma_bins
        cfg["algo"]["discrete"]["soft_truncate_bins"] = args.soft_truncate_bins
    cfg["observation"]["modalities"]["obs"]["low_dim"] = LOW_DIM_KEYS
    if args.obs_mode == "state_only":
        cfg["observation"]["modalities"]["obs"]["rgb"] = []
        cfg["train"]["hdf5_cache_mode"] = "low_dim"
    else:
        cfg["observation"]["modalities"]["obs"]["rgb"] = RGB_KEYS[args.view]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cfg, indent=4) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
