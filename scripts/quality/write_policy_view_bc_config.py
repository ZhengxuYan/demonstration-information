#!/usr/bin/env python3
"""Generate robomimic BC configs for policy-view experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASETS = {
    "agent_wrist": "/iris/u/jasonyan/data/policy_view_experiments/square_ph/square_ph_agent_wrist_image.hdf5",
    "left_close_low_wrist": "/iris/u/jasonyan/data/policy_view_experiments/square_ph/square_ph_left_close_low_wrist_image.hdf5",
}

RGB_KEYS = {
    "agent_wrist": ["agentview_image", "robot0_eye_in_hand_image"],
    "left_close_low_wrist": ["left_close_low_image", "robot0_eye_in_hand_image"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", choices=["gmm", "discrete"], required=True)
    parser.add_argument("--view", choices=["agent_wrist", "left_close_low_wrist"], required=True)
    parser.add_argument("--repo", type=Path, default=Path("/iris/u/jasonyan/repos/demonstration-information"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-dir", type=str, default="/iris/u/jasonyan/data/robomimic_outputs/policy_view_experiments")
    parser.add_argument("--num-epochs", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_name = "square_ph_bc_gmm_wrist.json" if args.algo == "gmm" else "square_ph_bc_discrete_wrist.json"
    base_path = args.repo / "configs" / "robomimic" / base_name
    with base_path.open() as f:
        cfg = json.load(f)

    exp_name = f"square_ph_bc_{args.algo}_{args.view}_200_seed1"
    cfg["experiment"]["name"] = exp_name
    cfg["train"]["data"] = DATASETS[args.view]
    cfg["train"]["output_dir"] = args.output_dir
    cfg["train"]["num_epochs"] = args.num_epochs
    cfg["observation"]["modalities"]["obs"]["rgb"] = RGB_KEYS[args.view]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cfg, indent=4) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
