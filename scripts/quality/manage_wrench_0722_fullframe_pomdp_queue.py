#!/usr/bin/env python3
"""Manage corrected Wrench 0722 full-frame image+proprio density runs."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import manage_threading_action_dim_sequence as manager


ROOT = f"{manager.DATA}/wrench_on_hook_0722_pomdp"
STAGE = manager.Stage(
    key="w0722_fullframe",
    dataset_tag="cartvel7d_fullframe320x180",
    action_dim=7,
    source_hdf5="/scr/tiangao/wrench_on_hook_0722",
    hdf5=f"{ROOT}/wrench_on_hook_0722_image_proprio_cartvel7d_fullframe_320x180.hdf5",
    manifest=f"{ROOT}/wrench_on_hook_0722_fullframe_320x180_manifest.csv",
    run_prefix="wrench_on_hook_0722_pomdp_fullframe_320x180",
    out_root=f"{manager.DATA}/robomimic_outputs/wrench_on_hook_0722_pomdp_fullframe_320x180",
    config_root=f"{ROOT}/configs_fullframe_320x180",
    score_root=f"{ROOT}/scores_fullframe_320x180",
    report_root=f"{ROOT}/report_fullframe_320x180",
    title="Wrench-on-Hook 0722 POMDP Scores - Full-frame 320x180",
    description=(
        "Conditional models use complete 320x180 exterior and wrist frames "
        "without spatial cropping, plus end-effector proprio. Actions are raw "
        "7D Cartesian and gripper velocities; ordinal labels are 1 < 2 < 3."
    ),
    prepare_mode="wrench_0722_fullframe",
    labels_csv="/scr/tiangao/wrench_on_hook_0722_labels.csv",
    score_seed=20260725,
)


def submit_prepare(stage: manager.Stage) -> str:
    return manager.run(
        [
            "sbatch",
            "--parsable",
            "--account=iris",
            "--partition=iris-hi",
            "--nodelist=iris8",
            "--time=12:00:00",
            "--job-name=pip_w0722_fullframe_h5",
            "--export=ALL,"
            f"REPO={manager.REPO},"
            f"OUTPUT_HDF5={stage.hdf5},"
            f"MANIFEST={stage.manifest},"
            f"OUTPUT_ROOT={ROOT}",
            f"{manager.REPO}/scripts/slurm/prepare_wrench_0722_pomdp_hdf5.sh",
        ]
    )


def submit_train(
    stage: manager.Stage,
    regime: str,
    fold_tag: str,
    train_key: str,
    valid_key: str,
    algo: str,
    condition: str,
    item: manager.JobRecord,
) -> str:
    start_tier = item.tier
    if item.attempts > 0 and item.state in manager.BAD and start_tier is not None:
        start_tier = min(start_tier + 1, len(manager.TIERS) - 1)
    item.tier = manager.choose_tier(start_tier)
    tier = manager.TIERS[item.tier]
    env = {
        "REPO": manager.REPO,
        "TASK_TAG": "wrench_on_hook_0722_fullframe",
        "RUN_PREFIX": stage.run_prefix,
        "DATASET_TAG": stage.dataset_tag,
        "DATASET_HDF5": stage.hdf5,
        "OUT_ROOT": stage.out_root,
        "CONFIG_ROOT": stage.config_root,
        "ACTION_SOURCE": "image_proprio",
        "ACTION_TARGET": "single",
        "ACTION_NORMALIZATION": "none",
        "ALGOS": algo,
        "CONDITIONS": condition,
        "FOLD_TAG": fold_tag,
        "TRAIN_FILTER_KEY": train_key,
        "VALID_FILTER_KEY": valid_key,
        "NUM_EPOCHS": "2000",
        "BATCH_SIZE": "32",
        "EPOCH_STEPS": "100",
        "VALIDATION_STEPS": "25",
        "SAVE_EVERY_N_EPOCHS": "50",
        "GAUSSIAN_MIN_STD": "0.0001",
        "GMM_MODES": "5",
        "DISABLE_RGB_RANDOMIZER": "1",
        "RESUME": "1" if tier.preemptible or item.attempts > 0 else "0",
        "WANDB_PROJECT": "wrench-0722-pomdp-fullframe-density",
    }
    name = f"wff_{regime}_{algo}_{condition}"
    command = manager.env_command(
        env, f"{manager.REPO}/scripts/slurm/train_pen_in_cup_density_models_array.sh"
    )
    return manager.gpu_sbatch(tier, name, command, "24:00:00", "64G", 10)


def submit_score(
    stage: manager.Stage,
    regime: str,
    fold_tag: str,
    filter_key: str,
    algo: str,
    item: manager.JobRecord,
) -> str:
    start_tier = item.tier
    if item.attempts > 0 and item.state in manager.BAD and start_tier is not None:
        start_tier = min(start_tier + 1, len(manager.TIERS) - 1)
    item.tier = manager.choose_tier(start_tier)
    tier = manager.TIERS[item.tier]
    env = {
        "REPO": manager.REPO,
        "DATASET_TAG": stage.dataset_tag,
        "DATASET_HDF5": stage.hdf5,
        "OUT_ROOT": stage.out_root,
        "SCORE_ROOT": stage.score_root,
        "RUN_PREFIX": stage.run_prefix,
        "ALGO": algo,
        "REGIME": regime,
        "FOLD_TAG": fold_tag,
        "FILTER_KEY": filter_key,
        "ACTION_SOURCE": "image_proprio",
        "CONDITIONAL_CONDITION": "image_proprio_euler",
        "M": "16",
        "K": "512",
        "SEED": str(stage.score_seed),
    }
    name = f"wff_{regime}_{algo}_score"
    command = manager.env_command(
        env, f"{manager.REPO}/scripts/slurm/score_threading_pomdp_6.sh"
    )
    return manager.gpu_sbatch(tier, name, command, "12:00:00", "64G", 8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state-file",
        type=Path,
        default=Path(f"{ROOT}/fullframe_320x180_manager.json"),
    )
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--max-active-gpu", type=int, default=12)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--pending-migrate-seconds", type=int, default=1200)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manager.STAGES = (STAGE,)
    manager.CONDITIONS = ("image_proprio_euler", "action_prior")
    manager.submit_prepare = submit_prepare
    manager.submit_train = submit_train
    manager.submit_score = submit_score

    sys.argv = [
        sys.argv[0],
        "--state-file",
        str(args.state_file),
        "--poll-seconds",
        str(args.poll_seconds),
        "--max-active-gpu",
        str(args.max_active_gpu),
        "--max-attempts",
        str(args.max_attempts),
        "--pending-migrate-seconds",
        str(args.pending_migrate_seconds),
        "--stages",
        STAGE.key,
    ]
    if args.once:
        sys.argv.append("--once")
    manager.main()


if __name__ == "__main__":
    main()
