#!/usr/bin/env python3
"""Manage Wrench 0722 GMM visual-input control experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import manage_threading_action_dim_sequence as manager


ROOT = f"{manager.DATA}/wrench_on_hook_0722_pomdp"
STAGE = manager.Stage(
    key="w0722_visual_controls",
    dataset_tag="cartvel7d_fullframe320x180",
    action_dim=7,
    source_hdf5=f"{ROOT}/wrench_on_hook_0722_image_proprio_cartvel7d_fullframe_320x180.hdf5",
    hdf5=f"{ROOT}/wrench_on_hook_0722_image_proprio_cartvel7d_fullframe_320x180.hdf5",
    manifest=f"{ROOT}/wrench_on_hook_0722_fullframe_320x180_manifest.csv",
    run_prefix="wrench_on_hook_0722_visual_controls_gmm_minstd1em2",
    out_root=f"{manager.DATA}/robomimic_outputs/wrench_on_hook_0722_visual_controls_gmm_minstd1em2",
    config_root=f"{ROOT}/configs_visual_controls_gmm_minstd1em2",
    score_root=f"{ROOT}/scores_visual_controls_gmm_minstd1em2",
    report_root=f"{ROOT}/report_score_root_cause",
    title="Wrench 0722 score root-cause analysis",
    description="Controlled GMM comparison of proprio, single-camera, and dual-camera action densities.",
    prepare_mode="existing",
    labels_csv=f"{ROOT}/wrench_on_hook_0722_labels_full.csv",
    score_seed=20260725,
)

CONDITIONS = (
    "proprio_euler",
    "exterior_proprio_euler",
    "wrist_proprio_euler",
)


def submit_prepare(stage: manager.Stage) -> str:
    raise RuntimeError(f"Prepared HDF5 is required: {stage.hdf5}")


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
    if algo != "gmm" or condition not in CONDITIONS:
        raise ValueError(f"Unsupported control: {algo}/{condition}")
    start_tier = item.tier
    if item.attempts > 0 and item.state in manager.BAD and start_tier is not None:
        start_tier = min(start_tier + 1, len(manager.TIERS) - 1)
    item.tier = manager.choose_tier(start_tier)
    tier = manager.TIERS[item.tier]
    env = {
        "REPO": manager.REPO,
        "TASK_TAG": "wrench_on_hook_0722_visual_controls",
        "RUN_PREFIX": stage.run_prefix,
        "DATASET_TAG": stage.dataset_tag,
        "DATASET_HDF5": stage.hdf5,
        "OUT_ROOT": stage.out_root,
        "CONFIG_ROOT": stage.config_root,
        "ACTION_SOURCE": "image_proprio",
        "ACTION_TARGET": "single",
        "ACTION_NORMALIZATION": "none",
        "ALGOS": "gmm",
        "CONDITIONS": condition,
        "FOLD_TAG": fold_tag,
        "TRAIN_FILTER_KEY": train_key,
        "VALID_FILTER_KEY": valid_key,
        # Existing dual-camera GMM validation optima are at epochs 16-23.
        # One hundred epochs covers that range while keeping the nine-way
        # controlled comparison tractable.
        "NUM_EPOCHS": "100",
        "BATCH_SIZE": "32",
        "EPOCH_STEPS": "100",
        "VALIDATION_STEPS": "25",
        "SAVE_EVERY_N_EPOCHS": "50",
        "GAUSSIAN_MIN_STD": "0.01",
        "GMM_MODES": "5",
        "DISABLE_RGB_RANDOMIZER": "1",
        "RESUME": "1" if tier.preemptible or item.attempts > 0 else "0",
        "WANDB_PROJECT": "wrench-0722-score-root-cause",
    }
    name = f"wrc_{regime}_{condition}"
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
        "DATASET_HDF5": stage.hdf5,
        "MANIFEST": stage.manifest,
        "LABELS_CSV": stage.labels_csv,
        "CONTROL_OUT_ROOT": stage.out_root,
        "CONTROL_RUN_PREFIX": stage.run_prefix,
        "DATASET_TAG": stage.dataset_tag,
        "SCORE_ROOT": stage.score_root,
        "REGIME": regime,
        "FOLD_TAG": fold_tag,
        "FILTER_KEY": filter_key,
        "SEED": str(stage.score_seed),
    }
    name = f"wrc_{regime}_score"
    command = manager.env_command(
        env, f"{manager.REPO}/scripts/slurm/score_wrench_0722_visual_controls.sh"
    )
    return manager.gpu_sbatch(tier, name, command, "16:00:00", "64G", 8)


def submit_report(stage: manager.Stage) -> str:
    command = (
        "source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh && "
        "conda activate openx && "
        f"cd {manager.quote(manager.REPO)} && "
        "python scripts/quality/build_wrench_0722_root_cause_report.py "
        f"--dataset {manager.quote(stage.hdf5)} "
        f"--manifest {manager.quote(stage.manifest)} "
        f"--labels-csv {manager.quote(stage.labels_csv)} "
        f"--baseline-score-root {manager.quote(ROOT + '/snapshot_20260727_0700_cst/baseline_1e-4/best_validation/scores')} "
        f"--control-score-root {manager.quote(stage.score_root)} "
        f"--output {manager.quote(stage.report_root)}"
    )
    return manager.submit_cpu("wrc_report", command, "04:00:00")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state-file",
        type=Path,
        default=Path(f"{ROOT}/visual_controls_manager.json"),
    )
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--max-active-gpu", type=int, default=9)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--pending-migrate-seconds", type=int, default=1200)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manager.STAGES = (STAGE,)
    manager.ALGOS = ("gmm",)
    manager.CONDITIONS = CONDITIONS
    manager.submit_prepare = submit_prepare
    manager.submit_train = submit_train
    manager.submit_score = submit_score
    manager.submit_report = submit_report
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
