#!/usr/bin/env python3
"""Launch the staged Threading-D0 / Square density-recipe experiment funnel."""

from __future__ import annotations

import argparse
import itertools
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


REPO = "/iris/u/jasonyan/repos/demonstration-information"
DATA = "/iris/u/jasonyan/data"
ROOT = f"{DATA}/density_normalization_covariance_20260804"
SCORE_SLURM_SCRIPT = os.environ.get(
    "SCORE_SLURM_SCRIPT", f"{REPO}/scripts/slurm/score_threading_pomdp_6.sh"
)
EXCLUDED_GPU_NODES = (
    "iris1,iris2,iris3,iris4,iris-hp-z8,iliad1,iliad2,iliad3,iliad4,"
    "cocoflops-hgx-1,iliad-hgx-1,iris-hgx-1,iris-hgx-2,"
    "pasteur-hgx-1,pasteur-hgx-2,tiger-hgx-1,viscam-hgx-1,viscam-hgx-2,"
    "iliad7,jagupard28,jagupard29,jagupard30,jagupard31,"
    "tiger6,tiger7,tiger8,viscam3,viscam4,viscam6,viscam7,viscam8,viscam9"
)


@dataclass(frozen=True)
class Stage:
    key: str
    dataset_tag: str
    hdf5: str
    labels: str
    run_prefix: str


@dataclass(frozen=True)
class ResourceTier:
    partition: str
    account: str
    preemptible: bool = False
    constraint: str | None = None


STAGES = {
    "threading_d0_absolute": Stage(
        "threading_d0_absolute",
        "d0_final200_joint_absolute",
        f"{DATA}/threading_d0_final200_abs_delta_20260730/hdf5/image_final200_joint_absolute_fixedobs.hdf5",
        f"{DATA}/threading_d0_final200_abs_delta_20260730/manifests/joint_absolute.csv",
        "density_recipe_threading_d0_absolute",
    ),
    "square_7d": Stage(
        "square_7d",
        "square_mh_300_7d",
        f"{DATA}/pomdp_image_proprio_20260723/square_mh_300_image_proprio_7d.hdf5",
        f"{REPO}/observability_annotations.csv",
        "density_recipe_square_7d",
    ),
    "square_6d": Stage(
        "square_6d",
        "square_mh_300_6d",
        f"{DATA}/pomdp_image_proprio_20260723/square_mh_300_image_proprio_6d.hdf5",
        f"{REPO}/observability_annotations.csv",
        "density_recipe_square_6d",
    ),
}

RECIPES = {
    "identity_linear": ("identity", "none", "diag"),
    "zscore_linear": ("zscore", "none", "diag"),
    "robust_scale_linear": ("robust_scale", "none", "diag"),
    "zca_linear": ("zca", "none", "diag"),
    "minmax_tanh_legacy": ("minmax", "tanh", "diag"),
    # Controlled minmax factorial: isolate mean support from the variance floor.
    "minmax_linear_floor1e4": ("minmax", "none", "diag"),
    "minmax_tanh_floor1e2": ("minmax", "tanh", "diag"),
    "minmax_linear_floor1e2": ("minmax", "none", "diag"),
}

RECIPE_MIN_STD = {
    "minmax_tanh_legacy": "0.0001",
    "minmax_linear_floor1e4": "0.0001",
    "minmax_tanh_floor1e2": "0.01",
    "minmax_linear_floor1e2": "0.01",
}

REGIMES = {
    "normal": ("", "train", "valid", "score_all"),
    "fold0": ("fold0", "fold0_train", "fold0_valid", "fold0_score"),
    "fold1": ("fold1", "fold1_train", "fold1_valid", "fold1_score"),
}

STAGE2_FAMILIES = {
    "fullcov_gaussian": (("gaussian",), "full"),
    "fullcov_gmm": (("gmm",), "full"),
    "realnvp": (("flow",), "diag"),
}

RESOURCE_TIERS = {
    "iris_hi": ResourceTier("iris-hi", "iris"),
    "iliad": ResourceTier("iliad", "iliad"),
    "sc_loprio": ResourceTier(
        "sc-loprio", "iliad", preemptible=True, constraint="ampere"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=[
            "screen",
            "oof",
            "confirm",
            "covariance",
            "covariance_oof",
            "covariance_confirm",
            "square6",
            "proprio",
        ],
        required=True,
    )
    parser.add_argument("--stage", action="append", choices=sorted(STAGES))
    parser.add_argument(
        "--recipe",
        action="append",
        choices=sorted(RECIPES),
        help="Restrict screen phase to selected recipes.",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        help="Selected recipe as recipe:algo (required outside screen/covariance).",
    )
    parser.add_argument(
        "--winner-transform",
        choices=["identity", "minmax", "zscore", "robust_scale", "zca"],
        help="First-stage winning transform for all covariance phases.",
    )
    parser.add_argument(
        "--stage2-candidate",
        action="append",
        choices=sorted(STAGE2_FAMILIES),
        help="Selected stage-two family for OOF or three-seed confirmation.",
    )
    parser.add_argument("--submit", action="store_true")
    parser.add_argument(
        "--resource-tier",
        action="append",
        choices=sorted(RESOURCE_TIERS),
        help="Round-robin resource tiers. Defaults to iris-hi, iliad, then sc-loprio.",
    )
    parser.add_argument("--save-every-n-epochs", type=int, default=200)
    parser.add_argument("--latest-save-interval", type=int, default=200)
    parser.add_argument("--root", default=ROOT)
    parser.add_argument("--num-epochs", type=int, default=2000)
    parser.add_argument("--save-best-validation", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-min-epoch", type=int, default=0)
    parser.add_argument("--local-best-sync-interval", type=int, default=0)
    parser.add_argument(
        "--score-checkpoint-mode",
        action="append",
        choices=["best_validation", "latest_epoch"],
        help="Checkpoint selection to score. May be repeated; default is latest_epoch.",
    )
    return parser.parse_args()


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def run_or_print(command: list[str], submit: bool) -> str | None:
    print(shell_join(command), flush=True)
    if not submit:
        return None
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    job_id = result.stdout.strip().split(";", 1)[0]
    print(f"submitted={job_id}", flush=True)
    return job_id


def export_arg(values: dict[str, str]) -> str:
    return "ALL," + ",".join(f"{key}={value}" for key, value in values.items())


def submit_pair(
    stage: Stage,
    regime: str,
    recipe_name: str,
    transform: str,
    mean_squash: str,
    covariance: str,
    algorithms: tuple[str, ...],
    seed: int,
    submit: bool,
    resource: ResourceTier,
    root: str,
    num_epochs: int,
    save_best_validation: bool,
    early_stopping_patience: int,
    early_stopping_min_epoch: int,
    local_best_sync_interval: int,
    score_checkpoint_modes: tuple[str, ...],
    save_every_n_epochs: int,
    latest_save_interval: int,
    normalize_obs: bool = False,
    reuse_prior_variant: str | None = None,
) -> None:
    fold_tag, train_key, valid_key, score_key = REGIMES[regime]
    conditions = "image_proprio" if reuse_prior_variant else "image_proprio,action_prior"
    output_root = f"{root}/models/{stage.key}"
    config_root = f"{root}/configs/{stage.key}"
    score_root = f"{root}/scores/{stage.key}"
    env = {
        "REPO": REPO,
        "TASK_TAG": "density_recipe",
        "RUN_PREFIX": stage.run_prefix,
        "DATASET_TAG": stage.dataset_tag,
        "DATASET_HDF5": stage.hdf5,
        "OUT_ROOT": output_root,
        "CONFIG_ROOT": config_root,
        "ACTION_SOURCE": "image_proprio",
        "ACTION_TARGET": "single",
        "ACTION_NORMALIZATION": "none",
        "ACTION_TRANSFORM": transform,
        "MEAN_SQUASH": mean_squash,
        "COVARIANCE_TYPE": covariance,
        # sbatch --export reserves commas as variable separators. Use a colon
        # on the command line and let the training script decode it.
        "ALGOS": ":".join(algorithms),
        "CONDITIONS": conditions.replace(",", ":"),
        "FOLD_TAG": fold_tag,
        "TRAIN_FILTER_KEY": train_key,
        "VALID_FILTER_KEY": valid_key,
        "GAUSSIAN_MIN_STD": RECIPE_MIN_STD.get(recipe_name, "0.01"),
        "NUM_EPOCHS": str(num_epochs),
        "SAVE_EVERY_N_EPOCHS": str(save_every_n_epochs),
        "SAVE_BEST_VALIDATION": "1" if save_best_validation else "0",
        "EARLY_STOPPING_PATIENCE": str(early_stopping_patience),
        "EARLY_STOPPING_MIN_EPOCH": str(early_stopping_min_epoch),
        "LOCAL_BEST_SYNC_INTERVAL": str(local_best_sync_interval),
        "ROBOMIMIC_LATEST_SAVE_INTERVAL": str(latest_save_interval),
        "TRAIN_SEED": str(seed),
        "RUN_REGIME": regime,
        "VARIANT_TAG": recipe_name,
        "DISABLE_RGB_RANDOMIZER": "1",
        "RESUME": "1",
        "WANDB_PROJECT": "density-normalization-covariance",
    }
    if normalize_obs:
        env["HDF5_NORMALIZE_OBS"] = "1"
    total = len(algorithms) * (1 if reuse_prior_variant else 2)
    training = [
        "sbatch",
        "--parsable",
        "--partition",
        resource.partition,
        "--account",
        resource.account,
        "--exclude",
        EXCLUDED_GPU_NODES,
    ]
    if resource.preemptible:
        training.append("--requeue")
    if resource.constraint:
        training.extend(("--constraint", resource.constraint))
    training.extend([
        "--array",
        f"1-{total}%{total}",
        "--export",
        export_arg(env),
        f"{REPO}/scripts/slurm/train_pen_in_cup_density_models_array.sh",
    ])
    training_job = run_or_print(training, submit)
    for algo in algorithms:
        for checkpoint_mode in score_checkpoint_modes:
            score_env = {
                "REPO": REPO,
                "DATASET_TAG": stage.dataset_tag,
                "DATASET_HDF5": stage.hdf5,
                "OUT_ROOT": output_root,
                "SCORE_ROOT": score_root,
                "RUN_PREFIX": stage.run_prefix,
                "ALGO": algo,
                "REGIME": regime,
                "FOLD_TAG": fold_tag,
                "FILTER_KEY": score_key,
                "ACTION_SOURCE": "image_proprio",
                "CONDITIONAL_CONDITION": "image_proprio",
                "VARIANT_TAG": recipe_name,
                "PRIOR_VARIANT_TAG": reuse_prior_variant or recipe_name,
                "TRAIN_SEED": str(seed),
                "M": "16",
                "K": "512",
                "SCORE_PY": os.environ.get(
                    "SCORE_PY", f"{REPO}/scripts/quality/score_threading_pomdp_6.py"
                ),
                "CKPT_MODE": checkpoint_mode,
                "SCORE_MODE_TAG": (
                    "exact_best_validation"
                    if checkpoint_mode == "best_validation"
                    else "early_stop_final"
                ),
            }
            scoring = [
                "sbatch",
                "--parsable",
                "--partition",
                resource.partition,
                "--account",
                resource.account,
                "--exclude",
                EXCLUDED_GPU_NODES,
            ]
            if resource.preemptible:
                scoring.append("--requeue")
            if resource.constraint:
                scoring.extend(("--constraint", resource.constraint))
            if training_job:
                scoring.extend(("--dependency", f"afterok:{training_job}"))
            scoring.extend(
                (
                    "--export",
                    export_arg(score_env),
                    SCORE_SLURM_SCRIPT,
                )
            )
            run_or_print(scoring, submit)


def candidates(values: list[str] | None) -> list[tuple[str, str]]:
    result = []
    for value in values or []:
        recipe, separator, algo = value.partition(":")
        if not separator or recipe not in RECIPES or algo not in ("gaussian", "gmm"):
            raise ValueError(f"Invalid candidate {value!r}; expected recipe:gaussian|gmm")
        result.append((recipe, algo))
    return result


def main() -> None:
    args = parse_args()
    if args.stage:
        stage_names = args.stage
    elif args.phase == "square6":
        stage_names = ["square_6d"]
    else:
        stage_names = ["threading_d0_absolute", "square_7d"]
    selected = candidates(args.candidate)
    tier_names = args.resource_tier or ["iris_hi", "iliad", "sc_loprio", "sc_loprio"]
    resource_cycle = itertools.cycle(RESOURCE_TIERS[name] for name in tier_names)

    def launch(*launch_args, **launch_kwargs):
        submit_pair(
            *launch_args,
            **launch_kwargs,
            submit=args.submit,
            resource=next(resource_cycle),
            root=args.root,
            num_epochs=args.num_epochs,
            save_best_validation=args.save_best_validation,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_epoch=args.early_stopping_min_epoch,
            local_best_sync_interval=args.local_best_sync_interval,
            score_checkpoint_modes=tuple(
                args.score_checkpoint_mode or ["latest_epoch"]
            ),
            save_every_n_epochs=args.save_every_n_epochs,
            latest_save_interval=args.latest_save_interval,
        )
    if args.phase in ("oof", "confirm", "square6", "proprio") and not selected:
        raise ValueError(f"--phase {args.phase} requires at least one --candidate")
    if args.phase.startswith("covariance") and args.winner_transform is None:
        raise ValueError(f"--phase {args.phase} requires --winner-transform")
    if args.phase in ("covariance_oof", "covariance_confirm") and not args.stage2_candidate:
        raise ValueError(f"--phase {args.phase} requires --stage2-candidate")

    for stage_name in stage_names:
        stage = STAGES[stage_name]
        if args.phase == "screen":
            for recipe, (transform, squash, covariance) in RECIPES.items():
                if args.recipe and recipe not in args.recipe:
                    continue
                launch(stage, "normal", recipe, transform, squash, covariance, ("gaussian", "gmm"), 1)
        elif args.phase == "oof":
            for recipe, algo in selected:
                transform, squash, covariance = RECIPES[recipe]
                for regime in ("fold0", "fold1"):
                    launch(stage, regime, recipe, transform, squash, covariance, (algo,), 1)
        elif args.phase == "confirm":
            for recipe, algo in selected:
                transform, squash, covariance = RECIPES[recipe]
                for seed in (1, 2, 3):
                    for regime in REGIMES:
                        launch(stage, regime, recipe, transform, squash, covariance, (algo,), seed)
        elif args.phase == "covariance":
            for recipe, (algos, covariance) in STAGE2_FAMILIES.items():
                launch(stage, "normal", recipe, args.winner_transform, "none", covariance, algos, 1)
        elif args.phase == "covariance_oof":
            for recipe in args.stage2_candidate:
                algos, covariance = STAGE2_FAMILIES[recipe]
                for regime in ("fold0", "fold1"):
                    launch(stage, regime, recipe, args.winner_transform, "none", covariance, algos, 1)
        elif args.phase == "covariance_confirm":
            for recipe in args.stage2_candidate:
                algos, covariance = STAGE2_FAMILIES[recipe]
                for seed in (1, 2, 3):
                    for regime in REGIMES:
                        launch(stage, regime, recipe, args.winner_transform, "none", covariance, algos, seed)
        elif args.phase == "square6":
            if stage.key != "square_6d":
                continue
            for recipe, algo in selected:
                transform, squash, covariance = RECIPES[recipe]
                for regime in REGIMES:
                    launch(stage, regime, recipe, transform, squash, covariance, (algo,), 1)
        elif args.phase == "proprio":
            for recipe, algo in selected:
                transform, squash, covariance = RECIPES[recipe]
                for regime in REGIMES:
                    launch(
                        stage,
                        regime,
                        recipe + "_proprio_zscore",
                        transform,
                        squash,
                        covariance,
                        (algo,),
                        1,
                        normalize_obs=True,
                        reuse_prior_variant=recipe,
                    )


if __name__ == "__main__":
    main()
