#!/usr/bin/env python3
"""Prepare the six selected final-density-score Threading filtered-BC sweeps.

This script only reads small episode-level score CSVs and BC JSON templates. It
does not copy or modify the source HDF5. Compute jobs add the generated masks to
their node-local staged HDF5 copy.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path


EXPECTED_DEMOS = 200
FILTER_PERCENTS = (25, 50, 75)  # percentage dropped; highest scores retained
SEEDS = (1, 2, 3)

CONDITIONS = (
    {
        "condition": "gaussian_zscore_score2",
        "algo": "gaussian",
        "recipe": "zscore_linear",
        "score_number": 2,
        "score_field": "data_mi_learned_marginal",
    },
    {
        "condition": "gaussian_robust_score2",
        "algo": "gaussian",
        "recipe": "robust_scale_linear",
        "score_number": 2,
        "score_field": "data_mi_learned_marginal",
    },
    {
        "condition": "gaussian_zscore_score6",
        "algo": "gaussian",
        "recipe": "zscore_linear",
        "score_number": 6,
        "score_field": "model_mi_reference_prior",
    },
    {
        "condition": "gaussian_robust_score6",
        "algo": "gaussian",
        "recipe": "robust_scale_linear",
        "score_number": 6,
        "score_field": "model_mi_reference_prior",
    },
    {
        "condition": "gaussian_identity_score6",
        "algo": "gaussian",
        "recipe": "identity_linear",
        "score_number": 6,
        "score_field": "model_mi_reference_prior",
    },
    {
        "condition": "gmm_robust_score6",
        "algo": "gmm",
        "recipe": "robust_scale_linear",
        "score_number": 6,
        "score_field": "model_mi_reference_prior",
    },
)


def parse_args() -> argparse.Namespace:
    user = os.environ.get("USER", "jasonyan")
    data_root = Path(f"/iris/u/{user}/data")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=data_root / "threading_d0_final200_abs_delta_20260730",
    )
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=data_root / "threading_d0_final_score_filtered_bc_results_20260808",
    )
    return parser.parse_args()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(text)
    os.replace(temporary, path)


def read_scores(path: Path, field: str) -> list[tuple[int, float]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_DEMOS:
        raise ValueError(f"{path}: expected {EXPECTED_DEMOS} rows, found {len(rows)}")
    values = []
    for row in rows:
        ep_idx = int(row["ep_idx"])
        score = float(row[field])
        if not math.isfinite(score):
            raise ValueError(f"{path}: non-finite {field} at ep_idx={ep_idx}")
        values.append((ep_idx, score))
    indices = [item[0] for item in values]
    if sorted(indices) != list(range(EXPECTED_DEMOS)):
        raise ValueError(f"{path}: ep_idx must cover 0..{EXPECTED_DEMOS - 1}")
    return values


def mask_name(condition: str, filter_percent: int) -> str:
    return f"density_final_{condition}_filter{filter_percent}pct"


def configure(base: dict, run_name: str, filter_key: str, seed: int) -> dict:
    config = copy.deepcopy(base)
    config["experiment"]["name"] = run_name
    config["experiment"]["validate"] = False
    config["experiment"]["render_video"] = False
    config["experiment"]["logging"]["terminal_output_to_txt"] = False
    config["experiment"]["logging"]["log_tb"] = False
    config["experiment"]["logging"]["log_wandb"] = False
    config["experiment"]["save"] = {
        "enabled": False,
        "every_n_seconds": None,
        "every_n_epochs": None,
        "epochs": [],
        "on_best_validation": False,
        "on_best_rollout_return": False,
        "on_best_rollout_success_rate": False,
    }
    config["experiment"]["rollout"].update(
        {
            "enabled": True,
            "n": 100,
            "horizon": 800,
            "rate": 50,
            "warmstart": 100,
            "terminate_on_success": True,
        }
    )
    config["train"]["num_epochs"] = 600
    config["train"]["seed"] = seed
    config["train"]["hdf5_filter_key"] = filter_key
    config["train"]["hdf5_validation_filter_key"] = None
    # The worker replaces both paths with node-local paths before training.
    config["train"]["data"] = "/tmp/threading_d0_filtered_bc_dataset.hdf5"
    config["train"]["output_dir"] = "/tmp/threading_d0_filtered_bc_outputs"
    return config


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.result_root.mkdir(parents=True, exist_ok=True)

    selections: dict[str, list[str]] = {}
    selection_rows = []
    manifest_rows = []
    task_id = 0

    for condition in CONDITIONS:
        score_path = (
            args.score_root
            / condition["algo"]
            / condition["recipe"]
            / "threading_pomdp_8_scores.csv"
        )
        values = read_scores(score_path, condition["score_field"])
        ranked = sorted(values, key=lambda item: (-item[1], item[0]))
        for filter_percent in FILTER_PERCENTS:
            retained_count = EXPECTED_DEMOS * (100 - filter_percent) // 100
            retained = sorted(item[0] for item in ranked[:retained_count])
            filter_key = mask_name(condition["condition"], filter_percent)
            selections[filter_key] = [f"demo_{ep_idx}" for ep_idx in retained]
            selection_rows.append(
                {
                    **condition,
                    "filter_percent": filter_percent,
                    "retained_count": retained_count,
                    "filter_key": filter_key,
                    "score_csv": str(score_path),
                    "retained_ep_indices": retained,
                }
            )

            for seed in SEEDS:
                task_id += 1
                base_path = (
                    args.source_root
                    / "bc_configs/joint_absolute/full/gmm"
                    / f"seed{seed}.json"
                )
                base = json.loads(base_path.read_text())
                run_name = (
                    "bc_gmm_threading_d0_final_"
                    f"{condition['condition']}_filter{filter_percent}pct_seed{seed}"
                )
                config = configure(base, run_name, filter_key, seed)
                config_path = (
                    args.output_root
                    / "configs"
                    / condition["condition"]
                    / f"filter{filter_percent}pct"
                    / f"seed{seed}.json"
                )
                atomic_write_text(config_path, json.dumps(config, indent=2) + "\n")
                manifest_rows.append(
                    {
                        "task_id": task_id,
                        **condition,
                        "filter_percent": filter_percent,
                        "retained_count": retained_count,
                        "seed": seed,
                        "filter_key": filter_key,
                        "config_path": str(config_path),
                        "run_name": run_name,
                        "result_path": str(args.result_root / f"task_{task_id:02d}.json"),
                    }
                )

    if task_id != 54 or len(selections) != 18:
        raise AssertionError((task_id, len(selections)))

    atomic_write_text(
        args.output_root / "score_selections.json",
        json.dumps(
            {
                "schema_version": 1,
                "expected_demos": EXPECTED_DEMOS,
                "direction": "drop_lowest_scores_retain_highest_scores",
                "selections": selections,
                "details": selection_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    manifest_path = args.output_root / "config_manifest.csv"
    fields = list(manifest_rows[0])
    temporary = manifest_path.with_name(manifest_path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)
    os.replace(temporary, manifest_path)
    atomic_write_text(
        args.output_root / "PREP_COMPLETE.json",
        json.dumps(
            {
                "conditions": len(CONDITIONS),
                "masks": len(selections),
                "runs": len(manifest_rows),
                "filter_semantics": {
                    "25": "drop 50, retain top 150",
                    "50": "drop 100, retain top 100",
                    "75": "drop 150, retain top 50",
                },
                "checkpoints_saved": False,
                "rollout_episodes": 100,
                "final_epoch": 600,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    print(json.dumps({"runs": 54, "masks": 18, "manifest": str(manifest_path)}))


if __name__ == "__main__":
    main()
