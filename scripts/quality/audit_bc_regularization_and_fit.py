#!/usr/bin/env python3
"""Summarize robomimic BC regularization, logging, validation, and NLL fit."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def parse_key_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected KEY=PATH")
    key, path = value.split("=", 1)
    if not key:
        raise argparse.ArgumentTypeError("KEY cannot be empty")
    return key, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", type=parse_key_path, default=[], help="Run config as KEY=PATH")
    parser.add_argument("--score", action="append", type=parse_key_path, default=[], help="NLL pickle as KEY=PATH")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def summarize_config(path: Path) -> dict:
    with path.open() as f:
        cfg = json.load(f)
    logging = cfg["experiment"].get("logging", {})
    regularization = cfg["algo"]["optim_params"]["policy"].get("regularization", {})
    return {
        "path": str(path),
        "experiment_name": cfg["experiment"].get("name"),
        "optimizer": cfg["algo"]["optim_params"]["policy"].get("optimizer_type", "adam"),
        "l2_regularization": regularization.get("L2", 0.0),
        "validate": bool(cfg["experiment"].get("validate", False)),
        "log_wandb": bool(logging.get("log_wandb", False)),
        "wandb_project": logging.get("wandb_proj_name"),
        "train_filter": cfg["train"].get("hdf5_filter_key"),
        "valid_filter": cfg["train"].get("hdf5_validation_filter_key"),
        "dataset": cfg["train"].get("data"),
    }


def summarize_score(path: Path) -> dict:
    with path.open("rb") as f:
        scores = pickle.load(f)
    sample = np.asarray(scores["sample_score"], dtype=np.float64)
    demo = np.asarray(list(scores["ep_idx"].values()), dtype=np.float64)
    return {
        "path": str(path),
        "filter_key": scores.get("filter_key"),
        "num_transitions": int(sample.size),
        "num_demos": int(demo.size),
        "transition_mean": float(sample.mean()),
        "transition_std": float(sample.std()),
        "transition_min": float(sample.min()),
        "transition_max": float(sample.max()),
        "demo_mean": float(demo.mean()),
        "demo_std": float(demo.std()),
        "demo_min": float(demo.min()),
        "demo_max": float(demo.max()),
    }


def write_markdown(report: dict, path: Path) -> None:
    lines = ["# BC Regularization / Fit Audit", ""]
    lines.extend(
        [
            "## Configs",
            "",
            "| run | validate | wandb | train/valid filter | L2 | dataset |",
            "| --- | --- | --- | --- | ---: | --- |",
        ]
    )
    for key, item in sorted(report["configs"].items()):
        filters = f"{item['train_filter']} / {item['valid_filter']}"
        lines.append(
            f"| {key} | {item['validate']} | {item['log_wandb']} | {filters} | "
            f"{item['l2_regularization']} | `{item['dataset']}` |"
        )

    lines.extend(
        [
            "",
            "## NLL Scores",
            "",
            "| score | filter | demos | transitions | transition mean +/- std | demo mean +/- std | min / max |",
            "| --- | --- | ---: | ---: | --- | --- | --- |",
        ]
    )
    for key, item in sorted(report["scores"].items()):
        lines.append(
            f"| {key} | {item['filter_key']} | {item['num_demos']} | {item['num_transitions']} | "
            f"{item['transition_mean']:.4f} +/- {item['transition_std']:.4f} | "
            f"{item['demo_mean']:.4f} +/- {item['demo_std']:.4f} | "
            f"{item['demo_min']:.4f} / {item['demo_max']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    report = {
        "configs": {key: summarize_config(path) for key, path in args.config},
        "scores": {key: summarize_score(path) for key, path in args.score},
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_markdown(report, args.output_md)
    print(args.output_json)
    print(args.output_md)


if __name__ == "__main__":
    main()
