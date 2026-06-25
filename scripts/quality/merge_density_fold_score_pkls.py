#!/usr/bin/env python3
"""Merge held-out score PKLs from multiple density cross-validation folds."""

from __future__ import annotations

import argparse
import csv
import pickle
from collections import OrderedDict
from pathlib import Path

import numpy as np


ARRAY_KEYS = (
    "sample_score",
    "sample_nll",
    "sample_log_prob",
    "sample_entropy",
    "sample_ep_idx",
    "sample_step_idx",
    "sample_demo_key",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True, help="Fold score PKL. Repeat per fold.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, default=None)
    return parser.parse_args()


def ordered_episode_means(ep: np.ndarray, values: np.ndarray) -> OrderedDict[int, float]:
    out = OrderedDict()
    for ep_idx in sorted(np.unique(ep).astype(int).tolist()):
        mask = ep == ep_idx
        out[int(ep_idx)] = float(np.asarray(values[mask], dtype=np.float64).mean())
    return out


def merge(inputs: list[Path]) -> dict:
    bundles = []
    for path in inputs:
        with path.open("rb") as f:
            bundles.append(pickle.load(f))
    merged = dict(bundles[0])
    for key in ARRAY_KEYS:
        merged[key] = np.concatenate([np.asarray(bundle[key]) for bundle in bundles], axis=0)

    order = np.lexsort((merged["sample_step_idx"], merged["sample_ep_idx"]))
    for key in ARRAY_KEYS:
        merged[key] = merged[key][order]

    sample_ids = list(zip(merged["sample_ep_idx"].astype(int), merged["sample_step_idx"].astype(int)))
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Fold score files contain overlapping (ep_idx, step_idx) samples")

    merged["ep_idx"] = ordered_episode_means(merged["sample_ep_idx"], merged["sample_nll"])
    merged["ep_idx_nll"] = merged["ep_idx"]
    merged["ep_idx_entropy"] = ordered_episode_means(merged["sample_ep_idx"], merged["sample_entropy"])
    merged["filter_key"] = "+".join(str(bundle.get("filter_key", "")) for bundle in bundles)
    merged["fold_score_files"] = [str(path) for path in inputs]
    return merged


def write_csv(scores: dict, path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ep_idx", "mean_nll", "mean_entropy"])
        for ep_idx, mean_nll in scores["ep_idx"].items():
            writer.writerow([ep_idx, mean_nll, scores["ep_idx_entropy"][ep_idx]])


def main() -> None:
    args = parse_args()
    scores = merge(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        pickle.dump(scores, f)
    if args.csv_output is not None:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        write_csv(scores, args.csv_output)
    print(f"wrote {args.output}")
    print(f"episodes={len(scores['ep_idx'])} samples={len(scores['sample_nll'])}")


if __name__ == "__main__":
    main()
