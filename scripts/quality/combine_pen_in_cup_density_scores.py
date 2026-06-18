#!/usr/bin/env python3
"""Combine robomimic density-model NLL / entropy outputs into Tian's 8 scores."""

from __future__ import annotations

import argparse
import csv
import pickle
from collections import OrderedDict
from pathlib import Path

import numpy as np


CONDITIONS = ("image_state", "image", "state", "action_prior")

SCORE_DEFS = [
    ("nll_image_state_minus_state", "sample_nll", "image_state", "state"),
    ("nll_image_minus_image_state", "sample_nll", "image", "image_state"),
    ("nll_image_state_minus_action_prior", "sample_nll", "image_state", "action_prior"),
    ("nll_image_minus_action_prior", "sample_nll", "image", "action_prior"),
    ("entropy_image_state_minus_state", "sample_entropy", "image_state", "state"),
    ("entropy_image_minus_image_state", "sample_entropy", "image", "image_state"),
    ("entropy_image_state", "sample_entropy", "image_state", None),
    ("entropy_image", "sample_entropy", "image", None),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-root", type=Path, required=True, help="Directory containing <condition>.pkl files.")
    parser.add_argument("--output-pkl", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--prefix", default="", help="Optional prefix added to score column names.")
    return parser.parse_args()


def load_scores(root: Path) -> dict[str, dict]:
    out = {}
    for condition in CONDITIONS:
        path = root / f"{condition}.pkl"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("rb") as f:
            out[condition] = pickle.load(f)
    return out


def sample_map(bundle: dict, key: str) -> dict[tuple[int, int], float]:
    ep = np.asarray(bundle["sample_ep_idx"], dtype=np.int64)
    step = np.asarray(bundle["sample_step_idx"], dtype=np.int64)
    values = np.asarray(bundle[key], dtype=np.float64)
    if not (len(ep) == len(step) == len(values)):
        raise ValueError(f"Length mismatch for key={key}")
    return {(int(e), int(s)): float(v) for e, s, v in zip(ep, step, values)}


def aggregate(values_by_sample: dict[tuple[int, int], float]) -> OrderedDict[int, float]:
    grouped: dict[int, list[float]] = {}
    for (ep_idx, _), value in values_by_sample.items():
        grouped.setdefault(ep_idx, []).append(value)
    out = OrderedDict()
    for ep_idx in sorted(grouped):
        out[int(ep_idx)] = float(np.mean(grouped[ep_idx]))
    return out


def combine_pair(left: dict[tuple[int, int], float], right: dict[tuple[int, int], float] | None) -> OrderedDict[int, float]:
    if right is None:
        return aggregate(left)
    common = sorted(set(left) & set(right))
    if not common:
        raise ValueError("No overlapping samples between score files")
    return aggregate({key: left[key] - right[key] for key in common})


def main() -> None:
    args = parse_args()
    bundles = load_scores(args.score_root)

    sample_values = {
        condition: {
            "sample_nll": sample_map(bundle, "sample_nll"),
            "sample_entropy": sample_map(bundle, "sample_entropy"),
        }
        for condition, bundle in bundles.items()
    }

    combined: dict[str, OrderedDict[int, float]] = {}
    for name, sample_key, left_condition, right_condition in SCORE_DEFS:
        score_name = f"{args.prefix}{name}" if args.prefix else name
        combined[score_name] = combine_pair(
            sample_values[left_condition][sample_key],
            None if right_condition is None else sample_values[right_condition][sample_key],
        )

    args.output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_pkl.open("wb") as f:
        pickle.dump({"ep_idx": combined}, f)

    ep_indices = sorted(set().union(*(scores.keys() for scores in combined.values())))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["ep_idx"] + list(combined.keys())
        writer.writerow(header)
        for ep_idx in ep_indices:
            writer.writerow([ep_idx] + [combined[name].get(ep_idx, "") for name in combined])

    print(f"wrote {args.output_pkl}")
    print(f"wrote {args.output_csv}")
    print(f"scores={len(combined)} episodes={len(ep_indices)}")


if __name__ == "__main__":
    main()
