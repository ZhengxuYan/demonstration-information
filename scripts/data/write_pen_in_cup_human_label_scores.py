#!/usr/bin/env python3
"""Convert pen-in-cup annotation CSV labels into OpenX score pkls.

The output pkls use the same structure as random drop scores:
{"ep_idx": {0: score0, 1: score1, ...}}

Higher score means the episode is kept by filter_by_scores at a given
percentile. Categorical ties are broken by a tiny deterministic random jitter
so 25/50/75 percentile filters have stable counts without changing the label
ordering.
"""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np


OBS_SCORES = {"full": 1.0, "partial": 0.0}
OPT_SCORES = {"better": 2.0, "okay": 1.0, "worse": 0.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-csv", type=Path, default=Path("pen_in_cup_annotations.csv"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--drop-percents", nargs="+", type=int, default=[25, 50, 75])
    parser.add_argument("--env", default="pen_in_cup")
    parser.add_argument("--tie-seed", type=int, default=1)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def read_scores(path: Path, column: str, mapping: dict[str, float], rng: np.random.Generator) -> dict[int, float]:
    scores: dict[int, float] = {}
    missing: list[int] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ep_idx = int(row["ep_idx"])
            label = (row.get(column) or "unlabeled").strip()
            if label not in mapping:
                missing.append(ep_idx)
                continue
            scores[ep_idx] = mapping[label] + float(rng.uniform(0.0, 1e-4))
    if missing:
        print(f"{column}: missing_or_unlabeled={sorted(missing)}")
    return scores


def kept_and_dropped(scores: dict[int, float], drop_percent: int) -> tuple[list[int], list[int]]:
    threshold = np.percentile(np.array(list(scores.values())), drop_percent)
    kept = sorted(int(k) for k, v in scores.items() if v >= threshold)
    dropped = sorted(int(k) for k, v in scores.items() if v < threshold)
    return kept, dropped


def write_score_set(
    output_root: Path,
    env: str,
    label_name: str,
    scores: dict[int, float],
    drop_percents: list[int],
) -> None:
    score_dir = output_root / env / label_name
    score_dir.mkdir(parents=True, exist_ok=True)
    score_pkl = score_dir / f"{label_name}_scores.pkl"
    with score_pkl.open("wb") as f:
        pickle.dump({"ep_idx": scores}, f)

    manifest = score_dir / f"{label_name}_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["label_name", "drop_percent", "num_kept", "num_dropped", "kept_ep_indices", "dropped_ep_indices", "score_pkl"],
        )
        writer.writeheader()
        for drop_percent in drop_percents:
            kept, dropped = kept_and_dropped(scores, drop_percent)
            writer.writerow(
                {
                    "label_name": label_name,
                    "drop_percent": drop_percent,
                    "num_kept": len(kept),
                    "num_dropped": len(dropped),
                    "kept_ep_indices": " ".join(map(str, kept)),
                    "dropped_ep_indices": " ".join(map(str, dropped)),
                    "score_pkl": str(score_pkl),
                }
            )
            print(f"{label_name} drop={drop_percent}: kept={len(kept)} dropped={len(dropped)}")
    print(f"{label_name}_score_pkl={score_pkl}")
    print(f"{label_name}_manifest={manifest}")


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.tie_seed)
    obs_scores = read_scores(args.annotations_csv, "observability", OBS_SCORES, rng)
    opt_scores = read_scores(args.annotations_csv, "optimality", OPT_SCORES, rng)

    if not args.allow_incomplete:
        obs_keys = set(obs_scores)
        opt_keys = set(opt_scores)
        if obs_keys != opt_keys:
            missing_obs = sorted(opt_keys - obs_keys)
            missing_opt = sorted(obs_keys - opt_keys)
            raise SystemExit(f"incomplete annotations: missing_observability={missing_obs}, missing_optimality={missing_opt}")

    if not obs_scores:
        raise SystemExit("no observability labels found")
    if not opt_scores:
        raise SystemExit("no optimality labels found")

    write_score_set(args.output_root, args.env, "observability", obs_scores, args.drop_percents)
    write_score_set(args.output_root, args.env, "optimality", opt_scores, args.drop_percents)


if __name__ == "__main__":
    main()
