#!/usr/bin/env python3
"""Write deterministic random episode score pkls for OpenX filter_by_scores."""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-episodes", type=int, required=True)
    parser.add_argument("--drop-percents", nargs="+", type=int, default=[25, 50, 75])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--env", default="pen_in_cup")
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def kept_and_dropped(scores: dict[int, float], drop_percent: int) -> tuple[list[int], list[int]]:
    threshold = np.percentile(np.array(list(scores.values())), drop_percent)
    kept = sorted(int(k) for k, v in scores.items() if v >= threshold)
    dropped = sorted(int(k) for k, v in scores.items() if v < threshold)
    return kept, dropped


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    scores = {int(ep_idx): float(score) for ep_idx, score in enumerate(rng.random(args.num_episodes))}
    score_dir = args.output_root / args.env / "random" / f"seed-{args.seed}"
    score_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = score_dir / "random_drop_manifest.csv"

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["drop_percent", "num_kept", "num_dropped", "kept_ep_indices", "dropped_ep_indices", "score_pkl"],
        )
        writer.writeheader()
        for drop_percent in args.drop_percents:
            kept, dropped = kept_and_dropped(scores, drop_percent)
            pkl_path = score_dir / f"random_drop_{drop_percent:02d}_seed{args.seed}.pkl"
            with pkl_path.open("wb") as pf:
                pickle.dump({"ep_idx": scores}, pf)
            writer.writerow(
                {
                    "drop_percent": drop_percent,
                    "num_kept": len(kept),
                    "num_dropped": len(dropped),
                    "kept_ep_indices": " ".join(map(str, kept)),
                    "dropped_ep_indices": " ".join(map(str, dropped)),
                    "score_pkl": str(pkl_path),
                }
            )
            print(f"drop={drop_percent}: kept={len(kept)} dropped={len(dropped)} -> {pkl_path}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
