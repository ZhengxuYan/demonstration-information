#!/usr/bin/env python3
"""Convert a DemInf score pickle into an episode_scores.csv file."""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-pkl", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", default="rollout")
    parser.add_argument("--view", default="agentview")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.score_pkl.open("rb") as f:
        data = pickle.load(f)
    ep_scores = data.get("ep_idx")
    if not isinstance(ep_scores, dict):
        raise ValueError(f"{args.score_pkl} is missing dict key 'ep_idx'")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "ep_idx", "score", "source", "view", "score_pickle"])
        writer.writeheader()
        for ep_idx, score in sorted(ep_scores.items(), key=lambda item: int(item[0])):
            writer.writerow(
                {
                    "dataset": args.dataset,
                    "ep_idx": int(ep_idx),
                    "score": float(score),
                    "source": args.source,
                    "view": args.view,
                    "score_pickle": str(args.score_pkl),
                }
            )
    print(args.output)


if __name__ == "__main__":
    main()
