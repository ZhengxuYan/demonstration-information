#!/usr/bin/env python3
"""Export DemInf episode scores from a pickle to a sorted CSV."""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-pkl", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=None)
    parser.add_argument(
        "--score-key",
        default="ep_idx",
        help="Pickle key containing episode-index to score mapping. Defaults to ep_idx.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.score_pkl.open("rb") as f:
        scores = pickle.load(f)

    if args.score_key not in scores:
        raise KeyError(f"{args.score_pkl} missing {args.score_key}; available keys: {sorted(scores.keys())}")

    rows = [(int(ep_idx), float(score)) for ep_idx, score in scores[args.score_key].items()]
    rows.sort(key=lambda item: item[1], reverse=True)

    if args.expected_episodes is not None and len(rows) != args.expected_episodes:
        raise AssertionError(f"expected {args.expected_episodes} scored episodes, got {len(rows)}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "ep_idx", "demo_key", "information_score"])
        for rank, (ep_idx, score) in enumerate(rows, start=1):
            writer.writerow([rank, ep_idx, f"demo_{ep_idx}", score])

    print(f"wrote {args.output_csv} rows={len(rows)}")


if __name__ == "__main__":
    main()
