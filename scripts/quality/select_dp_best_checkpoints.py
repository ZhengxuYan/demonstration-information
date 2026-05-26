#!/usr/bin/env python3
"""Select best Diffusion Policy checkpoints by success-rate metric in filename."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


CKPT_RE = re.compile(r"epoch=(?P<epoch>\d+)-test_mean_score=(?P<score>[-+0-9.]+)\.ckpt$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--glob", default="seed_*/checkpoints/*.ckpt")
    return parser.parse_args()


def parse_checkpoint(path: Path) -> tuple[int, float] | None:
    match = CKPT_RE.search(path.name)
    if match is None:
        return None
    return int(match.group("epoch")), float(match.group("score"))


def seed_name(path: Path, run_root: Path) -> str:
    rel = path.relative_to(run_root)
    return rel.parts[0]


def main() -> None:
    args = parse_args()
    rows_by_seed = {}
    for ckpt in sorted(args.run_root.glob(args.glob)):
        parsed = parse_checkpoint(ckpt)
        if parsed is None:
            continue
        epoch, score = parsed
        seed = seed_name(ckpt, args.run_root)
        row = {
            "seed_name": seed,
            "epoch": epoch,
            "test_mean_score": score,
            "checkpoint": str(ckpt),
        }
        old = rows_by_seed.get(seed)
        if old is None or (score, epoch) > (old["test_mean_score"], old["epoch"]):
            rows_by_seed[seed] = row

    rows = [rows_by_seed[key] for key in sorted(rows_by_seed)]
    if not rows:
        raise FileNotFoundError(f"No checkpoint files matching {args.run_root / args.glob}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed_name", "epoch", "test_mean_score", "checkpoint"])
        writer.writeheader()
        writer.writerows(rows)

    print(args.output_csv)
    for row in rows:
        print(f"{row['seed_name']} epoch={row['epoch']} score={row['test_mean_score']:.3f} {row['checkpoint']}")


if __name__ == "__main__":
    main()
