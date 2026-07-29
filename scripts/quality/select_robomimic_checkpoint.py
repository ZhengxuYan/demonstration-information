#!/usr/bin/env python3
"""Select a robomimic checkpoint from an experiment directory."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["best_validation", "latest_epoch", "last"],
        default="best_validation",
    )
    parser.add_argument(
        "--max-epoch",
        type=int,
        help="Ignore model_epoch checkpoints above this epoch.",
    )
    return parser.parse_args()


def epoch(path: Path) -> int:
    match = re.search(r"model_epoch_(\d+)", path.name)
    return int(match.group(1)) if match else -1


def checkpoints(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("*/models/*.pth"))


def main() -> None:
    args = parse_args()
    ckpts = checkpoints(args.run_dir)
    if args.max_epoch is not None:
        ckpts = [
            path
            for path in ckpts
            if epoch(path) < 0 or epoch(path) <= args.max_epoch
        ]
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints under {args.run_dir}")

    if args.mode == "last":
        last = sorted(args.run_dir.glob("*/models/last.pth")) or sorted(args.run_dir.glob("*/last.pth"))
        if not last:
            raise FileNotFoundError(f"No last.pth under {args.run_dir}")
        print(last[-1])
        return

    if args.mode == "best_validation":
        best = [
            p
            for p in ckpts
            if "best_validation" in p.name
            or "valid_best" in p.name
            or "best_valid" in p.name
            or "model_best" in p.name
        ]
        if best:
            best.sort(key=lambda p: (p.stat().st_mtime, epoch(p), str(p)))
            print(best[-1])
            return

    epoch_ckpts = [p for p in ckpts if epoch(p) >= 0]
    if not epoch_ckpts:
        raise FileNotFoundError(f"No epoch checkpoints under {args.run_dir}")
    epoch_ckpts.sort(key=lambda p: (epoch(p), p.stat().st_mtime, str(p)))
    print(epoch_ckpts[-1])


if __name__ == "__main__":
    main()
