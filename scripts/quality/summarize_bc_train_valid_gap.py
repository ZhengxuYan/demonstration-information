#!/usr/bin/env python3
"""Summarize robomimic BC train/validation loss gaps from log.txt files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


EPOCH_RE = re.compile(r"^(Train|Validation) Epoch (\d+)$")


def parse_log(path: Path) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    lines = path.read_text(errors="replace").splitlines()
    i = 0
    while i < len(lines):
        match = EPOCH_RE.match(lines[i].strip())
        if match is None:
            i += 1
            continue

        split, epoch_s = match.groups()
        epoch = int(epoch_s)
        i += 1
        block = []
        brace_depth = 0
        started = False
        while i < len(lines):
            line = lines[i]
            if "{" in line:
                started = True
            if started:
                block.append(line)
                brace_depth += line.count("{") - line.count("}")
                if brace_depth == 0:
                    break
            i += 1

        if block:
            try:
                metrics = json.loads("\n".join(block))
            except json.JSONDecodeError:
                i += 1
                continue
            prefix = "train" if split == "Train" else "valid"
            row = rows.setdefault(epoch, {})
            for key in ("Loss", "Log_Likelihood", "Policy_Grad_Norms"):
                if key in metrics:
                    row[f"{prefix}_{key}"] = float(metrics[key])
        i += 1
    return rows


def summarize(path: Path) -> dict[str, float | int | str]:
    rows = parse_log(path)
    paired = {
        epoch: row
        for epoch, row in rows.items()
        if "train_Loss" in row and "valid_Loss" in row
    }
    if not paired:
        raise ValueError(f"no paired train/valid epochs found in {path}")

    last_epoch = max(paired)
    best_epoch = min(paired, key=lambda epoch: paired[epoch]["valid_Loss"])
    first_epoch = min(paired)

    def gap(epoch: int) -> float:
        return paired[epoch]["valid_Loss"] - paired[epoch]["train_Loss"]

    return {
        "run": str(path.parent.parent),
        "first_epoch": first_epoch,
        "first_train": paired[first_epoch]["train_Loss"],
        "first_valid": paired[first_epoch]["valid_Loss"],
        "best_valid_epoch": best_epoch,
        "best_valid": paired[best_epoch]["valid_Loss"],
        "best_train": paired[best_epoch]["train_Loss"],
        "best_gap": gap(best_epoch),
        "last_epoch": last_epoch,
        "last_train": paired[last_epoch]["train_Loss"],
        "last_valid": paired[last_epoch]["valid_Loss"],
        "last_gap": gap(last_epoch),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path, help="robomimic logs/log.txt paths")
    args = parser.parse_args()

    headers = [
        "run",
        "first_epoch",
        "first_train",
        "first_valid",
        "best_valid_epoch",
        "best_train",
        "best_valid",
        "best_gap",
        "last_epoch",
        "last_train",
        "last_valid",
        "last_gap",
    ]
    print("\t".join(headers))
    for log_path in args.logs:
        summary = summarize(log_path)
        print("\t".join(str(summary[key]) for key in headers))


if __name__ == "__main__":
    main()
