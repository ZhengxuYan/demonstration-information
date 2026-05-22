#!/usr/bin/env python3
"""Summarize closed-loop rollouts for DemInf / MI-score filtered BC policies."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


RUN_RE = re.compile(r"deminf_mi_bc_(?P<algo>gmm|discrete)_(?P<dataset>.+)_drop_(?P<drop>\d+)_seed\d+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rollout-root",
        type=Path,
        default=Path("/iris/u/jasonyan/data/robomimic_rollout_scores/deminf_mi_filtered_bc"),
    )
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-fig", type=Path, default=None)
    return parser.parse_args()


def parse_run_name(run_name: str) -> dict[str, str]:
    match = RUN_RE.fullmatch(run_name)
    if match is None:
        raise ValueError(f"Cannot parse run name: {run_name}")
    return match.groupdict()


def read_rows(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for summary_path in sorted(root.glob("deminf_mi_bc_*/*_summary.csv")):
        run_name = summary_path.parent.name
        meta = parse_run_name(run_name)
        with summary_path.open(newline="") as f:
            reader = csv.DictReader(f)
            entries = list(reader)
        if not entries:
            continue
        best = max(entries, key=lambda row: (float(row.get("Success_Rate", 0.0)), int(row.get("Epoch", 0))))
        rows.append(
            {
                "algo": meta["algo"],
                "dataset": meta["dataset"],
                "drop_percent": int(meta["drop"]),
                "run_name": run_name,
                "checkpoint_label": best.get("Checkpoint_Label", ""),
                "epoch": int(best.get("Epoch", 0)),
                "num_rollouts": int(float(best.get("Num_Rollouts", 0))),
                "num_success": int(float(best.get("Num_Success", 0))),
                "success_rate": float(best.get("Success_Rate", 0.0)),
                "return": float(best.get("Return", 0.0)),
                "horizon": float(best.get("Horizon", 0.0)),
                "summary_csv": str(summary_path),
            }
        )
    return sorted(rows, key=lambda row: (str(row["algo"]), str(row["dataset"]), int(row["drop_percent"])))


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "algo",
        "dataset",
        "drop_percent",
        "run_name",
        "checkpoint_label",
        "epoch",
        "num_rollouts",
        "num_success",
        "success_rate",
        "return",
        "horizon",
        "summary_csv",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    datasets = sorted({str(row["dataset"]) for row in rows})
    algos = sorted({str(row["algo"]) for row in rows})
    fig, axes = plt.subplots(1, len(algos), figsize=(6.2 * len(algos), 4.2), squeeze=False)
    for ax, algo in zip(axes[0], algos):
        for dataset in datasets:
            subset = [row for row in rows if row["algo"] == algo and row["dataset"] == dataset]
            if not subset:
                continue
            xs = [int(row["drop_percent"]) for row in subset]
            ys = [float(row["success_rate"]) for row in subset]
            ax.plot(xs, ys, marker="o", linewidth=2.0, label=dataset)
        ax.set_title(f"{algo} BC")
        ax.set_xlabel("Dropped episodes by MI score (%)")
        ax.set_ylabel("Closed-loop success rate")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Policy performance after MI-score filtering")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = read_rows(args.rollout_root)
    output_csv = args.output_csv or args.rollout_root / "summary.csv"
    output_fig = args.output_fig or args.rollout_root / "success_vs_drop.png"
    write_csv(rows, output_csv)
    plot(rows, output_fig)
    print(output_csv)
    print(output_fig)


if __name__ == "__main__":
    main()
