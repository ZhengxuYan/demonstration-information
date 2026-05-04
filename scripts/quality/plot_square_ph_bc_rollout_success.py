#!/usr/bin/env python3
"""Plot Square PH BC rollout success rate across checkpoints."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_SUMMARIES = {
    "GMM agent+wrist": "/iris/u/jasonyan/data/robomimic_rollout_scores/square_ph_bc_checkpoint_sweep/square_ph_bc_gmm_agent_wrist_200_seed1/square_ph_bc_gmm_agent_wrist_200_seed1_summary.csv",
    "GMM left-close-low+wrist": "/iris/u/jasonyan/data/robomimic_rollout_scores/square_ph_bc_checkpoint_sweep/square_ph_bc_gmm_left_close_low_wrist_200_seed1/square_ph_bc_gmm_left_close_low_wrist_200_seed1_summary.csv",
    "Discrete agent+wrist": "/iris/u/jasonyan/data/robomimic_rollout_scores/square_ph_bc_checkpoint_sweep/square_ph_bc_discrete_agent_wrist_200_seed1/square_ph_bc_discrete_agent_wrist_200_seed1_summary.csv",
    "Discrete left-close-low+wrist": "/iris/u/jasonyan/data/robomimic_rollout_scores/square_ph_bc_checkpoint_sweep/square_ph_bc_discrete_left_close_low_wrist_200_seed1/square_ph_bc_discrete_left_close_low_wrist_200_seed1_summary.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="append", default=[], help="Optional label=/path/to/summary.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Square PH BC Checkpoint Rollout Success")
    return parser.parse_args()


def parse_summary_spec(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"Invalid --summary {raw!r}; expected label=/path/to/summary.csv")
    label, path = raw.split("=", 1)
    return label, Path(path)


def read_summary(path: Path) -> list[tuple[int, float]]:
    points = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row or row.get("Epoch") in (None, "Epoch"):
                continue
            points.append((int(row["Epoch"]), float(row["Success_Rate"])))
    return sorted(points)


def main() -> None:
    args = parse_args()
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    specs = [parse_summary_spec(x) for x in args.summary]
    if not specs:
        specs = [(label, Path(path)) for label, path in DEFAULT_SUMMARIES.items()]

    series = []
    for label, path in specs:
        points = read_summary(path)
        if not points:
            raise ValueError(f"No points found in {path}")
        series.append((label, points))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    colors = {
        "GMM agent+wrist": "#b94f24",
        "GMM left-close-low+wrist": "#d7903f",
        "Discrete agent+wrist": "#1f6f63",
        "Discrete left-close-low+wrist": "#244f8f",
    }
    for label, points in series:
        xs = [x for x, _ in points]
        ys = [y for _, y in points]
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=2.4,
            markersize=6,
            label=label,
            color=colors.get(label),
        )
        best_epoch, best_success = max(points, key=lambda item: item[1])
        ax.annotate(
            f"{best_success:.2f}",
            xy=(best_epoch, best_success),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=colors.get(label, "#222222"),
        )

    ax.set_title(args.title, fontsize=15, pad=14)
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Success rate")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, axis="y", color="#d9d0bf", linewidth=0.9)
    ax.grid(True, axis="x", color="#eee6d8", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(args.output, dpi=220)
    print(args.output)


if __name__ == "__main__":
    main()
