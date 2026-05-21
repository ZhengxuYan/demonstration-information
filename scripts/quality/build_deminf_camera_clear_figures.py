#!/usr/bin/env python3
"""Build clear figures from DemInf camera episode score CSV."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATASETS = ["ph_agentview", "400_agentview", "400_left_close_low", "400_mix"]
COLORS = {
    "ph": "#4C78A8",
    "rollout": "#F58518",
    "agentview": "#54A24B",
    "left_close_low": "#B279A2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["score"] = float(row["score"])
        row["ep_idx"] = int(row["ep_idx"])
    return rows


def stats(values: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    ci = 1.96 * sd / math.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return mean, sd, ci


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_dataset_overview(rows: list[dict], out: Path, ylim: tuple[float, float]) -> None:
    data = [[r["score"] for r in rows if r["dataset"] == d] for d in DATASETS]
    positions = np.arange(len(DATASETS)) + 1
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bp = ax.boxplot(data, positions=positions, widths=0.55, showfliers=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set(facecolor="#D9EAF7", edgecolor="#2F4B7C", linewidth=1.2)
    for med in bp["medians"]:
        med.set(color="#1B1B1B", linewidth=1.6)
    for pos, vals in zip(positions, data):
        mean, _, _ = stats(vals)
        ax.scatter([pos], [mean], marker="D", s=40, color="#E45756", zorder=3)
        ax.text(pos, ylim[1] - 0.06, f"n={len(vals)}\nmean={mean:.3f}", ha="center", va="top", fontsize=9)
    ax.axhline(0, color="0.55", linestyle="--", linewidth=1)
    ax.set_xticks(positions, DATASETS, rotation=15, ha="right")
    ax.set_ylabel("DemInf trajectory score")
    ax.set_title("Trained DemInf score by dataset")
    ax.set_ylim(ylim)
    ax.grid(axis="y", alpha=0.25)
    savefig(out / "dataset_score_box_overview.png")


def plot_ph_rollout(rows: list[dict], out: Path, ylim: tuple[float, float]) -> None:
    bins = np.linspace(ylim[0], ylim[1], 36)
    for dataset in DATASETS:
        subset = [r for r in rows if r["dataset"] == dataset]
        present = [g for g in ("ph", "rollout") if any(r["source"] == g for r in subset)]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.55, 1.0]})

        ax = axes[0]
        for group in present:
            vals = [r["score"] for r in subset if r["source"] == group]
            mean, _, _ = stats(vals)
            ax.hist(vals, bins=bins, alpha=0.48, color=COLORS[group], label=f"{group} n={len(vals)}, mean={mean:.3f}")
            ax.axvline(mean, color=COLORS[group], linewidth=2)
        ax.axvline(0, color="0.55", linestyle="--", linewidth=1)
        ax.set_xlim(ylim)
        ax.set_xlabel("DemInf trajectory score")
        ax.set_ylabel("episodes")
        ax.set_title(f"{dataset}: distribution by source")
        ax.legend(frameon=False, fontsize=9)
        ax.grid(axis="y", alpha=0.2)

        ax = axes[1]
        vals_by = [[r["score"] for r in subset if r["source"] == group] for group in present]
        bp = ax.boxplot(vals_by, labels=present, widths=0.55, showfliers=False, patch_artist=True)
        for patch, group in zip(bp["boxes"], present):
            patch.set(facecolor=COLORS[group], alpha=0.45, edgecolor=COLORS[group], linewidth=1.4)
        for i, (group, vals) in enumerate(zip(present, vals_by), start=1):
            mean, _, ci = stats(vals)
            ax.errorbar(i, mean, yerr=ci, fmt="D", color="black", capsize=4, markersize=4)
            ax.text(i, ylim[1] - 0.06, f"n={len(vals)}\nmean={mean:.3f}", ha="center", va="top", fontsize=9)
        if len(vals_by) == 2:
            mean0, _, _ = stats(vals_by[0])
            mean1, _, _ = stats(vals_by[1])
            ax.text(1.5, ylim[0] + 0.08, f"delta={mean1 - mean0:.3f}", ha="center", va="bottom", fontsize=10)
        ax.axhline(0, color="0.55", linestyle="--", linewidth=1)
        ax.set_ylim(ylim)
        ax.set_ylabel("DemInf trajectory score")
        ax.set_title("boxplot with mean +/- 95% CI")
        ax.grid(axis="y", alpha=0.2)
        savefig(out / f"{dataset}_ph_vs_rollout_clear.png")


def plot_mean_ci(rows: list[dict], out: Path) -> None:
    labels, means, cis, colors = [], [], [], []
    for dataset in ["400_agentview", "400_left_close_low", "400_mix"]:
        for group in ["ph", "rollout"]:
            vals = [r["score"] for r in rows if r["dataset"] == dataset and r["source"] == group]
            mean, _, ci = stats(vals)
            labels.append(f"{dataset}\n{group}")
            means.append(mean)
            cis.append(ci)
            colors.append(COLORS[group])
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    xs = np.arange(len(labels))
    ax.bar(xs, means, yerr=cis, color=colors, alpha=0.72, capsize=4, edgecolor="black", linewidth=0.6)
    ax.axhline(0, color="0.55", linestyle="--", linewidth=1)
    ax.set_xticks(xs, labels, rotation=20, ha="right")
    ax.set_ylabel("Mean DemInf trajectory score")
    ax.set_title("PH vs rollout mean scores by dataset")
    ax.grid(axis="y", alpha=0.25)
    savefig(out / "ph_vs_rollout_mean_ci_by_dataset.png")


def plot_mix_breakdowns(rows: list[dict], out: Path, ylim: tuple[float, float]) -> None:
    mix = [r for r in rows if r["dataset"] == "400_mix"]
    combos = [("agentview", "ph"), ("agentview", "rollout"), ("left_close_low", "ph"), ("left_close_low", "rollout")]
    vals_by = [[r["score"] for r in mix if r["view"] == view and r["source"] == source] for view, source in combos]
    labels = [f"{view}\n{source}" for view, source in combos]

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    bp = ax.boxplot(vals_by, labels=labels, showfliers=False, patch_artist=True)
    for patch, (view, source) in zip(bp["boxes"], combos):
        patch.set(facecolor=COLORS[source], alpha=0.45, edgecolor=COLORS[view], linewidth=1.6)
    for i, vals in enumerate(vals_by, start=1):
        mean, _, ci = stats(vals)
        ax.errorbar(i, mean, yerr=ci, fmt="D", color="black", capsize=4, markersize=4)
        ax.text(i, ylim[1] - 0.06, f"n={len(vals)}\nmean={mean:.3f}", ha="center", va="top", fontsize=8.5)
    ax.axhline(0, color="0.55", linestyle="--", linewidth=1)
    ax.set_ylim(ylim)
    ax.set_ylabel("DemInf trajectory score")
    ax.set_title("400_mix: score by camera view and source")
    ax.grid(axis="y", alpha=0.25)
    savefig(out / "400_mix_view_source_box_clear.png")

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    bins = np.linspace(ylim[0], ylim[1], 36)
    for view in ["agentview", "left_close_low"]:
        vals = [r["score"] for r in mix if r["view"] == view]
        mean, _, _ = stats(vals)
        ax.hist(vals, bins=bins, alpha=0.48, color=COLORS[view], label=f"{view} n={len(vals)}, mean={mean:.3f}")
        ax.axvline(mean, color=COLORS[view], linewidth=2)
    ax.axvline(0, color="0.55", linestyle="--", linewidth=1)
    ax.set_xlim(ylim)
    ax.set_xlabel("DemInf trajectory score")
    ax.set_ylabel("episodes")
    ax.set_title("400_mix: score distribution by camera view")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    savefig(out / "400_mix_view_distribution_clear.png")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.episode_csv)
    scores = np.asarray([r["score"] for r in rows])
    ylim = (
        float(np.floor((scores.min() - 0.05) * 10) / 10),
        float(np.ceil((scores.max() + 0.05) * 10) / 10),
    )
    plot_dataset_overview(rows, args.output_dir, ylim)
    plot_ph_rollout(rows, args.output_dir, ylim)
    plot_mean_ci(rows, args.output_dir)
    plot_mix_breakdowns(rows, args.output_dir, ylim)
    print(f"wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
