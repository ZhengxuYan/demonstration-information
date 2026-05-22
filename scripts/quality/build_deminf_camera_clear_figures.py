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

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    views = ["agentview", "left_close_low"]
    view_values = [[r["score"] for r in mix if r["view"] == view] for view in views]
    parts = ax.violinplot(view_values, positions=[1, 2], widths=0.75, showmeans=False, showextrema=False, showmedians=False)
    for body, view in zip(parts["bodies"], views):
        body.set_facecolor(COLORS[view])
        body.set_edgecolor(COLORS[view])
        body.set_alpha(0.28)

    bp = ax.boxplot(view_values, positions=[1, 2], widths=0.32, showfliers=False, patch_artist=True)
    for patch, view in zip(bp["boxes"], views):
        patch.set(facecolor="white", edgecolor=COLORS[view], linewidth=1.7)
    for med in bp["medians"]:
        med.set(color="black", linewidth=1.6)

    rng = np.random.default_rng(7)
    for i, (view, vals) in enumerate(zip(views, view_values), start=1):
        vals_arr = np.asarray(vals, dtype=float)
        mean, _, ci = stats(vals)
        jitter = rng.normal(0, 0.045, size=len(vals_arr))
        ax.scatter(np.full(len(vals_arr), i) + jitter, vals_arr, s=10, color=COLORS[view], alpha=0.28, linewidths=0)
        ax.errorbar(i, mean, yerr=ci, fmt="D", color="black", capsize=5, markersize=5, zorder=4)
        ax.text(i, ylim[1] - 0.06, f"n={len(vals)}\nmean={mean:.3f}", ha="center", va="top", fontsize=9.5)

    mean_agent, _, _ = stats(view_values[0])
    mean_left, _, _ = stats(view_values[1])
    ax.text(1.5, ylim[0] + 0.08, f"left - agent = {mean_left - mean_agent:.3f}", ha="center", va="bottom", fontsize=10)
    ax.axhline(0, color="0.55", linestyle="--", linewidth=1)
    ax.set_xticks([1, 2], views)
    ax.set_ylim(ylim)
    ax.set_ylabel("DemInf trajectory score")
    ax.set_title("400_mix: camera view score comparison")
    ax.grid(axis="y", alpha=0.25)
    savefig(out / "400_mix_view_distribution_clear.png")


def plot_mix_filtered_quality_curve(rows: list[dict], out: Path) -> None:
    mix = [r for r in rows if r["dataset"] == "400_mix"]
    if not mix:
        return

    labels = []
    scores = []
    for row in mix:
        if row["view"] == "agentview":
            labels.append(2.0)
        elif row["view"] == "left_close_low":
            labels.append(1.0)
        else:
            continue
        scores.append(float(row["score"]))

    labels_arr = np.asarray(labels, dtype=float)
    scores_arr = np.asarray(scores, dtype=float)
    n = len(labels_arr)
    if n == 0:
        return

    # Remove low DemInf-score episodes first. At x=0, all episodes are kept; at
    # x=n-1, only the single highest-scored episode remains.
    sorted_by_score = np.argsort(scores_arr)[::-1]
    quality_sorted_by_score = labels_arr[sorted_by_score]
    kept_counts = np.arange(1, n + 1)
    avg_quality_top_k = np.cumsum(quality_sorted_by_score) / kept_counts
    episodes_filtered = n - kept_counts

    oracle_quality = np.sort(labels_arr)[::-1]
    oracle_avg_quality_top_k = np.cumsum(oracle_quality) / kept_counts

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(episodes_filtered[::-1], avg_quality_top_k[::-1], color="#2F4B7C", linewidth=2.2, label="DemInf ranking")
    ax.plot(episodes_filtered[::-1], oracle_avg_quality_top_k[::-1], color="#777777", linewidth=2.0, linestyle="--", label="Oracle")
    ax.axhline(labels_arr.mean(), color="#E45756", linewidth=1.6, linestyle=":", label=f"Random / no filter ({labels_arr.mean():.2f})")
    ax.set_xlabel("Number of episodes filtered")
    ax.set_ylabel("Average observability label of remaining episodes")
    ax.set_title("400_mix: filtering curve (agentview=2, left_close_low=1)")
    ax.set_xlim(0, n - 1)
    ax.set_ylim(0.95, 2.05)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    ax.text(
        0.02,
        0.05,
        f"n={n}; full={int(np.sum(labels_arr == 2.0))}; partial={int(np.sum(labels_arr == 1.0))}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
    )
    savefig(out / "400_mix_filtered_avg_observability_curve.png")


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
    plot_mix_filtered_quality_curve(rows, args.output_dir)
    print(f"wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
