#!/usr/bin/env python3
"""Evaluate density scores by retained average human label after filtering."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


OBS_LABELS = {
    "full": 2.0,
    "fully_observable": 2.0,
    "1": 1.0,
    "true": 1.0,
    "partial": 1.0,
    "partially_observable": 1.0,
    "0": 0.0,
    "false": 0.0,
    "better": 3.0,
    "good": 3.0,
    "okay": 2.0,
    "ok": 2.0,
    "worse": 1.0,
    "bad": 1.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-csv", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, default=None)
    parser.add_argument("--label-column", default="observability")
    parser.add_argument("--ylabel", default=None)
    parser.add_argument("--higher-is-better", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-filtered", type=int, default=None)
    return parser.parse_args()


def parse_label(value: str) -> float:
    text = str(value).strip().lower()
    if text in OBS_LABELS:
        return OBS_LABELS[text]
    return float(text)


def read_labels(path: Path, column: str) -> dict[int, float]:
    labels = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if not row.get(column, "").strip():
                continue
            labels[int(row["ep_idx"])] = parse_label(row[column])
    return labels


def read_scores(path: Path) -> tuple[list[str], dict[str, dict[int, float]]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {path}")
    score_names = [name for name in rows[0] if name != "ep_idx"]
    scores = {name: {} for name in score_names}
    for row in rows:
        ep_idx = int(row["ep_idx"])
        for name in score_names:
            value = row.get(name, "")
            if value != "":
                scores[name][ep_idx] = float(value)
    return score_names, scores


def retained_curve(scores: dict[int, float], labels: dict[int, float], higher_is_better: bool) -> list[dict[str, float]]:
    common = sorted(set(scores) & set(labels))
    if not common:
        raise ValueError("No overlapping ep_idx values between scores and labels")
    ranked = sorted(common, key=lambda ep: scores[ep], reverse=higher_is_better)
    out = []
    for filtered in range(0, len(ranked)):
        retained = ranked[: len(ranked) - filtered]
        avg = float(np.mean([labels[ep] for ep in retained]))
        out.append(
            {
                "filtered_episodes": filtered,
                "retained_episodes": len(retained),
                "avg_human_label": avg,
                "score_min_retained": float(min(scores[ep] for ep in retained)),
                "score_max_retained": float(max(scores[ep] for ep in retained)),
            }
        )
    return out


def write_plot(rows: list[dict[str, str]], path: Path, higher_is_better: bool, ylabel: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_score: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_score.setdefault(row["score"], []).append(row)

    fig, ax = plt.subplots(figsize=(10, 6))
    for score, score_rows in by_score.items():
        x = [int(r["filtered_episodes"]) for r in score_rows]
        y = [float(r["avg_human_label"]) for r in score_rows]
        ax.plot(x, y, label=score, linewidth=1.8)
    ax.set_xlabel("Filtered episodes")
    ax.set_ylabel(ylabel)
    direction = "higher score retained first" if higher_is_better else "lower score retained first"
    ax.set_title(f"Retained observability curves ({direction})")
    values = [float(r["avg_human_label"]) for r in rows]
    if values:
        pad = max(0.05, 0.05 * (max(values) - min(values)))
        ax.set_ylim(min(values) - pad, max(values) + pad)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=1, loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)


def main() -> None:
    args = parse_args()
    labels = read_labels(args.labels_csv, args.label_column)
    score_names, scores = read_scores(args.scores_csv)

    output_rows = []
    for name in score_names:
        curve = retained_curve(scores[name], labels, args.higher_is_better)
        if args.max_filtered is not None:
            curve = [row for row in curve if row["filtered_episodes"] <= args.max_filtered]
        for row in curve:
            output_rows.append({"score": name, **row})

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        fieldnames = [
            "score",
            "filtered_episodes",
            "retained_episodes",
            "avg_human_label",
            "score_min_retained",
            "score_max_retained",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    if args.output_png is not None:
        ylabel = args.ylabel or f"Average human {args.label_column} among retained episodes"
        write_plot(output_rows, args.output_png, args.higher_is_better, ylabel)

    print(f"wrote {args.output_csv}")
    if args.output_png is not None:
        print(f"wrote {args.output_png}")
    print(f"scores={len(score_names)} labels={len(labels)}")


if __name__ == "__main__":
    main()
