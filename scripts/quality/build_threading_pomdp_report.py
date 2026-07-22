#!/usr/bin/env python3
"""Build retained-label curves for Threading POMDP 6-score outputs."""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path

import numpy as np


SCORE_NAMES = (
    "neg_h_data_cond",
    "neg_h_model_cond",
    "mi_data_direct",
    "mi_data_mc_marginal",
    "mi_model_direct",
    "mi_model_mc_marginal",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Threading POMDP 6 Scores")
    parser.add_argument(
        "--description",
        default="Scores are higher-is-better. Labels use full=1, partial=0. Dashed black curve is human_label_filter.",
    )
    return parser.parse_args()


def read_labels(manifest: Path) -> dict[int, float]:
    labels = {}
    with manifest.open(newline="") as f:
        for idx, row in enumerate(csv.DictReader(f)):
            label = row["label"].strip().lower()
            ep_idx = int(row.get("ep_idx", idx))
            if label == "full":
                labels[ep_idx] = 1.0
            elif label == "partial":
                labels[ep_idx] = 0.0
    return labels


def read_scores(path: Path) -> dict[str, dict[int, float]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    out = {name: {} for name in SCORE_NAMES}
    for row in rows:
        ep_idx = int(row["ep_idx"])
        for name in SCORE_NAMES:
            out[name][ep_idx] = float(row[name])
    return out


def retained_curve(scores: dict[int, float], labels: dict[int, float]) -> list[dict[str, float]]:
    common = sorted(set(scores) & set(labels))
    if not common:
        raise ValueError("No overlapping ep_idx between scores and labels")
    ranked = sorted(common, key=lambda ep: scores[ep], reverse=True)
    rows = []
    for filtered in range(len(ranked)):
        retained = ranked[: len(ranked) - filtered]
        rows.append(
            {
                "filtered_episodes": filtered,
                "retained_episodes": len(retained),
                "avg_human_label": float(np.mean([labels[ep] for ep in retained])),
                "score_min_retained": float(min(scores[ep] for ep in retained)),
                "score_max_retained": float(max(scores[ep] for ep in retained)),
            }
        )
    return rows


def write_curves(curves: list[dict[str, object]], csv_path: Path, png_path: Path, title: str) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        fieldnames = ["score", "filtered_episodes", "retained_episodes", "avg_human_label", "score_min_retained", "score_max_retained"]
        writer = csv.DictWriter(f, fieldnames)
        writer.writeheader()
        writer.writerows(curves)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_score: dict[str, list[dict[str, object]]] = {}
    for row in curves:
        by_score.setdefault(str(row["score"]), []).append(row)
    fig, ax = plt.subplots(figsize=(10, 6))
    for score, rows in by_score.items():
        x = [int(row["filtered_episodes"]) for row in rows]
        y = [float(row["avg_human_label"]) for row in rows]
        if score == "human_label_filter":
            ax.plot(x, y, label=score, color="#111111", linestyle="--", linewidth=2.4)
        else:
            ax.plot(x, y, label=score, linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("Filtered episodes (higher score retained first)")
    ax.set_ylabel("Average retained label (full=1, partial=0)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def discover_score_files(score_root: Path) -> list[tuple[str, str, Path]]:
    found = []
    for algo in ("gaussian", "gmm"):
        for regime in ("normal", "2fold"):
            path = score_root / algo / regime / "threading_pomdp_6_scores.csv"
            if path.exists():
                found.append((algo, regime, path))
    if not found:
        raise FileNotFoundError(f"No threading_pomdp_6_scores.csv files under {score_root}")
    return found


def main() -> None:
    args = parse_args()
    labels = read_labels(args.manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    cards = []
    summary_rows = []
    for algo, regime, score_csv in discover_score_files(args.score_root):
        scores = read_scores(score_csv)
        curves = []
        for name in SCORE_NAMES:
            for row in retained_curve(scores[name], labels):
                curves.append({"score": name, **row})
        for row in retained_curve(labels, labels):
            curves.append({"score": "human_label_filter", **row})
        out_dir = args.output / algo / regime
        curve_csv = out_dir / "retained_label_curves.csv"
        curve_png = out_dir / "retained_label_curves.png"
        title = f"{args.title}: {algo} {regime}"
        write_curves(curves, curve_csv, curve_png, title)
        score_copy = out_dir / "threading_pomdp_6_scores.csv"
        score_copy.write_text(score_csv.read_text())
        cards.append((algo, regime, curve_png.relative_to(args.output), curve_csv.relative_to(args.output), score_copy.relative_to(args.output)))

        for name in SCORE_NAMES:
            score_rows = [row for row in curves if row["score"] == name]
            auc = float(np.mean([float(row["avg_human_label"]) for row in score_rows]))
            summary_rows.append({"algo": algo, "regime": regime, "score": name, "mean_curve_label": auc})

    with (args.output / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, ["algo", "regime", "score", "mean_curve_label"])
        writer.writeheader()
        writer.writerows(summary_rows)

    html_cards = []
    for algo, regime, png, curve_csv, score_csv in cards:
        html_cards.append(
            f"<section class='card'><h2>{html.escape(algo)} · {html.escape(regime)}</h2>"
            f"<img src='{png.as_posix()}' alt='{html.escape(algo)} {html.escape(regime)} curve'>"
            f"<p><a href='{curve_csv.as_posix()}'>curves csv</a> · "
            f"<a href='{score_csv.as_posix()}'>scores csv</a></p></section>"
        )
    index = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(args.title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; background: #f7f8f4; color: #17201b; }}
    h1 {{ margin: 0 0 8px; }}
    .meta {{ color: #647067; margin-bottom: 20px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(560px, 1fr)); gap: 18px; }}
    .card {{ background: white; border: 1px solid #dce3dc; border-radius: 8px; padding: 14px; }}
    .card h2 {{ margin: 0 0 10px; font-size: 18px; }}
    .card img {{ width: 100%; border: 1px solid #eef1ee; }}
    a {{ color: #245c9f; text-decoration: none; }}
  </style>
</head>
<body>
  <h1>{html.escape(args.title)}</h1>
  <div class="meta">{html.escape(args.description)}</div>
  <div class="grid">{''.join(html_cards)}</div>
</body>
</html>
"""
    (args.output / "index.html").write_text(index)
    print(args.output / "index.html")
    print(f"cards={len(cards)}")


if __name__ == "__main__":
    main()
