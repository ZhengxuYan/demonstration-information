#!/usr/bin/env python3
"""Build distribution plots for robomimic policy NLL and entropy scores."""

from __future__ import annotations

import argparse
import csv
import html
import os
import pickle
import re
import tempfile
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "xdg-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = [
    ("nll_images_robot", "NLL images+robot"),
    ("entropy_images_robot", "Entropy images+robot"),
    ("nll_delta", "NLL images+robot - NLL robot"),
    ("entropy_delta", "Entropy images+robot - Entropy robot"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="CSV with image_score and robot_score columns.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annotations-csv", type=Path, default=Path("observability_annotations.csv"))
    parser.add_argument(
        "--mh-label-root",
        type=Path,
        action="append",
        default=[],
        help="Directory to scan recursively for Square MH demo_*_label_{1,2,3} videos.",
    )
    parser.add_argument("--bins", type=int, default=30)
    return parser.parse_args()


def load_score(path: Path) -> dict[str, dict[int, float]]:
    with path.open("rb") as f:
        data = pickle.load(f)
    nll = data.get("ep_idx_nll", data.get("ep_idx", {}))
    entropy = data.get("ep_idx_entropy")
    if entropy is None:
        raise KeyError(f"{path} does not contain ep_idx_entropy; rescore with the updated scorer")
    return {
        "nll": {int(k): float(v) for k, v in nll.items()},
        "entropy": {int(k): float(v) for k, v in entropy.items()},
    }


def load_observability_labels(path: Path) -> dict[str, dict[int, str]]:
    labels: dict[str, dict[int, str]] = defaultdict(dict)
    if not path.exists():
        return labels
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            dataset = row.get("dataset", "").strip()
            label = row.get("label", "").strip().lower()
            if not dataset or label not in {"full", "partial"}:
                continue
            try:
                ep_idx = int(row["ep_idx"])
            except (KeyError, ValueError):
                continue
            labels[dataset][ep_idx] = label
            if dataset == "expert200":
                labels["expert200_random_post"][ep_idx] = label
    return labels


def load_mh_numeric_labels(roots: list[Path]) -> dict[int, str]:
    labels: dict[int, str] = {}
    pattern = re.compile(r"demo_(\d+).*label_([123])(?:\D|$)")
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            match = pattern.search(path.stem)
            if match:
                labels[int(match.group(1))] = f"label_{match.group(2)}"
    return labels


def label_for(dataset: str, ep_idx: int, obs_labels: dict[str, dict[int, str]], mh_labels: dict[int, str]) -> str | None:
    if dataset == "square_mh":
        return mh_labels.get(ep_idx)
    return obs_labels.get(dataset, {}).get(ep_idx)


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def histogram_curve(values: list[float], bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts, edges = np.histogram(np.asarray(values, dtype=np.float64), bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts


def plot_metric(
    values_by_label: dict[str, list[float]],
    title: str,
    xlabel: str,
    output_base: Path,
    bins_count: int,
) -> tuple[Path, Path] | None:
    non_empty = {k: v for k, v in values_by_label.items() if v}
    if not non_empty:
        return None
    all_values = np.asarray([v for values in non_empty.values() for v in values], dtype=np.float64)
    if np.allclose(all_values.min(), all_values.max()):
        delta = 0.5 if all_values.min() == 0 else abs(all_values.min()) * 0.05
        bins = np.linspace(all_values.min() - delta, all_values.max() + delta, max(3, bins_count))
    else:
        bins = np.linspace(all_values.min(), all_values.max(), bins_count + 1)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for label in sorted(non_empty):
        xs, ys = histogram_curve(non_empty[label], bins)
        ax.step(xs, ys, where="mid", linewidth=2, label=f"{label} (n={len(non_empty[label])})")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of episodes")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()

    png = output_base.with_suffix(".png")
    svg = output_base.with_suffix(".svg")
    fig.savefig(png, dpi=180)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg


def metric_values(image: dict[str, dict[int, float]], robot: dict[str, dict[int, float]], metric: str) -> dict[int, float]:
    if metric == "nll_images_robot":
        return image["nll"]
    if metric == "entropy_images_robot":
        return image["entropy"]
    if metric == "nll_delta":
        common = set(image["nll"]) & set(robot["nll"])
        return {ep: image["nll"][ep] - robot["nll"][ep] for ep in common}
    if metric == "entropy_delta":
        common = set(image["entropy"]) & set(robot["entropy"])
        return {ep: image["entropy"][ep] - robot["entropy"][ep] for ep in common}
    raise ValueError(metric)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_html(output: Path, rows: list[dict[str, str]]) -> None:
    cards = []
    for row in rows:
        rel_png = Path(row["png"]).relative_to(output).as_posix()
        rel_svg = Path(row["svg"]).relative_to(output).as_posix()
        title = html.escape(row["title"])
        cards.append(
            f"<section><h2>{title}</h2><p>{html.escape(row['summary'])}</p>"
            f"<a href='{html.escape(rel_svg)}'><img src='{html.escape(rel_png)}' alt='{title}'></a></section>"
        )
    body = "\n".join(cards)
    (output / "index.html").write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Policy NLL + Entropy Distributions</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 28px; color: #17201b; }}
    h1 {{ font-size: 24px; }}
    section {{ border-top: 1px solid #d7ded8; padding: 20px 0; max-width: 920px; }}
    h2 {{ font-size: 18px; margin: 0 0 6px; }}
    p {{ color: #526057; margin: 0 0 12px; }}
    img {{ width: min(860px, 100%); border: 1px solid #d7ded8; }}
  </style>
</head>
<body>
  <h1>Policy NLL + Entropy Distributions</h1>
  <p><a href="manifest_summary.csv">manifest_summary.csv</a></p>
  {body}
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plot_dir = args.output / "plots"
    plot_dir.mkdir(exist_ok=True)

    obs_labels = load_observability_labels(args.annotations_csv)
    mh_labels = load_mh_numeric_labels(args.mh_label_root)
    rows = read_manifest(args.manifest)

    summary_rows: list[dict[str, str]] = []
    plot_rows: list[dict[str, str]] = []
    score_cache: dict[Path, dict[str, dict[int, float]]] = {}

    def cached_score(path_text: str) -> dict[str, dict[int, float]]:
        path = Path(path_text)
        if path not in score_cache:
            score_cache[path] = load_score(path)
        return score_cache[path]

    for row in rows:
        dataset = row["dataset"]
        image = cached_score(row["image_score"])
        robot = cached_score(row["robot_score"])
        for metric, label in METRICS:
            ep_values = metric_values(image, robot, metric)
            values_by_label: dict[str, list[float]] = defaultdict(list)
            for ep_idx, value in ep_values.items():
                bucket = label_for(dataset, int(ep_idx), obs_labels, mh_labels)
                if bucket is not None:
                    values_by_label[bucket].append(float(value))

            title = " / ".join(
                [
                    dataset,
                    row.get("policy", ""),
                    row.get("view", ""),
                    row.get("checkpoint_label", ""),
                    label,
                ]
            )
            stem = safe_name(title)
            plotted = plot_metric(values_by_label, title, label, plot_dir / stem, args.bins)
            if plotted is None:
                continue
            png, svg = plotted
            plot_rows.append(
                {
                    "png": str(png),
                    "svg": str(svg),
                    "title": title,
                    "summary": ", ".join(f"{k}: {len(v)} eps" for k, v in sorted(values_by_label.items())),
                }
            )
            for bucket, values in sorted(values_by_label.items()):
                arr = np.asarray(values, dtype=np.float64)
                if len(arr) == 0:
                    continue
                summary_rows.append(
                    {
                        "dataset": dataset,
                        "policy": row.get("policy", ""),
                        "view": row.get("view", ""),
                        "checkpoint_label": row.get("checkpoint_label", ""),
                        "metric": metric,
                        "label": bucket,
                        "n": str(len(arr)),
                        "mean": f"{arr.mean():.8g}",
                        "std": f"{arr.std():.8g}",
                        "min": f"{arr.min():.8g}",
                        "max": f"{arr.max():.8g}",
                        "png": str(png),
                        "svg": str(svg),
                        "title": title,
                        "summary": ", ".join(f"{k}: {len(v)} eps" for k, v in sorted(values_by_label.items())),
                    }
                )

    summary_path = args.output / "manifest_summary.csv"
    fieldnames = [
        "dataset",
        "policy",
        "view",
        "checkpoint_label",
        "metric",
        "label",
        "n",
        "mean",
        "std",
        "min",
        "max",
        "png",
        "svg",
        "title",
        "summary",
    ]
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    write_html(args.output, plot_rows)
    print(f"wrote {summary_path}")
    print(f"wrote {args.output / 'index.html'}")


if __name__ == "__main__":
    main()
