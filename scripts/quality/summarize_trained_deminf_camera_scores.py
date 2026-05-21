#!/usr/bin/env python3
"""Summarize and visualize trained DemInf camera-view scores.

The scoring pickles produced by estimate_quality_combined_robomimic.py store
episode-level DemInf scores. This script normalizes those pickles into one CSV
and makes lightweight plots for PH vs rollout, view comparisons, and the mixed
view dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

try:
    import h5py
except Exception:  # pragma: no cover
    h5py = None


DATASETS = ("ph_agentview", "400_agentview", "400_left_close_low", "400_mix")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--score-root",
        type=Path,
        required=True,
        help="Root containing image_proprio/<dataset>/<dataset>.pkl or directly <dataset>/<dataset>.pkl.",
    )
    parser.add_argument(
        "--hdf5-root",
        type=Path,
        default=None,
        help="Optional root containing <dataset>/image.hdf5 with dataset_build_metadata_json.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--better-view", default="left_close_low")
    parser.add_argument("--worse-view", default="agentview")
    return parser.parse_args()


def read_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict pickle at {path}, got {type(obj)}")
    return obj


def find_score_pickle(score_root: Path, dataset: str) -> Path:
    candidates = [
        score_root / dataset / f"{dataset}.pkl",
        score_root / "image_proprio" / dataset / f"{dataset}.pkl",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No score pickle found for {dataset} under {score_root}")


def as_int_key_dict(value: Any) -> dict[int, float]:
    if not isinstance(value, dict):
        return {}
    out = {}
    for key, score in value.items():
        try:
            ep_idx = int(key)
            out[ep_idx] = float(score)
        except Exception:
            continue
    return out


def episode_scores(ds_scores: dict[str, Any]) -> dict[int, float]:
    # Preferred form used by quality_estimators: ep_idx maps episode index to
    # the DemInf score. If unavailable, fall back to averaging per-step samples.
    scores = as_int_key_dict(ds_scores.get("ep_idx"))
    if scores:
        return scores

    sample_ep_idx = ds_scores.get("sample_ep_idx")
    sample_score = ds_scores.get("sample_score")
    if sample_ep_idx is None or sample_score is None:
        raise KeyError(f"Could not find episode score fields. Keys: {sorted(ds_scores.keys())}")

    grouped: dict[int, list[float]] = defaultdict(list)
    for ep_idx, score in zip(np.asarray(sample_ep_idx), np.asarray(sample_score)):
        grouped[int(ep_idx)].append(float(score))
    return {ep_idx: float(np.mean(values)) for ep_idx, values in grouped.items()}


def read_metadata(hdf5_root: Path | None, dataset: str) -> dict[int, dict[str, Any]]:
    if hdf5_root is None or h5py is None:
        return {}
    path = hdf5_root / dataset / "image.hdf5"
    if not path.exists():
        return {}
    with h5py.File(path, "r") as f:
        raw = f.attrs.get("dataset_build_metadata_json", "")
    if raw is None or raw == "":
        return {}
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    meta = json.loads(raw)
    out = {}
    for key, value in meta.items():
        if isinstance(key, str) and key.startswith("demo_"):
            ep_idx = int(key.split("_", 1)[1])
        else:
            ep_idx = int(key)
        out[ep_idx] = value if isinstance(value, dict) else {}
    return out


def infer_source(dataset: str, ep_idx: int, meta: dict[str, Any]) -> str:
    for key in ("source_type", "source", "split"):
        value = str(meta.get(key, "")).lower()
        if "rollout" in value:
            return "rollout"
        if value in {"ph", "human", "demo"} or "ph" in value:
            return "ph"
    source_path = str(meta.get("source_path", "")).lower()
    if "low_dim_bc_gmm" in source_path or "rollout" in source_path:
        return "rollout"
    if dataset == "ph_agentview":
        return "ph"
    if dataset in {"400_agentview", "400_left_close_low"}:
        return "ph" if ep_idx < 200 else "rollout"
    if dataset == "400_mix" and meta:
        # The mixed dataset is shuffled, so ep_idx does not encode source. When
        # metadata does not identify a rollout source path, the episode is from
        # the PH half of the constructed dataset.
        return "ph"
    return "unknown"


def infer_view(dataset: str, meta: dict[str, Any]) -> str:
    view = meta.get("view") or meta.get("camera") or meta.get("camera_name")
    if view:
        return str(view)
    if dataset in {"ph_agentview", "400_agentview"}:
        return "agentview"
    if dataset == "400_left_close_low":
        return "left_close_low"
    return "unknown"


def summarize(values: list[float]) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_hist_by_group(rows: list[dict[str, Any]], group_key: str, path: Path, title: str) -> None:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        groups[str(row[group_key])].append(float(row["score"]))
    plt.figure(figsize=(7, 4))
    for group, values in sorted(groups.items()):
        if not values:
            continue
        plt.hist(values, bins=30, alpha=0.45, label=f"{group} (n={len(values)})")
    plt.xlabel("DemInf trajectory score")
    plt.ylabel("episodes")
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


def plot_box(rows: list[dict[str, Any]], label_key: str, path: Path, title: str) -> None:
    labels = sorted({str(row[label_key]) for row in rows})
    values = [[float(row["score"]) for row in rows if str(row[label_key]) == label] for label in labels]
    plt.figure(figsize=(max(5, len(labels) * 1.2), 4))
    plt.boxplot(values, labels=labels, showfliers=False)
    plt.ylabel("DemInf trajectory score")
    plt.title(title)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []

    for dataset in DATASETS:
        pkl_path = find_score_pickle(args.score_root, dataset)
        ds_scores = read_pickle(pkl_path)
        scores = episode_scores(ds_scores)
        metadata = read_metadata(args.hdf5_root, dataset)

        for ep_idx, score in sorted(scores.items()):
            meta = metadata.get(ep_idx, {})
            source = infer_source(dataset, ep_idx, meta)
            view = infer_view(dataset, meta)
            if view == args.better_view:
                observability_label = "better"
            elif view == args.worse_view:
                observability_label = "worse"
            else:
                observability_label = str(meta.get("observability_label", "unknown"))
            rows.append(
                {
                    "dataset": dataset,
                    "ep_idx": ep_idx,
                    "score": score,
                    "source": source,
                    "view": view,
                    "observability_label": observability_label,
                    "score_pickle": str(pkl_path),
                }
            )

    args.output_root.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "ep_idx", "score", "source", "view", "observability_label", "score_pickle"]
    write_csv(args.output_root / "episode_scores.csv", rows, fields)

    summary_rows = []
    for dataset in DATASETS:
        values = [float(row["score"]) for row in rows if row["dataset"] == dataset]
        summary_rows.append({"group": "dataset", "name": dataset, **summarize(values)})
    for key in ("source", "view", "observability_label"):
        for value in sorted({str(row[key]) for row in rows}):
            values = [float(row["score"]) for row in rows if str(row[key]) == value]
            summary_rows.append({"group": key, "name": value, **summarize(values)})
    write_csv(
        args.output_root / "score_summary.csv",
        summary_rows,
        ["group", "name", "n", "mean", "std", "min", "max", "median", "p25", "p75"],
    )

    fig_root = args.output_root / "figures"
    plot_box(rows, "dataset", fig_root / "score_by_dataset_box.png", "DemInf scores by dataset")
    plot_hist_by_group(rows, "source", fig_root / "score_by_source_hist.png", "PH vs rollout score distribution")

    for dataset in ("400_agentview", "400_left_close_low", "400_mix"):
        subset = [row for row in rows if row["dataset"] == dataset]
        plot_hist_by_group(
            subset,
            "source",
            fig_root / f"{dataset}_ph_vs_rollout_hist.png",
            f"{dataset}: PH vs rollout",
        )

    mix_rows = [row for row in rows if row["dataset"] == "400_mix"]
    plot_hist_by_group(mix_rows, "view", fig_root / "400_mix_by_view_hist.png", "400_mix by camera view")
    plot_box(mix_rows, "observability_label", fig_root / "400_mix_better_worse_box.png", "400_mix better/worse")

    report = [
        "# Trained DemInf Camera Score Summary",
        "",
        f"Rows: {len(rows)} episodes",
        "",
        "## Outputs",
        "",
        "- `episode_scores.csv`: per-episode normalized table.",
        "- `score_summary.csv`: descriptive statistics.",
        "- `figures/`: score distributions and box plots.",
        "",
        "## Notes",
        "",
        f"- Better/worse labels use view mapping: `{args.better_view}` -> better, `{args.worse_view}` -> worse.",
        "- Change `--better-view` / `--worse-view` if the camera observability ordering should be reversed.",
    ]
    (args.output_root / "analysis-report.md").write_text("\n".join(report) + "\n")

    print(f"wrote {args.output_root / 'episode_scores.csv'}")
    print(f"wrote {args.output_root / 'score_summary.csv'}")
    print(f"wrote figures under {fig_root}")


if __name__ == "__main__":
    main()
