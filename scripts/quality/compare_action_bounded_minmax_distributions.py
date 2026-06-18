#!/usr/bin/env python3
"""Compare raw and percentile-bounded minmax action distributions across HDF5 datasets."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Dataset spec as name=/path/to/file.hdf5. Can be passed multiple times.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--low-percentile", type=float, default=1.0)
    parser.add_argument("--high-percentile", type=float, default=99.0)
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument(
        "--all-demos-for-bounds",
        action="store_true",
        help="Use all demos for percentile bounds instead of HDF5 mask/train when available.",
    )
    return parser.parse_args()


def parse_dataset_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Expected dataset spec name=/path/to/file.hdf5, got {spec!r}")
    name, path = spec.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Dataset name is empty in {spec!r}")
    return name, Path(path).expanduser()


def mask_demo_keys(f: h5py.File, mask_name: str) -> list[str]:
    if "mask" not in f or mask_name not in f["mask"]:
        return []
    return [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in f["mask"][mask_name][:]]


def read_actions(path: Path, use_train_bounds: bool) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as f:
        if "data" not in f:
            raise KeyError(f"{path} missing /data group")
        all_keys = sorted(f["data"].keys())
        train_keys = mask_demo_keys(f, "train") if use_train_bounds else []
        bound_keys = train_keys or all_keys

        def read_for_keys(keys: list[str]) -> np.ndarray:
            arrays = []
            for key in keys:
                grp = f["data"][key]
                action_key = "actions_raw" if "actions_raw" in grp else "actions"
                arrays.append(np.asarray(grp[action_key][:], dtype=np.float64))
            if not arrays:
                raise ValueError(f"No action arrays found in {path}")
            return np.concatenate(arrays, axis=0)

        all_actions = read_for_keys(all_keys)
        bound_actions = read_for_keys(bound_keys)
    if all_actions.ndim != 2:
        raise ValueError(f"Expected actions with shape (N, D), got {all_actions.shape} in {path}")
    if bound_actions.shape[-1] != all_actions.shape[-1]:
        raise ValueError(f"Action dim mismatch in {path}: all={all_actions.shape} bounds={bound_actions.shape}")
    return all_actions, bound_actions


def bounded_minmax(actions: np.ndarray, bounds_actions: np.ndarray, low_pct: float, high_pct: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    low = np.percentile(bounds_actions, low_pct, axis=0)
    high = np.percentile(bounds_actions, high_pct, axis=0)
    scale = np.where((high - low) < 1e-6, 1.0, high - low)
    low_count = np.sum(actions < low, axis=0)
    high_count = np.sum(actions > high, axis=0)
    clipped = np.clip(actions, low, high)
    normalized = 2.0 * (clipped - low) / scale - 1.0
    return normalized, {
        "low": low,
        "high": high,
        "scale": scale,
        "low_count": low_count,
        "high_count": high_count,
        "clipped_fraction": (low_count + high_count) / max(len(actions), 1),
    }


def write_stats_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "dataset",
        "dim",
        "num_samples",
        "raw_mean",
        "raw_std",
        "raw_min",
        "raw_p01",
        "raw_p50",
        "raw_p99",
        "raw_max",
        "bound_low",
        "bound_high",
        "clip_low_count",
        "clip_high_count",
        "clip_fraction",
        "normalized_mean",
        "normalized_std",
        "normalized_min",
        "normalized_p01",
        "normalized_p50",
        "normalized_p99",
        "normalized_max",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_plots(output_dir: Path, datasets: dict[str, dict[str, np.ndarray]], bins: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    action_dim = next(iter(datasets.values()))["raw"].shape[-1]
    for kind, title, filename in (
        ("raw", "Raw absolute action distributions", "raw_action_distributions.png"),
        ("normalized", "Percentile-bounded minmax action distributions", "bounded_minmax_action_distributions.png"),
    ):
        fig, axes = plt.subplots(action_dim, 1, figsize=(10, max(2.2 * action_dim, 4)), squeeze=False)
        for dim in range(action_dim):
            ax = axes[dim, 0]
            for name, data in datasets.items():
                values = data[kind][:, dim]
                ax.hist(values, bins=bins, density=True, alpha=0.32, label=name)
            ax.set_title(f"action dim {dim}")
            ax.grid(alpha=0.2)
            if kind == "normalized":
                ax.axvline(-1.0, color="black", linestyle="--", linewidth=1)
                ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    width = 0.8 / max(len(datasets), 1)
    x = np.arange(action_dim)
    for offset, (name, data) in enumerate(datasets.items()):
        ax.bar(x + offset * width, data["clip_fraction"], width=width, label=name)
    ax.set_xticks(x + width * (len(datasets) - 1) / 2)
    ax.set_xticklabels([str(i) for i in range(action_dim)])
    ax.set_xlabel("action dim")
    ax.set_ylabel("clipped fraction")
    ax.set_title("Fraction of samples clipped by percentile bounds")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "clip_fraction_by_dim.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not 0 <= args.low_percentile < args.high_percentile <= 100:
        raise ValueError("Expected 0 <= low-percentile < high-percentile <= 100")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets: dict[str, dict[str, np.ndarray]] = {}
    rows: list[dict[str, object]] = []
    for spec in args.dataset:
        name, path = parse_dataset_spec(spec)
        raw, bound_actions = read_actions(path, use_train_bounds=not args.all_demos_for_bounds)
        normalized, stats = bounded_minmax(raw, bound_actions, args.low_percentile, args.high_percentile)
        if not np.all(np.isfinite(normalized)):
            raise ValueError(f"Non-finite normalized actions for {name}")
        if normalized.min() < -1.00001 or normalized.max() > 1.00001:
            raise ValueError(f"Normalized actions outside [-1, 1] for {name}: {normalized.min()} {normalized.max()}")
        datasets[name] = {
            "raw": raw,
            "normalized": normalized,
            "clip_fraction": stats["clipped_fraction"],
        }
        for dim in range(raw.shape[-1]):
            rows.append(
                {
                    "dataset": name,
                    "dim": dim,
                    "num_samples": len(raw),
                    "raw_mean": float(np.mean(raw[:, dim])),
                    "raw_std": float(np.std(raw[:, dim])),
                    "raw_min": float(np.min(raw[:, dim])),
                    "raw_p01": float(np.percentile(raw[:, dim], 1)),
                    "raw_p50": float(np.percentile(raw[:, dim], 50)),
                    "raw_p99": float(np.percentile(raw[:, dim], 99)),
                    "raw_max": float(np.max(raw[:, dim])),
                    "bound_low": float(stats["low"][dim]),
                    "bound_high": float(stats["high"][dim]),
                    "clip_low_count": int(stats["low_count"][dim]),
                    "clip_high_count": int(stats["high_count"][dim]),
                    "clip_fraction": float(stats["clipped_fraction"][dim]),
                    "normalized_mean": float(np.mean(normalized[:, dim])),
                    "normalized_std": float(np.std(normalized[:, dim])),
                    "normalized_min": float(np.min(normalized[:, dim])),
                    "normalized_p01": float(np.percentile(normalized[:, dim], 1)),
                    "normalized_p50": float(np.percentile(normalized[:, dim], 50)),
                    "normalized_p99": float(np.percentile(normalized[:, dim], 99)),
                    "normalized_max": float(np.max(normalized[:, dim])),
                }
            )

    write_stats_csv(output_dir / "action_distribution_stats.csv", rows)
    write_plots(output_dir, datasets, args.bins)
    print(f"wrote {output_dir / 'action_distribution_stats.csv'}")
    print(f"wrote {output_dir / 'raw_action_distributions.png'}")
    print(f"wrote {output_dir / 'bounded_minmax_action_distributions.png'}")
    print(f"wrote {output_dir / 'clip_fraction_by_dim.png'}")


if __name__ == "__main__":
    main()
