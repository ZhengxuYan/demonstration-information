#!/usr/bin/env python3
"""Create robomimic BC datasets filtered by DemInf / MI episode score."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import h5py
import numpy as np


DEFAULT_DATASETS = ("ph_agentview", "400_agentview", "400_left_close_low", "400_mix")
DEFAULT_DROP_FRACTIONS = (0.0, 0.1, 0.2, 0.3, 0.4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episode-csv",
        type=Path,
        required=True,
        help="CSV with dataset, ep_idx, and score columns.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/iris/u/jasonyan/data/deminf_camera_view_datasets"),
        help="Root containing <dataset>/image.hdf5.",
    )
    parser.add_argument(
        "--source-hdf5",
        action="append",
        default=[],
        help="Explicit source mapping in the form dataset=/path/to/image.hdf5. Overrides --source-root for that dataset.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/iris/u/jasonyan/data/deminf_filtered_bc_datasets/mi_score"),
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--drop-fractions", nargs="+", type=float, default=list(DEFAULT_DROP_FRACTIONS))
    parser.add_argument(
        "--drop-side",
        choices=["low", "high"],
        default="low",
        help="Which end of the score distribution to remove. Default removes the lowest MI scores.",
    )
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--symlink-zero-drop",
        action="store_true",
        help="Symlink 0%% drop datasets instead of rewriting them with fresh train/valid masks.",
    )
    return parser.parse_args()


def sorted_demo_keys(data_group: h5py.Group) -> list[str]:
    def key_index(name: str) -> int:
        if name.startswith("demo_"):
            return int(name.removeprefix("demo_"))
        return int(name)

    return sorted(data_group.keys(), key=key_index)


def copy_attrs(src, dst) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def copy_group(src: h5py.Group, dst: h5py.Group) -> None:
    copy_attrs(src, dst)
    for key, item in src.items():
        if isinstance(item, h5py.Group):
            child = dst.create_group(key)
            copy_group(item, child)
        else:
            src.copy(item, dst, name=key)


def write_masks(out_file: h5py.File, demo_keys: list[str], valid_ratio: float, seed: int) -> None:
    if not 0.0 <= valid_ratio < 1.0:
        raise ValueError(f"valid_ratio must be in [0, 1); got {valid_ratio}")

    if "mask" in out_file:
        del out_file["mask"]

    if valid_ratio > 0.0 and len(demo_keys) > 1:
        rng = np.random.default_rng(seed)
        num_valid = max(1, int(round(valid_ratio * len(demo_keys))))
        valid_indexes = set(rng.choice(np.arange(len(demo_keys)), size=num_valid, replace=False).astype(int).tolist())
        train_keys = [key for idx, key in enumerate(demo_keys) if idx not in valid_indexes]
        valid_keys = [key for idx, key in enumerate(demo_keys) if idx in valid_indexes]
    else:
        train_keys = list(demo_keys)
        valid_keys = []

    mask = out_file.create_group("mask")
    mask.create_dataset("train", data=np.asarray([key.encode("utf-8") for key in train_keys], dtype="S"))
    mask.create_dataset("valid", data=np.asarray([key.encode("utf-8") for key in valid_keys], dtype="S"))
    out_file.attrs["mask_split_seed"] = int(seed)
    out_file.attrs["mask_valid_ratio"] = float(valid_ratio)


def read_scores(path: Path) -> dict[str, dict[int, dict[str, str]]]:
    by_dataset: dict[str, dict[int, dict[str, str]]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"dataset", "ep_idx", "score"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        for row in reader:
            dataset = row["dataset"]
            ep_idx = int(row["ep_idx"])
            score = float(row["score"])
            if not math.isfinite(score):
                raise ValueError(f"Non-finite score for {dataset} ep {ep_idx}: {score}")
            row = dict(row)
            row["score"] = str(score)
            by_dataset.setdefault(dataset, {})[ep_idx] = row
    return by_dataset


def parse_source_hdf5(values: list[str]) -> dict[str, Path]:
    out = {}
    for value in values:
        dataset, sep, path = value.partition("=")
        if not sep or not dataset or not path:
            raise ValueError(f"Invalid --source-hdf5 {value!r}; expected dataset=/path/to/image.hdf5")
        out[dataset] = Path(path)
    return out


def selected_indices(scores: dict[int, dict[str, str]], num_demos: int, drop_fraction: float, drop_side: str) -> tuple[list[int], list[int]]:
    if not 0.0 <= drop_fraction < 1.0:
        raise ValueError(f"drop_fraction must be in [0, 1); got {drop_fraction}")
    missing = sorted(set(range(num_demos)) - set(scores))
    if missing:
        raise ValueError(f"Missing scores for {len(missing)} episodes; first missing indices: {missing[:10]}")

    num_drop = int(round(drop_fraction * num_demos))
    ordered = sorted(range(num_demos), key=lambda idx: float(scores[idx]["score"]))
    dropped = ordered[:num_drop] if drop_side == "low" else ordered[-num_drop:] if num_drop else []
    dropped_set = set(dropped)
    kept = [idx for idx in range(num_demos) if idx not in dropped_set]
    return kept, list(dropped)


def maybe_symlink(src: Path, dst: Path, overwrite: bool) -> bool:
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            print(f"exists, skipping: {dst}")
            return True
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    try:
        dst.symlink_to(src)
        return True
    except OSError:
        return False


def write_filtered_dataset(
    src_path: Path,
    dst_path: Path,
    scores: dict[int, dict[str, str]],
    keep_indices: list[int],
    drop_indices: list[int],
    drop_fraction: float,
    drop_side: str,
    valid_ratio: float,
    split_seed: int,
    overwrite: bool,
) -> None:
    if dst_path.exists() or dst_path.is_symlink():
        if not overwrite:
            print(f"exists, skipping: {dst_path}")
            return
        if dst_path.is_symlink() or dst_path.is_file():
            dst_path.unlink()
        else:
            shutil.rmtree(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        copy_attrs(src, dst)
        src_keys = sorted_demo_keys(src["data"])
        data_out = dst.create_group("data")
        copy_attrs(src["data"], data_out)
        new_keys = []
        mapping = {}
        total = 0
        for new_idx, old_idx in enumerate(keep_indices):
            old_key = src_keys[old_idx]
            new_key = f"demo_{new_idx}"
            new_keys.append(new_key)
            mapping[new_key] = {
                "source_demo_key": old_key,
                "source_ep_idx": int(old_idx),
                "score": float(scores[old_idx]["score"]),
                "source": scores[old_idx].get("source", ""),
                "view": scores[old_idx].get("view", ""),
            }
            demo_out = data_out.create_group(new_key)
            copy_group(src["data"][old_key], demo_out)
            if "actions" in demo_out:
                total += int(demo_out["actions"].shape[0])

        data_out.attrs["num_demos"] = len(new_keys)
        data_out.attrs["total"] = total
        write_masks(dst, new_keys, valid_ratio=valid_ratio, seed=split_seed)
        dst.attrs["source_path"] = str(src_path)
        dst.attrs["mi_filter_drop_fraction"] = float(drop_fraction)
        dst.attrs["mi_filter_drop_side"] = drop_side
        dst.attrs["mi_filter_num_dropped"] = int(len(drop_indices))
        dst.attrs["mi_filter_num_kept"] = int(len(keep_indices))
        dst.attrs["mi_filter_kept_ep_indices_json"] = json.dumps([int(i) for i in keep_indices])
        dst.attrs["mi_filter_dropped_ep_indices_json"] = json.dumps([int(i) for i in drop_indices])
        dst.attrs["mi_filter_mapping_json"] = json.dumps(mapping, sort_keys=True)


def write_manifest_row(writer: csv.DictWriter, dataset: str, drop_fraction: float, path: Path, keep: list[int], drop: list[int]) -> None:
    writer.writerow(
        {
            "dataset": dataset,
            "drop_fraction": drop_fraction,
            "drop_percent": int(round(drop_fraction * 100)),
            "num_kept": len(keep),
            "num_dropped": len(drop),
            "dataset_path": str(path),
        }
    )


def main() -> None:
    args = parse_args()
    by_dataset = read_scores(args.episode_csv)
    source_hdf5 = parse_source_hdf5(args.source_hdf5)
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "manifest.csv"

    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "drop_fraction", "drop_percent", "num_kept", "num_dropped", "dataset_path"],
        )
        writer.writeheader()
        for dataset in args.datasets:
            src_path = source_hdf5.get(dataset, args.source_root / dataset / "image.hdf5")
            if not src_path.exists():
                raise FileNotFoundError(src_path)
            if dataset not in by_dataset:
                raise ValueError(f"No scores found for dataset={dataset} in {args.episode_csv}")
            with h5py.File(src_path, "r") as src:
                num_demos = len(src["data"])

            for drop_fraction in args.drop_fractions:
                keep, drop = selected_indices(by_dataset[dataset], num_demos, drop_fraction, args.drop_side)
                drop_percent = int(round(drop_fraction * 100))
                dst_path = args.output_root / dataset / f"drop_{drop_percent:02d}" / "image.hdf5"
                if drop_percent == 0 and args.symlink_zero_drop:
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    if not maybe_symlink(src_path, dst_path, overwrite=args.overwrite):
                        write_filtered_dataset(
                            src_path,
                            dst_path,
                            by_dataset[dataset],
                            keep,
                            drop,
                            drop_fraction,
                            args.drop_side,
                            args.valid_ratio,
                            args.split_seed,
                            args.overwrite,
                        )
                else:
                    write_filtered_dataset(
                        src_path,
                        dst_path,
                        by_dataset[dataset],
                        keep,
                        drop,
                        drop_fraction,
                        args.drop_side,
                        args.valid_ratio,
                        args.split_seed,
                        args.overwrite,
                    )
                write_manifest_row(writer, dataset, drop_fraction, dst_path, keep, drop)
                print(f"{dataset} drop={drop_percent:02d}: kept={len(keep)} dropped={len(drop)} -> {dst_path}")

    print(manifest_path)


if __name__ == "__main__":
    main()
