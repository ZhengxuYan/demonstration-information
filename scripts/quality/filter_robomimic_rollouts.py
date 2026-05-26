#!/usr/bin/env python3
"""Filter robomimic rollout HDF5s by success, return, and horizon."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-demos", type=int, default=1000)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def copy_attrs(src, dst) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def copy_group(src: h5py.Group, dst: h5py.Group) -> None:
    copy_attrs(src, dst)
    for key, item in src.items():
        if isinstance(item, h5py.Dataset):
            src.copy(item, dst, name=key)
        else:
            child = dst.create_group(key)
            copy_group(item, child)


def demo_score(path: Path, key: str) -> tuple:
    with h5py.File(path, "r") as f:
        demo = f["data"][key]
        success = int(demo.attrs.get("success", np.max(demo["rewards"][:]) > 0))
        ret = float(demo.attrs.get("return", np.sum(demo["rewards"][:]) if "rewards" in demo else success))
        horizon = int(demo.attrs.get("horizon", demo["actions"].shape[0]))
    return success, ret, horizon


def write_masks(f: h5py.File, demo_keys: list[str], valid_ratio: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    keys = np.asarray(demo_keys)
    perm = rng.permutation(len(keys))
    n_valid = int(round(len(keys) * valid_ratio))
    valid_idx = set(perm[:n_valid].tolist())
    train = [key.encode("utf-8") for i, key in enumerate(demo_keys) if i not in valid_idx]
    valid = [key.encode("utf-8") for i, key in enumerate(demo_keys) if i in valid_idx]
    mask = f.create_group("mask")
    mask.create_dataset("train", data=np.asarray(train))
    mask.create_dataset("valid", data=np.asarray(valid))


def main() -> None:
    args = parse_args()
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(args.output)
        args.output.unlink()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    candidates = []
    for path in args.inputs:
        with h5py.File(path, "r") as f:
            for key in f["data"].keys():
                success, ret, horizon = demo_score(path, key)
                candidates.append(
                    {
                        "path": path,
                        "key": key,
                        "success": success,
                        "return": ret,
                        "horizon": horizon,
                    }
                )
    candidates.sort(key=lambda row: (-row["success"], row["horizon"], -row["return"], str(row["path"]), row["key"]))
    selected = candidates[: args.num_demos]
    if len(selected) < args.num_demos:
        raise ValueError(f"Only found {len(selected)} demos, requested {args.num_demos}")

    total = 0
    new_keys = []
    with h5py.File(selected[0]["path"], "r") as first, h5py.File(args.output, "w") as out:
        copy_attrs(first, out)
        out.attrs["filtered_from_json"] = json.dumps([str(path) for path in args.inputs])
        data = out.create_group("data")
        copy_attrs(first["data"], data)
        for idx, row in enumerate(selected):
            new_key = f"demo_{idx}"
            with h5py.File(row["path"], "r") as src:
                copy_group(src["data"][row["key"]], data.create_group(new_key))
            data[new_key].attrs["source_hdf5"] = str(row["path"])
            data[new_key].attrs["source_demo_key"] = row["key"]
            total += int(data[new_key]["actions"].shape[0])
            new_keys.append(new_key)
        data.attrs["num_demos"] = len(new_keys)
        data.attrs["total"] = total
        write_masks(out, new_keys, valid_ratio=args.valid_ratio, seed=args.seed)

    successes = sum(row["success"] for row in selected)
    print(args.output)
    print(f"num_demos={len(selected)} successes={successes} success_rate={successes / len(selected):.3f} total={total}")


if __name__ == "__main__":
    main()
