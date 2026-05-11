#!/usr/bin/env python3
"""Create expert200 non-abs BC datasets from repaired abs datasets.

The repaired expert200 files keep the original delta actions in actions_delta
and overwrite actions with robomimic absolute actions. This helper copies those
files and restores actions from actions_delta so BC training can use the same
non-abs action convention as PH and MH.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np


DEFAULT_SRC_ROOT = Path("/iris/u/jasonyan/data/policy_view_experiments/expert200_random_post_bc")
DEFAULT_DST_ROOT = Path("/iris/u/jasonyan/data/policy_view_experiments/expert200_random_post_bc")
DEFAULT_ENV_META = Path("/iris/u/jasonyan/data/diffusion_policy/robomimic/datasets/square/ph/image.hdf5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-root", type=Path, default=DEFAULT_SRC_ROOT)
    parser.add_argument("--dst-root", type=Path, default=DEFAULT_DST_ROOT)
    parser.add_argument("--env-meta-source", type=Path, default=DEFAULT_ENV_META)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sorted_demo_keys(data_group) -> list[str]:
    return sorted(data_group.keys(), key=lambda key: int(key.split("_")[-1]))


def load_env_args(path: Path) -> str:
    with h5py.File(path, "r") as f:
        return f["data"].attrs["env_args"]


def restore_delta_actions(path: Path, env_args: str, env_meta_source: Path) -> None:
    with h5py.File(path, "r+") as f:
        f["data"].attrs["env_args"] = env_args
        for demo_key in sorted_demo_keys(f["data"]):
            demo = f["data"][demo_key]
            if "actions_delta" not in demo:
                raise KeyError(f"{path}:{demo_key} is missing actions_delta")
            actions_delta = demo["actions_delta"][:]
            if actions_delta.shape != demo["actions"].shape:
                raise ValueError(
                    f"{path}:{demo_key} actions_delta shape {actions_delta.shape} "
                    f"!= actions shape {demo['actions'].shape}"
                )
            demo["actions"][:] = actions_delta
        f.attrs["nonabs_source"] = "actions_delta"
        f.attrs["nonabs_env_meta_source"] = str(env_meta_source)


def verify_actions_equal_delta(path: Path) -> None:
    with h5py.File(path, "r") as f:
        for demo_key in sorted_demo_keys(f["data"]):
            demo = f["data"][demo_key]
            if not np.array_equal(demo["actions"][:], demo["actions_delta"][:]):
                max_abs = float(np.max(np.abs(demo["actions"][:] - demo["actions_delta"][:])))
                raise AssertionError(f"{path}:{demo_key} actions != actions_delta; max_abs={max_abs}")


def copy_restore(src: Path, dst: Path, env_args: str, env_meta_source: Path, overwrite: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists():
        if not overwrite:
            print(f"exists, verifying without overwrite: {dst}")
            verify_actions_equal_delta(dst)
            return
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    restore_delta_actions(dst, env_args, env_meta_source)
    verify_actions_equal_delta(dst)
    print(dst)


def main() -> None:
    args = parse_args()
    env_args = load_env_args(args.env_meta_source)
    specs = [
        (
            args.src_root / "expert200_random_post_agent_wrist_image_abs.hdf5",
            args.dst_root / "expert200_random_post_agent_wrist_image.hdf5",
        ),
        (
            args.src_root / "expert200_random_post_left_close_low_wrist_image_abs.hdf5",
            args.dst_root / "expert200_random_post_left_close_low_wrist_image.hdf5",
        ),
    ]
    print(json.dumps({"src_root": str(args.src_root), "dst_root": str(args.dst_root)}, indent=2))
    for src, dst in specs:
        copy_restore(src, dst, env_args, args.env_meta_source, args.overwrite)


if __name__ == "__main__":
    main()
