#!/usr/bin/env python3
"""Copy a RoboMimic image HDF5 and store a chosen image key as agentview_image.

OpenX's RoboMimic RLDS builder and original DemInf image configs treat
obs/agentview_image as the third-person "agent" stream. Use this script when a
dataset contains the desired third-person stream under a different key but the
downstream pipeline should keep using camera=agent unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Source RoboMimic image HDF5.")
    parser.add_argument("--output", type=Path, required=True, help="Output HDF5 to write.")
    parser.add_argument(
        "--source-image-key",
        default="thirdperson_1_image",
        help="Image observation key to copy into agentview_image.",
    )
    parser.add_argument("--target-image-key", default="agentview_image")
    parser.add_argument("--expected-demos", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sorted_demo_keys(data_group: h5py.Group) -> list[str]:
    return sorted(data_group.keys(), key=lambda key: int(key.split("_")[-1]))


def copy_attrs(src, dst) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def replace_dataset(group: h5py.Group, target_key: str, source_dataset: h5py.Dataset) -> None:
    data = source_dataset[:]
    if target_key in group:
        del group[target_key]

    kwargs = {}
    if data.ndim >= 3:
        kwargs.update(compression="gzip", compression_opts=1)
    group.create_dataset(target_key, data=data, **kwargs)


def verify_alias(src_demo: h5py.Group, dst_demo: h5py.Group, group_name: str, source_key: str, target_key: str) -> None:
    src_arr = src_demo[group_name][source_key][:]
    dst_arr = dst_demo[group_name][target_key][:]
    if src_arr.shape != dst_arr.shape:
        raise AssertionError(
            f"{src_demo.name}/{group_name}: {target_key} shape {dst_arr.shape} != {source_key} shape {src_arr.shape}"
        )
    if not np.array_equal(src_arr, dst_arr):
        raise AssertionError(f"{src_demo.name}/{group_name}: {target_key} does not equal source {source_key}")


def main() -> None:
    args = parse_args()
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(args.output)
        args.output.unlink()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.input, "r") as src, h5py.File(args.output, "w") as dst:
        for key in src.keys():
            src.copy(key, dst)
        copy_attrs(src, dst)

        demo_keys = sorted_demo_keys(src["data"])
        if args.expected_demos is not None and len(demo_keys) != args.expected_demos:
            raise AssertionError(f"expected {args.expected_demos} demos, found {len(demo_keys)}")

        for demo_key in tqdm(demo_keys, desc=f"aliasing {args.source_image_key} -> {args.target_image_key}"):
            src_demo = src["data"][demo_key]
            dst_demo = dst["data"][demo_key]
            for group_name in ("obs", "next_obs"):
                if group_name not in src_demo:
                    continue
                if args.source_image_key not in src_demo[group_name]:
                    raise KeyError(
                        f"{args.input}:data/{demo_key}/{group_name} missing {args.source_image_key}; "
                        f"available keys: {sorted(src_demo[group_name].keys())}"
                    )
                replace_dataset(dst_demo[group_name], args.target_image_key, src_demo[group_name][args.source_image_key])
                verify_alias(src_demo, dst_demo, group_name, args.source_image_key, args.target_image_key)

        dst.attrs["source_path"] = str(args.input)
        dst.attrs["agentview_image_alias_source_key"] = args.source_image_key
        dst.attrs["agentview_image_alias_target_key"] = args.target_image_key
        dst.attrs["agentview_image_alias_note"] = (
            "agentview_image was intentionally overwritten so existing camera=agent DemInf configs read "
            f"{args.source_image_key}."
        )

    print(args.output)


if __name__ == "__main__":
    main()
