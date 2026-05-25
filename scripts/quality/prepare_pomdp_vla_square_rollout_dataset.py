#!/usr/bin/env python3
"""Merge POMDP-VLA Square rollout HDF5s and ensure robomimic image observations."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from prepare_policy_view_datasets import (
    copy_attrs,
    copy_group,
    create_env,
    ensure_required_observations,
    load_env_meta,
    sorted_demo_keys,
    validate_source,
    write_masks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=None, help="Directory containing source HDF5 files.")
    parser.add_argument("--input-hdf5", type=Path, action="append", default=[], help="Source HDF5. Repeatable.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--glob", default="*.hdf5")
    parser.add_argument("--max-demos", type=int, default=0, help="0 keeps all demos.")
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--render-height", type=int, default=84)
    parser.add_argument("--render-width", type=int, default=84)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def discover_inputs(input_root: Path | None, input_hdf5: list[Path], pattern: str) -> list[Path]:
    paths = list(input_hdf5)
    if input_root is not None:
        paths.extend(sorted(input_root.rglob(pattern)))
    deduped = []
    seen = set()
    for path in paths:
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        if path.name.startswith(".") or not path.is_file():
            continue
        deduped.append(path)
    if not deduped:
        raise FileNotFoundError("No input HDF5 files found.")
    return deduped


def has_required_obs(demo: h5py.Group) -> bool:
    if "obs" not in demo:
        return False
    obs = demo["obs"]
    required = (
        "agentview_image",
        "robot0_eye_in_hand_image",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    )
    return all(key in obs for key in required)


def action_len(demo: h5py.Group) -> int:
    if "actions" in demo:
        return int(demo["actions"].shape[0])
    return 0


def sanitize_env_meta(env_meta: dict) -> dict:
    """Drop newer robosuite kwargs that robosuite 1.2.0 cannot replay."""
    env_meta = copy.deepcopy(env_meta)
    env_kwargs = env_meta.get("env_kwargs", {})
    for key in ("lite_physics",):
        env_kwargs.pop(key, None)
    controller_config = env_kwargs.get("controller_configs")
    if isinstance(controller_config, dict) and controller_config.get("type") == "BASIC":
        body_parts = controller_config.get("body_parts", {})
        if isinstance(body_parts, dict):
            for part_key in ("right", "arm0", "robot0_right"):
                part_config = body_parts.get(part_key)
                if isinstance(part_config, dict) and part_config.get("type"):
                    controller_config = dict(part_config)
                    break
            else:
                for part_config in body_parts.values():
                    if isinstance(part_config, dict) and part_config.get("type"):
                        controller_config = dict(part_config)
                        break
        env_kwargs["controller_configs"] = controller_config
    controller_config = env_kwargs.get("controller_configs")
    if isinstance(controller_config, dict):
        controller_config.setdefault("interpolation", None)
    return env_meta


def copy_one_demo(
    src_path: Path,
    src_demo_key: str,
    dst_data: h5py.Group,
    dst_key: str,
    env,
    height: int,
    width: int,
) -> int:
    with h5py.File(src_path, "r") as src:
        demo_out = dst_data.create_group(dst_key)
        copy_group(src["data"][src_demo_key], demo_out)

    if not has_required_obs(demo_out):
        if env is None:
            raise RuntimeError(f"{src_path}:{src_demo_key} needs rendered obs but env was not initialized")
        if "states" not in demo_out:
            raise KeyError(f"{src_path}:{src_demo_key} needs rendered obs but has no states dataset")
        ensure_required_observations(
            demo_out,
            env,
            np.asarray(demo_out["states"]),
            "agentview_image",
            height,
            width,
            model_xml=demo_out.attrs.get("model_file"),
        )
    return action_len(demo_out)


def main() -> None:
    args = parse_args()
    inputs = discover_inputs(args.input_root, args.input_hdf5, args.glob)
    for path in inputs:
        validate_source(path, expected_action_dim=7)

    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(args.output)
        args.output.unlink()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    remaining = args.max_demos if args.max_demos > 0 else None
    mapping = {}
    new_keys = []
    total = 0

    with h5py.File(inputs[0], "r") as first, h5py.File(args.output, "w") as dst:
        copy_attrs(first, dst)
        data_out = dst.create_group("data")
        copy_attrs(first["data"], data_out)

        new_idx = 0
        for src_path in inputs:
            with h5py.File(src_path, "r") as src:
                env_meta = sanitize_env_meta(load_env_meta(src, src_path, None))
                demo_keys = sorted_demo_keys(src["data"])
                needs_env = any(not has_required_obs(src["data"][demo_key]) for demo_key in demo_keys)
            env = None
            if needs_env:
                env = create_env(env_meta, args.render_height, args.render_width, ["agentview", "robot0_eye_in_hand"])
                env.reset()
            for src_demo_key in tqdm(demo_keys, desc=src_path.name):
                if remaining is not None and remaining <= 0:
                    break
                dst_key = f"demo_{new_idx}"
                total += copy_one_demo(
                    src_path=src_path,
                    src_demo_key=src_demo_key,
                    dst_data=data_out,
                    dst_key=dst_key,
                    env=env,
                    height=args.render_height,
                    width=args.render_width,
                )
                mapping[dst_key] = {"source_hdf5": str(src_path), "source_demo_key": src_demo_key}
                new_keys.append(dst_key)
                new_idx += 1
                if remaining is not None:
                    remaining -= 1
            if remaining is not None and remaining <= 0:
                break

        data_out.attrs["num_demos"] = len(new_keys)
        data_out.attrs["total"] = total
        write_masks(dst, new_keys, valid_ratio=args.valid_ratio, seed=args.split_seed)
        dst.attrs["source_hdf5s_json"] = json.dumps([str(path) for path in inputs])
        dst.attrs["demo_key_mapping_json"] = json.dumps(mapping, sort_keys=True)

    print(args.output)
    print(f"num_demos={len(new_keys)} total={total}")


if __name__ == "__main__":
    main()
