#!/usr/bin/env python3
"""Canonicalize POMDP-VLA Square rollouts into a RoboMimic-MH-like HDF5.

The POMDP-VLA rollouts already contain usable RoboMimic image observations and
actions, but their metadata and optional fields differ from the original
RoboMimic Square-MH image dataset. This script writes a new HDF5 with the
fields consumed by the existing OpenX / RoboMimic pipeline:

- data/demo_*/obs and next_obs image + low-dimensional keys
- data/demo_*/rewards and dones
- mask/train, mask/valid, and a quality mask so RLDS quality_score is finite
- data.attrs env_args with camera metadata set consistently

It can borrow env_args from an original MH image.hdf5 file when available.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from tqdm import tqdm


REQUIRED_OBS_KEYS = (
    "agentview_image",
    "robot0_eye_in_hand_image",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)
ZERO_LOW_DIM_SPECS = {
    "robot0_joint_pos": (7,),
    "robot0_joint_vel": (7,),
    "object": (44,),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reference-hdf5",
        type=Path,
        default=None,
        help="Optional original RoboMimic MH image.hdf5 to borrow data.attrs['env_args'] from.",
    )
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--quality-mask", choices=["better", "okay", "worse"], default="better")
    parser.add_argument(
        "--model-file-policy",
        choices=["drop", "keep"],
        default="drop",
        help="Drop per-demo model_file attrs by default because robosuite 1.5 XML is not loadable in the 1.2 stack.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sorted_demo_keys(data_group: h5py.Group) -> list[str]:
    return sorted(data_group.keys(), key=lambda key: int(key.split("_")[-1]))


def copy_attrs(src: Any, dst: Any, skip: set[str] | None = None) -> None:
    skip = skip or set()
    for key, value in src.attrs.items():
        if key in skip:
            continue
        dst.attrs[key] = value


def copy_group(src: h5py.Group, dst: h5py.Group, skip_attrs: set[str] | None = None) -> None:
    copy_attrs(src, dst, skip=skip_attrs)
    for key, item in src.items():
        if isinstance(item, h5py.Group):
            copy_group(item, dst.create_group(key))
        else:
            src.copy(item, dst, name=key)


def decode_env_args(raw: Any) -> dict:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not isinstance(raw, str):
        raise TypeError(f"env_args must be str or bytes, got {type(raw)}")
    return json.loads(raw)


def load_env_args(input_file: h5py.File, reference_hdf5: Path | None) -> dict:
    if reference_hdf5 is not None:
        with h5py.File(reference_hdf5, "r") as ref:
            return decode_env_args(ref["data"].attrs["env_args"])
    return decode_env_args(input_file["data"].attrs["env_args"])


def canonicalize_env_args(env_args: dict) -> str:
    env_args = copy.deepcopy(env_args)
    env_args["env_name"] = "NutAssemblySquare"
    env_kwargs = env_args.setdefault("env_kwargs", {})
    env_kwargs["use_camera_obs"] = True
    env_kwargs["use_object_obs"] = True
    env_kwargs["camera_names"] = ["agentview", "robot0_eye_in_hand"]
    env_kwargs["camera_heights"] = 84
    env_kwargs["camera_widths"] = 84
    env_kwargs.setdefault("reward_shaping", False)
    env_kwargs.pop("lite_physics", None)
    env_args["type"] = env_args.get("type", 1)
    return json.dumps(env_args)


def assert_required_source_obs(demo: h5py.Group, source: str) -> None:
    if "obs" not in demo:
        raise KeyError(f"{source} is missing obs group")
    missing = [key for key in REQUIRED_OBS_KEYS if key not in demo["obs"]]
    if missing:
        raise KeyError(f"{source}/obs is missing required keys: {missing}")


def upsert_dataset(group: h5py.Group, key: str, value: np.ndarray, compression: bool = False) -> None:
    if key in group:
        del group[key]
    kwargs = {"compression": "gzip", "compression_opts": 1} if compression else {}
    group.create_dataset(key, data=value, **kwargs)


def shifted_next(values: np.ndarray) -> np.ndarray:
    if values.shape[0] == 0:
        return values.copy()
    return np.concatenate([values[1:], values[-1:]], axis=0)


def ensure_next_obs(demo: h5py.Group) -> None:
    obs = demo["obs"]
    next_obs = demo.require_group("next_obs")
    for key in obs.keys():
        if key in next_obs and next_obs[key].shape == obs[key].shape:
            continue
        values = obs[key][:]
        compression = values.dtype == np.uint8 and values.ndim == 4
        upsert_dataset(next_obs, key, shifted_next(values), compression=compression)


def ensure_zero_lowdim(demo: h5py.Group, length: int) -> None:
    obs = demo["obs"]
    next_obs = demo.require_group("next_obs")
    for key, trailing_shape in ZERO_LOW_DIM_SPECS.items():
        shape = (length, *trailing_shape)
        if key not in obs:
            obs.create_dataset(key, data=np.zeros(shape, dtype=np.float32))
        if key not in next_obs:
            next_obs.create_dataset(key, data=shifted_next(obs[key][:].astype(np.float32)))


def ensure_rewards_dones(demo: h5py.Group, length: int) -> None:
    if "rewards" not in demo:
        rewards = np.zeros((length,), dtype=np.float32)
        if length:
            rewards[-1] = 1.0
        demo.create_dataset("rewards", data=rewards)
    if "dones" not in demo:
        dones = np.zeros((length,), dtype=np.int64)
        if length:
            dones[-1] = 1
        demo.create_dataset("dones", data=dones)


def write_masks(out_file: h5py.File, demo_keys: list[str], valid_ratio: float, seed: int, quality_mask: str) -> None:
    if not 0.0 <= valid_ratio < 1.0:
        raise ValueError(f"valid_ratio must be in [0, 1), got {valid_ratio}")
    if "mask" in out_file:
        del out_file["mask"]
    mask = out_file.create_group("mask")

    rng = np.random.default_rng(seed)
    if valid_ratio > 0 and len(demo_keys) > 1:
        num_valid = max(1, int(round(valid_ratio * len(demo_keys))))
        num_valid = min(num_valid, len(demo_keys) - 1)
        valid_indexes = set(rng.choice(np.arange(len(demo_keys)), size=num_valid, replace=False).astype(int).tolist())
    else:
        valid_indexes = set()
    train = [key for i, key in enumerate(demo_keys) if i not in valid_indexes]
    valid = [key for i, key in enumerate(demo_keys) if i in valid_indexes]

    def encoded(keys: list[str]) -> np.ndarray:
        return np.asarray([key.encode("utf-8") for key in keys], dtype="S")

    mask.create_dataset("train", data=encoded(train))
    mask.create_dataset("valid", data=encoded(valid))
    mask.create_dataset(quality_mask, data=encoded(demo_keys))
    out_file.attrs["mask_valid_ratio"] = float(valid_ratio)
    out_file.attrs["mask_split_seed"] = int(seed)
    out_file.attrs["quality_mask"] = quality_mask


def canonicalize(input_path: Path, output_path: Path, reference_hdf5: Path | None, args: argparse.Namespace) -> None:
    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(output_path)
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_path, "r") as src, h5py.File(output_path, "w") as dst:
        copy_attrs(src, dst)
        data_out = dst.create_group("data")
        copy_attrs(src["data"], data_out, skip={"env_args", "num_demos", "total"})
        data_out.attrs["env_args"] = canonicalize_env_args(load_env_args(src, reference_hdf5))

        source_demo_keys = sorted_demo_keys(src["data"])
        output_demo_keys = []
        total = 0
        mapping = {}
        for new_idx, old_key in enumerate(tqdm(source_demo_keys, desc="canonicalizing demos")):
            old_demo = src["data"][old_key]
            assert_required_source_obs(old_demo, f"{input_path}:data/{old_key}")
            new_key = f"demo_{new_idx}"
            skip_attrs = {"model_file"} if args.model_file_policy == "drop" else set()
            demo_out = data_out.create_group(new_key)
            copy_group(old_demo, demo_out, skip_attrs=skip_attrs)

            length = int(demo_out["actions"].shape[0])
            ensure_next_obs(demo_out)
            ensure_zero_lowdim(demo_out, length)
            ensure_rewards_dones(demo_out, length)
            demo_out.attrs["num_samples"] = length

            output_demo_keys.append(new_key)
            total += length
            mapping[new_key] = old_key

        data_out.attrs["num_demos"] = len(output_demo_keys)
        data_out.attrs["total"] = total
        write_masks(dst, output_demo_keys, args.valid_ratio, args.split_seed, args.quality_mask)
        dst.attrs["source_path"] = str(input_path)
        dst.attrs["reference_hdf5"] = "" if reference_hdf5 is None else str(reference_hdf5)
        dst.attrs["demo_key_mapping_json"] = json.dumps(mapping, sort_keys=True)
        dst.attrs["canonicalized_for"] = "robomimic_square_mh_openx"


def main() -> None:
    args = parse_args()
    canonicalize(args.input, args.output, args.reference_hdf5, args)
    print(args.output)


if __name__ == "__main__":
    main()
