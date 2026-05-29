#!/usr/bin/env python3
"""Validate a PVLA -> RoboMimic-MH-compatible HDF5 conversion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


REAL_OBS_KEYS = (
    "agentview_image",
    "robot0_eye_in_hand_image",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)
PLACEHOLDER_OBS_SHAPES = {
    "robot0_joint_pos": (7,),
    "robot0_joint_vel": (7,),
    "object": (44,),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converted", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--reference-hdf5", type=Path, default=None)
    parser.add_argument("--max-demos", type=int, default=0, help="0 validates all demos.")
    parser.add_argument("--expect-no-model-file", action="store_true")
    return parser.parse_args()


def sorted_demo_keys(data_group: h5py.Group) -> list[str]:
    return sorted(data_group.keys(), key=lambda key: int(key.split("_")[-1]))


def read_mask(mask_group: h5py.Group, key: str) -> list[str]:
    if key not in mask_group:
        return []
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in mask_group[key][:]]


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def assert_shifted_next(obs: h5py.Group, next_obs: h5py.Group, key: str) -> None:
    values = obs[key][:]
    next_values = next_obs[key][:]
    expected = np.concatenate([values[1:], values[-1:]], axis=0) if len(values) else values.copy()
    assert_true(np.array_equal(next_values, expected), f"next_obs/{key} is not shifted obs/{key}")


def compare_env_args(converted: h5py.File, reference: h5py.File | None) -> None:
    converted_env = json.loads(converted["data"].attrs["env_args"])
    env_kwargs = converted_env["env_kwargs"]
    assert_true(converted_env["env_name"] == "NutAssemblySquare", "env_name is not NutAssemblySquare")
    assert_true(env_kwargs.get("use_camera_obs") is True, "use_camera_obs is not true")
    assert_true(env_kwargs.get("camera_names") == ["agentview", "robot0_eye_in_hand"], "camera_names mismatch")
    assert_true(env_kwargs.get("camera_heights") == 84, "camera_heights mismatch")
    assert_true(env_kwargs.get("camera_widths") == 84, "camera_widths mismatch")
    if reference is not None:
        ref_env = json.loads(reference["data"].attrs["env_args"])
        assert_true(converted_env["env_name"] == ref_env["env_name"], "env_name differs from reference")


def validate_masks(f: h5py.File, demos: list[str]) -> None:
    assert_true("mask" in f, "missing mask group")
    mask = f["mask"]
    train = read_mask(mask, "train")
    valid = read_mask(mask, "valid")
    assert_true(train, "mask/train is empty")
    assert_true(valid, "mask/valid is empty")
    assert_true(set(train).isdisjoint(valid), "train and valid masks overlap")
    assert_true(set(train + valid) == set(demos), "train+valid masks do not cover all demos")
    for key in ("better", "okay", "worse"):
        assert_true(key in mask, f"missing mask/{key}")
    finite_quality_mask = None
    for key in ("better", "okay", "worse"):
        keys = read_mask(mask, key)
        if keys:
            finite_quality_mask = key
            assert_true(set(keys) == set(demos), f"mask/{key} does not cover all demos")
            break
    assert_true(finite_quality_mask is not None, "missing finite quality mask better/okay/worse")
    print(f"quality_mask={finite_quality_mask} train={len(train)} valid={len(valid)}")


def validate_demo(
    demo_key: str,
    demo: h5py.Group,
    source_demo: h5py.Group | None,
    expect_no_model_file: bool,
) -> int:
    assert_true("actions" in demo, f"{demo_key} missing actions")
    assert_true("states" in demo, f"{demo_key} missing states")
    assert_true("obs" in demo, f"{demo_key} missing obs")
    assert_true("next_obs" in demo, f"{demo_key} missing next_obs")
    length = int(demo["actions"].shape[0])
    assert_true(length > 0, f"{demo_key} has zero length")
    assert_true(demo.attrs.get("num_samples") == length, f"{demo_key} num_samples mismatch")
    if expect_no_model_file:
        assert_true("model_file" not in demo.attrs, f"{demo_key} still has model_file attr")

    obs = demo["obs"]
    next_obs = demo["next_obs"]
    for key in REAL_OBS_KEYS:
        assert_true(key in obs, f"{demo_key}/obs missing {key}")
        assert_true(key in next_obs, f"{demo_key}/next_obs missing {key}")
        assert_true(obs[key].shape[0] == length, f"{demo_key}/obs/{key} length mismatch")
        assert_true(next_obs[key].shape == obs[key].shape, f"{demo_key}/next_obs/{key} shape mismatch")
        assert_shifted_next(obs, next_obs, key)

    for key, trailing_shape in PLACEHOLDER_OBS_SHAPES.items():
        assert_true(key in obs, f"{demo_key}/obs missing placeholder {key}")
        assert_true(key in next_obs, f"{demo_key}/next_obs missing placeholder {key}")
        assert_true(obs[key].shape == (length, *trailing_shape), f"{demo_key}/obs/{key} shape mismatch")
        assert_true(np.allclose(obs[key][:], 0), f"{demo_key}/obs/{key} is not zero placeholder")
        assert_true(np.allclose(next_obs[key][:], 0), f"{demo_key}/next_obs/{key} is not zero placeholder")

    assert_true("rewards" in demo, f"{demo_key} missing rewards")
    assert_true("dones" in demo, f"{demo_key} missing dones")
    rewards = demo["rewards"][:]
    dones = demo["dones"][:]
    assert_true(rewards.shape == (length,), f"{demo_key} rewards shape mismatch")
    assert_true(dones.shape == (length,), f"{demo_key} dones shape mismatch")
    assert_true(np.allclose(rewards[:-1], 0) and np.isclose(rewards[-1], 1), f"{demo_key} rewards not demo-style")
    assert_true(np.all(dones[:-1] == 0) and dones[-1] == 1, f"{demo_key} dones not demo-style")

    if source_demo is not None:
        assert_true(np.array_equal(demo["actions"][:], source_demo["actions"][:]), f"{demo_key} actions changed")
        assert_true(np.array_equal(demo["states"][:], source_demo["states"][:]), f"{demo_key} states changed")
        for key in REAL_OBS_KEYS:
            assert_true(np.array_equal(obs[key][:], source_demo["obs"][key][:]), f"{demo_key}/obs/{key} changed")
    return length


def main() -> None:
    args = parse_args()
    with h5py.File(args.converted, "r") as converted:
        reference = h5py.File(args.reference_hdf5, "r") if args.reference_hdf5 else None
        source = h5py.File(args.source, "r") if args.source else None
        try:
            compare_env_args(converted, reference)
            demos = sorted_demo_keys(converted["data"])
            expected_num = int(converted["data"].attrs["num_demos"])
            assert_true(expected_num == len(demos), "data.attrs num_demos mismatch")
            validate_masks(converted, demos)

            mapping = json.loads(converted.attrs.get("demo_key_mapping_json", "{}"))
            selected = demos if args.max_demos <= 0 else demos[: args.max_demos]
            total = 0
            for demo_key in tqdm(selected, desc="validating demos"):
                source_demo = None
                if source is not None:
                    source_key = mapping.get(demo_key, demo_key)
                    source_demo = source["data"][source_key]
                total += validate_demo(
                    demo_key,
                    converted["data"][demo_key],
                    source_demo,
                    expect_no_model_file=args.expect_no_model_file,
                )
            if args.max_demos <= 0:
                assert_true(int(converted["data"].attrs["total"]) == total, "data.attrs total mismatch")
            print(f"ok converted={args.converted} demos_checked={len(selected)} frames_checked={total}")
        finally:
            if reference is not None:
                reference.close()
            if source is not None:
                source.close()


if __name__ == "__main__":
    main()
