#!/usr/bin/env python3
"""Replace obs/agentview_image with left_close_low renders for a RoboMimic HDF5.

The OpenX RoboMimic RLDS builder always reads obs/agentview_image as the
third-person "agent" image. To run the same pipeline with the left_close_low
view, this script copies a compatible image.hdf5 and overwrites agentview_image
with frames rendered from the left_close_low camera pose, while keeping the key
name unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from prepare_policy_view_datasets import (
    LEFT_CLOSE_LOW_POS,
    LEFT_CLOSE_LOW_QUAT_WXYZ,
    copy_attrs,
    copy_group,
    create_env,
    load_env_meta,
    sorted_demo_keys,
    upsert_image_pair,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render-height", type=int, default=84)
    parser.add_argument("--render-width", type=int, default=84)
    parser.add_argument("--render-gpu-device-id", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def render_left_close_low(env, states: np.ndarray, height: int, width: int) -> np.ndarray:
    cam_id = env.env.sim.model.camera_name2id("agentview")
    original_pos = env.env.sim.model.cam_pos[cam_id].copy()
    original_quat = env.env.sim.model.cam_quat[cam_id].copy()
    frames = []
    try:
        for state in states:
            env.env.sim.set_state_from_flattened(state)
            env.env.sim.model.cam_pos[cam_id] = LEFT_CLOSE_LOW_POS
            env.env.sim.model.cam_quat[cam_id] = LEFT_CLOSE_LOW_QUAT_WXYZ
            env.env.sim.forward()
            frame = env.render(mode="rgb_array", height=height, width=width, camera_name="agentview")
            frames.append(np.asarray(frame, dtype=np.uint8))
    finally:
        env.env.sim.model.cam_pos[cam_id] = original_pos
        env.env.sim.model.cam_quat[cam_id] = original_quat
        env.env.sim.forward()
    return np.stack(frames, axis=0)


def create_left_close_low_env(input_path: Path, height: int, width: int, render_gpu_device_id: int | None):
    with h5py.File(input_path, "r") as src:
        env_meta = load_env_meta(src, input_path, None)
    if render_gpu_device_id is not None:
        env_meta = json.loads(json.dumps(env_meta))
        env_meta.setdefault("env_kwargs", {})["render_gpu_device_id"] = int(render_gpu_device_id)
    env = create_env(env_meta, height, width, ["agentview", "robot0_eye_in_hand"])
    env.reset()
    return env


def main() -> None:
    args = parse_args()
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(args.output)
        args.output.unlink()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    env = create_left_close_low_env(
        args.input,
        height=args.render_height,
        width=args.render_width,
        render_gpu_device_id=args.render_gpu_device_id,
    )

    with h5py.File(args.input, "r") as src, h5py.File(args.output, "w") as dst:
        copy_attrs(src, dst)
        data_out = dst.create_group("data")
        copy_attrs(src["data"], data_out)
        demo_keys = sorted_demo_keys(src["data"])
        for demo_key in tqdm(demo_keys, desc="rendering left_close_low"):
            demo_out = data_out.create_group(demo_key)
            copy_group(src["data"][demo_key], demo_out)
            frames = render_left_close_low(
                env,
                states=demo_out["states"][:],
                height=args.render_height,
                width=args.render_width,
            )
            upsert_image_pair(demo_out, "agentview_image", frames)

        if "mask" in src:
            copy_group(src["mask"], dst.create_group("mask"))
        dst.attrs["source_path"] = str(args.input)
        dst.attrs["agentview_image_replaced_with"] = "left_close_low"
        dst.attrs["left_close_low_pos"] = LEFT_CLOSE_LOW_POS
        dst.attrs["left_close_low_quat_wxyz"] = LEFT_CLOSE_LOW_QUAT_WXYZ

    print(args.output)


if __name__ == "__main__":
    main()
