#!/usr/bin/env python3
"""Prepare the three camera-view HDF5 datasets requested for DemInf.

Outputs:
  ph_agentview/image.hdf5
    200 PH episodes with agentview_image.
  400_agentview/image.hdf5
    200 PH episodes + 200 rollout episodes, all with agentview_image.
  400_mix/image.hdf5
    Same 400 episodes, shuffled, with half rendered/stored as agentview_image
    from agentview and half rendered/stored as agentview_image from left_close_low.

The RLDS builder reads obs/agentview_image, so mixed-view episodes intentionally
store the selected third-person view under agentview_image.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from prepare_policy_view_datasets import (
    LEFT_CLOSE_LOW_POS,
    LEFT_CLOSE_LOW_QUAT_WXYZ,
    copy_attrs,
    copy_group,
    create_env,
    ensure_required_observations,
    load_env_meta,
    reset_env_to_model,
    sorted_demo_keys,
    upsert_image_pair,
    validate_source,
    write_masks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ph-image", type=Path, required=True)
    parser.add_argument("--rollout-image", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--num-ph", type=int, default=200)
    parser.add_argument("--num-rollouts", type=int, default=200)
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--render-height", type=int, default=84)
    parser.add_argument("--render-width", type=int, default=84)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def selected_indices(path: Path, count: int) -> list[int]:
    with h5py.File(path, "r") as f:
        n = len(f["data"])
    if count > n:
        raise ValueError(f"{path} only has {n} demos; requested {count}")
    return list(range(count))


def render_left_close_low_as_agentview(env, states: np.ndarray, height: int, width: int, model_xml=None) -> np.ndarray:
    reset_env_to_model(env, model_xml)
    cam_id = env.env.sim.model.camera_name2id("agentview")
    frames = []
    for state in states:
        env.env.sim.set_state_from_flattened(state)
        env.env.sim.model.cam_pos[cam_id] = LEFT_CLOSE_LOW_POS
        env.env.sim.model.cam_quat[cam_id] = LEFT_CLOSE_LOW_QUAT_WXYZ
        env.env.sim.forward()
        frame = env.render(mode="rgb_array", height=height, width=width, camera_name="agentview")
        frames.append(np.asarray(frame, dtype=np.uint8))
    return np.stack(frames, axis=0)


def copy_episode(
    src_path: Path,
    src_index: int,
    dst_data,
    dst_key: str,
    env,
    height: int,
    width: int,
    use_left_close_low: bool,
) -> dict[str, object]:
    with h5py.File(src_path, "r") as src:
        src_keys = sorted_demo_keys(src["data"])
        src_key = src_keys[src_index]
        demo_out = dst_data.create_group(dst_key)
        copy_group(src["data"][src_key], demo_out)
        states = demo_out["states"][:]
        model_xml = demo_out.attrs.get("model_file")

    ensure_required_observations(
        demo_out,
        env,
        states,
        "agentview_image",
        height,
        width,
        model_xml=model_xml,
    )
    if use_left_close_low:
        frames = render_left_close_low_as_agentview(env, states, height, width, model_xml=model_xml)
        upsert_image_pair(demo_out, "agentview_image", frames)

    return {
        "source_path": str(src_path),
        "source_demo_index": int(src_index),
        "source_demo_key": src_key,
        "view": "left_close_low_as_agentview" if use_left_close_low else "agentview",
    }


def build_dataset(
    dst_path: Path,
    ph_path: Path,
    rollout_path: Path | None,
    ph_indices: list[int],
    rollout_indices: list[int],
    mix_views: bool,
    shuffle: bool,
    seed: int,
    height: int,
    width: int,
    overwrite: bool,
) -> None:
    if dst_path.exists():
        if not overwrite:
            print(f"exists, skipping: {dst_path}")
            return
        dst_path.unlink()
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(ph_path, "r") as src:
        env_meta = load_env_meta(src, ph_path, None)
    env = create_env(env_meta, height, width, ["agentview", "robot0_eye_in_hand"])
    env.reset()

    items = [("ph", ph_path, idx) for idx in ph_indices]
    if rollout_path is not None:
        items.extend(("rollout", rollout_path, idx) for idx in rollout_indices)
    if shuffle:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(items)).astype(int).tolist()
        items = [items[i] for i in order]

    left_indexes = set()
    if mix_views:
        rng = np.random.default_rng(seed + 17)
        left_indexes = set(rng.choice(np.arange(len(items)), size=len(items) // 2, replace=False).astype(int).tolist())

    metadata = {}
    with h5py.File(ph_path, "r") as src, h5py.File(dst_path, "w") as dst:
        copy_attrs(src, dst)
        data_out = dst.create_group("data")
        copy_attrs(src["data"], data_out)
        new_keys = []
        for new_idx, (_, src_path, src_index) in enumerate(items):
            dst_key = f"demo_{new_idx}"
            new_keys.append(dst_key)
            metadata[dst_key] = copy_episode(
                src_path=src_path,
                src_index=src_index,
                dst_data=data_out,
                dst_key=dst_key,
                env=env,
                height=height,
                width=width,
                use_left_close_low=new_idx in left_indexes,
            )
        write_masks(dst, new_keys)
        dst.attrs["dataset_build_metadata_json"] = json.dumps(metadata, indent=2, sort_keys=True)
        dst.attrs["shuffle_seed"] = int(seed)
        dst.attrs["left_close_low_pos"] = LEFT_CLOSE_LOW_POS
        dst.attrs["left_close_low_quat_wxyz"] = LEFT_CLOSE_LOW_QUAT_WXYZ


def main() -> None:
    args = parse_args()
    validate_source(args.ph_image, expected_action_dim=7)
    validate_source(args.rollout_image, expected_action_dim=7)

    ph_indices = selected_indices(args.ph_image, args.num_ph)
    rollout_indices = selected_indices(args.rollout_image, args.num_rollouts)

    build_dataset(
        args.out_root / "ph_agentview" / "image.hdf5",
        args.ph_image,
        None,
        ph_indices,
        [],
        mix_views=False,
        shuffle=False,
        seed=args.shuffle_seed,
        height=args.render_height,
        width=args.render_width,
        overwrite=args.overwrite,
    )
    build_dataset(
        args.out_root / "400_agentview" / "image.hdf5",
        args.ph_image,
        args.rollout_image,
        ph_indices,
        rollout_indices,
        mix_views=False,
        shuffle=False,
        seed=args.shuffle_seed,
        height=args.render_height,
        width=args.render_width,
        overwrite=args.overwrite,
    )
    build_dataset(
        args.out_root / "400_mix" / "image.hdf5",
        args.ph_image,
        args.rollout_image,
        ph_indices,
        rollout_indices,
        mix_views=True,
        shuffle=True,
        seed=args.shuffle_seed,
        height=args.render_height,
        width=args.render_width,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
