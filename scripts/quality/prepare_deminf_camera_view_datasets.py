#!/usr/bin/env python3
"""Prepare the camera-view HDF5 datasets requested for DemInf.

Outputs:
  ph_agentview/image.hdf5
    200 PH episodes with agentview_image.
  400_agentview/image.hdf5
    200 PH episodes + 200 rollout episodes, all with agentview_image.
  400_left_close_low/image.hdf5
    200 PH episodes + 200 rollout episodes, all rendered from left_close_low
    and stored under agentview_image.
  400_mix/image.hdf5
    Same 400 episodes, shuffled, with half rendered/stored as agentview_image
    from agentview and half rendered/stored as agentview_image from left_close_low.

The RLDS builder reads obs/agentview_image, so mixed-view episodes intentionally
store the selected third-person view under agentview_image.
"""

from __future__ import annotations

import argparse
import csv
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
    parser.add_argument("--rollout-image", type=Path, action="append", required=True)
    parser.add_argument(
        "--rollout-annotations",
        type=Path,
        help="Optional CSV from serve_yes_no_video_annotation_app.py. Uses rows with --positive-label.",
    )
    parser.add_argument("--positive-label", default="yes")
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


def all_indices(path: Path) -> list[int]:
    with h5py.File(path, "r") as f:
        return list(range(len(f["data"])))


def rollout_selections(paths: list[Path], annotations: Path | None, positive_label: str, count: int) -> list[tuple[Path, int]]:
    if annotations is None:
        selected = []
        for path in paths:
            for idx in all_indices(path):
                selected.append((path, idx))
                if len(selected) >= count:
                    return selected
        raise ValueError(f"Only found {len(selected)} rollout demos across {paths}; requested {count}")

    by_stem = {path.stem: path for path in paths}
    selected: list[tuple[Path, int]] = []
    with annotations.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("label", "").strip().lower() != positive_label:
                continue
            source = row.get("source", "").strip()
            demo_key = row.get("demo_key", "").strip()
            if source not in by_stem:
                continue
            if not demo_key.startswith("demo_"):
                continue
            selected.append((by_stem[source], int(demo_key.removeprefix("demo_"))))
            if len(selected) >= count:
                break
    if len(selected) < count:
        raise ValueError(
            f"Only found {len(selected)} rows with label={positive_label!r} in {annotations}; requested {count}"
        )
    return selected


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
    view_mode: str,
) -> dict[str, object]:
    with h5py.File(src_path, "r") as src:
        src_keys = sorted_demo_keys(src["data"])
        src_key = src_keys[src_index]
        demo_out = dst_data.create_group(dst_key)
        copy_group(src["data"][src_key], demo_out)
        states = demo_out["states"][:]
        model_xml = demo_out.attrs.get("model_file")

    render_key = "left_close_low_image" if view_mode == "left_close_low_named" else "agentview_image"
    ensure_required_observations(
        demo_out,
        env,
        states,
        render_key,
        height,
        width,
        model_xml=model_xml,
    )
    if view_mode in {"left_close_low", "left_close_low_named"}:
        frames = render_left_close_low_as_agentview(env, states, height, width, model_xml=model_xml)
        upsert_image_pair(demo_out, "agentview_image", frames)

    return {
        "source_path": str(src_path),
        "source_demo_index": int(src_index),
        "source_demo_key": src_key,
        "view": view_mode,
    }


def build_dataset(
    dst_path: Path,
    ph_path: Path,
    rollout_items: list[tuple[Path, int]],
    ph_indices: list[int],
    view_mode: str,
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
    items.extend(("rollout", path, idx) for path, idx in rollout_items)
    if shuffle:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(items)).astype(int).tolist()
        items = [items[i] for i in order]

    left_indexes = set()
    if view_mode == "mix":
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
            episode_view = view_mode
            if view_mode == "mix":
                episode_view = "left_close_low" if new_idx in left_indexes else "agentview"
            metadata[dst_key] = copy_episode(
                src_path=src_path,
                src_index=src_index,
                dst_data=data_out,
                dst_key=dst_key,
                env=env,
                height=height,
                width=width,
                view_mode=episode_view,
            )
        write_masks(dst, new_keys)
        dst.attrs["dataset_build_metadata_json"] = json.dumps(metadata, indent=2, sort_keys=True)
        dst.attrs["shuffle_seed"] = int(seed)
        dst.attrs["left_close_low_pos"] = LEFT_CLOSE_LOW_POS
        dst.attrs["left_close_low_quat_wxyz"] = LEFT_CLOSE_LOW_QUAT_WXYZ


def main() -> None:
    args = parse_args()
    validate_source(args.ph_image, expected_action_dim=7)
    for rollout_image in args.rollout_image:
        validate_source(rollout_image, expected_action_dim=7)

    ph_indices = selected_indices(args.ph_image, args.num_ph)
    selected_rollouts = rollout_selections(
        paths=args.rollout_image,
        annotations=args.rollout_annotations,
        positive_label=args.positive_label,
        count=args.num_rollouts,
    )

    build_dataset(
        args.out_root / "ph_agentview" / "image.hdf5",
        args.ph_image,
        [],
        ph_indices,
        view_mode="agentview",
        shuffle=False,
        seed=args.shuffle_seed,
        height=args.render_height,
        width=args.render_width,
        overwrite=args.overwrite,
    )
    build_dataset(
        args.out_root / "400_agentview" / "image.hdf5",
        args.ph_image,
        selected_rollouts,
        ph_indices,
        view_mode="agentview",
        shuffle=False,
        seed=args.shuffle_seed,
        height=args.render_height,
        width=args.render_width,
        overwrite=args.overwrite,
    )
    build_dataset(
        args.out_root / "400_left_close_low" / "image.hdf5",
        args.ph_image,
        selected_rollouts,
        ph_indices,
        view_mode="left_close_low",
        shuffle=False,
        seed=args.shuffle_seed,
        height=args.render_height,
        width=args.render_width,
        overwrite=args.overwrite,
    )
    build_dataset(
        args.out_root / "400_mix" / "image.hdf5",
        args.ph_image,
        selected_rollouts,
        ph_indices,
        view_mode="mix",
        shuffle=True,
        seed=args.shuffle_seed,
        height=args.render_height,
        width=args.render_width,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
