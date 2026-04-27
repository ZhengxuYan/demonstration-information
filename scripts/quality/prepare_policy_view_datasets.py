#!/usr/bin/env python3
"""Prepare policy-view HDF5 datasets for Square PH and expert200 experiments."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import types
import zipfile
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


LEFT_CLOSE_LOW_POS = np.asarray([0.42205740, -0.23999999, 1.15230719], dtype=np.float64)
LEFT_CLOSE_LOW_QUAT_WXYZ = np.asarray([0.81392215, 0.36066498, 0.18452251, 0.41641680], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["ph", "expert200"])
    parser.add_argument("--out-root", type=Path, default=Path("/iris/u/jasonyan/data/policy_view_experiments"))
    parser.add_argument(
        "--ph-image",
        type=Path,
        default=Path("/iris/u/jasonyan/data/diffusion_policy/robomimic/datasets/square/ph/image.hdf5"),
    )
    parser.add_argument(
        "--ph-image-abs",
        type=Path,
        default=Path("/iris/u/jasonyan/data/diffusion_policy/robomimic/datasets/square/ph/image_abs.hdf5"),
    )
    parser.add_argument("--expert200-zip", type=Path, default=None)
    parser.add_argument("--expert200-source", type=Path, default=None)
    parser.add_argument("--ph-dp-num-demos", type=int, default=50)
    parser.add_argument("--ph-dp-seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--render-height", type=int, default=84)
    parser.add_argument("--render-width", type=int, default=84)
    return parser.parse_args()


def sorted_demo_keys(data_group) -> list[str]:
    return sorted(data_group.keys(), key=lambda key: int(key.split("_")[-1]))


def copy_attrs(src, dst) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def copy_group(src, dst) -> None:
    copy_attrs(src, dst)
    for key, item in src.items():
        if isinstance(item, h5py.Group):
            child = dst.create_group(key)
            copy_group(item, child)
        else:
            src.copy(item, dst, name=key)


def write_masks(out_file: h5py.File, new_demo_keys: list[str]) -> None:
    mask = out_file.create_group("mask")
    encoded = np.asarray([key.encode("utf-8") for key in new_demo_keys])
    mask.create_dataset("train", data=encoded)
    mask.create_dataset("valid", data=np.asarray([], dtype=encoded.dtype))


def selected_demo_indices(num_demos: int, count: int, seed: int) -> list[int]:
    if count > num_demos:
        raise ValueError(f"Cannot sample {count} demos from only {num_demos}")
    rng = np.random.RandomState(seed)
    return sorted(rng.choice(np.arange(num_demos), size=count, replace=False).astype(int).tolist())


def validate_source(path: Path, expected_action_dim: int) -> None:
    with h5py.File(path, "r") as f:
        demos = sorted_demo_keys(f["data"])
        if not demos:
            raise ValueError(f"No demos found in {path}")
        action_dim = int(f["data"][demos[0]]["actions"].shape[-1])
        if action_dim != expected_action_dim:
            raise ValueError(f"{path} has action dim {action_dim}; expected {expected_action_dim}")


def install_lang_utils_stub() -> None:
    """Avoid loading CLIP / transformers when robomimic only needs env reset/render.

    The local robomimic fork imports lang_utils at EnvRobosuite import time, and
    lang_utils eagerly imports transformers plus CLIP weights. Dataset rendering
    passes lang=None, so language embeddings are unused here.
    """
    module_name = "robomimic.utils.lang_utils"
    if module_name in sys.modules:
        return
    stub = types.ModuleType(module_name)
    stub.LANG_EMB_OBS_KEY = "lang_emb"
    stub.get_lang_emb = lambda lang: None
    stub.get_lang_emb_shape = lambda: []
    sys.modules[module_name] = stub


def create_env(env_meta: dict, height: int, width: int):
    install_lang_utils_stub()

    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils

    env_meta = upgrade_controller_config(env_meta)
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": ["robot0_eef_pos"], "rgb": []}})
    return EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=["agentview"],
        camera_height=height,
        camera_width=width,
        reward_shaping=False,
        render=False,
        render_offscreen=True,
        use_image_obs=True,
    )


def upgrade_controller_config(env_meta: dict) -> dict:
    try:
        from robosuite.controllers.composite.composite_controller_factory import (
            refactor_composite_controller_config,
        )
    except Exception:
        return env_meta

    env_meta = json.loads(json.dumps(env_meta))
    env_kwargs = env_meta["env_kwargs"]
    controller_config = env_kwargs.get("controller_configs")
    robots = env_kwargs.get("robots", [])
    if controller_config is None or not robots:
        return env_meta
    if "body_parts" in controller_config:
        return env_meta

    robot_type = robots[0] if isinstance(robots, (list, tuple)) else robots
    try:
        env_kwargs["controller_configs"] = refactor_composite_controller_config(
            controller_config=controller_config,
            robot_type=robot_type,
            arms=["right"],
        )
    except Exception:
        return env_meta
    return env_meta


def set_left_close_low_pose(env) -> None:
    cam_id = env.env.sim.model.camera_name2id("agentview")
    env.env.sim.model.cam_pos[cam_id] = LEFT_CLOSE_LOW_POS
    env.env.sim.model.cam_quat[cam_id] = LEFT_CLOSE_LOW_QUAT_WXYZ
    env.env.sim.forward()


def render_left_close_low_images(env, states: np.ndarray, height: int, width: int) -> np.ndarray:
    frames = []
    set_left_close_low_pose(env)
    for state in states:
        # Avoid EnvRobosuite.reset_to here: it also materializes observations,
        # which can trigger an extra image render before our custom camera render.
        env.env.sim.set_state_from_flattened(state)
        env.env.sim.forward()
        frame = env.render(mode="rgb_array", height=height, width=width, camera_name="agentview")
        frames.append(np.asarray(frame, dtype=np.uint8))
    return np.stack(frames, axis=0)


def add_custom_view(demo_out, images: np.ndarray) -> None:
    for group_name in ("obs", "next_obs"):
        group = demo_out[group_name]
        if "left_close_low_image" in group:
            del group["left_close_low_image"]
    demo_out["obs"].create_dataset("left_close_low_image", data=images, compression="gzip", compression_opts=1)
    next_images = np.concatenate([images[1:], images[-1:]], axis=0)
    demo_out["next_obs"].create_dataset(
        "left_close_low_image",
        data=next_images,
        compression="gzip",
        compression_opts=1,
    )


def build_dataset(
    src_path: Path,
    dst_path: Path,
    demo_indices: list[int] | None,
    add_left_close_low: bool,
    render_height: int,
    render_width: int,
    overwrite: bool,
) -> None:
    if dst_path.exists():
        if not overwrite:
            print(f"exists, skipping: {dst_path}")
            return
        dst_path.unlink()
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    env = None
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        copy_attrs(src, dst)
        src_demo_keys = sorted_demo_keys(src["data"])
        if demo_indices is None:
            demo_indices = list(range(len(src_demo_keys)))
        data_out = dst.create_group("data")
        copy_attrs(src["data"], data_out)

        if add_left_close_low:
            env_meta = json.loads(src["data"].attrs["env_args"])
            env = create_env(env_meta, render_height, render_width)
            env.reset()

        new_demo_keys = []
        mapping = {}
        for new_idx, old_idx in enumerate(tqdm(demo_indices, desc=f"writing {dst_path.name}")):
            old_key = src_demo_keys[old_idx]
            new_key = f"demo_{new_idx}"
            mapping[new_key] = old_key
            new_demo_keys.append(new_key)
            demo_out = data_out.create_group(new_key)
            copy_group(src["data"][old_key], demo_out)
            if add_left_close_low:
                images = render_left_close_low_images(env, demo_out["states"][:], render_height, render_width)
                add_custom_view(demo_out, images)

        write_masks(dst, new_demo_keys)
        dst.attrs["source_path"] = str(src_path)
        dst.attrs["demo_key_mapping_json"] = json.dumps(mapping, sort_keys=True)
        dst.attrs["left_close_low_pos"] = LEFT_CLOSE_LOW_POS
        dst.attrs["left_close_low_quat_wxyz"] = LEFT_CLOSE_LOW_QUAT_WXYZ


def write_selection(path: Path, seed: int, indices: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"seed": seed, "selected_original_ep_idx": indices}, indent=2) + "\n")


def find_expert200_source(args: argparse.Namespace) -> Path:
    if args.expert200_source is not None:
        return args.expert200_source

    expert_root = args.out_root / "expert200"
    expert_root.mkdir(parents=True, exist_ok=True)
    zip_value = args.expert200_zip or os.environ.get("EXPERT200_ZIP")
    zip_path = Path(zip_value) if zip_value else None
    if zip_path is None or not zip_path.exists():
        raise FileNotFoundError("Provide --expert200-source or --expert200-zip, or set EXPERT200_ZIP")

    extract_dir = expert_root / "expert200_unzipped"
    if not extract_dir.exists():
        extract_dir.mkdir(parents=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

    candidates = sorted(extract_dir.rglob("image_abs.hdf5")) + sorted(extract_dir.rglob("*.hdf5"))
    for candidate in candidates:
        try:
            validate_source(candidate, expected_action_dim=7)
            return candidate
        except Exception as exc:
            print(f"skipping non-DP-raw-action candidate {candidate}: {exc}")
    raise FileNotFoundError(f"No robomimic-compatible absolute-action HDF5 found in {extract_dir}")


def prepare_ph(args: argparse.Namespace) -> None:
    out = args.out_root / "square_ph"
    validate_source(args.ph_image, expected_action_dim=7)
    # Diffusion Policy stores robomimic absolute actions as raw 7D actions in
    # HDF5 and expands them to 10D internally when abs_action=True.
    validate_source(args.ph_image_abs, expected_action_dim=7)
    with h5py.File(args.ph_image_abs, "r") as f:
        num_abs_demos = len(f["data"])
    selected = selected_demo_indices(num_abs_demos, args.ph_dp_num_demos, args.ph_dp_seed)
    write_selection(out / f"square_ph_dp_50_seed{args.ph_dp_seed}_selection.json", args.ph_dp_seed, selected)

    build_dataset(args.ph_image, out / "square_ph_agent_wrist_image.hdf5", None, False, args.render_height, args.render_width, args.overwrite)
    build_dataset(args.ph_image, out / "square_ph_left_close_low_wrist_image.hdf5", None, True, args.render_height, args.render_width, args.overwrite)
    build_dataset(args.ph_image_abs, out / "square_ph_agent_wrist_image_abs_50_seed42.hdf5", selected, False, args.render_height, args.render_width, args.overwrite)
    build_dataset(args.ph_image_abs, out / "square_ph_left_close_low_wrist_image_abs_50_seed42.hdf5", selected, True, args.render_height, args.render_width, args.overwrite)


def prepare_expert200(args: argparse.Namespace) -> None:
    out = args.out_root / "expert200"
    src = find_expert200_source(args)
    validate_source(src, expected_action_dim=7)
    build_dataset(src, out / "expert200_agent_wrist_image_abs.hdf5", None, False, args.render_height, args.render_width, args.overwrite)
    build_dataset(src, out / "expert200_left_close_low_wrist_image_abs.hdf5", None, True, args.render_height, args.render_width, args.overwrite)


def main() -> None:
    args = parse_args()
    if args.dataset == "ph":
        prepare_ph(args)
    else:
        prepare_expert200(args)


if __name__ == "__main__":
    main()
