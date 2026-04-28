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
    parser.add_argument("--expert200-num-demos", type=int, default=None)
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


def create_env(env_meta: dict, height: int, width: int, camera_names: list[str] | None = None):
    install_lang_utils_stub()

    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils

    env_meta = upgrade_controller_config(env_meta)
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": ["robot0_eef_pos"], "rgb": []}})
    return EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=camera_names or ["agentview"],
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


def upsert_image_pair(demo_out, key: str, images: np.ndarray) -> None:
    obs = demo_out.require_group("obs")
    next_obs = demo_out.require_group("next_obs")
    for group in (obs, next_obs):
        if key in group:
            del group[key]
    obs.create_dataset(key, data=images, compression="gzip", compression_opts=1)
    next_images = np.concatenate([images[1:], images[-1:]], axis=0)
    next_obs.create_dataset(key, data=next_images, compression="gzip", compression_opts=1)


def upsert_obs_pair(demo_out, key: str, values: np.ndarray) -> None:
    obs = demo_out.require_group("obs")
    next_obs = demo_out.require_group("next_obs")
    for group in (obs, next_obs):
        if key in group:
            del group[key]
    obs.create_dataset(key, data=values)
    next_values = np.concatenate([values[1:], values[-1:]], axis=0)
    next_obs.create_dataset(key, data=next_values)


def select_robot_value(value, arm: str):
    if isinstance(value, dict):
        return value[arm]
    if isinstance(value, (list, tuple)):
        return value[0]
    return value


def as_index_list(value) -> list[int]:
    return np.asarray(value).reshape(-1).astype(np.int64).tolist()


def render_required_observations(
    env,
    states: np.ndarray,
    image_key: str,
    height: int,
    width: int,
    need_image_key: bool,
    need_wrist: bool,
    need_lowdim: bool,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    import robosuite.utils.transform_utils as T

    image_values: dict[str, list[np.ndarray]] = {}
    lowdim_values: dict[str, list[np.ndarray]] = {}
    if need_image_key:
        image_values[image_key] = []
    if need_wrist:
        image_values["robot0_eye_in_hand_image"] = []
    if need_lowdim:
        lowdim_values = {
            "robot0_eef_pos": [],
            "robot0_eef_quat": [],
            "robot0_gripper_qpos": [],
        }

    cam_id = env.env.sim.model.camera_name2id("agentview")
    original_agent_pos = env.env.sim.model.cam_pos[cam_id].copy()
    original_agent_quat = env.env.sim.model.cam_quat[cam_id].copy()
    robot = env.env.robots[0]
    arm = robot.arms[0] if getattr(robot, "arms", None) else "right"
    eef_site_id = select_robot_value(robot.eef_site_id, arm)
    eef_body_name = select_robot_value(robot.robot_model.eef_name, arm)
    gripper_joint_pos_indexes = as_index_list(select_robot_value(robot._ref_gripper_joint_pos_indexes, arm))

    for state in states:
        # Avoid EnvRobosuite.reset_to: it materializes observations and can
        # trigger extra image rendering before our selected camera renders.
        env.env.sim.set_state_from_flattened(state)
        env.env.sim.forward()

        if need_lowdim:
            eef_pos = np.asarray(env.env.sim.data.site_xpos[eef_site_id], dtype=np.float32)
            eef_quat = T.convert_quat(env.env.sim.data.get_body_xquat(eef_body_name), to="xyzw")
            eef_quat = np.asarray(eef_quat, dtype=np.float32)
            gripper_qpos = np.asarray(
                [env.env.sim.data.qpos[index] for index in gripper_joint_pos_indexes],
                dtype=np.float32,
            )
            lowdim_values["robot0_eef_pos"].append(eef_pos)
            lowdim_values["robot0_eef_quat"].append(eef_quat)
            lowdim_values["robot0_gripper_qpos"].append(gripper_qpos)

        if need_image_key and image_key == "agentview_image":
            frame = env.render(mode="rgb_array", height=height, width=width, camera_name="agentview")
            image_values[image_key].append(np.asarray(frame, dtype=np.uint8))
        elif need_image_key and image_key == "left_close_low_image":
            env.env.sim.model.cam_pos[cam_id] = LEFT_CLOSE_LOW_POS
            env.env.sim.model.cam_quat[cam_id] = LEFT_CLOSE_LOW_QUAT_WXYZ
            env.env.sim.forward()
            frame = env.render(mode="rgb_array", height=height, width=width, camera_name="agentview")
            image_values[image_key].append(np.asarray(frame, dtype=np.uint8))
            env.env.sim.model.cam_pos[cam_id] = original_agent_pos
            env.env.sim.model.cam_quat[cam_id] = original_agent_quat
            env.env.sim.forward()
        elif need_image_key:
            raise ValueError(f"Unsupported image key {image_key}")

        if need_wrist:
            frame = env.render(mode="rgb_array", height=height, width=width, camera_name="robot0_eye_in_hand")
            image_values["robot0_eye_in_hand_image"].append(np.asarray(frame, dtype=np.uint8))

    stacked_images = {key: np.stack(values, axis=0) for key, values in image_values.items()}
    stacked_lowdim = {key: np.stack(values, axis=0) for key, values in lowdim_values.items()}
    return stacked_images, stacked_lowdim


def ensure_required_observations(demo_out, env, states: np.ndarray, image_key: str, height: int, width: int) -> None:
    obs = demo_out.get("obs")
    need_image_key = obs is None or image_key not in obs
    need_wrist = obs is None or "robot0_eye_in_hand_image" not in obs
    required = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
    need_lowdim = obs is None or any(key not in obs for key in required)
    if not need_image_key and not need_wrist and not need_lowdim:
        return

    images, lowdim = render_required_observations(
        env=env,
        states=states,
        image_key=image_key,
        height=height,
        width=width,
        need_image_key=need_image_key,
        need_wrist=need_wrist,
        need_lowdim=need_lowdim,
    )
    for key, values in images.items():
        upsert_image_pair(demo_out, key, values)
    for key, values in lowdim.items():
        upsert_obs_pair(demo_out, key, values)


def load_env_meta(src, src_path: Path, fallback_path: Path | None) -> dict:
    env_args = src["data"].attrs.get("env_args")
    if env_args is None:
        if fallback_path is None:
            raise KeyError(f"{src_path} is missing data.attrs['env_args']")
        with h5py.File(fallback_path, "r") as fallback:
            env_args = fallback["data"].attrs["env_args"]
    return json.loads(env_args)


def source_needs_generated_obs(src, demo_key: str, image_key: str) -> bool:
    demo = src["data"][demo_key]
    if "obs" not in demo:
        return True
    obs = demo["obs"]
    required = (image_key, "robot0_eye_in_hand_image", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
    return any(key not in obs for key in required)


def build_dataset(
    src_path: Path,
    dst_path: Path,
    demo_indices: list[int] | None,
    add_left_close_low: bool,
    render_height: int,
    render_width: int,
    overwrite: bool,
    env_meta_fallback_path: Path | None = None,
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

        image_key = "left_close_low_image" if add_left_close_low else "agentview_image"
        needs_generated_obs = source_needs_generated_obs(src, src_demo_keys[demo_indices[0]], image_key)
        if add_left_close_low or needs_generated_obs:
            env_meta = load_env_meta(src, src_path, env_meta_fallback_path)
            env = create_env(env_meta, render_height, render_width, ["agentview", "robot0_eye_in_hand"])
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
            if add_left_close_low or needs_generated_obs:
                states = demo_out["states"][:]
                ensure_required_observations(demo_out, env, states, image_key, render_height, render_width)

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
    with h5py.File(src, "r") as f:
        num_demos = len(f["data"])
    selected_count = args.expert200_num_demos or num_demos
    selected = list(range(min(selected_count, num_demos)))
    if num_demos != selected_count:
        print(f"expert200 source has {num_demos} demos; using first {len(selected)} demos")
    build_dataset(
        src,
        out / "expert200_agent_wrist_image_abs.hdf5",
        selected,
        False,
        args.render_height,
        args.render_width,
        args.overwrite,
        env_meta_fallback_path=args.ph_image_abs,
    )
    build_dataset(
        src,
        out / "expert200_left_close_low_wrist_image_abs.hdf5",
        selected,
        True,
        args.render_height,
        args.render_width,
        args.overwrite,
        env_meta_fallback_path=args.ph_image_abs,
    )


def main() -> None:
    args = parse_args()
    if args.dataset == "ph":
        prepare_ph(args)
    else:
        prepare_expert200(args)


if __name__ == "__main__":
    main()
