#!/usr/bin/env python3
"""Render low-dimensional robomimic rollout HDF5 files to MP4 videos.

This is for rollout datasets that store simulator states but not image
observations, e.g. files with ``data/demo_*/states`` and ``data.attrs.env_args``.

Example:

python scripts/quality/render_low_dim_rollout_videos.py \
  --input-root /scr/tiangao/pomdp_vla/square_rollouts \
  --output-root /iris/u/jasonyan/data/pomdp_vla_square_rollouts/videos \
  --camera-name agentview \
  --fps 20
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import types
from pathlib import Path

import h5py
import imageio
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--camera-name", default="agentview")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--video-skip", type=int, default=1)
    parser.add_argument("--max-demos-per-file", type=int, default=0, help="0 means all demos.")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all frames.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def install_lang_utils_stub() -> None:
    """Avoid importing optional CLIP / transformers when only rendering env states."""
    module_name = "robomimic.utils.lang_utils"
    if module_name in sys.modules:
        return
    stub = types.ModuleType(module_name)
    stub.LANG_EMB_OBS_KEY = "lang_emb"
    stub.get_lang_emb = lambda lang: None
    stub.get_lang_emb_shape = lambda: []
    sys.modules[module_name] = stub


def initialize_obs_utils() -> None:
    import robomimic.utils.obs_utils as ObsUtils

    dummy_spec = dict(obs=dict(low_dim=["robot0_eef_pos"], rgb=[]))
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)


def upgrade_controller_config(env_meta: dict) -> dict:
    """Handle robosuite releases that expect composite controller configs."""
    try:
        from robosuite.controllers.composite.composite_controller_factory import (
            refactor_composite_controller_config,
        )
    except Exception:
        return env_meta

    env_meta = copy.deepcopy(env_meta)
    env_kwargs = env_meta["env_kwargs"]
    controller_config = env_kwargs.get("controller_configs")
    robots = env_kwargs.get("robots", [])
    if controller_config is None or not robots:
        return env_meta

    robot_type = robots[0] if isinstance(robots, (list, tuple)) else robots
    env_kwargs["controller_configs"] = refactor_composite_controller_config(
        controller_config=controller_config,
        robot_type=robot_type,
        arms=["right"],
    )
    return env_meta


def create_env(env_meta: dict, camera_name: str, height: int, width: int):
    install_lang_utils_stub()
    initialize_obs_utils()

    import robomimic.utils.env_utils as EnvUtils

    env_meta = upgrade_controller_config(env_meta)
    return EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=[camera_name],
        camera_height=height,
        camera_width=width,
        reward_shaping=False,
    )


def demo_sort_key(name: str) -> tuple[int, str]:
    if name.startswith("demo_"):
        try:
            return int(name.removeprefix("demo_")), name
        except ValueError:
            pass
    return 10**12, name


def render_demo(
    env,
    states: np.ndarray,
    out_path: Path,
    camera_name: str,
    height: int,
    width: int,
    fps: int,
    video_skip: int,
    max_frames: int,
    overwrite: bool,
) -> int:
    if out_path.exists() and not overwrite:
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(states) if max_frames <= 0 else min(len(states), max_frames)
    frame_indices = range(0, total, max(video_skip, 1))

    with imageio.get_writer(out_path, fps=fps) as writer:
        count = 0
        for i in frame_indices:
            env.env.sim.set_state_from_flattened(states[i])
            env.env.sim.forward()
            frame = env.render(mode="rgb_array", height=height, width=width, camera_name=camera_name)
            writer.append_data(np.asarray(frame, dtype=np.uint8))
            count += 1
    return count


def render_file(path: Path, args: argparse.Namespace) -> list[dict[str, str]]:
    rel = path.relative_to(args.input_root)
    file_out_dir = args.output_root / rel.with_suffix("")
    rows: list[dict[str, str]] = []

    with h5py.File(path, "r") as f:
        env_meta = json.loads(f["data"].attrs["env_args"])
        demos = sorted(f["data"].keys(), key=demo_sort_key)
        if args.max_demos_per_file > 0:
            demos = demos[: args.max_demos_per_file]

        env = create_env(env_meta, args.camera_name, args.height, args.width)
        try:
            for demo_key in demos:
                states = np.asarray(f["data"][demo_key]["states"])
                out_path = file_out_dir / f"{demo_key}_{args.camera_name}.mp4"
                print(f"render {path.name} {demo_key} -> {out_path}", flush=True)
                n_frames = render_demo(
                    env=env,
                    states=states,
                    out_path=out_path,
                    camera_name=args.camera_name,
                    height=args.height,
                    width=args.width,
                    fps=args.fps,
                    video_skip=args.video_skip,
                    max_frames=args.max_frames,
                    overwrite=args.overwrite,
                )
                rows.append(
                    {
                        "source_hdf5": str(path),
                        "source_file": path.name,
                        "demo_key": demo_key,
                        "camera_name": args.camera_name,
                        "video": str(out_path),
                        "rendered_frames": str(n_frames),
                    }
                )
        finally:
            env.env.close()
    return rows


def main() -> None:
    args = parse_args()
    files = sorted(list(args.input_root.rglob("*.hdf5")) + list(args.input_root.rglob("*.h5")))
    if not files:
        raise SystemExit(f"No .hdf5 or .h5 files found under {args.input_root}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, str]] = []
    for path in files:
        all_rows.extend(render_file(path, args))

    manifest = args.output_root / "video_manifest.csv"
    with manifest.open("w", newline="") as f:
        fieldnames = ["source_hdf5", "source_file", "demo_key", "camera_name", "video", "rendered_frames"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"wrote {manifest} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
