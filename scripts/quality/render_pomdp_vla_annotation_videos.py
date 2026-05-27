#!/usr/bin/env python3
"""Render POMDP-VLA Square rollout videos for human annotation.

Each output MP4 shows the same episode from two synchronized views:
left_close_low third-person on the left and wrist on the right.

Example:
python scripts/quality/render_pomdp_vla_annotation_videos.py \
  --input-hdf5 /iris/u/jasonyan/data/pomdp_vla_square_rollouts_raw/low_dim_bc_gmm_seed1_success200.hdf5 \
  --output-root /iris/u/jasonyan/data/pomdp_vla_seed1_annotation
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import imageio
import numpy as np

from prepare_policy_view_datasets import LEFT_CLOSE_LOW_POS, LEFT_CLOSE_LOW_QUAT_WXYZ
from render_low_dim_rollout_videos import create_env, demo_sort_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-hdf5", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--video-skip", type=int, default=1)
    parser.add_argument("--max-demos", type=int, default=200)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all frames.")
    parser.add_argument("--show-sites", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def hide_sites(env) -> np.ndarray:
    site_rgba = np.array(env.env.sim.model.site_rgba, copy=True)
    env.env.sim.model.site_rgba[:, 3] = 0.0
    return site_rgba


def restore_sites(env, site_rgba: np.ndarray) -> None:
    env.env.sim.model.site_rgba[:] = site_rgba


def render_left_close_low(env, height: int, width: int) -> np.ndarray:
    cam_id = env.env.sim.model.camera_name2id("agentview")
    old_pos = np.array(env.env.sim.model.cam_pos[cam_id], copy=True)
    old_quat = np.array(env.env.sim.model.cam_quat[cam_id], copy=True)
    try:
        env.env.sim.model.cam_pos[cam_id] = LEFT_CLOSE_LOW_POS
        env.env.sim.model.cam_quat[cam_id] = LEFT_CLOSE_LOW_QUAT_WXYZ
        env.env.sim.forward()
        return np.asarray(env.render(mode="rgb_array", height=height, width=width, camera_name="agentview"), dtype=np.uint8)
    finally:
        env.env.sim.model.cam_pos[cam_id] = old_pos
        env.env.sim.model.cam_quat[cam_id] = old_quat
        env.env.sim.forward()


def render_wrist(env, height: int, width: int) -> np.ndarray:
    return np.asarray(
        env.render(mode="rgb_array", height=height, width=width, camera_name="robot0_eye_in_hand"),
        dtype=np.uint8,
    )


def render_demo(
    env,
    states: np.ndarray,
    out_path: Path,
    height: int,
    width: int,
    fps: int,
    video_skip: int,
    max_frames: int,
    show_sites: bool,
    overwrite: bool,
) -> int:
    if out_path.exists() and not overwrite:
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(states) if max_frames <= 0 else min(len(states), max_frames)
    frame_indices = range(0, total, max(video_skip, 1))
    site_rgba = None if show_sites else hide_sites(env)

    try:
        count = 0
        with imageio.get_writer(out_path, fps=fps) as writer:
            for index in frame_indices:
                env.env.sim.set_state_from_flattened(states[index])
                env.env.sim.forward()
                left = render_left_close_low(env, height=height, width=width)
                wrist = render_wrist(env, height=height, width=width)
                frame = np.concatenate([left, wrist], axis=1)
                writer.append_data(frame)
                count += 1
        return count
    finally:
        if site_rgba is not None:
            restore_sites(env, site_rgba)


def demo_ep_idx(demo_key: str) -> int:
    if demo_key.startswith("demo_"):
        return int(demo_key.removeprefix("demo_"))
    raise ValueError(f"Cannot parse demo index from {demo_key}")


def main() -> None:
    args = parse_args()
    video_dir = args.output_root / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    with h5py.File(args.input_hdf5, "r") as f:
        env_meta = json.loads(f["data"].attrs["env_args"])
        demos = sorted(f["data"].keys(), key=demo_sort_key)
        if args.max_demos > 0:
            demos = demos[: args.max_demos]

        env = create_env(env_meta, "agentview", args.height, args.width)
        try:
            for demo_key in demos:
                demo = f["data"][demo_key]
                states = np.asarray(demo["states"])
                ep_idx = demo_ep_idx(demo_key)
                out_path = video_dir / f"demo_{ep_idx:04d}_left_close_low_wrist.mp4"
                print(f"render {demo_key} -> {out_path}", flush=True)
                rendered_frames = render_demo(
                    env=env,
                    states=states,
                    out_path=out_path,
                    height=args.height,
                    width=args.width,
                    fps=args.fps,
                    video_skip=args.video_skip,
                    max_frames=args.max_frames,
                    show_sites=args.show_sites,
                    overwrite=args.overwrite,
                )
                rows.append(
                    {
                        "source_hdf5": str(args.input_hdf5),
                        "demo_key": demo_key,
                        "ep_idx": ep_idx,
                        "video": str(out_path),
                        "num_states": len(states),
                        "rendered_frames": rendered_frames,
                    }
                )
        finally:
            env.env.close()

    manifest = args.output_root / "manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source_hdf5", "demo_key", "ep_idx", "video", "num_states", "rendered_frames"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {manifest} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
