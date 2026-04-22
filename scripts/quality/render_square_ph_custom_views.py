"""
Render one Square PH rollout from multiple front-facing third-person camera views.

This is intended for quickly previewing candidate camera poses before regenerating
the whole dataset with mixed third-person views.

Example:

source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate openx

python scripts/quality/render_square_ph_custom_views.py \
  --dataset /iris/u/jasonyan/data/diffusion_policy/robomimic/datasets/square/ph/image.hdf5 \
  --demo-idx 0 \
  --output-dir /iris/u/jasonyan/data/camera_view_previews/square_ph_demo0
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import h5py
import imageio
import numpy as np
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
import robosuite.utils.transform_utils as T


CORRECTION = np.array(
    [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--demo-idx", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means render the full demo.")
    return parser.parse_args()


def initialize_obs_utils() -> None:
    dummy_spec = dict(obs=dict(low_dim=["robot0_eef_pos"], rgb=[]))
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)


def normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        raise ValueError(f"Cannot normalize near-zero vector: {vec}")
    return vec / norm


def look_at_rotation(camera_pos: np.ndarray, target_pos: np.ndarray, world_up: np.ndarray) -> np.ndarray:
    """
    Return a camera-to-world rotation matrix in the corrected robomimic / OpenCV-style frame:
    x = right, y = down-ish, z = forward.
    """
    forward = normalize(target_pos - camera_pos)
    right = normalize(np.cross(forward, world_up))
    down = normalize(np.cross(forward, right))
    return np.stack([right, down, forward], axis=1)


def corrected_rot_to_mujoco_quat(corrected_rot: np.ndarray) -> np.ndarray:
    raw_rot = corrected_rot @ CORRECTION
    quat_xyzw = T.mat2quat(raw_rot)
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
    return quat_wxyz.astype(np.float64)


def set_camera_pose(env, camera_name: str, pos: np.ndarray, corrected_rot: np.ndarray) -> None:
    cam_id = env.env.sim.model.camera_name2id(camera_name)
    env.env.sim.model.cam_pos[cam_id] = np.asarray(pos, dtype=np.float64)
    env.env.sim.model.cam_quat[cam_id] = corrected_rot_to_mujoco_quat(corrected_rot)
    env.env.sim.forward()


def current_agentview_pose(env, camera_name: str = "agentview") -> tuple[np.ndarray, np.ndarray]:
    cam_id = env.env.sim.model.camera_name2id(camera_name)
    base_pos = np.asarray(env.env.sim.data.cam_xpos[cam_id], dtype=np.float64)
    raw_rot = np.asarray(env.env.sim.data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)
    corrected_rot = raw_rot @ CORRECTION
    return base_pos, corrected_rot


def view_specs(base_pos: np.ndarray, base_rot: np.ndarray) -> list[dict[str, object]]:
    base_right = normalize(base_rot[:, 0])
    base_forward = normalize(base_rot[:, 2])
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    target = base_pos + 0.70 * base_forward

    specs = [
        dict(name="agentview", pos=base_pos, target=target),
        dict(name="front_left_high", pos=base_pos - 0.10 * base_right + np.array([0.0, 0.0, 0.08]), target=target),
        dict(name="front_left_low", pos=base_pos - 0.10 * base_right + np.array([0.0, 0.0, -0.06]), target=target),
        dict(name="front_right_high", pos=base_pos + 0.10 * base_right + np.array([0.0, 0.0, 0.08]), target=target),
        dict(name="front_right_low", pos=base_pos + 0.10 * base_right + np.array([0.0, 0.0, -0.06]), target=target),
    ]
    for spec in specs:
        spec["rot"] = look_at_rotation(np.asarray(spec["pos"]), target, world_up)
    return specs


def load_demo(path: Path, demo_idx: int) -> tuple[dict, np.ndarray]:
    demo_key = f"demo_{demo_idx}"
    with h5py.File(path, "r") as f:
        env_meta = json.loads(f["data"].attrs["env_args"])
        states = np.asarray(f[f"data/{demo_key}/states"])
        initial_state = {"states": states[0], "model": f[f"data/{demo_key}"].attrs["model_file"]}
        ep_meta = f[f"data/{demo_key}"].attrs.get("ep_meta", None)
        if ep_meta is not None:
            initial_state["ep_meta"] = ep_meta
    return env_meta, initial_state, states


def create_env(env_meta: dict, camera_height: int, camera_width: int):
    initialize_obs_utils()
    return EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=["agentview"],
        camera_height=camera_height,
        camera_width=camera_width,
        reward_shaping=False,
    )


def render_views(
    env,
    initial_state: dict,
    states: np.ndarray,
    specs: list[dict[str, object]],
    output_dir: Path,
    width: int,
    height: int,
    fps: int,
    max_frames: int,
) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    writers = {}
    outputs = []
    for spec in specs:
        out_path = output_dir / f"{spec['name']}.mp4"
        writers[spec["name"]] = imageio.get_writer(out_path, fps=fps)
        outputs.append({"name": spec["name"], "video": out_path.name})

    env.reset_to(initial_state)
    total = states.shape[0] if max_frames <= 0 else min(states.shape[0], max_frames)
    try:
        for i in range(total):
            env.reset_to({"states": states[i]})
            for spec in specs:
                set_camera_pose(env, "agentview", spec["pos"], spec["rot"])
                frame = env.render(mode="rgb_array", height=height, width=width, camera_name="agentview")
                writers[spec["name"]].append_data(frame)
    finally:
        for writer in writers.values():
            writer.close()
    return outputs


def build_html(output_dir: Path, demo_idx: int, videos: list[dict[str, object]]) -> None:
    cards = []
    for item in videos:
        cards.append(
            f"""
        <article class="card">
          <h2>{html.escape(item['name'])}</h2>
          <video controls preload="metadata" src="{html.escape(item['video'])}"></video>
        </article>
        """
        )

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Square PH Camera View Preview</title>
  <style>
    body {{
      margin: 0;
      padding: 24px;
      background: #efe9dd;
      color: #1f1b17;
      font-family: "Iowan Old Style", "Palatino Linotype", serif;
    }}
    h1 {{ margin: 0 0 8px; }}
    p {{ margin: 0 0 18px; color: #6f6a63; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
    }}
    .card {{
      background: #fffaf0;
      border: 1px solid #d9d0c0;
      border-radius: 16px;
      box-shadow: 0 12px 30px rgba(34, 26, 18, 0.08);
      padding: 12px;
    }}
    h2 {{ margin: 0 0 10px; font-size: 18px; }}
    video {{
      width: 100%;
      display: block;
      aspect-ratio: 1 / 1;
      background: #050403;
    }}
  </style>
</head>
<body>
  <h1>Square PH Demo {demo_idx} Camera View Preview</h1>
  <p>Original agentview plus four custom front-facing third-person views.</p>
  <div class="grid">
    {''.join(cards)}
  </div>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_doc)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    env_meta, initial_state, states = load_demo(args.dataset, args.demo_idx)
    env = create_env(env_meta, args.camera_height, args.camera_width)
    env.reset_to(initial_state)
    base_pos, base_rot = current_agentview_pose(env, "agentview")
    specs = view_specs(base_pos, base_rot)
    videos = render_views(
        env=env,
        initial_state=initial_state,
        states=states,
        specs=specs,
        output_dir=args.output_dir,
        width=args.camera_width,
        height=args.camera_height,
        fps=args.fps,
        max_frames=args.max_frames,
    )
    build_html(args.output_dir, args.demo_idx, videos)
    print(args.output_dir / "index.html")


if __name__ == "__main__":
    main()
