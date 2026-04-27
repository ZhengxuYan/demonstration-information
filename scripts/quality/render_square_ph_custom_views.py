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
import copy
import csv
import html
import json
from pathlib import Path

import h5py
import imageio
import numpy as np
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
import robosuite
import robosuite.utils.transform_utils as T
from robosuite.controllers.composite.composite_controller_factory import (
    refactor_composite_controller_config,
)


CORRECTION = np.array(
    [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--demo-idx", type=int, default=0)
    parser.add_argument("--annotations-csv", type=Path)
    parser.add_argument("--demos-per-label", type=int, default=5)
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
    work_target = base_pos + 0.58 * base_forward + np.array([0.0, 0.0, -0.08], dtype=np.float64)

    specs = [
        dict(name="agentview", pos=base_pos, target=work_target),
        dict(
            name="left_moderate_high",
            pos=base_pos - 0.12 * base_right + 0.03 * base_forward + np.array([0.0, 0.0, 0.05]),
            target=work_target + np.array([0.0, 0.0, -0.01]),
        ),
        dict(
            name="left_moderate_low",
            pos=base_pos - 0.14 * base_right + 0.05 * base_forward + np.array([0.0, 0.0, -0.04]),
            target=work_target + np.array([0.0, 0.0, -0.04]),
        ),
        dict(
            name="right_moderate_high",
            pos=base_pos + 0.12 * base_right + 0.03 * base_forward + np.array([0.0, 0.0, 0.05]),
            target=work_target + np.array([0.0, 0.0, -0.01]),
        ),
        dict(
            name="right_moderate_low",
            pos=base_pos + 0.14 * base_right + 0.05 * base_forward + np.array([0.0, 0.0, -0.04]),
            target=work_target + np.array([0.0, 0.0, -0.04]),
        ),
        dict(
            name="left_close_high",
            pos=base_pos - 0.22 * base_right + 0.08 * base_forward + np.array([0.0, 0.0, 0.00]),
            target=work_target + np.array([0.0, 0.0, -0.02]),
        ),
        dict(
            name="left_close_low",
            pos=base_pos - 0.24 * base_right + 0.11 * base_forward + np.array([0.0, 0.0, -0.12]),
            target=work_target + np.array([0.0, 0.0, -0.08]),
        ),
        dict(
            name="right_close_high",
            pos=base_pos + 0.22 * base_right + 0.08 * base_forward + np.array([0.0, 0.0, 0.00]),
            target=work_target + np.array([0.0, 0.0, -0.02]),
        ),
        dict(
            name="right_close_low",
            pos=base_pos + 0.24 * base_right + 0.11 * base_forward + np.array([0.0, 0.0, -0.12]),
            target=work_target + np.array([0.0, 0.0, -0.08]),
        ),
    ]
    for spec in specs:
        spec["rot"] = look_at_rotation(np.asarray(spec["pos"]), np.asarray(spec["target"]), world_up)
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


def select_demo_indices(annotations_csv: Path, demos_per_label: int) -> list[dict[str, object]]:
    rows = []
    with annotations_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["label"].strip()
            if label not in {"full", "partial"}:
                continue
            rows.append(
                {
                    "ep_idx": int(row["ep_idx"]),
                    "label": label,
                    "note": row.get("note", "").strip(),
                }
            )

    selected = []
    for label in ("full", "partial"):
        group = sorted([row for row in rows if row["label"] == label], key=lambda row: row["ep_idx"])
        if len(group) <= demos_per_label:
            picks = group
        else:
            idxs = np.linspace(0, len(group) - 1, demos_per_label, dtype=int)
            picks = [group[i] for i in idxs]
        selected.extend(picks)
    return selected


def rewrite_legacy_model_xml(model_xml: str) -> str:
    """
    Normalize dataset-era robosuite asset paths to the local robosuite checkout.

    Older Square datasets reference absolute paths on the data-generation machine and
    Panda visual meshes that were later reorganized. For viewpoint preview we only
    need geometrically reasonable robot meshes, so legacy *_vis.stl files are mapped
    to the corresponding collision STL meshes that still exist locally.
    """
    robosuite_root = Path(robosuite.__file__).resolve().parent
    assets_root = robosuite_root / "models" / "assets"

    xml = model_xml.replace(
        "/home/robot/installed_libraries/robosuite/robosuite/models/assets",
        str(assets_root),
    )
    xml = xml.replace(
        str(assets_root / "mounts"),
        str(assets_root / "bases"),
    )

    legacy_mesh_rewrites = {
        "robots/panda/meshes/link0_vis.stl": "robots/panda/meshes/link0.stl",
        "robots/panda/meshes/link1_vis.stl": "robots/panda/meshes/link1.stl",
        "robots/panda/meshes/link2_vis.stl": "robots/panda/meshes/link2.stl",
        "robots/panda/meshes/link3_vis.stl": "robots/panda/meshes/link3.stl",
        "robots/panda/meshes/link4_vis.stl": "robots/panda/meshes/link4.stl",
        "robots/panda/meshes/link5_vis.stl": "robots/panda/meshes/link5.stl",
        "robots/panda/meshes/link6_vis.stl": "robots/panda/meshes/link6.stl",
        "robots/panda/meshes/link7_vis.stl": "robots/panda/meshes/link7.stl",
        "grippers/meshes/panda_gripper/hand_vis.stl": "grippers/meshes/panda_gripper/hand.stl",
        "grippers/meshes/panda_gripper/finger_vis.stl": "grippers/meshes/panda_gripper/finger.stl",
    }
    for old_rel, new_rel in legacy_mesh_rewrites.items():
        xml = xml.replace(str(assets_root / old_rel), str(assets_root / new_rel))

    return xml


def upgrade_controller_config(env_meta: dict) -> dict:
    """
    Robomimic Square datasets were produced with robosuite's older part-controller
    config format. Newer robosuite releases expect a composite-controller config.
    """
    env_meta = copy.deepcopy(env_meta)
    env_kwargs = env_meta["env_kwargs"]
    controller_config = env_kwargs.get("controller_configs")
    robots = env_kwargs.get("robots", [])
    if controller_config is None or not robots:
        return env_meta

    if isinstance(robots, (list, tuple)):
        robot_type = robots[0]
    else:
        robot_type = robots

    env_kwargs["controller_configs"] = refactor_composite_controller_config(
        controller_config=controller_config,
        robot_type=robot_type,
        arms=["right"],
    )
    return env_meta


def create_env(env_meta: dict, camera_height: int, camera_width: int):
    initialize_obs_utils()
    env_meta = upgrade_controller_config(env_meta)
    return EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=["agentview"],
        camera_height=camera_height,
        camera_width=camera_width,
        reward_shaping=False,
    )


def render_views(
    env,
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

    env.reset()
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


def build_html(output_dir: Path, demos: list[dict[str, object]]) -> None:
    sections = []
    for demo in demos:
        cards = []
        for item in demo["videos"]:
            cards.append(
                f"""
            <article class="card">
              <h3>{html.escape(item['name'])}</h3>
              <video controls preload="metadata" src="{html.escape(item['video'])}"></video>
            </article>
            """
            )
        sections.append(
            f"""
        <section class="demo">
          <div class="demo-header">
            <h2>Demo {demo['demo_idx']}</h2>
            <div class="meta">observability: {html.escape(demo['label'])}</div>
          </div>
          <div class="grid">
            {''.join(cards)}
          </div>
        </section>
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
    .demo {{
      margin: 0 0 28px;
      padding: 16px;
      border: 1px solid #d9d0c0;
      border-radius: 20px;
      background: rgba(255, 250, 240, 0.55);
    }}
    .demo-header {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 16px;
      margin: 0 0 12px;
    }}
    .demo-header h2 {{
      margin: 0;
      font-size: 22px;
    }}
    .meta {{
      color: #6f6a63;
      font-size: 14px;
      text-transform: lowercase;
    }}
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
    h3 {{ margin: 0 0 10px; font-size: 18px; }}
    video {{
      width: 100%;
      display: block;
      aspect-ratio: 1 / 1;
      background: #050403;
    }}
  </style>
</head>
<body>
  <h1>Square PH Camera View Preview</h1>
  <p>Ten demos total: five labeled full observability and five labeled partial observability. Each demo shows agentview plus eight candidate third-person views: four moderate variants and four more aggressive variants chosen to increase occlusion.</p>
  <p>How these views were generated: I used the original robosuite <code>agentview</code> as the reference camera, extracted its position and orientation, then created new candidate views by manually perturbing the camera pose in a front-facing regime. The moderate views use smaller left/right, forward, and height offsets; the close views use larger lateral shifts, move slightly closer to the workspace, and lower the camera more to induce stronger occlusion between the gripper, peg, and hole. All candidate cameras were then re-aimed toward the same task workspace before replaying the same trajectories.</p>
  {''.join(sections)}
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_doc)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.annotations_csv is not None:
        selected = select_demo_indices(args.annotations_csv, args.demos_per_label)
    else:
        selected = [{"ep_idx": args.demo_idx, "label": "unspecified", "note": ""}]

    env_meta, initial_state, states = load_demo(args.dataset, selected[0]["ep_idx"])
    env = create_env(env_meta, args.camera_height, args.camera_width)
    env.reset()
    env.reset_to({"states": initial_state["states"]})
    base_pos, base_rot = current_agentview_pose(env, "agentview")
    specs = view_specs(base_pos, base_rot)

    demo_sections = []
    for row in selected:
        demo_idx = row["ep_idx"]
        _, _, states = load_demo(args.dataset, demo_idx)
        demo_dir = args.output_dir / f"demo_{demo_idx}"
        videos = render_views(
            env=env,
            states=states,
            specs=specs,
            output_dir=demo_dir,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.fps,
            max_frames=args.max_frames,
        )
        demo_sections.append(
            {
                "demo_idx": demo_idx,
                "label": row["label"],
                "videos": [{**video, "video": f"demo_{demo_idx}/{video['video']}"} for video in videos],
            }
        )

    build_html(args.output_dir, demo_sections)
    print(args.output_dir / "index.html")


if __name__ == "__main__":
    main()
