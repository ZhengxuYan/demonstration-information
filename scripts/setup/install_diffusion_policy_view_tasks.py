#!/usr/bin/env python3
"""Install Diffusion Policy task configs and custom-camera wrapper patch."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


LEFT_CLOSE_LOW_PATCH = """    def _render_left_close_low_image(self):
        sim = self.env.env.sim
        cam_id = sim.model.camera_name2id('agentview')
        old_pos = sim.model.cam_pos[cam_id].copy()
        old_quat = sim.model.cam_quat[cam_id].copy()
        try:
            sim.model.cam_pos[cam_id] = np.array([0.42205740, -0.23999999, 1.15230719], dtype=np.float64)
            sim.model.cam_quat[cam_id] = np.array([0.81392215, 0.36066498, 0.18452251, 0.41641680], dtype=np.float64)
            sim.forward()
            shape = self.shape_meta['obs']['left_close_low_image']['shape']
            _, height, width = shape
            frame = self.env.render(mode='rgb_array', height=height, width=width, camera_name='agentview')
        finally:
            sim.model.cam_pos[cam_id] = old_pos
            sim.model.cam_quat[cam_id] = old_quat
            sim.forward()
        return np.moveaxis(frame.astype(np.float32) / 255.0, -1, 0)

"""

GET_OBSERVATION_PATCH = """    def get_observation(self, raw_obs=None):
        if raw_obs is None:
            raw_obs = self.env.get_observation()

        obs_keys = self.observation_space.spaces.keys()
        needs_left_close_low = (
            self.render_obs_key == 'left_close_low_image'
            or 'left_close_low_image' in obs_keys
        )
        if needs_left_close_low and 'left_close_low_image' not in raw_obs:
            raw_obs = dict(raw_obs)
            raw_obs['left_close_low_image'] = self._render_left_close_low_image()

        self.render_cache = raw_obs[self.render_obs_key]

        obs = dict()
        for key in obs_keys:
            obs[key] = raw_obs[key]
        return obs

"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dp-repo", type=Path, default=Path("/iris/u/jasonyan/repos/diffusion_policy"))
    parser.add_argument("--data-root", type=Path, default=Path("/iris/u/jasonyan/data/policy_view_experiments"))
    return parser.parse_args()


def task_yaml(name: str, dataset_path: Path, rgb_key: str, render_obs_key: str) -> str:
    return f"""name: {name}

shape_meta: &shape_meta
  obs:
    {rgb_key}:
      shape: [3, 84, 84]
      type: rgb
    robot0_eye_in_hand_image:
      shape: [3, 84, 84]
      type: rgb
    robot0_eef_pos:
      shape: [3]
    robot0_eef_quat:
      shape: [4]
    robot0_gripper_qpos:
      shape: [2]
  action:
    shape: [10]

dataset_path: &dataset_path {dataset_path}
abs_action: &abs_action True

env_runner:
  _target_: diffusion_policy.env_runner.robomimic_image_runner.RobomimicImageRunner
  dataset_path: *dataset_path
  shape_meta: *shape_meta
  n_train: 6
  n_train_vis: 2
  train_start_idx: 0
  n_test: 50
  n_test_vis: 4
  test_start_seed: 100000
  max_steps: 400
  n_obs_steps: ${{n_obs_steps}}
  n_action_steps: ${{n_action_steps}}
  render_obs_key: '{render_obs_key}'
  fps: 10
  crf: 22
  past_action: ${{past_action_visible}}
  abs_action: *abs_action
  tqdm_interval_sec: 1.0
  n_envs: 16

dataset:
  _target_: diffusion_policy.dataset.robomimic_replay_image_dataset.RobomimicReplayImageDataset
  shape_meta: *shape_meta
  dataset_path: *dataset_path
  horizon: ${{horizon}}
  pad_before: ${{eval:'${{n_obs_steps}}-1+${{n_latency_steps}}'}}
  pad_after: ${{eval:'${{n_action_steps}}-1'}}
  n_obs_steps: ${{dataset_obs_steps}}
  abs_action: *abs_action
  rotation_rep: 'rotation_6d'
  use_legacy_normalizer: False
  use_cache: True
  seed: 42
  val_ratio: 0.02
"""


def write_tasks(dp_repo: Path, data_root: Path) -> None:
    task_dir = dp_repo / "diffusion_policy" / "config" / "task"
    task_dir.mkdir(parents=True, exist_ok=True)
    specs = {
        "square_ph_dp_agent_wrist_abs_50_seed42": (
            data_root / "square_ph" / "square_ph_agent_wrist_image_abs_50_seed42.hdf5",
            "agentview_image",
            "agentview_image",
        ),
        "square_ph_dp_left_close_low_wrist_abs_50_seed42": (
            data_root / "square_ph" / "square_ph_left_close_low_wrist_image_abs_50_seed42.hdf5",
            "left_close_low_image",
            "left_close_low_image",
        ),
        "expert200_dp_agent_wrist_abs_212": (
            data_root / "expert200" / "expert200_agent_wrist_image_abs.hdf5",
            "agentview_image",
            "agentview_image",
        ),
        "expert200_dp_left_close_low_wrist_abs_212": (
            data_root / "expert200" / "expert200_left_close_low_wrist_image_abs.hdf5",
            "left_close_low_image",
            "left_close_low_image",
        ),
    }
    for name, (dataset_path, rgb_key, render_obs_key) in specs.items():
        path = task_dir / f"{name}.yaml"
        path.write_text(task_yaml(name, dataset_path, rgb_key, render_obs_key))
        print(f"wrote {path}")


def patch_wrapper(dp_repo: Path) -> None:
    path = dp_repo / "diffusion_policy" / "env" / "robomimic" / "robomimic_image_wrapper.py"
    text = path.read_text()
    text = re.sub(
        r"    def _render_left_close_low_image\(self\):\n.*?\n\n    def get_observation\(self, raw_obs=None\):",
        LEFT_CLOSE_LOW_PATCH + "    def get_observation(self, raw_obs=None):",
        text,
        flags=re.S,
    )
    if "_render_left_close_low_image" not in text:
        marker = "    def get_observation(self, raw_obs=None):"
        if marker not in text:
            raise RuntimeError(f"Could not find get_observation marker in {path}")
        text = text.replace(marker, LEFT_CLOSE_LOW_PATCH + marker, 1)
    text, count = re.subn(
        r"    def get_observation\(self, raw_obs=None\):\n.*?\n    def seed\(self, seed=None\):",
        GET_OBSERVATION_PATCH + "    def seed(self, seed=None):",
        text,
        count=1,
        flags=re.S,
    )
    if count != 1:
        raise RuntimeError(f"Could not replace get_observation in {path}")
    path.write_text(text)
    print(f"patched {path}")


def main() -> None:
    args = parse_args()
    write_tasks(args.dp_repo, args.data_root)
    patch_wrapper(args.dp_repo)


if __name__ == "__main__":
    main()
