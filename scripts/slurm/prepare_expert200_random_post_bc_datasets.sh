#!/bin/bash
#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --job-name=expert200_bc_data
#SBATCH --output=/iris/u/jasonyan/slurm/%j_expert200_bc_data.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_expert200_bc_data.err

set -euo pipefail

source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate openx

SRC_ROOT="${SRC_ROOT:-/iris/u/jasonyan/data/policy_view_experiments/expert200}"
DST_ROOT="${DST_ROOT:-/iris/u/jasonyan/data/policy_view_experiments/expert200_random_post_bc}"
VALID_RATIO="${VALID_RATIO:-0.1}"
SPLIT_SEED="${SPLIT_SEED:-0}"

mkdir -p /iris/u/jasonyan/slurm "${DST_ROOT}"

cp -v "${SRC_ROOT}/expert200_agent_wrist_image_abs.hdf5" \
  "${DST_ROOT}/expert200_random_post_agent_wrist_image_abs.hdf5"
cp -v "${SRC_ROOT}/expert200_left_close_low_wrist_image_abs.hdf5" \
  "${DST_ROOT}/expert200_random_post_left_close_low_wrist_image_abs.hdf5"

python - <<PY
import h5py
import numpy as np
from pathlib import Path

paths = [
    Path("${DST_ROOT}/expert200_random_post_agent_wrist_image_abs.hdf5"),
    Path("${DST_ROOT}/expert200_random_post_left_close_low_wrist_image_abs.hdf5"),
]
valid_ratio = float("${VALID_RATIO}")
split_seed = int("${SPLIT_SEED}")

for path in paths:
    with h5py.File(path, "r+") as f:
        demos = sorted(f["data"].keys(), key=lambda name: int(name.split("_")[-1]))
        for demo in demos:
            demo_group = f["data"][demo]
            if "num_samples" not in demo_group.attrs:
                demo_group.attrs["num_samples"] = int(demo_group["actions"].shape[0])
        rng = np.random.default_rng(split_seed)
        num_valid = max(1, int(round(valid_ratio * len(demos)))) if len(demos) > 1 else 0
        valid_idx = set(rng.choice(np.arange(len(demos)), size=num_valid, replace=False).astype(int).tolist())
        train = [demo for idx, demo in enumerate(demos) if idx not in valid_idx]
        valid = [demo for idx, demo in enumerate(demos) if idx in valid_idx]

        if "mask" in f:
            del f["mask"]
        mask = f.create_group("mask")
        mask.create_dataset("train", data=np.asarray([x.encode("utf-8") for x in train], dtype="S"))
        mask.create_dataset("valid", data=np.asarray([x.encode("utf-8") for x in valid], dtype="S"))
        f.attrs["mask_valid_ratio"] = valid_ratio
        f.attrs["mask_split_seed"] = split_seed

        print(path)
        print("train", len(train), "valid", len(valid), "overlap", len(set(train) & set(valid)))
        if set(train) & set(valid):
            raise SystemExit("train/valid overlap")
        if not train or not valid:
            raise SystemExit("empty train or valid split")
PY

cd /iris/u/jasonyan/repos/demonstration-information

python scripts/quality/verify_policy_view_dataset.py \
  "${DST_ROOT}/expert200_random_post_agent_wrist_image_abs.hdf5" \
  --expected-demos 212 --expected-action-dim 7 \
  --required-obs-key agentview_image \
  --required-obs-key robot0_eye_in_hand_image \
  --required-obs-key robot0_eef_pos \
  --required-obs-key robot0_eef_quat \
  --required-obs-key robot0_gripper_qpos \
  --expected-obs-shape agentview_image=84,84,3 \
  --expected-obs-shape robot0_eye_in_hand_image=84,84,3 \
  --expected-obs-shape robot0_eef_pos=3 \
  --expected-obs-shape robot0_eef_quat=4 \
  --expected-obs-shape robot0_gripper_qpos=2

python scripts/quality/verify_policy_view_dataset.py \
  "${DST_ROOT}/expert200_random_post_left_close_low_wrist_image_abs.hdf5" \
  --expected-demos 212 --expected-action-dim 7 \
  --required-obs-key left_close_low_image \
  --required-obs-key robot0_eye_in_hand_image \
  --required-obs-key robot0_eef_pos \
  --required-obs-key robot0_eef_quat \
  --required-obs-key robot0_gripper_qpos \
  --expected-obs-shape left_close_low_image=84,84,3 \
  --expected-obs-shape robot0_eye_in_hand_image=84,84,3 \
  --expected-obs-shape robot0_eef_pos=3 \
  --expected-obs-shape robot0_eef_quat=4 \
  --expected-obs-shape robot0_gripper_qpos=2
