#!/bin/bash
#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=random_post_views
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris-hgx-1,iris-hgx-2,iris-hp-z8
#SBATCH --output=/iris/u/jasonyan/slurm/%j_random_post_views.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_random_post_views.err

set -euo pipefail

if [[ -z "${RANDOM_POST_IMAGE_HDF5:-}" ]]; then
  echo "Set RANDOM_POST_IMAGE_HDF5=/path/to/randomized-post/image.hdf5"
  exit 2
fi

source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate openx

REPO=/iris/u/jasonyan/repos/demonstration-information
OUT_ROOT=/iris/u/jasonyan/data/policy_view_experiments
OVERWRITE_FLAG=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_FLAG=(--overwrite)
fi

mkdir -p /iris/u/jasonyan/slurm
cd "${REPO}"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID="${EGL_DEVICE_ID:-0}"
export PYTHONPATH="${REPO}/robomimic:${PYTHONPATH:-}"

python scripts/quality/prepare_policy_view_datasets.py random_post \
  --random-post-image "${RANDOM_POST_IMAGE_HDF5}" \
  --out-root "${OUT_ROOT}" \
  --valid-ratio "${VALID_RATIO:-0.1}" \
  --split-seed "${SPLIT_SEED:-0}" \
  "${OVERWRITE_FLAG[@]}"

python scripts/quality/verify_policy_view_dataset.py \
  "${OUT_ROOT}/random_post/random_post_agent_wrist_image.hdf5" \
  --expected-action-dim 7 \
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
  "${OUT_ROOT}/random_post/random_post_left_close_low_wrist_image.hdf5" \
  --expected-action-dim 7 \
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

python - <<'PY'
import h5py
from pathlib import Path

for path in [
    Path("/iris/u/jasonyan/data/policy_view_experiments/random_post/random_post_agent_wrist_image.hdf5"),
    Path("/iris/u/jasonyan/data/policy_view_experiments/random_post/random_post_left_close_low_wrist_image.hdf5"),
]:
    with h5py.File(path, "r") as f:
        train = {x.decode("utf-8") for x in f["mask/train"][:]}
        valid = {x.decode("utf-8") for x in f["mask/valid"][:]}
    overlap = train & valid
    print(f"{path}: train={len(train)} valid={len(valid)} overlap={len(overlap)}")
    if overlap or not train or not valid:
        raise SystemExit(f"bad split in {path}")
PY
