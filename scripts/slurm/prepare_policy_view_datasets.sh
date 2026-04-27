#!/bin/bash
#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=prep_policy_views
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris-hgx-1,iris-hgx-2,iris-hp-z8
#SBATCH --output=/iris/u/jasonyan/slurm/%j_prep_policy_views.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_prep_policy_views.err

set -euo pipefail

TARGET="${1:-ph}"
if [[ "${TARGET}" != "ph" && "${TARGET}" != "expert200" ]]; then
  echo "Usage: sbatch $0 ph|expert200"
  exit 2
fi

source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate robodiff

REPO=/iris/u/jasonyan/repos/demonstration-information
mkdir -p /iris/u/jasonyan/slurm

cd "${REPO}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID=0
export PYTHONPATH="${REPO}/robomimic:${PYTHONPATH:-}"

EXTRA_ARGS=()
if [[ "${TARGET}" == "expert200" && -n "${EXPERT200_ZIP:-}" ]]; then
  EXTRA_ARGS+=(--expert200-zip "${EXPERT200_ZIP}")
fi
if [[ "${TARGET}" == "expert200" && -n "${EXPERT200_SOURCE:-}" ]]; then
  EXTRA_ARGS+=(--expert200-source "${EXPERT200_SOURCE}")
fi

python scripts/quality/prepare_policy_view_datasets.py "${TARGET}" "${EXTRA_ARGS[@]}"

if [[ "${TARGET}" == "ph" ]]; then
  python scripts/quality/verify_policy_view_dataset.py \
    /iris/u/jasonyan/data/policy_view_experiments/square_ph/square_ph_agent_wrist_image.hdf5 \
    --expected-demos 200 \
    --expected-action-dim 7 \
    --required-obs-key agentview_image \
    --required-obs-key robot0_eye_in_hand_image
  python scripts/quality/verify_policy_view_dataset.py \
    /iris/u/jasonyan/data/policy_view_experiments/square_ph/square_ph_left_close_low_wrist_image.hdf5 \
    --expected-demos 200 \
    --expected-action-dim 7 \
    --required-obs-key left_close_low_image \
    --required-obs-key robot0_eye_in_hand_image
  python scripts/quality/verify_policy_view_dataset.py \
    /iris/u/jasonyan/data/policy_view_experiments/square_ph/square_ph_agent_wrist_image_abs_50_seed42.hdf5 \
    --expected-demos 50 \
    --expected-action-dim 7 \
    --required-obs-key agentview_image \
    --required-obs-key robot0_eye_in_hand_image
  python scripts/quality/verify_policy_view_dataset.py \
    /iris/u/jasonyan/data/policy_view_experiments/square_ph/square_ph_left_close_low_wrist_image_abs_50_seed42.hdf5 \
    --expected-demos 50 \
    --expected-action-dim 7 \
    --required-obs-key left_close_low_image \
    --required-obs-key robot0_eye_in_hand_image
else
  python scripts/quality/verify_policy_view_dataset.py \
    /iris/u/jasonyan/data/policy_view_experiments/expert200/expert200_agent_wrist_image_abs.hdf5 \
    --expected-demos 200 \
    --expected-action-dim 7 \
    --required-obs-key agentview_image \
    --required-obs-key robot0_eye_in_hand_image
  python scripts/quality/verify_policy_view_dataset.py \
    /iris/u/jasonyan/data/policy_view_experiments/expert200/expert200_left_close_low_wrist_image_abs.hdf5 \
    --expected-demos 200 \
    --expected-action-dim 7 \
    --required-obs-key left_close_low_image \
    --required-obs-key robot0_eye_in_hand_image
fi
