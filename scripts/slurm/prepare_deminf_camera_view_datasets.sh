#!/bin/bash
# Prepare Tian's three camera-view datasets for DemInf and convert them to RLDS.
#
# Required:
#   ROLLOUT_IMAGE=/path/to/200_successful_rollouts_image.hdf5 sbatch ...
#
# Optional:
#   PH_IMAGE=/path/to/square/ph/image.hdf5
#   OUT_ROOT=/iris/u/jasonyan/data/deminf_camera_view_datasets
#   RLDS_ROOT=/iris/u/jasonyan/data/deminf_camera_view_rlds

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=prep_deminf_cam
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris-hgx-1,iris-hgx-2,iris-hp-z8
#SBATCH --output=/iris/u/jasonyan/slurm/%j_prep_deminf_cam.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_prep_deminf_cam.err

set -euo pipefail

if [[ -z "${ROLLOUT_IMAGE:-}" ]]; then
  echo "Set ROLLOUT_IMAGE=/path/to/200_successful_rollouts_image.hdf5" >&2
  exit 2
fi

source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-openx}"

REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"
PH_IMAGE="${PH_IMAGE:-/iris/u/jasonyan/data/diffusion_policy/robomimic/datasets/square/ph/image.hdf5}"
OUT_ROOT="${OUT_ROOT:-/iris/u/jasonyan/data/deminf_camera_view_datasets}"
RLDS_ROOT="${RLDS_ROOT:-/iris/u/jasonyan/data/deminf_camera_view_rlds}"

mkdir -p /iris/u/jasonyan/slurm "${OUT_ROOT}" "${RLDS_ROOT}"
cd "${REPO}"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID="${EGL_DEVICE_ID:-0}"
export PYTHONPATH="${REPO}/robomimic:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

python scripts/quality/prepare_deminf_camera_view_datasets.py \
  --ph-image "${PH_IMAGE}" \
  --rollout-image "${ROLLOUT_IMAGE}" \
  --out-root "${OUT_ROOT}" \
  --overwrite

for name in ph_agentview 400_agentview 400_mix; do
  dataset_dir="${OUT_ROOT}/${name}"
  data_dir="${RLDS_ROOT}/${name}"
  rm -rf "${data_dir}/robo_mimic"
  (
    cd rlds/robomimic
    tfds build \
      --manual_dir "${dataset_dir}" \
      --data_dir "${data_dir}"
  )
  echo "${name} RLDS: ${data_dir}/robo_mimic/1.0.0"
done
