#!/bin/bash
# Prepare Tian's camera-view datasets for DemInf and convert them to RLDS.
#
# Required:
#   ROLLOUT_IMAGES="/path/seed1.hdf5 /path/seed2.hdf5" ROLLOUT_ANNOTATIONS=/path/quality_annotations.csv sbatch ...
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

if [[ -z "${ROLLOUT_IMAGES:-${ROLLOUT_IMAGE:-}}" ]]; then
  echo "Set ROLLOUT_IMAGES='/path/seed1.hdf5 /path/seed2.hdf5' or ROLLOUT_IMAGE=/path/to/rollouts.hdf5" >&2
  exit 2
fi
ROLLOUT_IMAGES="${ROLLOUT_IMAGES:-${ROLLOUT_IMAGE}}"
read -r -a ROLLOUT_IMAGE_ARRAY <<< "${ROLLOUT_IMAGES}"
ROLLOUT_ARGS=()
for rollout_image in "${ROLLOUT_IMAGE_ARRAY[@]}"; do
  if [[ ! -f "${rollout_image}" ]]; then
    echo "ROLLOUT_IMAGE does not exist: ${rollout_image}" >&2
    exit 2
  fi
  ROLLOUT_ARGS+=(--rollout-image "${rollout_image}")
done
ANNOTATION_ARGS=()
if [[ -n "${ROLLOUT_ANNOTATIONS:-}" ]]; then
  if [[ ! -f "${ROLLOUT_ANNOTATIONS}" ]]; then
    echo "ROLLOUT_ANNOTATIONS does not exist: ${ROLLOUT_ANNOTATIONS}" >&2
    exit 2
  fi
  ANNOTATION_ARGS+=(--rollout-annotations "${ROLLOUT_ANNOTATIONS}")
fi

set +u
source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-openx}"
set -u

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
  "${ROLLOUT_ARGS[@]}" \
  --out-root "${OUT_ROOT}" \
  "${ANNOTATION_ARGS[@]}" \
  --positive-label "${POSITIVE_LABEL:-yes}" \
  --overwrite

for name in ph_agentview 400_agentview 400_left_close_low 400_mix; do
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
