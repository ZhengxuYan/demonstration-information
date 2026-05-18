#!/bin/bash
# Score Tian's three camera-view datasets with Square PH DemInf checkpoints.
#
# Usage:
#   sbatch scripts/slurm/score_deminf_camera_view_datasets.sh image
#   sbatch scripts/slurm/score_deminf_camera_view_datasets.sh image_proprio
#   sbatch scripts/slurm/score_deminf_camera_view_datasets.sh all

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=score_deminf_cam
#SBATCH --output=/iris/u/jasonyan/slurm/%j_score_deminf_cam.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_score_deminf_cam.err

set -euo pipefail

MODE="${1:-all}"
if [[ "${MODE}" != "image" && "${MODE}" != "image_proprio" && "${MODE}" != "all" ]]; then
  echo "Expected mode to be one of: image, image_proprio, all. Got: ${MODE}" >&2
  exit 2
fi

source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-openx}"

REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"
RLDS_ROOT="${RLDS_ROOT:-/iris/u/jasonyan/data/deminf_camera_view_rlds}"
CKPT_ROOT="${CKPT_ROOT:-/iris/u/jasonyan/data/deminf_outputs/square_ph_wrist_image}"
OUT_ROOT="${OUT_ROOT:-/iris/u/jasonyan/data/deminf_outputs/camera_view_scores}"

mkdir -p /iris/u/jasonyan/slurm "${OUT_ROOT}"
cd "${REPO}"

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TF_FORCE_GPU_ALLOW_GROWTH=true

latest_ckpt_dir() {
  local pattern="$1"
  find "${CKPT_ROOT}" -maxdepth 1 -type d -name "${pattern}" | sort | tail -n 1
}

IMAGE_OBS_CKPT="${IMAGE_OBS_CKPT:-$(latest_ckpt_dir 'square_ph_wrist_image_only_obs_vae_seed1_*')}"
IMAGE_PROPRIO_OBS_CKPT="${IMAGE_PROPRIO_OBS_CKPT:-$(latest_ckpt_dir 'square_ph_wrist_image_proprio_obs_vae_seed1_*')}"
ACTION_CKPT="${ACTION_CKPT:-$(latest_ckpt_dir 'square_ph_action_vae_seed1_*')}"

if [[ -z "${ACTION_CKPT}" ]]; then
  echo "Could not find square_ph_action_vae_seed1_* under ${CKPT_ROOT}" >&2
  exit 1
fi

run_score() {
  local obs_ckpt="$1"
  local variant="$2"
  local dataset_name="$3"
  local rlds_path="${RLDS_ROOT}/${dataset_name}/robo_mimic/1.0.0"
  local output="${OUT_ROOT}/${variant}/${dataset_name}"

  if [[ -z "${obs_ckpt}" ]]; then
    echo "Missing obs checkpoint for ${variant}" >&2
    exit 1
  fi
  if [[ ! -d "${rlds_path}" ]]; then
    echo "Missing RLDS path: ${rlds_path}" >&2
    echo "Run scripts/slurm/prepare_deminf_camera_view_datasets.sh first." >&2
    exit 1
  fi

  python scripts/quality/estimate_quality_combined_robomimic.py \
    --obs_ckpt="${obs_ckpt}" \
    --action_ckpt="${ACTION_CKPT}" \
    --square_dataset_name="square_ph" \
    --square_path_override="${rlds_path}" \
    --batch_size=1024 \
    --output="${output}"
}

for dataset_name in ph_agentview 400_agentview 400_mix; do
  if [[ "${MODE}" == "image" || "${MODE}" == "all" ]]; then
    run_score "${IMAGE_OBS_CKPT}" image_only "${dataset_name}"
  fi
  if [[ "${MODE}" == "image_proprio" || "${MODE}" == "all" ]]; then
    run_score "${IMAGE_PROPRIO_OBS_CKPT}" image_proprio "${dataset_name}"
  fi
done
