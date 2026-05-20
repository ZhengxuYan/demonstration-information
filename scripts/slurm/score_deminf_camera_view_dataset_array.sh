#!/bin/bash
# Score one camera-view DemInf dataset per Slurm array task.
#
# Default usage, image+proprio only over four datasets:
#   sbatch --array=1-4%4 scripts/slurm/score_deminf_camera_view_dataset_array.sh
#
# Optional:
#   MODE=image_proprio|image
#   DATASETS="ph_agentview 400_agentview 400_left_close_low 400_mix"
#   CONDA_ENV=openx

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=score_deminf_cam
#SBATCH --output=/iris/u/jasonyan/slurm/%A_%a_score_deminf_cam.out
#SBATCH --error=/iris/u/jasonyan/slurm/%A_%a_score_deminf_cam.err

set -euo pipefail

MODE="${MODE:-image_proprio}"
if [[ "${MODE}" != "image" && "${MODE}" != "image_proprio" ]]; then
  echo "Expected MODE=image or MODE=image_proprio. Got: ${MODE}" >&2
  exit 2
fi

DATASETS="${DATASETS:-ph_agentview 400_agentview 400_left_close_low 400_mix}"
read -r -a DATASET_ARRAY <<< "${DATASETS}"

ROW="${SLURM_ARRAY_TASK_ID:?Submit with --array=1-N}"
if (( ROW < 1 || ROW > ${#DATASET_ARRAY[@]} )); then
  echo "Array row ${ROW} outside 1..${#DATASET_ARRAY[@]}" >&2
  exit 2
fi
DATASET_NAME="${DATASET_ARRAY[$((ROW - 1))]}"

set +u
source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-openx}"
set -u

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

case "${MODE}" in
  image)
    OBS_CKPT="${IMAGE_OBS_CKPT}"
    OUT_VARIANT="image_only"
    ;;
  image_proprio)
    OBS_CKPT="${IMAGE_PROPRIO_OBS_CKPT}"
    OUT_VARIANT="image_proprio"
    ;;
esac

if [[ -z "${OBS_CKPT}" ]]; then
  echo "Missing obs checkpoint for ${MODE}" >&2
  exit 1
fi

RLDS_PATH="${RLDS_ROOT}/${DATASET_NAME}/robo_mimic/1.0.0"
OUTPUT="${OUT_ROOT}/${OUT_VARIANT}/${DATASET_NAME}"

if [[ ! -d "${RLDS_PATH}" ]]; then
  echo "Missing RLDS path: ${RLDS_PATH}" >&2
  echo "Run scripts/slurm/prepare_deminf_camera_view_datasets.sh first." >&2
  exit 1
fi

echo "mode=${MODE}"
echo "dataset=${DATASET_NAME}"
echo "obs_ckpt=${OBS_CKPT}"
echo "action_ckpt=${ACTION_CKPT}"
echo "rlds=${RLDS_PATH}"
echo "output=${OUTPUT}"

python scripts/quality/estimate_quality_combined_robomimic.py \
  --obs_ckpt="${OBS_CKPT}" \
  --action_ckpt="${ACTION_CKPT}" \
  --square_dataset_name="square_ph" \
  --square_path_override="${RLDS_PATH}" \
  --batch_size=1024 \
  --output="${OUTPUT}"
