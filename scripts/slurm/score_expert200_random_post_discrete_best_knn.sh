#!/bin/bash
# Score selected expert200 random-post discrete BC checkpoints and build BC-latent kNN pages.
#
# Usage:
#   sbatch scripts/slurm/score_expert200_random_post_discrete_best_knn.sh agent_wrist
#   sbatch scripts/slurm/score_expert200_random_post_discrete_best_knn.sh left_close_low_wrist
#   sbatch scripts/slurm/score_expert200_random_post_discrete_best_knn.sh both

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=expert200_knn
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris-hgx-1,iris-hgx-2,iris-hp-z8
#SBATCH --output=/iris/u/jasonyan/slurm/%j_expert200_knn.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_expert200_knn.err

set -euo pipefail

MODE="${1:-both}"
if [[ "${MODE}" != "agent_wrist" && "${MODE}" != "left_close_low_wrist" && "${MODE}" != "both" ]]; then
  echo "Usage: sbatch $0 agent_wrist|left_close_low_wrist|both" >&2
  exit 2
fi

source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate openx

REPO=/iris/u/jasonyan/repos/demonstration-information
DATA_ROOT="${DATA_ROOT:-/iris/u/jasonyan/data/policy_view_experiments/expert200_random_post_bc}"
SCORE_ROOT="${SCORE_ROOT:-/iris/u/jasonyan/data/robomimic_policy_scores/expert200_random_post_bc}"
KNN_ROOT="${KNN_ROOT:-/iris/u/jasonyan/data/knn_entropy/expert200_random_post}"

mkdir -p /iris/u/jasonyan/slurm "${SCORE_ROOT}" "${KNN_ROOT}"
cd "${REPO}"

export PYTHONPATH="${REPO}/robomimic:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID="${EGL_DEVICE_ID:-0}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

run_one() {
  local view="$1"
  local ckpt="$2"
  local dataset="$3"
  local image_key="$4"
  local label="$5"

  local score_dir="${SCORE_ROOT}/${label}"
  local score_name="${label}_all"
  local score_pkl="${score_dir}/${score_name}.pkl"
  local knn_dir="${KNN_ROOT}/${label}_bc_latent"

  mkdir -p "${score_dir}" "${knn_dir}"

  if [[ ! -f "${score_pkl}" ]]; then
    python scripts/quality/score_robomimic_policy_nll.py \
      --checkpoint "${ckpt}" \
      --dataset "${dataset}" \
      --output "${score_dir}" \
      --name "${score_name}" \
      --batch-size "${BATCH_SIZE:-128}" \
      --num-workers "${NUM_WORKERS:-0}"
  else
    echo "reusing existing score ${score_pkl}"
  fi

  python scripts/quality/visualize_random_post_knn_entropy.py \
    --checkpoint "${ckpt}" \
    --dataset "${dataset}" \
    --score-pkl "${score_pkl}" \
    --output "${knn_dir}" \
    --view-key "${image_key}" \
    --run-label "${label}_bc_latent" \
    --num-queries "${NUM_QUERIES:-24}" \
    --top-k "${TOP_K:-8}" \
    --batch-size "${BATCH_SIZE:-128}"

  echo "finished ${view}"
  echo "score ${score_pkl}"
  echo "knn ${knn_dir}/index.html"
}

if [[ "${MODE}" == "agent_wrist" || "${MODE}" == "both" ]]; then
  run_one \
    agent_wrist \
    /iris/u/jasonyan/data/robomimic_outputs/policy_view_experiments/expert200_random_post_bc_discrete_agent_wrist_seed1/20260502215527/models/model_epoch_24_best_validation_13.947654819488525.pth \
    "${DATA_ROOT}/expert200_random_post_agent_wrist_image_abs.hdf5" \
    agentview_image \
    discrete_agent_wrist_best_epoch24
fi

if [[ "${MODE}" == "left_close_low_wrist" || "${MODE}" == "both" ]]; then
  run_one \
    left_close_low_wrist \
    /iris/u/jasonyan/data/robomimic_outputs/policy_view_experiments/expert200_random_post_bc_discrete_left_close_low_wrist_seed1/20260502215524/models/model_epoch_20_best_validation_15.11133508682251.pth \
    "${DATA_ROOT}/expert200_random_post_left_close_low_wrist_image_abs.hdf5" \
    left_close_low_image \
    discrete_left_close_low_wrist_best_epoch20
fi
