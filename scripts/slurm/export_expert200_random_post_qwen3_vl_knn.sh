#!/bin/bash
# Export Qwen3-VL-Embedding-8B frame embeddings and build same-dataset kNN pages.
#
# Usage:
#   sbatch scripts/slurm/export_expert200_random_post_qwen3_vl_knn.sh agent_wrist
#   sbatch scripts/slurm/export_expert200_random_post_qwen3_vl_knn.sh left_close_low_wrist
#   sbatch scripts/slurm/export_expert200_random_post_qwen3_vl_knn.sh both

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=expert200_qwen_knn
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris-hgx-1,iris-hgx-2,iris-hp-z8
#SBATCH --output=/iris/u/jasonyan/slurm/%j_expert200_qwen_knn.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_expert200_qwen_knn.err

set -euo pipefail

MODE="${1:-both}"
if [[ "${MODE}" != "agent_wrist" && "${MODE}" != "left_close_low_wrist" && "${MODE}" != "both" ]]; then
  echo "Usage: sbatch $0 agent_wrist|left_close_low_wrist|both" >&2
  exit 2
fi

source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-openx}"

REPO=/iris/u/jasonyan/repos/demonstration-information
DATA_ROOT="${DATA_ROOT:-/iris/u/jasonyan/data/policy_view_experiments/expert200_random_post_bc}"
LATENT_ROOT="${LATENT_ROOT:-/iris/u/jasonyan/data/knn_entropy/expert200_random_post/qwen3_vl_latents}"
KNN_ROOT="${KNN_ROOT:-/iris/u/jasonyan/data/knn_entropy/expert200_random_post/qwen3_vl}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-Embedding-8B}"

mkdir -p /iris/u/jasonyan/slurm "${LATENT_ROOT}" "${KNN_ROOT}"
cd "${REPO}"

export PYTHONPATH="${REPO}:${REPO}/robomimic:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/iris/u/jasonyan/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"

run_one() {
  local view="$1"
  local dataset="$2"
  local latent_npz="${LATENT_ROOT}/expert200_random_post_${view}_qwen3_vl_embedding.npz"
  local output_dir="${KNN_ROOT}/${view}"

  if [[ ! -f "${latent_npz}" ]]; then
    extra_args=()
    if [[ -n "${MAX_FRAMES:-}" ]]; then
      extra_args+=(--max-frames "${MAX_FRAMES}")
    fi
    python scripts/quality/export_qwen3_vl_frame_embeddings.py \
      --dataset "${dataset}" \
      --output "${latent_npz}" \
      --view "${view}" \
      --model-name "${MODEL_NAME}" \
      --batch-size "${BATCH_SIZE:-8}" \
      "${extra_args[@]}"
  else
    echo "reusing existing latents ${latent_npz}"
  fi

  python scripts/quality/build_qwen3_vl_knn_review_page.py \
    --latent-npz "${latent_npz}" \
    --dataset "${dataset}" \
    --output "${output_dir}" \
    --view "${view}" \
    --run-label "expert200_random_post_${view}_qwen3_vl" \
    --num-queries "${NUM_QUERIES:-24}" \
    --top-k "${TOP_K:-8}"

  echo "latent ${latent_npz}"
  echo "knn ${output_dir}/index.html"
}

if [[ "${MODE}" == "agent_wrist" || "${MODE}" == "both" ]]; then
  run_one agent_wrist "${DATA_ROOT}/expert200_random_post_agent_wrist_image_abs.hdf5"
fi

if [[ "${MODE}" == "left_close_low_wrist" || "${MODE}" == "both" ]]; then
  run_one left_close_low_wrist "${DATA_ROOT}/expert200_random_post_left_close_low_wrist_image_abs.hdf5"
fi
