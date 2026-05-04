#!/bin/bash
# Export expert200/random-post state-action VAE latents and build kNN entropy pages.
#
# Usage:
#   sbatch scripts/slurm/export_expert200_random_post_sa_vae_knn.sh agent_wrist
#   sbatch scripts/slurm/export_expert200_random_post_sa_vae_knn.sh left_close_low_wrist
#   sbatch scripts/slurm/export_expert200_random_post_sa_vae_knn.sh both
#
# Note: the trained SA-VAE uses agentview+wrist observations. The left-close-low
# page can display left-close-low frames and left policy NLL, but its kNN latent
# still comes from the agentview+wrist SA-VAE.

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=expert200_sa_knn
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris-hgx-1,iris-hgx-2,iris-hp-z8
#SBATCH --output=/iris/u/jasonyan/slurm/%j_expert200_sa_knn.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_expert200_sa_knn.err

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
LATENT_ROOT="${LATENT_ROOT:-/iris/u/jasonyan/data/knn_entropy/expert200_random_post/latents}"
SA_VAE_CKPT="${SA_VAE_CKPT:-/iris/u/jasonyan/data/deminf_outputs/expert200_random_post_image/expert200_random_post_both_sa_vae_seed1_20260503_231630}"

AGENT_DATA="${DATA_ROOT}/expert200_random_post_agent_wrist_image_abs.hdf5"
LATENT_NPZ="${LATENT_ROOT}/expert200_random_post_agent_wrist_sa_vae_latents.npz"

mkdir -p /iris/u/jasonyan/slurm "${LATENT_ROOT}" "${KNN_ROOT}"
cd "${REPO}"

export PYTHONPATH="${REPO}/robomimic:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID="${EGL_DEVICE_ID:-0}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TF_FORCE_GPU_ALLOW_GROWTH=true

if [[ ! -d "${SA_VAE_CKPT}" ]]; then
  echo "Missing SA_VAE_CKPT=${SA_VAE_CKPT}" >&2
  exit 1
fi

if [[ ! -f "${LATENT_NPZ}" ]]; then
  python scripts/quality/export_robomimic_sa_vae_latents.py \
    --checkpoint "${SA_VAE_CKPT}" \
    --dataset "${AGENT_DATA}" \
    --output "${LATENT_NPZ}" \
    --batch-size "${LATENT_BATCH_SIZE:-512}"
else
  echo "reusing existing latents ${LATENT_NPZ}"
fi

build_page() {
  local view="$1"
  local dataset="$2"
  local score_pkl="$3"
  local image_key="$4"
  local output_dir="$5"
  local label="$6"

  if [[ ! -f "${score_pkl}" ]]; then
    echo "Missing score pkl ${score_pkl}" >&2
    echo "Run scripts/slurm/score_expert200_random_post_discrete_best_knn.sh first." >&2
    exit 1
  fi

  python scripts/quality/visualize_random_post_knn_entropy.py \
    --latent-npz "${LATENT_NPZ}" \
    --latent-label "SA-VAE state-action latent (agent+wrist)" \
    --dataset "${dataset}" \
    --score-pkl "${score_pkl}" \
    --output "${output_dir}" \
    --view-key "${image_key}" \
    --run-label "${label}" \
    --num-queries "${NUM_QUERIES:-24}" \
    --top-k "${TOP_K:-8}" \
    --batch-size "${BATCH_SIZE:-128}"

  echo "finished ${view}"
  echo "knn ${output_dir}/index.html"
}

if [[ "${MODE}" == "agent_wrist" || "${MODE}" == "both" ]]; then
  build_page \
    agent_wrist \
    "${AGENT_DATA}" \
    "${SCORE_ROOT}/discrete_agent_wrist_best_epoch24/discrete_agent_wrist_best_epoch24_all.pkl" \
    agentview_image \
    "${KNN_ROOT}/discrete_agent_wrist_best_epoch24_sa_vae_latent" \
    "discrete_agent_wrist_best_epoch24_sa_vae_latent"
fi

if [[ "${MODE}" == "left_close_low_wrist" || "${MODE}" == "both" ]]; then
  build_page \
    left_close_low_wrist \
    "${DATA_ROOT}/expert200_random_post_left_close_low_wrist_image_abs.hdf5" \
    "${SCORE_ROOT}/discrete_left_close_low_wrist_best_epoch20/discrete_left_close_low_wrist_best_epoch20_all.pkl" \
    left_close_low_image \
    "${KNN_ROOT}/discrete_left_close_low_wrist_best_epoch20_sa_vae_latent_agent_wrist" \
    "discrete_left_close_low_wrist_best_epoch20_sa_vae_latent_agent_wrist"
fi
