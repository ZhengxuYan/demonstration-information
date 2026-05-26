#!/bin/bash
# Score the 1400 POMDP-VLA Square rollout dataset using VAEs trained on it.
#
# Usage:
#   sbatch scripts/slurm/score_trained_pomdp_vla_square_rollouts_1400_deminf.sh

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris9,iris-hgx-1,iris-hgx-2,iris-hp-z8
#SBATCH --job-name=score_tr_pvla
#SBATCH --output=/iris/u/jasonyan/slurm/%j_score_tr_pvla.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_score_tr_pvla.err

set -euo pipefail

set +u
source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-openx}"
set -u

REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"
DATASET_NAME="${DATASET_NAME:-pomdp_vla_square_rollouts_1400}"
RLDS_ROOT="${RLDS_ROOT:-/iris/u/jasonyan/data/pomdp_vla_square_rollouts_1400_rlds}"
TRAIN_ROOT="${TRAIN_ROOT:-/iris/u/jasonyan/data/deminf_outputs/pomdp_vla_square_rollouts_1400_train}"
OUT_ROOT="${OUT_ROOT:-/iris/u/jasonyan/data/deminf_outputs/pomdp_vla_square_rollouts_1400_scores_trained}"

RLDS="${RLDS_ROOT}/${DATASET_NAME}/robo_mimic/1.0.0"
TRAIN_DIR="${TRAIN_ROOT}/${DATASET_NAME}"
OUTPUT="${OUT_ROOT}/image_proprio/${DATASET_NAME}"

if [[ ! -d "${RLDS}" ]]; then
  echo "missing RLDS=${RLDS}" >&2
  exit 1
fi
if [[ ! -d "${TRAIN_DIR}" ]]; then
  echo "missing TRAIN_DIR=${TRAIN_DIR}" >&2
  exit 1
fi

OBS_CKPT="$(find "${TRAIN_DIR}" -maxdepth 1 -type d -name "${DATASET_NAME}_image_proprio_obs_vae_seed1_*" | sort | tail -1)"
ACTION_CKPT="$(find "${TRAIN_DIR}" -maxdepth 1 -type d -name "${DATASET_NAME}_action_vae_seed1_*" | sort | tail -1)"

if [[ -z "${OBS_CKPT}" || ! -d "${OBS_CKPT}/100000" ]]; then
  echo "missing completed obs checkpoint: ${OBS_CKPT}" >&2
  exit 1
fi
if [[ -z "${ACTION_CKPT}" || ! -d "${ACTION_CKPT}/100000" ]]; then
  echo "missing completed action checkpoint: ${ACTION_CKPT}" >&2
  exit 1
fi

mkdir -p /iris/u/jasonyan/slurm "${OUTPUT}"
cd "${REPO}"

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TF_FORCE_GPU_ALLOW_GROWTH=true

echo "hostname=$(hostname)"
echo "dataset_name=${DATASET_NAME}"
echo "obs_ckpt=${OBS_CKPT}"
echo "action_ckpt=${ACTION_CKPT}"
echo "rlds=${RLDS}"
echo "output=${OUTPUT}"

python scripts/quality/estimate_quality_combined_robomimic.py \
  --obs_ckpt "${OBS_CKPT}" \
  --action_ckpt "${ACTION_CKPT}" \
  --square_dataset_name "${DATASET_NAME}" \
  --square_path_override "${RLDS}" \
  --score_split "${SCORE_SPLIT:-train+val}" \
  --output "${OUTPUT}" \
  --batch_size "${BATCH_SIZE:-1024}"

python scripts/quality/deminf_score_pkl_to_episode_csv.py \
  --score-pkl "${OUTPUT}/${DATASET_NAME}.pkl" \
  --dataset "${DATASET_NAME}" \
  --output "${OUT_ROOT}/image_proprio/episode_scores.csv" \
  --source rollout \
  --view agentview
