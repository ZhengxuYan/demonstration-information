#!/bin/bash
# Evaluate available checkpoints for POMDP-VLA filtered BC policies.
#
# Usage:
#   CONDA_ENV=robodiff sbatch --array=1-4%4 scripts/slurm/evaluate_pomdp_vla_square_rollouts_1400_filtered_bc_rollouts_array.sh

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=pvla1400_eval
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris9,iris-hgx-1,iris-hgx-2,iris-hp-z8
#SBATCH --output=/iris/u/jasonyan/slurm/%A_%a_pvla1400_eval.out
#SBATCH --error=/iris/u/jasonyan/slurm/%A_%a_pvla1400_eval.err

set -euo pipefail

REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"
CONDA_ENV="${CONDA_ENV:-robodiff}"

CKPT_ROOT=/iris/u/jasonyan/data/robomimic_outputs/pomdp_vla_square_rollouts_1400_filtered_bc
OUT_ROOT=/iris/u/jasonyan/data/robomimic_rollout_scores/pomdp_vla_square_rollouts_1400_filtered_bc
DATASETS="pomdp_vla_square_rollouts_1400"
DROP_PCTS="0 25 50 75"
ALGO="${ALGO:-gmm}"
N_ROLLOUTS="${N_ROLLOUTS:-50}"
HORIZON="${HORIZON:-400}"
EPOCH_START="${EPOCH_START:-50}"
EPOCH_END="${EPOCH_END:-2000}"
EPOCH_STEP="${EPOCH_STEP:-50}"
SKIP_MISSING_EPOCHS="${SKIP_MISSING_EPOCHS:-1}"
ROBOSUITE_MIN_VERSION="${ROBOSUITE_MIN_VERSION:-1.2.0}"

export REPO CONDA_ENV CKPT_ROOT OUT_ROOT DATASETS DROP_PCTS ALGO N_ROLLOUTS HORIZON EPOCH_START EPOCH_END EPOCH_STEP SKIP_MISSING_EPOCHS ROBOSUITE_MIN_VERSION

exec "${REPO}/scripts/slurm/evaluate_deminf_mi_filtered_bc_rollouts_array.sh"
