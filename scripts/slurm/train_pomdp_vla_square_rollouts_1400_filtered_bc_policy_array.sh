#!/bin/bash
# Train GMM BC policies on full / 75% / 50% / 25% POMDP-VLA rollout datasets.
#
# Usage:
#   sbatch --array=1-4%4 scripts/slurm/train_pomdp_vla_square_rollouts_1400_filtered_bc_policy_array.sh

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=pvla1400_bc
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris-hgx-1,iris-hgx-2,iris-hp-z8
#SBATCH --output=/iris/u/jasonyan/slurm/%A_%a_pvla1400_bc.out
#SBATCH --error=/iris/u/jasonyan/slurm/%A_%a_pvla1400_bc.err

set -euo pipefail

REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"

DATASET_ROOT=/iris/u/jasonyan/data/pomdp_vla_square_rollouts_1400_filtered_bc_datasets/mi_score
CONFIG_DIR=/iris/u/jasonyan/data/pomdp_vla_square_rollouts_1400_filtered_bc_datasets/configs/robomimic
OUTPUT_DIR=/iris/u/jasonyan/data/robomimic_outputs/pomdp_vla_square_rollouts_1400_filtered_bc
DATASETS="pomdp_vla_square_rollouts_1400"
DROP_PCTS="0 25 50 75"
ALGO="${ALGO:-gmm}"
NUM_EPOCHS="${NUM_EPOCHS:-2000}"
WANDB_PROJECT="${WANDB_PROJECT:-pvla1400-filtered-bc}"

export REPO DATASET_ROOT CONFIG_DIR OUTPUT_DIR DATASETS DROP_PCTS ALGO NUM_EPOCHS WANDB_PROJECT

exec "${REPO}/scripts/slurm/train_deminf_mi_filtered_bc_policy_array.sh"
