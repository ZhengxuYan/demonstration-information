#!/bin/bash
# Train a state-action VAE on the expert200/random-post distribution.
#
# The VAE config type is "sa": one latent embeds observation state, both
# agent+wrist images, and action. The training input must be an RLDS/TFDS
# builder directory; set EXPERT200_RANDOM_POST_RLDS if the default is wrong.

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=expert200_sa_vae
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris-hgx-1,iris-hgx-2,iris-hp-z8
#SBATCH --output=/iris/u/jasonyan/slurm/%j_expert200_sa_vae.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_expert200_sa_vae.err

set -euo pipefail

source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate openx

REPO=/iris/u/jasonyan/repos/demonstration-information
OUT="${OUT:-/iris/u/jasonyan/data/deminf_outputs/expert200_random_post_image}"
EXPERT200_RANDOM_POST_RLDS="${EXPERT200_RANDOM_POST_RLDS:-/iris/u/jasonyan/data/expert200_random_post_rlds/robo_mimic/1.0.0}"
NAME="${NAME:-expert200_random_post_both_sa_vae_seed1}"

mkdir -p /iris/u/jasonyan/slurm "${OUT}"

cd "${REPO}"

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TF_FORCE_GPU_ALLOW_GROWTH=true

if [[ ! -d "${EXPERT200_RANDOM_POST_RLDS}" ]]; then
  echo "Missing EXPERT200_RANDOM_POST_RLDS=${EXPERT200_RANDOM_POST_RLDS}" >&2
  echo "Set EXPERT200_RANDOM_POST_RLDS to the TFDS/RLDS builder dir for expert200/random-post." >&2
  exit 1
fi

python scripts/train.py \
  --config="configs/quality/vae_robomimic_image.py:expert200_random_post,sa,1,both,expert200_random_post=${EXPERT200_RANDOM_POST_RLDS}" \
  --path "${OUT}" \
  --name "${NAME}"
