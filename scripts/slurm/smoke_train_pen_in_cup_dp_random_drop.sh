#!/bin/bash
# Short debug run for the pen-in-cup DemInf/OpenX diffusion policy config.
#
# Usage:
#   sbatch scripts/slurm/smoke_train_pen_in_cup_dp_random_drop.sh

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=smoke_pic_dp
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris-hgx-1,iris-hgx-2,iris-hp-z8
#SBATCH --output=/iris/u/jasonyan/slurm/%j_smoke_pic_dp.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_smoke_pic_dp.err

set -euo pipefail

set +u
source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-openx}"
set -u

REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"
RLDS_PATH="${RLDS_PATH:-/iris/u/jasonyan/data/droid_pen_in_cup_06072026_rlds/droid_pen_in_cup/1.0.0}"
SCORE_ROOT="${SCORE_ROOT:-/iris/u/jasonyan/data/droid_pen_in_cup_06072026_random_drop_scores}"
OUT_ROOT="${OUT_ROOT:-/iris/u/jasonyan/data/deminf_dp_pen_in_cup_06072026_smoke}"
DROP_PERCENT="${DROP_PERCENT:-25}"
SEED="${SEED:-1}"

cd "${REPO}"
python scripts/train.py \
  --config="configs/bc/droid_pen_in_cup_dp_random_drop.py:pen_in_cup,${DROP_PERCENT},random,${SEED},${SCORE_ROOT},${RLDS_PATH}" \
  --config.steps=2 \
  --config.log_freq=1 \
  --config.val_freq=1000000 \
  --config.save_freq=1000000 \
  --config.dataloader.batch_size=2 \
  --config.dataloader.shuffle_size=10 \
  --config.dataloader.cache=false \
  --config.dataloader.prefetch=0 \
  --path="${OUT_ROOT}" \
  --name="smoke_pen_in_cup_dp_random_drop_${DROP_PERCENT}_seed${SEED}" \
  --project="${WANDB_PROJECT:-deminf-dp-pen-in-cup-smoke}" \
  --include_timestamp=false
