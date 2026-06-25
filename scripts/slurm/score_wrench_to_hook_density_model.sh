#!/bin/bash
# Score one trained wrench-to-hook robomimic BC action-density model.

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=wth_den_score
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris-hp-z8,iliad1,iliad2,iliad3,iliad4
#SBATCH --output=/iris/u/jasonyan/slurm/%j_wth_den_score.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_wth_den_score.err

set -euo pipefail

export TASK_TAG="${TASK_TAG:-wrench_to_hook}"
export RUN_PREFIX="${RUN_PREFIX:-wrench_to_hook}"
export ACTION_SOURCE="${ACTION_SOURCE:-cartesian_velocity}"
export ACTION_TARGET="${ACTION_TARGET:-single}"
export ACTION_NORMALIZATION="${ACTION_NORMALIZATION:-none}"
export ALGOS="${ALGOS:?Set ALGOS to one algo, e.g. gaussian or gmm}"
export CONDITIONS="${CONDITIONS:?Set CONDITIONS to one condition, e.g. image_state}"

REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"
exec "${REPO}/scripts/slurm/score_pen_in_cup_density_models_array.sh"
