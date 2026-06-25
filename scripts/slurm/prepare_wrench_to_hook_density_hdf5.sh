#!/bin/bash
# Export one wrench-to-hook RLDS builder directory to robomimic HDF5 for density sweeps.

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --job-name=wth_density_h5
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris-hp-z8
#SBATCH --output=/iris/u/jasonyan/slurm/%j_wth_density_h5.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_wth_density_h5.err

set -euo pipefail

export TASK_TAG="${TASK_TAG:-wrench_to_hook}"
export ACTION_SOURCE="${ACTION_SOURCE:-cartesian_velocity}"
export ACTION_TARGET="${ACTION_TARGET:-single}"
export ACTION_NORMALIZATION="${ACTION_NORMALIZATION:-none}"
export NUM_FOLDS="${NUM_FOLDS:-2}"
export ENV_NAME="${ENV_NAME:-wrench_to_hook_density}"

REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"
exec "${REPO}/scripts/slurm/prepare_pen_in_cup_density_hdf5.sh"
