#!/bin/bash
# Score the merged 1400 POMDP-VLA Square rollout dataset with DemInf.
#
# Usage:
#   sbatch scripts/slurm/score_pomdp_vla_square_rollouts_1400_deminf.sh

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --gres=gpu:1
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris9,iris-hgx-1,iris-hgx-2,iris-hp-z8
#SBATCH --job-name=score_pvla1400
#SBATCH --output=/iris/u/jasonyan/slurm/%j_score_pvla1400.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_score_pvla1400.err

set -euo pipefail

set +u
source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-openx}"
set -u

REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"
DATASET_NAME="${DATASET_NAME:-pomdp_vla_square_rollouts_1400}"
DATASET_HDF5="${DATASET_HDF5:-/iris/u/jasonyan/data/pomdp_vla_square_rollouts_1400/image.hdf5}"
CKPT_ROOT="${CKPT_ROOT:-/iris/u/jasonyan/data/deminf_outputs/square_ph_wrist_image}"
OUT_ROOT="${OUT_ROOT:-/iris/u/jasonyan/data/deminf_outputs/pomdp_vla_square_rollouts_1400_scores}"
OUTPUT="${OUT_ROOT}/image_proprio"

if [[ ! -f "${DATASET_HDF5}" ]]; then
  echo "missing DATASET_HDF5=${DATASET_HDF5}" >&2
  exit 1
fi

OBS_CKPT="${OBS_CKPT:-$(find "${CKPT_ROOT}" -maxdepth 1 -type d -name "square_ph_wrist_image_proprio_obs_vae_seed1_*" | sort | tail -1)}"
ACTION_CKPT="${ACTION_CKPT:-$(find "${CKPT_ROOT}" -maxdepth 1 -type d -name "square_ph_action_vae_seed1_*" | sort | tail -1)}"

if [[ -z "${OBS_CKPT}" || ! -d "${OBS_CKPT}/100000" ]]; then
  echo "missing obs checkpoint: ${OBS_CKPT}" >&2
  exit 1
fi
if [[ -z "${ACTION_CKPT}" || ! -d "${ACTION_CKPT}/100000" ]]; then
  echo "missing action checkpoint: ${ACTION_CKPT}" >&2
  exit 1
fi

mkdir -p /iris/u/jasonyan/slurm "${OUTPUT}"
cd "${REPO}"

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TF_FORCE_GPU_ALLOW_GROWTH=true

echo "hostname=$(hostname)"
echo "dataset_name=${DATASET_NAME}"
echo "dataset_hdf5=${DATASET_HDF5}"
echo "obs_ckpt=${OBS_CKPT}"
echo "action_ckpt=${ACTION_CKPT}"
echo "output=${OUTPUT}"

python scripts/quality/score_robomimic_hdf5.py \
  --obs_ckpt "${OBS_CKPT}" \
  --action_ckpt "${ACTION_CKPT}" \
  --dataset "${DATASET_NAME}=1=${DATASET_HDF5}" \
  --camera both \
  --batch_size "${BATCH_SIZE:-1024}" \
  --output "${OUTPUT}"

python scripts/quality/deminf_score_pkl_to_episode_csv.py \
  --score-pkl "${OUTPUT}/${DATASET_NAME}.pkl" \
  --dataset "${DATASET_NAME}" \
  --output "${OUTPUT}/episode_scores.csv" \
  --source rollout \
  --view agentview
