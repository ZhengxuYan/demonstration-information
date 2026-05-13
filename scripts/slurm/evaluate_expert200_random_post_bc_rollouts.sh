#!/bin/bash
# Evaluate Expert200 random-post robomimic BC checkpoints by rollout success.
#
# Usage:
#   sbatch scripts/slurm/evaluate_expert200_random_post_bc_rollouts.sh gmm agent_wrist
#   sbatch scripts/slurm/evaluate_expert200_random_post_bc_rollouts.sh discrete left_close_low_wrist
#   sbatch scripts/slurm/evaluate_expert200_random_post_bc_rollouts.sh discrete_smooth agent_wrist
#
# Optional env:
#   RUN_NAME=exact_run_dir_name
#   EPOCHS="50 500 1000 1500 2000"
#   N_ROLLOUTS=20 HORIZON=400

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=rp_bc_rollout
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris9,iris-hgx-1,iris-hgx-2,iris-hp-z8
#SBATCH --output=/iris/u/jasonyan/slurm/%j_rp_bc_rollout.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_rp_bc_rollout.err

set -euo pipefail

POLICY="${1:?usage: $0 gmm|discrete|discrete_smooth agent_wrist|left_close_low_wrist}"
VIEW="${2:?usage: $0 gmm|discrete|discrete_smooth agent_wrist|left_close_low_wrist}"

if [[ "${POLICY}" != "gmm" && "${POLICY}" != "discrete" && "${POLICY}" != "discrete_smooth" ]]; then
  echo "bad policy: ${POLICY}" >&2
  exit 2
fi
if [[ "${VIEW}" != "agent_wrist" && "${VIEW}" != "left_close_low_wrist" ]]; then
  echo "bad view: ${VIEW}" >&2
  exit 2
fi

source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
CONDA_ENV="${CONDA_ENV:-openx}"
conda activate "${CONDA_ENV}"

REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"
CKPT_ROOT="${CKPT_ROOT:-/iris/u/jasonyan/data/robomimic_outputs/policy_view_experiments}"
OUT_ROOT="${OUT_ROOT:-/iris/u/jasonyan/data/robomimic_rollout_scores/expert200_random_post_bc_success}"
N_ROLLOUTS="${N_ROLLOUTS:-20}"
HORIZON="${HORIZON:-400}"
SEED="${SEED:-0}"

if [[ -z "${RUN_NAME:-}" ]]; then
  RUN_NAME="expert200_random_post_bc_${POLICY}_${VIEW}_seed1"
fi
RUN_ROOT="${CKPT_ROOT}/${RUN_NAME}"

mkdir -p /iris/u/jasonyan/slurm "${OUT_ROOT}/${RUN_NAME}"
cd "${REPO}"

python "${REPO}/scripts/setup/patch_robomimic_optional_diffusion.py"
python "${REPO}/scripts/setup/patch_robomimic_discrete_action.py"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID="${EGL_DEVICE_ID:-0}"
export PYTHONPATH="${REPO}/robomimic:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

mapfile -t ALL_CKPTS < <(find "${RUN_ROOT}" -path "*/models/model_epoch_*.pth" -type f | sort -V)
if [[ "${#ALL_CKPTS[@]}" -eq 0 ]]; then
  echo "missing checkpoints under ${RUN_ROOT}" >&2
  exit 1
fi

epoch_of() {
  basename "$1" | sed -E 's/model_epoch_([0-9]+).*/\1/'
}

CKPTS=()
if [[ -n "${EPOCHS:-}" ]]; then
  for epoch in ${EPOCHS}; do
    match=""
    for ckpt in "${ALL_CKPTS[@]}"; do
      if [[ "$(basename "${ckpt}")" == model_epoch_${epoch}* ]]; then
        match="${ckpt}"
        break
      fi
    done
    if [[ -z "${match}" ]]; then
      echo "missing epoch ${epoch} for ${RUN_NAME}" >&2
      exit 1
    fi
    CKPTS+=("${match}")
  done
else
  BEST=""
  for ckpt in "${ALL_CKPTS[@]}"; do
    if [[ "$(basename "${ckpt}")" == *best_validation* ]]; then
      BEST="${ckpt}"
    fi
  done
  COUNT="${#ALL_CKPTS[@]}"
  for q in 0 25 50 75 100; do
    if [[ "${q}" == "0" && -n "${BEST}" ]]; then
      candidate="${BEST}"
    else
      idx=$(( (COUNT - 1) * q / 100 ))
      candidate="${ALL_CKPTS[$idx]}"
    fi
    exists=0
    for ckpt in "${CKPTS[@]}"; do
      if [[ "${ckpt}" == "${candidate}" ]]; then
        exists=1
      fi
    done
    if [[ "${exists}" == "0" ]]; then
      CKPTS+=("${candidate}")
    fi
  done
fi

LEFT_ARGS=()
if [[ "${VIEW}" == "left_close_low_wrist" ]]; then
  LEFT_ARGS=(--left-close-low)
fi

printf 'run %s\n' "${RUN_NAME}"
printf 'policy %s view %s\n' "${POLICY}" "${VIEW}"
printf 'n_rollouts %s horizon %s seed %s\n' "${N_ROLLOUTS}" "${HORIZON}" "${SEED}"
printf 'checkpoints %s\n' "${#CKPTS[@]}"
printf '%s\n' "${CKPTS[@]}"

python scripts/quality/evaluate_robomimic_bc_checkpoint_rollouts.py \
  --dataset expert200_random_post \
  --policy "${POLICY}" \
  --view "${VIEW}" \
  --run-name "${RUN_NAME}" \
  --checkpoints "${CKPTS[@]}" \
  --output-dir "${OUT_ROOT}/${RUN_NAME}" \
  --n-rollouts "${N_ROLLOUTS}" \
  --horizon "${HORIZON}" \
  --seed "${SEED}" \
  "${LEFT_ARGS[@]}"
