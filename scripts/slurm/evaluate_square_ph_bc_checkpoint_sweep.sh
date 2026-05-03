#!/bin/bash
#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=ph_bc_rollout
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris9,iris-hgx-1,iris-hgx-2,iris-hp-z8
#SBATCH --output=/iris/u/jasonyan/slurm/%j_ph_bc_rollout.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_ph_bc_rollout.err

set -euo pipefail

ALGO="${1:?usage: $0 gmm|discrete agent_wrist|left_close_low_wrist}"
VIEW="${2:?usage: $0 gmm|discrete agent_wrist|left_close_low_wrist}"

if [[ "${ALGO}" != "gmm" && "${ALGO}" != "discrete" ]]; then
  echo "bad algo: ${ALGO}" >&2
  exit 2
fi
if [[ "${VIEW}" != "agent_wrist" && "${VIEW}" != "left_close_low_wrist" ]]; then
  echo "bad view: ${VIEW}" >&2
  exit 2
fi

source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
CONDA_ENV="${CONDA_ENV:-openx}"
conda activate "${CONDA_ENV}"

REPO=/iris/u/jasonyan/repos/demonstration-information
CKPT_ROOT="${CKPT_ROOT:-/iris/u/jasonyan/data/robomimic_outputs/policy_view_experiments}"
OUT_ROOT="${OUT_ROOT:-/iris/u/jasonyan/data/robomimic_rollout_scores/square_ph_bc_checkpoint_sweep}"
RUN_NAME="square_ph_bc_${ALGO}_${VIEW}_200_seed1"
RUN_ROOT="${CKPT_ROOT}/${RUN_NAME}"
N_ROLLOUTS="${N_ROLLOUTS:-50}"
HORIZON="${HORIZON:-400}"
SEED="${SEED:-0}"

mkdir -p /iris/u/jasonyan/slurm "${OUT_ROOT}/${RUN_NAME}"
cd "${REPO}"

python "${REPO}/scripts/setup/patch_robomimic_optional_diffusion.py"
python "${REPO}/scripts/setup/patch_robomimic_discrete_action.py"

export MUJOCO_GL=egl
export PYTHONPATH="${REPO}/robomimic:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

python - <<'PY'
import robosuite
print("robosuite", getattr(robosuite, "__version__", "unknown"))
PY

mapfile -t ALL_CKPTS < <(find "${RUN_ROOT}" -path "*/models/model_epoch_*.pth" -type f | sort -V)
if [[ "${#ALL_CKPTS[@]}" -eq 0 ]]; then
  echo "missing checkpoints under ${RUN_ROOT}" >&2
  exit 1
fi

CKPTS=()
if [[ -n "${EPOCHS:-}" ]]; then
  for epoch in ${EPOCHS}; do
    match=""
    for ckpt in "${ALL_CKPTS[@]}"; do
      if [[ "$(basename "${ckpt}")" == "model_epoch_${epoch}.pth" ]]; then
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
  CKPTS=("${ALL_CKPTS[@]}")
fi

LEFT_ARGS=()
if [[ "${VIEW}" == "left_close_low_wrist" ]]; then
  LEFT_ARGS=(--left-close-low)
fi

printf 'run %s\n' "${RUN_NAME}"
printf 'conda_env %s\n' "${CONDA_ENV}"
printf 'n_rollouts %s horizon %s seed %s\n' "${N_ROLLOUTS}" "${HORIZON}" "${SEED}"
printf 'checkpoints %s\n' "${#CKPTS[@]}"
printf '%s\n' "${CKPTS[@]}"

python scripts/quality/evaluate_square_ph_bc_checkpoint_rollouts.py \
  --run-name "${RUN_NAME}" \
  --checkpoints "${CKPTS[@]}" \
  --output-dir "${OUT_ROOT}/${RUN_NAME}" \
  --n-rollouts "${N_ROLLOUTS}" \
  --horizon "${HORIZON}" \
  --seed "${SEED}" \
  "${LEFT_ARGS[@]}"
