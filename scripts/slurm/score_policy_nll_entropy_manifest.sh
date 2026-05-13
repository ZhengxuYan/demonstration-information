#!/bin/bash
# Score one row from a policy NLL+entropy manifest.
#
# Usage:
#   sbatch --array=1-N scripts/slurm/score_policy_nll_entropy_manifest.sh /path/to/score_manifest.csv

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=policy_ne
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris-hgx-1,iris-hgx-2,iris-hp-z8
#SBATCH --output=/iris/u/jasonyan/slurm/%A_%a_policy_ne.out
#SBATCH --error=/iris/u/jasonyan/slurm/%A_%a_policy_ne.err

set -euo pipefail

MANIFEST="${1:?Usage: sbatch --array=1-N $0 /path/to/score_manifest.csv}"
ROW_INDEX="${SLURM_ARRAY_TASK_ID:?Submit with --array=1-N; row 1 is first data row}"

set +u
source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate openx
set -u

REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"
mkdir -p /iris/u/jasonyan/slurm
cd "${REPO}"

python scripts/setup/patch_robomimic_optional_diffusion.py
python scripts/setup/patch_robomimic_discrete_action.py

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID="${EGL_DEVICE_ID:-0}"
export PYTHONPATH="${REPO}/robomimic:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

eval "$(
  python - "${MANIFEST}" "${ROW_INDEX}" <<'PY'
import csv
import shlex
import sys

manifest, row_index = sys.argv[1], int(sys.argv[2])
with open(manifest, newline="") as f:
    rows = list(csv.DictReader(f))
if row_index < 1 or row_index > len(rows):
    raise SystemExit(f"row index {row_index} outside 1..{len(rows)}")
row = rows[row_index - 1]
for key in ("dataset", "policy", "view", "baseline", "checkpoint_label", "checkpoint", "dataset_path", "output", "name"):
    print(f"{key.upper()}={shlex.quote(row[key])}")
PY
)"

mkdir -p "${OUTPUT}"
echo "dataset=${DATASET} policy=${POLICY} view=${VIEW} baseline=${BASELINE} checkpoint_label=${CHECKPOINT_LABEL}"
echo "checkpoint=${CHECKPOINT}"
echo "dataset_path=${DATASET_PATH}"
echo "output=${OUTPUT}"

python scripts/quality/score_robomimic_policy_nll.py \
  --checkpoint "${CHECKPOINT}" \
  --dataset "${DATASET_PATH}" \
  --output "${OUTPUT}" \
  --name "${NAME}" \
  --batch-size "${BATCH_SIZE:-128}" \
  --num-workers "${NUM_WORKERS:-0}" \
  --gmm-entropy-samples "${GMM_ENTROPY_SAMPLES:-128}" \
  --entropy-seed "${ENTROPY_SEED:-0}"
