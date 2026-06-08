#!/bin/bash
# Build RLDS/TFDS and random episode-drop score files for DROID pen-in-cup.
#
# Usage:
#   sbatch scripts/slurm/prepare_droid_pen_in_cup_rlds.sh

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --nodelist=iris10
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --job-name=prep_pic_rlds
#SBATCH --output=/iris/u/jasonyan/slurm/%j_prep_pic_rlds.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_prep_pic_rlds.err

set -euo pipefail

set +u
source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-openx}"
set -u

REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"
RAW_ROOT="${RAW_ROOT:-/scr/rbhowmik/collected-data/pen-in-cup/06-07-2026-total-102}"
RLDS_ROOT="${RLDS_ROOT:-/iris/u/jasonyan/data/droid_pen_in_cup_06072026_rlds}"
RLDS_PATH="${RLDS_PATH:-${RLDS_ROOT}/droid_pen_in_cup/1.0.0}"
SCORE_ROOT="${SCORE_ROOT:-/iris/u/jasonyan/data/droid_pen_in_cup_06072026_random_drop_scores}"
DROP_PERCENTS="${DROP_PERCENTS:-25 50 75}"
SEED="${SEED:-1}"
EXPECTED_EPISODES="${EXPECTED_EPISODES:-102}"

mkdir -p /iris/u/jasonyan/slurm "${RLDS_ROOT}" "${SCORE_ROOT}"
cd "${REPO}"

echo "hostname=$(hostname)"
echo "raw_root=${RAW_ROOT}"
echo "rlds_root=${RLDS_ROOT}"
echo "rlds_path=${RLDS_PATH}"
echo "score_root=${SCORE_ROOT}"
echo "drop_percents=${DROP_PERCENTS}"
echo "seed=${SEED}"

python - <<'PY'
import importlib
missing = [name for name in ("cv2", "h5py", "tensorflow", "tensorflow_datasets") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Missing conversion dependencies: {missing}")
PY

python scripts/data/validate_droid_pen_in_cup_raw.py \
  --raw-root "${RAW_ROOT}" \
  --expected-episodes "${EXPECTED_EPISODES}"

export DROID_PEN_IN_CUP_RAW_ROOT="${RAW_ROOT}"
cd "${REPO}/rlds/droid_pen_in_cup"
tfds build --overwrite --data_dir "${RLDS_ROOT}"

cd "${REPO}"
python scripts/data/write_random_episode_drop_scores.py \
  --num-episodes "${EXPECTED_EPISODES}" \
  --drop-percents ${DROP_PERCENTS} \
  --seed "${SEED}" \
  --env pen_in_cup \
  --output-root "${SCORE_ROOT}"

python scripts/data/validate_droid_pen_in_cup_rlds.py \
  --rlds-path "${RLDS_PATH}" \
  --score-root "${SCORE_ROOT}" \
  --drop-percent 25 \
  --seed "${SEED}"

echo "PREP_DROID_PEN_IN_CUP_RLDS_OK"
echo "${RLDS_PATH}"
