#!/bin/bash
set -euo pipefail

for algo in gmm discrete; do
  for view in agent_wrist left_close_low_wrist; do
    sbatch --export=ALL,xml_catalog_files_libxml2="" \
      scripts/slurm/evaluate_square_ph_bc_checkpoint_sweep.sh "${algo}" "${view}"
  done
done
