# Threading D0 filtered BC: 54 runs

This bundle reproduces the filtered-BC sweep used in the report.

## Matrix

Six density filters, three filtering levels, and three BC seeds: `6 × 3 × 3 = 54` runs.

| Density filter | Score |
| --- | --- |
| Gaussian, z-score + linear | Score 2: data MI, learned marginal |
| Gaussian, robust scale + linear | Score 2: data MI, learned marginal |
| Gaussian, z-score + linear | Score 6: model MI, reference prior |
| Gaussian, robust scale + linear | Score 6: model MI, reference prior |
| Gaussian, identity + linear | Score 6: model MI, reference prior |
| GMM, robust scale + linear | Score 6: model MI, reference prior |

All scores use the epoch-2000 final density checkpoint. `filter25/50/75` means drop the lowest 25/50/75% and retain the top 150/100/50 demonstrations. Seeds are `1, 2, 3`.

BC settings: GMM policy, 600 epochs, 100 rollout episodes, horizon 800, rollout every 50 epochs, no saved checkpoints. The reported value is the best rollout success rate for each seed.

## Files

- `configs/`: the exact 54 robomimic JSON configs
- `score_inputs/`: the four episode-level score files used to rank demonstrations
- `score_selections.json`: the exact 18 retained-demo sets
- `config_manifest.csv`: task IDs and run metadata

## Run

```bash
git checkout threading-density-configs
git submodule update --init robomimic

REPO=$PWD
SOURCE_ROOT=/iris/u/jasonyan/data/threading_d0_final200_abs_delta_20260730
RUN_ROOT=/iris/u/$USER/data/threading_d0_filtered_bc_54run
RESULT_ROOT=/iris/u/$USER/data/threading_d0_filtered_bc_54run_results
BUNDLE=$REPO/configs/filtered_bc/threading_d0/final_score_54run

python scripts/quality/prepare_threading_d0_final_score_filtered_bc.py \
  --source-root "$SOURCE_ROOT" \
  --score-root "$BUNDLE/score_inputs" \
  --output-root "$RUN_ROOT" \
  --result-root "$RESULT_ROOT"

REPO="$REPO" ROOT="$RUN_ROOT" RESULT_ROOT="$RESULT_ROOT" \
SOURCE_HDF5="$SOURCE_ROOT/hdf5/image_final200_joint_absolute_fixedobs_contiguous.hdf5" \
bash scripts/slurm/launch_threading_d0_final_score_filtered_bc.sh
```

The launcher submits 6 concurrent jobs to `iris-hi`, 8 to `iliad`, and 13 to `sc-loprio`. The worker stages the HDF5 to node-local `/tmp`; only `sc-loprio` writes sparse temporary resume checkpoints.
