# Threading D0 density models

This directory records the density-model configurations used for the promising
Threading D0 score filters. The evidence comes from the normal seed-1 density
sweep and the 54-run filtered-BC validation sweep (six filters, three retained
fractions, and three BC seeds per cell).

## Recommendation

Use an 8D absolute joint-action density with a **Gaussian action head**, a
**linear (unsquashed) mean**, and **diagonal covariance**.

1. **Primary:** `identity_linear` with Score 6 (model MI against a learned
   action prior). It produced the best 50%-retained filtered-BC result:
   `81.3%` mean success over three seeds, with seed range `81%–82%`.
2. **Aggressive filtering:** `zscore_linear` with Score 2 (data MI against a
   learned marginal). It was the best option with only 25% of demonstrations
   retained: `53.3%` mean, seed range `49%–59%`.
3. **Secondary robustness check:** `robust_scale_linear` with Gaussian Score 2
   or Score 6. It is competitive but did not beat the two recommendations
   above at their strongest retained fractions.
4. **GMM:** keep `robust_scale_linear` + 5-mode GMM as an ablation, not the
   default. The matched Gaussian filters were stronger in downstream BC.

Select the **epoch-2000 final checkpoint**, not the validation-best checkpoint.
For the six selected Threading scores, final-checkpoint normalized oracle gaps
were substantially lower than validation-best gaps. For example:

| Density / score | Final gap | Validation-best gap |
| --- | ---: | ---: |
| Gaussian identity, Score 6 | 0.166 | 1.038 |
| Gaussian z-score, Score 2 | 0.275 | 0.827 |
| Gaussian z-score, Score 6 | 0.161 | 1.133 |
| Gaussian robust, Score 2 | 0.305 | 0.529 |
| Gaussian robust, Score 6 | 0.192 | 0.889 |
| GMM robust, Score 6 | 0.538 | 0.850 |

Gap convention: `0` is the label oracle, `1` is random ranking, and lower is
better. Validation-best is still saved for auditing; it is not the recommended
Threading scoring checkpoint.

## Shared hyperparameters

| Parameter | Value |
| --- | --- |
| Action representation | 8D absolute joint action |
| Conditional inputs | agent-view RGB + wrist RGB + proprioception |
| Prior inputs | action-prior dummy observation only |
| Action head | Gaussian recommended; 5-mode GMM ablation |
| Mean | linear (`mean_squash=none`) |
| Covariance | diagonal |
| Minimum std, z-score / robust | `0.01` in transformed coordinates |
| Minimum std, identity | per dimension: `0.01 × train-split raw std` |
| Transform scale floor | `1e-6` |
| Hidden layers | `[1024, 1024]` |
| Optimizer / learning rate | Adam / `1e-4` |
| L2 regularization | `0` |
| Batch size | `128` |
| Train / validation steps per epoch | `100 / 25` |
| Epochs | `2000` |
| RGB crop randomizer | disabled |
| Checkpoint / best synchronization | every 200 epochs |

Normalization statistics are fit on the **train split only**. The validation
split reuses those exact statistics. Identity-linear's raw-coordinate minimum
std vector is computed after the train statistics are available; the exact
vectors are preserved in the config snapshots.

## Exact configs

`actual_runs/` contains the eight JSON snapshots used by the selected scores:

- Gaussian × identity, z-score, and robust scaling
- GMM × robust scaling
- conditional and action-prior models for each density family

The conditional snapshots point to the lossless contiguous Threading HDF5 used
by the corrected fast-I/O runs. The prior snapshots preserve the original data
path; priors read actions only. For a new run, using the contiguous HDF5 for
both models is safe and avoids layout-dependent I/O.

## Training

Clone with the robomimic submodule and apply the pinned density extension:

```bash
git submodule update --init robomimic
git -C robomimic apply --check ../patches/robomimic_density_normalization.patch
git -C robomimic apply ../patches/robomimic_density_normalization.patch
```

The patch adds train-only action transforms, validation-stat reuse, linear
Gaussian/GMM means, vector minimum std, and diagonal/full covariance support.

On the Stanford cluster, launch one recipe as a two-element array: conditional
and action prior. The following example trains the primary Gaussian identity
recipe; change `ACTION_TRANSFORM` and `VARIANT_TAG` together for `zscore_linear`
or `robust_scale_linear`.

```bash
REPO=/iris/u/$USER/repos/demonstration-information
DATA=/iris/u/$USER/data
HDF5=$DATA/threading_d0_final200_abs_delta_20260730/hdf5/image_final200_joint_absolute_fixedobs_contiguous.hdf5
OUT=$DATA/threading_d0_density_promising/models
CFG=$DATA/threading_d0_density_promising/configs

sbatch --partition=iris-hi --account=iris --array=1-2%2 \
  --export=ALL,REPO=$REPO,DATASET_TAG=d0_final200_joint_absolute,DATASET_HDF5=$HDF5,\
OUT_ROOT=$OUT,CONFIG_ROOT=$CFG,TASK_TAG=density_recipe,RUN_PREFIX=threading_d0_density,\
ALGOS=gaussian,CONDITIONS=image_proprio:action_prior,ACTION_SOURCE=image_proprio,\
ACTION_TARGET=single,ACTION_NORMALIZATION=none,ACTION_TRANSFORM=identity,\
MEAN_SQUASH=none,COVARIANCE_TYPE=diag,VARIANT_TAG=identity_linear,\
TRAIN_FILTER_KEY=train,VALID_FILTER_KEY=valid,GAUSSIAN_MIN_STD=0.01,\
NUM_EPOCHS=2000,BATCH_SIZE=128,EPOCH_STEPS=100,VALIDATION_STEPS=25,\
SAVE_EVERY_N_EPOCHS=200,ROBOMIMIC_LATEST_SAVE_INTERVAL=200,\
SAVE_BEST_VALIDATION=1,LOCAL_BEST_SYNC_INTERVAL=200,DISABLE_RGB_RANDOMIZER=1,\
STAGE_DATASET_TO_TMP=1,RESUME=1,TRAIN_SEED=1 \
  $REPO/scripts/slurm/train_pen_in_cup_density_models_array.sh
```

For the GMM ablation, set `ALGOS=gmm`, `ACTION_TRANSFORM=robust_scale`, and
`VARIANT_TAG=robust_scale_linear`. The worker creates the robomimic configs,
stages the HDF5 once per task to `/tmp`, keeps the rolling validation-best copy
local, and writes shared checkpoints only at the configured sparse interval.

For preemptible partitions, retain `RESUME=1` and add Slurm `--requeue`.

## Scoring checkpoint

Use the latest epoch checkpoint when producing Threading scores:

```bash
CKPT_MODE=latest_epoch SCORE_MODE_TAG=early_stop_final \
  sbatch scripts/slurm/score_threading_pomdp_6.sh
```

Score 2 is `data_mi_learned_marginal`; Score 6 is
`model_mi_reference_prior`. The paired conditional and action-prior models must
use the same action transform and density head.

## Reproducibility notes

- These conclusions are descriptive: the density sweep used normal seed 1,
  while downstream filtered BC used three policy seeds.
- Do not select a checkpoint using human labels. The final checkpoint was fixed
  before the three-seed filtered-BC comparison.
- Keep the validation-best checkpoint for diagnostics even though it is not the
  recommended Threading scoring checkpoint.
