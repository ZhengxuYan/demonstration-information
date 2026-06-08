# DROID Pen-in-Cup RLDS Builder

This builder converts raw DROID collection folders into the RLDS / TFDS schema
expected by `openx.data.datasets.oxe.droid_dataset_transform`.

It is based on the public `kpertsch/droid_dataset_builder` example, with two
task-specific adjustments:

- The source data path is read from `DROID_PEN_IN_CUP_RAW_ROOT`.
- If a trajectory only has one exterior camera, `exterior_image_2_left` is
  filled by duplicating `exterior_image_1_left` so the downstream DROID schema
  remains unchanged.

Build:

```bash
cd rlds/droid_pen_in_cup
DROID_PEN_IN_CUP_RAW_ROOT=/scr/rbhowmik/collected-data/pen-in-cup/06-07-2026-total-102 \
tfds build --overwrite --data_dir /iris/u/jasonyan/data/droid_pen_in_cup_06072026_rlds
```

The trainable builder directory will be:

```text
/iris/u/jasonyan/data/droid_pen_in_cup_06072026_rlds/droid_pen_in_cup/1.0.0
```
