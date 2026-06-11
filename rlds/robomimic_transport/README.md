# RoboMimic Transport RLDS

Builds a TFDS/RLDS dataset from a bimanual RoboMimic transport `image.hdf5`.

```bash
mkdir -p /path/to/manual
ln -sfn /path/to/transport_mh_image_v15.hdf5 /path/to/manual/image.hdf5
cd rlds/robomimic_transport
tfds build --manual_dir /path/to/manual --data_dir /path/to/output
```

The builder maps `shouldercamera0_image` and `shouldercamera1_image` to the
OpenX image keys `agent` and `wrist`, concatenates both robot low-dimensional
states plus object state into one transport state vector, and preserves the
full 14D action.
