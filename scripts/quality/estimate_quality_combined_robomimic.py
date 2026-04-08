"""
Run a single combined quality-estimation pass over square_mh plus extra robomimic RLDS datasets.

This is the cleanest way to compare custom robomimic demos against the original
RoboMimic square dataset under one shared scoring run and one shared normalization
distribution.

Expected workflow:
1. Build each custom robomimic image.hdf5 into an RLDS/TFDS directory with
   rlds/robomimic/robomimic_dataset_builder.py.
2. Run this script with the original square_mh dataset plus the new RLDS roots.

Example:

python scripts/quality/estimate_quality_combined_robomimic.py \
    --obs_ckpt /iris/u/jasonyan/data/deminf_outputs/robomimic_image/square_mh_wrist_obs_vae_seed1 \
    --action_ckpt /iris/u/jasonyan/data/deminf_outputs/robomimic_image/square_mh_action_vae_seed1 \
    --batch_size 1024 \
    --output /iris/u/jasonyan/data/deminf_outputs/combined_square_fb \
    --extra-dataset forward_grab=/iris/u/jasonyan/data/fb_demos_rlds/forward_grab/1.0.0 \
    --extra-dataset backward_grab=/iris/u/jasonyan/data/fb_demos_rlds/backward_grab/1.0.0
"""

from __future__ import annotations

import json
import os
import pickle
import pprint
from dataclasses import dataclass

import jax
import numpy as np
import quality_estimators
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app, flags
from jax.experimental import compilation_cache, multihost_utils
from matplotlib import pyplot as plt
from ml_collections import ConfigDict
from scipy import stats

from openx.data.dataloader import make_dataloader
from openx.data.datasets.robomimic import robomimic_dataset_transform
from openx.utils.evaluate import load_checkpoint
from openx.utils.spec import ModuleSpec

FLAGS = flags.FLAGS
flags.DEFINE_string("output", None, "Directory to save result pickles and plots.", required=True)
flags.DEFINE_string("obs_ckpt", None, "Path to the obs logs and checkpoints.", required=True)
flags.DEFINE_string("action_ckpt", None, "Path to the action logs and checkpoints.", required=True)
flags.DEFINE_integer("batch_size", 1024, "Batch size for evaluation.", required=False)
flags.DEFINE_string(
    "square_dataset_name",
    "square_mh",
    "Dataset key inside the checkpoint config for the original RoboMimic square dataset.",
    required=False,
)
flags.DEFINE_string(
    "square_path_override",
    None,
    "Optional TFDS builder directory override for square_mh. Defaults to the checkpoint config path.",
    required=False,
)
flags.DEFINE_multi_string(
    "extra_dataset",
    None,
    "Extra dataset in the form name=/path/to/tfds_builder_dir. Repeat for multiple datasets.",
)


@dataclass
class ExtraDatasetSpec:
    name: str
    path: str


def parse_extra_dataset(spec: str) -> ExtraDatasetSpec:
    name, path = spec.split("=", 1)
    if not name or not path:
        raise ValueError(f"Invalid --extra-dataset spec: {spec}")
    return ExtraDatasetSpec(name=name, path=path)


def _checkpoint_root(path: str) -> str:
    return os.path.dirname(os.path.normpath(path)) if os.path.basename(os.path.normpath(path)).isdigit() else path


def _load_config(path: str) -> ConfigDict:
    with tf.io.gfile.GFile(os.path.join(_checkpoint_root(path), "config.json"), "r") as f:
        return ConfigDict(json.load(f))


def _build_dataset_configs(base_config: ConfigDict, extra_specs: list[ExtraDatasetSpec]) -> dict:
    datasets = base_config.dataloader.to_dict()["datasets"]
    if FLAGS.square_dataset_name not in datasets:
        raise KeyError(f"{FLAGS.square_dataset_name} not found in checkpoint dataloader config")

    square_cfg = dict(datasets[FLAGS.square_dataset_name])
    if FLAGS.square_path_override is not None:
        square_cfg["path"] = FLAGS.square_path_override
    if "val_split" in square_cfg:
        del square_cfg["val_split"]
    square_cfg["train_split"] = square_cfg.get("train_split", "train")

    combined = {FLAGS.square_dataset_name: square_cfg}
    for spec in extra_specs:
        combined[spec.name] = dict(
            path=spec.path,
            train_split="train",
            transform=ModuleSpec.create(robomimic_dataset_transform),
        )
    return combined


def _count_effective_steps(builder_dir: str, split: str = "train") -> int:
    builder = tfds.builder_from_directory(builder_dir=builder_dir)
    ds = builder.as_dataset(split=split, decoders=dict(steps=tfds.decode.SkipDecoding()), shuffle_files=False)
    total = 0
    for ep in tfds.as_numpy(ds):
        ep_len = int(ep["steps"]["is_first"].shape[0])
        total += max(ep_len - 1, 0)  # match dataloader behavior: terminal transition is removed
    return total


def main(_):
    compilation_cache.compilation_cache.set_cache_dir(os.path.expanduser("~/.jax_compilation_cache"))

    mesh = jax.sharding.Mesh(jax.devices(), axis_names="batch")
    dp_spec = jax.sharding.PartitionSpec("batch")
    dp_sharding = jax.sharding.NamedSharding(mesh, dp_spec)
    rep_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    def shard(batch):
        batch = jax.tree.map(lambda x: x._numpy(), batch)
        return multihost_utils.host_local_array_to_global_array(batch, mesh, dp_spec)

    tf.config.set_visible_devices([], "GPU")

    config = _load_config(FLAGS.obs_ckpt)
    extra_specs = [parse_extra_dataset(spec) for spec in FLAGS.extra_dataset or []]
    dataset_cfgs = _build_dataset_configs(config, extra_specs)

    for name, ds_cfg in dataset_cfgs.items():
        steps = _count_effective_steps(ds_cfg["path"], ds_cfg.get("train_split", "train"))
        ds_cfg["weight"] = float(steps)
        print(f"Using size-based weight for {name}: {steps} effective transitions")

    obs_alg, obs_state, _, _ = load_checkpoint(FLAGS.obs_ckpt)
    action_alg, action_state, _, _ = load_checkpoint(FLAGS.action_ckpt)

    pred_fn = jax.jit(
        lambda batch, rng: (
            quality_estimators.ksg_estimator(
                batch,
                rng,
                ks=np.arange(5, 8),
                obs_alg=obs_alg,
                obs_state=obs_state,
                action_alg=action_alg,
                action_state=action_state,
            ),
            {k: batch[k] for k in quality_estimators.AGGREGATION_KEYS},
        ),
        in_shardings=(dp_sharding, None),
        out_shardings=(rep_sharding, rep_sharding),
    )

    dataloader_config = config.dataloader.to_dict()
    dataloader_config["datasets"] = dataset_cfgs
    dataloader_config["batch_size"] = FLAGS.batch_size
    dataloader_config["repeat"] = 4
    dataloader_config["discard_fraction"] = 0.5
    dataloader_config["repeat_early"] = True
    dataloader_config["recompute_statistics"] = False
    if dataloader_config.get("goal_conditioning", None) is not None:
        dataloader_config["goal_conditioning"] = "last"

    structure = config.structure.to_dict()
    for key in quality_estimators.AGGREGATION_KEYS:
        if key in ("dataset_id", "step_idx"):
            continue
        structure[key] = None

    ds, _, _, dataset_ids = make_dataloader(**dataloader_config, structure=structure, split_for_jax=True)
    ds = map(shard, ds)
    rng = jax.random.key(jax.process_index())
    scores = quality_estimators.estimate_quality(ds, pred_fn, dataset_ids, rng)

    pprint.pprint(scores)

    if jax.process_index() != 0:
        return

    tf.io.gfile.makedirs(FLAGS.output)
    for ds_name, ds_scores in scores.items():
        with tf.io.gfile.GFile(os.path.join(FLAGS.output, ds_name + ".pkl"), "wb") as f:
            pickle.dump(ds_scores, f)

        if "quality_by_ep_idx" not in ds_scores:
            continue

        idxs = list(ds_scores["ep_idx"].keys())
        x = np.array([ds_scores["quality_by_ep_idx"][idx] for idx in idxs])
        y = np.array([ds_scores["ep_idx"][idx] for idx in idxs])

        title = "r=n/a"
        if len(np.unique(x)) > 1 and len(y) > 1:
            r = stats.pearsonr(x, y)
            print(ds_name, "R:", r)
            title = f"r={r[0]:.2f}"

        _, ax = plt.subplots(1, 1, figsize=(3, 3))
        for v in np.sort(np.unique(x)):
            ax.hist(y[x == v], bins=20, alpha=0.5, label=f"Quality {v}")
        ax.legend()
        ax.set_title(title)
        plt.tight_layout()
        with tf.io.gfile.GFile(os.path.join(FLAGS.output, ds_name + "_hist.png"), "wb") as f:
            plt.savefig(f)
        plt.close()

        plt.figure(figsize=(3, 3))
        sort_idx = np.argsort(y)
        rev_sorted_quality_labels = x[sort_idx][::-1]
        total_quality_labels = np.cumsum(rev_sorted_quality_labels)
        num_data_points = 1 + np.arange(total_quality_labels.shape[0])
        avg_quality_label = total_quality_labels / num_data_points
        plt.plot(np.arange(avg_quality_label.shape[0]), avg_quality_label[::-1], label="method")
        oracle_labels = np.cumsum(np.sort(x)[::-1]) / num_data_points
        plt.plot(np.arange(oracle_labels.shape[0]), oracle_labels[::-1], color="gray", label="oracle")
        plt.gca().hlines(np.mean(x), xmin=0, xmax=oracle_labels.shape[0], color="red", linestyles="dashed")
        plt.xlabel("Episodes Removed")
        plt.ylabel("Average Quality Label")
        plt.legend(frameon=False)
        if np.min(x) != np.max(x):
            plt.ylim(np.min(x), np.max(x))
        plt.title(title)
        plt.tight_layout()
        with tf.io.gfile.GFile(os.path.join(FLAGS.output, ds_name + "_curve.png"), "wb") as f:
            plt.savefig(f)
        plt.close()


if __name__ == "__main__":
    app.run(main)
