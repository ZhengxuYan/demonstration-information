#!/usr/bin/env python3
"""Smoke-test the DROID pen-in-cup RLDS dataset and OpenX dataloader."""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds

from configs.bc.droid_pen_in_cup_dp_random_drop import DEFAULT_SCORE_ROOT, get_config
from openx.data.dataloader import make_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rlds-path", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, default=Path(DEFAULT_SCORE_ROOT))
    parser.add_argument("--drop-percent", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    builder = tfds.builder_from_directory(builder_dir=str(args.rlds_path))
    print(f"builder={builder.name} version={builder.version}")
    print(f"splits={builder.info.splits}")

    raw = builder.as_dataset(split="train", decoders={"steps": tfds.decode.SkipDecoding()}, shuffle_files=False)
    raw_count = 0
    first_ep_idx = None
    for episode in tfds.as_numpy(raw):
        raw_count += 1
        if first_ep_idx is None:
            first_ep_idx = int(episode["episode_metadata"]["ep_idx"])
    print(f"raw_episode_count={raw_count} first_ep_idx={first_ep_idx}")

    cfg = get_config(f"pen_in_cup,{args.drop_percent},random,{args.seed},{args.score_root},{args.rlds_path}")
    dataloader_cfg = cfg.dataloader.to_dict()
    dataloader_cfg["batch_size"] = 2
    dataloader_cfg["shuffle_size"] = 10
    dataloader_cfg["cache"] = False
    dataloader_cfg["prefetch"] = 0
    train_dataset, _, _, _ = make_dataloader(
        **dataloader_cfg,
        structure=cfg.structure.to_dict(),
        split_for_jax=False,
    )
    batch = next(iter(train_dataset.take(1)))
    print("batch keys:", batch.keys())
    print("agent_1", batch["observation"]["image"]["agent_1"].shape, batch["observation"]["image"]["agent_1"].dtype)
    print("wrist", batch["observation"]["image"]["wrist"].shape, batch["observation"]["image"]["wrist"].dtype)
    print("state leaves", [x.shape for x in tf.nest.flatten(batch["observation"]["state"])])
    print("action", batch["action"].shape, batch["action"].dtype)
    if int(batch["action"].shape[-1]) != 10:
        raise SystemExit(f"Expected action dim 10, got {batch['action'].shape}")
    print("RLDS_VALIDATION_OK")


if __name__ == "__main__":
    main()
