#!/usr/bin/env python3
"""Prepare canonical image+proprio density HDF5s for Square MH or Threading."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np


OBS_KEYS = (
    "agentview_image",
    "robot0_eye_in_hand_image",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)
LABEL_VALUE = {"partial": 0.0, "full": 1.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-type", choices=("square", "threading"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--robosuite-root", type=Path, default=Path("/iris/u/jasonyan/repos/robosuite-pomdp"))
    parser.add_argument("--env-name", default="Threading")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def numeric_key(value: str) -> tuple[int, str]:
    tail = value.rsplit("_", 1)[-1]
    return (int(tail), value) if tail.isdigit() else (sys.maxsize, value)


def stratified_split(
    indices_by_label: dict[str, list[int]], valid_ratio: float, rng: np.random.Generator
) -> tuple[list[int], list[int]]:
    train: list[int] = []
    valid: list[int] = []
    for label, indices in sorted(indices_by_label.items()):
        values = np.asarray(indices, dtype=np.int64)
        rng.shuffle(values)
        n_valid = min(len(values) - 1, max(1, int(round(valid_ratio * len(values)))))
        valid.extend(values[:n_valid].tolist())
        train.extend(values[n_valid:].tolist())
    return sorted(train), sorted(valid)


def build_splits(labels: list[str], valid_ratio: float, seed: int) -> dict[str, list[int]]:
    by_label: dict[str, list[int]] = {}
    for idx, label in enumerate(labels):
        by_label.setdefault(label, []).append(idx)
    rng = np.random.default_rng(seed)
    train, valid = stratified_split(by_label, valid_ratio, rng)
    folds = [[], []]
    fold_rng = np.random.default_rng(seed + 1)
    for indices in by_label.values():
        values = np.asarray(indices, dtype=np.int64)
        fold_rng.shuffle(values)
        for fold_idx, part in enumerate(np.array_split(values, 2)):
            folds[fold_idx].extend(part.tolist())
    result = {"train": train, "valid": valid, "score_all": list(range(len(labels))), "all": list(range(len(labels)))}
    for fold_idx, score in enumerate(folds):
        score_set = set(score)
        pool = {label: [idx for idx in indices if idx not in score_set] for label, indices in by_label.items()}
        fold_train, fold_valid = stratified_split(pool, valid_ratio, np.random.default_rng(seed + 10 + fold_idx))
        result[f"fold{fold_idx}_train"] = fold_train
        result[f"fold{fold_idx}_valid"] = fold_valid
        result[f"fold{fold_idx}_score"] = sorted(score)
    return result


def write_masks(root: h5py.File, demo_keys: list[str], splits: dict[str, list[int]]) -> None:
    mask = root.create_group("mask")
    for name, indices in splits.items():
        mask.create_dataset(name, data=np.asarray([demo_keys[idx].encode() for idx in indices], dtype="S"))


def write_labels(root: h5py.File, demo_keys: list[str], labels: list[str]) -> None:
    group = root.create_group("labels")
    group.create_dataset("demo_key", data=np.asarray([key.encode() for key in demo_keys], dtype="S"))
    group.create_dataset("label", data=np.asarray([label.encode() for label in labels], dtype="S"))
    group.create_dataset(
        "label_value",
        data=np.asarray([LABEL_VALUE.get(label, np.nan) for label in labels], dtype=np.float32),
    )
    group.attrs["mapping_json"] = json.dumps(LABEL_VALUE, sort_keys=True)


def read_square_labels(path: Path, count: int) -> list[str]:
    with path.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["dataset"] == "square_mh"]
    labels = {int(row["ep_idx"]): row["label"].strip().lower() for row in rows}
    if set(labels) != set(range(count)):
        raise ValueError(f"Square labels do not cover 0..{count - 1}")
    return [labels[idx] for idx in range(count)]


def prepare_square(args: argparse.Namespace, output: Path) -> None:
    if args.labels_csv is None:
        raise ValueError("--labels-csv is required for Square MH")
    with h5py.File(args.input, "r") as src, h5py.File(output, "w") as dst:
        source_keys = sorted(src["data"], key=numeric_key)
        labels = read_square_labels(args.labels_csv, len(source_keys))
        splits = build_splits(labels, args.valid_ratio, args.seed)
        env_args = src.attrs.get("env_args", src["data"].attrs.get("env_args", "{}"))
        dst.attrs["env_args"] = env_args
        dst.attrs["source_hdf5"] = str(args.input)
        dst.attrs["action_dim"] = 7
        dst.attrs["observation_keys_json"] = json.dumps(OBS_KEYS)
        data = dst.create_group("data")
        data.attrs["env_args"] = env_args
        total = 0
        demo_keys = []
        for idx, source_key in enumerate(source_keys):
            source = src[f"data/{source_key}"]
            actions = source["actions"]
            if actions.ndim != 2 or actions.shape[1] != 7:
                raise ValueError(f"{source_key} actions must be (T,7), got {actions.shape}")
            demo_key = f"demo_{idx}"
            demo_keys.append(demo_key)
            demo = data.create_group(demo_key)
            obs = demo.create_group("obs")
            for key in OBS_KEYS:
                if key not in source["obs"]:
                    raise KeyError(f"{source_key}/obs is missing {key}")
                source["obs"].copy(key, obs)
            obs.create_dataset("action_prior_dummy", data=np.zeros((len(actions), 1), dtype=np.float32))
            source.copy("actions", demo)
            if "states" in source:
                source.copy("states", demo)
            for key, value in source.attrs.items():
                demo.attrs[key] = value
            demo.attrs["ep_idx"] = idx
            demo.attrs["source_demo"] = source_key
            demo.attrs["label"] = labels[idx]
            demo.attrs["label_value"] = LABEL_VALUE.get(labels[idx], np.nan)
            demo.attrs["num_samples"] = len(actions)
            total += len(actions)
        data.attrs["num_demos"] = len(demo_keys)
        data.attrs["total"] = total
        dst.attrs["total"] = total
        write_masks(dst, demo_keys, splits)
        write_labels(dst, demo_keys, labels)


def read_threading_rows(root: Path) -> list[dict[str, str]]:
    with (root / "annotations.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {root / 'annotations.csv'}")
    return rows


def make_env(args: argparse.Namespace):
    if str(args.robosuite_root) not in sys.path:
        sys.path.insert(0, str(args.robosuite_root))
    import robosuite as suite

    return suite.make(
        args.env_name,
        robots=["Panda"],
        ignore_done=True,
        use_camera_obs=True,
        use_object_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=84,
        camera_widths=84,
        horizon=1000,
    )


def replay_observation(env, state: np.ndarray) -> dict[str, np.ndarray]:
    env.sim.set_state_from_flattened(state)
    env.sim.forward()
    try:
        raw = env._get_observations(force_update=True)
    except TypeError:
        raw = env._get_observations()
    result = {}
    for key in OBS_KEYS:
        value = np.asarray(raw[key])
        if key.endswith("_image"):
            value = value[::-1].copy()
        result[key] = value
    return result


def prepare_threading(args: argparse.Namespace, output: Path) -> None:
    rows = read_threading_rows(args.input)
    labels = [row["manual_label"].strip().lower() for row in rows]
    if any(label not in LABEL_VALUE for label in labels):
        raise ValueError(f"Threading labels must be full/partial, got {sorted(set(labels))}")
    splits = build_splits(labels, args.valid_ratio, args.seed)
    env = make_env(args)
    env_args = json.dumps({"env_name": args.env_name, "type": 1, "env_kwargs": {"robots": ["Panda"]}})
    with h5py.File(output, "w") as dst:
        dst.attrs["env_args"] = env_args
        dst.attrs["source_dataset_root"] = str(args.input)
        dst.attrs["action_dim"] = 8
        dst.attrs["observation_keys_json"] = json.dumps(OBS_KEYS)
        data = dst.create_group("data")
        data.attrs["env_args"] = env_args
        demo_keys = []
        total = 0
        for idx, row in enumerate(rows):
            npz_path = args.input / row["npz"]
            with np.load(npz_path, allow_pickle=True) as payload:
                states = np.asarray(payload["states"], dtype=np.float64)
                actions = np.asarray([info["actions"] for info in payload["action_infos"]], dtype=np.float32)
            if states.shape[0] != actions.shape[0] + 1 or actions.ndim != 2 or actions.shape[1] != 8:
                raise ValueError(f"Invalid state/action shapes in {npz_path}: {states.shape}, {actions.shape}")
            model_xml = (args.input / row["xml"]).read_text()
            env.reset()
            if hasattr(env, "reset_from_xml_string"):
                env.reset_from_xml_string(model_xml)
            obs_rows = [replay_observation(env, state) for state in states[:-1]]
            demo_key = f"demo_{idx}"
            demo_keys.append(demo_key)
            demo = data.create_group(demo_key)
            obs = demo.create_group("obs")
            for key in OBS_KEYS:
                values = np.asarray([item[key] for item in obs_rows])
                kwargs = {"compression": "gzip"} if values.dtype == np.uint8 else {}
                obs.create_dataset(key, data=values, **kwargs)
            obs.create_dataset("action_prior_dummy", data=np.zeros((len(actions), 1), dtype=np.float32))
            demo.create_dataset("actions", data=actions)
            demo.create_dataset("states", data=states[:-1])
            demo.attrs["num_samples"] = len(actions)
            demo.attrs["ep_idx"] = idx
            demo.attrs["episode"] = row["episode"]
            demo.attrs["source_demo"] = row["source_demo"]
            demo.attrs["label"] = labels[idx]
            demo.attrs["label_value"] = LABEL_VALUE[labels[idx]]
            demo.attrs["model_file"] = model_xml
            demo.attrs["source_manifest_row_json"] = json.dumps(row, sort_keys=True)
            total += len(actions)
            print(f"rendered {idx + 1}/{len(rows)} {row['episode']} steps={len(actions)}", flush=True)
        data.attrs["num_demos"] = len(demo_keys)
        data.attrs["total"] = total
        dst.attrs["total"] = total
        write_masks(dst, demo_keys, splits)
        write_labels(dst, demo_keys, labels)


def validate(path: Path) -> None:
    with h5py.File(path, "r") as handle:
        demos = sorted(handle["data"], key=numeric_key)
        if not demos:
            raise ValueError("Output contains no demos")
        for demo_key in demos:
            demo = handle[f"data/{demo_key}"]
            length = len(demo["actions"])
            expected_action_dim = int(handle.attrs["action_dim"])
            if demo["actions"].shape[1] != expected_action_dim:
                raise ValueError(
                    f"{demo_key} action dim {demo['actions'].shape[1]} != {expected_action_dim}"
                )
            for key in (*OBS_KEYS, "action_prior_dummy"):
                if key not in demo["obs"] or len(demo[f"obs/{key}"]) != length:
                    raise ValueError(f"{demo_key}/obs/{key} is missing or misaligned")
        required = {"train", "valid", "score_all", "fold0_train", "fold0_valid", "fold0_score", "fold1_train", "fold1_valid", "fold1_score"}
        if not required.issubset(handle["mask"]):
            raise ValueError(f"Missing masks: {sorted(required - set(handle['mask']))}")


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".tmp.hdf5")
    if tmp.exists():
        tmp.unlink()
    if args.source_type == "square":
        prepare_square(args, tmp)
    else:
        prepare_threading(args, tmp)
    validate(tmp)
    if args.output.exists():
        args.output.unlink()
    tmp.replace(args.output)
    print(f"wrote and validated {args.output}")


if __name__ == "__main__":
    main()
