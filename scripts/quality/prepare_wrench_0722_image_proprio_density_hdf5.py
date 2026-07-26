#!/usr/bin/env python3
"""Export raw Wrench 0722 DROID episodes for image+proprio density scoring."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import h5py
import numpy as np


IMAGE_HEIGHT = 180
IMAGE_WIDTH = 320
SEED_DEFAULT = 20260725


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_label_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "episode",
        "source_video",
        "image_file",
        "middle_frame",
        "total_frames",
        "score",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} must contain columns {sorted(required)}")
    episodes = [row["episode"] for row in rows]
    duplicates = [key for key, count in Counter(episodes).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate label episodes: {duplicates[:10]}")
    scores = sorted({int(row["score"]) for row in rows})
    if scores != [1, 2, 3]:
        raise ValueError(f"Expected ordinal scores [1, 2, 3], got {scores}")
    return rows


def stratified_train_valid(
    indices_by_label: dict[int, list[int]],
    valid_ratio: float,
    rng: np.random.Generator,
) -> tuple[list[int], list[int]]:
    train: list[int] = []
    valid: list[int] = []
    for label, indices in sorted(indices_by_label.items()):
        values = np.asarray(indices, dtype=np.int64)
        rng.shuffle(values)
        count = min(len(values) - 1, max(1, int(round(valid_ratio * len(values)))))
        valid.extend(values[:count].tolist())
        train.extend(values[count:].tolist())
    return sorted(train), sorted(valid)


def build_splits(
    scores: list[int], valid_ratio: float, seed: int
) -> dict[str, list[int]]:
    by_label: dict[int, list[int]] = {}
    for index, score in enumerate(scores):
        by_label.setdefault(score, []).append(index)

    train, valid = stratified_train_valid(
        by_label, valid_ratio, np.random.default_rng(seed)
    )
    fold_rng = np.random.default_rng(seed + 1)
    folds = [[], []]
    for label, indices in sorted(by_label.items()):
        values = np.asarray(indices, dtype=np.int64)
        fold_rng.shuffle(values)
        for fold_index, values_in_fold in enumerate(np.array_split(values, 2)):
            folds[fold_index].extend(values_in_fold.tolist())

    result = {
        "train": train,
        "valid": valid,
        "score_all": list(range(len(scores))),
        "all": list(range(len(scores))),
    }
    for fold_index, score_indices in enumerate(folds):
        score_set = set(score_indices)
        remaining = {
            label: [index for index in indices if index not in score_set]
            for label, indices in by_label.items()
        }
        fold_train, fold_valid = stratified_train_valid(
            remaining,
            valid_ratio,
            np.random.default_rng(seed + 10 + fold_index),
        )
        result[f"fold{fold_index}_train"] = fold_train
        result[f"fold{fold_index}_valid"] = fold_valid
        result[f"fold{fold_index}_score"] = sorted(score_indices)
    return result


def camera_serials(handle: h5py.File) -> tuple[str, str]:
    types = handle["observation/camera_type"]
    wrist = [serial for serial in types if int(types[serial][0]) == 0]
    exterior = [serial for serial in types if int(types[serial][0]) != 0]
    if len(wrist) != 1 or len(exterior) != 1:
        raise ValueError(
            f"Expected exactly one wrist and one exterior camera, got "
            f"wrist={wrist}, exterior={exterior}"
        )
    return wrist[0], exterior[0]


def video_frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if count <= 0:
        raise ValueError(f"Could not read frame count from {path}")
    return count


def read_resized_full_frames(
    path: Path, selected_indices: np.ndarray
) -> np.ndarray:
    if len(selected_indices) == 0:
        return np.empty((0, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    selected = set(int(index) for index in selected_indices)
    last = int(selected_indices[-1])
    frames: list[np.ndarray] = []
    capture = cv2.VideoCapture(str(path))
    try:
        for index in range(last + 1):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"{path} ended before direct frame index {index}")
            if index not in selected:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(
                rgb,
                (IMAGE_WIDTH, IMAGE_HEIGHT),
                interpolation=cv2.INTER_AREA,
            )
            frames.append(resized.astype(np.uint8, copy=False))
    finally:
        capture.release()
    if len(frames) != len(selected_indices):
        raise ValueError(
            f"Read {len(frames)} selected frames from {path}, expected "
            f"{len(selected_indices)}"
        )
    return np.stack(frames)


def copy_episode(
    destination: h5py.Group,
    episode_dir: Path,
    row: dict[str, str],
    ep_idx: int,
) -> dict[str, int]:
    trajectory = episode_dir / "trajectory.h5"
    with h5py.File(trajectory, "r") as source:
        length = int(source["action/cartesian_velocity"].shape[0])
        wrist_serial, exterior_serial = camera_serials(source)
        wrist_video = episode_dir / "recordings/MP4" / f"{wrist_serial}.mp4"
        exterior_video = (
            episode_dir / "recordings/MP4" / f"{exterior_serial}.mp4"
        )
        wrist_count = video_frame_count(wrist_video)
        exterior_count = video_frame_count(exterior_video)
        label_frames = int(row["total_frames"])
        if wrist_count != label_frames:
            raise ValueError(
                f"{episode_dir.name}: label total_frames={label_frames}, "
                f"wrist MP4={wrist_count}"
            )
        if exterior_count != wrist_count:
            raise ValueError(
                f"{episode_dir.name}: wrist/exterior frame counts differ "
                f"({wrist_count}, {exterior_count})"
            )
        if length != wrist_count + 1:
            raise ValueError(
                f"{episode_dir.name}: expected trajectory length = MP4 + 1, "
                f"got {length} and {wrist_count}"
            )

        skip_action = np.asarray(
            source["observation/timestamp/skip_action"][:], dtype=bool
        )
        movement_enabled = np.asarray(
            source["observation/controller_info/movement_enabled"][:],
            dtype=bool,
        )
        if not np.array_equal(skip_action, ~movement_enabled):
            raise ValueError(
                f"{episode_dir.name}: skip_action and movement_enabled disagree"
            )
        valid = ~(skip_action | ~movement_enabled)
        # Public DROID conversion reads MP4 frame i for trajectory step i.
        # The final trajectory step has no MP4 frame in this dataset.
        valid[wrist_count:] = False
        indices = np.flatnonzero(valid)
        if len(indices) == 0:
            raise ValueError(f"{episode_dir.name}: no valid aligned transitions")

        cartesian_state = np.asarray(
            source["observation/robot_state/cartesian_position"][:],
            dtype=np.float32,
        )[indices]
        eef_position = cartesian_state[:, :3]
        eef_euler_xyz = cartesian_state[:, 3:6]
        gripper_state = np.asarray(
            source["observation/robot_state/gripper_position"][:],
            dtype=np.float32,
        )[indices].reshape(-1, 1)
        cartesian_velocity = np.asarray(
            source["action/cartesian_velocity"][:], dtype=np.float32
        )[indices]
        gripper_velocity = np.asarray(
            source["action/gripper_velocity"][:], dtype=np.float32
        )[indices].reshape(-1, 1)
        actions = np.concatenate(
            [cartesian_velocity, gripper_velocity], axis=1
        )
        if actions.shape[1] != 7 or not np.isfinite(actions).all():
            raise ValueError(
                f"{episode_dir.name}: invalid action shape/values {actions.shape}"
            )

    wrist_images = read_resized_full_frames(wrist_video, indices)
    exterior_images = read_resized_full_frames(exterior_video, indices)

    observation = destination.create_group("obs")
    observation.create_dataset(
        "agentview_image",
        data=exterior_images,
        compression="gzip",
        compression_opts=1,
        chunks=(1, IMAGE_HEIGHT, IMAGE_WIDTH, 3),
    )
    observation.create_dataset(
        "robot0_eye_in_hand_image",
        data=wrist_images,
        compression="gzip",
        compression_opts=1,
        chunks=(1, IMAGE_HEIGHT, IMAGE_WIDTH, 3),
    )
    observation.create_dataset("robot0_eef_pos", data=eef_position)
    observation.create_dataset("robot0_eef_euler", data=eef_euler_xyz)
    observation.create_dataset("robot0_gripper_qpos", data=gripper_state)
    observation.create_dataset(
        "action_prior_dummy",
        data=np.zeros((len(indices), 1), dtype=np.float32),
    )
    destination.create_dataset("actions", data=actions)
    destination.create_dataset(
        "source_step_index", data=indices.astype(np.int32)
    )
    destination.attrs["num_samples"] = len(indices)
    destination.attrs["ep_idx"] = ep_idx
    destination.attrs["episode"] = episode_dir.name
    destination.attrs["label"] = row["score"]
    destination.attrs["label_value"] = float(row["score"])
    destination.attrs["source_video"] = row["source_video"]
    destination.attrs["wrist_serial"] = wrist_serial
    destination.attrs["exterior_serial"] = exterior_serial
    destination.attrs["trajectory_length"] = length
    destination.attrs["video_frame_count"] = wrist_count
    return {
        "trajectory_steps": length,
        "video_frames": wrist_count,
        "valid_transitions": len(indices),
    }


def write_masks(
    root: h5py.File, demo_keys: list[str], splits: dict[str, list[int]]
) -> None:
    mask = root.create_group("mask")
    for name, indices in splits.items():
        values = np.asarray(
            [demo_keys[index].encode() for index in indices], dtype="S"
        )
        mask.create_dataset(name, data=values)


def validate_output(path: Path, expected_episodes: int) -> None:
    required_masks = {
        "train",
        "valid",
        "score_all",
        "fold0_train",
        "fold0_valid",
        "fold0_score",
        "fold1_train",
        "fold1_valid",
        "fold1_score",
    }
    with h5py.File(path, "r") as handle:
        if len(handle["data"]) != expected_episodes:
            raise ValueError(
                f"Output has {len(handle['data'])} demos, expected "
                f"{expected_episodes}"
            )
        if not required_masks.issubset(handle["mask"]):
            raise ValueError(
                f"Missing masks: {sorted(required_masks - set(handle['mask']))}"
            )
        score_folds = [
            set(value.decode() for value in handle[f"mask/fold{i}_score"][:])
            for i in range(2)
        ]
        all_demos = set(handle["data"])
        if score_folds[0] & score_folds[1] or score_folds[0] | score_folds[1] != all_demos:
            raise ValueError("2-fold score masks must be disjoint and exhaustive")
        for demo_key in handle["data"]:
            demo = handle[f"data/{demo_key}"]
            length = len(demo["actions"])
            if demo["actions"].shape != (length, 7):
                raise ValueError(f"{demo_key}: bad action shape")
            for key in (
                "agentview_image",
                "robot0_eye_in_hand_image",
                "robot0_eef_pos",
                "robot0_eef_euler",
                "robot0_gripper_qpos",
                "action_prior_dummy",
            ):
                if key not in demo["obs"] or len(demo[f"obs/{key}"]) != length:
                    raise ValueError(f"{demo_key}: missing/misaligned obs/{key}")


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")
    rows = read_label_rows(args.labels_csv)
    if args.max_episodes is not None:
        rows = rows[: args.max_episodes]
    root_episodes = {
        path.name for path in args.root.iterdir() if path.is_dir()
    }
    label_episodes = {row["episode"] for row in rows}
    if args.max_episodes is None and root_episodes != label_episodes:
        raise ValueError(
            f"Dataset/label mismatch: missing={sorted(label_episodes-root_episodes)}, "
            f"unlabeled={sorted(root_episodes-label_episodes)}"
        )

    scores = [int(row["score"]) for row in rows]
    splits = build_splits(scores, args.valid_ratio, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp.hdf5")
    if temporary.exists():
        temporary.unlink()

    totals = Counter()
    with h5py.File(temporary, "w") as output:
        output.attrs["dataset_name"] = "wrench_on_hook_0722"
        output.attrs["source_root"] = str(args.root)
        output.attrs["source_labels_csv"] = str(args.labels_csv)
        output.attrs["action_dim"] = 7
        output.attrs["action_source"] = (
            "action/cartesian_velocity + action/gripper_velocity"
        )
        output.attrs["action_normalization"] = "none"
        output.attrs["conditional_observation"] = (
            "wrist_rgb + exterior_rgb + eef_position + "
            "raw eef_euler + gripper_position"
        )
        output.attrs["image_alignment"] = (
            "direct trajectory step i to MP4 frame i; unmatched final "
            "trajectory step dropped"
        )
        output.attrs["image_preprocessing"] = (
            "decode complete 1280x720 camera MP4 frame; RGB conversion; "
            "aspect-preserving INTER_AREA resize to 320x180; no spatial crop"
        )
        output.attrs["image_source_frame_shape"] = np.asarray(
            [720, 1280, 3], dtype=np.int32
        )
        output.attrs["image_stored_shape"] = np.asarray(
            [IMAGE_HEIGHT, IMAGE_WIDTH, 3], dtype=np.int32
        )
        output.attrs["orientation_representation"] = (
            "raw observation/robot_state/cartesian_position dimensions 3:6"
        )
        output.attrs["label_semantics"] = (
            "ordinal score; provisional assumption 1 < 2 < 3"
        )
        output.attrs["split_seed"] = args.seed
        output.attrs["valid_ratio"] = args.valid_ratio
        output.attrs["observation_keys_json"] = json.dumps(
            [
                "agentview_image",
                "robot0_eye_in_hand_image",
                "robot0_eef_pos",
                "robot0_eef_euler",
                "robot0_gripper_qpos",
                "action_prior_dummy",
            ]
        )
        env_args = json.dumps(
            {"env_name": "DROIDWrenchOnHook", "type": 1, "env_kwargs": {}}
        )
        output.attrs["env_args"] = env_args
        data = output.create_group("data")
        data.attrs["env_args"] = env_args
        demo_keys = []
        for ep_idx, row in enumerate(rows):
            demo_key = f"demo_{ep_idx}"
            demo_keys.append(demo_key)
            demo = data.create_group(demo_key)
            stats = copy_episode(
                demo, args.root / row["episode"], row, ep_idx
            )
            totals.update(stats)
            print(
                f"exported {ep_idx + 1}/{len(rows)} {row['episode']} "
                f"valid={stats['valid_transitions']}",
                flush=True,
            )
        data.attrs["num_demos"] = len(demo_keys)
        data.attrs["total"] = totals["valid_transitions"]
        output.attrs["total"] = totals["valid_transitions"]
        write_masks(output, demo_keys, splits)
        labels = output.create_group("labels")
        labels.create_dataset(
            "demo_key",
            data=np.asarray([key.encode() for key in demo_keys], dtype="S"),
        )
        labels.create_dataset(
            "label",
            data=np.asarray([str(score).encode() for score in scores], dtype="S"),
        )
        labels.create_dataset(
            "label_value", data=np.asarray(scores, dtype=np.float32)
        )
        labels.attrs["mapping_json"] = json.dumps(
            {"1": 1.0, "2": 2.0, "3": 3.0}
        )

    validate_output(temporary, len(rows))
    if args.output.exists():
        args.output.unlink()
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "episodes": len(rows),
                "valid_transitions": totals["valid_transitions"],
                "score_counts": dict(sorted(Counter(scores).items())),
                "split_sizes": {
                    name: len(indices) for name, indices in splits.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
