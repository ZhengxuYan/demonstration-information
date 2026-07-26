#!/usr/bin/env python3
"""Validate a Wrench 0722 density HDF5 against its raw DROID source."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import h5py
import numpy as np


def read_full_rgb(path: Path, index: int, height: int, width: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise ValueError(f"Could not read {path} frame {index}")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--contact-sheet", type=Path)
    args = parser.parse_args()

    action_max_error = 0.0
    position_max_error = 0.0
    euler_max_error = 0.0
    image_max_error = 0
    label_counts = Counter()
    preview_rows = []
    with h5py.File(args.input, "r") as handle:
        for demo_key in handle["data"]:
            demo = handle[f"data/{demo_key}"]
            episode = str(demo.attrs["episode"])
            label_counts[str(demo.attrs["label"])] += 1
            indices = np.asarray(demo["source_step_index"][:], dtype=np.int64)
            with h5py.File(
                args.raw_root / episode / "trajectory.h5", "r"
            ) as source:
                expected_actions = np.concatenate(
                    [
                        np.asarray(
                            source["action/cartesian_velocity"][:],
                            dtype=np.float32,
                        )[indices],
                        np.asarray(
                            source["action/gripper_velocity"][:],
                            dtype=np.float32,
                        )[indices].reshape(-1, 1),
                    ],
                    axis=1,
                )
                expected_position = np.asarray(
                    source[
                        "observation/robot_state/cartesian_position"
                    ][:],
                    dtype=np.float32,
                )[indices, :3]
                expected_euler = np.asarray(
                    source[
                        "observation/robot_state/cartesian_position"
                    ][:],
                    dtype=np.float32,
                )[indices, 3:6]
            action_max_error = max(
                action_max_error,
                float(
                    np.max(
                        np.abs(
                            np.asarray(demo["actions"][:])
                            - expected_actions
                        )
                    )
                ),
            )
            position_max_error = max(
                position_max_error,
                float(
                    np.max(
                        np.abs(
                            np.asarray(demo["obs/robot0_eef_pos"][:])
                            - expected_position
                        )
                    )
                ),
            )
            euler_max_error = max(
                euler_max_error,
                float(
                    np.max(
                        np.abs(
                            np.asarray(demo["obs/robot0_eef_euler"][:])
                            - expected_euler
                        )
                    )
                ),
            )

            if len(preview_rows) < 3:
                sample_positions = sorted(
                    {0, len(indices) // 2, len(indices) - 1}
                )
                wrist_serial = str(demo.attrs["wrist_serial"])
                exterior_serial = str(demo.attrs["exterior_serial"])
                for local_index in sample_positions:
                    source_index = int(indices[local_index])
                    stored_wrist = np.asarray(
                        demo["obs/robot0_eye_in_hand_image"][local_index]
                    )
                    stored_exterior = np.asarray(
                        demo["obs/agentview_image"][local_index]
                    )
                    height, width = stored_wrist.shape[:2]
                    raw_wrist = read_full_rgb(
                        args.raw_root
                        / episode
                        / "recordings/MP4"
                        / f"{wrist_serial}.mp4",
                        source_index,
                        height,
                        width,
                    )
                    raw_exterior = read_full_rgb(
                        args.raw_root
                        / episode
                        / "recordings/MP4"
                        / f"{exterior_serial}.mp4",
                        source_index,
                        height,
                        width,
                    )
                    image_max_error = max(
                        image_max_error,
                        int(np.max(np.abs(stored_wrist.astype(int) - raw_wrist))),
                        int(
                            np.max(
                                np.abs(
                                    stored_exterior.astype(int)
                                    - raw_exterior
                                )
                            )
                        ),
                    )
                    preview_rows.append(
                        np.concatenate(
                            [stored_exterior, stored_wrist], axis=1
                        )
                    )

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
        missing_masks = sorted(required_masks - set(handle["mask"]))

    report = {
        "demos": sum(label_counts.values()),
        "label_counts": dict(sorted(label_counts.items())),
        "action_max_abs_error": action_max_error,
        "eef_position_max_abs_error": position_max_error,
        "eef_euler_max_abs_error": euler_max_error,
        "image_roundtrip_max_abs_error": image_max_error,
        "missing_masks": missing_masks,
    }
    if any(
        [
            action_max_error != 0,
            position_max_error != 0,
            euler_max_error != 0,
            image_max_error != 0,
            missing_masks,
        ]
    ):
        raise ValueError(json.dumps(report, indent=2))
    if args.contact_sheet is not None and preview_rows:
        sheet = np.concatenate(preview_rows, axis=0)
        args.contact_sheet.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(args.contact_sheet),
            cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR),
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
