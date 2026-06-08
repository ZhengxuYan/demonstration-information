import json
import os
from pathlib import Path
from typing import Any, Iterator, Tuple

import cv2
import h5py
import numpy as np
import tensorflow_datasets as tfds


LANGUAGE_INSTRUCTION = "put the pen in the cup"
IMAGE_RES = (180, 320)
DEFAULT_RAW_ROOT = "/scr/rbhowmik/collected-data/pen-in-cup/06-07-2026-total-102"


def _raw_root() -> Path:
    return Path(os.environ.get("DROID_PEN_IN_CUP_RAW_ROOT", DEFAULT_RAW_ROOT)).expanduser()


def _episode_dirs(raw_root: Path) -> list[Path]:
    return sorted(
        path
        for path in raw_root.iterdir()
        if path.is_dir()
        and (path / "trajectory.h5").is_file()
        and (path / "recordings" / "MP4").is_dir()
    )


def _load_metadata(episode_dir: Path) -> dict[str, Any]:
    candidates = sorted(episode_dir.glob("metadata_*.json"))
    if not candidates:
        return {}
    with candidates[0].open("r", encoding="utf-8") as f:
        return json.load(f)


def _camera_ids(h5: h5py.File, mp4_dir: Path, metadata: dict[str, Any]) -> tuple[str, str, str]:
    camera_types = h5["observation"]["camera_type"]
    type_by_serial = {serial: int(camera_types[serial][0]) for serial in camera_types}
    available_serials = {
        path.stem
        for path in mp4_dir.glob("*.mp4")
        if "stereo" not in path.stem.lower() and path.stem.upper() != "N/A"
    }

    wrist_ids = [serial for serial, camera_type in type_by_serial.items() if camera_type == 0 and serial in available_serials]
    exterior_ids = [serial for serial, camera_type in type_by_serial.items() if camera_type != 0 and serial in available_serials]

    wrist_hint = str(metadata.get("wrist_cam_serial", ""))
    ext1_hint = str(metadata.get("ext1_cam_serial", ""))
    ext2_hint = str(metadata.get("ext2_cam_serial", ""))

    wrist = wrist_hint if wrist_hint in available_serials else (wrist_ids[0] if wrist_ids else "")
    ext1 = ext1_hint if ext1_hint in available_serials else (exterior_ids[0] if exterior_ids else "")
    ext2_candidates = [serial for serial in exterior_ids if serial != ext1]
    ext2 = ext2_hint if ext2_hint in available_serials else (ext2_candidates[0] if ext2_candidates else ext1)

    if not wrist or not ext1:
        raise ValueError(f"Could not resolve wrist/exterior cameras in {mp4_dir}")
    return wrist, ext1, ext2


def _open_capture(mp4_dir: Path, serial: str) -> cv2.VideoCapture:
    path = mp4_dir / f"{serial}.mp4"
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open MP4 {path}")
    return capture


def _read_left_rgb(capture: cv2.VideoCapture, serial: str, step_idx: int) -> np.ndarray:
    ok, frame = capture.read()
    if not ok or frame is None:
        raise ValueError(f"Could not read frame {step_idx} from camera {serial}")
    left = frame[:, : frame.shape[1] // 2, :]
    resized = cv2.resize(left, (IMAGE_RES[1], IMAGE_RES[0]), interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def _read_step_group(group: h5py.Group, index: int) -> dict[str, Any]:
    out = {}
    for key, value in group.items():
        if isinstance(value, h5py.Group):
            out[key] = _read_step_group(value, index)
        else:
            out[key] = value[index]
    return out


def _trajectory_length(h5: h5py.File) -> int:
    return int(h5["action"]["cartesian_position"].shape[0])


def _episode_key(episode_dir: Path, ep_idx: int) -> str:
    return f"{ep_idx:05d}_{episode_dir.name}"


def _parse_episode(episode_dir: Path, ep_idx: int) -> tuple[str, dict[str, Any]]:
    h5_path = episode_dir / "trajectory.h5"
    mp4_dir = episode_dir / "recordings" / "MP4"
    metadata = _load_metadata(episode_dir)

    with h5py.File(h5_path, "r") as h5:
        length = _trajectory_length(h5)
        wrist_serial, ext1_serial, ext2_serial = _camera_ids(h5, mp4_dir, metadata)

        captures = {
            "wrist": _open_capture(mp4_dir, wrist_serial),
            "ext1": _open_capture(mp4_dir, ext1_serial),
        }
        if ext2_serial == ext1_serial:
            captures["ext2"] = None
        else:
            captures["ext2"] = _open_capture(mp4_dir, ext2_serial)

        episode = []
        try:
            for i in range(length):
                obs_robot = _read_step_group(h5["observation"]["robot_state"], i)
                action = _read_step_group(h5["action"], i)
                wrist_image = _read_left_rgb(captures["wrist"], wrist_serial, i)
                ext1_image = _read_left_rgb(captures["ext1"], ext1_serial, i)
                ext2_image = ext1_image.copy() if captures["ext2"] is None else _read_left_rgb(captures["ext2"], ext2_serial, i)

                episode.append(
                    {
                        "observation": {
                            "exterior_image_1_left": ext1_image,
                            "exterior_image_2_left": ext2_image,
                            "wrist_image_left": wrist_image,
                            "cartesian_position": obs_robot["cartesian_position"],
                            "joint_position": obs_robot["joint_positions"],
                            "gripper_position": np.asarray([obs_robot["gripper_position"]], dtype=np.float64),
                        },
                        "action_dict": {
                            "cartesian_position": action["cartesian_position"],
                            "cartesian_velocity": action["cartesian_velocity"],
                            "gripper_position": np.asarray([action["gripper_position"]], dtype=np.float64),
                            "gripper_velocity": np.asarray([action["gripper_velocity"]], dtype=np.float64),
                            "joint_position": action["joint_position"],
                            "joint_velocity": action["joint_velocity"],
                        },
                        "action": np.concatenate(
                            [action["cartesian_position"], np.asarray([action["gripper_position"]], dtype=np.float64)]
                        ),
                        "discount": np.float32(1.0),
                        "reward": np.float32(i == length - 1 and bool(metadata.get("success", True))),
                        "is_first": i == 0,
                        "is_last": i == length - 1,
                        "is_terminal": i == length - 1,
                        "language_instruction": LANGUAGE_INSTRUCTION,
                    }
                )
        finally:
            for capture in captures.values():
                if capture is not None:
                    capture.release()

    return (
        _episode_key(episode_dir, ep_idx),
        {
            "steps": episode,
            "episode_metadata": {
                "file_path": str(h5_path),
                "recording_folderpath": str(mp4_dir),
                "ep_idx": np.int32(ep_idx),
                "quality_score": np.float32(1.0),
            },
        },
    )


class DroidPenInCup(tfds.core.GeneratorBasedBuilder):
    """RLDS builder for the June 2026 DROID pen-in-cup demos."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial release."}
    MANUAL_DOWNLOAD_INSTRUCTIONS = "Set DROID_PEN_IN_CUP_RAW_ROOT to the raw DROID collection directory."

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "exterior_image_1_left": tfds.features.Image(
                                        shape=(*IMAGE_RES, 3), dtype=np.uint8, encoding_format="jpeg"
                                    ),
                                    "exterior_image_2_left": tfds.features.Image(
                                        shape=(*IMAGE_RES, 3), dtype=np.uint8, encoding_format="jpeg"
                                    ),
                                    "wrist_image_left": tfds.features.Image(
                                        shape=(*IMAGE_RES, 3), dtype=np.uint8, encoding_format="jpeg"
                                    ),
                                    "cartesian_position": tfds.features.Tensor(shape=(6,), dtype=np.float64),
                                    "gripper_position": tfds.features.Tensor(shape=(1,), dtype=np.float64),
                                    "joint_position": tfds.features.Tensor(shape=(7,), dtype=np.float64),
                                }
                            ),
                            "action_dict": tfds.features.FeaturesDict(
                                {
                                    "cartesian_position": tfds.features.Tensor(shape=(6,), dtype=np.float64),
                                    "cartesian_velocity": tfds.features.Tensor(shape=(6,), dtype=np.float64),
                                    "gripper_position": tfds.features.Tensor(shape=(1,), dtype=np.float64),
                                    "gripper_velocity": tfds.features.Tensor(shape=(1,), dtype=np.float64),
                                    "joint_position": tfds.features.Tensor(shape=(7,), dtype=np.float64),
                                    "joint_velocity": tfds.features.Tensor(shape=(7,), dtype=np.float64),
                                }
                            ),
                            "action": tfds.features.Tensor(shape=(7,), dtype=np.float64),
                            "discount": tfds.features.Scalar(dtype=np.float32),
                            "reward": tfds.features.Scalar(dtype=np.float32),
                            "is_first": tfds.features.Scalar(dtype=np.bool_),
                            "is_last": tfds.features.Scalar(dtype=np.bool_),
                            "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                            "language_instruction": tfds.features.Text(),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(),
                            "recording_folderpath": tfds.features.Text(),
                            "ep_idx": tfds.features.Scalar(dtype=np.int32),
                            "quality_score": tfds.features.Scalar(dtype=np.float32),
                        }
                    ),
                }
            )
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        del dl_manager
        episodes = _episode_dirs(_raw_root())
        print(f"Found {len(episodes)} DROID pen-in-cup episodes under {_raw_root()}")
        return {"train": self._generate_examples(episodes)}

    def _generate_examples(self, episodes: list[Path]) -> Iterator[Tuple[str, Any]]:
        for ep_idx, episode_dir in enumerate(episodes):
            try:
                yield _parse_episode(episode_dir, ep_idx)
            except Exception as exc:
                print(f"Skipping {episode_dir}: {exc}")
