import os
from typing import Any, Iterator, Tuple

import h5py
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


TRANSPORT_STATE_SIZE = 87
ACTION_SIZE = 14


def _read_or_zeros(group: h5py.Group, key: str, shape: tuple[int, ...], dtype=np.float32) -> np.ndarray:
    if key in group:
        return group[key][:].astype(dtype)
    return np.zeros(shape, dtype=dtype)


def _read_one_or_zeros(group: h5py.Group, key: str, index: int, shape: tuple[int, ...], dtype=np.float32) -> np.ndarray:
    if key in group:
        return group[key][index].astype(dtype)
    return np.zeros(shape, dtype=dtype)


def _gripper_width(group: h5py.Group, robot: int, demo_length: int) -> int:
    key = f"robot{robot}_gripper_qpos"
    if key in group:
        return int(group[key].shape[-1])
    return 2


def _joint_width(group: h5py.Group, robot: int, demo_length: int) -> int:
    key = f"robot{robot}_joint_pos"
    if key in group:
        return int(group[key].shape[-1])
    return 7


def _transport_state(group: h5py.Group, demo_length: int) -> np.ndarray:
    parts = []
    for robot in (0, 1):
        parts.extend(
            [
                _read_or_zeros(group, f"robot{robot}_eef_pos", (demo_length, 3)),
                _read_or_zeros(group, f"robot{robot}_eef_quat", (demo_length, 4)),
                _read_or_zeros(group, f"robot{robot}_gripper_qpos", (demo_length, _gripper_width(group, robot, demo_length))),
                _read_or_zeros(group, f"robot{robot}_joint_pos", (demo_length, _joint_width(group, robot, demo_length))),
                _read_or_zeros(group, f"robot{robot}_joint_vel", (demo_length, _joint_width(group, robot, demo_length))),
            ]
        )
    object_state = _read_or_zeros(group, "object", (demo_length, 0))
    parts.append(object_state)
    state = np.concatenate(parts, axis=-1).astype(np.float32)
    if state.shape[-1] != TRANSPORT_STATE_SIZE:
        raise ValueError(f"Expected transport state dim {TRANSPORT_STATE_SIZE}, got {state.shape[-1]}")
    return state


def _transport_state_one(group: h5py.Group, index: int) -> np.ndarray:
    parts = []
    for robot in (0, 1):
        parts.extend(
            [
                _read_one_or_zeros(group, f"robot{robot}_eef_pos", index, (3,)),
                _read_one_or_zeros(group, f"robot{robot}_eef_quat", index, (4,)),
                _read_one_or_zeros(group, f"robot{robot}_gripper_qpos", index, (_gripper_width(group, robot, index + 1),)),
                _read_one_or_zeros(group, f"robot{robot}_joint_pos", index, (_joint_width(group, robot, index + 1),)),
                _read_one_or_zeros(group, f"robot{robot}_joint_vel", index, (_joint_width(group, robot, index + 1),)),
            ]
        )
    parts.append(_read_one_or_zeros(group, "object", index, (41,)))
    state = np.concatenate(parts, axis=-1).astype(np.float32)
    if state.shape[-1] != TRANSPORT_STATE_SIZE:
        raise ValueError(f"Expected transport state dim {TRANSPORT_STATE_SIZE}, got {state.shape[-1]}")
    return state


def _decode_demo_names(mask_dataset) -> list[str]:
    return [elem.decode("utf-8") if isinstance(elem, bytes) else str(elem) for elem in np.array(mask_dataset[:])]


class RoboMimicTransport(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial transport release."}
    MANUAL_DOWNLOAD_INSTRUCTIONS = "Provide a manual_dir containing image.hdf5."

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "agent_image": tfds.features.Image(
                                        shape=(84, 84, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Transport shoulder camera 0 RGB observation.",
                                    ),
                                    "wrist_image": tfds.features.Image(
                                        shape=(84, 84, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Transport shoulder camera 1 RGB observation.",
                                    ),
                                    "state": tfds.features.FeaturesDict(
                                        {
                                            "transport": tfds.features.Tensor(
                                                shape=(TRANSPORT_STATE_SIZE,),
                                                dtype=np.float32,
                                                doc="Both robot proprioception plus object state.",
                                            ),
                                        }
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(ACTION_SIZE,),
                                dtype=np.float32,
                                doc="Full bimanual transport action.",
                            ),
                            "discount": tfds.features.Scalar(dtype=np.float32),
                            "reward": tfds.features.Scalar(dtype=np.float32),
                            "is_first": tfds.features.Scalar(dtype=np.bool_),
                            "is_last": tfds.features.Scalar(dtype=np.bool_),
                            "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                            "language_instruction": tfds.features.Text(doc="Language instruction."),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(doc="Path to the original data file."),
                            "ep_idx": tfds.features.Scalar(dtype=np.int32),
                            "quality_score": tfds.features.Scalar(dtype=np.float32),
                            "operator": tfds.features.Text(doc="Operator, if present."),
                        }
                    ),
                }
            )
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        dataset_path = os.path.join(dl_manager.manual_dir, "image.hdf5")
        with h5py.File(dataset_path, "r") as f:
            has_train = "mask" in f and "train" in f["mask"] and len(f["mask/train"]) > 0
            has_valid = "mask" in f and "valid" in f["mask"] and len(f["mask/valid"]) > 0

        splits = {
            "train": self._generate_examples(
                path=dataset_path,
                train=True,
                use_mask=has_train,
            )
        }
        if has_valid:
            splits["val"] = self._generate_examples(path=dataset_path, train=False, use_mask=True)
        return splits

    def _generate_examples(self, path: str, train: bool = True, use_mask: bool = True) -> Iterator[Tuple[str, Any]]:
        f = h5py.File(path, "r")
        if use_mask:
            mask_name = "train" if train else "valid"
            demos = _decode_demo_names(f[f"mask/{mask_name}"])
        else:
            demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]))

        language_instruction = "Transport the object."
        for demo in demos:
            demo_group = f["data"][demo]
            obs = demo_group["obs"]
            next_obs = demo_group["next_obs"]
            demo_length = demo_group["actions"].shape[0]
            actions = demo_group["actions"][:].astype(np.float32)
            if actions.shape[-1] != ACTION_SIZE:
                raise ValueError(f"{path}:{demo} expected action dim {ACTION_SIZE}, got {actions.shape[-1]}")
            for image_key in ("shouldercamera0_image", "shouldercamera1_image"):
                if image_key not in obs:
                    raise KeyError(f"{path}:{demo}/obs missing {image_key}; available keys: {sorted(obs.keys())}")

            rewards = (
                demo_group["rewards"][:].astype(np.float32)
                if "rewards" in demo_group
                else np.zeros(demo_length, dtype=np.float32)
            )
            data = dict(
                action=actions,
                observation=dict(
                    agent_image=obs["shouldercamera0_image"][:],
                    wrist_image=obs["shouldercamera1_image"][:],
                    state=dict(transport=_transport_state(obs, demo_length)),
                ),
                is_first=np.zeros(demo_length, dtype=np.bool_),
                is_last=np.zeros(demo_length, dtype=np.bool_),
                is_terminal=np.zeros(demo_length, dtype=np.bool_),
                discount=np.ones(demo_length, dtype=np.float32),
                reward=rewards,
            )
            data["is_first"][0] = True

            episode = []
            for i in range(demo_length):
                step = tf.nest.map_structure(lambda x, i=i: x[i], data)
                step["language_instruction"] = language_instruction
                episode.append(step)

            terminal_step = dict(
                action=np.zeros(ACTION_SIZE, dtype=np.float32),
                observation=dict(
                    agent_image=next_obs["shouldercamera0_image"][demo_length - 1],
                    wrist_image=next_obs["shouldercamera1_image"][demo_length - 1],
                    state=dict(transport=_transport_state_one(next_obs, demo_length - 1)),
                ),
                is_first=False,
                is_last=True,
                is_terminal=True,
                discount=1.0,
                reward=1.0,
                language_instruction=language_instruction,
            )
            episode.append(terminal_step)

            metadata = dict(
                ep_idx=int(demo.split("_")[-1]),
                file_path=os.path.join(path, demo),
                quality_score=-np.inf,
                operator="",
            )
            yield demo, dict(steps=episode, episode_metadata=metadata)
        f.close()
