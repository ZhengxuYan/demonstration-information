"""Check Robomimic HDF5 / RLDS / eval-env observation alignment.

This diagnostic is meant for dataset compatibility debugging. It exports image
observations from:

1. the HDF5 dataset at each demo's recorded first state,
2. the current installed robosuite eval env reset to that recorded first state,
3. optionally the RLDS / TFDS builder after the OpenX standardization transform.

It also prints compact state/action/image structure summaries and simple image
distance metrics under vertical-flip variants.
"""

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image, ImageDraw
from robomimic.utils import obs_utils as ObsUtils
from robomimic.utils import env_utils

from openx.data.datasets.robomimic import robomimic_dataset_transform
from openx.envs.robomimic import _sanitize_env_metadata_for_installed_robosuite


HDF5_CAMERA_KEYS = ("agentview_image", "robot0_eye_in_hand_image")
RLDS_IMAGE_KEYS = ("agent", "wrist")


def _tree_summary(x: Any, prefix: str = ""):
    if isinstance(x, dict):
        for k in sorted(x):
            yield from _tree_summary(x[k], f"{prefix}/{k}" if prefix else str(k))
        return
    if isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            yield from _tree_summary(v, f"{prefix}/{i}" if prefix else str(i))
        return
    if hasattr(x, "numpy"):
        x = x.numpy()
    arr = np.asarray(x)
    if arr.size and np.issubdtype(arr.dtype, np.number):
        suffix = f"min={arr.min()} max={arr.max()}"
    else:
        suffix = "non_numeric"
    yield f"{prefix}: shape={arr.shape} dtype={arr.dtype} {suffix}"


def _to_uint8_image(x: Any) -> np.ndarray:
    if isinstance(x, (bytes, bytearray)):
        x = tf.io.decode_image(x, channels=3, expand_animations=False).numpy()
    elif hasattr(x, "numpy"):
        x = x.numpy()
    x = np.asarray(x)
    if x.ndim == 4:
        x = x[0]
    if x.dtype != np.uint8:
        x = np.asarray(np.clip(x, 0, 1) * 255, dtype=np.uint8) if x.max(initial=0) <= 1.0 else x.astype(np.uint8)
    return x


def _save_image(path: Path, image: Any, label: str | None = None) -> np.ndarray:
    arr = _to_uint8_image(image)
    img = Image.fromarray(arr)
    if label:
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, min(img.width, 420), 14), fill=(0, 0, 0))
        draw.text((3, 1), label, fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return arr


def _image_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    a = _to_uint8_image(a).astype(np.float32)
    b = _to_uint8_image(b).astype(np.float32)
    if a.shape != b.shape:
        return {"shape_a": a.shape, "shape_b": b.shape}
    diff = a - b
    flat_a = a.reshape(-1)
    flat_b = b.reshape(-1)
    corr = np.corrcoef(flat_a, flat_b)[0, 1] if np.std(flat_a) > 0 and np.std(flat_b) > 0 else np.nan
    return {
        "mse": float(np.mean(diff * diff)),
        "mae": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "corr": float(corr),
    }


def _load_env_meta(dataset_path: str, render_gpu_device_id: int | None = None) -> dict:
    with h5py.File(os.path.expanduser(dataset_path), "r") as f:
        env_meta = json.loads(f["data"].attrs["env_args"])
    env_meta = _sanitize_env_metadata_for_installed_robosuite(env_meta)
    env_meta = deepcopy(env_meta)
    env_kwargs = env_meta["env_kwargs"]
    env_kwargs["use_camera_obs"] = True
    env_kwargs["camera_names"] = ["agentview", "robot0_eye_in_hand"]
    env_kwargs.setdefault("camera_heights", 84)
    env_kwargs.setdefault("camera_widths", 84)
    if render_gpu_device_id is not None:
        env_kwargs["render_gpu_device_id"] = render_gpu_device_id
    return env_meta


def _make_resettable_env(dataset_path: str, render_gpu_device_id: int | None = None):
    ObsUtils.initialize_obs_utils_with_obs_specs(
        obs_modality_specs={
            "obs": {
                "low_dim": [
                    "object",
                    "robot0_eef_pos",
                    "robot0_eef_quat",
                    "robot0_gripper_qpos",
                    "robot0_joint_pos",
                    "robot0_joint_vel",
                ],
                "rgb": ["agentview_image", "robot0_eye_in_hand_image"],
            }
        }
    )
    env_meta = _load_env_meta(dataset_path, render_gpu_device_id=render_gpu_device_id)
    env = env_utils.create_env_from_metadata(
        env_meta=env_meta,
        env_name=env_meta["env_name"],
        render=False,
        render_offscreen=True,
        use_image_obs=True,
    )
    env.env.ignore_done = False
    return env


def _demo_names(f, max_demos: int):
    demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]))
    return demos[:max_demos]


def _export_hdf5_vs_env(
    hdf5_path: str,
    output_dir: Path,
    max_demos: int,
    render_gpu_device_id: int | None = None,
):
    env = _make_resettable_env(hdf5_path, render_gpu_device_id=render_gpu_device_id)
    rows = []

    with h5py.File(os.path.expanduser(hdf5_path), "r") as f:
        print("\nHDF5 root:", hdf5_path)
        print("env_args:", json.loads(f["data"].attrs["env_args"]).get("env_version"))
        demos = _demo_names(f, max_demos)
        for demo in demos:
            group = f["data"][demo]
            print(f"\n== {demo} ==")
            print("hdf5 obs keys:", sorted(group["obs"].keys()))
            print("actions:", group["actions"].shape, group["actions"].dtype)
            print("states:", group["states"].shape, group["states"].dtype)
            for key in HDF5_CAMERA_KEYS:
                if key in group["obs"]:
                    print(f"hdf5 obs/{key}:", group["obs"][key].shape, group["obs"][key].dtype)

            initial_state = {"states": group["states"][0]}
            model_file = group.attrs.get("model_file")
            used_model_file = False
            if model_file is not None:
                try:
                    env_obs = env.reset_to({"states": group["states"][0], "model": model_file})
                    used_model_file = True
                except Exception as e:
                    msg_lines = str(e).splitlines()
                    msg = msg_lines[-1] if msg_lines else repr(e)
                    print(f"model_file reset failed: {type(e).__name__}: {msg}")
                    env_obs = env.reset_to(initial_state)
            else:
                env_obs = env.reset_to(initial_state)
            print("env obs keys:", sorted(env_obs.keys()))
            print("used_model_file:", used_model_file)

            demo_dir = output_dir / demo
            for key in HDF5_CAMERA_KEYS:
                if key not in group["obs"] or key not in env_obs:
                    print(f"{key}: missing hdf5={key in group['obs']} env={key in env_obs}")
                    continue
                hdf5_img = _save_image(demo_dir / f"hdf5_{key}.png", group["obs"][key][0], f"hdf5 {demo} {key}")
                env_img = _save_image(demo_dir / f"env_reset_{key}.png", env_obs[key], f"env reset {demo} {key}")
                _save_image(demo_dir / f"hdf5_flipud_{key}.png", np.flip(hdf5_img, axis=0), f"hdf5 flipud {key}")
                _save_image(demo_dir / f"env_flipud_{key}.png", np.flip(env_img, axis=0), f"env flipud {key}")

                metrics = {
                    "demo": demo,
                    "key": key,
                    "used_model_file": used_model_file,
                    "hdf5_vs_env": _image_metrics(hdf5_img, env_img),
                    "hdf5_vs_env_flipud": _image_metrics(hdf5_img, np.flip(env_img, axis=0)),
                    "hdf5_flipud_vs_env": _image_metrics(np.flip(hdf5_img, axis=0), env_img),
                }
                rows.append(metrics)
                print(key, json.dumps(metrics, sort_keys=True))
    return rows


def _export_rlds_sample(rlds_path: str, output_dir: Path):
    print("\nRLDS root:", rlds_path)
    builder = tfds.builder_from_directory(builder_dir=rlds_path)
    ds = builder.as_dataset(split="train", decoders=dict(steps=tfds.decode.SkipDecoding()), shuffle_files=False)
    ep = next(iter(ds))
    steps = ep["steps"]
    def _take_first(x):
        return x[:1] if hasattr(x, "shape") and len(x.shape) > 0 else x

    if "episode_metadata" in ep:
        print("\nepisode metadata summary:")
        for line in _tree_summary(ep["episode_metadata"]):
            print(line)

    print("\nraw RLDS step summary:")
    for line in _tree_summary(tf.nest.map_structure(_take_first, steps)):
        print(line)

    steps_for_transform = dict(steps)
    if "episode_metadata" in ep:
        ep_len = tf.shape(steps["is_first"])[0]
        steps_for_transform["episode_metadata"] = tf.nest.map_structure(
            lambda x: tf.repeat(x, ep_len), ep["episode_metadata"]
        )

    try:
        transformed = robomimic_dataset_transform(steps_for_transform)
    except Exception as e:
        print(f"\nRLDS transform failed: {type(e).__name__}: {e}")
        return
    print("\ntransformed RLDS step summary:")
    for line in _tree_summary(transformed):
        print(line)

    img_root = output_dir / "rlds_first_train_episode"
    for key in RLDS_IMAGE_KEYS:
        try:
            img = transformed["observation"]["image"][key][0]
        except Exception as e:
            print(f"could not export transformed RLDS image {key}: {e}")
            continue
        _save_image(img_root / f"transformed_{key}.png", img, f"RLDS transformed {key}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", required=True)
    parser.add_argument("--rlds", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_demos", type=int, default=3)
    parser.add_argument("--render_gpu_device_id", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _export_hdf5_vs_env(
        args.hdf5,
        output_dir=output_dir / "hdf5_vs_env",
        max_demos=args.max_demos,
        render_gpu_device_id=args.render_gpu_device_id,
    )
    with (output_dir / "image_metrics.json").open("w") as f:
        json.dump(rows, f, indent=2)
    if args.rlds:
        _export_rlds_sample(args.rlds, output_dir=output_dir)
    print("\nwrote:", output_dir)


if __name__ == "__main__":
    main()
