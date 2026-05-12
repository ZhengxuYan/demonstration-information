#!/usr/bin/env python3
"""Export expert200 wrist-view videos from robomimic states.

This is intentionally narrower than prepare_policy_view_datasets.py: it only
renders robot0_eye_in_hand and writes browser-ready MP4 files for annotation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from prepare_policy_view_datasets import create_env, load_env_meta, sorted_demo_keys
from serve_observability_annotation_app import write_video


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "expert200" / "demo.hdf5"
DEFAULT_FALLBACK = REPO_ROOT / "image.hdf5"
DEFAULT_OUT = REPO_ROOT / "observability_annotation_app" / "videos" / "expert200_wrist"
ROBOMIMIC_SOURCE = REPO_ROOT / "robomimic"
ROBOSUITE_SITE_PACKAGES = REPO_ROOT / ".venv-robomimic" / "lib" / "python3.10" / "site-packages"


def install_runtime_paths() -> None:
    if ROBOMIMIC_SOURCE.exists() and str(ROBOMIMIC_SOURCE) not in sys.path:
        sys.path.insert(0, str(ROBOMIMIC_SOURCE))
    if ROBOSUITE_SITE_PACKAGES.exists() and str(ROBOSUITE_SITE_PACKAGES) not in sys.path:
        sys.path.append(str(ROBOSUITE_SITE_PACKAGES))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--env-meta-fallback", type=Path, default=DEFAULT_FALLBACK)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--height", type=int, default=84)
    parser.add_argument("--width", type=int, default=84)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--start", type=int, default=1, help="1-based expert demo index to start from.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--from-obs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export existing obs/robot0_eye_in_hand_image frames when present. "
        "Use --no-from-obs to replay simulator states instead.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def render_wrist_frames(
    env,
    states: np.ndarray,
    height: int,
    width: int,
    model_xml: str | bytes | None = None,
) -> np.ndarray:
    if model_xml is not None:
        if isinstance(model_xml, bytes):
            model_xml = model_xml.decode("utf-8")
        env.reset_to({"model": model_xml})

    frames = []
    for state in states:
        env.env.sim.set_state_from_flattened(state)
        env.env.sim.forward()
        frame = env.render(
            mode="rgb_array",
            height=height,
            width=width,
            camera_name="robot0_eye_in_hand",
        )
        frames.append(np.asarray(frame, dtype=np.uint8))
    return np.stack(frames, axis=0)


def output_ep_idx_by_demo_key(hdf5_path: Path) -> dict[str, int]:
    with h5py.File(hdf5_path, "r") as f:
        raw_mapping = f.attrs.get("demo_key_mapping_json")
    if raw_mapping is None:
        return {}
    if isinstance(raw_mapping, bytes):
        raw_mapping = raw_mapping.decode("utf-8")
    mapping = json.loads(str(raw_mapping))
    return {new_key: int(str(old_key).split("_")[-1]) for new_key, old_key in mapping.items()}


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    install_runtime_paths()

    print(f"loading metadata from {args.source}", flush=True)
    with h5py.File(args.source, "r") as f:
        demo_keys = sorted_demo_keys(f["data"])
        has_wrist_obs = all(
            "obs" in f["data"][demo_key] and "robot0_eye_in_hand_image" in f["data"][demo_key]["obs"]
            for demo_key in demo_keys[:1]
        )
        env_meta = None if args.from_obs and has_wrist_obs else load_env_meta(f, args.source, args.env_meta_fallback)

    env = None
    if env_meta is not None:
        print("creating robosuite env for wrist rendering", flush=True)
        env = create_env(env_meta, args.height, args.width, ["robot0_eye_in_hand"])
        print("resetting env", flush=True)
        env.reset()
    else:
        print("exporting existing obs/robot0_eye_in_hand_image frames", flush=True)

    selected = demo_keys[args.start - 1 :]
    if args.limit is not None:
        selected = selected[: args.limit]
    print(f"exporting {len(selected)} expert200 wrist videos to {args.out_dir}", flush=True)

    output_ep_idx = output_ep_idx_by_demo_key(args.source)
    manifest = {}
    for demo_key in tqdm(selected, desc="exporting expert200 wrist videos"):
        ep_idx = output_ep_idx.get(demo_key, int(demo_key.split("_")[-1]))
        output_path = args.out_dir / f"demo_{ep_idx:04d}.mp4"
        manifest[demo_key] = str(output_path)
        if output_path.exists() and not args.overwrite:
            continue
        with h5py.File(args.source, "r") as f:
            demo = f["data"][demo_key]
            if args.from_obs and "obs" in demo and "robot0_eye_in_hand_image" in demo["obs"]:
                frames = demo["obs"]["robot0_eye_in_hand_image"][:]
            else:
                states = demo["states"][:]
                if env is None:
                    raise RuntimeError("state replay requested but env was not created")
                frames = render_wrist_frames(
                    env,
                    states,
                    args.height,
                    args.width,
                    model_xml=demo.attrs.get("model_file"),
                )
        write_video(output_path, frames, args.fps)

    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
