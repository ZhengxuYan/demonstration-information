#!/usr/bin/env python3
"""Export Qwen3-VL frame embeddings for expert200 random-post datasets.

The script builds a side-by-side primary-view + wrist composite for each frame
and embeds it with Qwen/Qwen3-VL-Embedding-8B using the official Qwen embedder
interface when available.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


DEFAULT_MODEL = "Qwen/Qwen3-VL-Embedding-8B"
DEFAULT_PROMPT = (
    "Represent this robot manipulation frame for retrieval by task state in a Square insertion task. "
    "Focus on whether the gripper is approaching, grasping, lifting, moving toward the square hole, "
    "aligning, inserting/placing, releasing, or resetting. Ignore lighting and camera-specific appearance."
)
VIEW_KEYS = {
    "agent_wrist": ("agentview_image", "robot0_eye_in_hand_image"),
    "left_close_low_wrist": ("left_close_low_image", "robot0_eye_in_hand_image"),
}

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Robomimic image.hdf5 dataset.")
    parser.add_argument("--output", type=Path, required=True, help="Output .npz path.")
    parser.add_argument("--view", choices=sorted(VIEW_KEYS), required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=None, help="Optional smoke-test frame cap.")
    parser.add_argument("--device", default=None, help="Optional model device, e.g. cuda:0.")
    return parser.parse_args()


def load_embedder(model_name: str, device: str | None):
    try:
        from scripts.qwen3_vl_embedding import Qwen3VLEmbedder
    except Exception as exc:
        raise RuntimeError(
            "Could not import scripts.qwen3_vl_embedding.Qwen3VLEmbedder. "
            "Install the official Qwen3-VL-Embedding requirements in this environment before running."
        ) from exc

    kwargs = {"model_name_or_path": model_name}
    if device is not None:
        kwargs["device"] = device
    try:
        return Qwen3VLEmbedder(**kwargs)
    except TypeError:
        if device is not None:
            kwargs.pop("device", None)
        return Qwen3VLEmbedder(**kwargs)


def iter_frames(dataset: Path, image_keys: tuple[str, str], max_frames: int | None):
    yielded = 0
    with h5py.File(dataset, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda name: int(name.split("_")[-1]))
        for demo_key in demos:
            obs = f["data"][demo_key]["obs"]
            missing = [key for key in image_keys if key not in obs]
            if missing:
                raise KeyError(f"{dataset}:{demo_key}/obs missing {missing}; available keys: {sorted(obs.keys())}")
            length = obs[image_keys[0]].shape[0]
            for step_idx in range(length):
                yield demo_key, int(demo_key.split("_")[-1]), step_idx, obs[image_keys[0]][step_idx], obs[image_keys[1]][step_idx]
                yielded += 1
                if max_frames is not None and yielded >= max_frames:
                    return


def composite_image(primary: np.ndarray, wrist: np.ndarray) -> Image.Image:
    left = Image.fromarray(np.asarray(primary, dtype=np.uint8)).convert("RGB")
    right = Image.fromarray(np.asarray(wrist, dtype=np.uint8)).convert("RGB")
    height = max(left.height, right.height)
    if left.height != height:
        left = left.resize((round(left.width * height / left.height), height), Image.BILINEAR)
    if right.height != height:
        right = right.resize((round(right.width * height / right.height), height), Image.BILINEAR)
    out = Image.new("RGB", (left.width + right.width, height), color=(0, 0, 0))
    out.paste(left, (0, 0))
    out.paste(right, (left.width, 0))
    return out


def embed_batch(embedder, prompt: str, image_paths: list[Path]) -> np.ndarray:
    inputs = [{"text": prompt, "image": str(path)} for path in image_paths]
    if hasattr(embedder, "encode"):
        embeddings = embedder.encode(inputs)
    elif hasattr(embedder, "embed"):
        embeddings = embedder.embed(inputs)
    else:
        raise AttributeError("Qwen3VLEmbedder exposes neither encode(...) nor embed(...).")
    return np.asarray(embeddings, dtype=np.float32)


def main() -> None:
    args = parse_args()
    image_keys = VIEW_KEYS[args.view]
    embedder = load_embedder(args.model_name, args.device)

    latents = []
    ep_idxs = []
    step_idxs = []
    demo_keys = []

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="qwen3_vl_frames_") as tmp:
        tmpdir = Path(tmp)
        batch_paths: list[Path] = []
        batch_meta: list[tuple[str, int, int]] = []

        def flush() -> None:
            if not batch_paths:
                return
            emb = embed_batch(embedder, args.prompt, batch_paths)
            latents.append(emb)
            demo_keys.extend(meta[0] for meta in batch_meta)
            ep_idxs.extend(meta[1] for meta in batch_meta)
            step_idxs.extend(meta[2] for meta in batch_meta)
            batch_paths.clear()
            batch_meta.clear()

        iterator = iter_frames(args.dataset, image_keys, args.max_frames)
        for demo_key, ep_idx, step_idx, primary, wrist in tqdm(iterator, desc="embedding qwen3-vl frames"):
            img = composite_image(primary, wrist)
            path = tmpdir / f"{demo_key}_{step_idx:04d}.jpg"
            img.save(path, quality=92)
            batch_paths.append(path)
            batch_meta.append((demo_key, ep_idx, step_idx))
            if len(batch_paths) >= args.batch_size:
                flush()
        flush()

    if not latents:
        raise ValueError(f"No frames read from {args.dataset}")

    latent = np.concatenate(latents, axis=0).astype(np.float32)
    np.savez_compressed(
        args.output,
        latent=latent,
        ep_idx=np.asarray(ep_idxs, dtype=np.int64),
        step_idx=np.asarray(step_idxs, dtype=np.int64),
        demo_key=np.asarray(demo_keys, dtype=object),
        view=np.asarray(args.view),
        image_keys=np.asarray(image_keys, dtype=object),
        prompt=np.asarray(args.prompt),
        model_name=np.asarray(args.model_name),
        embedding_dim=np.asarray(latent.shape[-1], dtype=np.int64),
    )
    print(args.output)
    print(f"frames={latent.shape[0]} embedding_dim={latent.shape[-1]}")


if __name__ == "__main__":
    main()
