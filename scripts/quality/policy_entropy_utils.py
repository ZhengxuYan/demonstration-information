"""Policy entropy helpers used by robomimic scoring scripts."""

from __future__ import annotations

import torch


def sample_entropy_from_distribution(
    dist,
    num_samples: int,
    seed: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Estimate entropy with action samples from a torch distribution."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if device is None:
        device = torch.device("cpu")
    devices = [device.index] if device.type == "cuda" and device.index is not None else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        samples = dist.sample(sample_shape=torch.Size([num_samples]))
    log_probs = dist.log_prob(samples)
    return -log_probs.mean(dim=0)
