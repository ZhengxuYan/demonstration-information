"""Utilities for hard and soft discrete-action negative log likelihood."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def actions_to_bins(actions: torch.Tensor, num_bins: int) -> torch.Tensor:
    clipped = torch.clamp(actions, -1.0, 1.0 - 1e-6)
    bins = torch.floor((clipped + 1.0) * 0.5 * num_bins).long()
    return torch.clamp(bins, 0, num_bins - 1)


def hard_discrete_nll_from_logits(logits: torch.Tensor, actions: torch.Tensor, num_bins: int) -> torch.Tensor:
    target_bins = actions_to_bins(actions, num_bins)
    logits = _align_logits_to_targets(logits, target_bins)
    ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target_bins.reshape(-1),
        reduction="none",
    ).reshape(target_bins.shape)
    return ce.sum(dim=-1)


def soft_discrete_targets(
    target_bins: torch.Tensor,
    num_bins: int,
    sigma_bins: float,
    truncate_bins: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    dtype = dtype or torch.float32
    if sigma_bins <= 0:
        return F.one_hot(target_bins, num_classes=num_bins).to(dtype)
    offsets = torch.arange(num_bins, device=target_bins.device, dtype=dtype)
    distances = offsets.view(*([1] * target_bins.ndim), num_bins) - target_bins.unsqueeze(-1).to(dtype)
    weights = torch.exp(-0.5 * (distances / float(sigma_bins)) ** 2)
    if truncate_bins >= 0:
        weights = weights * (torch.abs(distances) <= int(truncate_bins)).to(weights.dtype)
    return weights / torch.clamp(weights.sum(dim=-1, keepdim=True), min=1e-12)


def soft_discrete_nll_from_logits(
    logits: torch.Tensor,
    actions: torch.Tensor,
    num_bins: int,
    sigma_bins: float,
    truncate_bins: int,
) -> torch.Tensor:
    target_bins = actions_to_bins(actions, num_bins)
    logits = _align_logits_to_targets(logits, target_bins)
    targets = soft_discrete_targets(target_bins, num_bins, sigma_bins, truncate_bins, logits.dtype)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(targets * log_probs).sum(dim=-1).sum(dim=-1)


def _align_logits_to_targets(logits: torch.Tensor, target_bins: torch.Tensor) -> torch.Tensor:
    if logits.ndim == target_bins.ndim + 2 and logits.shape[1] == 1 and target_bins.ndim == 2:
        return logits[:, -1]
    if logits.ndim != target_bins.ndim + 1:
        raise ValueError(f"logits shape {tuple(logits.shape)} is incompatible with target bins {tuple(target_bins.shape)}")
    return logits
