from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "quality"))

from discrete_nll_utils import actions_to_bins, hard_discrete_nll_from_logits, soft_discrete_nll_from_logits, soft_discrete_targets


def test_soft_targets_sum_to_one_and_prefer_near_bins():
    target_bins = torch.tensor([[10]])
    weights = soft_discrete_targets(target_bins, num_bins=32, sigma_bins=1.5, truncate_bins=6)
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)))
    assert weights[0, 0, 10] > weights[0, 0, 11] > weights[0, 0, 14]
    assert weights[0, 0, 17] == 0


def test_sigma_zero_matches_hard_cross_entropy():
    logits = torch.randn(4, 3, 16)
    actions = torch.linspace(-0.9, 0.9, 12).reshape(4, 3)
    target_bins = actions_to_bins(actions, 16)
    expected = F.cross_entropy(logits.reshape(-1, 16), target_bins.reshape(-1), reduction="none").reshape(4, 3).sum(dim=-1)
    hard = hard_discrete_nll_from_logits(logits, actions, 16)
    soft = soft_discrete_nll_from_logits(logits, actions, 16, sigma_bins=0.0, truncate_bins=0)
    assert torch.allclose(hard, expected)
    assert torch.allclose(soft, expected)
