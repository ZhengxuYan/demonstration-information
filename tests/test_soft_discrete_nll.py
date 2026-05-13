from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "quality"))

from discrete_nll_utils import (
    actions_to_bins,
    discrete_entropy_from_logits,
    hard_discrete_nll_from_logits,
    soft_discrete_nll_from_logits,
    soft_discrete_targets,
)
from policy_entropy_utils import sample_entropy_from_distribution


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


def test_discrete_entropy_matches_manual_categorical_entropy():
    logits = torch.tensor([[[0.0, 1.0, 2.0], [1.5, -0.5, 0.25]]])
    log_probs = F.log_softmax(logits, dim=-1)
    expected = -(log_probs.exp() * log_probs).sum(dim=-1).sum(dim=-1)
    actual = discrete_entropy_from_logits(logits)
    assert torch.allclose(actual, expected)


def test_sample_entropy_from_distribution_is_deterministic_and_shaped():
    mixture = torch.distributions.Categorical(logits=torch.tensor([[0.0, 0.5], [1.0, -1.0]]))
    component = torch.distributions.Independent(
        torch.distributions.Normal(
            loc=torch.zeros(2, 2, 3),
            scale=torch.ones(2, 2, 3),
        ),
        1,
    )
    dist = torch.distributions.MixtureSameFamily(mixture, component)
    first = sample_entropy_from_distribution(dist, num_samples=16, seed=42)
    second = sample_entropy_from_distribution(dist, num_samples=16, seed=42)
    assert first.shape == (2,)
    assert torch.allclose(first, second)
