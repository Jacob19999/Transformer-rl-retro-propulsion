"""Unit tests for reward terms that exercise per-env target plumbing."""

import torch
from types import SimpleNamespace

from tvc_env.common.constants import ContactState
from tvc_env.envs.rewards import compute_pad_accuracy_reward


def _state(positions: torch.Tensor, contact: torch.Tensor) -> SimpleNamespace:
    """Build a minimal env-state stub for reward functions."""
    return SimpleNamespace(position=positions, contact_state=contact)


def test_pad_accuracy_reward_uses_per_env_target_xy_columns():
    """Regression: target[:, :2] (xy components per env), not target[:2] (first 2 envs)."""
    # 3 envs, each landed exactly on its own world target -> max accuracy.
    target = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [8.0, 2.0, 0.0],
        ],
        dtype=torch.float32,
    )
    positions = target.clone()
    contact = torch.tensor(
        [ContactState.LANDED, ContactState.LANDED, ContactState.LANDED]
    )
    state = _state(positions, contact)

    config = {"_target_position_world": target}
    reward = compute_pad_accuracy_reward(state, config)

    assert reward.shape == (3,)
    assert torch.allclose(reward, torch.ones(3), atol=1e-6)


def test_pad_accuracy_reward_decays_with_horizontal_offset():
    """1 m horizontal offset from per-env target gives exp(-2.0) accuracy when landed."""
    target = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [10.0, 10.0, 0.0],
        ],
        dtype=torch.float32,
    )
    positions = torch.tensor(
        [
            [1.0, 0.0, 0.0],   # 1 m off in x
            [10.0, 11.0, 0.0],  # 1 m off in y
        ],
        dtype=torch.float32,
    )
    contact = torch.tensor([ContactState.LANDED, ContactState.LANDED])
    state = _state(positions, contact)

    reward = compute_pad_accuracy_reward(state, {"_target_position_world": target})

    expected = torch.exp(torch.tensor([-2.0, -2.0]))
    assert torch.allclose(reward, expected, atol=1e-6)


def test_pad_accuracy_reward_zero_when_not_landed():
    target = torch.zeros(2, 3)
    positions = torch.zeros(2, 3)
    contact = torch.tensor([ContactState.AIRBORNE, ContactState.AIRBORNE])
    state = _state(positions, contact)

    reward = compute_pad_accuracy_reward(state, {"_target_position_world": target})

    assert torch.allclose(reward, torch.zeros(2))
