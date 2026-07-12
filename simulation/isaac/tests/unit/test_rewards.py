"""Unit tests for reward terms that exercise per-env target plumbing."""

import torch
from types import SimpleNamespace

from tvc_env.common.constants import ContactState
from tvc_env.envs.rewards import (
    compute_horizontal_closure_reward,
    compute_landing_success_reward,
    compute_off_pad_landing_penalty,
    compute_pad_accuracy_reward,
    compute_touchdown_softness_reward,
)


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
    """1 m horizontal offset from per-env target gives exp(-1.0) accuracy when landed."""
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

    expected = torch.exp(torch.tensor([-1.0, -1.0]))
    assert torch.allclose(reward, expected, atol=1e-6)


def test_pad_accuracy_reward_zero_when_not_landed():
    target = torch.zeros(2, 3)
    positions = torch.zeros(2, 3)
    contact = torch.tensor([ContactState.AIRBORNE, ContactState.AIRBORNE])
    state = _state(positions, contact)

    reward = compute_pad_accuracy_reward(state, {"_target_position_world": target})

    assert torch.allclose(reward, torch.zeros(2))


def test_touchdown_softness_reward_only_pays_when_landed():
    positions = torch.zeros(3, 3)
    contact = torch.tensor(
        [ContactState.AIRBORNE, ContactState.GROUND_CONTACT_CANDIDATE, ContactState.LANDED]
    )
    state = _state(positions, contact)
    state.linear_vel_frd = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.2],
            [0.0, 0.0, 0.2],
        ],
        dtype=torch.float32,
    )

    reward = compute_touchdown_softness_reward(state, {})

    assert torch.allclose(reward[:2], torch.zeros(2))
    assert torch.allclose(reward[2], torch.exp(torch.tensor(-0.2)))


def test_touchdown_rewards_use_first_contact_speed_after_dwell():
    state = _state(torch.zeros(1, 3), torch.tensor([ContactState.LANDED]))
    state.linear_vel_frd = torch.zeros(1, 3)
    state.touchdown_speed = torch.tensor([1.5])

    softness = compute_touchdown_softness_reward(state, {})
    success = compute_landing_success_reward(
        state,
        {"task": {"success": {"max_pad_distance": 0.5, "max_touchdown_speed": 0.25}}},
    )

    assert torch.allclose(softness, torch.exp(torch.tensor([-1.5])))
    assert torch.equal(success, torch.zeros(1))


def test_landing_success_reward_requires_pad_radius():
    target = torch.zeros(3, 3)
    positions = torch.tensor(
        [
            [0.25, 0.0, 0.0],
            [0.75, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    contact = torch.tensor([ContactState.LANDED, ContactState.LANDED, ContactState.AIRBORNE])
    state = _state(positions, contact)

    reward = compute_landing_success_reward(
        state,
        {
            "_target_position_world": target,
            "task": {"success": {"max_pad_distance": 0.5}},
        },
    )

    assert torch.allclose(reward, torch.tensor([1.0, 0.0, 0.0]))


def test_landing_success_reward_gates_on_touchdown_speed_when_configured():
    """When success.max_touchdown_speed is set, hard-touchdown landings inside
    the pad radius no longer earn the success bonus."""
    target = torch.zeros(3, 3)
    # All three envs land on-pad (xy = 0).
    positions = torch.zeros(3, 3, dtype=torch.float32)
    contact = torch.tensor([ContactState.LANDED, ContactState.LANDED, ContactState.LANDED])
    state = _state(positions, contact)
    # FRD: +z is down, so positive values = downward speed at touchdown.
    state.linear_vel_frd = torch.tensor(
        [
            [0.0, 0.0, 0.10],   # soft  (under 0.25 gate) → counts
            [0.0, 0.0, 0.30],   # hard  (over 0.25 gate)  → no success
            [0.0, 0.0, 0.25],   # exactly at gate         → counts (≤)
        ],
        dtype=torch.float32,
    )

    reward = compute_landing_success_reward(
        state,
        {
            "_target_position_world": target,
            "task": {"success": {"max_pad_distance": 0.5, "max_touchdown_speed": 0.25}},
        },
    )

    assert torch.allclose(reward, torch.tensor([1.0, 0.0, 1.0]))


def test_horizontal_closure_reward_sign_tracks_pad_closing_velocity():
    target = torch.zeros(3, 3)
    positions = torch.tensor(
        [
            [1.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
        ],
        dtype=torch.float32,
    )
    contact = torch.full((3,), ContactState.AIRBORNE)
    state = _state(positions, contact)
    state.linear_vel_world = torch.tensor(
        [
            [-0.4, 0.0, 0.0],  # toward target
            [0.4, 0.0, 0.0],   # away from target
            [0.4, 0.0, 0.0],   # at target, direction degenerates to zero
        ],
        dtype=torch.float32,
    )

    reward = compute_horizontal_closure_reward(state, {"_target_position_world": target})

    assert torch.allclose(reward, torch.tensor([0.4, -0.4, 0.0]), atol=1e-6)


def test_off_pad_landing_penalty_only_flags_landed_outside_pad():
    target = torch.zeros(4, 3)
    positions = torch.tensor(
        [
            [0.25, 0.0, 0.0],
            [0.75, 0.0, 0.0],
            [0.75, 0.0, 0.0],
            [0.75, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    contact = torch.tensor(
        [
            ContactState.LANDED,
            ContactState.LANDED,
            ContactState.AIRBORNE,
            ContactState.CRASHED,
        ]
    )
    state = _state(positions, contact)

    penalty = compute_off_pad_landing_penalty(
        state,
        {
            "_target_position_world": target,
            "task": {"success": {"max_pad_distance": 0.5}},
        },
    )

    assert torch.allclose(penalty, torch.tensor([0.0, 1.0, 0.0, 0.0]))
