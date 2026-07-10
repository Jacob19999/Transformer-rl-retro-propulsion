"""Unit tests for task termination logic."""

import torch

from tvc_env.common.constants import ContactState
from tvc_env.envs.terminations import check_all_terminations, check_altitude_termination


def test_landing_success_state_terminates_episode():
    quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    position = torch.zeros(2, 3)
    target = torch.zeros(2, 3)
    contact = torch.tensor([ContactState.AIRBORNE, ContactState.LANDED])
    step_count = torch.zeros(2, dtype=torch.int32)
    config = {
        "task": {
            "episode_length_s": 30.0,
            "success": {"state": "LANDED"},
            "termination": {"crash": True, "max_tilt": 1.57, "max_altitude_error": 10.0},
        }
    }

    dones = check_all_terminations(
        quaternion,
        position,
        target,
        contact,
        step_count,
        config,
        physics_dt=1.0 / 120.0,
        decimation=4,
    )

    assert dones.tolist() == [False, True]


def test_altitude_termination_ignores_horizontal_pad_distance():
    position = torch.tensor(
        [
            [20.0, 0.0, 5.0],
            [0.0, 0.0, 16.0],
        ],
        dtype=torch.float32,
    )
    target = torch.tensor([0.0, 0.0, 5.0], dtype=torch.float32)

    dones = check_altitude_termination(position, target, max_altitude_error=10.0)

    assert dones.tolist() == [False, True]


def test_check_all_terminations_does_not_mark_timeout_as_terminal():
    quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    position = torch.zeros(1, 3)
    target = torch.zeros(1, 3)
    contact = torch.tensor([ContactState.AIRBORNE])
    step_count = torch.tensor([10_000], dtype=torch.int32)
    config = {
        "task": {
            "episode_length_s": 30.0,
            "success": {"state": "LANDED"},
            "termination": {"crash": True, "max_tilt": 1.57, "max_altitude_error": 10.0},
        }
    }

    dones = check_all_terminations(
        quaternion,
        position,
        target,
        contact,
        step_count,
        config,
        physics_dt=1.0 / 120.0,
        decimation=4,
    )

    assert dones.tolist() == [False]
