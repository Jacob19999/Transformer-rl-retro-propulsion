"""Unit tests for vectorized reset coordinate handling."""

import torch

from tvc_env.sim.reset_logic import ResetManager


class _Body:
    def __init__(self):
        self.root_state = None
        self.env_ids = None
        self.joint_targets = None

    def set_root_state(self, position, quaternion_wxyz, linear_vel, angular_vel, env_ids=None):
        self.root_state = (position, quaternion_wxyz, linear_vel, angular_vel)
        self.env_ids = env_ids

    def set_fin_joint_targets(self, positions):
        self.joint_targets = positions.clone()


class _Servo:
    def reset(self, num_envs, device):
        return torch.ones(num_envs, 4, device=device)


class _Edf:
    omega_max = 4300.0

    def reset(self, num_envs, device):
        return torch.ones(num_envs, device=device)


class _Contacts:
    def __init__(self):
        self.env_ids = None

    def reset(self, env_ids):
        self.env_ids = env_ids


def test_reset_adds_env_origins_and_forwards_env_ids():
    body = _Body()
    contacts = _Contacts()
    env_origins = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    config = {
        "task": {
            "spawn": {
                "position_range": [[0.0, 0.0, 5.0], [0.0, 0.0, 5.0]],
                "velocity_range": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                "attitude_range": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            }
        }
    }
    manager = ResetManager(body, _Servo(), _Edf(), contacts, config, env_origins=env_origins)
    manager.initialize(num_envs=3, device=torch.device("cpu"))

    env_ids = torch.tensor([1, 2], dtype=torch.int64)
    manager.reset_envs(env_ids)

    positions = body.root_state[0]
    assert torch.allclose(positions, torch.tensor([[4.0, 0.0, 5.0], [8.0, 0.0, 5.0]]))
    assert torch.equal(body.env_ids, env_ids)
    assert torch.allclose(manager.servo_state[env_ids], torch.zeros(2, 4))
    assert torch.allclose(manager.omega_state[env_ids], torch.zeros(2))
