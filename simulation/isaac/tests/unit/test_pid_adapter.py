"""Unit tests for PID frame-convention boundaries."""

import torch

from tvc_env.common.quaternions import from_euler
from tvc_env.controllers.pid_adapter import PIDController


def _base_obs(quat):
    obs = torch.zeros(1, 24)
    obs[:, 3:7] = quat
    return obs


def test_isaac_positive_pitch_is_negated_before_attitude_error():
    controller = PIDController(
        kp_att=1.0,
        ki_att=0.0,
        kd_att=0.0,
        kp_yaw=0.0,
        kd_yaw=0.0,
        throttle_hover=0.0,
        max_fin_angle=1.0,
        num_envs=1,
    )
    quat = from_euler(torch.zeros(1), torch.tensor([0.1]), torch.zeros(1))

    action = controller.compute_action(_base_obs(quat))

    # With Isaac pitch negated to FRD, desired_pitch - pitch_frd is positive,
    # so the +X pitch fin command is positive.
    assert action[0, 0].item() > 0.0


def test_isaac_positive_yaw_is_negated_before_attitude_error():
    controller = PIDController(
        kp_att=0.0,
        kd_att=0.0,
        kp_yaw=1.0,
        ki_yaw=0.0,
        kd_yaw=0.0,
        throttle_hover=0.0,
        max_fin_angle=1.0,
        num_envs=1,
    )
    quat = from_euler(torch.zeros(1), torch.zeros(1), torch.tensor([0.1]))

    controller.compute_action(_base_obs(quat))
    att_err = controller.get_debug_state()["attitude_error_rpy"]

    assert att_err[2] > 0.0
