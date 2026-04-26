"""Unit tests for PID frame-convention boundaries."""

import torch

from tvc_env.common.quaternions import from_euler
from tvc_env.controllers.pid_adapter import PIDController


def _base_obs(quat):
    obs = torch.zeros(1, 24)
    obs[:, 3:7] = quat
    return obs


def _level_obs_with_position_error(x_error: float = 0.0, y_error: float = 0.0):
    obs = torch.zeros(1, 24)
    obs[:, 0] = x_error
    obs[:, 1] = y_error
    obs[:, 3] = 1.0
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


def test_x_position_error_commands_opposite_pitch_for_body_dynamics():
    controller = PIDController(
        kp_att=1.0,
        ki_att=0.0,
        kd_att=0.0,
        kp_yaw=0.0,
        kd_yaw=0.0,
        k_pos_xy=1.0,
        ki_pos_xy=0.0,
        k_vel_xy=0.0,
        max_tilt_cmd=1.0,
        max_tilt_rate=10.0,
        throttle_hover=0.0,
        max_fin_angle=1.0,
        num_envs=1,
    )

    controller.compute_action(_level_obs_with_position_error(x_error=0.1))
    att_err = controller.get_debug_state()["attitude_error_rpy"]

    assert att_err[1] < 0.0


def test_body_right_position_error_commands_positive_roll():
    controller = PIDController(
        kp_att=1.0,
        ki_att=0.0,
        kd_att=0.0,
        kp_yaw=0.0,
        kd_yaw=0.0,
        k_pos_xy=1.0,
        ki_pos_xy=0.0,
        k_vel_xy=0.0,
        max_tilt_cmd=1.0,
        max_tilt_rate=10.0,
        throttle_hover=0.0,
        max_fin_angle=1.0,
        num_envs=1,
    )

    # Isaac world +Y maps to body-FRD -Y at level attitude, so use negative
    # world-Y error to represent positive body-right error.
    controller.compute_action(_level_obs_with_position_error(y_error=-0.1))
    att_err = controller.get_debug_state()["attitude_error_rpy"]

    assert att_err[0] > 0.0
