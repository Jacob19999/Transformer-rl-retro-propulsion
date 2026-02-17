"""Unit tests for RewardFunction (Stage 14)."""

from __future__ import annotations

import numpy as np

from simulation.training.reward import RewardFunction


def test_potential_shaping_positive_when_making_progress() -> None:
    rf = RewardFunction(
        {
            "alive_bonus": 0.0,
            "shaping": {"distance_coeff": 1.0, "velocity_coeff": 0.0, "gamma": 0.99},
            "orientation_weight": 0.0,
            "jerk_weight": 0.0,
            "fuel_weight": 0.0,
            "action_smooth_weight": 0.0,
        }
    )
    rf.reset()

    R = np.eye(3, dtype=float)
    p_target = np.zeros(3, dtype=float)
    a = np.zeros(5, dtype=float)

    # First call initializes prev_potential (shaping is 0.0 by design on step 0)
    r0 = rf.step_reward(
        p=np.array([10.0, 0.0, 0.0], dtype=float),
        v_b=np.zeros(3, dtype=float),
        R_body_to_inertial=R,
        p_target=p_target,
        action=a,
        T_cmd=0.0,
        T_max=1.0,
        dt_policy=0.025,
    )
    assert np.isfinite(r0)

    # Second call moves closer: potential becomes less negative => positive shaping
    r1 = rf.step_reward(
        p=np.array([5.0, 0.0, 0.0], dtype=float),
        v_b=np.zeros(3, dtype=float),
        R_body_to_inertial=R,
        p_target=p_target,
        action=a,
        T_cmd=0.0,
        T_max=1.0,
        dt_policy=0.025,
    )
    assert r1 > 0.0


def test_crash_terminal_penalty_is_negative() -> None:
    rf = RewardFunction({"crash_penalty": 123.0})
    r = rf.terminal_reward(
        landed=False,
        crashed=True,
        out_of_bounds=False,
        p=np.zeros(3, dtype=float),
        v_b=np.zeros(3, dtype=float),
        R_body_to_inertial=np.eye(3, dtype=float),
        p_target=np.zeros(3, dtype=float),
        v_max_touchdown=0.5,
    )
    assert r == -123.0


def test_perfect_landing_terminal_reward_matches_sum() -> None:
    rf = RewardFunction(
        {
            "landing_success": 100.0,
            "precision_bonus": 50.0,
            "precision_sigma": 0.1,
            "soft_touchdown": 20.0,
        }
    )
    r = rf.terminal_reward(
        landed=True,
        crashed=False,
        out_of_bounds=False,
        p=np.zeros(3, dtype=float),
        v_b=np.zeros(3, dtype=float),
        R_body_to_inertial=np.eye(3, dtype=float),
        p_target=np.zeros(3, dtype=float),
        v_max_touchdown=0.5,
    )
    assert r == 170.0

