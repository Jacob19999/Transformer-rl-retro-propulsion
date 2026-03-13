"""Unit tests for PIDController (Stage 17)."""

from __future__ import annotations

import numpy as np

from simulation.training.controllers.pid_controller import PIDController


def _base_cfg() -> dict:
    return {
        "pid": {
            "dt": 0.025,
            "delta_max": 0.26,
            "outer_loop": {
                "altitude": {"Kp": 1.0, "Ki": 0.5, "Kd": 0.2, "integral_limit": 1.0},
                "lateral_x": {"Kp": 0.5, "Kd": 0.1},
                "lateral_y": {"Kp": 0.5, "Kd": 0.1},
                "max_tilt_cmd_deg": 20.0,
            },
            "inner_loop": {
                "roll": {"Kp": 4.0, "Kd": 0.5},
                "pitch": {"Kp": 4.0, "Kd": 0.5},
            },
            "gain_schedule": {"enabled": False, "phases": []},
        }
    }


def _obs(
    *,
    target_body: np.ndarray | None = None,
    v_b: np.ndarray | None = None,
    g_body: np.ndarray | None = None,
    omega: np.ndarray | None = None,
    twr: float = 1.0,
    h_agl: float = 0.0,
) -> np.ndarray:
    o = np.zeros(20, dtype=float)
    o[0:3] = np.zeros(3) if target_body is None else np.asarray(target_body, dtype=float).reshape(3)
    o[3:6] = np.zeros(3) if v_b is None else np.asarray(v_b, dtype=float).reshape(3)
    o[6:9] = np.array([0.0, 0.0, 1.0], dtype=float) if g_body is None else np.asarray(g_body, dtype=float).reshape(3)
    o[9:12] = np.zeros(3) if omega is None else np.asarray(omega, dtype=float).reshape(3)
    o[12] = float(twr)
    o[16] = float(h_agl)
    return o.astype(np.float32)


def test_hover_like_state_produces_near_zero_fin_commands() -> None:
    ctrl = PIDController(_base_cfg())
    ctrl.reset()

    a = ctrl.get_action(_obs(h_agl=0.0))
    assert a.shape == (5,)
    assert np.all(np.isfinite(a))
    # No lateral/attitude error -> no fin command.
    np.testing.assert_allclose(a[1:5], np.zeros(4, dtype=float), atol=1e-6)


def test_lateral_error_maps_to_expected_fin_signs() -> None:
    ctrl = PIDController(_base_cfg())
    ctrl.reset()

    # Target is forward (+x): expect negative pitch command (nose down),
    # which maps to common-mode negative deflection on Fin_1+Fin_2.
    a_fwd = ctrl.get_action(_obs(target_body=np.array([1.0, 0.0, 0.0]), h_agl=0.0))
    assert a_fwd[1] < 0.0
    assert a_fwd[2] < 0.0

    # Target is right (+y): expect positive roll command,
    # which maps to common-mode negative deflection on Fin_3+Fin_4.
    a_right = ctrl.get_action(_obs(target_body=np.array([0.0, 1.0, 0.0]), h_agl=0.0))
    assert a_right[3] < 0.0
    assert a_right[4] < 0.0


def test_actions_are_clipped_to_unit_box() -> None:
    ctrl = PIDController(_base_cfg())
    ctrl.reset()

    # Large errors should saturate outputs within [-1, 1].
    a = ctrl.get_action(
        _obs(
            target_body=np.array([100.0, 100.0, 0.0]),
            v_b=np.array([50.0, 50.0, 0.0]),
            omega=np.array([10.0, 10.0, 0.0]),
            h_agl=10.0,
        )
    )
    assert np.all(a <= 1.0 + 1e-12)
    assert np.all(a >= -1.0 - 1e-12)


def test_reset_clears_integral_state() -> None:
    ctrl = PIDController(_base_cfg())
    ctrl.reset()

    o = _obs(h_agl=1.0)
    a1 = ctrl.get_action(o).copy()
    _ = ctrl.get_action(o)  # accumulate integral
    ctrl.reset()
    a2 = ctrl.get_action(o).copy()

    # After reset, the first-step response should match again.
    np.testing.assert_allclose(a1, a2, atol=1e-9)


def test_get_action_with_debug_matches_get_action() -> None:
    ctrl_plain = PIDController(_base_cfg())
    ctrl_plain.reset()
    ctrl_debug = PIDController(_base_cfg())
    ctrl_debug.reset()

    o = _obs(
        target_body=np.array([0.5, -0.25, 0.0]),
        v_b=np.array([0.1, -0.2, 0.3]),
        g_body=np.array([0.1, -0.1, 0.98]),
        omega=np.array([0.05, -0.02, 0.01]),
        h_agl=1.5,
    )
    a_plain = ctrl_plain.get_action(o)
    a_dbg, dbg = ctrl_debug.get_action_with_debug(o)

    np.testing.assert_allclose(a_plain, a_dbg, atol=1e-12)
    assert "alt_error" in dbg and "pitch_cmd" in dbg and "yaw_total" in dbg


def test_debug_sign_conventions_match_existing_tests() -> None:
    ctrl = PIDController(_base_cfg())
    ctrl.reset()

    # Forward (+x) target should still drive negative pitch_cmd and negative Fin_1/Fin_2.
    o_fwd = _obs(target_body=np.array([1.0, 0.0, 0.0]), h_agl=0.0)
    a_fwd, dbg_fwd = ctrl.get_action_with_debug(o_fwd)
    assert a_fwd[1] < 0.0 and a_fwd[2] < 0.0
    assert dbg_fwd["pitch_cmd"] < 0.0

    # Right (+y) target should still drive positive roll_cmd and negative Fin_3/Fin_4.
    o_right = _obs(target_body=np.array([0.0, 1.0, 0.0]), h_agl=0.0)
    a_right, dbg_right = ctrl.get_action_with_debug(o_right)
    assert a_right[3] < 0.0 and a_right[4] < 0.0
    assert dbg_right["roll_cmd"] > 0.0

    # Yaw damping pattern [-d, +d, +d, -d] should appear for pure yaw-rate input.
    o_yaw = _obs(omega=np.array([0.0, 0.0, 1.0]), h_agl=0.0)
    a_yaw, dbg_yaw = ctrl.get_action_with_debug(o_yaw)
    d = dbg_yaw["yaw_total"]
    # Fin pattern implied by yaw_total sign:
    # fin1 = pitch_cmd - d, fin2 = pitch_cmd + d, fin3 = -roll_cmd + d, fin4 = -roll_cmd - d
    assert np.sign(a_yaw[1] - a_yaw[2]) == -np.sign(d) or d == 0.0

