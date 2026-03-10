"""
Tests for the Isaac Sim EDF landing environment.

T023: Single-env reset stability, observation dimensions, gravity fall
T024: test_fins.py analogue — fin deflection limits
T025: test_fin_deflection_limits
T026: test_action_scaling
T027: test_fin_lag_response
T029: test_parallel_independence

All tests are marked 'isaac' and skipped when IsaacLab is not available.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Conditional skip — IsaacLab only available inside Isaac Sim Python env
# ---------------------------------------------------------------------------
isaaclab = pytest.importorskip("isaaclab", reason="IsaacLab not available; skipping Isaac tests")

from simulation.isaac.envs.edf_isaac_env import EDFIsaacEnv  # noqa: E402

pytestmark = pytest.mark.isaac

_SINGLE_CFG = REPO_ROOT / "simulation" / "isaac" / "configs" / "isaac_env_single.yaml"
_CFG_128    = REPO_ROOT / "simulation" / "isaac" / "configs" / "isaac_env_128.yaml"

# Physical constants
_DELTA_MAX  = 0.2618   # rad, ±15°
_T_MAX      = 45.0     # N
_TAU_SERVO  = 0.04     # s
_TAU_MOTOR  = 0.10     # s
_DT         = 1.0 / 120.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def single_env():
    env = EDFIsaacEnv(config_path=_SINGLE_CFG, seed=0)
    yield env
    env.close()


# ---------------------------------------------------------------------------
# T023-a: 1000 reset-step-reset cycles headless
# ---------------------------------------------------------------------------
@pytest.mark.isaac
def test_single_env_reset_stability():
    """1000 reset-step-reset cycles must complete without exception."""
    env = EDFIsaacEnv(config_path=_SINGLE_CFG, seed=0)
    try:
        zero_action = np.zeros(5, dtype=np.float32)
        for i in range(1000):
            obs, info = env.reset()
            obs, rew, term, trunc, info = env.step(zero_action)
    finally:
        env.close()


# ---------------------------------------------------------------------------
# T023-b: Observation shape and dtype
# ---------------------------------------------------------------------------
@pytest.mark.isaac
def test_observation_dimensions(single_env):
    """Obs shape must be (20,), dtype float32, all finite after reset."""
    obs, _ = single_env.reset()
    assert obs.shape == (20,), f"Expected (20,), got {obs.shape}"
    assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"
    assert np.all(np.isfinite(obs)), "Observation contains non-finite values"


# ---------------------------------------------------------------------------
# T023-c: Gravity fall — zero actions
# ---------------------------------------------------------------------------
@pytest.mark.isaac
def test_gravity_fall(single_env):
    """With zero actions, h_agl must decrease monotonically until contact within 600 steps."""
    single_env.reset()
    zero_action = np.zeros(5, dtype=np.float32)

    prev_h = float("inf")
    contacted = False
    for step in range(600):
        obs, _, term, trunc, _ = single_env.step(zero_action)
        h_agl = float(obs[16])
        # h_agl should be non-increasing (allow tiny numerical noise ≤ 1 mm)
        assert h_agl <= prev_h + 0.001, (
            f"Step {step}: h_agl increased from {prev_h:.4f} to {h_agl:.4f}"
        )
        prev_h = h_agl
        if h_agl < 0.1:
            contacted = True
        if term or trunc:
            break

    assert contacted, "Drone did not contact ground within 600 steps"


# ---------------------------------------------------------------------------
# T025: Fin deflection limits ±15°
# ---------------------------------------------------------------------------
@pytest.mark.isaac
def test_fin_deflection_limits(single_env):
    """Command each fin to ±1.0, step 60 frames, verify clamped to ±15° (0.2618 rad)."""
    eps = 0.001  # rad

    for fin_idx in range(4):
        for direction in [+1.0, -1.0]:
            single_env.reset()
            action = np.zeros(5, dtype=np.float32)
            action[fin_idx + 1] = direction

            for _ in range(60):
                obs, _, term, trunc, _ = single_env.step(action)
                if term or trunc:
                    break

            # Read fin_deflections_actual from underlying task
            fin_actual = single_env._task.fin_deflections_actual[0, fin_idx].item()
            assert abs(fin_actual) <= _DELTA_MAX + eps, (
                f"Fin {fin_idx+1} deflection {fin_actual:.4f} rad exceeds limit {_DELTA_MAX:.4f}"
            )


# ---------------------------------------------------------------------------
# T026: Action scaling — thrust_cmd=1.0 → T_max within 5%
# ---------------------------------------------------------------------------
@pytest.mark.isaac
def test_action_scaling(single_env):
    """Full thrust command should produce T_actual ≈ T_max after lag settles (>5×tau_motor)."""
    single_env.reset()
    action = np.zeros(5, dtype=np.float32)
    action[0] = 1.0  # full thrust

    settle_steps = int(5 * _TAU_MOTOR / _DT)  # ≈ 60 steps
    for _ in range(settle_steps + 10):
        single_env.step(action)

    thrust = single_env._task.thrust_actual[0].item()
    assert abs(thrust - _T_MAX) / _T_MAX < 0.05, (
        f"Thrust {thrust:.2f} N not within 5% of T_max={_T_MAX:.1f} N"
    )


# ---------------------------------------------------------------------------
# T027: Fin lag response — 63% of target within tau_servo/dt steps
# ---------------------------------------------------------------------------
@pytest.mark.isaac
def test_fin_lag_response():
    """Fin deflection reaches 63% of target within tau_servo/dt steps (first-order lag)."""
    env = EDFIsaacEnv(config_path=_SINGLE_CFG, seed=1)
    try:
        env.reset()
        action = np.zeros(5, dtype=np.float32)
        action[1] = 1.0  # fin 1 to max

        settle_steps = int(_TAU_SERVO / _DT)  # ≈ 5 steps

        for step in range(settle_steps):
            env.step(action)

        fin_actual = env._task.fin_deflections_actual[0, 0].item()
        target = _DELTA_MAX
        # First-order lag: after 1 tau, response should be ≥ 63% of target
        assert fin_actual >= 0.63 * target, (
            f"After {settle_steps} steps, fin={fin_actual:.4f} rad < 63% of target={target:.4f}"
        )
    finally:
        env.close()


# ---------------------------------------------------------------------------
# T029: Parallel env independence (128 envs)
# ---------------------------------------------------------------------------
@pytest.mark.isaac
def test_parallel_independence():
    """Reset env 42 mid-episode; verify its episode_step resets while env 41 continues."""
    env = EDFIsaacEnv(config_path=_CFG_128, seed=42)
    try:
        env.reset()
        zero_action = np.zeros((128, 5), dtype=np.float32)

        # Step 10 frames
        for _ in range(10):
            env.step(zero_action)

        # Episode step before forced reset
        step_41_before = int(env._task._episode_step[41].item())
        assert step_41_before == 10, f"Expected step_41=10, got {step_41_before}"

        # Force reset env 42
        import torch
        env._task._reset_idx(torch.tensor([42], device=env._task.device))

        step_42_after = int(env._task._episode_step[42].item())
        assert step_42_after == 0, f"Env 42 episode_step should be 0 after reset, got {step_42_after}"

        # Step 10 more frames
        for _ in range(10):
            env.step(zero_action)

        step_41_final = int(env._task._episode_step[41].item())
        step_42_final = int(env._task._episode_step[42].item())
        assert step_41_final == 20, f"Env 41 should be at step 20, got {step_41_final}"
        assert step_42_final == 10, f"Env 42 should be at step 10, got {step_42_final}"
    finally:
        env.close()
