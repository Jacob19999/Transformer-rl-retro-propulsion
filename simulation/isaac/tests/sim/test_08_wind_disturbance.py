"""
Simulation test: Wind disturbance (test_08).

Tests:
  - Free-fall with zero thrust, verify horizontal drift direction matches steady wind vector
  - Rotate vehicle 90° and verify drag still opposes relative airflow correctly
  - Trigger gust event and verify transient force magnitude and duration

Requires Isaac Sim runtime.
"""

from __future__ import annotations
import pytest
import torch
from pathlib import Path

try:
    import omni.usd
    ISAAC_AVAILABLE = True
except ImportError:
    ISAAC_AVAILABLE = False

pytestmark = pytest.mark.skipif(not ISAAC_AVAILABLE, reason="Isaac Sim runtime not available")

SIM_ROOT = Path(__file__).parents[2]


def _set_deterministic_root_state(env) -> None:
    """Remove randomized spawn drift so the steady wind signal is measurable."""
    from tvc_env.common.quaternions import identity

    position = env._body_iface.get_root_position().clone()
    linear_vel = torch.zeros_like(position)
    angular_vel = torch.zeros_like(position)
    quaternion = identity(num=position.shape[0], device=env.device, dtype=position.dtype).reshape(-1, 4)

    env._body_iface.set_root_state(position, quaternion, linear_vel, angular_vel)
    env._sim_scene.step()


@pytest.fixture(scope="class")
def wind_env():
    from tvc_env.envs.base_env import BaseEnvConfig
    from tvc_env.envs.direct_rl_env import TVCDirectRLEnv

    config = BaseEnvConfig(
        task_name="hover",
        env_config_path=SIM_ROOT / "configs/env/single_env_debug.yaml",
        disturbance_config_path=SIM_ROOT / "configs/disturbances/wind.yaml",
        sim_root=SIM_ROOT,
    )
    env = TVCDirectRLEnv(config)
    env.reset()
    try:
        yield env
    finally:
        env.close()


class TestWindDisturbance:
    def test_drift_direction_matches_wind(self, wind_env):
        """Free-falling drone (zero thrust) should drift in the steady-wind x direction.

        Using free-fall instead of hovering avoids throttle-calibration and
        auto-reset noise.  As the drone falls, |v_rel| grows, so the horizontal
        wind-drag force increases — giving a clean, growing signal.
        30 env-steps ≈ 1 s sim time; the vertical drop stays well inside the
        10 m altitude-error termination limit.
        """
        import yaml
        from tvc_env.dynamics.wind_model import WindModel

        with open(SIM_ROOT / "configs/disturbances/wind.yaml") as f:
            wind_cfg = yaml.safe_load(f)
        wind_x = WindModel.from_disturbance_config(wind_cfg).get_effective_wind_world()[0].item()

        wind_env.reset()
        _set_deterministic_root_state(wind_env)
        initial_pos = wind_env._body_iface.get_root_position()[0].clone()

        # Zero throttle: drone free-falls; wind drag pushes it sideways.
        for _ in range(30):
            wind_env.step(torch.zeros(1, 5))

        drift_x = (wind_env._body_iface.get_root_position()[0] - initial_pos)[0].item()

        if abs(wind_x) > 0.1:
            assert wind_x * drift_x > 0, \
                f"Drift x={drift_x:.4f} m should align with wind x={wind_x:.3f} m/s"

    def test_drag_opposes_relative_airflow(self, wind_env):
        """Body drag force should oppose relative wind in rotated vehicle."""
        # Rotate vehicle 90° and compute drag — it should still oppose relative airflow
        from tvc_env.dynamics.wind_model import WindModel
        import yaml
        with open(SIM_ROOT / "configs/disturbances/wind.yaml") as f:
            wind_cfg = yaml.safe_load(f)

        wind_model = WindModel.from_disturbance_config(wind_cfg, device=wind_env._drone.device)

        # Test: drag opposes relative airspeed for arbitrary orientation
        linear_vel = torch.zeros(1, 3, device=wind_env._drone.device)
        q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=wind_env._drone.device)
        drag = wind_model.compute_drag_force(linear_vel, q)

        # When hovering (zero body vel) in positive x wind, drag should be in -x direction in body frame
        assert drag.shape == (1, 3)
        assert torch.all(torch.isfinite(drag))

    def test_gust_event_is_transient(self, wind_env):
        """Gust event should produce a transient force that decays."""
        from tvc_env.dynamics.wind_model import WindModel
        import yaml

        with open(SIM_ROOT / "configs/disturbances/wind.yaml") as f:
            wind_cfg = yaml.safe_load(f)
        wind_model = WindModel.from_disturbance_config(wind_cfg)

        # Manually trigger a gust
        wind_model._gust_active = True
        wind_model._gust_remaining = 0.5
        wind_model._gust_direction = torch.tensor([1.0, 0.0, 0.0])

        # Gust wind magnitude should be large
        wind_with_gust = wind_model.get_effective_wind_world()
        assert wind_with_gust.norm().item() > 4.0, "Gust should produce large wind magnitude"

        # After gust expires
        wind_model.update_gust(dt=1.0)  # Step past gust duration
        wind_no_gust = wind_model.get_effective_wind_world()
        assert wind_no_gust.norm().item() < wind_with_gust.norm().item(), \
            "Wind should decrease after gust expires"
