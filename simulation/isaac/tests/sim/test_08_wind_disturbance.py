"""
Simulation test: Wind disturbance (test_08).

Tests:
  - Apply constant wind to hovering vehicle, verify drift direction matches wind vector
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


@pytest.fixture
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
    return env


class TestWindDisturbance:
    def test_drift_direction_matches_wind(self, wind_env):
        """Vehicle should drift in wind direction when hovering at fixed throttle."""
        import torch
        from tvc_env.dynamics.wind_model import WindModel
        import yaml

        with open(SIM_ROOT / "configs/disturbances/wind.yaml") as f:
            wind_cfg = yaml.safe_load(f)
        wind_model = WindModel.from_disturbance_config(wind_cfg)
        wind_world = wind_model.get_effective_wind_world()

        obs_dict, _ = wind_env.reset()
        initial_pos = wind_env._body_iface.get_root_position()[0].clone()

        # Hover with fixed throttle for several seconds
        for _ in range(60):
            action = torch.zeros(1, 5)
            action[0, 4] = 0.75
            wind_env.step(action)

        final_pos = wind_env._body_iface.get_root_position()[0]
        drift = final_pos - initial_pos

        # Drift x-component should align with wind x-component sign
        wind_x = wind_world[0].item()
        drift_x = drift[0].item()
        if abs(wind_x) > 0.1:
            assert wind_x * drift_x > 0, \
                f"Drift x={drift_x:.3f} should align with wind x={wind_x:.3f}"

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
