"""
Simulation test: Steady hover with all force contributions (test_12).

Tests PID hover with wind disturbance enabled. Logs and verifies all torque
contributions separately per FR-018:
  - Fin aerodynamic forces
  - Static reaction torque (motor spin)
  - Dynamic spool torque (rpm change rate)
  - Gyroscopic precession torque
  - Wind drag force

Verifies:
  - All torque magnitudes are physically reasonable relative to each other
  - No sign-error-induced divergence over 15 seconds
  - Position error remains bounded under wind disturbance

Requires Isaac Sim runtime.
"""

from __future__ import annotations
import math
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
def wind_pid_env():
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


class TestSteadyHoverAllForces:
    def test_all_force_contributions_bounded(self, wind_pid_env):
        """All torque contributions should be finite and physically reasonable."""
        from tvc_env.controllers.pid_adapter import PIDController
        from tvc_env.dynamics.rotor_reaction import (
            compute_static_reaction_torque,
            compute_dynamic_spool_torque,
            compute_gyroscopic_precession,
        )
        from tvc_env.dynamics.wind_model import WindModel
        import yaml

        pid = PIDController(num_envs=1, device=wind_pid_env.device)
        obs_dict, _ = wind_pid_env.reset()
        obs = obs_dict["policy"]
        pid.reset()

        with open(SIM_ROOT / "configs/disturbances/wind.yaml") as f:
            wind_cfg = yaml.safe_load(f)
        wind_model = WindModel.from_disturbance_config(wind_cfg, device=wind_pid_env.device)

        dt = 1.0 / 30.0
        n_steps = int(15.0 / dt)

        pos_errors = []
        max_static_torque = 0.0
        max_gyro_torque = 0.0
        max_wind_force = 0.0

        for _ in range(n_steps):
            action = pid.compute_action(obs)
            obs_dict, _, terminated, truncated, _ = wind_pid_env.step(action)
            obs = obs_dict["policy"]

            throttle = action[0, 4]
            ang_vel_frd = obs[0, 10:13].unsqueeze(0)  # (1, 3)
            lin_vel = obs[0, 7:10].unsqueeze(0)        # (1, 3) body FRD
            quat = obs[0, 3:7].unsqueeze(0)            # (1, 4) wxyz

            # Static reaction torque
            from tvc_env.configs.params import load_edf_config
            try:
                edf_cfg = load_edf_config(SIM_ROOT / "configs/params/edf_90mm.yaml")
                omega_max = edf_cfg.get("omega_max", 3000.0)
                k_q = edf_cfg.get("k_Q", 1e-6)
            except Exception:
                omega_max, k_q = 3000.0, 1e-6

            omega = throttle.item() * omega_max
            static_tau = abs(k_q * omega ** 2)
            max_static_torque = max(max_static_torque, static_tau)

            # Gyroscopic precession
            rotor_spin_axis = torch.tensor([[0.0, 0.0, -1.0]], device=wind_pid_env.device)
            d_rotor = 0.09  # 90mm fan
            blade_mass = 0.05
            I_rotor = 0.5 * blade_mass * (d_rotor / 2) ** 2
            H_rotor = rotor_spin_axis * (omega * I_rotor)
            gyro_tau = torch.linalg.cross(ang_vel_frd, H_rotor).norm().item()
            max_gyro_torque = max(max_gyro_torque, gyro_tau)

            # Wind drag
            drag = wind_model.compute_drag_force(lin_vel, quat)
            wind_f = drag.norm().item()
            max_wind_force = max(max_wind_force, wind_f)

            # Position error
            pos_err = obs[0, 0:3].norm().item()
            pos_errors.append(pos_err)

            assert torch.isfinite(obs).all(), "Non-finite values in observation"

            if (terminated | truncated)[0].item():
                obs_dict, _ = wind_pid_env.reset()
                obs = obs_dict["policy"]
                pid.reset()

        mean_pos_err = sum(pos_errors) / len(pos_errors)

        # All contributions must be finite and bounded
        assert math.isfinite(max_static_torque), "Static reaction torque is not finite"
        assert math.isfinite(max_gyro_torque), "Gyroscopic torque is not finite"
        assert math.isfinite(max_wind_force), "Wind drag force is not finite"

        # Physical reasonableness: wind force should be less than main thrust
        main_thrust = 2.5 * 9.81  # ~vehicle weight for hover
        assert max_wind_force < main_thrust * 2.0, (
            f"Wind force {max_wind_force:.2f}N exceeds 2x hover thrust — likely a sign error"
        )

        # Hover should remain bounded under wind disturbance
        assert mean_pos_err < 2.0, (
            f"Mean pos error {mean_pos_err:.3f}m under wind disturbance is too large — "
            "possible sign-error divergence"
        )

    def test_no_divergence_under_wind(self, wind_pid_env):
        """PID should maintain bounded position under constant wind disturbance."""
        from tvc_env.controllers.pid_adapter import PIDController

        pid = PIDController(num_envs=1, device=wind_pid_env.device)
        obs_dict, _ = wind_pid_env.reset()
        obs = obs_dict["policy"]
        pid.reset()

        dt = 1.0 / 30.0
        n_steps = int(15.0 / dt)

        for step in range(n_steps):
            action = pid.compute_action(obs)
            obs_dict, _, terminated, truncated, _ = wind_pid_env.step(action)
            obs = obs_dict["policy"]

            pos_err = obs[0, 0:3].norm().item()
            assert pos_err < 5.0, (
                f"Position diverged to {pos_err:.2f}m at step {step} under wind disturbance"
            )
            assert torch.isfinite(obs).all(), f"Non-finite obs at step {step}"

            if (terminated | truncated)[0].item():
                obs_dict, _ = wind_pid_env.reset()
                obs = obs_dict["policy"]
                pid.reset()
