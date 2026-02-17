"""
Gymnasium environment wrapper for EDF landing.

Implements Implementation Tracker Stage 12:
`EDFLandingEnv` wraps `VehicleDynamics` + `EnvironmentModel` with Gymnasium API
and provides standardized action/observation spaces for RL training.

Notes
-----
- Observation pipeline is implemented directly here for Stage 12 completeness.
  Stage 13 will factor this into `training/observation.py` with noise injection.
- Reward is implemented as a lightweight in-env version of training.md §5 so
  the environment is usable immediately; Stage 14 will factor into `reward.py`.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Gymnasium is required for EDFLandingEnv. Install with `pip install gymnasium`."
    ) from e

from simulation.config_loader import load_config
from simulation.dynamics.quaternion_utils import euler_to_quat, quat_to_dcm
from simulation.dynamics.vehicle import CONTROL_DIM, VehicleDynamics
from simulation.environment.environment_model import EnvironmentModel
from simulation.training.observation import ObservationConfig, ObservationPipeline
from simulation.training.reward import RewardFunction


OBS_DIM = 20
ACT_DIM = 5


@dataclass(slots=True)
class TerminationResult:
    terminated: bool
    landed: bool
    crashed: bool
    oob: bool
    reason: str


class EDFLandingEnv(gym.Env):
    """Shared Gymnasium environment for retro-propulsive EDF landing."""

    metadata = {"render_modes": []}

    def __init__(self, config: Mapping[str, Any] | str | Path | None = None) -> None:
        super().__init__()

        cfg = self._load_root_config(config)

        # Core models
        env_cfg = cfg.get("environment", cfg.get("env", {}))
        vehicle_cfg = cfg.get("vehicle", {})
        self.env_model = EnvironmentModel(env_cfg)
        self.vehicle = VehicleDynamics(vehicle_cfg, self.env_model)

        # Spaces (training.md §2.1, §3.2, §4.1)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32
        )

        # Episode timing (training.md §2.1–2.2)
        self.dt_physics = float(self.vehicle.dt)
        self.dt_policy = float(cfg.get("dt_policy", 0.025))
        self.substeps = int(round(self.dt_policy / self.dt_physics))
        if self.substeps < 1:
            raise ValueError("dt_policy must be >= dt_physics.")
        # Keep dt_policy consistent with substeps and physics dt to avoid drift.
        self.dt_policy = float(self.substeps * self.dt_physics)

        self.max_time = float(cfg.get("max_episode_time", 15.0))
        self.max_steps = int(cfg.get("max_steps", int(round(self.max_time / self.dt_policy))))

        # Target (NED position)
        self.p_target = np.asarray(cfg.get("target_position", [0.0, 0.0, 0.0]), dtype=float).reshape(3)

        # Action scaling (training.md §4.2)
        self.throttle_range = float(cfg.get("throttle_range", 0.5))
        self.delta_max = float(cfg.get("delta_max", 0.26))  # rad (~15°)

        # Ground contact (training.md §6.4)
        ground_cfg = cfg.get("ground_contact", {})
        self.k_ground = float(ground_cfg.get("k_spring", 10000.0))
        self.c_ground = float(ground_cfg.get("c_damper", 500.0))
        self.settle_steps = int(ground_cfg.get("settle_substeps", 5))

        # Reward function (Stage 14)
        self.reward_fn = RewardFunction(cfg.get("reward"))

        # Termination thresholds (training.md §6.3)
        term_cfg = cfg.get("termination", {})
        self.h_land = float(term_cfg.get("h_land", 0.05))
        self.v_land = float(term_cfg.get("v_land", 0.5))
        self.tilt_land = float(term_cfg.get("tilt_land", np.deg2rad(15.0)))
        self.omega_land = float(term_cfg.get("omega_land", 0.5))
        self.vz_hard_crash = float(term_cfg.get("vz_hard_crash", 2.0))
        self.tilt_abort = float(term_cfg.get("tilt_abort", np.deg2rad(60.0)))
        self.oob_radius = float(term_cfg.get("oob_radius", 20.0))
        self.oob_alt_margin = float(term_cfg.get("oob_alt_margin", 5.0))
        self.landing_grace_steps = int(term_cfg.get("landing_grace_steps", 40))

        # Initial-condition sampling configuration (training.md §6.1)
        ic_cfg = cfg.get("initial_conditions", {})
        self.ic_pos_xy_range = self._range2(ic_cfg.get("pos_xy_range", [-2.0, 2.0]), name="pos_xy_range")
        self.ic_altitude_range = self._range2(ic_cfg.get("altitude_range", [5.0, 10.0]), name="altitude_range")
        self.ic_vel_xy_range = self._range2(ic_cfg.get("vel_xy_range", [-2.0, 2.0]), name="vel_xy_range")
        self.ic_descent_rate_range = self._range2(
            ic_cfg.get("descent_rate_range", [0.0, 3.0]), name="descent_rate_range"
        )
        self.ic_tilt_range_rad = float(ic_cfg.get("tilt_range_rad", np.deg2rad(5.0)))
        self.ic_yaw_range = self._range2(ic_cfg.get("yaw_range", [0.0, 2.0 * np.pi]), name="yaw_range")
        self.ic_omega_range = self._range2(ic_cfg.get("omega_range", [-0.2, 0.2]), name="omega_range")

        # Internal episode state
        self.step_count = 0
        self.h0 = 0.0
        self._touched_ground = False
        self._hard_impact = False
        self._landing_grace_count = 0
        self._impact_vz_down: float | None = None

        # Observation pipeline (Stage 13)
        obs_section = dict(cfg.get("observation", {}))
        # Backward-compat: Stage 12 stored this at the root.
        if "wind_ema_alpha" in cfg and "wind_ema_alpha" not in obs_section:
            obs_section["wind_ema_alpha"] = cfg["wind_ema_alpha"]
        self.obs_pipeline = ObservationPipeline(ObservationConfig.from_config(obs_section))

        # Domain randomization: actuator delay (training.md §6.2.1)
        ad_cfg = cfg.get("actuator_delay", {})
        self._actuator_delay_enabled = bool(ad_cfg.get("enabled", False))
        esc_range = ad_cfg.get("esc_delay_range", [0.010, 0.040])
        servo_range = ad_cfg.get("servo_delay_range", [0.005, 0.020])
        self._esc_delay_range = self._range2(esc_range, name="esc_delay_range")
        self._servo_delay_range = self._range2(servo_range, name="servo_delay_range")

        # Domain randomization: observation latency (training.md §6.2.2)
        ol_cfg = cfg.get("obs_latency", {})
        self._obs_latency_enabled = bool(ol_cfg.get("enabled", False))
        ds_range = ol_cfg.get("delay_steps_range", [0, 3])
        ds_arr = np.asarray(ds_range, dtype=float).reshape(-1)
        if ds_arr.size != 2:
            raise ValueError("delay_steps_range must have 2 elements [lo, hi].")
        self._obs_delay_steps_lo = int(ds_arr[0])
        self._obs_delay_steps_hi = int(ds_arr[1])
        if self._obs_delay_steps_hi < self._obs_delay_steps_lo:
            raise ValueError("delay_steps_range must satisfy hi >= lo.")

        # Will be set at reset
        self._delay_policy_steps: int = 0
        self._action_buffer: deque[np.ndarray] = deque()
        self._prev_applied_u: np.ndarray = np.zeros(CONTROL_DIM, dtype=float)
        self._obs_delay_steps: int = 0
        self._obs_buffer: deque[np.ndarray] = deque()

    def _load_root_config(self, config: Mapping[str, Any] | str | Path | None) -> dict[str, Any]:
        """Load env config dict.

        Accepts:
        - dict-like with keys {'vehicle', 'environment', ...}
        - path to YAML file containing those keys
        - None: uses `simulation/configs/default_*.yaml`
        """
        if config is None:
            base = Path(__file__).resolve().parents[1] / "configs"
            v = load_config(base / "default_vehicle.yaml")
            e = load_config(base / "default_environment.yaml")
            r = load_config(base / "reward.yaml")
            dr = load_config(base / "domain_randomization.yaml")
            return {
                "vehicle": v.get("vehicle", v),
                "environment": e.get("environment", e),
                "reward": r.get("reward", r),
                **{k: v for k, v in dr.items() if k not in ("vehicle", "environment", "reward")},
            }

        if isinstance(config, (str, Path)):
            cfg = load_config(config)
            # allow a top-level 'training' key if used later
            return dict(cfg.get("training", cfg))

        return dict(config)

    @staticmethod
    def _range2(value: object, *, name: str) -> tuple[float, float]:
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size != 2:
            raise ValueError(f"{name} must have 2 elements [lo, hi], got {arr!r}.")
        lo = float(arr[0])
        hi = float(arr[1])
        if hi < lo:
            raise ValueError(f"{name} must satisfy hi >= lo, got [{lo}, {hi}].")
        return lo, hi

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        """Convert normalized [-1,1] action to physical units [T_cmd, delta_1..4]."""
        a = np.asarray(action, dtype=float).reshape(ACT_DIM)

        T_hover = float(self.vehicle.mass * self.vehicle.g)
        T_cmd = float(T_hover * (1.0 + a[0] * self.throttle_range))
        T_cmd = max(0.0, T_cmd)

        T_max = self.vehicle.thrust_model.config.T_max
        if T_max is None:
            # Keep within the intended envelope if no hardware max provided.
            T_max = T_hover * (1.0 + self.throttle_range)
        T_cmd = float(np.clip(T_cmd, 0.0, float(T_max)))

        deltas = a[1:5] * float(self.delta_max)
        return np.concatenate([[T_cmd], deltas]).astype(float)

    def _sample_initial_conditions(self) -> np.ndarray:
        """Randomized initial conditions (training.md §6.1)."""
        rng = self.np_random

        # Position (NED): z is down, so altitude h -> p_z = -h
        off_x = float(rng.uniform(*self.ic_pos_xy_range))
        off_y = float(rng.uniform(*self.ic_pos_xy_range))
        h0 = float(rng.uniform(*self.ic_altitude_range))
        p0 = self.p_target + np.array([off_x, off_y, -h0], dtype=float)

        # Inertial velocity (NED)
        vx_i = float(rng.uniform(*self.ic_vel_xy_range))
        vy_i = float(rng.uniform(*self.ic_vel_xy_range))
        vz_i = float(rng.uniform(*self.ic_descent_rate_range))  # positive = downward in NED

        # Orientation: small roll/pitch tilt + random yaw
        roll = float(rng.uniform(-self.ic_tilt_range_rad, self.ic_tilt_range_rad))
        pitch = float(rng.uniform(-self.ic_tilt_range_rad, self.ic_tilt_range_rad))
        yaw = float(rng.uniform(*self.ic_yaw_range))
        q0 = euler_to_quat(roll, pitch, yaw)
        R = quat_to_dcm(q0)

        v_b0 = R.T @ np.array([vx_i, vy_i, vz_i], dtype=float)

        omega_lo, omega_hi = self.ic_omega_range
        omega0 = np.asarray(rng.uniform(omega_lo, omega_hi, size=3), dtype=float).reshape(3)
        T_init = float(self.vehicle.mass * self.vehicle.g)

        self.h0 = h0
        return np.concatenate([p0, v_b0, q0, omega0, [T_init]]).astype(float)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        ic = self._sample_initial_conditions()
        # Seed both models for episode-level reproducibility (tracker 12.4).
        self.env_model.reset(seed)
        self.vehicle.reset(ic, seed=seed)

        self.step_count = 0
        self._touched_ground = False
        self._hard_impact = False
        self._impact_vz_down = None
        self._landing_grace_count = 0
        self.obs_pipeline.reset(self.np_random)
        self.reward_fn.reset()

        # Domain randomization: sample per-episode delays
        rng = self.np_random
        if self._actuator_delay_enabled:
            tau_esc = float(rng.uniform(*self._esc_delay_range))
            tau_servo = float(rng.uniform(*self._servo_delay_range))
            tau_act = max(tau_esc, tau_servo)
            # Delay in policy steps (training.md: "1–2 policy steps at 40 Hz")
            self._delay_policy_steps = max(1, int(np.ceil(tau_act / self.dt_policy)))
            self._action_buffer.clear()
            T_hover = float(self.vehicle.mass * self.vehicle.g)
            self._prev_applied_u = np.array([T_hover, 0, 0, 0, 0], dtype=float)
        else:
            self._delay_policy_steps = 0
            self._action_buffer.clear()
            self._prev_applied_u = np.zeros(CONTROL_DIM, dtype=float)

        if self._obs_latency_enabled:
            self._obs_delay_steps = int(
                rng.integers(self._obs_delay_steps_lo, self._obs_delay_steps_hi + 1)
            )
            self._obs_buffer.clear()
        else:
            self._obs_delay_steps = 0
            self._obs_buffer.clear()

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def _state_terms(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Convenience: unpack current vehicle state and rotation."""
        p, v_b, q, omega, T, _ = self.vehicle._unpack(self.vehicle.state)
        R = quat_to_dcm(q)
        return np.asarray(p), np.asarray(v_b), R, np.asarray(omega), float(T)

    def _get_obs(self) -> np.ndarray:
        """Compute 20-dim observation (training.md §3.2)."""
        p, v_b, R, omega, T = self._state_terms()
        env_vars = self.env_model.sample_at_state(self.vehicle.time, p)
        v_wind_ned = np.asarray(env_vars["wind"], dtype=float).reshape(3)
        return self.obs_pipeline.get_obs(
            p=p,
            v_b=v_b,
            R_body_to_inertial=R,
            omega=omega,
            T=T,
            mass=self.vehicle.mass,
            g=self.vehicle.g,
            p_target=self.p_target,
            v_wind_ned=v_wind_ned,
            t=self.vehicle.time,
            max_time=self.max_time,
        )

    def _tilt_angle(self, R: np.ndarray) -> float:
        g_body = R.T @ np.array([0.0, 0.0, 1.0], dtype=float)
        return float(np.arccos(np.clip(g_body[2], -1.0, 1.0)))

    def _check_terminated(self) -> TerminationResult:
        p, v_b, R, omega, _T = self._state_terms()
        v_inertial = R @ v_b
        h_agl = max(float(-p[2]), 0.0)
        tilt = self._tilt_angle(R)

        # Immediate hard-impact crash (training.md §6.3). This is detected at the
        # instant of first ground penetration, before the contact impulse settles.
        if self._hard_impact:
            return TerminationResult(True, False, True, False, "hard_ground_contact")

        # Ground contact / touchdown checks with settling grace period.
        # Real vehicles settle over a brief period after first ground contact.
        # The grace window lets the vehicle stabilise for a configurable number
        # of steps before declaring a crash.
        in_landing_zone = h_agl < self.h_land
        conditions_met = (
            in_landing_zone
            and float(np.linalg.norm(v_inertial)) < self.v_land
            and tilt < self.tilt_land
            and float(np.linalg.norm(omega)) < self.omega_land
        )

        if in_landing_zone:
            self._landing_grace_count += 1
        else:
            self._landing_grace_count = 0

        grace_expired = self._landing_grace_count > self.landing_grace_steps
        crashed = in_landing_zone and not conditions_met and grace_expired
        extreme_tilt = tilt > self.tilt_abort

        # Out of bounds (relative to target)
        e_xy = p[:2] - self.p_target[:2]
        oob = float(np.linalg.norm(e_xy)) > self.oob_radius or h_agl > (self.h0 + self.oob_alt_margin)

        if conditions_met:
            return TerminationResult(True, True, False, False, "landed")
        if crashed:
            return TerminationResult(True, False, True, False, "crashed")
        if extreme_tilt:
            return TerminationResult(True, False, False, False, "extreme_tilt")
        if oob:
            return TerminationResult(True, False, False, True, "out_of_bounds")
        return TerminationResult(False, False, False, False, "")

    def _apply_ground_contact(self) -> None:
        """Apply spring-damper ground contact correction (training.md §6.4).

        Implemented as an external impulse in inertial z, then mapped back to body
        velocity. Position is clamped to the ground plane (p_z <= 0) after the impulse
        to avoid numerical tunneling.
        """
        p, v_b, R, _omega, _T = self._state_terms()
        pz = float(p[2])  # NED: ground at 0, below ground => pz > 0
        penetration = max(0.0, pz)
        if penetration <= 0.0:
            return

        v_inertial = R @ v_b
        vz_down = float(v_inertial[2])  # positive = downward
        if (not self._hard_impact) and vz_down > self.vz_hard_crash:
            self._hard_impact = True
            self._impact_vz_down = vz_down

        F = self.k_ground * penetration + self.c_ground * max(0.0, vz_down)
        F = max(0.0, float(F))

        # Upward normal force => negative NED z
        a_inertial = np.array([0.0, 0.0, -F / self.vehicle.mass], dtype=float)
        v_inertial = v_inertial + a_inertial * float(self.dt_physics)
        v_b_new = R.T @ v_inertial

        # Update state in-place: v_b and clamp position to ground plane.
        self.vehicle.state[3:6] = v_b_new
        if self.vehicle.state[2] > 0.0:
            self.vehicle.state[2] = 0.0
        self._touched_ground = True

    def step(self, action: np.ndarray):
        a = np.asarray(action, dtype=float).reshape(ACT_DIM)
        a = np.clip(a, -1.0, 1.0)
        u = self._scale_action(a)

        # Actuator delay: buffer action for N policy steps (training.md §6.2.1)
        if self._actuator_delay_enabled and self._delay_policy_steps > 0:
            self._action_buffer.append(u.copy())
            if len(self._action_buffer) > self._delay_policy_steps:
                u_delayed = self._action_buffer.popleft()
            else:
                u_delayed = self._prev_applied_u.copy()
            self._prev_applied_u = u_delayed.copy()
            u = u_delayed
        else:
            self._prev_applied_u = u.copy()

        # Physics stepping
        self._touched_ground = False
        for _ in range(self.substeps):
            self.vehicle.step(u)
            self._apply_ground_contact()

        # If we touched ground, allow settling (training.md §6.4)
        if self._touched_ground:
            for _ in range(max(0, self.settle_steps)):
                self.vehicle.step(u)
                self._apply_ground_contact()

        obs_fresh = self._get_obs()

        # Observation latency: return stale obs (training.md §6.2.2)
        if self._obs_latency_enabled and self._obs_delay_steps > 0:
            self._obs_buffer.append(obs_fresh.copy())
            if len(self._obs_buffer) > self._obs_delay_steps:
                obs = self._obs_buffer[-(self._obs_delay_steps + 1)].copy()
            else:
                obs = obs_fresh.copy()
            while len(self._obs_buffer) > self._obs_delay_steps + 1:
                self._obs_buffer.popleft()
        else:
            obs = obs_fresh
        term = self._check_terminated()
        terminated = bool(term.terminated)
        truncated = bool(self.step_count >= self.max_steps)

        p, v_b, R, _omega, _T = self._state_terms()
        T_cmd = float(u[0])
        T_max = self.vehicle.thrust_model.config.T_max
        if T_max is None:
            T_max = float(self.vehicle.mass * self.vehicle.g) * (1.0 + self.throttle_range)

        reward = self.reward_fn.step_reward(
            p=p,
            v_b=v_b,
            R_body_to_inertial=R,
            p_target=self.p_target,
            action=a,
            T_cmd=T_cmd,
            T_max=float(T_max),
            dt_policy=float(self.dt_policy),
        )
        if terminated:
            reward += self.reward_fn.terminal_reward(
                landed=bool(term.landed),
                crashed=bool(term.crashed),
                out_of_bounds=bool(term.oob),
                p=p,
                v_b=v_b,
                R_body_to_inertial=R,
                p_target=self.p_target,
                v_max_touchdown=float(self.v_land),
            )

        info = self._get_info()
        info.update(
            {
                "termination_reason": term.reason,
                "landed": bool(term.landed),
                "crashed": bool(term.crashed),
                "out_of_bounds": bool(term.oob),
            }
        )

        self.step_count += 1
        return obs, float(reward), terminated, truncated, info

    def _get_info(self) -> dict[str, Any]:
        p, v_b, R, omega, T = self._state_terms()
        v_inertial = R @ v_b
        h_agl = max(float(-p[2]), 0.0)
        tilt = self._tilt_angle(R)

        touchdown_velocity: float | None = None
        if h_agl < 0.1:
            touchdown_velocity = float(np.linalg.norm(v_inertial))

        return {
            "position": p.copy(),
            "velocity_body": v_b.copy(),
            "velocity_inertial": v_inertial.copy(),
            "altitude": h_agl,
            "tilt_angle": tilt,
            "angular_rate": float(np.linalg.norm(omega)),
            "thrust": float(T),
            "time": float(self.vehicle.time),
            "cep": float(np.linalg.norm(p[:2] - self.p_target[:2])),
            "touchdown_velocity": touchdown_velocity,
            "step_count": int(self.step_count),
        }


__all__ = ["EDFLandingEnv", "OBS_DIM", "ACT_DIM"]

