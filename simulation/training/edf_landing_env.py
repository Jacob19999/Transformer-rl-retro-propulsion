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

        # Reward (training.md §5) — minimal in-env implementation for Stage 12
        reward_cfg = cfg.get("reward", {})
        self.gamma = float(reward_cfg.get("gamma", 0.99))
        self.alive_bonus = float(reward_cfg.get("alive_bonus", 0.1))
        self.c_d = float(reward_cfg.get("c_d", 1.0))
        self.c_v = float(reward_cfg.get("c_v", 0.2))
        self.w_theta = float(reward_cfg.get("w_theta", 0.5))
        self.w_j = float(reward_cfg.get("w_j", 0.05))
        self.j_ref = float(reward_cfg.get("j_ref", 10.0))
        self.w_f = float(reward_cfg.get("w_f", 0.01))
        self.w_a = float(reward_cfg.get("w_a", 0.02))
        self.R_land = float(reward_cfg.get("R_land", 100.0))
        self.R_prec = float(reward_cfg.get("R_prec", 50.0))
        self.sigma_prec = float(reward_cfg.get("sigma_prec", 0.1))
        self.R_soft = float(reward_cfg.get("R_soft", 20.0))
        self.R_crash = float(reward_cfg.get("R_crash", 100.0))
        self.R_oob = float(reward_cfg.get("R_oob", 50.0))

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

        # Internal episode state
        self.step_count = 0
        self.prev_action = np.zeros(ACT_DIM, dtype=float)
        self.wind_ema = np.zeros(3, dtype=float)
        self.wind_ema_alpha = float(cfg.get("wind_ema_alpha", 0.05))
        self.prev_velocity: np.ndarray | None = None
        self.prev_accel: np.ndarray | None = None
        self.prev_potential: float | None = None
        self.h0 = 0.0
        self._touched_ground = False

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
            return {"vehicle": v.get("vehicle", v), "environment": e.get("environment", e)}

        if isinstance(config, (str, Path)):
            cfg = load_config(config)
            # allow a top-level 'training' key if used later
            return dict(cfg.get("training", cfg))

        return dict(config)

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
        off_x = float(rng.uniform(-2.0, 2.0))
        off_y = float(rng.uniform(-2.0, 2.0))
        h0 = float(rng.uniform(5.0, 10.0))
        p0 = self.p_target + np.array([off_x, off_y, -h0], dtype=float)

        # Inertial velocity (NED)
        vx_i = float(rng.uniform(-2.0, 2.0))
        vy_i = float(rng.uniform(-2.0, 2.0))
        vz_i = float(rng.uniform(0.0, 3.0))  # positive = downward in NED

        # Orientation: small roll/pitch tilt + random yaw
        roll = float(rng.uniform(-np.deg2rad(5.0), np.deg2rad(5.0)))
        pitch = float(rng.uniform(-np.deg2rad(5.0), np.deg2rad(5.0)))
        yaw = float(rng.uniform(0.0, 2.0 * np.pi))
        q0 = euler_to_quat(roll, pitch, yaw)
        R = quat_to_dcm(q0)

        v_b0 = R.T @ np.array([vx_i, vy_i, vz_i], dtype=float)

        omega0 = np.asarray(rng.uniform(-0.2, 0.2, size=3), dtype=float).reshape(3)
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
        self.prev_action = np.zeros(ACT_DIM, dtype=float)
        self.wind_ema = np.zeros(3, dtype=float)
        self.prev_velocity = None
        self.prev_accel = None
        self.prev_potential = None
        self._touched_ground = False

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
        v_inertial = R @ v_b
        env_vars = self.env_model.sample_at_state(self.vehicle.time, p)
        v_wind_ned = np.asarray(env_vars["wind"], dtype=float).reshape(3)
        v_wind_body = R.T @ v_wind_ned

        # Target offset in body frame
        e_p_body = R.T @ (self.p_target - p)

        # Gravity direction in body frame (NED gravity direction = +z)
        g_body = R.T @ np.array([0.0, 0.0, 1.0], dtype=float)

        # Wind EMA estimate (body frame)
        a = float(self.wind_ema_alpha)
        self.wind_ema = (1.0 - a) * self.wind_ema + a * v_wind_body

        h_agl = max(float(-p[2]), 0.0)
        speed = float(np.linalg.norm(v_b))
        ang_speed = float(np.linalg.norm(omega))
        twr = float(T / (self.vehicle.mass * self.vehicle.g + 1e-9))
        time_frac = float(np.clip(self.vehicle.time / max(self.max_time, 1e-9), 0.0, 1.0))

        obs = np.concatenate(
            [
                e_p_body.astype(float),
                v_b.astype(float),
                g_body.astype(float),
                omega.astype(float),
                np.array([twr], dtype=float),
                self.wind_ema.astype(float),
                np.array([h_agl, speed, ang_speed, time_frac], dtype=float),
            ]
        )
        if obs.size != OBS_DIM:
            raise RuntimeError(f"Expected obs dim {OBS_DIM}, got {obs.size}.")
        return obs.astype(np.float32)

    def _tilt_angle(self, R: np.ndarray) -> float:
        g_body = R.T @ np.array([0.0, 0.0, 1.0], dtype=float)
        return float(np.arccos(np.clip(g_body[2], -1.0, 1.0)))

    def _check_terminated(self) -> TerminationResult:
        p, v_b, R, omega, _T = self._state_terms()
        v_inertial = R @ v_b
        h_agl = max(float(-p[2]), 0.0)
        tilt = self._tilt_angle(R)

        # Ground contact / touchdown checks
        landed = (
            h_agl < self.h_land
            and float(np.linalg.norm(v_inertial)) < self.v_land
            and tilt < self.tilt_land
            and float(np.linalg.norm(omega)) < self.omega_land
        )
        on_ground = h_agl <= 0.0
        hard_ground_contact = on_ground and float(abs(v_inertial[2])) > self.vz_hard_crash

        crashed = (h_agl < self.h_land and not landed) or hard_ground_contact
        extreme_tilt = tilt > self.tilt_abort

        # Out of bounds (relative to target)
        e_xy = p[:2] - self.p_target[:2]
        oob = float(np.linalg.norm(e_xy)) > self.oob_radius or h_agl > (self.h0 + self.oob_alt_margin)

        if landed:
            return TerminationResult(True, True, False, False, "landed")
        if crashed:
            return TerminationResult(True, False, True, False, "crashed")
        if hard_ground_contact:
            return TerminationResult(True, False, True, False, "hard_ground_contact")
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

    def _potential(self) -> float:
        p, v_b, _R, _omega, _T = self._state_terms()
        e_p = self.p_target - p
        return float(-self.c_d * np.linalg.norm(e_p) - self.c_v * np.linalg.norm(v_b))

    def _terminal_reward(self, term: TerminationResult) -> float:
        if term.landed:
            p, v_b, R, _omega, _T = self._state_terms()
            v_inertial = R @ v_b
            e_xy = p[:2] - self.p_target[:2]
            cep = float(np.linalg.norm(e_xy))
            v_touch = float(np.linalg.norm(v_inertial))
            r_prec = self.R_prec * float(np.exp(-(cep * cep) / (2.0 * self.sigma_prec * self.sigma_prec)))
            r_soft = self.R_soft * max(0.0, (1.0 - v_touch / max(self.v_land, 1e-9)))
            return float(self.R_land + r_prec + r_soft)
        if term.crashed:
            return float(-self.R_crash)
        if term.oob:
            return float(-self.R_oob)
        return 0.0

    def _step_reward(self, action: np.ndarray, u_phys: np.ndarray) -> float:
        # Alive + potential shaping
        phi = self._potential()
        if self.prev_potential is None:
            shape = 0.0
        else:
            shape = self.gamma * phi - float(self.prev_potential)
        self.prev_potential = phi

        # Orientation penalty
        _p, _v_b, R, omega, T = self._state_terms()
        g_body = R.T @ np.array([0.0, 0.0, 1.0], dtype=float)
        orient = -self.w_theta * (1.0 - float(g_body[2]))

        # Fuel penalty (proxy)
        T_cmd = float(u_phys[0])
        T_max = self.vehicle.thrust_model.config.T_max
        if T_max is None:
            T_max = float(self.vehicle.mass * self.vehicle.g) * (1.0 + self.throttle_range)
        fuel = -self.w_f * (T_cmd / max(float(T_max), 1e-9)) * float(self.dt_policy)

        # Action smoothness
        act_smooth = -self.w_a * float(np.linalg.norm(np.asarray(action, dtype=float) - self.prev_action))

        # Jerk penalty (finite-difference; skip first 2 steps)
        jerk_pen = 0.0
        if self.prev_velocity is not None:
            accel = (np.asarray(self.vehicle.state[3:6], dtype=float) - self.prev_velocity) / float(self.dt_policy)
            if self.prev_accel is not None and self.step_count >= 2:
                jerk = float(np.linalg.norm(accel - self.prev_accel) / float(self.dt_policy))
                jerk_pen = -self.w_j * (jerk / max(self.j_ref, 1e-9))
            self.prev_accel = accel
        self.prev_velocity = np.asarray(self.vehicle.state[3:6], dtype=float).copy()

        return float(self.alive_bonus + shape + orient + jerk_pen + fuel + act_smooth)

    def step(self, action: np.ndarray):
        a = np.asarray(action, dtype=float).reshape(ACT_DIM)
        a = np.clip(a, -1.0, 1.0)
        u = self._scale_action(a)

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

        obs = self._get_obs()
        term = self._check_terminated()
        terminated = bool(term.terminated)
        truncated = bool(self.step_count >= self.max_steps)

        reward = self._step_reward(a, u)
        if terminated:
            reward += self._terminal_reward(term)

        info = self._get_info()
        info.update(
            {
                "termination_reason": term.reason,
                "landed": bool(term.landed),
                "crashed": bool(term.crashed),
                "out_of_bounds": bool(term.oob),
            }
        )

        self.prev_action = a.copy()
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

