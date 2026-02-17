"""
Reward function for EDFLandingEnv (Stage 14).

Implements training.md §5:
  r_t = r_alive + r_shape + r_orient + r_jerk + r_fuel + r_action + 1_terminal * r_terminal
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


def _as_vec(x: object, n: int, *, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size != n:
        raise ValueError(f"{name} must have shape ({n},), got {arr.shape}.")
    return arr


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Configuration for RewardFunction (training.md §15.2)."""

    # Step rewards
    alive_bonus: float = 0.1
    shaping_distance_coeff: float = 1.0  # c_d
    shaping_velocity_coeff: float = 0.2  # c_v
    shaping_gamma: float = 0.99
    orientation_weight: float = 0.5  # w_theta
    jerk_weight: float = 0.05  # w_j
    jerk_reference: float = 10.0  # j_ref
    fuel_weight: float = 0.01  # w_f
    action_smooth_weight: float = 0.02  # w_a

    # Terminal rewards/penalties
    landing_success: float = 100.0  # R_land
    precision_bonus: float = 50.0  # R_prec
    precision_sigma: float = 0.1  # sigma_prec (m)
    soft_touchdown: float = 20.0  # R_soft
    crash_penalty: float = 100.0  # R_crash
    oob_penalty: float = 50.0  # R_oob

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "RewardConfig":
        """Parse reward config.

        Supports:
        - training.md §15.2 nested schema
        - backward-compatible flat keys used in early Stage 12 env reward code
        """
        cfg = dict(config or {})
        defaults = cls()

        # Preferred schema (training.md §15.2). Allow partial configs that only
        # override terminal weights (e.g. {'crash_penalty': 123.0}).
        new_schema_keys = {
            "alive_bonus",
            "shaping",
            "orientation_weight",
            "jerk_weight",
            "jerk_reference",
            "fuel_weight",
            "action_smooth_weight",
            "landing_success",
            "precision_bonus",
            "precision_sigma",
            "soft_touchdown",
            "crash_penalty",
            "oob_penalty",
        }
        use_new_schema = ("shaping" in cfg) or any(k in cfg for k in (new_schema_keys - {"shaping"}))
        shaping = dict(cfg.get("shaping", {}) or {})
        if use_new_schema:
            return cls(
                alive_bonus=float(cfg.get("alive_bonus", defaults.alive_bonus)),
                shaping_distance_coeff=float(
                    shaping.get("distance_coeff", defaults.shaping_distance_coeff)
                ),
                shaping_velocity_coeff=float(
                    shaping.get("velocity_coeff", defaults.shaping_velocity_coeff)
                ),
                shaping_gamma=float(shaping.get("gamma", defaults.shaping_gamma)),
                orientation_weight=float(
                    cfg.get("orientation_weight", defaults.orientation_weight)
                ),
                jerk_weight=float(cfg.get("jerk_weight", defaults.jerk_weight)),
                jerk_reference=float(cfg.get("jerk_reference", defaults.jerk_reference)),
                fuel_weight=float(cfg.get("fuel_weight", defaults.fuel_weight)),
                action_smooth_weight=float(
                    cfg.get("action_smooth_weight", defaults.action_smooth_weight)
                ),
                landing_success=float(cfg.get("landing_success", defaults.landing_success)),
                precision_bonus=float(cfg.get("precision_bonus", defaults.precision_bonus)),
                precision_sigma=float(cfg.get("precision_sigma", defaults.precision_sigma)),
                soft_touchdown=float(cfg.get("soft_touchdown", defaults.soft_touchdown)),
                crash_penalty=float(cfg.get("crash_penalty", defaults.crash_penalty)),
                oob_penalty=float(cfg.get("oob_penalty", defaults.oob_penalty)),
            )

        # Backward compatible schema (Stage 12 in-env reward fields)
        return cls(
            alive_bonus=float(cfg.get("alive_bonus", defaults.alive_bonus)),
            shaping_distance_coeff=float(cfg.get("c_d", defaults.shaping_distance_coeff)),
            shaping_velocity_coeff=float(cfg.get("c_v", defaults.shaping_velocity_coeff)),
            shaping_gamma=float(cfg.get("gamma", defaults.shaping_gamma)),
            orientation_weight=float(cfg.get("w_theta", defaults.orientation_weight)),
            jerk_weight=float(cfg.get("w_j", defaults.jerk_weight)),
            jerk_reference=float(cfg.get("j_ref", defaults.jerk_reference)),
            fuel_weight=float(cfg.get("w_f", defaults.fuel_weight)),
            action_smooth_weight=float(cfg.get("w_a", defaults.action_smooth_weight)),
            landing_success=float(cfg.get("R_land", defaults.landing_success)),
            precision_bonus=float(cfg.get("R_prec", defaults.precision_bonus)),
            precision_sigma=float(cfg.get("sigma_prec", defaults.precision_sigma)),
            soft_touchdown=float(cfg.get("R_soft", defaults.soft_touchdown)),
            crash_penalty=float(cfg.get("R_crash", defaults.crash_penalty)),
            oob_penalty=float(cfg.get("R_oob", defaults.oob_penalty)),
        )


class RewardFunction:
    """Stateful reward function (potential shaping + smoothness penalties)."""

    def __init__(self, config: RewardConfig | Mapping[str, Any] | None = None) -> None:
        self.config = config if isinstance(config, RewardConfig) else RewardConfig.from_config(config)
        self.reset()

    def reset(self) -> None:
        """Reset episode-dependent reward state (tracker 14.10)."""
        self.prev_potential: float | None = None
        self.prev_velocity: np.ndarray | None = None
        self.prev_accel: np.ndarray | None = None
        self.prev_action: np.ndarray | None = None
        self.step_count: int = 0

    def _potential(self, *, p: np.ndarray, v_b: np.ndarray, p_target: np.ndarray) -> float:
        e_p = _as_vec(p_target, 3, name="p_target") - _as_vec(p, 3, name="p")
        v_b = _as_vec(v_b, 3, name="v_b")
        c_d = float(self.config.shaping_distance_coeff)
        c_v = float(self.config.shaping_velocity_coeff)
        return float(-c_d * np.linalg.norm(e_p) - c_v * np.linalg.norm(v_b))

    def step_reward(
        self,
        *,
        p: np.ndarray,
        v_b: np.ndarray,
        R_body_to_inertial: np.ndarray,
        p_target: np.ndarray,
        action: np.ndarray,
        T_cmd: float,
        T_max: float,
        dt_policy: float,
    ) -> float:
        """Compute step reward after the environment has advanced to s_{t+1}."""
        p = _as_vec(p, 3, name="p")
        v_b = _as_vec(v_b, 3, name="v_b")
        p_target = _as_vec(p_target, 3, name="p_target")
        a = np.asarray(action, dtype=float).reshape(-1)
        R = np.asarray(R_body_to_inertial, dtype=float).reshape(3, 3)

        # Alive bonus
        r_alive = float(self.config.alive_bonus)

        # Potential-based shaping (training.md §5.3.2)
        phi = self._potential(p=p, v_b=v_b, p_target=p_target)
        if self.prev_potential is None:
            r_shape = 0.0
        else:
            gamma = float(self.config.shaping_gamma)
            r_shape = gamma * phi - float(self.prev_potential)
        self.prev_potential = phi

        # Orientation penalty (training.md §5.3.3)
        g_body = R.T @ np.array([0.0, 0.0, 1.0], dtype=float)
        g_body_z = float(g_body[2])
        r_orient = -float(self.config.orientation_weight) * (1.0 - g_body_z)

        # Jerk penalty (training.md §5.3.4), skip first 2 policy steps
        r_jerk = 0.0
        if self.prev_velocity is not None:
            accel = (v_b - self.prev_velocity) / max(float(dt_policy), 1e-9)
            if self.prev_accel is not None and self.step_count >= 2:
                jerk = float(np.linalg.norm((accel - self.prev_accel) / max(float(dt_policy), 1e-9)))
                r_jerk = -float(self.config.jerk_weight) * (jerk / max(float(self.config.jerk_reference), 1e-9))
            self.prev_accel = accel
        self.prev_velocity = v_b.copy()

        # Fuel penalty (training.md §5.3.5)
        r_fuel = -float(self.config.fuel_weight) * (abs(float(T_cmd)) / max(float(T_max), 1e-9)) * float(dt_policy)

        # Action smoothness penalty (training.md §5.3.6)
        if self.prev_action is None:
            r_action = -float(self.config.action_smooth_weight) * float(np.linalg.norm(a))
        else:
            r_action = -float(self.config.action_smooth_weight) * float(np.linalg.norm(a - self.prev_action))
        self.prev_action = a.copy()

        self.step_count += 1
        return float(r_alive + r_shape + r_orient + r_jerk + r_fuel + r_action)

    def terminal_reward(
        self,
        *,
        landed: bool,
        crashed: bool,
        out_of_bounds: bool,
        p: np.ndarray,
        v_b: np.ndarray,
        R_body_to_inertial: np.ndarray,
        p_target: np.ndarray,
        v_max_touchdown: float,
    ) -> float:
        """Terminal reward applied once at episode end (training.md §5.4)."""
        if landed:
            p = _as_vec(p, 3, name="p")
            v_b = _as_vec(v_b, 3, name="v_b")
            p_target = _as_vec(p_target, 3, name="p_target")
            R = np.asarray(R_body_to_inertial, dtype=float).reshape(3, 3)

            e_xy = p[:2] - p_target[:2]
            cep = float(np.linalg.norm(e_xy))
            v_touch = float(np.linalg.norm(R @ v_b))

            sigma = max(float(self.config.precision_sigma), 1e-9)
            r_prec = float(self.config.precision_bonus) * float(
                np.exp(-(cep * cep) / (2.0 * sigma * sigma))
            )
            v_max = max(float(v_max_touchdown), 1e-9)
            r_soft = float(self.config.soft_touchdown) * max(0.0, (1.0 - v_touch / v_max))
            return float(self.config.landing_success + r_prec + r_soft)

        if crashed:
            return float(-self.config.crash_penalty)
        if out_of_bounds:
            return float(-self.config.oob_penalty)
        return 0.0


__all__ = ["RewardConfig", "RewardFunction"]

