"""
EDF thrust model.

Implements vehicle.md §6.1 and tracker Stage 3:
- Static thrust curve: T = k_thrust * omega_fan^2 and inverse omega_fan = sqrt(T/k)
- First-order motor lag in thrust: T_dot = (T_cmd - T) / tau_motor
- Ground effect: T_eff = T * (1 + 0.5 * (r_duct/h)^2), with clamp h >= 0.01 m
- Density correction: T_eff *= rho / rho_ref (rho_ref defaults to 1.225 kg/m^3)
- Force/torque about origin: F_thrust = [0, 0, T_eff], tau = r_offset x F_thrust
- Motor reaction torque: tau_motor = -I_fan * omega_fan_dot * [0, 0, 1]

Frames / conventions
--------------------
- Body frame is FRD (Forward-Right-Down).
- By convention in this project plan, thrust force is along +body-z:
    F_thrust = [0, 0, T_eff]
  (Sign conventions are handled consistently at the full vehicle assembly stage.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


def _as_vec3(x: Sequence[float], *, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}.")
    return arr


@dataclass(frozen=True, slots=True)
class ThrustModelConfig:
    """Immutable configuration for ThrustModel."""

    k_thrust: float  # N/(rad/s)^2
    tau_motor: float  # s
    r_thrust: np.ndarray  # (3,) m, thrust application point (body frame)
    r_duct: float  # m, duct radius for ground effect
    I_fan: float  # kg*m^2, rotor MoI about spin axis
    rho_ref: float = 1.225  # kg/m^3
    T_max: float | None = None  # N, optional clamp (e.g., max_static_thrust)
    k_torque: float = 0.0  # N·m/(rad/s)^2, steady-state anti-torque coefficient
    anti_torque_enabled: bool = True  # gate for steady-state anti-torque in outputs()

    def __post_init__(self) -> None:
        if float(self.k_thrust) <= 0.0:
            raise ValueError(f"k_thrust must be > 0, got {self.k_thrust}.")
        if float(self.tau_motor) <= 0.0:
            raise ValueError(f"tau_motor must be > 0, got {self.tau_motor}.")
        if float(self.r_duct) <= 0.0:
            raise ValueError(f"r_duct must be > 0, got {self.r_duct}.")
        if float(self.I_fan) < 0.0:
            raise ValueError(f"I_fan must be >= 0, got {self.I_fan}.")
        if float(self.rho_ref) <= 0.0:
            raise ValueError(f"rho_ref must be > 0, got {self.rho_ref}.")
        if self.T_max is not None and float(self.T_max) <= 0.0:
            raise ValueError(f"T_max must be > 0 when provided, got {self.T_max}.")
        if float(self.k_torque) < 0.0:
            raise ValueError(f"k_torque must be >= 0, got {self.k_torque}.")

        object.__setattr__(self, "r_thrust", _as_vec3(self.r_thrust, name="r_thrust"))


class ThrustModel:
    """EDF thrust + actuator dynamics model.

    The model maintains an internal thrust state `T` (N). Integration is done at the
    vehicle level (Stage 7); this class provides derivatives and algebraic outputs.
    """

    def __init__(self, config: ThrustModelConfig) -> None:
        self.config = config
        self.T: float = 0.0

    @classmethod
    def from_edf_config(cls, edf: Mapping[str, Any]) -> "ThrustModel":
        anti_torque_cfg = edf.get("anti_torque", {})
        cfg = ThrustModelConfig(
            k_thrust=float(edf["k_thrust"]),
            tau_motor=float(edf["tau_motor"]),
            r_thrust=_as_vec3(edf["r_thrust"], name="r_thrust"),
            r_duct=float(edf["r_duct"]),
            I_fan=float(edf["I_fan"]),
            rho_ref=1.225,
            T_max=float(edf["max_static_thrust"]) if "max_static_thrust" in edf else None,
            k_torque=float(edf.get("k_torque", 0.0)),
            anti_torque_enabled=bool(anti_torque_cfg.get("enabled", True)),
        )
        return cls(cfg)

    def reset(self, *, T0: float = 0.0) -> None:
        """Reset internal thrust state at episode init."""
        self.T = float(max(0.0, T0))

    def _clip_T_cmd(self, T_cmd: float) -> float:
        T_cmd_f = float(T_cmd)
        if T_cmd_f < 0.0:
            T_cmd_f = 0.0
        if self.config.T_max is not None:
            T_cmd_f = float(min(T_cmd_f, float(self.config.T_max)))
        return T_cmd_f

    def thrust_from_omega(self, omega_fan: float) -> float:
        """Static thrust curve: T = k * omega^2."""
        w = float(omega_fan)
        return float(self.config.k_thrust * w * w)

    def omega_from_thrust(self, T: float) -> float:
        """Inverse thrust curve: omega = sqrt(T/k)."""
        T_pos = float(max(0.0, T))
        return float(np.sqrt(T_pos / float(self.config.k_thrust)))

    def thrust_dot(self, *, T: float, T_cmd: float) -> float:
        """First-order motor lag in thrust.

        Returns T_dot for integration (e.g., within RK4).
        """
        T_cmd_c = self._clip_T_cmd(T_cmd)
        return float((T_cmd_c - float(T)) / float(self.config.tau_motor))

    def ground_effect_factor(self, *, h: float) -> float:
        """Ground effect multiplier."""
        h_eff = float(max(0.01, h))
        ratio = float(self.config.r_duct) / h_eff
        return float(1.0 + 0.5 * ratio * ratio)

    def effective_thrust(self, *, T: float, h: float, rho: float) -> float:
        """Thrust after ground effect + density correction."""
        ge = self.ground_effect_factor(h=h)
        dens = float(rho) / float(self.config.rho_ref)
        return float(float(T) * ge * dens)

    def thrust_force(self, *, T_eff: float) -> np.ndarray:
        """Return thrust force vector in body frame."""
        return np.array([0.0, 0.0, float(T_eff)], dtype=float)

    def thrust_torque(self, *, r_offset: Sequence[float], F_thrust: Sequence[float]) -> np.ndarray:
        """Torque from thrust force about origin: tau = r x F."""
        r = _as_vec3(r_offset, name="r_offset")
        F = _as_vec3(F_thrust, name="F_thrust")
        return np.cross(r, F).astype(float)

    def motor_reaction_torque(self, *, T: float, T_dot: float) -> np.ndarray:
        """Motor reaction torque from fan spin-up/down.

        tau_motor = -I_fan * omega_dot * e_z, with omega derived from thrust curve.
        """
        omega = self.omega_from_thrust(T)
        omega_safe = max(omega, 1e-6)
        omega_dot = float(T_dot) / (2.0 * float(self.config.k_thrust) * omega_safe)
        return np.array([0.0, 0.0, -float(self.config.I_fan) * omega_dot], dtype=float)

    def steady_state_anti_torque(self, *, T: float) -> np.ndarray:
        """Steady-state fan anti-torque about body +z."""
        omega = self.omega_from_thrust(T)
        tau_z = -float(self.config.k_torque) * omega * omega
        return np.array([0.0, 0.0, tau_z], dtype=float)

    def outputs(self, *, T: float, T_cmd: float, h: float, rho: float) -> tuple[np.ndarray, np.ndarray, float]:
        """Compute thrust force, total torque, and T_dot.

        Total torque includes:
        - Offset moment from force application point
        - Motor reaction torque due to fan angular acceleration
        """
        T_dot = self.thrust_dot(T=T, T_cmd=T_cmd)
        T_eff = self.effective_thrust(T=T, h=h, rho=rho)
        F = self.thrust_force(T_eff=T_eff)
        tau_offset = self.thrust_torque(r_offset=self.config.r_thrust, F_thrust=F)
        tau_reaction = self.motor_reaction_torque(T=T, T_dot=T_dot)
        tau_anti = (
            self.steady_state_anti_torque(T=T)
            if self.config.anti_torque_enabled
            else np.zeros(3, dtype=float)
        )
        return F, (tau_offset + tau_reaction + tau_anti), float(T_dot)

