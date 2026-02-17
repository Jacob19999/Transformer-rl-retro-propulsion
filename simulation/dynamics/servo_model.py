"""
Servo actuator dynamics model for 4 fin servos.

Implements vehicle.md §6.3.6 and tracker Stage 6:
- Rate-limited first-order lag per fin:
    delta_dot = clip((delta_cmd - delta_actual) / tau_servo,
                     -rate_max_eff, +rate_max_eff)
- Aerodynamic-load derating applied to max slew:
    rate_max_eff = rate_max * (1 - derating), derating ~ Uniform[0.2, 0.5] per episode

This module is intentionally lightweight and standalone so it can be used both:
- as a stateful helper (via `ServoModel.step`) for simple experiments, and
- as a pure rate provider (via `ServoModel.compute_rate`) inside the full RK4 vehicle dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


def _as_1d(x: Sequence[float], *, name: str, n: int) -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    if arr.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},), got {arr.shape}.")
    return arr


@dataclass(frozen=True, slots=True)
class ServoModelConfig:
    """Immutable configuration for ServoModel."""

    n_fins: int
    tau_servo: float  # s, nominal position lag time constant
    tau_servo_range: tuple[float, float]  # s, domain randomization range [min, max]
    rate_max: float  # rad/s, no-load max angular velocity
    aero_load_derating: float  # fraction in [0,1), nominal effective rate reduction
    derating_range: tuple[float, float] = (0.2, 0.5)  # per-episode DR range

    def __post_init__(self) -> None:
        if int(self.n_fins) <= 0:
            raise ValueError(f"n_fins must be > 0, got {self.n_fins}.")
        if float(self.tau_servo) <= 0.0:
            raise ValueError(f"tau_servo must be > 0, got {self.tau_servo}.")
        lo, hi = (float(self.tau_servo_range[0]), float(self.tau_servo_range[1]))
        if lo <= 0.0 or hi <= 0.0 or lo > hi:
            raise ValueError(f"tau_servo_range must be positive and ordered, got {self.tau_servo_range}.")
        if float(self.rate_max) <= 0.0:
            raise ValueError(f"rate_max must be > 0, got {self.rate_max}.")

        d0 = float(self.aero_load_derating)
        if not (0.0 <= d0 < 1.0):
            raise ValueError(f"aero_load_derating must be in [0, 1), got {self.aero_load_derating}.")

        dlo, dhi = (float(self.derating_range[0]), float(self.derating_range[1]))
        if not (0.0 <= dlo <= dhi < 1.0):
            raise ValueError(f"derating_range must be in [0,1) and ordered, got {self.derating_range}.")

        object.__setattr__(self, "n_fins", int(self.n_fins))
        object.__setattr__(self, "tau_servo_range", (lo, hi))
        object.__setattr__(self, "derating_range", (dlo, dhi))

    @classmethod
    def from_config(cls, fins: Mapping[str, Any]) -> "ServoModelConfig":
        """Build config from the `vehicle.fins` YAML section."""
        fins_d = dict(fins)
        servo = dict(fins_d.get("servo", {}))

        n_fins = int(fins_d.get("count", 4))
        tau_servo = float(servo.get("tau_servo", 0.04))
        tau_range_raw = servo.get("tau_servo_range", [tau_servo, tau_servo])
        tau_range = (float(tau_range_raw[0]), float(tau_range_raw[1]))

        # Prefer servo.max_angular_velocity when present; otherwise fall back to fins.rate_limit.
        rate_max = float(servo.get("max_angular_velocity", fins_d.get("rate_limit", 10.5)))

        aero_load_derating = float(servo.get("aero_load_derating", 0.0))

        return cls(
            n_fins=n_fins,
            tau_servo=tau_servo,
            tau_servo_range=tau_range,
            rate_max=rate_max,
            aero_load_derating=aero_load_derating,
            derating_range=(0.2, 0.5),
        )


class ServoModel:
    """Rate-limited first-order lag for fin servo actuators."""

    def __init__(self, config: ServoModelConfig) -> None:
        self.config = config
        self.n_fins = int(config.n_fins)

        # Episode-varying parameters (tau and derating may be randomized in reset()).
        self.tau: float = float(config.tau_servo)
        self.derating: float = float(config.aero_load_derating)

        # Stateful servo positions for convenience (VehicleDynamics will integrate these as states).
        self.delta_actual: np.ndarray = np.zeros(self.n_fins, dtype=float)

    @classmethod
    def from_config(cls, fins: Mapping[str, Any]) -> "ServoModel":
        """Build ServoModel from the `vehicle.fins` YAML section."""
        return cls(ServoModelConfig.from_config(fins))

    def reset(self, seed: int | None = None) -> None:
        """Reset servo positions; optionally randomize tau + derating for domain randomization."""
        self.delta_actual = np.zeros(self.n_fins, dtype=float)

        # Restore nominal values first; DR only applies when seed is provided.
        self.tau = float(self.config.tau_servo)
        self.derating = float(self.config.aero_load_derating)

        if seed is None:
            return

        rng = np.random.default_rng(int(seed))
        tau_lo, tau_hi = self.config.tau_servo_range
        self.tau = float(rng.uniform(tau_lo, tau_hi))

        d_lo, d_hi = self.config.derating_range
        self.derating = float(rng.uniform(d_lo, d_hi))

    def rate_max_eff(self) -> float:
        """Effective max angular velocity after derating."""
        return float(max(0.0, float(self.config.rate_max) * (1.0 - float(self.derating))))

    def compute_rate(self, delta_cmd: Sequence[float], delta_actual: Sequence[float]) -> np.ndarray:
        """Compute delta_dot from commanded/actual deflections.

        Parameters
        ----------
        delta_cmd
            Commanded fin deflections (n_fins,) rad.
        delta_actual
            Current physical fin deflections (n_fins,) rad.

        Returns
        -------
        delta_dot
            Fin deflection rates (n_fins,) rad/s.
        """
        cmd = _as_1d(delta_cmd, name="delta_cmd", n=self.n_fins)
        act = _as_1d(delta_actual, name="delta_actual", n=self.n_fins)

        error = cmd - act
        rate_desired = error / float(self.tau)
        rmax = self.rate_max_eff()
        delta_dot = np.clip(rate_desired, -rmax, rmax)
        return delta_dot.astype(float)

    def step(self, delta_cmd: Sequence[float], dt: float) -> np.ndarray:
        """Euler step update of internal servo positions for standalone usage/testing."""
        delta_dot = self.compute_rate(delta_cmd, self.delta_actual)
        self.delta_actual = self.delta_actual + delta_dot * float(dt)
        return self.delta_actual.copy()

