"""
RK4 integrator utilities.

Stage 7 — RK4 Integrator
------------------------
Implements:
- `rk4_step(f, y, u, t, dt)` : generic fixed-step 4th-order Runge-Kutta.
- `RK4Integrator` : small stateful wrapper that optionally re-normalizes a
  quaternion slice in the state every N steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, TypeVar

import numpy as np

from .quaternion_utils import quat_normalize

Y = TypeVar("Y", bound=np.ndarray)


DerivFn = Callable[[np.ndarray, object, float], np.ndarray]


def rk4_step(f: DerivFn, y: np.ndarray, u: object, t: float, dt: float) -> np.ndarray:
    """Take a single fixed-step RK4 step.

    Parameters
    ----------
    f:
        Derivative function, `dy = f(y, u, t)`.
    y:
        Current state array (any shape).
    u:
        Control / input passed through to `f` (may be None).
    t:
        Current time.
    dt:
        Integration step (seconds).

    Returns
    -------
    np.ndarray
        Next state `y_next` with same shape as `y`.
    """
    y0 = np.asarray(y, dtype=float)

    k1 = np.asarray(f(y0, u, t), dtype=float)
    k2 = np.asarray(f(y0 + 0.5 * dt * k1, u, t + 0.5 * dt), dtype=float)
    k3 = np.asarray(f(y0 + 0.5 * dt * k2, u, t + 0.5 * dt), dtype=float)
    k4 = np.asarray(f(y0 + dt * k3, u, t + dt), dtype=float)

    return y0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@dataclass
class RK4Integrator:
    """RK4 integrator with optional periodic quaternion normalization.

    Notes
    -----
    By default, assumes the vehicle state layout:
    `[p(3), v_b(3), q(4), omega(3), T(1), delta_actual(4)]`
    so the quaternion is at indices 6:10.
    """

    quat_slice: Optional[slice] = field(default_factory=lambda: slice(6, 10))
    quat_normalize_every_n: int = 10
    step_count: int = field(default=0, init=False)

    def reset(self) -> None:
        """Reset the internal step counter."""
        self.step_count = 0

    def step(self, f: DerivFn, y: np.ndarray, u: object, t: float, dt: float) -> np.ndarray:
        """Advance one RK4 step and optionally normalize quaternion state."""
        y_next = rk4_step(f, y, u, t, dt)

        self.step_count += 1

        if self.quat_slice is None:
            return y_next

        if self.quat_normalize_every_n <= 0:
            raise ValueError("quat_normalize_every_n must be >= 1 when quat_slice is set.")

        if (self.step_count % self.quat_normalize_every_n) != 0:
            return y_next

        q = np.asarray(y_next[self.quat_slice], dtype=float)
        if q.shape != (4,):
            raise ValueError(
                f"Quaternion slice must select shape (4,), got {q.shape}. "
                f"Check quat_slice={self.quat_slice!r}."
            )

        y_norm = np.array(y_next, copy=True)
        y_norm[self.quat_slice] = quat_normalize(q)
        return y_norm

