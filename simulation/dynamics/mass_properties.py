"""
Mass properties aggregation from geometric primitives.

Implements vehicle.md §5:
- Primitive inertia formulas for cylinder/box/sphere
- Composite mass, center of mass, and inertia via parallel axis theorem
- Aerodynamic bookkeeping (total surface area + per-axis projected areas)
- Optional per-episode mass randomization via `randomize_mass` field
- CAD override via `MassProperties.from_cad`

Frames / conventions
--------------------
- Body frame is FRD (Forward-Right-Down).
- Primitive `position` is expressed in body coordinates (meters).
- Primitive `orientation` is [roll, pitch, yaw] in degrees, applied as an intrinsic
  yaw(z) → pitch(y) → roll(x) rotation. The resulting DCM R maps vectors from the
  primitive's local frame to the body frame. Inertia transforms as:
      I_body = R @ I_local @ R.T
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


def _euler_deg_to_dcm(orientation_deg: Sequence[float]) -> np.ndarray:
    """Return DCM R = Rz(yaw) @ Ry(pitch) @ Rx(roll) with angles in degrees."""
    roll_deg, pitch_deg, yaw_deg = _as_vec3(orientation_deg, name="orientation_deg")
    roll, pitch, yaw = np.deg2rad([roll_deg, pitch_deg, yaw_deg])

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    # Rx(roll)
    Rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cr, -sr],
            [0.0, sr, cr],
        ],
        dtype=float,
    )
    # Ry(pitch)
    Ry = np.array(
        [
            [cp, 0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp],
        ],
        dtype=float,
    )
    # Rz(yaw)
    Rz = np.array(
        [
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return Rz @ Ry @ Rx


def _primitive_inertia(prim: Mapping[str, Any], *, mass: float) -> np.ndarray:
    """Return a 3x3 inertia tensor about the primitive's own CoM (local axes)."""
    shape = prim.get("shape", None)
    if not isinstance(shape, str):
        raise ValueError("Primitive 'shape' must be a string.")
    shape = shape.lower()

    m = float(mass)
    if m <= 0.0:
        raise ValueError(f"Primitive mass must be > 0, got {m}.")

    if shape == "cylinder":
        r = float(prim["radius"])
        h = float(prim["height"])
        if r <= 0.0 or h <= 0.0:
            raise ValueError(f"Cylinder radius/height must be > 0, got r={r}, h={h}.")
        Ixx = (1.0 / 12.0) * m * (3.0 * r * r + h * h)
        Iyy = Ixx
        Izz = 0.5 * m * r * r
        return np.diag([Ixx, Iyy, Izz]).astype(float)

    if shape == "box":
        a, b, c = _as_vec3(prim["dimensions"], name="dimensions")  # x, y, z lengths
        if a <= 0.0 or b <= 0.0 or c <= 0.0:
            raise ValueError(f"Box dimensions must be > 0, got {a}, {b}, {c}.")
        Ixx = (1.0 / 12.0) * m * (b * b + c * c)
        Iyy = (1.0 / 12.0) * m * (a * a + c * c)
        Izz = (1.0 / 12.0) * m * (a * a + b * b)
        return np.diag([Ixx, Iyy, Izz]).astype(float)

    if shape == "sphere":
        r = float(prim["radius"])
        if r <= 0.0:
            raise ValueError(f"Sphere radius must be > 0, got r={r}.")
        I = (2.0 / 5.0) * m * r * r
        return np.diag([I, I, I]).astype(float)

    raise ValueError(f"Unsupported primitive shape: {shape!r}.")


@dataclass(frozen=True, slots=True)
class MassProperties:
    """Immutable aggregate mass properties about the vehicle CoM."""

    total_mass: float
    center_of_mass: np.ndarray  # (3,)
    inertia_tensor: np.ndarray  # (3,3) about CoM in body axes
    inertia_tensor_inv: np.ndarray  # (3,3)
    total_surface_area: float
    projected_area_x: float
    projected_area_y: float
    projected_area_z: float

    @classmethod
    def from_cad(cls, cad_config: Mapping[str, Any]) -> "MassProperties":
        """Create MassProperties from CAD-exported values (bypasses primitives)."""
        m = float(cad_config["total_mass"])
        com = _as_vec3(cad_config["center_of_mass"], name="center_of_mass")
        I = np.asarray(cad_config["inertia_tensor"], dtype=float)
        if I.shape != (3, 3):
            raise ValueError(f"cad inertia_tensor must be shape (3,3), got {I.shape}.")
        I_inv = np.linalg.inv(I)

        total_surface_area = float(cad_config.get("total_surface_area", 0.0) or 0.0)

        # Directional projected areas are typically not exported directly from CAD.
        return cls(
            total_mass=m,
            center_of_mass=com,
            inertia_tensor=I,
            inertia_tensor_inv=I_inv,
            total_surface_area=total_surface_area,
            projected_area_x=0.0,
            projected_area_y=0.0,
            projected_area_z=0.0,
        )


def compute_mass_properties(
    primitives: Sequence[Mapping[str, Any]],
    *,
    rng: np.random.Generator | None = None,
) -> MassProperties:
    """Aggregate mass, CoM, inertia, and aero areas from a primitive list.

    If `rng` is provided, primitives with a `randomize_mass` field will have their
    mass perturbed uniformly in ±randomize_mass (fraction) before aggregation.
    """
    if len(primitives) == 0:
        raise ValueError("primitives must be a non-empty sequence.")

    masses = np.empty((len(primitives),), dtype=float)
    positions = np.empty((len(primitives), 3), dtype=float)

    for i, p in enumerate(primitives):
        base_mass = float(p["mass"])
        frac = float(p.get("randomize_mass", 0.0) or 0.0)
        if frac < 0.0:
            raise ValueError(f"randomize_mass must be >= 0, got {frac}.")
        if rng is not None and frac > 0.0:
            delta = rng.uniform(-frac, +frac)
            m_i = base_mass * (1.0 + float(delta))
        else:
            m_i = base_mass

        masses[i] = m_i
        positions[i] = _as_vec3(p["position"], name="position")

    total_mass = float(np.sum(masses))
    if total_mass <= 0.0:
        raise ValueError(f"Total mass must be > 0, got {total_mass}.")

    center_of_mass = (masses[:, None] * positions).sum(axis=0) / total_mass

    I_total = np.zeros((3, 3), dtype=float)
    total_surface_area = 0.0
    proj_x = proj_y = proj_z = 0.0

    for m_i, p_i, p in zip(masses, positions, primitives, strict=True):
        I_local = _primitive_inertia(p, mass=float(m_i))
        orientation = p.get("orientation", None)
        if orientation is not None:
            R = _euler_deg_to_dcm(orientation)
            I_local = R @ I_local @ R.T

        d = p_i - center_of_mass
        I_parallel = float(m_i) * (float(d @ d) * np.eye(3) - np.outer(d, d))
        I_total += I_local + I_parallel

        total_surface_area += float(p.get("surface_area", 0.0) or 0.0)
        drag = p.get("drag_facing", {}) or {}
        proj_x += float(drag.get("x", 0.0) or 0.0)
        proj_y += float(drag.get("y", 0.0) or 0.0)
        proj_z += float(drag.get("z", 0.0) or 0.0)

    # Numerical symmetry guard.
    I_total = 0.5 * (I_total + I_total.T)
    I_inv = np.linalg.inv(I_total)

    return MassProperties(
        total_mass=total_mass,
        center_of_mass=center_of_mass,
        inertia_tensor=I_total,
        inertia_tensor_inv=I_inv,
        total_surface_area=float(total_surface_area),
        projected_area_x=float(proj_x),
        projected_area_y=float(proj_y),
        projected_area_z=float(proj_z),
    )

