"""
parts_registry.py — Single source of truth for drone prim names and fin configuration.

Both the USD builders/post-processors and the IsaacLab task code import from here
to ensure naming stays consistent. Part names defined here must match what you
name objects in Blender (see BLENDER_EXPORT_GUIDE.md).

Blender object name → USD prim path mapping:

  Articulated parts (direct children of Drone — get joints from postprocess_usd):
    Drone        →  /Drone             (root empty)
    Fin_1..4     →  /Drone/Fin_N       (fin mesh, origin at hinge point)

  Rigid body parts (children of Body — no joints, grouped for rigid physics):
    Body         →  /Drone/Body
    edf_drone    →  /Drone/Body/edf_drone  # combined duct + fan + legs mesh

  Note (drone_v2): Legs are modeled as part of the main body geometry in ``edf_drone`` —
  there is no separate ``Legs`` prim in the Blender-exported USD.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


# ---------------------------------------------------------------------------
# FRD → Z-up coordinate conversion
# ---------------------------------------------------------------------------
def frd_to_zup(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Convert FRD body-frame coordinates to Z-up world coordinates.

    FRD +X (fwd) → world +X,  FRD +Y (right) → world -Y,  FRD +Z (down) → world -Z.
    """
    return (x, -y, -z)


def zup_to_frd(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Convert Z-up world coordinates to FRD body-frame coordinates.

    Inverse of frd_to_zup(). The mapping is its own inverse:
    World +X → FRD +X,  World +Y → FRD -Y (left),  World +Z → FRD -Z (up).
    """
    return (x, -y, -z)


def reconstruct_inertia_tensor(
    diagonal: tuple[float, float, float],
    principal_axes_quat: tuple[float, float, float, float],
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    """Reconstruct full 3x3 inertia tensor from USD diagonal + principal axes quaternion.

    USD Physics stores inertia as principal moments (diagonal) and the rotation
    of the principal axes frame relative to the prim frame (quaternion).
    Converts to a full symmetric 3x3 tensor: I = R @ diag(I') @ R^T.

    Args:
        diagonal: Principal moments (Ixx', Iyy', Izz') in principal axes frame (kg·m²).
        principal_axes_quat: Quaternion (qx, qy, qz, qw), scalar-last. If zero-magnitude,
            treated as identity (principal axes == prim frame axes).

    Returns:
        3x3 inertia tensor as nested tuples (row-major) in prim frame (kg·m²).
    """
    import math

    Ixx_p, Iyy_p, Izz_p = diagonal
    qx, qy, qz, qw = principal_axes_quat

    mag = math.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    if mag < 1e-9:
        return (
            (Ixx_p, 0.0, 0.0),
            (0.0, Iyy_p, 0.0),
            (0.0, 0.0, Izz_p),
        )
    qx, qy, qz, qw = qx / mag, qy / mag, qz / mag, qw / mag

    # Rotation matrix from quaternion (scalar-last)
    R = [
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)],
    ]
    D = [[Ixx_p, 0.0, 0.0], [0.0, Iyy_p, 0.0], [0.0, 0.0, Izz_p]]

    def _mm(A: list, B: list) -> list:
        return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

    I = _mm(_mm(R, D), [[R[j][i] for j in range(3)] for i in range(3)])  # R @ D @ R^T
    return (
        (I[0][0], I[0][1], I[0][2]),
        (I[1][0], I[1][1], I[1][2]),
        (I[2][0], I[2][1], I[2][2]),
    )


# ---------------------------------------------------------------------------
# Prim path constants
# ---------------------------------------------------------------------------
DRONE_ROOT  = "/Drone"
BODY_PRIM   = "/Drone/Body"

# Body sub-parts: Blender object names that should live under /Drone/Body.
# Only MVP parts are listed; add more as the model grows.
# In the current drone_v2 asset, the combined EDF, frame, and legs geometry is a single
# mesh named ``edf_drone`` parented under ``Body``.
BODY_PARTS_MVP = ["edf_drone"]


# ---------------------------------------------------------------------------
# Fin paths
# ---------------------------------------------------------------------------
def fin_prim_path(index: int) -> str:
    """USD prim path for fin at 1-based index (1–4)."""
    return f"/Drone/Fin_{index}"


def fin_joint_path(index: int) -> str:
    """USD prim path for fin joint at 1-based index (1–4).

    In the manual Isaac Sim scene, joints are children of each fin Xform:
      /Drone/Fin_1/Fin_1_Joint
    IsaacLab resolves joints by the leaf name: find_joints(["Fin_1_Joint", ...])
    """
    return f"/Drone/Fin_{index}/Fin_{index}_Joint"


def expected_fin_prim_paths(n: int = 4) -> list[str]:
    return [fin_prim_path(i) for i in range(1, n + 1)]


def expected_joint_paths(n: int = 4) -> list[str]:
    return [fin_joint_path(i) for i in range(1, n + 1)]


# ---------------------------------------------------------------------------
# Full MVP validation set
# ---------------------------------------------------------------------------
def expected_mvp_prim_paths(num_fins: int = 4) -> dict[str, list[str]]:
    """Return all expected prim paths grouped by category.

    Returns dict with keys 'required' (block export) and 'recommended' (warn only).
    """
    return {
        "required": [
            DRONE_ROOT,
            BODY_PRIM,
            *expected_fin_prim_paths(num_fins),
        ],
        # Recommended prims are non-blocking; for drone_v2 the only MVP sub-part
        # we expect is /Drone/Body/edf_drone.
        "recommended": [
            *[f"{BODY_PRIM}/{p}" for p in BODY_PARTS_MVP],
        ],
    }


# ---------------------------------------------------------------------------
# FinSpec dataclass
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FinSpec:
    """Configuration for one fin control surface."""

    prim_name: str                             # "Fin_1", "Fin_2", etc.
    hinge_pos_frd: tuple[float, float, float]  # hinge position in FRD body frame (m)
    hinge_axis: str                            # USD joint axis token: "X", "Y", or "Z" (fin-prim local)
    lift_direction: tuple[float, float, float] # unit vector in FRD body frame


def _axis_vec_to_token(axis_vec: Sequence[float]) -> str:
    """Convert a unit-vector axis [1,0,0] to USD axis token "X"/"Y"/"Z"."""
    ax = [int(v) for v in axis_vec]
    if ax == [1, 0, 0]:
        return "X"
    elif ax == [0, 1, 0]:
        return "Y"
    elif ax == [0, 0, 1]:
        return "Z"
    else:
        raise ValueError(
            f"Non-axis-aligned hinge_axis not supported: {list(axis_vec)}. "
            "Only [1,0,0], [0,1,0], [0,0,1] are valid."
        )


@dataclass(frozen=True)
class ExplicitMassProps:
    """Explicit vehicle mass properties from config (no primitive aggregation)."""

    total_mass: float                              # kg
    center_of_mass_frd: tuple[float, float, float] # body frame (FRD), m
    inertia_tensor: tuple[                         # 3×3 about CoM, body frame, kg·m²
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]


def load_explicit_mass_props(vehicle_cfg: dict) -> ExplicitMassProps:
    """Load explicit mass properties from vehicle config.

    Reads from ``vehicle_cfg["mass_properties"]``. Raises KeyError if the
    section is missing or ``use_explicit`` is false.
    """
    mp = vehicle_cfg.get("mass_properties", {})
    if not mp.get("use_explicit", False):
        raise KeyError(
            "mass_properties.use_explicit is false or missing. "
            "Set it to true or fall back to compute_mass_properties()."
        )
    com = tuple(float(v) for v in mp["center_of_mass"])
    inertia = tuple(
        tuple(float(v) for v in row) for row in mp["inertia_tensor"]
    )
    return ExplicitMassProps(
        total_mass=float(mp["total_mass"]),
        center_of_mass_frd=com,  # type: ignore[arg-type]
        inertia_tensor=inertia,  # type: ignore[arg-type]
    )


def load_fin_specs(vehicle_cfg: dict) -> list[FinSpec]:
    """Load fin specs from a vehicle config dict (the 'vehicle' sub-dict).

    Args:
        vehicle_cfg: The 'vehicle' section of default_vehicle.yaml (already extracted).

    Returns:
        List of FinSpec, one per fin in order (Fin_1 … Fin_N).

    Hinge axis: If ``hinge_axis_usd`` is set per fin (string "X", "Y", or "Z"), it is
    used as the RevoluteJoint axis in USD (fin-prim local frame). Otherwise the axis
    is derived from ``hinge_axis`` (body-frame vector). Set ``hinge_axis_usd`` when the
    fin prim's local axes in Isaac Sim differ from body FRD so the joint rotates about
    the correct local axis.
    """
    fins_cfg = vehicle_cfg.get("fins", {})
    fins_config = fins_cfg.get("fins_config", [])

    specs: list[FinSpec] = []
    for i, fin in enumerate(fins_config):
        prim_name = f"Fin_{i + 1}"
        pos = tuple(float(v) for v in fin["position"])
        axis_token = fin.get("hinge_axis_usd", "").strip().upper()
        if axis_token not in ("X", "Y", "Z"):
            axis_token = _axis_vec_to_token(fin["hinge_axis"])
        lift_dir = tuple(float(v) for v in fin["lift_direction"])
        specs.append(FinSpec(
            prim_name=prim_name,
            hinge_pos_frd=pos,       # type: ignore[arg-type]
            hinge_axis=axis_token,
            lift_direction=lift_dir, # type: ignore[arg-type]
        ))

    return specs
