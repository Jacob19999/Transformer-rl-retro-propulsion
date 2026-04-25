"""
Hinge axis extraction from USD revolute joint prims.

Extracts revolute joint axes from USD PhysicsRevoluteJoint prims,
validates they are unit vectors along cardinal directions, and compares
against the hinge_axes field in asset metadata YAML.
"""

from __future__ import annotations
import math
from typing import Any


_CARDINAL_AXES = {
    (1, 0, 0): "+X",
    (-1, 0, 0): "-X",
    (0, 1, 0): "+Y",
    (0, -1, 0): "-Y",
    (0, 0, 1): "+Z",
    (0, 0, -1): "-Z",
}


def extract_joint_axis_from_usd(stage, joint_path: str) -> list[float]:
    """Extract the revolute joint axis from a USD PhysicsRevoluteJoint prim.

    Args:
        stage: USD Stage object.
        joint_path: Full USD prim path to the revolute joint.

    Returns:
        Unit vector [x, y, z] representing the joint axis in local frame.

    Raises:
        ImportError: If USD/Isaac Sim is not available.
        ValueError: If prim is not a valid RevoluteJoint or axis is not set.
    """
    try:
        import pxr.UsdPhysics as UsdPhysics
    except ImportError as e:
        raise ImportError("Isaac Sim runtime required for USD prim access.") from e

    prim = stage.GetPrimAtPath(joint_path)
    if not prim.IsValid():
        raise ValueError(f"Joint prim not found at path: {joint_path}")

    joint_api = UsdPhysics.RevoluteJoint(prim)
    axis_attr = joint_api.GetAxisAttr()
    if not axis_attr.IsValid():
        raise ValueError(f"No axis attribute on joint: {joint_path}")

    axis_token = axis_attr.Get()
    axis_map = {"X": [1, 0, 0], "Y": [0, 1, 0], "Z": [0, 0, 1]}
    if axis_token not in axis_map:
        raise ValueError(f"Unexpected axis token '{axis_token}' at {joint_path}")
    return axis_map[axis_token]


def validate_hinge_axes_against_metadata(
    usd_axes: list[list[float]],
    metadata: dict[str, Any],
    atol: float = 1e-4,
) -> list[str]:
    """Compare extracted USD joint axes against metadata YAML hinge_axes.

    Args:
        usd_axes: List of 3-vectors extracted from USD joints (one per fin).
        metadata: Asset metadata dict containing 'hinge_axes'.
        atol: Absolute tolerance for component comparison.

    Returns:
        List of warning strings (empty if all match).
    """
    warnings = []
    expected_axes = metadata.get("hinge_axes", [])
    for i, (usd_axis, expected_axis) in enumerate(zip(usd_axes, expected_axes)):
        for j, (u, e) in enumerate(zip(usd_axis, expected_axis)):
            if abs(u - e) > atol:
                warnings.append(
                    f"Fin {i}: USD axis component [{j}] = {u:.4f}, "
                    f"metadata = {e:.4f} (diff={abs(u-e):.4f} > atol={atol})"
                )
    return warnings


def is_unit_vector(v: list[float], atol: float = 1e-4) -> bool:
    """Check if a 3-vector is approximately unit length."""
    norm = math.sqrt(sum(x * x for x in v))
    return abs(norm - 1.0) < atol


def classify_cardinal(v: list[float], atol: float = 0.1) -> str | None:
    """Classify a vector as a cardinal direction or return None."""
    for cardinal, label in _CARDINAL_AXES.items():
        if all(abs(v[i] - cardinal[i]) < atol for i in range(3)):
            return label
    return None
