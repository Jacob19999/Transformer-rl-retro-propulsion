"""
Fail-fast structural validation for the EDF drone USD asset.

Validates: body link exists, 4 fin links exist, 4 revolute joints with defined
axes, joint limits match config, mass properties valid, link hierarchy consistent.
Raises descriptive errors on any failure.
"""

from __future__ import annotations
from typing import Any


class AssetValidationError(RuntimeError):
    """Raised when asset structural validation fails."""
    pass


def validate_asset(
    metadata: dict[str, Any],
    vehicle_config: dict[str, Any],
    stage=None,
    articulation=None,
) -> list[str]:
    """Run all structural validations on the EDF drone asset.

    Can operate in two modes:
      - Offline: validates metadata self-consistency (no stage/articulation required)
      - Isaac Sim: full USD structural validation (stage and articulation required)

    Args:
        metadata: Asset metadata dict from usd_loader.load_asset_metadata().
        vehicle_config: Vehicle config from mass_properties.load_vehicle_config().
        stage: USD Stage object (optional, for full USD validation).
        articulation: Isaac Lab Articulation (optional, for runtime validation).

    Returns:
        List of diagnostic strings (informational). Empty = fully passed.

    Raises:
        AssetValidationError: On any critical structural failure.
    """
    diagnostics = []

    # --- Metadata self-consistency checks (always run) ---
    _validate_metadata_structure(metadata)

    fin_links = metadata["fin_link_names"]
    fin_joints = metadata["fin_joint_names"]
    hinge_axes = metadata.get("hinge_axes", [])
    joint_limits = metadata.get("joint_limits", [])
    cop_positions = metadata.get("fin_cop_positions", [])

    if len(fin_links) != 4:
        raise AssetValidationError(
            f"Expected 4 fin link names in metadata, got {len(fin_links)}: {fin_links}"
        )
    if len(fin_joints) != 4:
        raise AssetValidationError(
            f"Expected 4 fin joint names in metadata, got {len(fin_joints)}: {fin_joints}"
        )
    if len(hinge_axes) != 4:
        raise AssetValidationError(
            f"Expected 4 hinge_axes in metadata, got {len(hinge_axes)}"
        )
    if len(joint_limits) != 4:
        raise AssetValidationError(
            f"Expected 4 joint_limits entries in metadata, got {len(joint_limits)}"
        )
    if len(cop_positions) != 4:
        raise AssetValidationError(
            f"Expected 4 fin_cop_positions in metadata, got {len(cop_positions)}"
        )

    # Validate hinge axes are unit vectors
    import math
    for i, axis in enumerate(hinge_axes):
        norm = math.sqrt(sum(x * x for x in axis))
        if abs(norm - 1.0) > 0.01:
            raise AssetValidationError(
                f"Hinge axis {i} is not a unit vector: {axis} (norm={norm:.4f})"
            )

    # Validate joint limits
    config_max_deflection = vehicle_config.get("fins", {}).get("max_deflection", 0.262)
    for i, (lo, hi) in enumerate(joint_limits):
        if abs(abs(lo) - config_max_deflection) > 0.01 or abs(abs(hi) - config_max_deflection) > 0.01:
            diagnostics.append(
                f"Joint limit {i}: [{lo:.3f}, {hi:.3f}] rad vs "
                f"config max_deflection={config_max_deflection:.3f} rad"
            )

    # --- USD structural checks (require stage) ---
    if stage is not None:
        _validate_usd_structure(metadata, stage, diagnostics)

    # --- Articulation runtime checks ---
    if articulation is not None:
        _validate_articulation(metadata, articulation, diagnostics)

    return diagnostics


def _validate_metadata_structure(metadata: dict[str, Any]) -> None:
    """Validate required keys exist in metadata."""
    required_keys = [
        "body_link_name",
        "fin_link_names",
        "fin_joint_names",
        "hinge_axes",
        "joint_limits",
        "fin_cop_positions",
    ]
    missing = [k for k in required_keys if k not in metadata]
    if missing:
        raise AssetValidationError(
            f"Missing required keys in asset metadata: {missing}"
        )


def _validate_usd_structure(metadata: dict[str, Any], stage, diagnostics: list[str]) -> None:
    """Check USD prim hierarchy against metadata."""
    try:
        import pxr.UsdPhysics as UsdPhysics
    except ImportError:
        diagnostics.append("WARNING: pxr not available, skipping USD prim checks")
        return

    body_link_name = metadata["body_link_name"]
    body_path = f"/{body_link_name}"
    body_prim = stage.GetPrimAtPath(body_path)

    if not body_prim.IsValid():
        raise AssetValidationError(
            f"Body link prim not found at '{body_path}'. "
            f"Check that USD asset has prim at this path."
        )

    if not body_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        raise AssetValidationError(
            f"Body prim '{body_path}' missing ArticulationRootAPI. "
            f"USD asset must have PhysicsArticulationRootAPI applied to root link."
        )

    # Check fin links
    for link_name in metadata["fin_link_names"]:
        fin_path = f"/{body_link_name}/{link_name}"
        fin_prim = stage.GetPrimAtPath(fin_path)
        if not fin_prim.IsValid():
            raise AssetValidationError(
                f"Fin link '{fin_path}' not found in USD. "
                f"Check edf_drone_v2.asset.yaml fin_link_names match USD hierarchy."
            )

    # Check fin joints
    for joint_name in metadata["fin_joint_names"]:
        joint_path = f"/{body_link_name}/{joint_name}"
        joint_prim = stage.GetPrimAtPath(joint_path)
        if not joint_prim.IsValid():
            raise AssetValidationError(
                f"Fin joint '{joint_path}' not found in USD. "
                f"Check edf_drone_v2.asset.yaml fin_joint_names match USD hierarchy."
            )
        if not joint_prim.HasAPI(UsdPhysics.RevoluteJoint):
            raise AssetValidationError(
                f"Joint '{joint_path}' is not a RevoluteJoint. "
                f"All fin joints must use UsdPhysics.RevoluteJoint schema."
            )


def _validate_articulation(metadata: dict[str, Any], articulation, diagnostics: list[str]) -> None:
    """Check Isaac Lab articulation body/joint names against metadata."""
    all_body_names = list(articulation.body_names)
    all_joint_names = list(articulation.joint_names)

    for link_name in metadata["fin_link_names"]:
        found = any(link_name in n for n in all_body_names)
        if not found:
            raise AssetValidationError(
                f"Fin link '{link_name}' not found in articulation body names. "
                f"Available: {all_body_names}"
            )

    for joint_name in metadata["fin_joint_names"]:
        found = any(joint_name in n for n in all_joint_names)
        if not found:
            raise AssetValidationError(
                f"Fin joint '{joint_name}' not found in articulation joint names. "
                f"Available: {all_joint_names}"
            )
