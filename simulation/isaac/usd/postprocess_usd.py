"""
postprocess_usd.py — Post-process a Blender-exported USD to add IsaacLab physics APIs.

Workflow:
  1. Model the drone in Blender with the required part names (see BLENDER_EXPORT_GUIDE.md)
  2. Export as USD from Blender (Z-up axis, unit = meters)
  3. Run this script — it adds ArticulationRootAPI, RigidBodyAPI, MassAPI,
     RevoluteJoint, and DriveAPI using parameters from default_vehicle.yaml.

This replaces the geometry-authoring path in drone_builder.py: geometry comes
from Blender, physics comes from this script + the YAML config.

CLI:
  # Add physics and write final drone.usd
  python -m simulation.isaac.usd.postprocess_usd \\
      --input  simulation/isaac/usd/drone_blender.usd \\
      --output simulation/isaac/usd/drone.usd \\
      --config simulation/configs/default_vehicle.yaml

  # Validate prim names only (no output written)
  python -m simulation.isaac.usd.postprocess_usd \\
      --input simulation/isaac/usd/drone_blender.usd \\
      --validate-only
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
except ImportError as e:
    raise ImportError(
        "USD Python bindings (pxr) not found. "
        "Run this script from within the Isaac Sim Python environment."
    ) from e

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from simulation.config_loader import load_config                       # noqa: E402
from simulation.isaac.usd.parts_registry import (                      # noqa: E402
    DRONE_ROOT,
    BODY_PRIM,
    expected_fin_prim_paths,
    expected_mvp_prim_paths,
    load_explicit_mass_props,
    load_fin_specs,
    frd_to_zup,
    FinSpec,
)

_DRIVE_STIFFNESS = 20.0
_DRIVE_DAMPING   = 1.0

def _bake_nonuniform_scale_into_mesh(prim: Usd.Prim) -> None:
    """Bake prim's *local* non-uniform scale into mesh points, then set scale to 1.

    Blender exports often leave object scales unapplied (non-1). PhysX can behave
    poorly with non-uniformly scaled rigid bodies. This helper preserves visuals
    by pushing scale into the mesh geometry.
    """
    # Robustly extract local scale using full xform matrix decomposition.
    # Some USD builds shipped with Isaac Sim don't expose ComputeLocalToParentTransform,
    # so we compute it from world transforms.
    xformable = UsdGeom.Xformable(prim)
    time = Usd.TimeCode.Default()
    world_m = xformable.ComputeLocalToWorldTransform(time)
    parent_prim = prim.GetParent()
    if parent_prim.IsValid() and UsdGeom.Xformable(parent_prim):
        parent_world_m = UsdGeom.Xformable(parent_prim).ComputeLocalToWorldTransform(time)
        local_m = parent_world_m.GetInverse() * world_m
    else:
        local_m = world_m
    xf = Gf.Transform()
    xf.SetMatrix(local_m)
    s = xf.GetScale()
    sx, sy, sz = float(s[0]), float(s[1]), float(s[2])

    # Only act on materially non-identity scale.
    if abs(sx - 1.0) < 1e-6 and abs(sy - 1.0) < 1e-6 and abs(sz - 1.0) < 1e-6:
        return

    # Bake into mesh points for prim itself (if Mesh) and immediate mesh children.
    def _bake_mesh(mesh_prim: Usd.Prim) -> None:
        if not mesh_prim.IsA(UsdGeom.Mesh):
            return
        mesh = UsdGeom.Mesh(mesh_prim)
        pts = mesh.GetPointsAttr().Get(Usd.TimeCode.Default())
        if not pts:
            return
        scaled = [Gf.Vec3f(float(p[0]) * sx, float(p[1]) * sy, float(p[2]) * sz) for p in pts]
        mesh.GetPointsAttr().Set(scaled)

    _bake_mesh(prim)
    for child in prim.GetChildren():
        _bake_mesh(child)

    # Reset the prim's authored local scale to identity while preserving other ops.
    # We do this by clearing ops and re-authoring TR (no scale).
    try:
        t = xf.GetTranslation()
        r = xf.GetRotation()
        quatd = r.GetQuat()
        qw = float(quatd.GetReal())
        qx, qy, qz = (
            float(quatd.GetImaginary()[0]),
            float(quatd.GetImaginary()[1]),
            float(quatd.GetImaginary()[2]),
        )
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(Gf.Vec3d(float(t[0]), float(t[1]), float(t[2])))
        xformable.AddOrientOp().Set(Gf.Quatf(qw, Gf.Vec3f(qx, qy, qz)))
        xformable.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
    except Exception:
        # Best-effort fallback: do nothing if ops can't be rewritten.
        pass


def _ensure_legs_under_body(stage: Usd.Stage) -> None:
    """Ensure legs follow the simulated base link.

    With /Drone as an articulation-root Xform (not a rigid body) and /Drone/Body
    as the base rigid link, purely-visual children under /Drone will not follow
    physics motion. To avoid fragile USD namespace editing (which varies across
    USD builds and can break with composed references), we instead:

    - apply RigidBodyAPI to /Drone/Legs
    - create a FixedJoint from /Drone/Body to /Drone/Legs

    This makes legs a rigid link that follows the base link deterministically.
    """
    legs_prim = stage.GetPrimAtPath(LEGS_PRIM)
    if not legs_prim.IsValid():
        return

    # Remove problematic non-uniform scale on legs rigid link.
    _bake_nonuniform_scale_into_mesh(legs_prim)

    # Ensure legs are a rigid body link.
    UsdPhysics.RigidBodyAPI.Apply(legs_prim)

    # Create a fixed joint tying Body (base link) to Legs.
    joint_path = f"{DRONE_ROOT}/Legs_FixedJoint"
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.GetBody0Rel().SetTargets([Sdf.Path(BODY_PRIM)])
    joint.GetBody1Rel().SetTargets([Sdf.Path(LEGS_PRIM)])

    # Set joint anchors so no snap/rotation occurs: compute Legs pose in Body frame.
    body_prim = stage.GetPrimAtPath(BODY_PRIM)
    try:
        body_xf = UsdGeom.Xformable(body_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        legs_xf = UsdGeom.Xformable(legs_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        rel = body_xf.GetInverse() * legs_xf
        # Decompose with Gf.Transform to remove scale/shear from the joint frame.
        rel_xf = Gf.Transform()
        rel_xf.SetMatrix(rel)
        t = rel_xf.GetTranslation()
        quatd = rel_xf.GetRotation().GetQuat()  # (real, imag)
        qw = float(quatd.GetReal())
        qx, qy, qz = (float(quatd.GetImaginary()[0]), float(quatd.GetImaginary()[1]), float(quatd.GetImaginary()[2]))
        joint.GetLocalPos0Attr().Set(Gf.Vec3f(float(t[0]), float(t[1]), float(t[2])))
        joint.GetLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.GetLocalRot0Attr().Set(Gf.Quatf(qw, Gf.Vec3f(qx, qy, qz)))
        joint.GetLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    except Exception as exc:
        print(f"[postprocess_usd] WARNING: Could not compute fixed-joint anchors for legs: {exc}")

    print(f"[postprocess_usd] Legs fixed to Body via {joint_path}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _validate_stage(stage: Usd.Stage, fin_specs: list[FinSpec]) -> tuple[list[str], list[str]]:
    """Validate prim names in the stage.

    Returns:
        (missing_required, missing_recommended) — both are lists of prim path strings.
        missing_required blocks processing; missing_recommended emits warnings only.
    """
    mvp = expected_mvp_prim_paths(num_fins=len(fin_specs))
    missing_required    = [p for p in mvp["required"]    if not stage.GetPrimAtPath(p).IsValid()]
    missing_recommended = [p for p in mvp["recommended"] if not stage.GetPrimAtPath(p).IsValid()]
    return missing_required, missing_recommended


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
def _ensure_stage_metadata(stage: Usd.Stage) -> None:
    """Enforce Z-up axis and meters-per-unit = 1.0."""
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        print("[postprocess_usd] WARNING: up-axis is not Z — setting to Z.")
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    if UsdGeom.GetStageMetersPerUnit(stage) != 1.0:
        print("[postprocess_usd] WARNING: meters-per-unit is not 1.0 — setting to 1.0.")
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)


# ---------------------------------------------------------------------------
# Physics API authoring
# ---------------------------------------------------------------------------
def _add_root_physics(stage: Usd.Stage, mass_props) -> None:
    """Apply ArticulationRootAPI to /Drone and RigidBodyAPI + MassAPI to /Drone/Body.

    Args:
        mass_props: ExplicitMassProps from parts_registry.
    """
    root_prim = stage.GetPrimAtPath(DRONE_ROOT)
    body_prim = stage.GetPrimAtPath(BODY_PRIM)
    UsdPhysics.ArticulationRootAPI.Apply(root_prim)
    UsdPhysics.RigidBodyAPI.Apply(body_prim)
    mass_api = UsdPhysics.MassAPI.Apply(body_prim)
    mass_api.GetMassAttr().Set(mass_props.total_mass)

    com_frd = mass_props.center_of_mass_frd
    com_zup = frd_to_zup(float(com_frd[0]), float(com_frd[1]), float(com_frd[2]))
    mass_api.GetCenterOfMassAttr().Set(Gf.Vec3f(*com_zup))

    I    = mass_props.inertia_tensor
    diag = (float(I[0][0]), float(I[1][1]), float(I[2][2]))
    mass_api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*diag))


def _add_fin_physics(stage: Usd.Stage, fin_specs: list[FinSpec], fin_mass: float) -> None:
    """Apply RigidBodyAPI + MassAPI to each fin prim.

    Fins are authored as direct children of /Drone. By making the *base* rigid body
    live at /Drone/Body (not /Drone), we avoid nested rigid bodies and therefore do
    not need resetXformStack on fins. This preserves inherited transforms and avoids
    fins drifting into the ground or losing scale.
    """
    for spec in fin_specs:
        fin_prim = stage.GetPrimAtPath(f"{DRONE_ROOT}/{spec.prim_name}")
        UsdPhysics.RigidBodyAPI.Apply(fin_prim)
        mass_api = UsdPhysics.MassAPI.Apply(fin_prim)
        mass_api.GetMassAttr().Set(fin_mass)


def _create_fin_joints(
    stage: Usd.Stage,
    fin_specs: list[FinSpec],
    joint_lower_deg: float,
    joint_upper_deg: float,
) -> None:
    """Define RevoluteJoint + DriveAPI for each fin.

    localPos0 is taken from the YAML hinge position (FRD → Z-up converted).
    This must match the fin prim's world translate set in Blender — the guide
    documents the required positions so artists can position fins correctly.
    """
    for spec in fin_specs:
        fin_path   = f"{DRONE_ROOT}/{spec.prim_name}"
        # Place joint as child of the fin Xform using per-fin name (Fin_N_Joint)
        # so IsaacLab can resolve each joint by name: find_joints(["Fin_1_Joint",...])
        joint_path = f"{fin_path}/{spec.prim_name}_Joint"

        joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
        # Use joint_axis_local (fin prim local frame) when fin mesh axes differ from body
        axis_token = getattr(spec, "joint_axis_local", spec.hinge_axis)
        joint.GetAxisAttr().Set(axis_token)
        joint.GetLowerLimitAttr().Set(joint_lower_deg)
        joint.GetUpperLimitAttr().Set(joint_upper_deg)

        joint.GetBody0Rel().SetTargets([Sdf.Path(BODY_PRIM)])
        joint.GetBody1Rel().SetTargets([Sdf.Path(fin_path)])

        hinge_zup = frd_to_zup(*spec.hinge_pos_frd)
        joint.GetLocalPos0Attr().Set(Gf.Vec3f(*hinge_zup))
        joint.GetLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

        drive_api = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        drive_api.GetStiffnessAttr().Set(_DRIVE_STIFFNESS)
        drive_api.GetDampingAttr().Set(_DRIVE_DAMPING)
        drive_api.GetTargetPositionAttr().Set(0.0)


# ---------------------------------------------------------------------------
# Cleanup helpers
# ---------------------------------------------------------------------------
def _strip_rigid_body_from_visual_children(stage: Usd.Stage) -> None:
    """Remove RigidBodyAPI from visual-only prims under /Drone/Body.

    Prims like /Drone/Body/Legs are pure geometry — they should not have
    RigidBodyAPI applied (PhysX errors on nested rigid bodies in same hierarchy).
    """
    body_prim = stage.GetPrimAtPath(BODY_PRIM)
    if not body_prim.IsValid():
        return
    for child in body_prim.GetChildren():
        if child.HasAPI(UsdPhysics.RigidBodyAPI):
            child.RemoveAPI(UsdPhysics.RigidBodyAPI)
            print(f"[postprocess_usd] Removed RigidBodyAPI from visual prim {child.GetPath()}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def postprocess_drone_usd(
    input_path: str | Path,
    output_path: str | Path | None,
    config_path: str | Path = "simulation/configs/default_vehicle.yaml",
    *,
    validate_only: bool = False,
) -> None:
    """Post-process a Blender-exported USD to add IsaacLab physics APIs.

    Args:
        input_path:    Blender-exported .usd/.usda file.
        output_path:   Destination .usd file (ignored when validate_only=True).
        config_path:   Vehicle YAML config for mass/fin parameters.
        validate_only: If True, only check prim names exist; do not write output.

    Raises:
        ValueError:   If required prims are missing from the Blender USD.
        RuntimeError: If the USD stage cannot be opened.
    """
    input_path  = Path(input_path)
    config_path = Path(config_path)

    # Load config and fin specs
    cfg         = load_config(config_path)
    vehicle_cfg = cfg.get("vehicle", cfg)
    fin_specs   = load_fin_specs(vehicle_cfg)
    if not fin_specs:
        raise ValueError("No fins found in vehicle config. Check fins.fins_config.")

    fins_cfg = vehicle_cfg.get("fins", {})
    fin_mass = float(fins_cfg.get("servo", {}).get("weight_kg", 0.003))

    max_deflection_rad = float(fins_cfg.get("max_deflection", 0.349))
    joint_limit_deg    = math.degrees(max_deflection_rad)

    # Open stage
    stage = Usd.Stage.Open(str(input_path))
    if not stage:
        raise RuntimeError(f"Failed to open USD file: {input_path}")

    # Validate prim names
    missing_req, missing_rec = _validate_stage(stage, fin_specs)
    if missing_rec:
        print(
            "[postprocess_usd] WARNING: missing recommended prims (non-blocking):\n"
            + "\n".join(f"  {p}" for p in missing_rec)
        )
    if missing_req:
        raise ValueError(
            "Blender USD is missing required prims:\n"
            + "\n".join(f"  {p}" for p in missing_req)
            + "\n\nSee simulation/isaac/usd/BLENDER_EXPORT_GUIDE.md for naming requirements."
        )

    if validate_only:
        print(f"[postprocess_usd] Validation passed: {input_path}")
        return

    # Copy to output path, then post-process in-place
    output_path = Path(output_path)  # type: ignore[arg-type]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage.Export(str(output_path))
    stage = Usd.Stage.Open(str(output_path))

    _ensure_stage_metadata(stage)

    # Explicit mass properties from YAML (no primitive aggregation)
    mass_props = load_explicit_mass_props(vehicle_cfg)

    _add_root_physics(stage, mass_props)
    _strip_rigid_body_from_visual_children(stage)
    _add_fin_physics(stage, fin_specs, fin_mass)
    _create_fin_joints(stage, fin_specs, -joint_limit_deg, joint_limit_deg)

    stage.SetDefaultPrim(stage.GetPrimAtPath(DRONE_ROOT))
    stage.GetRootLayer().Save()

    com_zup = frd_to_zup(*mass_props.center_of_mass_frd)
    print(
        f"[postprocess_usd] Wrote {output_path}\n"
        f"  mass={mass_props.total_mass:.3f} kg  "
        f"CoM_zup={com_zup}  "
        f"fins={len(fin_specs)}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-process a Blender-exported USD to add IsaacLab physics APIs."
    )
    parser.add_argument("--input",  required=True, help="Blender-exported .usd file")
    parser.add_argument("--output", help="Output .usd file with physics (required unless --validate-only)")
    parser.add_argument(
        "--config",
        default="simulation/configs/default_vehicle.yaml",
        help="Vehicle YAML config (default: simulation/configs/default_vehicle.yaml)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate prim names; do not write output",
    )
    args = parser.parse_args()

    if not args.validate_only and not args.output:
        parser.error("--output is required unless --validate-only is set")

    def _resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else REPO_ROOT / path

    postprocess_drone_usd(
        input_path    = _resolve(args.input),
        output_path   = _resolve(args.output) if args.output else None,
        config_path   = _resolve(args.config),
        validate_only = args.validate_only,
    )


if __name__ == "__main__":
    _main()
