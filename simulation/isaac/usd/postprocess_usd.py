"""
postprocess_usd.py — Post-process a Blender-exported USD to add IsaacLab physics APIs.

Workflow:
  1. Model the drone in Blender with the required part names (see BLENDER_EXPORT_GUIDE.md)
  2. Export as USD from Blender (Z-up axis, unit = meters)
    3. Run this script — it adds ArticulationRootAPI, RigidBodyAPI, MassAPI,
        RevoluteJoint, and DriveAPI. Fin hinge positions come from the authored
        USD transforms, and Isaac Sim computes CoM/inertia from the colliders.

This replaces the geometry-authoring path in drone_builder.py: geometry comes
from Blender, while this script adds the physics APIs needed by Isaac Sim.

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
    BODY_PARTS_MVP,
    expected_mvp_prim_paths,
    load_fin_specs,
    FinSpec,
)

_DRIVE_STIFFNESS = 20.0
_DRIVE_DAMPING   = 1.0
# Negligible fin mass so runtime accepts the link; body carries effectively full vehicle mass.
_FIN_MASS_KG = 1e-5

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
def _warn_ignored_legacy_config(vehicle_cfg: dict) -> None:
    """Warn when legacy YAML geometry or inertia fields are present but ignored."""
    ignored_fields: list[str] = []

    mass_props_cfg = vehicle_cfg.get("mass_properties", {})
    if mass_props_cfg.get("center_of_mass") is not None:
        ignored_fields.append("vehicle.mass_properties.center_of_mass")
    if mass_props_cfg.get("inertia_tensor") is not None:
        ignored_fields.append("vehicle.mass_properties.inertia_tensor")

    fins_cfg = vehicle_cfg.get("fins", {}).get("fins_config", [])
    if any(fin.get("position") is not None for fin in fins_cfg):
        ignored_fields.append("vehicle.fins.fins_config[*].position")

    if ignored_fields:
        joined = ", ".join(ignored_fields)
        print(
            "[postprocess_usd] WARNING: ignoring legacy YAML geometry fields: "
            f"{joined}. Fin joints use authored USD transforms, and Isaac Sim/PhysX "
            "will compute CoM/inertia from the colliders."
        )


def _load_total_vehicle_mass(vehicle_cfg: dict) -> float:
    """Load the total vehicle mass from the legacy vehicle config."""
    mass_props_cfg = vehicle_cfg.get("mass_properties", {})
    if "total_mass" not in mass_props_cfg:
        raise KeyError(
            "vehicle.mass_properties.total_mass is required so the USD body mass can be authored."
        )
    return float(mass_props_cfg["total_mass"])


def _compute_body_mass(total_vehicle_mass: float, fin_mass: float, num_fins: int) -> float:
    """Split total vehicle mass into root-body mass plus articulated fin masses."""
    body_mass = total_vehicle_mass - num_fins * fin_mass
    if body_mass <= 0.0:
        print(
            "[postprocess_usd] WARNING: fin masses exceed total vehicle mass; "
            "using total mass directly on /Drone/Body."
        )
        return total_vehicle_mass
    return body_mass


def _clear_authored_mass_distribution(mass_api: UsdPhysics.MassAPI) -> None:
    """Remove authored CoM/inertia so Isaac Sim computes them from colliders."""
    for attr in (
        mass_api.GetCenterOfMassAttr(),
        mass_api.GetDiagonalInertiaAttr(),
        mass_api.GetPrincipalAxesAttr(),
    ):
        if attr.IsValid():
            attr.Clear()


def _clear_authored_inertia(mass_api: UsdPhysics.MassAPI) -> None:
    """Clear only inertia so Isaac Sim computes it; CoM is left as authored."""
    for attr in (
        mass_api.GetDiagonalInertiaAttr(),
        mass_api.GetPrincipalAxesAttr(),
    ):
        if attr.IsValid():
            attr.Clear()


def _body_local_bbox_center_z(stage: Usd.Stage) -> float:
    """Z component of body prim's local-axis-aligned bbox center (includes descendants)."""
    body_prim = stage.GetPrimAtPath(BODY_PRIM)
    if not body_prim or not UsdGeom.Imageable(body_prim):
        return 0.0
    time = Usd.TimeCode.Default()
    bbox = UsdGeom.Imageable(body_prim).ComputeLocalBound(time, UsdGeom.Tokens.default_)
    r = bbox.ComputeAlignedBox()
    return 0.5 * (r.GetMin()[2] + r.GetMax()[2])


def _add_root_physics(stage: Usd.Stage, body_mass: float) -> None:
    """Apply ArticulationRootAPI to /Drone and RigidBodyAPI + MassAPI to /Drone/Body.
    CoM is set to (0, 0, z) with z from body local bbox; inertia left for Isaac to compute.
    """
    root_prim = stage.GetPrimAtPath(DRONE_ROOT)
    body_prim = stage.GetPrimAtPath(BODY_PRIM)
    UsdPhysics.ArticulationRootAPI.Apply(root_prim)
    UsdPhysics.RigidBodyAPI.Apply(body_prim)
    mass_api = UsdPhysics.MassAPI.Apply(body_prim)
    mass_api.GetMassAttr().Set(body_mass)
    com_z = _body_local_bbox_center_z(stage)
    mass_api.GetCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, float(com_z)))
    _clear_authored_inertia(mass_api)


def _add_fin_physics(stage: Usd.Stage, fin_specs: list[FinSpec], fin_mass: float) -> None:
    """Apply RigidBodyAPI + MassAPI to each fin prim.

    Fins are authored as direct children of /Drone. Mass is set explicitly (0) and
    inertia zeroed so the runtime does not overwrite with collision-derived values.
    """
    for spec in fin_specs:
        fin_prim = stage.GetPrimAtPath(f"{DRONE_ROOT}/{spec.prim_name}")
        UsdPhysics.RigidBodyAPI.Apply(fin_prim)
        mass_api = UsdPhysics.MassAPI.Apply(fin_prim)
        mass_api.GetMassAttr().Set(float(fin_mass))
        # Override inertia to zero so Isaac/PhysX use authored mass=0 instead of recomputing from colliders.
        mass_api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        if mass_api.GetPrincipalAxesAttr().IsValid():
            mass_api.GetPrincipalAxesAttr().Clear()


def _read_fin_hinge_from_stage(stage: Usd.Stage, fin_path: str) -> tuple[float, float, float]:
    """Read the fin origin position in the Body frame from the authored USD transforms."""
    body_prim = stage.GetPrimAtPath(BODY_PRIM)
    fin_prim = stage.GetPrimAtPath(fin_path)
    if not body_prim.IsValid():
        raise ValueError(f"Missing body prim: {BODY_PRIM}")
    if not fin_prim.IsValid():
        raise ValueError(f"Missing fin prim: {fin_path}")

    time = Usd.TimeCode.Default()
    body_xf = UsdGeom.Xformable(body_prim).ComputeLocalToWorldTransform(time)
    fin_xf = UsdGeom.Xformable(fin_prim).ComputeLocalToWorldTransform(time)
    rel = body_xf.GetInverse() * fin_xf

    rel_xf = Gf.Transform()
    rel_xf.SetMatrix(rel)
    t = rel_xf.GetTranslation()
    return (float(t[0]), float(t[1]), float(t[2]))


def _add_collision_apis(stage: Usd.Stage, fin_specs: list[FinSpec]) -> None:
    """Attach collision APIs to body and fin mesh prims using convex decomposition.

    Colliders are applied to the prim that actually has the Mesh (so PhysX uses the
    right geometry). Body/fin assets may nest the mesh under the part xform, e.g.:
      /Drone/Body/edf_drone/edf_drone   /Drone/RightFin/RightFin
    We prefer the nested mesh path when present; otherwise use the part prim.
    """
    # Body: part prims and nested mesh prims when present (e.g. drone_v2).
    body_mesh_paths: list[str] = []
    for name in BODY_PARTS_MVP:
        nested = f"{BODY_PRIM}/{name}/{name}"
        if stage.GetPrimAtPath(nested).IsValid():
            body_mesh_paths.append(nested)
        else:
            body_mesh_paths.append(f"{BODY_PRIM}/{name}")

    # Fins: prefer nested mesh prim (e.g. /Drone/RightFin/RightFin) when present.
    fin_mesh_paths: list[str] = []
    for spec in fin_specs:
        nested = f"{DRONE_ROOT}/{spec.prim_name}/{spec.prim_name}"
        if stage.GetPrimAtPath(nested).IsValid():
            fin_mesh_paths.append(nested)
        else:
            fin_mesh_paths.append(f"{DRONE_ROOT}/{spec.prim_name}")

    for path in [*body_mesh_paths, *fin_mesh_paths]:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue
        # Base collision API (enables participation in contact generation).
        UsdPhysics.CollisionAPI.Apply(prim)
        # Mesh-specific approximation: convex decomposition of the mesh geometry.
        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_collision.CreateApproximationAttr("convexDecomposition")


def _create_fin_joints(
    stage: Usd.Stage,
    fin_specs: list[FinSpec],
    joint_lower_deg: float,
    joint_upper_deg: float,
) -> None:
    """Define RevoluteJoint + DriveAPI for each fin.

    The hinge position is read from the current USD transforms, so the CAD/Blender
    authored asset remains the source of truth for link placement.
    """
    for spec in fin_specs:
        fin_path   = f"{DRONE_ROOT}/{spec.prim_name}"
        # Place joint as child of the fin Xform using per-fin name (FwdFin_Joint etc.)
        # so IsaacLab can resolve each joint by name: find_joints(["FwdFin_Joint", ...]).
        joint_path = f"{fin_path}/{spec.prim_name}_Joint"

        joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)

        # Joint axis in fin-local frame:
        #   - RightFin, LeftFin: rotate about local Y (pitch-dominant pair)
        #   - FwdFin, AftFin: rotate about local X (roll-dominant pair)
        if spec.prim_name in ("RightFin", "LeftFin"):
            axis_token = "Y"
        elif spec.prim_name in ("FwdFin", "AftFin"):
            axis_token = "X"
        else:
            axis_token = getattr(spec, "joint_axis_local", spec.hinge_axis)
        joint.GetAxisAttr().Set(axis_token)
        joint.GetLowerLimitAttr().Set(joint_lower_deg)
        joint.GetUpperLimitAttr().Set(joint_upper_deg)

        joint.GetBody0Rel().SetTargets([Sdf.Path(BODY_PRIM)])
        joint.GetBody1Rel().SetTargets([Sdf.Path(fin_path)])

        hinge_zup = _read_fin_hinge_from_stage(stage, fin_path)
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
        config_path:   Vehicle YAML config for total mass, fin mass, and joint limits.
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
    _warn_ignored_legacy_config(vehicle_cfg)
    fin_specs   = load_fin_specs(vehicle_cfg)
    if not fin_specs:
        raise ValueError("No fins found in vehicle config. Check fins.fins_config.")

    fins_cfg = vehicle_cfg.get("fins", {})
    fin_mass = _FIN_MASS_KG
    total_vehicle_mass = _load_total_vehicle_mass(vehicle_cfg)
    body_mass = _compute_body_mass(total_vehicle_mass, fin_mass, len(fin_specs))

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

    # Author mass only; let Isaac Sim derive CoM/inertia from the colliders.
    _add_root_physics(stage, body_mass)
    _strip_rigid_body_from_visual_children(stage)
    _add_fin_physics(stage, fin_specs, fin_mass)
    _add_collision_apis(stage, fin_specs)
    _create_fin_joints(stage, fin_specs, -joint_limit_deg, joint_limit_deg)

    stage.SetDefaultPrim(stage.GetPrimAtPath(DRONE_ROOT))
    stage.GetRootLayer().Save()

    print(
        f"[postprocess_usd] Wrote {output_path}\n"
        f"  total_mass={total_vehicle_mass:.3f} kg  "
        f"body_mass={body_mass:.3f} kg  "
        f"fin_mass={fin_mass:.3f} kg each  "
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
        help=(
            "Vehicle YAML config used for total mass, fin mass, and joint limits. "
            "Fin positions and explicit CoM/inertia are ignored."
        ),
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
