"""
drone_builder.py — Generate drone.usd from default_vehicle.yaml.

Implements T005-T010:
- Loads default_vehicle.yaml via existing config_loader.py
- Computes composite mass / CoM / inertia via MassProperties
- Emits body geometry (cylinders, boxes, spheres) under /Drone/Body
- Emits 4 fin geometry cubes + RevoluteJoints + DriveAPIs under /Drone/Fin_N
- Applies -90° X-rotation at /Drone root to align FRD body with Isaac Y-up
- CLI: python -m simulation.isaac.usd.drone_builder [--config ...] [--output ...]

Coordinate convention:
  All child geometry is specified in FRD body frame coordinates.
  The /Drone root has a -90° X rotation that maps:
    FRD +X (forward)  → world +X
    FRD +Y (right)    → world +Z
    FRD +Z (down)     → world -Y  (correct: down in Y-up)
  No per-child coordinate conversion is needed.

USD schema contract: specs/001-isaac-sim-env/contracts/usd-asset-schema.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# USD imports — pxr ships with Isaac Sim's Python environment
# ---------------------------------------------------------------------------
try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
except ImportError as e:
    raise ImportError(
        "USD Python bindings (pxr) not found. "
        "Run this script from within the Isaac Sim Python environment."
    ) from e

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from simulation.config_loader import load_config  # noqa: E402
from simulation.dynamics.mass_properties import compute_mass_properties  # noqa: E402

# ---------------------------------------------------------------------------
# Constants (from usd-asset-schema.md)
# ---------------------------------------------------------------------------
_FIN_CHORD = 0.0325          # half-extent along local X (chord/2)
_FIN_SPAN  = 0.0275          # half-extent along local Y (span/2)
_FIN_THICK = 0.0039          # half-extent along local Z (thickness/2)
_FIN_MASS  = 0.003           # kg per fin

_JOINT_LOWER = -15.0         # degrees
_JOINT_UPPER =  15.0         # degrees
_DRIVE_STIFFNESS = 25.0
_DRIVE_DAMPING   = 0.05

# Fin hinge positions and axes in FRD body frame (from usd-asset-schema.md)
_FINS = [
    {"name": "Fin_1", "hinge_pos": (0.0,  0.055, 0.14), "axis": "X"},
    {"name": "Fin_2", "hinge_pos": (0.0, -0.055, 0.14), "axis": "X"},
    {"name": "Fin_3", "hinge_pos": (0.055, 0.0,  0.14), "axis": "Y"},
    {"name": "Fin_4", "hinge_pos": (-0.055, 0.0, 0.14), "axis": "Y"},
]

# Phase A testing: only emit these primitives (EDF duct + fins)
_PHASE_A_PRIMITIVES = {"edf_duct"}


# ---------------------------------------------------------------------------
# Stage setup
# ---------------------------------------------------------------------------
def _new_stage(output_path: str) -> Usd.Stage:
    stage = Usd.Stage.CreateNew(output_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    return stage


# ---------------------------------------------------------------------------
# Root drone prim
# ---------------------------------------------------------------------------
def _create_drone_root(stage: Usd.Stage, mass_props) -> UsdGeom.Xform:
    """Create /Drone root with RigidBodyAPI + MassAPI + -90° X-rotation.

    The -90° X rotation maps FRD body frame to Isaac Y-up world:
      Rot_X(-90°): (x,y,z) → (x, z, -y)
        FRD +X (fwd)   → world +X
        FRD +Y (right)  → world +Z
        FRD +Z (down)   → world -Y (down in Y-up) ✓
    Child geometry is in FRD coordinates; this root rotation handles everything.
    """
    drone_prim_path = "/Drone"
    drone_xform = UsdGeom.Xform.Define(stage, drone_prim_path)
    prim = drone_xform.GetPrim()

    # -90° rotation about X to align FRD body with Y-up world frame
    xform_api = UsdGeom.XformCommonAPI(drone_xform)
    xform_api.SetRotate(Gf.Vec3f(-90.0, 0.0, 0.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)

    # Physics APIs
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.GetMassAttr().Set(mass_props.total_mass)

    # CoM stays in FRD body frame (root rotation handles world mapping)
    com_frd = mass_props.center_of_mass
    mass_api.GetCenterOfMassAttr().Set(Gf.Vec3f(*[float(c) for c in com_frd]))

    # Diagonal inertia (principal axes approximation; off-diagonals ignored here)
    I = mass_props.inertia_tensor
    diag = (float(I[0, 0]), float(I[1, 1]), float(I[2, 2]))
    mass_api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*diag))

    return drone_xform


# ---------------------------------------------------------------------------
# Body geometry emitter (T006)
# ---------------------------------------------------------------------------
def _emit_body_geometry(stage: Usd.Stage, primitives: list[dict]) -> None:
    """Emit USD geometry prims for each YAML primitive under /Drone/Body.

    All positions and dimensions are in FRD body frame. The /Drone root
    rotation handles the FRD → Y-up world conversion.
    """
    body_path = "/Drone/Body"
    body_xform = UsdGeom.Xform.Define(stage, body_path)

    for i, prim_cfg in enumerate(primitives):
        name = prim_cfg.get("name", f"prim_{i}").replace(" ", "_")

        # Phase A filter: only emit EDF duct
        if name not in _PHASE_A_PRIMITIVES:
            continue

        shape = prim_cfg.get("shape", "").lower()
        child_path = f"{body_path}/{name}"

        # Position in FRD body frame (root rotation handles world mapping)
        pos = prim_cfg.get("position", [0.0, 0.0, 0.0])

        geom_prim = None

        if shape == "cylinder":
            cyl = UsdGeom.Cylinder.Define(stage, child_path)
            cyl.GetRadiusAttr().Set(float(prim_cfg["radius"]))
            cyl.GetHeightAttr().Set(float(prim_cfg["height"]))
            # Cylinder axis along Z to match FRD thrust axis
            cyl.GetAxisAttr().Set("Z")
            UsdGeom.XformCommonAPI(cyl).SetTranslate(Gf.Vec3d(*pos))
            geom_prim = cyl.GetPrim()

        elif shape == "box":
            dims = prim_cfg["dimensions"]  # [x, y, z] in FRD
            cube = UsdGeom.Cube.Define(stage, child_path)
            cube_xform = UsdGeom.XformCommonAPI(cube)
            cube_xform.SetTranslate(Gf.Vec3d(*pos))
            # Scale directly in FRD: x, y, z — no axis swapping
            cube_xform.SetScale(Gf.Vec3f(float(dims[0]), float(dims[1]), float(dims[2])))
            geom_prim = cube.GetPrim()

        elif shape == "sphere":
            sph = UsdGeom.Sphere.Define(stage, child_path)
            sph.GetRadiusAttr().Set(float(prim_cfg["radius"]))
            UsdGeom.XformCommonAPI(sph).SetTranslate(Gf.Vec3d(*pos))
            geom_prim = sph.GetPrim()

        else:
            continue  # Unknown shape; skip

        if geom_prim is not None:
            UsdPhysics.CollisionAPI.Apply(geom_prim)


# ---------------------------------------------------------------------------
# Fin geometry emitter (T007)
# ---------------------------------------------------------------------------
def _emit_fin_geometry(stage: Usd.Stage) -> None:
    """Emit fin Xform + Geom (UsdGeom.Cube) with MassAPI per fin."""
    for fin in _FINS:
        fin_path = f"/Drone/{fin['name']}"
        fin_xform = UsdGeom.Xform.Define(stage, fin_path)

        # Position fin Xform at hinge point in FRD body frame
        UsdGeom.XformCommonAPI(fin_xform).SetTranslate(Gf.Vec3d(*fin["hinge_pos"]))

        # MassAPI on the fin Xform
        fin_mass_api = UsdPhysics.MassAPI.Apply(fin_xform.GetPrim())
        fin_mass_api.GetMassAttr().Set(_FIN_MASS)

        # Geometry child (unit cube scaled to half-extents)
        geom_path = f"{fin_path}/Geom"
        geom_cube = UsdGeom.Cube.Define(stage, geom_path)
        geom_xform_api = UsdGeom.XformCommonAPI(geom_cube)
        # offset from hinge (leading edge) by chord/2 along local Z in fin frame
        geom_xform_api.SetTranslate(Gf.Vec3d(0.0, 0.0, _FIN_CHORD))
        geom_xform_api.SetScale(Gf.Vec3f(_FIN_CHORD, _FIN_SPAN, _FIN_THICK))

        UsdPhysics.CollisionAPI.Apply(geom_cube.GetPrim())


# ---------------------------------------------------------------------------
# Joint + drive emitter (T008)
# ---------------------------------------------------------------------------
def _emit_fin_joints(stage: Usd.Stage) -> None:
    """Emit RevoluteJoint + DriveAPI for each fin."""
    drone_prim = stage.GetPrimAtPath("/Drone")

    for fin in _FINS:
        fin_path = f"/Drone/{fin['name']}"
        joint_path = f"{fin_path}/RevoluteJoint"

        joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)

        # Axis in FRD body frame (root rotation handles world mapping)
        joint.GetAxisAttr().Set(fin["axis"])

        joint.GetLowerLimitAttr().Set(_JOINT_LOWER)
        joint.GetUpperLimitAttr().Set(_JOINT_UPPER)

        # body0 = /Drone root; body1 = fin Xform
        joint.GetBody0Rel().SetTargets([Sdf.Path("/Drone")])
        joint.GetBody1Rel().SetTargets([Sdf.Path(fin_path)])

        # Local frame on body0 at hinge position (FRD body frame)
        joint.GetLocalPos0Attr().Set(Gf.Vec3f(*fin["hinge_pos"]))
        joint.GetLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

        # DriveAPI
        drive_api = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        drive_api.GetStiffnessAttr().Set(_DRIVE_STIFFNESS)
        drive_api.GetDampingAttr().Set(_DRIVE_DAMPING)
        drive_api.GetTargetPositionAttr().Set(0.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_drone_usd(config_path: str | Path, output_path: str | Path) -> None:
    """Generate drone.usd from YAML config.

    Args:
        config_path: Path to default_vehicle.yaml (or any vehicle config).
        output_path: Destination .usd file path.
    """
    config_path = Path(config_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load vehicle config
    cfg = load_config(config_path)
    vehicle_cfg = cfg.get("vehicle", cfg)  # support both nested and flat
    primitives = vehicle_cfg["primitives"]

    # Compute composite mass properties
    mass_props = compute_mass_properties(primitives)

    # Create stage
    stage = _new_stage(str(output_path))

    # Root drone prim (T005, T009)
    _create_drone_root(stage, mass_props)

    # Body geometry (T006)
    _emit_body_geometry(stage, primitives)

    # Fin geometry (T007)
    _emit_fin_geometry(stage)

    # Fin joints + drives (T008)
    _emit_fin_joints(stage)

    stage.GetRootLayer().Save()
    print(f"[drone_builder] Wrote {output_path} "
          f"(mass={mass_props.total_mass:.3f} kg, "
          f"CoM={mass_props.center_of_mass.tolist()})")


# ---------------------------------------------------------------------------
# CLI (T010)
# ---------------------------------------------------------------------------
def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate drone.usd from default_vehicle.yaml"
    )
    parser.add_argument(
        "--config",
        default="simulation/configs/default_vehicle.yaml",
        help="Path to vehicle YAML config (default: simulation/configs/default_vehicle.yaml)",
    )
    parser.add_argument(
        "--output",
        default="simulation/isaac/usd/drone.usd",
        help="Output USD file path (default: simulation/isaac/usd/drone.usd)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path

    build_drone_usd(config_path, output_path)


if __name__ == "__main__":
    _main()
