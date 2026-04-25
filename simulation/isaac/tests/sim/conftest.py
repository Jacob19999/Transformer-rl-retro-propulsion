"""
Pytest fixtures for Isaac Sim asset validation tests.

Provides an in-memory USD stage whose prim hierarchy exactly matches
edf_drone_v2.asset.yaml and the drone_v2_physics.usd scene structure,
so tests are self-contained and do not depend on whatever stage happens
to be open in the Isaac Sim context.

USD hierarchy created here:
  /Drone                         (defaultPrim, Xform)
  /Drone/Body                    (Xform, ArticulationRootAPI)
  /Drone/FwdFin                  (Xform)  — +X, forward
  /Drone/RightFin                (Xform)  — +Y, right
  /Drone/AftFin                  (Xform)  — -X, aft
  /Drone/LeftFin                 (Xform)  — -Y, left
  /Drone/Body/joint_FwdFin       (RevoluteJoint)
  /Drone/Body/joint_RightFin     (RevoluteJoint)
  /Drone/Body/joint_AftFin       (RevoluteJoint)
  /Drone/Body/joint_LeftFin      (RevoluteJoint)
"""
from __future__ import annotations
import pytest

try:
    import pxr.Usd as Usd
    import pxr.UsdGeom as UsdGeom
    import pxr.UsdPhysics as UsdPhysics
    _PXR_AVAILABLE = True
except ImportError:
    _PXR_AVAILABLE = False


@pytest.fixture(scope="module")
def usd_stage():
    """In-memory USD stage matching edf_drone_v2 metadata structure."""
    if not _PXR_AVAILABLE:
        pytest.skip("pxr not available for in-memory USD stage creation")

    stage = Usd.Stage.CreateInMemory()

    drone = UsdGeom.Xform.Define(stage, "/Drone")
    stage.SetDefaultPrim(drone.GetPrim())

    body = UsdGeom.Xform.Define(stage, "/Drone/Body")
    UsdPhysics.ArticulationRootAPI.Apply(body.GetPrim())

    # Fin bodies: siblings of Body, in canonical +X/+Y/-X/-Y order
    for fin_name in ["FwdFin", "RightFin", "AftFin", "LeftFin"]:
        UsdGeom.Xform.Define(stage, f"/Drone/{fin_name}")

    # Revolute joints: children of Body
    for joint_name in ["joint_FwdFin", "joint_RightFin", "joint_AftFin", "joint_LeftFin"]:
        UsdPhysics.RevoluteJoint.Define(stage, f"/Drone/Body/{joint_name}")

    return stage
