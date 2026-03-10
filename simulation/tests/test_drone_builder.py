"""
Tests for simulation/isaac/usd/drone_builder.py.

T011: test_usd_round_trip
T032: test_yaml_to_usd_parameter_propagation
T033: test_fin_joint_limits_from_yaml
T034: test_composite_mass_from_yaml

These tests require pxr (USD Python bindings) — marked with 'isaac' marker.
Skip gracefully when pxr is not available (CI without GPU).
"""

from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Conditional skip: pxr only available inside Isaac Sim Python environment
# ---------------------------------------------------------------------------
pxr = pytest.importorskip("pxr", reason="pxr not available; skipping USD tests")
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from simulation.isaac.usd.drone_builder import build_drone_usd  # noqa: E402
from simulation.config_loader import load_config               # noqa: E402
from simulation.dynamics.mass_properties import compute_mass_properties  # noqa: E402

DEFAULT_VEHICLE_YAML = REPO_ROOT / "simulation" / "configs" / "default_vehicle.yaml"
pytestmark = pytest.mark.isaac


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def default_usd(tmp_path_factory):
    """Build drone.usd from default_vehicle.yaml once per module."""
    out = tmp_path_factory.mktemp("usd") / "drone.usd"
    build_drone_usd(DEFAULT_VEHICLE_YAML, out)
    stage = Usd.Stage.Open(str(out))
    return stage


@pytest.fixture(scope="module")
def default_mass_props():
    cfg = load_config(DEFAULT_VEHICLE_YAML)
    vehicle_cfg = cfg.get("vehicle", cfg)
    return compute_mass_properties(vehicle_cfg["primitives"])


# ---------------------------------------------------------------------------
# T011: USD round-trip
# ---------------------------------------------------------------------------
@pytest.mark.isaac
def test_usd_round_trip(default_usd, default_mass_props):
    """T011: Generated USD opens, mass within 1%, 4 revolute joints, correct fin half-extents."""
    stage = default_usd

    # --- Mass check ---
    drone_prim = stage.GetPrimAtPath("/Drone")
    assert drone_prim.IsValid(), "/Drone prim must exist"
    mass_api = UsdPhysics.MassAPI(drone_prim)
    usd_mass = mass_api.GetMassAttr().Get()
    expected_mass = default_mass_props.total_mass
    assert abs(usd_mass - expected_mass) / expected_mass < 0.01, (
        f"Mass mismatch: USD={usd_mass:.4f}, YAML={expected_mass:.4f}"
    )

    # --- Revolute joints count (4) ---
    joint_prims = [
        p for p in stage.Traverse()
        if p.HasAPI(UsdPhysics.RevoluteJoint)
    ]
    assert len(joint_prims) == 4, f"Expected 4 revolute joints, got {len(joint_prims)}"

    # --- Joint limits ---
    for jp in joint_prims:
        joint = UsdPhysics.RevoluteJoint(jp)
        lower = joint.GetLowerLimitAttr().Get()
        upper = joint.GetUpperLimitAttr().Get()
        assert abs(lower - (-15.0)) < 0.1, f"Lower limit wrong: {lower}"
        assert abs(upper -  15.0)  < 0.1, f"Upper limit wrong: {upper}"

    # --- Fin cube half-extents (any Cube child of a Fin_N prim) ---
    expected_scale = (0.0325, 0.0275, 0.0039)
    fin_geom_prims = [
        p for p in stage.Traverse()
        if p.GetName() == "Geom" and p.GetTypeName() == "Cube"
    ]
    assert len(fin_geom_prims) == 4, f"Expected 4 fin Geom cubes, got {len(fin_geom_prims)}"
    for gp in fin_geom_prims:
        xform = UsdGeom.XformCommonAPI(UsdGeom.Xform(gp))
        scale, *_ = xform.GetXformVectors(Usd.TimeCode.Default())
        # scale is returned as (translate, rotate, scale, pivot, rot_order)
        # Actually GetXformVectors returns (t, r, s, pivot, rot_order)
        # s is at index 2
        s = xform.GetXformVectors(Usd.TimeCode.Default())[2]
        for actual, exp in zip(s, expected_scale):
            assert abs(actual - exp) / exp < 0.01, (
                f"Fin half-extent mismatch: got {s}, expected ~{expected_scale}"
            )


# ---------------------------------------------------------------------------
# T032: YAML parameter propagation — duct radius
# ---------------------------------------------------------------------------
@pytest.mark.isaac
def test_yaml_to_usd_parameter_propagation(tmp_path):
    """T032: Override edf_duct.radius → regenerate USD → verify duct cylinder radius."""
    cfg = load_config(DEFAULT_VEHICLE_YAML)
    vehicle_cfg = cfg.get("vehicle", cfg)

    # Override duct radius
    modified_radius = 0.060
    for prim in vehicle_cfg["primitives"]:
        if prim.get("name") == "edf_duct":
            prim["radius"] = modified_radius
            break

    # Write modified YAML to temp file
    mod_yaml = tmp_path / "modified_vehicle.yaml"
    with open(mod_yaml, "w") as f:
        yaml.dump(cfg, f)

    out_usd = tmp_path / "drone_mod.usd"
    build_drone_usd(mod_yaml, out_usd)

    stage = Usd.Stage.Open(str(out_usd))
    duct_prim = stage.GetPrimAtPath("/Drone/Body/edf_duct")
    assert duct_prim.IsValid(), "/Drone/Body/edf_duct must exist"
    cyl = UsdGeom.Cylinder(duct_prim)
    usd_radius = cyl.GetRadiusAttr().Get()
    assert abs(usd_radius - modified_radius) / modified_radius < 0.01, (
        f"Duct radius: expected {modified_radius}, got {usd_radius}"
    )


# ---------------------------------------------------------------------------
# T033: Fin joint limits reflect YAML fins.max_deflection → 15° control limit
# ---------------------------------------------------------------------------
@pytest.mark.isaac
def test_fin_joint_limits_from_yaml(default_usd):
    """T033: All 4 RevoluteJoint prims have lowerLimit=-15.0 and upperLimit=+15.0."""
    stage = default_usd
    joint_prims = [
        p for p in stage.Traverse()
        if p.HasAPI(UsdPhysics.RevoluteJoint)
    ]
    assert len(joint_prims) == 4
    for jp in joint_prims:
        joint = UsdPhysics.RevoluteJoint(jp)
        assert joint.GetLowerLimitAttr().Get() == pytest.approx(-15.0, abs=0.01)
        assert joint.GetUpperLimitAttr().Get() == pytest.approx( 15.0, abs=0.01)


# ---------------------------------------------------------------------------
# T034: Composite mass matches sum of YAML primitives
# ---------------------------------------------------------------------------
@pytest.mark.isaac
def test_composite_mass_from_yaml(default_usd, default_mass_props):
    """T034: USD MassAPI mass within 1% of computed YAML primitive sum."""
    stage = default_usd
    drone_prim = stage.GetPrimAtPath("/Drone")
    mass_api = UsdPhysics.MassAPI(drone_prim)
    usd_mass = mass_api.GetMassAttr().Get()
    expected = default_mass_props.total_mass
    assert abs(usd_mass - expected) / expected < 0.01, (
        f"Mass: USD={usd_mass:.4f}, computed={expected:.4f}"
    )
