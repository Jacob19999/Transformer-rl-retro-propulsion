"""
test_mass_validation.py — Unit tests for USDC ↔ YAML mass property comparison.

Tests run headlessly (no Isaac Sim, no pxr required for pure-logic tests).
The compare_mass_properties() function is tested by mocking read_usd_mass_props()
and using a real test vehicle config.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TEST_VEHICLE_YAML = _REPO_ROOT / "simulation" / "configs" / "default_vehicle.yaml"


def _load_vehicle_cfg() -> dict:
    import yaml
    with open(_TEST_VEHICLE_YAML) as f:
        data = yaml.safe_load(f)
    return data.get("vehicle", data)


@pytest.fixture
def vehicle_cfg():
    return _load_vehicle_cfg()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _identity_usd_data(vehicle_cfg: dict) -> dict:
    """Return USD data dict that exactly matches the YAML config (should PASS).

    Computes eigendecomposition of the YAML inertia tensor to obtain the
    principal moments (diagonal) and principal axes quaternion — the same
    representation that USD Physics uses. reconstruct_inertia_tensor() should
    then reproduce the full 3x3 tensor within floating-point precision.
    """
    import numpy as np
    from simulation.isaac.usd.parts_registry import load_explicit_mass_props, frd_to_zup

    props = load_explicit_mass_props(vehicle_cfg)
    com_zup = frd_to_zup(*props.center_of_mass_frd)

    # Eigendecomposition: symmetric I = R @ diag(λ) @ R^T
    I_np = np.array(props.inertia_tensor, dtype=np.float64)
    eigenvalues, R = np.linalg.eigh(I_np)  # R columns are eigenvectors

    # Ensure right-handed rotation (det = +1)
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1

    # Rotation matrix → scalar-last quaternion (qx, qy, qz, qw)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s; qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s; qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s; qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s; qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s; qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s; qz = 0.25 * s

    return {
        "mass": props.total_mass,
        "com_zup": com_zup,
        "diagonal": tuple(float(v) for v in eigenvalues),
        "principal_axes_quat": (float(qx), float(qy), float(qz), float(qw)),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMassPropertyValidation:
    """Validates compare_mass_properties() logic without USD runtime."""

    def test_pass_case_matching_values(self, vehicle_cfg):
        """PASS: USD data exactly matches YAML config."""
        from simulation.isaac.scripts.validate_mass_props import compare_mass_properties

        usd_data = _identity_usd_data(vehicle_cfg)
        with patch(
            "simulation.isaac.scripts.validate_mass_props.read_usd_mass_props",
            return_value=usd_data,
        ):
            report = compare_mass_properties("fake.usdc", vehicle_cfg, tolerance=0.01)

        assert report.passed, (
            f"Expected PASS but got FAIL. Discrepancies: "
            + ", ".join(f"{d.field_name}: {d.relative_error:.4f}" for d in report.discrepancies if not d.within_tolerance)
        )
        assert len(report.discrepancies) == 10

    def test_fail_case_wrong_mass(self, vehicle_cfg):
        """FAIL: USD has wrong total mass (5.0 kg instead of 3.13 kg)."""
        from simulation.isaac.scripts.validate_mass_props import compare_mass_properties

        usd_data = _identity_usd_data(vehicle_cfg)
        usd_data["mass"] = 5.0  # intentionally wrong

        with patch(
            "simulation.isaac.scripts.validate_mass_props.read_usd_mass_props",
            return_value=usd_data,
        ):
            report = compare_mass_properties("fake.usdc", vehicle_cfg, tolerance=0.01)

        assert not report.passed, "Expected FAIL but got PASS"

        mass_disc = next(d for d in report.discrepancies if d.field_name == "total_mass")
        assert not mass_disc.within_tolerance
        assert mass_disc.actual == pytest.approx(5.0, abs=1e-6)

    def test_tolerance_edge_case_exactly_one_percent(self, vehicle_cfg):
        """Edge: error of exactly 1% should be within 1% tolerance (<=)."""
        from simulation.isaac.scripts.validate_mass_props import compare_mass_properties
        from simulation.isaac.usd.parts_registry import load_explicit_mass_props

        props = load_explicit_mass_props(vehicle_cfg)
        usd_data = _identity_usd_data(vehicle_cfg)
        # Set mass to exactly 1% above YAML value
        usd_data["mass"] = props.total_mass * 1.01

        with patch(
            "simulation.isaac.scripts.validate_mass_props.read_usd_mass_props",
            return_value=usd_data,
        ):
            report = compare_mass_properties("fake.usdc", vehicle_cfg, tolerance=0.01)

        mass_disc = next(d for d in report.discrepancies if d.field_name == "total_mass")
        # relative error == 0.01, tolerance == 0.01, so within_tolerance must be True
        assert mass_disc.within_tolerance, (
            f"Relative error {mass_disc.relative_error:.6f} should be <= 0.01"
        )

    def test_missing_mass_api_raises_error(self, vehicle_cfg):
        """ERROR: read_usd_mass_props() raises RuntimeError on missing MassAPI."""
        from simulation.isaac.scripts.validate_mass_props import compare_mass_properties

        with patch(
            "simulation.isaac.scripts.validate_mass_props.read_usd_mass_props",
            side_effect=RuntimeError("/Drone/Body does not have UsdPhysics.MassAPI applied."),
        ):
            with pytest.raises(RuntimeError, match="MassAPI"):
                compare_mass_properties("fake.usdc", vehicle_cfg, tolerance=0.01)

    def test_file_not_found_propagates(self, vehicle_cfg):
        """ERROR: FileNotFoundError propagates from read_usd_mass_props()."""
        from simulation.isaac.scripts.validate_mass_props import compare_mass_properties

        with patch(
            "simulation.isaac.scripts.validate_mass_props.read_usd_mass_props",
            side_effect=FileNotFoundError("USD file not found: missing.usdc"),
        ):
            with pytest.raises(FileNotFoundError):
                compare_mass_properties("missing.usdc", vehicle_cfg, tolerance=0.01)

    def test_report_has_ten_discrepancies(self, vehicle_cfg):
        """Report always contains exactly 10 comparison fields."""
        from simulation.isaac.scripts.validate_mass_props import compare_mass_properties

        usd_data = _identity_usd_data(vehicle_cfg)
        with patch(
            "simulation.isaac.scripts.validate_mass_props.read_usd_mass_props",
            return_value=usd_data,
        ):
            report = compare_mass_properties("fake.usdc", vehicle_cfg, tolerance=0.01)

        field_names = [d.field_name for d in report.discrepancies]
        assert "total_mass" in field_names
        assert "com_x" in field_names and "com_y" in field_names and "com_z" in field_names
        assert "Ixx" in field_names and "Iyy" in field_names and "Izz" in field_names
        assert "Ixy" in field_names and "Ixz" in field_names and "Iyz" in field_names
        assert len(field_names) == 10


class TestReconstructInertiaTensor:
    """Validates the inertia tensor reconstruction utility."""

    def test_identity_rotation_returns_diagonal(self):
        from simulation.isaac.usd.parts_registry import reconstruct_inertia_tensor

        diagonal = (0.01358, 0.01549, 0.00480)
        quat_identity = (0.0, 0.0, 0.0, 1.0)

        result = reconstruct_inertia_tensor(diagonal, quat_identity)

        assert result[0][0] == pytest.approx(0.01358, rel=1e-6)
        assert result[1][1] == pytest.approx(0.01549, rel=1e-6)
        assert result[2][2] == pytest.approx(0.00480, rel=1e-6)
        assert result[0][1] == pytest.approx(0.0, abs=1e-10)
        assert result[0][2] == pytest.approx(0.0, abs=1e-10)
        assert result[1][2] == pytest.approx(0.0, abs=1e-10)

    def test_zero_quaternion_treated_as_identity(self):
        from simulation.isaac.usd.parts_registry import reconstruct_inertia_tensor

        diagonal = (0.1, 0.2, 0.3)
        quat_zero = (0.0, 0.0, 0.0, 0.0)

        result = reconstruct_inertia_tensor(diagonal, quat_zero)

        assert result[0][0] == pytest.approx(0.1, rel=1e-6)
        assert result[1][1] == pytest.approx(0.2, rel=1e-6)
        assert result[2][2] == pytest.approx(0.3, rel=1e-6)

    def test_symmetry_preserved_under_rotation(self):
        import math
        from simulation.isaac.usd.parts_registry import reconstruct_inertia_tensor

        # 45° rotation about Z: quat = (0, 0, sin(π/8), cos(π/8))
        angle = math.pi / 4
        qz = math.sin(angle / 2)
        qw = math.cos(angle / 2)
        diagonal = (0.01358, 0.01549, 0.00480)

        result = reconstruct_inertia_tensor(diagonal, (0.0, 0.0, qz, qw))

        # Symmetry check
        assert result[0][1] == pytest.approx(result[1][0], abs=1e-12)
        assert result[0][2] == pytest.approx(result[2][0], abs=1e-12)
        assert result[1][2] == pytest.approx(result[2][1], abs=1e-12)
        # Trace must be preserved under rotation
        trace = diagonal[0] + diagonal[1] + diagonal[2]
        assert result[0][0] + result[1][1] + result[2][2] == pytest.approx(trace, rel=1e-6)


class TestFormatReport:
    """Validates report formatting."""

    def test_format_table_contains_pass(self, vehicle_cfg):
        from simulation.isaac.scripts.validate_mass_props import (
            compare_mass_properties,
            format_report,
        )

        usd_data = _identity_usd_data(vehicle_cfg)
        with patch(
            "simulation.isaac.scripts.validate_mass_props.read_usd_mass_props",
            return_value=usd_data,
        ):
            report = compare_mass_properties("fake.usdc", vehicle_cfg)

        text = format_report(report)
        assert "PASS" in text
        assert "total_mass" in text

    def test_format_json_is_valid(self, vehicle_cfg):
        import json as json_mod
        from simulation.isaac.scripts.validate_mass_props import (
            compare_mass_properties,
            format_report,
        )

        usd_data = _identity_usd_data(vehicle_cfg)
        with patch(
            "simulation.isaac.scripts.validate_mass_props.read_usd_mass_props",
            return_value=usd_data,
        ):
            report = compare_mass_properties("fake.usdc", vehicle_cfg)

        text = format_report(report, as_json=True)
        data = json_mod.loads(text)
        assert data["passed"] is True
        assert len(data["fields"]) == 10

    def test_quiet_mode_returns_pass_or_fail(self, vehicle_cfg):
        from simulation.isaac.scripts.validate_mass_props import (
            compare_mass_properties,
            format_report,
        )

        usd_data = _identity_usd_data(vehicle_cfg)
        with patch(
            "simulation.isaac.scripts.validate_mass_props.read_usd_mass_props",
            return_value=usd_data,
        ):
            report = compare_mass_properties("fake.usdc", vehicle_cfg)

        assert format_report(report, quiet=True) == "PASS"
