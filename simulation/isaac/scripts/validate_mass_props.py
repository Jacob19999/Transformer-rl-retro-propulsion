"""
validate_mass_props.py — Compare USDC scene mass properties against YAML config.

Runs headlessly (no Isaac Sim launch required). Uses pxr (OpenUSD) to read
UsdPhysics.MassAPI attributes from the drone USDC scene, then compares them
against the explicit mass properties defined in the vehicle YAML config.

Exit codes:
  0  PASS — all 10 values within tolerance
  1  FAIL — one or more values outside tolerance
  2  ERROR — file not found, schema missing, or config invalid
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Repo-root helper
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[4]


def _repo_path(rel: str) -> Path:
    return _REPO_ROOT / rel


# ---------------------------------------------------------------------------
# Dataclasses (data-model.md entities)
# ---------------------------------------------------------------------------

@dataclass
class Discrepancy:
    """Per-field comparison result."""
    field_name: str
    expected: float
    actual: float
    relative_error: float
    within_tolerance: bool


@dataclass
class MassPropertyReport:
    """Full comparison between YAML config and USDC scene mass properties."""
    yaml_mass: float
    usd_mass: float
    yaml_com_frd: tuple[float, float, float]
    usd_com_zup: tuple[float, float, float]
    usd_com_frd: tuple[float, float, float]
    yaml_inertia: tuple
    usd_inertia: tuple
    tolerance: float
    passed: bool
    discrepancies: list[Discrepancy] = field(default_factory=list)


# ---------------------------------------------------------------------------
# USD reading  (T005)
# ---------------------------------------------------------------------------

def read_usd_mass_props(usd_path: str | Path) -> dict:
    """Read MassAPI attributes from /Drone/Body prim in a USDC scene.

    Returns a dict with keys:
      mass (float), com_zup (3-tuple), diagonal (3-tuple),
      principal_axes_quat (4-tuple qx,qy,qz,qw)

    Raises:
      FileNotFoundError: if USD file does not exist.
      RuntimeError: if /Drone/Body prim or MassAPI schema is missing.
    """
    usd_path = Path(usd_path)
    if not usd_path.exists():
        raise FileNotFoundError(f"USD file not found: {usd_path}")

    try:
        from pxr import Usd, UsdPhysics, Gf  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "pxr (OpenUSD) is not installed. Install it or run from an Isaac Sim environment."
        ) from exc

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {usd_path}")

    body_prim = stage.GetPrimAtPath("/Drone/Body")
    if not body_prim.IsValid():
        raise RuntimeError(
            "Prim /Drone/Body not found in USD stage. "
            "Ensure the USDC has the standard drone layout."
        )

    if not body_prim.HasAPI(UsdPhysics.MassAPI):
        raise RuntimeError(
            "/Drone/Body does not have UsdPhysics.MassAPI applied. "
            "Run postprocess_usd.py to add physics APIs to the scene."
        )

    mass_api = UsdPhysics.MassAPI(body_prim)

    # Total mass
    mass_attr = mass_api.GetMassAttr()
    mass_val = mass_attr.Get() if mass_attr.IsValid() else None
    if mass_val is None:
        raise RuntimeError("/Drone/Body MassAPI has no 'physics:mass' attribute set.")
    mass = float(mass_val)

    # Center of mass (prim-local Z-up frame, GfVec3f)
    com_attr = mass_api.GetCenterOfMassAttr()
    com_val = com_attr.Get() if com_attr.IsValid() else None
    com_zup = tuple(float(v) for v in com_val) if com_val is not None else (0.0, 0.0, 0.0)

    # Diagonal inertia (principal moments)
    diag_attr = mass_api.GetDiagonalInertiaAttr()
    diag_val = diag_attr.Get() if diag_attr.IsValid() else None
    diagonal = tuple(float(v) for v in diag_val) if diag_val is not None else (0.0, 0.0, 0.0)

    # Principal axes quaternion (GfQuatf → xyzw)
    axes_attr = mass_api.GetPrincipalAxesAttr()
    axes_val = axes_attr.Get() if axes_attr.IsValid() else None
    if axes_val is not None:
        img = axes_val.GetImaginary()
        principal_axes_quat = (float(img[0]), float(img[1]), float(img[2]), float(axes_val.GetReal()))
    else:
        principal_axes_quat = (0.0, 0.0, 0.0, 1.0)  # identity

    return {
        "mass": mass,
        "com_zup": com_zup,
        "diagonal": diagonal,
        "principal_axes_quat": principal_axes_quat,
    }


# ---------------------------------------------------------------------------
# Comparison logic  (T006)
# ---------------------------------------------------------------------------

def compare_mass_properties(
    usd_path: str | Path,
    yaml_config: dict,
    tolerance: float = 0.01,
) -> MassPropertyReport:
    """Compare USDC scene mass properties against YAML explicit mass config.

    Args:
        usd_path: Path to the USDC scene file.
        yaml_config: Vehicle config dict (the 'vehicle' sub-section of default_vehicle.yaml).
        tolerance: Relative tolerance for comparison (default 0.01 = 1%).

    Returns:
        MassPropertyReport with per-field comparison and overall pass/fail.

    Raises:
        FileNotFoundError: if USD file does not exist.
        KeyError: if YAML missing mass_properties section or use_explicit is false.
        RuntimeError: if USD scene is missing required prims or schemas.
    """
    from simulation.isaac.usd.parts_registry import (
        load_explicit_mass_props,
        zup_to_frd,
        reconstruct_inertia_tensor,
    )

    # Load YAML (authoritative source)
    yaml_props = load_explicit_mass_props(yaml_config)

    # Load USD
    usd_data = read_usd_mass_props(usd_path)

    # Convert USD CoM from Z-up prim frame to FRD body frame
    usd_com_frd = zup_to_frd(*usd_data["com_zup"])

    # Reconstruct full inertia tensor from USD diagonal + principal axes
    usd_inertia = reconstruct_inertia_tensor(
        usd_data["diagonal"],
        usd_data["principal_axes_quat"],
    )

    # Helper: compute relative error, handle near-zero expected
    def rel_error(expected: float, actual: float) -> float:
        if abs(expected) < 1e-12:
            return abs(actual - expected)
        return abs(actual - expected) / abs(expected)

    discrepancies: list[Discrepancy] = []

    def _check(name: str, expected: float, actual: float) -> None:
        err = rel_error(expected, actual)
        discrepancies.append(Discrepancy(
            field_name=name,
            expected=expected,
            actual=actual,
            relative_error=err,
            within_tolerance=(err <= tolerance),
        ))

    # 1 mass
    _check("total_mass", yaml_props.total_mass, usd_data["mass"])

    # 3 CoM components
    yaml_com = yaml_props.center_of_mass_frd
    _check("com_x", yaml_com[0], usd_com_frd[0])
    _check("com_y", yaml_com[1], usd_com_frd[1])
    _check("com_z", yaml_com[2], usd_com_frd[2])

    # 6 unique inertia tensor components (symmetric: Ixx, Iyy, Izz, Ixy, Ixz, Iyz)
    yaml_I = yaml_props.inertia_tensor
    _check("Ixx", yaml_I[0][0], usd_inertia[0][0])
    _check("Iyy", yaml_I[1][1], usd_inertia[1][1])
    _check("Izz", yaml_I[2][2], usd_inertia[2][2])
    _check("Ixy", yaml_I[0][1], usd_inertia[0][1])
    _check("Ixz", yaml_I[0][2], usd_inertia[0][2])
    _check("Iyz", yaml_I[1][2], usd_inertia[1][2])

    passed = all(d.within_tolerance for d in discrepancies)

    return MassPropertyReport(
        yaml_mass=yaml_props.total_mass,
        usd_mass=usd_data["mass"],
        yaml_com_frd=yaml_com,
        usd_com_zup=usd_data["com_zup"],
        usd_com_frd=usd_com_frd,
        yaml_inertia=yaml_props.inertia_tensor,
        usd_inertia=usd_inertia,
        tolerance=tolerance,
        passed=passed,
        discrepancies=discrepancies,
    )


# ---------------------------------------------------------------------------
# Report formatting  (T007)
# ---------------------------------------------------------------------------

_UNITS = {
    "total_mass": "kg",
    "com_x": "m", "com_y": "m", "com_z": "m",
    "Ixx": "kg·m²", "Iyy": "kg·m²", "Izz": "kg·m²",
    "Ixy": "kg·m²", "Ixz": "kg·m²", "Iyz": "kg·m²",
}


def format_report(report: MassPropertyReport, as_json: bool = False, quiet: bool = False) -> str:
    """Format comparison report as human-readable table or JSON."""
    if as_json:
        data = {
            "passed": report.passed,
            "tolerance": report.tolerance,
            "fields": [
                {
                    "name": d.field_name,
                    "expected": d.expected,
                    "actual": d.actual,
                    "relative_error": d.relative_error,
                    "within_tolerance": d.within_tolerance,
                }
                for d in report.discrepancies
            ],
        }
        return json.dumps(data, indent=2)

    if quiet:
        return "PASS" if report.passed else "FAIL"

    lines = [
        "Mass Property Validation Report",
        "=" * 72,
        f"{'Field':<14} {'YAML':>14} {'USD':>14} {'Error':>8}  Status",
        "-" * 72,
    ]

    for d in report.discrepancies:
        unit = _UNITS.get(d.field_name, "")
        expected_s = f"{d.expected:.5g} {unit}".strip()
        actual_s   = f"{d.actual:.5g} {unit}".strip()
        error_s    = f"{d.relative_error * 100:.2f}%"
        status     = "✓" if d.within_tolerance else "✗ FAIL"
        lines.append(f"{d.field_name:<14} {expected_s:>14} {actual_s:>14} {error_s:>8}  {status}")

    n_pass = sum(1 for d in report.discrepancies if d.within_tolerance)
    n_total = len(report.discrepancies)
    lines.append("=" * 72)
    result = "PASS" if report.passed else "FAIL"
    lines.append(f"RESULT: {result} ({n_pass}/{n_total} within {report.tolerance * 100:.1f}% tolerance)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point  (T008)
# ---------------------------------------------------------------------------

def _load_vehicle_yaml(config_path: Path) -> dict:
    """Load vehicle config YAML; return the 'vehicle' sub-dict."""
    import yaml  # type: ignore
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return data.get("vehicle", data)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare USDC scene mass properties against YAML vehicle config.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--usd",
        type=Path,
        default=_repo_path("simulation/isaac/usd/drone_v2_physics.usdc"),
        help="Path to USDC scene (default: simulation/isaac/usd/drone_v2_physics.usdc)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_repo_path("simulation/configs/default_vehicle.yaml"),
        help="Path to vehicle YAML config (default: simulation/configs/default_vehicle.yaml)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Relative tolerance for comparison, e.g. 0.01 = 1%% (default: 0.01)",
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Output report as JSON",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print PASS or FAIL",
    )
    args = parser.parse_args(argv)

    try:
        vehicle_cfg = _load_vehicle_yaml(args.config)
    except FileNotFoundError:
        print(f"ERROR: Config file not found: {args.config}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"ERROR loading config: {exc}", file=sys.stderr)
        return 2

    try:
        report = compare_mass_properties(args.usd, vehicle_cfg, tolerance=args.tolerance)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except (KeyError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(format_report(report, as_json=args.as_json, quiet=args.quiet))
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
