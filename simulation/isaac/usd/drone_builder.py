"""
drone_builder.py — Validate and inspect an already-authored drone USD.

This repo originally used this module to *generate* a drone USD from YAML, including:
- emitting geometry primitives
- applying physics schemas (RigidBody, Mass, Collision)
- creating fin revolute joints + drives

That approach is brittle: programmatic authoring can diverge from what you set up
manually in Isaac Sim (or Blender), and can introduce hard-to-debug USD issues.

Current intent:
- Treat `simulation/isaac/usd/drone.usd` as the single source of truth.
- Provide a small, safe CLI to validate the USD “shape” expected by the codebase.
- Do not author or modify the USD.

CLI examples:
  python -m simulation.isaac.usd.drone_builder validate --usd simulation/isaac/usd/drone_v2_physics.usdc
  python -m simulation.isaac.usd.drone_builder info --usd simulation/isaac/usd/drone_v2_physics.usdc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# pxr requires the Carbonite/Omniverse runtime (loaded by SimulationApp).
# We defer the import until _main() so module-level tests can import
# the helper functions without triggering a full Isaac Sim launch.
# ---------------------------------------------------------------------------
Usd = None          # populated by _bootstrap_pxr()
UsdGeom = None
UsdPhysics = None


def _bootstrap_pxr() -> None:
    """Start SimulationApp headlessly and import pxr bindings."""
    global Usd, UsdGeom, UsdPhysics
    if Usd is not None:
        return  # already bootstrapped
    from isaacsim import SimulationApp
    # Store on module so it is not garbage-collected (would unload Carbonite).
    global _SIM_APP  # noqa: PLW0603
    _SIM_APP = SimulationApp({"headless": True, "hide_ui": True})
    from pxr import Usd as _Usd, UsdGeom as _UsdGeom, UsdPhysics as _UsdPhysics
    Usd, UsdGeom, UsdPhysics = _Usd, _UsdGeom, _UsdPhysics


_SIM_APP = None  # keeps SimulationApp alive for the process lifetime

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from simulation.isaac.usd.parts_registry import (
    DRONE_ROOT,
    FIN_PRIM_NAMES,
    expected_fin_prim_paths,
)


def open_usd_stage(usd_path: str | Path):  # -> Usd.Stage
    _bootstrap_pxr()
    usd_path = Path(usd_path)
    if not usd_path.is_absolute():
        usd_path = REPO_ROOT / usd_path
    if not usd_path.exists():
        raise FileNotFoundError(f"USD not found: {usd_path}")
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {usd_path}")
    return stage


def _iter_expected_paths() -> Iterable[str]:
    yield DRONE_ROOT
    yield f"{DRONE_ROOT}/Body"
    for p in expected_fin_prim_paths():
        yield p


def validate_drone_usd(stage) -> list[str]:  # stage: Usd.Stage
    """Validate the minimal prim layout and physics schemas we depend on.

    Returns:
        List of human-readable issues. Empty list means "looks good".
    """
    issues: list[str] = []

    # Stage metadata (best-effort; don't fail if missing).
    up_axis = UsdGeom.GetStageUpAxis(stage)
    if up_axis and up_axis != UsdGeom.Tokens.z:
        issues.append(f"Stage upAxis is '{up_axis}', expected 'z'.")

    default_prim = stage.GetDefaultPrim()
    if not default_prim or not default_prim.IsValid():
        issues.append("Stage defaultPrim is not set.")
    elif default_prim.GetPath() != DRONE_ROOT:
        issues.append(f"defaultPrim is '{default_prim.GetPath()}', expected '{DRONE_ROOT}'.")

    # Expected prims exist.
    for p in _iter_expected_paths():
        prim = stage.GetPrimAtPath(p)
        if not prim or not prim.IsValid():
            issues.append(f"Missing prim at '{p}'.")

    # Physics schemas (soft requirements; warn instead of hard-fail).
    drone = stage.GetPrimAtPath(DRONE_ROOT)
    if drone and drone.IsValid():
        if not UsdPhysics.ArticulationRootAPI(drone):
            issues.append("'/Drone' is missing UsdPhysics.ArticulationRootAPI.")

    body = stage.GetPrimAtPath(f"{DRONE_ROOT}/Body")
    if body and body.IsValid():
        if not UsdPhysics.RigidBodyAPI(body):
            issues.append("'/Drone/Body' is missing UsdPhysics.RigidBodyAPI.")

    for fin_name in FIN_PRIM_NAMES:
        fin = stage.GetPrimAtPath(f"{DRONE_ROOT}/{fin_name}")
        if fin and fin.IsValid():
            if not UsdPhysics.RigidBodyAPI(fin):
                issues.append(f"'{DRONE_ROOT}/{fin_name}' is missing UsdPhysics.RigidBodyAPI.")

    # Joints: postprocess_usd uses /Drone/<name>/<name>_Joint; also allow RevoluteJoint, legacy flat.
    for fin_name in FIN_PRIM_NAMES:
        candidates = [
            f"{DRONE_ROOT}/{fin_name}/RevoluteJoint",
            f"{DRONE_ROOT}/{fin_name}/{fin_name}_Joint",
            f"{DRONE_ROOT}/{fin_name}_Joint",
        ]
        found = False
        for c in candidates:
            prim = stage.GetPrimAtPath(c)
            if prim and prim.IsValid():
                found = True
                break
        if not found:
            issues.append(
                f"No RevoluteJoint found for {fin_name}. "
                f"Checked: {', '.join(candidates)}"
            )

    return issues


def print_stage_info(stage) -> None:  # stage: Usd.Stage
    up_axis = UsdGeom.GetStageUpAxis(stage)
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    default_prim = stage.GetDefaultPrim()
    print(f"[drone_usd] upAxis={up_axis} metersPerUnit={meters_per_unit}")
    print(f"[drone_usd] defaultPrim={default_prim.GetPath() if default_prim else None}")

    for p in _iter_expected_paths():
        prim = stage.GetPrimAtPath(p)
        t = prim.GetTypeName() if prim and prim.IsValid() else None
        print(f"[drone_usd] {p} type={t}")

    # Print joint info
    for fin_name in FIN_PRIM_NAMES:
        candidates = [
            f"{DRONE_ROOT}/{fin_name}/RevoluteJoint",
            f"{DRONE_ROOT}/{fin_name}/{fin_name}_Joint",
            f"{DRONE_ROOT}/{fin_name}_Joint",
        ]
        for jp in candidates:
            prim = stage.GetPrimAtPath(jp)
            if prim and prim.IsValid():
                print(f"[drone_usd] {jp} type={prim.GetTypeName()}")
                break
        else:
            print(f"[drone_usd] {DRONE_ROOT}/{fin_name} — no joint found")


# ---------------------------------------------------------------------------
# CLI (T010)
# ---------------------------------------------------------------------------
def _main() -> None:
    parser = argparse.ArgumentParser(description="Validate and inspect drone USD")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def _add_usd_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--usd",
            default="simulation/isaac/usd/drone_v2_physics.usdc",
            help="Path to drone USD (default: simulation/isaac/usd/drone_v2_physics.usdc)",
        )

    p_validate = sub.add_parser("validate", help="Validate expected prim layout / schemas")
    _add_usd_arg(p_validate)

    p_info = sub.add_parser("info", help="Print basic stage + prim info")
    _add_usd_arg(p_info)

    args = parser.parse_args()
    stage = open_usd_stage(args.usd)

    if args.cmd == "info":
        print_stage_info(stage)
        return

    issues = validate_drone_usd(stage)
    if issues:
        print("[drone_usd] VALIDATION FAILED")
        for msg in issues:
            print(f"- {msg}")
        raise SystemExit(2)
    print("[drone_usd] OK")


if __name__ == "__main__":
    _main()
