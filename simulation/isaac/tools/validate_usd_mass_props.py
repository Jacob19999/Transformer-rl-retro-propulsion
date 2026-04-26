"""
USD mass-property validator.

Opens the drone USD with ``pxr`` (no live Isaac Sim runtime required) and compares
the body link's mass / COM / diagonal inertia against the YAML truth in
``configs/vehicle/edf_drone_v2.yaml``.

Per constitution, YAML is authoritative; this script enforces a 1% relative
tolerance and exits non-zero on any mismatch so it can gate CI / pre-merge.

Usage::

    python simulation/isaac/tools/validate_usd_mass_props.py
    python simulation/isaac/tools/validate_usd_mass_props.py --tolerance 0.02 --verbose
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Resolve project layout regardless of where this script is invoked from.
_TOOLS_DIR = Path(__file__).resolve().parent
_SIM_ROOT = _TOOLS_DIR.parent
_DEFAULT_USD = _SIM_ROOT / "assets/usd/drone_v2_physics.usd"
_DEFAULT_METADATA = _SIM_ROOT / "assets/metadata/edf_drone_v2.asset.yaml"
_DEFAULT_VEHICLE = _SIM_ROOT / "configs/vehicle/edf_drone_v2.yaml"

# Make tvc_env importable so we can reuse the existing helpers.
sys.path.insert(0, str(_SIM_ROOT))

from tvc_env.asset.mass_properties import (  # noqa: E402
    MassProperties,
    load_vehicle_config,
    validate_mass_properties,
)


def _load_metadata(metadata_yaml: Path) -> dict:
    import yaml
    with open(metadata_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("asset", data)


def _extract_mass_properties(stage, body_path: str) -> MassProperties | None:
    """Extract MassAPI fields from the named body prim using pure pxr."""
    from pxr import UsdPhysics

    prim = stage.GetPrimAtPath(body_path)
    if not prim.IsValid():
        return None
    if not prim.HasAPI(UsdPhysics.MassAPI):
        return None

    mass_api = UsdPhysics.MassAPI(prim)

    mass_attr = mass_api.GetMassAttr()
    mass = float(mass_attr.Get()) if mass_attr.IsValid() and mass_attr.Get() is not None else None

    com_attr = mass_api.GetCenterOfMassAttr()
    com = [0.0, 0.0, 0.0]
    if com_attr.IsValid() and com_attr.Get() is not None:
        com_val = com_attr.Get()
        com = [float(com_val[0]), float(com_val[1]), float(com_val[2])]

    inertia = {"Ixx": 0.0, "Iyy": 0.0, "Izz": 0.0, "Ixy": 0.0, "Ixz": 0.0, "Iyz": 0.0}
    inertia_attr = mass_api.GetDiagonalInertiaAttr()
    if inertia_attr.IsValid() and inertia_attr.Get() is not None:
        inertia_val = inertia_attr.Get()
        inertia["Ixx"] = float(inertia_val[0])
        inertia["Iyy"] = float(inertia_val[1])
        inertia["Izz"] = float(inertia_val[2])

    if mass is None:
        return None
    return MassProperties(mass_kg=mass, com_offset=com, inertia=inertia)


def _resolve_body_prim_path(stage, metadata: dict) -> str | None:
    """Find the body prim path by name, anywhere under the default prim."""
    body_link = metadata.get("body_link_name", "Body")
    default_prim = stage.GetDefaultPrim()
    if not default_prim or not default_prim.IsValid():
        return f"/{body_link}"

    # Search the default prim's subtree for a prim named body_link.
    for prim in stage.Traverse():
        if prim.GetName() == body_link:
            return str(prim.GetPath())
    return f"{default_prim.GetPath()}/{body_link}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", type=Path, default=_DEFAULT_USD)
    parser.add_argument("--metadata", type=Path, default=_DEFAULT_METADATA)
    parser.add_argument("--vehicle", type=Path, default=_DEFAULT_VEHICLE)
    parser.add_argument("--tolerance", type=float, default=0.01,
                        help="Relative tolerance per field (default 0.01 = 1%)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if not args.usd.exists():
        print(f"ERROR: USD not found: {args.usd}", file=sys.stderr)
        return 2
    if not args.vehicle.exists():
        print(f"ERROR: Vehicle YAML not found: {args.vehicle}", file=sys.stderr)
        return 2

    from pxr import Usd  # local import so --help works without pxr installed

    stage = Usd.Stage.Open(str(args.usd))
    if stage is None:
        print(f"ERROR: failed to open USD stage: {args.usd}", file=sys.stderr)
        return 2

    metadata = _load_metadata(args.metadata) if args.metadata.exists() else {}
    body_path = _resolve_body_prim_path(stage, metadata)

    if args.verbose:
        print(f"USD:        {args.usd}")
        print(f"Vehicle:    {args.vehicle}")
        print(f"Body prim:  {body_path}")
        print(f"Tolerance:  {args.tolerance:.2%}")

    usd_props = _extract_mass_properties(stage, body_path)
    if usd_props is None:
        print(
            f"ERROR: no UsdPhysics.MassAPI on '{body_path}'. "
            f"Author MassAPI on the body link or fix --metadata.body_link_name.",
            file=sys.stderr,
        )
        return 2

    config = load_vehicle_config(args.vehicle)

    print("Mass properties read from USD:")
    print(f"  mass:    {usd_props.mass_kg:.6f} kg")
    print(f"  COM:     {usd_props.com_offset}")
    print(f"  inertia: Ixx={usd_props.inertia['Ixx']:.6f}  "
          f"Iyy={usd_props.inertia['Iyy']:.6f}  "
          f"Izz={usd_props.inertia['Izz']:.6f}")
    print()
    print("Vehicle YAML truth:")
    print(f"  mass:    {config.get('total_mass')} kg")
    print(f"  COM:     {config.get('body_com_offset')}")
    print(f"  inertia: {config.get('inertia_tensor')}")
    print()

    issues = validate_mass_properties(usd_props, config, tolerance=args.tolerance)
    if issues:
        print(f"FAIL ({len(issues)} mismatch{'es' if len(issues) != 1 else ''}):", file=sys.stderr)
        for w in issues:
            print(f"  - {w}", file=sys.stderr)
        return 1

    print("PASS: USD mass properties match YAML within tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
