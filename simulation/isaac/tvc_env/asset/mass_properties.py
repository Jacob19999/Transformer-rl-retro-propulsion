"""
Mass property extraction and validation for USD assets.

Extracts mass, COM offset, and inertia tensor from USD MassAPI prims,
then compares against edf_drone_v2.yaml with 1% tolerance per the constitution.
"""

from __future__ import annotations
import yaml
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Any


@dataclass
class MassProperties:
    """Extracted mass properties from USD or config."""
    mass_kg: float
    com_offset: list[float]   # [x, y, z] in body frame (m)
    inertia: dict[str, float]  # {Ixx, Iyy, Izz, Ixy, Ixz, Iyz}


def load_vehicle_config(vehicle_yaml_path: str | Path) -> dict[str, Any]:
    """Load vehicle configuration YAML.

    Args:
        vehicle_yaml_path: Path to edf_drone_v2.yaml.

    Returns:
        Parsed vehicle config dict.
    """
    path = Path(vehicle_yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Vehicle config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("vehicle", data)


def extract_mass_properties_from_usd(stage, body_path: str) -> MassProperties | None:
    """Extract mass properties from a USD prim with MassAPI.

    Args:
        stage: USD Stage object.
        body_path: Path to the body prim with UsdPhysics.MassAPI.

    Returns:
        MassProperties if extraction succeeds, None if MassAPI not found.

    Raises:
        ImportError: If USD/Isaac Sim is not available.
    """
    try:
        import pxr.UsdPhysics as UsdPhysics
        from pxr import Gf
    except ImportError as e:
        raise ImportError("Isaac Sim runtime required for USD prim access.") from e

    prim = stage.GetPrimAtPath(body_path)
    if not prim.IsValid() or not prim.HasAPI(UsdPhysics.MassAPI):
        return None

    mass_api = UsdPhysics.MassAPI(prim)

    mass = None
    mass_attr = mass_api.GetMassAttr()
    if mass_attr.IsValid():
        mass = float(mass_attr.Get())

    com = [0.0, 0.0, 0.0]
    com_attr = mass_api.GetCenterOfMassAttr()
    if com_attr.IsValid():
        com_val = com_attr.Get()
        if com_val is not None:
            com = [float(com_val[0]), float(com_val[1]), float(com_val[2])]

    inertia = {"Ixx": 0.0, "Iyy": 0.0, "Izz": 0.0, "Ixy": 0.0, "Ixz": 0.0, "Iyz": 0.0}
    inertia_attr = mass_api.GetDiagonalInertiaAttr()
    if inertia_attr.IsValid():
        inertia_val = inertia_attr.Get()
        if inertia_val is not None:
            inertia["Ixx"] = float(inertia_val[0])
            inertia["Iyy"] = float(inertia_val[1])
            inertia["Izz"] = float(inertia_val[2])

    if mass is None:
        return None
    return MassProperties(mass_kg=mass, com_offset=com, inertia=inertia)


def validate_mass_properties(
    usd_props: MassProperties,
    config: dict[str, Any],
    tolerance: float = 0.01,
) -> list[str]:
    """Compare USD mass properties against vehicle config YAML.

    Per constitution: mass/inertia in config MUST match USD within 1% (tolerance=0.01).

    Args:
        usd_props: Mass properties extracted from USD.
        config: Vehicle config dict from load_vehicle_config().
        tolerance: Relative tolerance (default 0.01 = 1%).

    Returns:
        List of warning/error strings describing mismatches. Empty if all pass.
    """
    warnings = []

    # Validate total mass
    config_mass = config.get("total_mass")
    if config_mass is not None:
        rel_err = abs(usd_props.mass_kg - config_mass) / max(abs(config_mass), 1e-12)
        if rel_err > tolerance:
            warnings.append(
                f"Mass mismatch: USD={usd_props.mass_kg:.4f}kg, "
                f"config={config_mass:.4f}kg, rel_err={rel_err:.4f} > {tolerance}"
            )

    # Config stores body offsets in FRD, while USD MassAPI uses the asset's
    # Isaac-local axes (x forward, y left, z up).
    config_com_frd = config.get("body_com_offset")
    if config_com_frd is not None:
        config_com_isaac = [
            float(config_com_frd[0]),
            -float(config_com_frd[1]),
            -float(config_com_frd[2]),
        ]
        for axis, usd_val, cfg_val in zip("xyz", usd_props.com_offset, config_com_isaac):
            abs_err = abs(float(usd_val) - cfg_val)
            allowed = max(1e-5, tolerance * max(abs(cfg_val), 1e-3))
            if abs_err > allowed:
                warnings.append(
                    f"COM mismatch [{axis}]: USD Isaac-local={float(usd_val):.6f}m, "
                    f"config FRD converted={cfg_val:.6f}m, abs_err={abs_err:.6f} > {allowed:.6f}"
                )

    # Validate inertia tensor diagonal
    config_inertia = config.get("inertia_tensor", {})
    for key in ["Ixx", "Iyy", "Izz"]:
        cfg_val = config_inertia.get(key)
        usd_val = usd_props.inertia.get(key)
        if cfg_val is not None and usd_val is not None and usd_val != 0.0:
            rel_err = abs(usd_val - cfg_val) / max(abs(cfg_val), 1e-12)
            if rel_err > tolerance:
                warnings.append(
                    f"Inertia mismatch [{key}]: USD={usd_val:.6f}, "
                    f"config={cfg_val:.6f}, rel_err={rel_err:.4f} > {tolerance}"
                )

    return warnings
