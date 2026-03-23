"""
USD scene loading and prim access for EDF drone asset.

Loads the USD scene, resolves prim paths, and extracts API schemas
(ArticulationRootAPI, RigidBodyAPI, MassAPI) for validation.

NOTE: This module uses Isaac Sim USD APIs. It requires the Isaac Sim runtime
to be initialized before calling any functions here. For offline unit tests,
mock this module or import only after Isaac Sim is available.
"""

from __future__ import annotations
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AssetPrimInfo:
    """Structured data extracted from USD scene for a single prim."""
    path: str
    has_articulation_root: bool = False
    has_rigid_body: bool = False
    has_mass_api: bool = False
    mass_kg: float | None = None
    com_offset: list[float] | None = None
    inertia: list[float] | None = None  # [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]


@dataclass
class LoadedAsset:
    """Complete asset data extracted from a loaded USD scene."""
    stage_path: str
    body_link_path: str
    fin_link_paths: list[str]
    fin_joint_paths: list[str]
    body_prim: AssetPrimInfo | None = None
    fin_prims: list[AssetPrimInfo] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def load_asset_metadata(metadata_yaml_path: str | Path) -> dict[str, Any]:
    """Load asset metadata YAML and return as dict.

    Args:
        metadata_yaml_path: Path to edf_drone_v2.asset.yaml.

    Returns:
        Parsed metadata dict with asset/link/joint/COP information.
    """
    path = Path(metadata_yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Asset metadata not found: {path}")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data.get("asset", data)


def load_usd_scene(usd_path: str | Path, metadata: dict[str, Any]) -> LoadedAsset:
    """Load USD scene and extract prim information.

    This function requires Isaac Sim runtime. Import omni.usd before calling.

    Args:
        usd_path: Path to the .usd or .usdc file.
        metadata: Asset metadata dict from load_asset_metadata().

    Returns:
        LoadedAsset with prim paths resolved from metadata.

    Raises:
        ImportError: If called without Isaac Sim runtime available.
        ValueError: If required USD prims are not found.
    """
    try:
        import omni.usd
        import pxr.Usd as Usd
        import pxr.UsdPhysics as UsdPhysics
    except ImportError as e:
        raise ImportError(
            "Isaac Sim runtime required for USD loading. "
            "Run this in an Isaac Sim app context."
        ) from e

    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        raise ValueError(f"Failed to open USD stage: {usd_path}")

    body_name = metadata["body_link_name"]
    fin_link_names = metadata["fin_link_names"]
    fin_joint_names = metadata["fin_joint_names"]

    # Resolve prim paths (assume prims live at /<name>)
    body_path = f"/{body_name}"
    fin_link_paths = [f"/{body_name}/{name}" for name in fin_link_names]
    fin_joint_paths = [f"/{body_name}/{name}" for name in fin_joint_names]

    # Extract body prim info
    body_prim_usd = stage.GetPrimAtPath(body_path)
    body_prim_info = _extract_prim_info(body_prim_usd, body_path) if body_prim_usd.IsValid() else None

    # Extract fin prim info
    fin_prim_infos = []
    for path in fin_link_paths:
        prim = stage.GetPrimAtPath(path)
        fin_prim_infos.append(_extract_prim_info(prim, path) if prim.IsValid() else AssetPrimInfo(path=path))

    return LoadedAsset(
        stage_path=str(usd_path),
        body_link_path=body_path,
        fin_link_paths=fin_link_paths,
        fin_joint_paths=fin_joint_paths,
        body_prim=body_prim_info,
        fin_prims=fin_prim_infos,
        metadata=metadata,
    )


def _extract_prim_info(prim, path: str) -> AssetPrimInfo:
    """Extract API schema info from a USD prim."""
    try:
        import pxr.UsdPhysics as UsdPhysics
    except ImportError:
        return AssetPrimInfo(path=path)

    info = AssetPrimInfo(path=path)
    info.has_articulation_root = prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    info.has_rigid_body = prim.HasAPI(UsdPhysics.RigidBodyAPI)
    info.has_mass_api = prim.HasAPI(UsdPhysics.MassAPI)

    if info.has_mass_api:
        mass_api = UsdPhysics.MassAPI(prim)
        mass_attr = mass_api.GetMassAttr()
        if mass_attr.IsValid():
            info.mass_kg = mass_attr.Get()

    return info
