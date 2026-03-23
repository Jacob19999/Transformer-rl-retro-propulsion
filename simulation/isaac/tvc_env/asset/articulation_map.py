"""
Link/joint name-to-index mapping for the EDF drone articulation.

Maps fin_link_names and fin_joint_names from asset metadata YAML to
Isaac Lab Articulation body/joint indices, with positional lookup (+X, +Y, -X, -Y).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any


# Canonical fin position labels in order
FIN_POSITIONS = ["+X", "+Y", "-X", "-Y"]


@dataclass
class ArticulationMap:
    """Mapping between fin names, positions, and Isaac articulation indices."""
    body_index: int
    fin_body_indices: list[int]   # Articulation body indices for each fin link
    fin_joint_indices: list[int]  # Articulation joint indices for each fin joint
    fin_link_names: list[str]     # Name-ordered list
    fin_joint_names: list[str]
    position_to_idx: dict[str, int]  # "+X" → 0, "+Y" → 1, etc.
    name_to_body_idx: dict[str, int]
    name_to_joint_idx: dict[str, int]

    def body_idx_for_position(self, position: str) -> int:
        """Get Isaac body index for a fin by position label (+X, +Y, -X, -Y)."""
        idx = self.position_to_idx[position]
        return self.fin_body_indices[idx]

    def joint_idx_for_position(self, position: str) -> int:
        """Get Isaac joint index for a fin by position label."""
        idx = self.position_to_idx[position]
        return self.fin_joint_indices[idx]

    def body_idx_for_name(self, name: str) -> int:
        """Get Isaac body index for a fin by link name."""
        return self.name_to_body_idx[name]

    def joint_idx_for_name(self, name: str) -> int:
        """Get Isaac joint index for a fin by joint name."""
        return self.name_to_joint_idx[name]


def build_articulation_map(
    metadata: dict[str, Any],
    articulation,
) -> ArticulationMap:
    """Build an ArticulationMap from asset metadata and an Isaac Lab Articulation object.

    Args:
        metadata: Asset metadata dict from usd_loader.load_asset_metadata().
        articulation: Isaac Lab Articulation object (must be initialized).

    Returns:
        ArticulationMap with all index mappings resolved.

    Raises:
        ValueError: If any expected link or joint is not found in the articulation.
    """
    fin_link_names = metadata["fin_link_names"]
    fin_joint_names = metadata["fin_joint_names"]
    body_link_name = metadata["body_link_name"]

    # Get all body/joint names from articulation
    all_body_names = articulation.body_names
    all_joint_names = articulation.joint_names

    def find_index(names: list[str], target: str) -> int:
        matches = [i for i, n in enumerate(names) if n == target or n.endswith(f"/{target}") or n.endswith(f"_{target}")]
        if not matches:
            raise ValueError(f"'{target}' not found in articulation. Available: {names}")
        return matches[0]

    body_index = find_index(all_body_names, body_link_name)
    fin_body_indices = [find_index(all_body_names, name) for name in fin_link_names]
    fin_joint_indices = [find_index(all_joint_names, name) for name in fin_joint_names]

    position_to_idx = {pos: i for i, pos in enumerate(FIN_POSITIONS)}
    name_to_body_idx = {name: idx for name, idx in zip(fin_link_names, fin_body_indices)}
    name_to_joint_idx = {name: idx for name, idx in zip(fin_joint_names, fin_joint_indices)}

    return ArticulationMap(
        body_index=body_index,
        fin_body_indices=fin_body_indices,
        fin_joint_indices=fin_joint_indices,
        fin_link_names=fin_link_names,
        fin_joint_names=fin_joint_names,
        position_to_idx=position_to_idx,
        name_to_body_idx=name_to_body_idx,
        name_to_joint_idx=name_to_joint_idx,
    )
