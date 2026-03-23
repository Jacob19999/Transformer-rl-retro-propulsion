"""
Fin spatial layout and COP position computation in body-FRD frame.

Computes fin COP positions from asset metadata, provides fin-local-to-body
transforms, and validates fin ordering (+X, +Y, -X, -Y).
"""

from __future__ import annotations
import torch
from torch import Tensor
from typing import Any

from tvc_env.common.quaternions import rotate_vector
from tvc_env.common.transforms import axis_angle_to_quat

# Canonical fin position labels in +X, +Y, -X, -Y order
FIN_LABELS = ["+X", "+Y", "-X", "-Y"]
NUM_FINS = 4


def load_cop_positions(metadata: dict[str, Any], device: torch.device = None, dtype: torch.dtype = torch.float32) -> Tensor:
    """Load fin COP positions from asset metadata.

    Args:
        metadata: Asset metadata dict with 'fin_cop_positions' key.
        device: Target device.
        dtype: Target dtype.

    Returns:
        Tensor of shape (4, 3) — COP positions in body-FRD frame (m).
    """
    cops = metadata["fin_cop_positions"]
    if len(cops) != NUM_FINS:
        raise ValueError(f"Expected 4 fin_cop_positions in metadata, got {len(cops)}")
    return torch.tensor(cops, dtype=dtype, device=device)


def load_hinge_axes(metadata: dict[str, Any], device: torch.device = None, dtype: torch.dtype = torch.float32) -> Tensor:
    """Load fin hinge axes from asset metadata.

    Args:
        metadata: Asset metadata dict with 'hinge_axes' key.
        device: Target device.
        dtype: Target dtype.

    Returns:
        Tensor of shape (4, 3) — unit hinge axis vectors in body-FRD frame.
    """
    axes = metadata["hinge_axes"]
    if len(axes) != NUM_FINS:
        raise ValueError(f"Expected 4 hinge_axes in metadata, got {len(axes)}")
    return torch.tensor(axes, dtype=dtype, device=device)


def compute_fin_body_transforms(
    hinge_axes: Tensor,
    fin_deflections: Tensor,
) -> Tensor:
    """Compute rotation quaternions from fin-local to body-FRD frame for each deflected fin.

    At zero deflection, fin-local frame aligns with body-FRD frame.
    At non-zero deflection, the fin is rotated around its hinge axis.

    Args:
        hinge_axes: Tensor of shape (4, 3) — unit hinge axes in body-FRD frame.
        fin_deflections: Tensor of shape (num_envs, 4) — current fin deflection angles (rad).

    Returns:
        Tensor of shape (num_envs, 4, 4) — quaternions (w,x,y,z) for fin-local→body rotation.
    """
    from tvc_env.common.transforms import axis_angle_to_quat
    num_envs = fin_deflections.shape[0]
    # hinge_axes: (4, 3) → expand to (num_envs, 4, 3)
    axes_expanded = hinge_axes.unsqueeze(0).expand(num_envs, -1, -1)
    # deflections: (num_envs, 4) → (num_envs, 4)
    return axis_angle_to_quat(axes_expanded, fin_deflections)


def validate_fin_ordering(metadata: dict[str, Any]) -> None:
    """Check that fin metadata is consistent with +X, +Y, -X, -Y ordering.

    Raises ValueError if COP positions do not roughly match expected ordering.
    """
    cops = metadata.get("fin_cop_positions", [])
    if len(cops) < 4:
        return
    # +X fin should have positive x COP component (relative to others)
    # +Y fin should have positive y COP component
    # This is a soft sanity check, not a hard constraint
    issues = []
    if cops[0][0] <= 0:
        issues.append(f"+X fin COP has non-positive x: {cops[0]}")
    if cops[1][1] <= 0:
        issues.append(f"+Y fin COP has non-positive y: {cops[1]}")
    if cops[2][0] >= 0:
        issues.append(f"-X fin COP has non-negative x: {cops[2]}")
    if cops[3][1] >= 0:
        issues.append(f"-Y fin COP has non-negative y: {cops[3]}")
    if issues:
        raise ValueError(
            "Fin COP positions do not match expected +X,+Y,-X,-Y ordering:\n"
            + "\n".join(issues)
        )
