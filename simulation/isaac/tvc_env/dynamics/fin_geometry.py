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
EDF_FLOW_AXIS_FRD = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)


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


def load_fin_chord_directions(
    metadata: dict[str, Any],
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Load zero-deflection fin chord/flow directions in body-FRD frame."""
    chord_dirs = metadata["fin_chord_directions"]
    if len(chord_dirs) != NUM_FINS:
        raise ValueError(f"Expected 4 fin_chord_directions in metadata, got {len(chord_dirs)}")
    return torch.tensor(chord_dirs, dtype=dtype, device=device)


def load_fin_normal_directions(
    metadata: dict[str, Any],
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Load zero-deflection fin normal directions in body-FRD frame."""
    normal_dirs = metadata["fin_normal_directions"]
    if len(normal_dirs) != NUM_FINS:
        raise ValueError(f"Expected 4 fin_normal_directions in metadata, got {len(normal_dirs)}")
    return torch.tensor(normal_dirs, dtype=dtype, device=device)


def validate_fin_force_geometry(metadata: dict[str, Any], tolerance: float = 1e-5) -> None:
    """Validate the jet-vane force basis in metadata.

    The EDF flow/chord axis is +Z_frd. Each fin normal must be radial and
    equal to ``hinge_axis x chord_axis`` so normal force cannot leak axial
    thrust into symmetric fin commands.
    """
    axes = load_hinge_axes(metadata, dtype=torch.float64)
    chords = load_fin_chord_directions(metadata, dtype=torch.float64)
    normals = load_fin_normal_directions(metadata, dtype=torch.float64)
    flow = torch.tensor(metadata.get("edf_thrust_axis", [0.0, 0.0, 1.0]), dtype=torch.float64)
    flow = flow / flow.norm().clamp(min=1e-12)

    issues = []
    for i in range(NUM_FINS):
        hinge_norm = axes[i].norm().clamp(min=1e-12)
        chord_norm = chords[i].norm().clamp(min=1e-12)
        normal_norm = normals[i].norm().clamp(min=1e-12)
        hinge = axes[i] / hinge_norm
        chord = chords[i] / chord_norm
        normal = normals[i] / normal_norm
        expected = torch.linalg.cross(hinge, flow)
        expected = expected / expected.norm().clamp(min=1e-12)

        if abs(float(chord_norm.item()) - 1.0) > tolerance:
            issues.append(f"fin {i} chord is not unit length: {chords[i].tolist()}")
        if abs(float(normal_norm.item()) - 1.0) > tolerance:
            issues.append(f"fin {i} normal is not unit length: {normals[i].tolist()}")
        if abs(float(torch.dot(normal, hinge).item())) > tolerance:
            issues.append(f"fin {i} normal is not orthogonal to hinge")
        if abs(float(torch.dot(normal, flow).item())) > tolerance:
            issues.append(f"fin {i} normal is not orthogonal to EDF flow")
        if not torch.allclose(chord, flow, atol=tolerance, rtol=0.0):
            issues.append(f"fin {i} chord must equal EDF flow axis {flow.tolist()}")
        if not torch.allclose(normal, expected, atol=tolerance, rtol=0.0):
            issues.append(
                f"fin {i} normal must equal hinge x flow; got {normal.tolist()}, expected {expected.tolist()}"
            )

    if issues:
        raise ValueError("Invalid fin force geometry:\n" + "\n".join(issues))


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
