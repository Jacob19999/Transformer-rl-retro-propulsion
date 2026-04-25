"""
Per-fin jet-vane force computation pipeline.

The EDF flow axis is body +Z_frd. For each fin, metadata defines the hinge
axis and the zero-deflection normal as ``hinge_axis x flow_axis``. The aero
model supplies scalar normal force and drag magnitudes; this dispatcher orients
the side force in body-FRD and exposes drag as bounded EDF thrust loss.
"""

from __future__ import annotations

import torch
from torch import Tensor

from tvc_env.common.datatypes import FinDispatchResult
from tvc_env.dynamics.fin_aero import FinAeroModel
from tvc_env.dynamics.fin_geometry import (
    load_cop_positions,
    load_fin_chord_directions,
    load_fin_normal_directions,
    load_hinge_axes,
    validate_fin_force_geometry,
)


class FinForceDispatch:
    """Orchestrates fin aero scalar computation and body-frame orientation."""

    def __init__(
        self,
        aero_model: FinAeroModel,
        cop_positions: Tensor,     # (4, 3) in body-FRD
        hinge_axes: Tensor,        # (4, 3) in body-FRD
        chord_dirs: Tensor,        # (4, 3) in body-FRD
        normal_dirs: Tensor,       # (4, 3) in body-FRD
    ):
        self.aero_model = aero_model
        self.cop_positions = cop_positions
        self.hinge_axes = hinge_axes
        self.chord_dirs = chord_dirs
        self.normal_dirs = normal_dirs

    @classmethod
    def from_metadata_and_config(
        cls,
        metadata: dict,
        vehicle_config: dict,
        edf_config: dict,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ) -> "FinForceDispatch":
        """Create from asset metadata and YAML configs."""
        validate_fin_force_geometry(metadata)
        aero_model = FinAeroModel.from_config(vehicle_config, edf_config)
        cops = load_cop_positions(metadata, device=device, dtype=dtype)
        axes = load_hinge_axes(metadata, device=device, dtype=dtype)
        chords = load_fin_chord_directions(metadata, device=device, dtype=dtype)
        normals = load_fin_normal_directions(metadata, device=device, dtype=dtype)
        return cls(aero_model, cops, axes, chords, normals)

    def compute_body_frame_forces(
        self,
        fin_angles: Tensor,        # (num_envs, 4), measured fin positions (rad)
        throttle: Tensor,          # (num_envs,), normalized throttle [0, 1]
    ) -> FinDispatchResult:
        """Compute per-fin side forces in body-FRD at COP positions.

        The normal force magnitude is odd in fin angle and is applied along
        the zero-deflection radial normal. This avoids artificial axial force
        injection from symmetric fin commands while preserving control torque
        signs through the force magnitude.
        """
        aero_result = self.aero_model.compute_forces(fin_angles, throttle)
        normals = self.normal_dirs.to(device=fin_angles.device, dtype=fin_angles.dtype)
        forces_body = aero_result.normal_force.unsqueeze(-1) * normals.unsqueeze(0)

        return FinDispatchResult(
            forces_body=forces_body,
            cop_positions=self.cop_positions.to(device=fin_angles.device, dtype=fin_angles.dtype),
            thrust_loss=aero_result.thrust_loss,
            normal_force=aero_result.normal_force,
            tangential_force=aero_result.tangential_force,
        )
