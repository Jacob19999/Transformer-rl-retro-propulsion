"""
Per-fin force computation pipeline.

Orchestrates: get actual fin angles from servo state → compute aero forces via
fin_aero.py → transform force vectors from fin-local to body frame using
fin_geometry.py → output per-fin force vectors at COP positions ready for dispatch.
"""

from __future__ import annotations
import torch
from torch import Tensor

from tvc_env.dynamics.fin_aero import FinAeroModel
from tvc_env.dynamics.fin_geometry import load_cop_positions, load_hinge_axes
from tvc_env.common.quaternions import rotate_vector
from tvc_env.common.transforms import axis_angle_to_quat


class FinForceDispatch:
    """Orchestrates fin aero force computation and transformation to body frame."""

    def __init__(
        self,
        aero_model: FinAeroModel,
        cop_positions: Tensor,     # (4, 3) in body-FRD
        hinge_axes: Tensor,        # (4, 3) in body-FRD
    ):
        self.aero_model = aero_model
        self.cop_positions = cop_positions
        self.hinge_axes = hinge_axes

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
        aero_model = FinAeroModel.from_config(vehicle_config, edf_config)
        cops = load_cop_positions(metadata, device=device, dtype=dtype)
        axes = load_hinge_axes(metadata, device=device, dtype=dtype)
        return cls(aero_model, cops, axes)

    def compute_body_frame_forces(
        self,
        fin_angles: Tensor,        # (num_envs, 4) — actual servo positions (rad)
        throttle: Tensor,          # (num_envs,) — normalized throttle [0, 1]
    ) -> tuple[Tensor, Tensor]:
        """Compute per-fin forces in body-FRD frame at COP positions.

        Pipeline:
          1. Compute fin-local aero forces via fin_aero.py
          2. Compute fin-local-to-body rotation quaternion per fin deflection
          3. Rotate force vectors from fin-local to body-FRD frame

        Args:
            fin_angles: Tensor (num_envs, 4) of actual fin angles (rad).
            throttle: Tensor (num_envs,) normalized throttle.

        Returns:
            Tuple (forces_body, cop_positions):
              - forces_body: Tensor (num_envs, 4, 3) force per fin in body-FRD frame (N)
              - cop_positions: Tensor (4, 3) COP offsets (same for all envs)
        """
        num_envs = fin_angles.shape[0]

        # Step 1: Get aero forces in fin-local frame
        aero_result = self.aero_model.compute_forces(fin_angles, throttle)
        forces_fin_local = aero_result.force_vector  # (num_envs, 4, 3)

        # Step 2: Compute fin-local → body-FRD rotation quaternion
        # Each fin is rotated around its hinge axis by its deflection angle
        hinge_axes = self.hinge_axes.to(fin_angles.device)  # (4, 3)
        hinge_axes_batch = hinge_axes.unsqueeze(0).expand(num_envs, -1, -1)  # (num_envs, 4, 3)
        fin_quats = axis_angle_to_quat(hinge_axes_batch, fin_angles)  # (num_envs, 4, 4)

        # Step 3: Rotate forces from fin-local to body-FRD
        forces_flat = forces_fin_local.reshape(-1, 3)   # (num_envs*4, 3)
        quats_flat = fin_quats.reshape(-1, 4)            # (num_envs*4, 4)
        forces_body_flat = rotate_vector(quats_flat, forces_flat)  # (num_envs*4, 3)
        forces_body = forces_body_flat.reshape(num_envs, 4, 3)    # (num_envs, 4, 3)

        return forces_body, self.cop_positions
