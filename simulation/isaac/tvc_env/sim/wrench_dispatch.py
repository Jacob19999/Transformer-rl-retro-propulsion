"""
Force dispatch mode switching.

Abstract dispatch layer per research decision R10:
  - per_link_force: applies forces via link_force_interface.py to each fin at COP
  - collapsed_body_wrench: sums all forces into net body wrench

Mode is selected from env config 'dispatch_mode' field.
"""

from __future__ import annotations
import torch
from torch import Tensor
from tvc_env.common.constants import DispatchMode
from tvc_env.common.frames import frd_force_to_isaac


class WrenchDispatch:
    """Routes fin force vectors to the appropriate dispatch implementation."""

    def __init__(
        self,
        mode: DispatchMode | str,
        link_force_interface=None,
        body_wrench_interface=None,
    ):
        if isinstance(mode, str):
            mode_map = {
                "per_link_force": DispatchMode.PER_LINK_FORCE,
                "collapsed_body_wrench": DispatchMode.COLLAPSED_BODY_WRENCH,
            }
            mode = mode_map.get(mode, DispatchMode.PER_LINK_FORCE)
        self.mode = mode
        self._link_iface = link_force_interface
        self._body_iface = body_wrench_interface

    def dispatch(
        self,
        forces_body_frd: Tensor,       # (num_envs, 4, 3) in body-FRD frame
        cop_positions: Tensor,          # (4, 3) in body-FRD
        root_quaternion_wxyz: Tensor,   # (num_envs, 4)
        edf_force_body: Tensor,         # (num_envs, 3) EDF thrust in body-FRD
    ) -> None:
        """Dispatch forces to simulation.

        Args:
            forces_body_frd: Per-fin aero forces in body-FRD frame (N).
            cop_positions: Fin COP positions in body-FRD frame (m).
            root_quaternion_wxyz: Body orientation (w,x,y,z).
            edf_force_body: EDF thrust force in body-FRD frame (N).
        """
        if self.mode == DispatchMode.PER_LINK_FORCE:
            self._dispatch_per_link(forces_body_frd, root_quaternion_wxyz)
        elif self.mode == DispatchMode.COLLAPSED_BODY_WRENCH:
            self._dispatch_collapsed(forces_body_frd, cop_positions, root_quaternion_wxyz, edf_force_body)

    def _dispatch_per_link(
        self,
        forces_body_frd: Tensor,
        root_quaternion_wxyz: Tensor,
    ) -> None:
        """Per-link mode: apply force at each fin COP via link_force_interface."""
        if self._link_iface is None:
            raise RuntimeError("per_link_force mode requires a LinkForceInterface")

        # Convert forces from body-FRD to Isaac world frame
        num_envs, num_fins, _ = forces_body_frd.shape
        forces_flat = forces_body_frd.reshape(-1, 3)  # (num_envs*4, 3)
        forces_world_flat = frd_force_to_isaac(forces_flat)
        forces_world = forces_world_flat.reshape(num_envs, num_fins, 3)

        torques_world = torch.zeros_like(forces_world)

        self._link_iface.apply_fin_forces_at_cop(
            forces_world, torques_world, root_quaternion_wxyz
        )
        self._link_iface.write_data_to_sim()

    def _dispatch_collapsed(
        self,
        forces_body_frd: Tensor,
        cop_positions: Tensor,
        root_quaternion_wxyz: Tensor,
        edf_force_body: Tensor,
    ) -> None:
        """Collapsed mode: sum all fin forces into net body wrench."""
        if self._body_iface is None:
            raise RuntimeError("collapsed_body_wrench mode requires a body wrench interface")

        # Sum forces
        total_force_body = forces_body_frd.sum(dim=1) + edf_force_body  # (num_envs, 3)

        # Compute torques from r × F for each fin
        # cop_positions: (4, 3), forces: (num_envs, 4, 3)
        cops = cop_positions.unsqueeze(0).expand(forces_body_frd.shape[0], -1, -1)
        torques = torch.linalg.cross(cops, forces_body_frd)  # (num_envs, 4, 3)
        total_torque_body = torques.sum(dim=1)  # (num_envs, 3)

        # Convert to Isaac world frame
        total_force_world = frd_force_to_isaac(total_force_body)
        total_torque_world = frd_force_to_isaac(total_torque_body)

        self._body_iface.apply_body_wrench(total_force_world, total_torque_world)
