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
from tvc_env.common.quaternions import rotate_vector


class WrenchDispatch:
    """Routes fin force vectors to the appropriate dispatch implementation."""

    def __init__(
        self,
        mode: DispatchMode | str,
        link_force_interface=None,
        body_wrench_interface=None,
        body_link_index: int = 0,
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
        self._body_link_index = body_link_index

    @staticmethod
    def _body_frd_to_world(vectors_body_frd: Tensor, quaternion_wxyz: Tensor) -> Tensor:
        """Rotate body-FRD vectors into Isaac world frame."""
        vectors_body_isaac = frd_force_to_isaac(vectors_body_frd)
        return rotate_vector(quaternion_wxyz, vectors_body_isaac)

    def dispatch(
        self,
        forces_body_frd: Tensor,       # (num_envs, 4, 3) in body-FRD frame
        cop_positions: Tensor,          # (4, 3) in body-FRD
        root_quaternion_wxyz: Tensor,   # (num_envs, 4)
        root_position_w: Tensor,        # (num_envs, 3)
        edf_force_body_frd: Tensor,     # (num_envs, 3) EDF thrust in body-FRD
        edf_torque_body_frd: Tensor,    # (num_envs, 3) EDF reaction torque in body-FRD
        wind_force_body_frd: Tensor | None = None,  # (num_envs, 3) wind drag in body-FRD
    ) -> None:
        """Dispatch forces to simulation.

        Args:
            forces_body_frd: Per-fin aero forces in body-FRD frame (N).
            cop_positions: Fin COP positions in body-FRD frame (m).
            root_quaternion_wxyz: Body orientation (w,x,y,z).
            root_position_w: Body position in Isaac world frame.
            edf_force_body_frd: EDF thrust force in body-FRD frame (N).
            edf_torque_body_frd: EDF body reaction torque in body-FRD frame (N*m).
            wind_force_body_frd: Wind drag force in body-FRD frame (N), optional.
        """
        # Combine all body-level forces
        body_force = edf_force_body_frd
        if wind_force_body_frd is not None:
            body_force = body_force + wind_force_body_frd

        if self.mode == DispatchMode.PER_LINK_FORCE:
            self._dispatch_per_link(
                forces_body_frd, root_quaternion_wxyz, root_position_w, body_force, edf_torque_body_frd
            )
        elif self.mode == DispatchMode.COLLAPSED_BODY_WRENCH:
            self._dispatch_collapsed(
                forces_body_frd, cop_positions, root_quaternion_wxyz, body_force, edf_torque_body_frd
            )

    def _dispatch_per_link(
        self,
        forces_body_frd: Tensor,
        root_quaternion_wxyz: Tensor,
        root_position_w: Tensor,
        body_force_frd: Tensor,
        body_torque_frd: Tensor,
    ) -> None:
        """Per-link mode: apply force at each fin COP via link_force_interface."""
        if self._link_iface is None:
            raise RuntimeError("per_link_force mode requires a LinkForceInterface")

        # Convert forces from body-FRD to Isaac world frame
        num_envs, num_fins, _ = forces_body_frd.shape
        forces_flat = forces_body_frd.reshape(-1, 3)  # (num_envs*4, 3)
        q_flat = root_quaternion_wxyz.unsqueeze(1).expand(-1, num_fins, -1).reshape(-1, 4)
        forces_world_flat = self._body_frd_to_world(forces_flat, q_flat)
        forces_world = forces_world_flat.reshape(num_envs, num_fins, 3)

        self._link_iface.apply_fin_forces_at_cop(
            forces_world,
            None,
            root_quaternion_wxyz,
            root_position_w,
        )

        # Apply body-level EDF thrust/wind force and EDF reaction torque on the body link.
        body_force_world = self._body_frd_to_world(body_force_frd, root_quaternion_wxyz)
        body_torque_world = self._body_frd_to_world(body_torque_frd, root_quaternion_wxyz)
        self._link_iface.apply_body_wrench(
            body_force_world,
            body_torque_world,
            body_id=self._body_link_index,
        )

        # TVCSimScene.step() performs the single scene.write_data_to_sim()
        # immediately after action application. Do not flush the articulation
        # here as well; that would run actuator conversion twice per substep.

    def _dispatch_collapsed(
        self,
        forces_body_frd: Tensor,
        cop_positions: Tensor,
        root_quaternion_wxyz: Tensor,
        body_force_frd: Tensor,
        body_torque_frd: Tensor,
    ) -> None:
        """Collapsed mode: sum all fin forces into net body wrench."""
        if self._link_iface is None:
            raise RuntimeError("collapsed_body_wrench mode requires a LinkForceInterface")

        # Sum forces
        total_force_body = forces_body_frd.sum(dim=1) + body_force_frd  # (num_envs, 3)

        # Compute torques from r × F for each fin
        # cop_positions: (4, 3), forces: (num_envs, 4, 3)
        cops = cop_positions.unsqueeze(0).expand(forces_body_frd.shape[0], -1, -1)
        torques = torch.linalg.cross(cops, forces_body_frd)  # (num_envs, 4, 3)
        total_torque_body = torques.sum(dim=1) + body_torque_frd  # (num_envs, 3)

        # Convert to Isaac world frame
        total_force_world = self._body_frd_to_world(total_force_body, root_quaternion_wxyz)
        total_torque_world = self._body_frd_to_world(total_torque_body, root_quaternion_wxyz)

        self._link_iface.apply_body_wrench(
            total_force_world,
            total_torque_world,
            body_id=self._body_link_index,
        )
        # Flushed once by TVCSimScene.step().
