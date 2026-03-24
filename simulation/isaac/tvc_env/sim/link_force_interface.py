"""
Per-link force application at COP using Isaac Lab's wrench composer API.

Uses Articulation.permanent_wrench_composer.set_forces_and_torques() with the positions parameter
per research decision R2, applying forces at fin COP offsets rather than link origins.

IMPORTANT: Forces must be in Isaac world frame (not body-FRD frame).
Frame conversion is the caller's responsibility (via frames.py boundary).
"""

from __future__ import annotations
import torch
from torch import Tensor
from tvc_env.common.frames import frd_position_to_isaac


class LinkForceInterface:
    """Apply external forces at fin COP positions via Isaac Lab's wrench composer."""

    def __init__(self, articulation, art_map, cop_positions_body: Tensor):
        """
        Args:
            articulation: Isaac Lab Articulation object.
            art_map: ArticulationMap with fin body index mapping.
            cop_positions_body: Tensor (4, 3) — COP offsets in body-FRD frame (m).
                                These are transformed to world frame at each step.
        """
        self._art = articulation
        self._map = art_map
        self._cop_positions_body = cop_positions_body  # (4, 3)

    def apply_fin_forces_at_cop(
        self,
        forces_world: Tensor,
        torques_world: Tensor,
        root_quaternion_wxyz: Tensor,
    ) -> None:
        """Apply external forces and torques at each fin's COP.

        Forces are applied at the COP offset from each fin's link origin,
        using Isaac Lab's permanent wrench composer with the positions argument.

        Args:
            forces_world: Tensor (num_envs, 4, 3) — force per fin in Isaac world frame (N).
            torques_world: Tensor (num_envs, 4, 3) — torque per fin in Isaac world frame (N·m).
            root_quaternion_wxyz: Tensor (num_envs, 4) — body orientation (w,x,y,z)
                                  used to transform COP offsets to world frame.
        """
        from tvc_env.common.quaternions import rotate_vector

        device = torch.device(self._art.device)
        forces_world = forces_world.to(device=device)
        torques_world = torques_world.to(device=device)
        root_quaternion_wxyz = root_quaternion_wxyz.to(device=device)

        num_envs = forces_world.shape[0]
        num_fins = 4
        fin_body_ids = torch.tensor(self._map.fin_body_indices, device=device)

        # Transform COP offsets from body-FRD to world frame.
        # Metadata stores COPs in the body-FRD convention, so convert them to
        # Isaac body axes before applying the root orientation.
        cop_body = self._cop_positions_body.to(device=device).unsqueeze(0).expand(num_envs, -1, -1)
        cop_body_isaac = frd_position_to_isaac(cop_body)
        # q: (num_envs, 4) → (num_envs, 1, 4) → broadcast over 4 fins
        q = root_quaternion_wxyz.unsqueeze(1).expand(-1, num_fins, -1)
        cop_world = rotate_vector(q.reshape(-1, 4), cop_body_isaac.reshape(-1, 3)).reshape(num_envs, num_fins, 3)

        # Apply forces at COP via Isaac Lab API
        # set_forces_and_torques expects:
        #   forces: (num_envs, num_bodies, 3)
        #   torques: (num_envs, num_bodies, 3)
        #   body_ids: (num_bodies,)
        #   positions: (num_envs, num_bodies, 3) — offset from body origin in world frame
        self._art.permanent_wrench_composer.set_forces_and_torques(
            forces=forces_world,
            torques=torques_world,
            body_ids=fin_body_ids,
            positions=cop_world,
        )

    def apply_body_force(
        self,
        force_world: Tensor,
        body_id: int,
    ) -> None:
        """Apply an external force to a single body link (e.g. EDF thrust, wind drag).

        Args:
            force_world: Tensor (num_envs, 3) — force in Isaac world frame (N).
            body_id: Articulation body index for the target link.
        """
        device = torch.device(self._art.device)
        force_world = force_world.to(device=device)
        num_envs = force_world.shape[0]

        # Reshape to (num_envs, 1, 3) for single body
        forces = force_world.unsqueeze(1)
        torques = torch.zeros_like(forces)
        body_ids = torch.tensor([body_id], device=device)

        self._art.permanent_wrench_composer.set_forces_and_torques(
            forces=forces,
            torques=torques,
            body_ids=body_ids,
        )

    def clear_external_forces(self) -> None:
        """Zero out all external forces on fin links."""
        num_envs = self._art.num_instances
        num_fins = 4
        zeros = torch.zeros(num_envs, num_fins, 3, device=self._art.device)
        fin_body_ids = torch.tensor(self._map.fin_body_indices, device=self._art.device)
        self._art.permanent_wrench_composer.set_forces_and_torques(
            forces=zeros,
            torques=zeros,
            body_ids=fin_body_ids,
        )

    def write_data_to_sim(self) -> None:
        """Flush all pending external force writes to the simulation.

        Must be called after setting wrench composer state and before stepping.
        """
        self._art.write_data_to_sim()
