"""
Per-link force application at COP using Isaac Lab's Articulation API.

Wraps Articulation.set_external_force_and_torque() with the positions parameter
per research decision R2, applying forces at fin COP offsets rather than link origins.

IMPORTANT: Forces must be in Isaac world frame (not body-FRD frame).
Frame conversion is the caller's responsibility (via frames.py boundary).
"""

from __future__ import annotations
import torch
from torch import Tensor


class LinkForceInterface:
    """Apply external forces at fin COP positions via Isaac Lab Articulation API."""

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
        using Isaac Lab's set_external_force_and_torque() with positions argument.

        Args:
            forces_world: Tensor (num_envs, 4, 3) — force per fin in Isaac world frame (N).
            torques_world: Tensor (num_envs, 4, 3) — torque per fin in Isaac world frame (N·m).
            root_quaternion_wxyz: Tensor (num_envs, 4) — body orientation (w,x,y,z)
                                  used to transform COP offsets to world frame.
        """
        from tvc_env.common.quaternions import rotate_vector

        num_envs = forces_world.shape[0]
        num_fins = 4
        fin_body_ids = torch.tensor(self._map.fin_body_indices, device=forces_world.device)

        # Transform COP offsets from body-FRD to world frame
        # cop_body: (4, 3) → (num_envs, 4, 3)
        cop_body = self._cop_positions_body.unsqueeze(0).expand(num_envs, -1, -1)
        # q: (num_envs, 4) → (num_envs, 1, 4) → broadcast over 4 fins
        q = root_quaternion_wxyz.unsqueeze(1).expand(-1, num_fins, -1)
        cop_world = rotate_vector(q.reshape(-1, 4), cop_body.reshape(-1, 3)).reshape(num_envs, num_fins, 3)

        # Apply forces at COP via Isaac Lab API
        # set_external_force_and_torque expects:
        #   forces: (num_envs, num_bodies, 3)
        #   torques: (num_envs, num_bodies, 3)
        #   body_ids: (num_bodies,)
        #   positions: (num_envs, num_bodies, 3) — offset from body origin in world frame
        self._art.set_external_force_and_torque(
            forces=forces_world,
            torques=torques_world,
            body_ids=fin_body_ids,
            positions=cop_world,
        )

    def clear_external_forces(self) -> None:
        """Zero out all external forces on fin links."""
        num_envs = self._art.num_instances
        num_fins = 4
        zeros = torch.zeros(num_envs, num_fins, 3, device=self._art.device)
        fin_body_ids = torch.tensor(self._map.fin_body_indices, device=self._art.device)
        self._art.set_external_force_and_torque(
            forces=zeros,
            torques=zeros,
            body_ids=fin_body_ids,
        )

    def write_data_to_sim(self) -> None:
        """Flush all pending external force writes to the simulation.

        Must be called after set_external_force_and_torque() and before stepping.
        """
        self._art.write_data_to_sim()
