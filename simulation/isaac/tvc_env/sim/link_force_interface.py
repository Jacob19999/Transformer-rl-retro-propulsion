"""
Per-link force application at COP using Isaac Lab's wrench composer API.

Uses Articulation.permanent_wrench_composer.set_forces_and_torques() with the
positions parameter per research decision R2, applying forces at fin COP offsets
rather than link origins.

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
            cop_positions_body: Tensor (4, 3) COP offsets in body-FRD frame (m).
                                These are transformed to world frame at each step.
        """
        self._art = articulation
        self._map = art_map
        self._cop_positions_body = cop_positions_body  # (4, 3)

    def refresh_link_poses(self) -> None:
        """Mark link poses as stale so the wrench composer re-reads them.

        The permanent wrench composer caches link poses and never refreshes
        them during normal stepping (only on articulation reset).  With
        ``is_global=True`` the kernel needs up-to-date link quaternions for
        the world-to-link rotation, so we must invalidate the cache each
        time we set new forces.
        """
        self._art.permanent_wrench_composer._link_poses_updated = False

    def apply_fin_forces_at_cop(
        self,
        forces_world: Tensor,
        torques_world: Tensor | None = None,
        root_quaternion_wxyz: Tensor | None = None,
        root_position_w: Tensor | None = None,
    ) -> None:
        """Apply external fin forces at each fin's COP.

        With ``is_global=True``, Isaac Lab expects force application positions
        to be world-absolute. It derives the COP moment internally as ``r x F``;
        explicit fin torques are intentionally ignored here to avoid
        double-counting or overwriting that moment.

        Args:
            forces_world: Tensor (num_envs, 4, 3) force per fin in Isaac world frame (N).
            torques_world: Deprecated compatibility argument; ignored.
            root_quaternion_wxyz: Tensor (num_envs, 4) body orientation (w,x,y,z)
                                  used to transform COP offsets to world frame.
            root_position_w: Tensor (num_envs, 3) body position in Isaac world frame.
        """
        from tvc_env.common.quaternions import rotate_vector

        device = torch.device(self._art.device)
        forces_world = forces_world.to(device=device)
        if root_quaternion_wxyz is None:
            root_quaternion_wxyz = self._art.data.root_quat_w.clone()
        root_quaternion_wxyz = root_quaternion_wxyz.to(device=device)
        if root_position_w is None:
            root_position_w = self._art.data.root_pos_w.clone()
        root_position_w = root_position_w.to(device=device)

        num_envs = forces_world.shape[0]
        num_fins = 4
        fin_body_ids = torch.tensor(self._map.fin_body_indices, device=device)

        # Transform COP offsets from body-FRD to world frame.
        # Metadata stores COPs in the body-FRD convention, so convert them to
        # Isaac body axes before applying the root orientation.
        cop_body = self._cop_positions_body.to(device=device).unsqueeze(0).expand(num_envs, -1, -1)
        cop_body_isaac = frd_position_to_isaac(cop_body)
        # q: (num_envs, 4) -> (num_envs, 1, 4) -> broadcast over 4 fins
        q = root_quaternion_wxyz.unsqueeze(1).expand(-1, num_fins, -1)
        cop_world_offset = rotate_vector(q.reshape(-1, 4), cop_body_isaac.reshape(-1, 3)).reshape(num_envs, num_fins, 3)
        cop_world = root_position_w.unsqueeze(1) + cop_world_offset

        self._art.permanent_wrench_composer.set_forces_and_torques(
            forces=forces_world,
            body_ids=fin_body_ids,
            positions=cop_world,
            is_global=True,
        )

    def apply_body_wrench(
        self,
        force_world: Tensor,
        torque_world: Tensor,
        body_id: int,
    ) -> None:
        """Apply a body-level force and torque with no COP position."""
        device = torch.device(self._art.device)
        force_world = force_world.to(device=device)
        torque_world = torque_world.to(device=device)

        forces = force_world.unsqueeze(1)
        torques = torque_world.unsqueeze(1)
        body_ids = torch.tensor([body_id], device=device)

        self._art.permanent_wrench_composer.set_forces_and_torques(
            forces=forces,
            torques=torques,
            body_ids=body_ids,
            is_global=True,
        )

    def apply_body_force(
        self,
        force_world: Tensor,
        body_id: int,
    ) -> None:
        """Apply an external force to a single body link (e.g. EDF thrust, wind drag).

        Args:
            force_world: Tensor (num_envs, 3) force in Isaac world frame (N).
            body_id: Articulation body index for the target link.
        """
        device = torch.device(self._art.device)
        force_world = force_world.to(device=device)
        torque_world = torch.zeros_like(force_world)
        self.apply_body_wrench(force_world, torque_world, body_id)

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
        """Flush all pending external force writes to the simulation."""
        self._art.write_data_to_sim()
