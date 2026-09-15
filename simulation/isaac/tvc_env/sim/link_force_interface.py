"""
Per-link force application at COP using Isaac Lab's instantaneous wrench composer API.

Uses Articulation.instantaneous_wrench_composer.set_forces_and_torques() with the
positions parameter per research decision R2, applying forces at fin COP offsets
rather than link origins.

IMPORTANT: Forces must be in Isaac world frame (not body-FRD frame).
Frame conversion is the caller's responsibility (via frames.py boundary).
"""

from __future__ import annotations
import torch
from torch import Tensor
from tvc_env.common.frames import frd_position_to_isaac
from tvc_env.common.quaternions import rotate_vector, inverse


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
        self._cop_positions_fin_local = None

    def get_fin_cop_positions_world(self, root_quaternion_wxyz: Tensor, root_position_w: Tensor) -> Tensor:
        """Attach each metadata COP to its moving fin, calibrated at zero joints.

        Initialize immediately after scene construction, before commanding fins.
        Metadata positions are neutral body-frame points, not fixed world points.
        """
        ids = self._map.fin_body_indices
        fin_q = self._art.data.body_link_quat_w[:, ids]
        fin_pos = self._art.data.body_link_pos_w[:, ids]
        if self._cop_positions_fin_local is None:
            neutral = frd_position_to_isaac(self._cop_positions_body.to(root_position_w))
            world = root_position_w[:, None] + rotate_vector(
                root_quaternion_wxyz[:, None].expand(-1, len(ids), -1),
                neutral[None].expand(root_position_w.shape[0], -1, -1),
            )
            self._cop_positions_fin_local = rotate_vector(inverse(fin_q), world - fin_pos)
        return fin_pos + rotate_vector(fin_q, self._cop_positions_fin_local)

    def apply_fin_forces_at_cop(
        self,
        forces_world: Tensor,
        torques_world: Tensor | None = None,
        root_quaternion_wxyz: Tensor | None = None,
        root_position_w: Tensor | None = None,
    ) -> None:
        """Apply external fin forces at each fin's COP.

        Compose the COP moment explicitly about the link COM in world space.
        Isaac Lab 2.3.2's installed WrenchComposer position kernel crosses a
        world lever arm with a local force, and its set path overwrites a
        supplied torque when positions is nonempty. Its downstream PhysX call
        applies the resultant at COM. Passing positions=None avoids both bugs.

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

        cop_world = self.get_fin_cop_positions_world(root_quaternion_wxyz, root_position_w)
        moment_world = torch.linalg.cross(
            cop_world - self._art.data.body_com_pos_w[:, self._map.fin_body_indices], forces_world)

        self._art.instantaneous_wrench_composer.set_forces_and_torques(
            forces=forces_world,
            torques=moment_world,
            body_ids=fin_body_ids,
            is_global=True,
        )

    def get_cop_velocity_relative_body_frd(self, cop_world: Tensor, root_quaternion: Tensor) -> Tensor:
        """Actual articulation COP velocity, relative to translating body origin.

        Use COM velocities with COM lever arms; mixing link-origin positions
        with COM velocity would create a spurious rotational inflow.
        """
        from tvc_env.common.frames import isaac_velocity_to_frd
        ids = self._map.fin_body_indices
        data = self._art.data
        velocity = data.body_com_lin_vel_w[:, ids] + torch.linalg.cross(
            data.body_com_ang_vel_w[:, ids], cop_world - data.body_com_pos_w[:, ids])
        velocity -= data.body_link_lin_vel_w[:, self._map.body_index, None]
        return isaac_velocity_to_frd(rotate_vector(inverse(root_quaternion)[:, None].expand(-1, 4, -1), velocity))

    def apply_body_wrench(
        self,
        force_world: Tensor,
        torque_world: Tensor,
        body_id: int,
        position_world: Tensor | None = None,
    ) -> None:
        """Apply force at an explicit body point, or COM when none is supplied.

        EDF thrust belongs to the airframe thrust line, not the randomized
        COM. Compute the application-point moment once, explicitly, because
        the installed composer position path uses mixed frames and overwrites
        supplied rotor reaction torques (2026-09-15 source/runtime audit).
        """
        device = torch.device(self._art.device)
        force_world = force_world.to(device=device)
        torque_world = torque_world.to(device=device)
        if position_world is not None:
            torque_world = torque_world + torch.linalg.cross(
                position_world.to(device=device) - self._art.data.body_com_pos_w[:, body_id], force_world)

        forces = force_world.unsqueeze(1)
        torques = torque_world.unsqueeze(1)
        body_ids = torch.tensor([body_id], device=device)

        self._art.instantaneous_wrench_composer.set_forces_and_torques(
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

    def clear_external_wrenches(self, env_ids: Tensor | None = None) -> None:
        """Clear all permanent external wrench slots for selected environments.

        Clear both composers before episode reset propagation so stale
        terminal-step forces cannot affect the new episode.
        """
        if env_ids is not None:
            env_ids = env_ids.to(device=self._art.device, dtype=torch.int64)
        self._art.instantaneous_wrench_composer.reset(env_ids)
        self._art.permanent_wrench_composer.reset(env_ids)

    def clear_external_forces(self) -> None:
        """Zero out external forces on fin links.

        Kept for compatibility with older tests/scripts. Prefer
        :meth:`clear_external_wrenches` when resetting an environment because
        body-level thrust and torque slots must be cleared too.
        """
        num_envs = self._art.num_instances
        num_fins = 4
        zeros = torch.zeros(num_envs, num_fins, 3, device=self._art.device)
        fin_body_ids = torch.tensor(self._map.fin_body_indices, device=self._art.device)
        self._art.instantaneous_wrench_composer.set_forces_and_torques(
            forces=zeros,
            torques=zeros,
            body_ids=fin_body_ids,
        )

    def write_data_to_sim(self) -> None:
        """Flush all pending external force writes to the simulation."""
        self._art.write_data_to_sim()
