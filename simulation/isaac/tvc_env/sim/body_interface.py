"""
Articulation state read/write interface for the EDF drone.

Reads root_state_w (position, quaternion wxyz, linear/angular velocity),
reads/writes joint positions and velocities, and provides body-frame velocity
computation using the frames.py conversion boundary.

All quaternions are in (w,x,y,z) convention per Isaac Lab 2.3.2.
Velocities in body-FRD frame are computed via frames.py (single boundary).
"""

from __future__ import annotations
import torch
from torch import Tensor

from tvc_env.common.frames import isaac_velocity_to_frd
from tvc_env.common.quaternions import rotate_vector, inverse as quat_inv, normalize


class BodyInterface:
    """Interface for reading and writing articulation body state."""

    def __init__(self, articulation, art_map):
        """
        Args:
            articulation: Isaac Lab Articulation object.
            art_map: ArticulationMap with body/joint index mappings.
        """
        self._art = articulation
        self._map = art_map

    # ---- Root state reading ----

    def get_root_position(self) -> Tensor:
        """Get root body position in Isaac world frame.

        Returns:
            Tensor of shape (num_envs, 3) in Isaac world frame (m).
        """
        return self._art.data.root_pos_w.clone()

    def get_root_quaternion_wxyz(self) -> Tensor:
        """Get root body orientation as (w,x,y,z) quaternion.

        Returns:
            Tensor of shape (num_envs, 4) in (w,x,y,z) ordering.
        """
        return self._art.data.root_quat_w.clone()

    def get_root_linear_velocity_world(self) -> Tensor:
        """Get root linear velocity in Isaac world frame.

        Returns:
            Tensor of shape (num_envs, 3) in Isaac world frame (m/s).
        """
        return self._art.data.root_lin_vel_w.clone()

    def get_root_angular_velocity_world(self) -> Tensor:
        """Get root angular velocity in Isaac world frame.

        Returns:
            Tensor of shape (num_envs, 3) in Isaac world frame (rad/s).
        """
        return self._art.data.root_ang_vel_w.clone()

    def get_linear_velocity_body_frd(self) -> Tensor:
        """Get root linear velocity in body-FRD frame.

        Converts from Isaac world frame using frames.py (single conversion boundary).

        Returns:
            Tensor of shape (num_envs, 3) in body-FRD frame (m/s).
        """
        vel_world = self.get_root_linear_velocity_world()
        q_wxyz = self.get_root_quaternion_wxyz()
        # Rotate world velocity into body frame: v_body = R^T * v_world
        q_inv = quat_inv(normalize(q_wxyz))
        vel_body_isaac = rotate_vector(q_inv, vel_world)
        # Convert from Isaac body convention to FRD
        return isaac_velocity_to_frd(vel_body_isaac)

    def get_angular_velocity_body_frd(self) -> Tensor:
        """Get root angular velocity in body-FRD frame.

        Returns:
            Tensor of shape (num_envs, 3) in body-FRD frame (rad/s).
        """
        ang_vel_world = self.get_root_angular_velocity_world()
        q_wxyz = self.get_root_quaternion_wxyz()
        q_inv = quat_inv(normalize(q_wxyz))
        ang_vel_body_isaac = rotate_vector(q_inv, ang_vel_world)
        return isaac_velocity_to_frd(ang_vel_body_isaac)

    # ---- Joint state reading ----

    def get_joint_positions(self) -> Tensor:
        """Get all joint positions.

        Returns:
            Tensor of shape (num_envs, num_joints) in radians.
        """
        return self._art.data.joint_pos.clone()

    def get_fin_joint_positions(self) -> Tensor:
        """Get fin joint positions in +X, +Y, -X, -Y order.

        Returns:
            Tensor of shape (num_envs, 4) in radians.
        """
        joint_indices = self._map.fin_joint_indices
        return self._art.data.joint_pos[:, joint_indices].clone()

    def get_fin_joint_velocities(self) -> Tensor:
        """Get fin joint angular velocities.

        Returns:
            Tensor of shape (num_envs, 4) in rad/s.
        """
        joint_indices = self._map.fin_joint_indices
        return self._art.data.joint_vel[:, joint_indices].clone()

    # ---- Joint state writing ----

    def set_fin_joint_targets(self, positions: Tensor) -> None:
        """Set fin joint position targets (for position-controlled joints).

        Args:
            positions: Tensor of shape (num_envs, 4) in radians.
        """
        joint_indices = torch.tensor(self._map.fin_joint_indices, device=positions.device)
        self._art.set_joint_position_target(positions, joint_ids=joint_indices)

    def set_root_state(
        self,
        position: Tensor,
        quaternion_wxyz: Tensor,
        linear_vel: Tensor,
        angular_vel: Tensor,
    ) -> None:
        """Set root body state for episode reset.

        Args:
            position: Tensor (num_envs, 3) in Isaac world frame (m).
            quaternion_wxyz: Tensor (num_envs, 4) in (w,x,y,z).
            linear_vel: Tensor (num_envs, 3) in Isaac world frame (m/s).
            angular_vel: Tensor (num_envs, 3) in Isaac world frame (rad/s).
        """
        root_state = torch.cat([position, quaternion_wxyz, linear_vel, angular_vel], dim=-1)
        self._art.write_root_state_to_sim(root_state)

    def get_altitude(self, ground_level: float = 0.0) -> Tensor:
        """Compute altitude above ground from Isaac world frame y-coordinate.

        In Isaac y-up frame, altitude = root_pos_y - ground_level.

        Returns:
            Tensor of shape (num_envs,) in meters.
        """
        pos_y = self._art.data.root_pos_w[:, 1]  # y-up in Isaac
        return pos_y - ground_level
