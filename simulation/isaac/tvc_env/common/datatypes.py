"""
Typed data structures for the TVC environment.

All tensor fields follow (num_envs, ...) batch convention.
"""

from dataclasses import dataclass
from torch import Tensor


@dataclass
class FinForceResult:
    """Per-fin aerodynamic force computation result.

    All tensors have shape (num_envs, num_fins) for scalar fields,
    or (num_envs, num_fins, 3) for vector fields.
    """
    force_vector: Tensor        # N, force vector in fin-local frame (num_envs, 4, 3)
    normal_force: Tensor        # N, force component normal to fin plane (num_envs, 4)
    tangential_force: Tensor    # N, force component in fin plane (drag) (num_envs, 4)
    thrust_loss: Tensor         # N, total EDF thrust reduction due to fin blockage (num_envs,)


@dataclass
class FinDispatchResult:
    """Per-fin force dispatch result in body-FRD coordinates."""
    forces_body: Tensor         # N, force per fin in body-FRD frame (num_envs, 4, 3)
    cop_positions: Tensor       # m, COP offsets in body-FRD frame (4, 3)
    thrust_loss: Tensor         # N, unclamped EDF thrust loss from fin drag (num_envs,)
    normal_force: Tensor        # N, per-fin normal force magnitudes (num_envs, 4)
    tangential_force: Tensor    # N, per-fin drag magnitudes (num_envs, 4)

    def __iter__(self):
        """Preserve legacy two-value unpacking: forces_body, cop_positions."""
        yield self.forces_body
        yield self.cop_positions


@dataclass
class EDFOutput:
    """EDF propulsion model output for a single timestep.

    All tensors have shape (num_envs,) for scalars,
    or (num_envs, 3) for vectors.
    """
    thrust_force: Tensor                # N, thrust along EDF axis (num_envs,)
    static_reaction_torque: Tensor      # N·m, reaction torque opposing spin (num_envs, 3)
    dynamic_spool_torque: Tensor        # N·m, torque from rotor angular acceleration (num_envs, 3)
    gyro_precession_torque: Tensor      # N·m, gyroscopic precession torque (num_envs, 3)
    current_omega: Tensor               # rad/s, current rotor angular velocity (num_envs,)


@dataclass
class VehicleState:
    """Full vehicle state for one timestep.

    Position and velocity in Isaac world frame.
    Quaternion in (w,x,y,z) per Isaac Lab 2.3.2.
    Body-frame velocities in FRD convention.
    """
    position: Tensor            # m, world frame (num_envs, 3)
    quaternion_wxyz: Tensor     # unit quat (w,x,y,z) (num_envs, 4)
    linear_vel_world: Tensor    # m/s, world frame (num_envs, 3)
    angular_vel_world: Tensor   # rad/s, world frame (num_envs, 3)
    linear_vel_frd: Tensor      # m/s, body-FRD frame (num_envs, 3)
    angular_vel_frd: Tensor     # rad/s, body-FRD frame (num_envs, 3)
    fin_angles: Tensor          # rad, actual fin joint angles from PhysX (num_envs, 4)
    fin_rates: Tensor           # rad/s, fin angular rates (num_envs, 4)
    motor_omega: Tensor         # rad/s, current rotor speed (num_envs,)
    contact_state: Tensor       # ContactState enum value, int (num_envs,)
    height: Tensor              # m, altitude above ground (num_envs,)
    touchdown_speed: Tensor | None = None  # m/s, first-contact downward speed (num_envs,)
