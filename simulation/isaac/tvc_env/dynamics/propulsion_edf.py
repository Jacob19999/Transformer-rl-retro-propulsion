"""
EDF thrust and spool dynamics model.

Implements:
  - Throttle-to-RPM mapping: ω_target = throttle * ω_max
  - First-order motor spool lag: dω/dt = (ω_target - ω) / τ_motor
  - Optional RPM rate limiting: clamp(dω, -dω_max, dω_max)
  - Thrust: T = k_T * ω²

Vectorized for (num_envs,) arrays.
"""

from __future__ import annotations
import torch
from torch import Tensor
import yaml
from pathlib import Path
from tvc_env.common.datatypes import EDFOutput


class EDFModel:
    """EDF propulsion model with spool dynamics and reaction torques."""

    def __init__(
        self,
        max_thrust: float = 48.0,          # N, source: estimate
        tau_motor: float = 0.15,           # s, source: estimate
        omega_max: float = 3000.0,         # rad/s, source: to-be-calibrated (placeholder)
        d_omega_max: float | None = None,  # rad/s², source: to-be-calibrated (optional clamp)
        k_T: float | None = None,          # N·s²/rad², source: to-be-calibrated
        k_Q: float | None = None,          # N·m·s²/rad², source: to-be-calibrated
        rotor_inertia: float = 0.0005,     # kg·m², source: estimate
        thrust_axis: list[float] | None = None,  # in body-FRD frame
    ):
        self.tau_motor = tau_motor
        self.omega_max = omega_max
        self.d_omega_max = d_omega_max
        self.rotor_inertia = rotor_inertia
        self.max_thrust = max_thrust
        # Thrust axis in body-FRD: EDF exhaust is +z (downward thrust)
        self.thrust_axis = torch.tensor(thrust_axis or [0.0, 0.0, 1.0], dtype=torch.float32)

        # Derive k_T if not provided: T = k_T * omega² → k_T = max_thrust / omega_max²
        if k_T is not None:
            self.k_T = k_T
        else:
            self.k_T = max_thrust / (omega_max ** 2) if omega_max > 0 else 0.0

        # Derive k_Q if not provided: Q = k_Q * omega²
        if k_Q is not None:
            self.k_Q = k_Q
        else:
            self.k_Q = self.k_T * 0.02  # rough estimate: Q/T ≈ 2% of radius

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "EDFModel":
        """Load EDF model from YAML config file."""
        path = Path(yaml_path)
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        edf = config.get("edf")
        if edf is None and isinstance(config.get("vehicle"), dict):
            edf = config["vehicle"].get("edf")

        # Repo convention: vehicle mass/inertia live under configs/vehicle/,
        # while EDF parameters live under configs/params/edf_90mm.yaml.
        if edf is None and path.parent.name == "vehicle":
            params_path = path.parent.parent / "params" / "edf_90mm.yaml"
            if params_path.exists():
                with open(params_path, "r", encoding="utf-8") as f:
                    params_config = yaml.safe_load(f) or {}
                edf = params_config.get("edf", params_config)

        if edf is None:
            edf = config

        tau_motor = edf.get("tau_motor", 0.15)
        omega_max = edf.get("omega_max") or 3000.0
        d_omega_max = edf.get("d_omega_max")
        k_T = edf.get("k_T")  # None is OK — computed from max_thrust/omega_max²
        k_Q = edf.get("k_Q")

        return cls(
            max_thrust=edf.get("max_thrust", 48.0),
            tau_motor=tau_motor,
            omega_max=omega_max,
            d_omega_max=d_omega_max,
            k_T=k_T,
            k_Q=k_Q,
            rotor_inertia=edf.get("rotor_inertia", 0.0005),
        )

    def update(
        self,
        omega_state: Tensor,    # (num_envs,) current rotor angular velocity (rad/s)
        throttle: Tensor,       # (num_envs,) normalized throttle [0, 1]
        dt: float,
    ) -> Tensor:
        """Update motor spool state by one timestep.

        Args:
            omega_state: Current rotor angular velocity (rad/s), shape (num_envs,).
            throttle: Normalized throttle command [0, 1], shape (num_envs,).
            dt: Simulation timestep (s).

        Returns:
            New rotor angular velocity (num_envs,) in rad/s.
        """
        omega_target = throttle * self.omega_max

        # First-order spool dynamics
        d_omega = (omega_target - omega_state) / self.tau_motor

        # Uncalibrated slew limits should not distort the configured first-order response.
        if self.d_omega_max is not None:
            d_omega = d_omega.clamp(-self.d_omega_max, self.d_omega_max)

        new_omega = omega_state + d_omega * dt
        new_omega = new_omega.clamp(0.0, self.omega_max)

        return new_omega

    def compute_thrust(self, omega: Tensor) -> Tensor:
        """Compute thrust from current rotor speed.

        Args:
            omega: Rotor angular velocity (rad/s), shape (num_envs,).

        Returns:
            Thrust (N), shape (num_envs,).
        """
        return self.k_T * omega ** 2

    def compute_output(
        self,
        omega: Tensor,              # (num_envs,) current rotor speed
        omega_prev: Tensor,         # (num_envs,) previous rotor speed (for d_omega/dt)
        body_angular_vel: Tensor,   # (num_envs, 3) body angular velocity in body-FRD frame
        dt: float,
        spin_axis: Tensor | None = None,  # (3,) rotor spin axis in body-FRD, default [0,0,1]
    ) -> EDFOutput:
        """Compute full EDF output including all torque components.

        Args:
            omega: Current rotor angular velocity (rad/s), shape (num_envs,).
            omega_prev: Previous rotor angular velocity (rad/s), shape (num_envs,).
            body_angular_vel: Body angular velocity in body-FRD (rad/s), shape (num_envs, 3).
            dt: Simulation timestep (s).
            spin_axis: Rotor spin axis in body-FRD (unit vector), default [0, 0, 1].

        Returns:
            EDFOutput with thrust and all torque components.
        """
        num_envs = omega.shape[0]
        device = omega.device

        if spin_axis is None:
            spin_axis = self.thrust_axis.to(device)

        # Thrust force along spin axis in body-FRD
        thrust_magnitude = self.compute_thrust(omega)  # (num_envs,)

        # Static reaction torque: opposes spin direction
        # Q = k_Q * omega², direction opposite to spin axis
        Q_magnitude = self.k_Q * omega ** 2
        static_reaction = -spin_axis.unsqueeze(0) * Q_magnitude.unsqueeze(-1)  # (num_envs, 3)

        # Dynamic spool reaction torque on the body: -I_rotor * dω/dt along spin axis.
        d_omega = (omega - omega_prev) / max(dt, 1e-8)
        dynamic_spool = -spin_axis.unsqueeze(0) * (self.rotor_inertia * d_omega).unsqueeze(-1)  # (num_envs, 3)

        # Gyroscopic precession on body: -ω_body × H_rotor (Newton's third law on
        # the rotor's precession torque). PhysX has no virtual rotor, so we
        # apply this reaction as an external body torque.
        H_rotor = spin_axis.unsqueeze(0) * (self.rotor_inertia * omega).unsqueeze(-1)  # (num_envs, 3)
        gyro_precession = -torch.linalg.cross(body_angular_vel, H_rotor)  # (num_envs, 3)

        return EDFOutput(
            thrust_force=thrust_magnitude,
            static_reaction_torque=static_reaction,
            dynamic_spool_torque=dynamic_spool,
            gyro_precession_torque=gyro_precession,
            current_omega=omega,
        )

    def reset(self, num_envs: int, device: torch.device = None) -> Tensor:
        """Return zeroed initial omega state.

        Returns:
            Tensor of shape (num_envs,) initialized to zero.
        """
        return torch.zeros(num_envs, device=device)
