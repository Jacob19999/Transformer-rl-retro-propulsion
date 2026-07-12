"""
MG996R servo dynamics model.

Implements first-order lag, angular rate limiting, clamping, and deadband
for servo actuators. Vectorized state update for (num_envs, 4) servo arrays.

Model equations per action_space contract:
  e = x_cmd - x               [tracking error]
  if deadband: e = 0 when |e| < deadband
  ẋ = e / τ_servo             [first-order lag]
  ẋ_clamped = clip(ẋ, -ω_max, ω_max)
  x_new = x + ẋ_clamped * dt
  x_final = clip(x_new, -max_cmd, max_cmd)
"""

from __future__ import annotations
import torch
from torch import Tensor
import yaml
from pathlib import Path


class ServoModel:
    """First-order servo dynamics model for MG996R or similar servos."""

    def __init__(
        self,
        tau_servo: float = 0.05,               # s, source: estimate
        max_angular_velocity: float = 7.54,    # rad/s, source: derived
        max_command_angle: float = 0.262,      # rad, source: measured
        deadband: float = 0.017,               # rad, source: estimate
        apply_deadband: bool = True,
    ):
        self.tau_servo = tau_servo
        self.max_angular_velocity = max_angular_velocity
        self.max_command_angle = max_command_angle
        self.deadband = deadband
        self.apply_deadband = apply_deadband

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "ServoModel":
        """Load servo model from YAML config file."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        servo = config.get("servo", config)
        return cls(
            tau_servo=servo.get("tau_servo", 0.05),
            max_angular_velocity=servo.get("max_angular_velocity", 7.54),
            max_command_angle=servo.get("max_command_angle", 0.262),
            deadband=servo.get("deadband", 0.017),
        )

    def update(
        self,
        state: Tensor,       # (num_envs, 4) current servo angles (rad)
        command: Tensor,     # (num_envs, 4) commanded angles (rad)
        dt: float,
    ) -> Tensor:
        """Update servo state by one timestep.

        Args:
            state: Current servo angle for each fin (rad), shape (num_envs, 4).
            command: Commanded servo angle for each fin (rad), shape (num_envs, 4).
            dt: Simulation timestep (s).

        Returns:
            New servo state (num_envs, 4) in radians.
        """
        # Clamp command to physical limits
        cmd_clamped = command.clamp(-self.max_command_angle, self.max_command_angle)

        # Tracking error (command deadband is applied on error, not on state).
        # Applying deadband directly on state each substep can pin the actuator
        # at zero under high-rate integration.
        error = cmd_clamped - state
        if self.apply_deadband:
            error = torch.where(
                error.abs() < self.deadband,
                torch.zeros_like(error),
                error,
            )

        # First-order lag: rate of change
        rate = error / self.tau_servo

        # Rate limit
        rate_limited = rate.clamp(-self.max_angular_velocity, self.max_angular_velocity)

        # Integrate
        new_state = state + rate_limited * dt

        # Clamp to position limits
        new_state = new_state.clamp(-self.max_command_angle, self.max_command_angle)

        return new_state

    def reset(self, num_envs: int, device: torch.device = None) -> Tensor:
        """Return zeroed initial servo state.

        Returns:
            Tensor of shape (num_envs, 4) initialized to zero.
        """
        return torch.zeros(num_envs, 4, device=device)
