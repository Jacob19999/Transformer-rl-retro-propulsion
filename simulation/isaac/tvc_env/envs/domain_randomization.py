"""
Per-reset domain randomization for 128-env vectorized training.

Randomizes spawn position/velocity/attitude from task config ranges.
Optional mass/inertia perturbation and servo parameter variation.
All randomization is seeded for reproducibility per constitution Principle IV.
"""

from __future__ import annotations
import torch
from torch import Tensor
from typing import Any


class DomainRandomizer:
    """Applies domain randomization on episode resets."""

    def __init__(
        self,
        task_config: dict[str, Any],
        randomize_mass: bool = False,       # Optional: perturb vehicle mass
        mass_variation: float = 0.05,       # ±5% mass variation
        randomize_servo: bool = False,      # Optional: perturb servo tau
        servo_tau_variation: float = 0.1,   # ±10% tau variation
        seed: int | None = None,
    ):
        self.task_config = task_config
        self.randomize_mass = randomize_mass
        self.mass_variation = mass_variation
        self.randomize_servo = randomize_servo
        self.servo_tau_variation = servo_tau_variation

        if seed is not None:
            torch.manual_seed(seed)

    def sample_spawn(
        self,
        num_envs: int,
        env_ids: Tensor,
        device: torch.device = None,
    ) -> dict[str, Tensor]:
        """Sample all randomized spawn parameters for specified environments.

        Args:
            num_envs: Total number of environments.
            env_ids: Subset of envs to randomize (indices).
            device: Target device.

        Returns:
            Dict with 'positions', 'quaternions', 'linear_vels', 'angular_vels'.
        """
        from tvc_env.sim.reset_logic import sample_spawn_state

        positions, quaternions, linear_vels, angular_vels = sample_spawn_state(
            self.task_config, env_ids, device
        )

        return {
            "positions": positions,
            "quaternions": quaternions,
            "linear_vels": linear_vels,
            "angular_vels": angular_vels,
        }

    def sample_mass_perturbation(
        self,
        base_mass: float,
        num_envs: int,
        device: torch.device = None,
    ) -> Tensor:
        """Sample per-environment mass perturbations.

        Returns:
            Tensor (num_envs,) — perturbed mass values (kg).
        """
        if not self.randomize_mass:
            return torch.full((num_envs,), base_mass, device=device)
        variation = torch.randn(num_envs, device=device) * self.mass_variation
        return (base_mass * (1.0 + variation)).clamp(base_mass * 0.7, base_mass * 1.3)

    def sample_servo_tau_perturbation(
        self,
        base_tau: float,
        num_envs: int,
        device: torch.device = None,
    ) -> Tensor:
        """Sample per-environment servo time constant perturbations.

        Returns:
            Tensor (num_envs,) — perturbed tau values (s).
        """
        if not self.randomize_servo:
            return torch.full((num_envs,), base_tau, device=device)
        variation = torch.randn(num_envs, device=device) * self.servo_tau_variation
        return (base_tau * (1.0 + variation)).clamp(base_tau * 0.5, base_tau * 2.0)
