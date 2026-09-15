"""Allocate body-axis efforts to vanes hinged along their radial spans.

Positive joint rotation turns the downstream jet toward hinge x flow; the
reaction on the body is opposite. Radial hinges produce tangential jet forces
and rank-three torque authority, including yaw through common-mode deflection.
Commands are fin-angle efforts, not measured body rates.
"""

from __future__ import annotations
import torch
from torch import Tensor


class PIDFinMixer:
    """Converts roll/pitch/yaw commands to 4-fin deflection angles."""

    def __init__(
        self,
        max_fin_angle: float = 0.262,   # rad (15°) per action_space contract
        yaw_coupling: float = 1.0,      # dimensionless fin-angle effort allocation
        device: torch.device = None,
    ):
        self._max_fin_angle = max_fin_angle
        self.device = device

        # Mixing matrix: shape (4, 3) — [roll, pitch, yaw]
        # Row ordering: [fin_+X, fin_+Y, fin_-X, fin_-Y]
        self._mix = torch.tensor(
            [
                [-1.0,  0.0, yaw_coupling],
                [ 0.0, -1.0, yaw_coupling],
                [ 1.0,  0.0, yaw_coupling],
                [ 0.0,  1.0, yaw_coupling],
            ],
            dtype=torch.float32,
            device=device,
        )  # (4, 3)

    def mix(
        self,
        roll_cmd: Tensor,   # (num_envs,)
        pitch_cmd: Tensor,  # (num_envs,)
        yaw_cmd: Tensor,    # (num_envs,)
    ) -> Tensor:
        """Compute fin angles from roll/pitch/yaw fin-angle efforts.

        Args:
            roll_cmd:  Roll effort  (num_envs,) rad
            pitch_cmd: Pitch effort (num_envs,) rad
            yaw_cmd:   Yaw fin-angle effort (num_envs,) rad

        Returns:
            Tensor (num_envs, 4) — fin angles clamped to ±max_fin_angle (rad).
        """
        mix = self._mix.to(roll_cmd.device)

        # cmd_vec: (num_envs, 3)
        cmd_vec = torch.stack([roll_cmd, pitch_cmd, yaw_cmd], dim=-1)

        # fin_angles: (num_envs, 4) = cmd_vec @ mix.T
        fin_angles = cmd_vec @ mix.t()

        return fin_angles.clamp(-self._max_fin_angle, self._max_fin_angle)
