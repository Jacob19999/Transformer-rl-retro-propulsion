"""
PID fin mixing logic for TVC environment.

Converts roll/pitch/yaw rate commands (rad/s) to 4 individual fin deflection
angles (rad) based on fin geometry:

  Fin 0 (+X, front): hinge along Y — primarily controls PITCH
  Fin 1 (+Y, right): hinge along X — primarily controls ROLL
  Fin 2 (-X, rear) : hinge along Y — PITCH with opposite sign from fin 0
  Fin 3 (-Y, left) : hinge along X — ROLL with opposite sign from fin 1

Mixing matrix in FRD frame (rows = fins, cols = [roll, pitch, yaw]):

         roll  pitch  yaw
  fin0:  [ 0,   +1,  +0.5 ]   (+X fin: pitch + yaw coupling)
  fin1:  [+1,    0,  -0.5 ]   (+Y fin: roll - yaw coupling)
  fin2:  [ 0,   -1,  +0.5 ]   (-X fin: -pitch + yaw coupling)
  fin3:  [-1,    0,  -0.5 ]   (-Y fin: -roll - yaw coupling)

Sign conventions (FRD):
  +roll_cmd  → roll right  (right side down)  → +fin1, -fin3
  +pitch_cmd → pitch down  (nose down)        → +fin0, -fin2
  +yaw_cmd   → yaw right   (nose right)       → differential coupling
"""

from __future__ import annotations
import torch
from torch import Tensor


class PIDFinMixer:
    """Converts roll/pitch/yaw commands to 4-fin deflection angles."""

    def __init__(
        self,
        max_fin_angle: float = 0.262,   # rad (15°) per action_space contract
        yaw_coupling: float = 0.5,      # yaw cross-coupling scale
        device: torch.device = None,
    ):
        self._max_fin_angle = max_fin_angle
        self.device = device

        # Mixing matrix: shape (4, 3) — [roll, pitch, yaw]
        # Row ordering: [fin_+X, fin_+Y, fin_-X, fin_-Y]
        self._mix = torch.tensor(
            [
                [ 0.0,  1.0,  yaw_coupling],   # fin_+X: pitch + yaw
                [ 1.0,  0.0, -yaw_coupling],   # fin_+Y: roll - yaw
                [ 0.0, -1.0,  yaw_coupling],   # fin_-X: -pitch + yaw
                [-1.0,  0.0, -yaw_coupling],   # fin_-Y: -roll - yaw
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
        """Compute fin angles from roll/pitch/yaw rate commands.

        Args:
            roll_cmd:  Roll rate command  (num_envs,) rad/s
            pitch_cmd: Pitch rate command (num_envs,) rad/s
            yaw_cmd:   Yaw rate command   (num_envs,) rad/s

        Returns:
            Tensor (num_envs, 4) — fin angles clamped to ±max_fin_angle (rad).
        """
        mix = self._mix.to(roll_cmd.device)

        # cmd_vec: (num_envs, 3)
        cmd_vec = torch.stack([roll_cmd, pitch_cmd, yaw_cmd], dim=-1)

        # fin_angles: (num_envs, 4) = cmd_vec @ mix.T
        fin_angles = cmd_vec @ mix.t()

        return fin_angles.clamp(-self._max_fin_angle, self._max_fin_angle)
