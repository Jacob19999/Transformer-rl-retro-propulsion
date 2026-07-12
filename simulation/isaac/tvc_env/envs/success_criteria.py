"""
Success condition checks for hover and landing tasks.

Hover success: position error + tilt + angular rate within tolerance for dwell_time seconds.
Landing success: contact state == LANDED and pad distance within tolerance.
"""

from __future__ import annotations
import torch
from torch import Tensor
from tvc_env.common.constants import ContactState
from tvc_env.common.quaternions import tilt_angle


class HoverSuccessTracker:
    """Tracks hover success with a dwell-time requirement."""

    def __init__(
        self,
        num_envs: int,
        max_position_error: float = 0.5,   # m
        max_tilt: float = 0.26,             # rad (15°)
        max_angular_rate: float = 1.0,      # rad/s
        dwell_time: float = 3.0,            # s
        rl_dt: float = 0.0333,             # s per RL step
        device: torch.device = None,
    ):
        self.max_position_error = max_position_error
        self.max_tilt = max_tilt
        self.max_angular_rate = max_angular_rate
        self.dwell_steps = int(dwell_time / rl_dt)
        self.device = device

        self._success_count = torch.zeros(num_envs, dtype=torch.int32, device=device)

    def update(
        self,
        position: Tensor,
        quaternion_wxyz: Tensor,
        angular_vel: Tensor,
        target_position: Tensor,
    ) -> Tensor:
        """Update success tracker for one RL step.

        Returns:
            Bool tensor (num_envs,) — True where hover success criteria met for dwell_time.
        """
        target = target_position.to(position.device)
        pos_err = (position - target).norm(dim=-1)  # (num_envs,)
        tilt = tilt_angle(quaternion_wxyz)
        ang_rate = angular_vel.norm(dim=-1)

        within = (pos_err < self.max_position_error) & (tilt < self.max_tilt) & (ang_rate < self.max_angular_rate)

        self._success_count[within] += 1
        self._success_count[~within] = 0

        return self._success_count >= self.dwell_steps

    def reset(self, env_ids: Tensor | None = None) -> None:
        """Reset dwell counter for specified environments."""
        if env_ids is None:
            self._success_count.fill_(0)
        else:
            self._success_count[env_ids] = 0


def check_landing_success(
    contact_state: Tensor,         # (num_envs,) ContactState int
    position: Tensor,              # (num_envs, 3)
    target_position: Tensor,       # (3,) pad center
    max_pad_distance: float = 0.5, # m
) -> Tensor:
    """Check landing success: LANDED state + within pad distance.

    Returns:
        Bool tensor (num_envs,) — True where landing success criteria met.
    """
    is_landed = contact_state == ContactState.LANDED
    target = target_position.to(position.device)
    horiz_dist = (position[:, :2] - target[:2]).norm(dim=-1)
    on_pad = horiz_dist < max_pad_distance
    return is_landed & on_pad
