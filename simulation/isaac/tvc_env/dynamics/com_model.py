"""
Center-of-mass offset model for domain randomization.

Samples COM offset from configured range on reset, applies as force application
point offset. Vectorized for num_envs environments.
"""

from __future__ import annotations
import torch
from torch import Tensor
from typing import Any


class COMOffsetModel:
    """Per-episode COM offset model for domain randomization."""

    def __init__(
        self,
        offset_range: list[list[float]] | None = None,
        device: torch.device = None,
    ):
        """
        Args:
            offset_range: [[min_x, min_y, min_z], [max_x, max_y, max_z]] in body-FRD (m).
            device: Target device.
        """
        if offset_range is None:
            offset_range = [[-0.005, -0.005, -0.005], [0.005, 0.005, 0.005]]

        self._offset_min = torch.tensor(offset_range[0], dtype=torch.float32, device=device)
        self._offset_max = torch.tensor(offset_range[1], dtype=torch.float32, device=device)
        self._current_offsets = None

    @classmethod
    def from_disturbance_config(cls, config: dict, device=None) -> "COMOffsetModel":
        """Create from disturbance config dict."""
        dist = config.get("disturbances", config)
        com = dist.get("com_offset", {})
        if not com.get("enabled", False):
            return cls(offset_range=[[0, 0, 0], [0, 0, 0]], device=device)
        return cls(offset_range=com.get("range", [[-0.005]*3, [0.005]*3]), device=device)

    def sample_offsets(self, num_envs: int, env_ids: Tensor | None = None) -> Tensor:
        """Sample new COM offsets for specified environments.

        Args:
            num_envs: Total number of environments.
            env_ids: Specific env indices to sample for. If None, sample all.

        Returns:
            Tensor (num_envs, 3) — COM offsets in body-FRD frame (m).
        """
        device = self._offset_min.device
        if self._current_offsets is None:
            self._current_offsets = torch.zeros(num_envs, 3, device=device)

        indices = env_ids if env_ids is not None else torch.arange(num_envs, device=device)
        n = len(indices)
        offsets = self._offset_min + torch.rand(n, 3, device=device) * (self._offset_max - self._offset_min)
        self._current_offsets[indices] = offsets

        return self._current_offsets

    @property
    def current_offsets(self) -> Tensor | None:
        """Current COM offsets for all environments."""
        return self._current_offsets
