"""Landing guidance wrapper.

Generates a moving altitude reference (a "premade trajectory") toward a
landing pad and disarms the controller on touchdown. The wrapper is
controller-agnostic: it rewrites the altitude error component of the
observation so any inner controller (PID, or PPO's residual-PID baseline)
sees a descending waypoint instead of a static pad target.

Trajectory profile per environment:
  Phase 1 (DESCEND): z_ref decreases at `descent_rate` until it is within
    `flare_alt` of the pad altitude.
  Phase 2 (FLARE):   z_ref decreases at `flare_descent_rate` until the pad.
  Phase 3 (LANDED):  once contact_state == LANDED, fins are zeroed and
                     throttle is forced to zero so the EDF spools down.

XY reference stays at pad center (no XY rerouting). The env's reward terms
continue to see the true pad target via `_target_position_world` since the
guidance only modifies the controller-facing observation, not the env state.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from tvc_env.common.constants import ContactState


class LandingGuidance:
    """Moving-altitude waypoint generator with touchdown disarm."""

    def __init__(
        self,
        num_envs: int,
        device,
        target_position: Tensor,
        descent_rate: float = 1.0,
        flare_alt: float = 0.5,
        flare_descent_rate: float = 0.25,
        xy_gate_radius: float = 0.75,
        far_descent_rate: float = 0.15,
        throttle_hover: float = 0.90,
        descent_brake_gain: float = 0.35,
        min_descent_throttle: float = 0.30,
        dt: float = 1.0 / 30.0,
    ):
        self.num_envs = int(num_envs)
        self.device = device
        target = target_position.to(device).clone().to(torch.float32)
        if target.dim() == 1:
            target = target.unsqueeze(0).expand(self.num_envs, -1).contiguous()
        self.target_position = target
        self.descent_rate = float(descent_rate)
        self.flare_alt = float(flare_alt)
        self.flare_descent_rate = float(flare_descent_rate)
        self.xy_gate_radius = float(xy_gate_radius)
        self.far_descent_rate = float(far_descent_rate)
        self.throttle_hover = float(throttle_hover)
        self.descent_brake_gain = float(descent_brake_gain)
        self.min_descent_throttle = float(min_descent_throttle)
        self._dt = float(dt)

        self._z_ref: Tensor | None = None
        self._target_down_rate = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
        self._landed_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=device)

    def _current_from_obs(self, obs: Tensor) -> Tensor:
        # obs[:, 0:3] is (target - current); recover current world position.
        return self.target_position - obs[:, 0:3]

    def reset(self, obs: Tensor | None = None, env_ids: Tensor | None = None) -> None:
        """Reset the waypoint state.

        If `obs` is provided, the initial waypoint altitude is seeded to the
        current vehicle altitude so the descent profile starts from the actual
        spawn height rather than a fixed pre-set value.
        """
        if self._z_ref is None:
            self._z_ref = self.target_position[:, 2].clone()
        if env_ids is None:
            if obs is not None:
                cur = self._current_from_obs(obs)
                self._z_ref = cur[:, 2].clone()
            else:
                self._z_ref = self.target_position[:, 2].clone()
            self._target_down_rate.zero_()
            self._landed_mask.zero_()
        else:
            if obs is not None:
                cur = self._current_from_obs(obs)
                self._z_ref[env_ids] = cur[env_ids, 2]
            else:
                self._z_ref[env_ids] = self.target_position[env_ids, 2]
            self._target_down_rate[env_ids] = 0.0
            self._landed_mask[env_ids] = False

    def _advance_waypoint(self, obs: Tensor) -> Tensor:
        assert self._z_ref is not None
        target_z = self.target_position[:, 2]
        height_above_pad = self._z_ref - target_z
        in_flare = height_above_pad < self.flare_alt
        horiz_error = obs[:, 0:2].norm(dim=-1)
        far_from_pad = horiz_error > self.xy_gate_radius
        rate = torch.where(
            in_flare,
            torch.full_like(height_above_pad, self.flare_descent_rate),
            torch.full_like(height_above_pad, self.descent_rate),
        )
        far_rate = torch.full_like(rate, self.far_descent_rate)
        rate = torch.where(far_from_pad, torch.minimum(rate, far_rate), rate)
        # Once landed, freeze the reference at the pad.
        rate = torch.where(self._landed_mask, torch.zeros_like(rate), rate)
        self._target_down_rate = rate
        next_z = self._z_ref - rate * self._dt
        next_z = torch.maximum(next_z, target_z)
        self._z_ref = next_z
        return self._z_ref

    def modify_obs(self, obs: Tensor) -> Tensor:
        """Update internal state and return obs with altitude error remapped."""
        if self._z_ref is None:
            self.reset(obs=obs)
        contact = obs[:, 23]
        self._landed_mask = self._landed_mask | (contact.long() == int(ContactState.LANDED))
        z_ref = self._advance_waypoint(obs)
        modified = obs.clone()
        cur_z = self.target_position[:, 2] - obs[:, 2]
        modified[:, 2] = z_ref - cur_z
        # The inner PID's altitude derivative term is tuned for hover, where
        # zero vertical speed is the target. During landing the target is a
        # controlled downward speed, so expose velocity error instead. Body-FRD
        # z velocity is positive downward.
        modified[:, 9] = obs[:, 9] - self._target_down_rate.to(dtype=obs.dtype)
        return modified

    def post_action(self, action: Tensor, obs: Tensor | None = None) -> Tensor:
        """Apply landing-specific throttle limits and disarm after touchdown."""
        out = action
        if obs is not None:
            out = out.clone()
            vertical_down_speed = obs[:, 9]
            target_down_rate = self._target_down_rate.to(dtype=obs.dtype)
            too_slow_or_climbing = vertical_down_speed < target_down_rate
            throttle_cap = (
                self.throttle_hover
                - self.descent_brake_gain * (target_down_rate - vertical_down_speed).clamp(min=0.0)
            ).clamp(self.min_descent_throttle, 1.0)
            out[:, 4] = torch.where(too_slow_or_climbing, torch.minimum(out[:, 4], throttle_cap), out[:, 4])

        if not bool(self._landed_mask.any()):
            return out
        if out is action:
            out = action.clone()
        idx = self._landed_mask
        out[idx, :4] = 0.0
        out[idx, 4] = 0.0
        return out

    def wrap(self, obs: Tensor, controller_fn: Callable[[Tensor], Tensor]) -> Tensor:
        """Convenience: modify obs, call controller, then disarm on landed."""
        modified = self.modify_obs(obs)
        action = controller_fn(modified)
        return self.post_action(action, obs)

    @property
    def z_ref(self) -> Tensor | None:
        return self._z_ref

    @property
    def landed_mask(self) -> Tensor:
        return self._landed_mask

    def get_debug_state(self, env_idx: int = 0) -> dict[str, float | bool]:
        if self._z_ref is None:
            return {}
        if env_idx < 0 or env_idx >= self.num_envs:
            env_idx = 0
        target_z = float(self.target_position[env_idx, 2].item())
        z_ref = float(self._z_ref[env_idx].item())
        return {
            "z_ref_world_m": z_ref,
            "z_ref_above_pad_m": z_ref - target_z,
            "descent_rate_m_s": self.descent_rate,
            "flare_alt_m": self.flare_alt,
            "flare_descent_rate_m_s": self.flare_descent_rate,
            "xy_gate_radius_m": self.xy_gate_radius,
            "far_descent_rate_m_s": self.far_descent_rate,
            "target_down_rate_m_s": float(self._target_down_rate[env_idx].item()),
            "descent_brake_gain": self.descent_brake_gain,
            "min_descent_throttle": self.min_descent_throttle,
            "is_landed": bool(self._landed_mask[env_idx].item()),
        }
