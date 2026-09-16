"""Flight rotation measured from true FRD body rates at the physics clock.

Angular travel is integral |p|, |q|, |r|, not wrapped Euler angle change.
It counts reversals and repeated turns. Limits are soft task preferences;
this module neither clips angular velocity nor changes termination gates.
"""
from __future__ import annotations

import math
import torch

AXES = ("roll", "pitch", "yaw")
DEFAULT_LIMITS_DEG_S = (90.0, 90.0, 180.0)


def _positive_linear_integral(a, b):
    """Mean positive part and positive time fraction for a linear interval."""
    lo, hi = torch.minimum(a, b), torch.maximum(a, b)
    fraction = torch.where(lo >= 0, (hi > 0).to(a.dtype),
                           (hi / (hi - lo).clamp(min=1e-12)).clamp(0, 1))
    return .5 * (lo.clamp(min=0) + hi.clamp(min=0)) * fraction, fraction


class RotationTracker:
    def __init__(self, num_envs, device, limits_deg_s=DEFAULT_LIMITS_DEG_S):
        if len(limits_deg_s) != 3 or any(not math.isfinite(v) or v <= 0 for v in limits_deg_s):
            raise ValueError("rotation soft_limits_deg_s must contain three positive finite FRD rates")
        self.limits = torch.tensor(limits_deg_s, device=device, dtype=torch.float32) * (math.pi / 180)
        self.episode = {
            name: torch.zeros(num_envs, 3 if name in (
                'peak_rate_rad_s', 'angular_travel_rad', 'excess_rotation_rad', 'time_above_limit_s'
            ) else 1, device=device)
            for name in ('peak_rate_rad_s', 'angular_travel_rad', 'excess_rotation_rad',
                         'time_above_limit_s', 'peak_rate_norm_rad_s', 'angular_path_rad',
                         'excess_cost_s', 'flight_time_s')
        }
        self.step_cost_s = torch.zeros(num_envs, device=device)

    def reset(self, env_ids=None):
        for value in self.episode.values():
            if env_ids is None:
                value.zero_()
            else:
                value[env_ids] = 0
        if env_ids is None:
            self.step_cost_s.zero_()
        else:
            self.step_cost_s[env_ids] = 0

    def begin_step(self):
        self.step_cost_s.zero_()

    def update(self, before, after, dt, active):
        """Accumulate one real physics interval, including its terminal impact.

        Linear endpoint interpolation handles rate reversals/limit crossings.
        Peak rates are sampled at both endpoints; finer physics is still
        required to resolve any variation within an integration interval.
        """
        mask = active.to(before.dtype)[:, None]
        abs_before, abs_after = before.abs(), after.abs()
        positive, _ = _positive_linear_integral(before, after)
        negative, _ = _positive_linear_integral(-before, -after)
        upper, upper_time = _positive_linear_integral(before-self.limits, after-self.limits)
        lower, lower_time = _positive_linear_integral(-before-self.limits, -after-self.limits)
        ep = self.episode
        ep['peak_rate_rad_s'] = torch.maximum(ep['peak_rate_rad_s'], torch.maximum(abs_before, abs_after)*mask)
        ep['angular_travel_rad'] += (positive+negative) * dt * mask
        ep['excess_rotation_rad'] += (upper+lower) * dt * mask
        ep['time_above_limit_s'] += (upper_time+lower_time) * dt * mask
        norms = torch.stack((before.norm(dim=-1), after.norm(dim=-1)), dim=-1)
        ep['peak_rate_norm_rad_s'] = torch.maximum(ep['peak_rate_norm_rad_s'], norms.max(-1).values[:,None]*mask)
        ep['angular_path_rad'] += norms.mean(-1)[:,None] * dt * mask
        # Bounded soft exceedance, zero within limits. Unlike a per-policy-step
        # rate norm this has seconds as units and a fixed episode cost budget.
        # Diagnostic: warm 16-20 m eval (12.058M, Sept 15) saw 33-36 rad/s peaks;
        # the old -.01*|w| per step could exceed the -200 crash payout over 30s.
        severity_before = ((abs_before-self.limits).clamp(min=0)/abs_before.clamp(min=1e-12)).max(-1).values
        severity_after = ((abs_after-self.limits).clamp(min=0)/abs_after.clamp(min=1e-12)).max(-1).values
        cost = .5*(severity_before+severity_after)*dt*active
        self.step_cost_s += cost
        ep['excess_cost_s'] += cost[:,None]
        ep['flight_time_s'] += dt * mask

    def snapshot(self):
        return {name: value.clone() for name, value in self.episode.items()}

    @property
    def peak_rate_rad_s(self):
        """Whole-flight per-axis peak rate used by the terminal quality reward."""
        return self.episode['peak_rate_rad_s']

    def record(self, env_id=0):
        return rotation_record(self.episode, self.limits, env_id)


def rotation_record(snapshot, limits, env_id):
    """JSON-friendly telemetry with explicit units and axis order."""
    result = dict(axes=list(AXES), soft_limits_deg_s=(limits*180/math.pi).tolist(),
                  source='true_body_frd_physics_substeps')
    for key, value in snapshot.items():
        item = value[env_id]
        if key.endswith('_rad_s'):
            key, item = key[:-6]+'_deg_s', item * (180/math.pi)
        elif key.endswith('_rad'):
            key, item = key[:-4]+'_deg', item * (180/math.pi)
        result[key] = item.tolist() if item.numel() > 1 else float(item.item())
    return result


class RotationSummary:
    """Accumulate completed episodes without weighting quick crashes twice."""
    def __init__(self, tracker):
        self.limits = tracker.limits
        self.count = torch.zeros((), device=tracker.limits.device)
        self.sums = {k: torch.zeros_like(v[:1]) for k, v in tracker.episode.items()}
        self.maxima = {k: torch.zeros_like(v[:1]) for k, v in tracker.episode.items()}

    def add(self, snapshot, completed):
        self.count += completed.sum()
        mask = completed[:,None]
        for key, value in snapshot.items():
            self.sums[key] += (value * mask).sum(0, keepdim=True)
            self.maxima[key] = torch.maximum(self.maxima[key], (value * mask).max(0, keepdim=True).values)

    def record(self):
        count = int(self.count.item())
        return dict(episodes=count,
                    mean=rotation_record({k: v/self.count.clamp(min=1) for k,v in self.sums.items()}, self.limits, 0) if count else None,
                    max=rotation_record(self.maxima, self.limits, 0) if count else None)
