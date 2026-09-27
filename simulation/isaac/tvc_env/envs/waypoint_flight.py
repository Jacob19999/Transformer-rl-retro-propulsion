"""Goal-conditioned waypoint flight: fly each leg to the next waypoint efficiently.

Task definition (versioned in configs/tasks/waypoint_flight.yaml):

* A mission is an ordered list of up to ``MAX_WAYPOINTS`` waypoints.
  FLYPASS is captured when the swept flight segment passes within its radius.
  HOVER needs a continuous hold with position, speed AND body rate inside
  limits. LAND is only allowed as the final waypoint and is captured by a real
  LANDED contact on the pad within the touchdown gates.
* The policy observes the next three waypoints in body FRD and chooses its own
  path and speed. There is no reference trajectory, speed profile, mixer or
  action override; the mission planner composes manoeuvres from waypoints.
* Reward is dense route progress and hover settle/hold terms (undiscounted
  differences, not refunded at termination), a per-step hover-tracking bonus,
  capture and mission bonuses, a time/energy cost weighted toward
  energy, a yaw-weighted body-rate cost and a small action-rate cost, plus one
  failure penalty. ``reward_budget`` checks the CLAUDE.md rule-2 magnitudes.

Yaw diagnosis behind the body-rate terms (2026-09-23). ppo_waypoints_staged_v5
curriculum_eval.jsonl held mean peak yaw at 1625-1651 deg/s from 32.5M to 200M
transitions while success rose 9% -> 36% and crashes stayed at 40%. With this
repo's CoupledJet model the vanes cancel steady residual swirl torque at zero
deflection (net 0.000 N m at 0.5-1.0 throttle) and offer +/-0.30 N m of yaw
authority at hover. The spin is rotor/body angular-momentum exchange:
I_rotor*omega_max/I_zz = 2e-4*4650/0.02 = 46.5 rad/s of body yaw per unit
throttle change, so a 0.84 -> 0.23 descent swing reproduces the 1625 deg/s
peak exactly. The old reward paid ~5 units per episode against +/-1600
terminals for holding yaw, so nothing taught the vanes to shed that momentum.

Nothing here touches Isaac; the environment passes tensors in and out.
"""
from __future__ import annotations

import copy
import math

import torch
from torch import Tensor

from tvc_env.common.constants import ContactState
from tvc_env.common.frames import isaac_position_to_frd, isaac_to_frd
from tvc_env.common.quaternions import from_euler, inverse, multiply, normalize, rotate_vector

FLYPASS, HOVER, LAND, NONE = 0, 1, 2, 3
KIND_NAMES = ('flypass', 'hover', 'land')
MAX_WAYPOINTS = 8
OBSERVATION_CONTRACT = 'waypoint_flight_v1'
# 22 waypoint + 20 vehicle + 4 battery + 5 previous action + 3 fine target channels.
OBS_DIM = 54

OUTCOMES = ('RUNNING', 'SUCCESS', 'CRASH', 'TILT', 'ALTITUDE', 'SPIN', 'GEOFENCE',
            'PREMATURE_LANDING', 'BAD_LANDING', 'TIMEOUT')
(RUNNING, SUCCESS, CRASH, TILT, ALTITUDE, SPIN, GEOFENCE,
 PREMATURE_LANDING, BAD_LANDING, TIMEOUT) = range(len(OUTCOMES))

REWARD_TERMS = ('progress', 'hover_settle', 'hover_hold', 'hover_track', 'waypoint_capture',
                'mission_success', 'landing_quality', 'failure', 'time', 'energy', 'body_rate', 'action_rate')
_BONUS_TERMS = ('progress', 'hover_settle', 'hover_hold', 'hover_track', 'waypoint_capture',
                'mission_success', 'landing_quality')
# Dense difference terms: Psi(s') - Psi(s), undiscounted and NOT refunded at
# termination (drone-racing style progress; CLAUDE.md rule 3 non-PBS terms).
_POTENTIAL_TERMS = ('progress', 'hover_settle', 'hover_hold')

GENERATOR_KEYS = ('count_range', 'final_land_probability', 'intermediate_hover_probability',
                  'leg_horizontal_m', 'leg_vertical_m', 'vertical_leg_probability', 'min_leg_m',
                  'altitude_m', 'arena_half_width_m', 'flypass_radius_m', 'hover_radius_m',
                  'land_radius_m', 'hover_hold_s', 'land_leg_horizontal_m')


def deep_merge(base: dict, overlay: dict) -> dict:
    from tvc_env.envs.task_registry import deep_merge as merge
    return merge(base, overlay)


def segment_distance(point: Tensor, a: Tensor, b: Tensor) -> Tensor:
    """Distance from ``point`` to the swept segment a->b, batched over rows."""
    delta = b - a
    alpha = ((point - a) * delta).sum(-1) / delta.square().sum(-1).clamp(min=1e-12)
    closest = a + alpha.clamp(0, 1)[:, None] * delta
    return (point - closest).norm(dim=-1)


def _clip_norm(vector: Tensor, limit: float) -> Tensor:
    return vector * (limit / vector.norm(dim=-1, keepdim=True).clamp(min=limit))


def _pair(value, name, low=-math.inf, high=math.inf):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f'{name} must be [low, high]')
    lo, hi = float(value[0]), float(value[1])
    if not (math.isfinite(lo) and math.isfinite(hi) and low <= lo <= hi <= high):
        raise ValueError(f'{name} must satisfy {low} <= low <= high <= {high}')
    return lo, hi


def validate_generator(generator: dict) -> dict:
    """Reject malformed or physically inconsistent random-mission settings."""
    missing = [key for key in GENERATOR_KEYS if key not in generator]
    unknown = sorted(set(generator) - set(GENERATOR_KEYS))
    if missing or unknown:
        raise ValueError(f'waypoint generator missing {missing}, unknown {unknown}')
    lo, hi = _pair(generator['count_range'], 'count_range', 1, MAX_WAYPOINTS)
    if lo != int(lo) or hi != int(hi):
        raise ValueError('count_range must hold integers')
    for key in ('final_land_probability', 'intermediate_hover_probability', 'vertical_leg_probability'):
        if not 0 <= float(generator[key]) <= 1:
            raise ValueError(f'{key} must be a probability')
    _pair(generator['leg_horizontal_m'], 'leg_horizontal_m', 0, 200)
    _pair(generator['leg_vertical_m'], 'leg_vertical_m', -100, 100)
    _pair(generator['altitude_m'], 'altitude_m', 1.0, 200)
    arena = float(generator['arena_half_width_m'])
    if not 0 < arena <= 500:
        raise ValueError('arena_half_width_m must be within (0, 500]')
    min_leg = float(generator['min_leg_m'])
    if not 0 <= min_leg < arena:
        raise ValueError('min_leg_m must be within [0, arena_half_width_m)')
    if max(float(generator['leg_horizontal_m'][1]), min_leg) > arena:
        raise ValueError('Horizontal legs cannot exceed the arena half width')
    for key in ('flypass_radius_m', 'hover_radius_m', 'land_radius_m'):
        _pair(generator[key], key, 0.05, 25)
    _pair(generator['hover_hold_s'], 'hover_hold_s', 0, 60)
    _pair(generator['land_leg_horizontal_m'], 'land_leg_horizontal_m', 0, arena)
    return generator


def parse_mission(waypoints: list[dict], touchdown_height: float, min_clearance: float) -> dict:
    """Explicit mission -> padded tensors in env-local coordinates.

    Each waypoint is ``{position: [x, y, z], type: flypass|hover|land,
    radius_m, hold_s}``. LAND may only be final and lies on the ground
    (its z is replaced by the body-origin touchdown height). A mission must
    end in HOVER or LAND so the vehicle is never left at speed.
    """
    if not isinstance(waypoints, list) or not 1 <= len(waypoints) <= MAX_WAYPOINTS:
        raise ValueError(f'A mission needs 1..{MAX_WAYPOINTS} waypoints')
    positions = torch.zeros(MAX_WAYPOINTS, 3)
    kinds = torch.full((MAX_WAYPOINTS,), NONE, dtype=torch.long)
    radii = torch.ones(MAX_WAYPOINTS)
    holds = torch.zeros(MAX_WAYPOINTS)
    for j, item in enumerate(waypoints):
        kind = str(item.get('type', 'flypass')).lower()
        if kind not in KIND_NAMES:
            raise ValueError(f'Waypoint {j + 1}: type must be one of {KIND_NAMES}')
        code = KIND_NAMES.index(kind)
        final = j == len(waypoints) - 1
        if code == LAND and not final:
            raise ValueError('LAND is only allowed as the final waypoint')
        if final and code == FLYPASS:
            raise ValueError('The final waypoint must be hover or land')
        position = [float(v) for v in item['position']]
        if len(position) != 3 or not all(math.isfinite(v) for v in position):
            raise ValueError(f'Waypoint {j + 1}: position needs three finite numbers')
        if code == LAND:
            position[2] = touchdown_height
        elif position[2] < min_clearance:
            raise ValueError(f'Waypoint {j + 1}: airborne waypoints need z >= {min_clearance} m')
        radius = float(item.get('radius_m', 1.0 if code == FLYPASS else 0.5))
        hold = float(item.get('hold_s', 2.0)) if code == HOVER else 0.0
        if not 0.05 <= radius <= 25 or not 0 <= hold <= 60:
            raise ValueError(f'Waypoint {j + 1}: radius or hold outside limits')
        positions[j] = torch.tensor(position)
        kinds[j], radii[j], holds[j] = code, radius, hold
    count = len(waypoints)
    positions[count:] = positions[count - 1]
    return dict(positions=positions, kinds=kinds, radii=radii, holds=holds,
                count=torch.tensor(count, dtype=torch.long))


def apply_stage(config: dict, final_task: dict, stage: dict | None) -> None:
    """Install a curriculum stage (or the full task when ``stage`` is None).

    Stages override reset sampling (spawn, mission generator), the episode
    length and the hover/touchdown capture gates (relaxed early, final in the
    last stages). Rewards, observations and physics never change, and absent
    fields always come from the immutable final task definition.
    """
    task = config['task']
    stage = stage or {}
    unknown = sorted(set(stage) - {'spawn', 'generator', 'hover_capture', 'landing_gates', 'episode_length_s',
                                   'min_steps', 'max_steps',
                                   'advance_success_fraction', 'min_terminations', 'name'})
    if unknown:
        raise ValueError(f'Unknown curriculum stage fields: {unknown}')
    task['spawn'] = deep_merge(final_task['spawn'], stage.get('spawn', {}))
    generator = deep_merge(final_task['waypoint_flight']['generator'], stage.get('generator', {}))
    task['waypoint_flight']['generator'] = validate_generator(generator)
    task['waypoint_flight']['hover_capture'] = deep_merge(final_task['waypoint_flight']['hover_capture'],
                                                          stage.get('hover_capture', {}))
    task['waypoint_flight']['landing_gates'] = deep_merge(final_task['waypoint_flight']['landing_gates'],
                                                          stage.get('landing_gates', {}))
    task['episode_length_s'] = float(stage.get('episode_length_s', final_task['episode_length_s']))


def reward_weights(config: dict) -> dict:
    weights = config['task']['reward']
    unknown = sorted(set(weights) - set(REWARD_TERMS))
    missing = sorted(set(REWARD_TERMS) - set(weights))
    if unknown or missing:
        raise ValueError(f'waypoint_flight reward needs exactly {REWARD_TERMS}; '
                         f'missing {missing}, unknown {unknown}')
    for key, value in weights.items():
        value = float(value)
        if not math.isfinite(value) or (value < 0 if key in _BONUS_TERMS else value > 0):
            raise ValueError(f'Reward weight {key}={value} has the wrong sign '
                             '(bonuses >= 0, costs <= 0)')
    return {key: float(weights[key]) for key in REWARD_TERMS}


def electrical_power_w(config: dict, rotor_fraction: float) -> float:
    battery = config['battery']
    shaft = float(battery['shaft_power_at_max_w']) * rotor_fraction ** 3
    return shaft / float(battery['motor_efficiency']) + float(battery['auxiliary_power_w'])


def reward_budget(config: dict, hover_fraction: float, max_episode_s: float | None = None) -> dict:
    """CLAUDE.md rule 2: terminal magnitudes must dominate integrated step costs.

    The nominal worst case flies the longest episode at full electrical power,
    with every body axis at its soft rate limit and a jittery action stream
    (sum of squared normalized action changes 0.1 per step). Returns the
    arithmetic so it can be logged, and raises if a terminal is too small.
    """
    weights = reward_weights(config)
    flight = config['task']['waypoint_flight']
    rl_dt = float(config['env']['physics_dt']) * int(config['env']['decimation'])
    if max_episode_s is None:
        stages = flight.get('curriculum', {}).get('stages', [])
        max_episode_s = max([float(config['task']['episode_length_s'])]
                            + [float(s.get('episode_length_s', 0)) for s in stages])
    hover_w = electrical_power_w(config, hover_fraction)
    max_w = electrical_power_w(config, 1.0)
    hover_cost_s = -weights['time'] - weights['energy'] * hover_w / 3600
    energy_share_hover = -weights['energy'] * hover_w / 3600 / hover_cost_s
    worst_cost_s = (-weights['time'] - weights['energy'] * max_w / 3600
                    - weights['body_rate'] * 3.0 - weights['action_rate'] * 0.1 / rl_dt)
    worst_episode = worst_cost_s * max_episode_s
    result = dict(rl_dt_s=rl_dt, max_episode_s=max_episode_s, hover_fraction=hover_fraction,
                  hover_power_w=hover_w, max_power_w=max_w, hover_cost_per_s=hover_cost_s,
                  energy_share_at_hover=energy_share_hover, worst_step_cost_per_s=worst_cost_s,
                  worst_episode_step_cost=worst_episode, failure=weights['failure'],
                  mission_success=weights['mission_success'])
    if -weights['failure'] <= worst_episode or weights['mission_success'] <= worst_episode:
        raise ValueError(f'Rule-2 violation: integrated step cost {worst_episode:.1f} is not dominated '
                         f'by failure {weights["failure"]} / success {weights["mission_success"]}')
    # hover_track is a per-second bonus: loitering beside a hover target for a
    # whole episode must stay worth less than completing the mission.
    result['max_hover_track_bonus'] = weights['hover_track'] * max_episode_s
    if result['max_hover_track_bonus'] >= weights['mission_success']:
        raise ValueError(f'Rule-2 violation: hover_track can accrue {result["max_hover_track_bonus"]:.1f} '
                         f'per episode, not dominated by success {weights["mission_success"]}')
    return result


class WaypointFlightTask:
    """Vectorized mission state, capture logic, observation and reward."""

    def __init__(self, num_envs: int, device, config: dict, env_origins: Tensor, rl_dt: float):
        self.config = config  # live merged config: the curriculum mutates spawn/generator
        self.n, self.device, self.dt = num_envs, device, float(rl_dt)
        self.origins = env_origins.to(device)
        self.ids = torch.arange(num_envs, device=device)
        self.measurement = None  # measured state behind the latest observation()
        flight = config['task']['waypoint_flight']
        self.touchdown_height = float(flight['touchdown_root_height_m'])
        self.min_clearance = float(flight['min_airborne_clearance_m'])
        self.touchdown_gate = float(flight['landing_gates']['max_touchdown_speed_m_s'])
        self.rate_soft = torch.tensor(flight['rate_soft_limits_deg_s'], device=device,
                                      dtype=torch.float32) * (math.pi / 180)
        self.rate_limit = torch.tensor(flight['max_body_rate_deg_s'], device=device,
                                       dtype=torch.float32) * (math.pi / 180)
        if self.rate_soft.numel() != 3 or self.rate_limit.numel() != 3 or bool((self.rate_soft <= 0).any()):
            raise ValueError('Rate limits need three positive FRD axes')
        self.geofence_margin = float(flight['geofence']['margin_m'])
        self.ceiling = float(flight['geofence']['ceiling_m'])
        obs = flight['observation']
        self.distance_scale = float(obs['distance_scale_m'])
        self.velocity_scale = float(obs['velocity_scale_m_s'])
        self.height_scale = float(obs['height_scale_m'])
        self.radius_scale = float(obs['radius_scale_m'])
        self.hold_scale = float(obs['hold_scale_s'])
        shaping = flight['hover_shaping']
        self.settle_distance = float(shaping['settle_distance_scale_m'])
        self.settle_speed_ref = float(shaping['speed_ref_m_s'])
        self.settle_rate_ref = float(shaping['rate_ref_rad_s'])
        self.weights = reward_weights(config)
        validate_generator(flight['generator'])

        M = MAX_WAYPOINTS
        self.positions = torch.zeros(num_envs, M, 3, device=device)
        self.kinds = torch.full((num_envs, M), NONE, dtype=torch.long, device=device)
        self.radii = torch.ones(num_envs, M, device=device)
        self.holds = torch.zeros(num_envs, M, device=device)
        self.count = torch.ones(num_envs, dtype=torch.long, device=device)
        self.index = torch.zeros_like(self.count)
        self.hold_elapsed = torch.zeros(num_envs, device=device)
        self.start = torch.zeros(num_envs, 3, device=device)
        # Per-episode operational volume: route bounding box + margin.
        self.fence_low = torch.zeros(num_envs, 3, device=device)
        self.fence_high = torch.zeros(num_envs, 3, device=device)
        self.last_potential = {name: torch.zeros(num_envs, device=device) for name in _POTENTIAL_TERMS}
        # Hover settle/hold value earned on already captured waypoints, so a
        # capture never reads as losing the hover potential.
        self.banked = {name: torch.zeros(num_envs, device=device) for name in ('hover_settle', 'hover_hold')}
        self.velocity = torch.zeros(num_envs, 3, device=device)
        self.rates = torch.zeros(num_envs, 3, device=device)
        self.outcome = torch.zeros_like(self.count)
        self.captured_now = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.success_now = torch.zeros_like(self.captured_now)
        self.failed_now = torch.zeros_like(self.captured_now)
        self.landing_quality_now = torch.zeros(num_envs, device=device)
        self.episode = {name: torch.zeros(num_envs, device=device)
                        for name in ('energy_wh', 'time_s', 'captured', 'route_m', 'flown_m',
                                     'touchdown_speed', 'pad_distance', 'final_land')}
        self.last_terms = {name: torch.zeros(num_envs, device=device) for name in REWARD_TERMS}
        self.explicit = None

    # ---- Mission definition ----

    @property
    def generator(self) -> dict:
        return self.config['task']['waypoint_flight']['generator']

    def set_explicit_missions(self, missions: list[list[dict]] | None) -> None:
        """Fly fixed missions (mission control, benchmarks); env i uses i % len."""
        if missions is None:
            self.explicit = None
            return
        if not missions:
            raise ValueError('Provide at least one explicit mission or None')
        parsed = [parse_mission(m, self.touchdown_height, self.min_clearance) for m in missions]
        self.explicit = {key: torch.stack([p[key] for p in parsed]).to(self.device) for key in parsed[0]}

    def _sample(self, env_ids: Tensor, start: Tensor):
        g = self.generator
        m, M, dev = len(env_ids), MAX_WAYPOINTS, self.device

        def uniform(pair, shape=(m,)):
            lo, hi = float(pair[0]), float(pair[1])
            return lo + torch.rand(shape, device=dev) * (hi - lo)

        origin = self.origins[env_ids]
        arena = float(g['arena_half_width_m'])
        alt_lo, alt_hi = (float(v) for v in g['altitude_m'])
        count = torch.randint(int(g['count_range'][0]), int(g['count_range'][1]) + 1, (m,), device=dev)
        final_land = torch.rand(m, device=dev) < float(g['final_land_probability'])
        positions = torch.zeros(m, M, 3, device=dev)
        kinds = torch.full((m, M), NONE, dtype=torch.long, device=dev)
        radii = torch.ones(m, M, device=dev)
        holds = torch.zeros(m, M, device=dev)
        previous = start.clone()
        for j in range(M):
            valid = j < count
            final = j == count - 1
            land = valid & final & final_land
            azimuth = torch.rand(m, device=dev) * (2 * math.pi)
            direction = torch.stack((azimuth.cos(), azimuth.sin()), dim=-1)
            horizontal = uniform(g['leg_horizontal_m'])
            horizontal = torch.where(torch.rand(m, device=dev) < float(g['vertical_leg_probability']),
                                     torch.zeros_like(horizontal), horizontal)
            local = previous - origin
            z = (local[:, 2] + uniform(g['leg_vertical_m'])).clamp(alt_lo, alt_hi)
            # Enforce a minimum leg by lengthening horizontally, then reflect
            # any arena exit (the previous point is inside, so this lands inside).
            vertical = (z - local[:, 2]).abs()
            needed = (float(g['min_leg_m']) ** 2 - vertical.square()).clamp(min=0).sqrt()
            horizontal = torch.maximum(horizontal, needed)
            candidate = local[:, :2] + horizontal[:, None] * direction
            outside = (candidate.abs() > arena).any(-1)
            candidate = torch.where(outside[:, None], local[:, :2] - horizontal[:, None] * direction, candidate)
            candidate = candidate.clamp(-arena, arena)
            airborne = torch.cat((candidate, z[:, None]), dim=-1)
            land_offset = uniform(g['land_leg_horizontal_m'])
            land_xy = (local[:, :2] + land_offset[:, None] * direction).clamp(-arena, arena)
            ground = torch.cat((land_xy, torch.full_like(z[:, None], self.touchdown_height)), dim=-1)
            point = torch.where(land[:, None], ground, airborne) + origin
            hover = final | (torch.rand(m, device=dev) < float(g['intermediate_hover_probability']))
            kind = torch.where(land, torch.full_like(count, LAND),
                               torch.where(hover, torch.full_like(count, HOVER), torch.full_like(count, FLYPASS)))
            radius = torch.where(kind == LAND, uniform(g['land_radius_m']),
                                 torch.where(kind == HOVER, uniform(g['hover_radius_m']), uniform(g['flypass_radius_m'])))
            hold = torch.where(kind == HOVER, uniform(g['hover_hold_s']), torch.zeros(m, device=dev))
            positions[:, j] = torch.where(valid[:, None], point, previous)
            kinds[:, j] = torch.where(valid, kind, torch.full_like(kind, NONE))
            radii[:, j] = torch.where(valid, radius, radii[:, j])
            holds[:, j] = torch.where(valid, hold, holds[:, j])
            previous = torch.where(valid[:, None], point, previous)
        return positions, kinds, radii, holds, count

    def reset(self, env_ids: Tensor, position: Tensor, velocity: Tensor | None = None,
              rates_frd: Tensor | None = None) -> None:
        if len(env_ids) == 0:
            return
        self.velocity[env_ids] = 0.0 if velocity is None else velocity[env_ids]
        self.rates[env_ids] = 0.0 if rates_frd is None else rates_frd[env_ids]
        start = position[env_ids]
        self.start[env_ids] = start
        if self.explicit is not None:
            select = env_ids % self.explicit['count'].shape[0]
            positions = self.explicit['positions'][select] + self.origins[env_ids, None, :]
            kinds, radii = self.explicit['kinds'][select], self.explicit['radii'][select]
            holds, count = self.explicit['holds'][select], self.explicit['count'][select]
        else:
            positions, kinds, radii, holds, count = self._sample(env_ids, start)
        self.positions[env_ids], self.kinds[env_ids] = positions, kinds
        self.radii[env_ids], self.holds[env_ids], self.count[env_ids] = radii, holds, count
        self.index[env_ids] = 0
        self.hold_elapsed[env_ids] = 0
        self.outcome[env_ids] = RUNNING
        for value in self.episode.values():
            value[env_ids] = 0
        route = torch.cat((start[:, None], positions), dim=1)
        legs = (route[:, 1:] - route[:, :-1]).norm(dim=-1)
        legs = legs * (torch.arange(MAX_WAYPOINTS, device=self.device)[None] < count[:, None])
        self.episode['route_m'][env_ids] = legs.sum(-1)
        # Mission-relative geofence. Run 20260923_151130 (v5) mastered stage 0,
        # then in stage 1 every episode climbed into the fixed 45 m ceiling: a
        # climb there postponed the -400 failure by ~12 s, longer than a
        # sinking crash, so "more throttle" was a stable local optimum. The
        # operational volume is the route bounding box (start + waypoints)
        # plus margin_m, capped by the arena and the absolute ceiling.
        valid = (torch.arange(MAX_WAYPOINTS + 1, device=self.device)[None] <= count[:, None])[..., None]
        low = torch.where(valid, route, torch.full_like(route, float('inf'))).amin(1)
        high = torch.where(valid, route, torch.full_like(route, -float('inf'))).amax(1)
        origin = self.origins[env_ids]
        arena = float(self.generator['arena_half_width_m']) + self.geofence_margin
        self.fence_low[env_ids] = torch.maximum(low - self.geofence_margin, origin - arena)
        self.fence_high[env_ids] = torch.minimum(high + self.geofence_margin,
                                                 origin + torch.tensor([arena, arena, self.ceiling],
                                                                       device=self.device))
        self.episode['final_land'][env_ids] = (kinds[torch.arange(len(env_ids), device=self.device), count - 1]
                                               == LAND).float()
        for value in self.banked.values():
            value[env_ids] = 0
        for name, value in self.potentials(position).items():
            self.last_potential[name][env_ids] = value[env_ids]

    # ---- Per-step logic ----

    def potentials(self, position: Tensor) -> dict:
        """Unweighted dense-reward potentials; the reward is their per-step change.

        The change is undiscounted and there is no refund at termination, so
        route progress lost by drifting stays lost. Diagnostic: runs up to
        20260923_200423 used potential-based shaping with Psi = 0 at absorbing
        states. A vehicle that drifted 10 m and hit the geofence got the lost
        progress back on its terminal step, inside GAE's ~0.7 s credit window
        of the drift, so failures carried no net dense signal. Stage-0
        position holds fell to 1-7% success while ~90% of episodes ended at the
        geofence. Totals are bounded by the route length (progress) and the
        number of hover waypoints (settle, hold), so rule 2 is unaffected.

        progress: -(distance to the active waypoint + remaining legs), m.
        hover_settle and hover_hold apply on HOVER legs only. Diagnostic: run
        20260923_144426 fit its critic (explained variance 0.99), yet by 7M
        transitions every stage-0 episode climbed into the 45 m geofence.
        With gamma = 0.999, postponing a -400 failure is worth
        (1 - gamma) * 400 = 0.4 per step (~12 /s), four times the progress paid
        at 3 m/s, and nothing dense preferred arriving slowly, so hover holds
        were never sampled. These potential-based terms (Ng et al. 1999) pay
        for settling and holding near a hover target without changing the
        optimal policy.

        hover_settle is exp(-d/scale) * exp(-motion): 1 when still on the
        target, 0 far away. Run 20260923_145644 used -exp(-d/scale) * motion,
        which is maximal (zero) far away and so paid a moving vehicle for
        leaving the target; stage-0 success peaked at 50% and then decayed
        to 10% as throttle drifted up into geofence climbs.
        """
        on_hover = (self.kinds[self.ids, self.index] == HOVER).float()
        hold = self.holds[self.ids, self.index]
        return dict(progress=self.potential(position),
                    hover_settle=self.banked['hover_settle'] + self.hover_stillness(position),
                    hover_hold=self.banked['hover_hold']
                    + on_hover * (self.hold_elapsed / hold.clamp(min=self.dt)).clamp(max=1.0))

    def hover_stillness(self, position: Tensor) -> Tensor:
        """exp(-d/scale - |v|/v_ref - |w|/w_ref) on HOVER legs, else 0; 1 = still on target."""
        on_hover = (self.kinds[self.ids, self.index] == HOVER).float()
        distance = (self.positions[self.ids, self.index] - position).norm(dim=-1)
        motion = (self.velocity.norm(dim=-1) / self.settle_speed_ref
                  + self.rates.norm(dim=-1) / self.settle_rate_ref)
        return on_hover * torch.exp(-distance / self.settle_distance - motion)

    def track_stillness(self, position: Tensor) -> Tensor:
        """hover_track shape on HOVER and LAND legs (LAND: distance to the pad point).

        Replay of run 20260924_151720's final (400M) checkpoint on LAND
        missions from 10 m: on every LAND leg the vehicle climbed to 15-17 m
        and circled there for 40 s (deterministic and stochastic alike); full-
        task evaluations at 300M and 400M had 0% land-mission success. LAND
        legs only had the telescoping progress term, so staying high cost at
        most the ~15 m of lost progress against a -400 risk at touchdown.
        """
        kind = self.kinds[self.ids, self.index]
        on_target = ((kind == HOVER) | (kind == LAND)).float()
        distance = (self.positions[self.ids, self.index] - position).norm(dim=-1)
        motion = (self.velocity.norm(dim=-1) / self.settle_speed_ref
                  + self.rates.norm(dim=-1) / self.settle_rate_ref)
        return on_target * torch.exp(-distance / self.settle_distance - motion)

    def potential(self, position: Tensor) -> Tensor:
        """Phi = -(distance to the active waypoint + remaining route legs), m."""
        current = self.positions[self.ids, self.index]
        remaining = (current - position).norm(dim=-1)
        legs = (self.positions[:, 1:] - self.positions[:, :-1]).norm(dim=-1)
        leg_index = torch.arange(MAX_WAYPOINTS - 1, device=self.device)[None]
        active = (leg_index >= self.index[:, None]) & (leg_index + 1 < self.count[:, None])
        return -(remaining + (legs * active).sum(-1))

    def step(self, before: Tensor, after: Tensor, velocity_world: Tensor, rates_frd: Tensor,
             contact_state: Tensor, touchdown_speed: Tensor, energy_wh: Tensor,
             tilt_failed: Tensor, altitude_failed: Tensor) -> Tensor:
        """Advance captures and classify the outcome; returns ``terminated``."""
        ids, index = self.ids, self.index
        self.velocity, self.rates = velocity_world, rates_frd
        goal, kind, radius = self.positions[ids, index], self.kinds[ids, index], self.radii[ids, index]
        landed = contact_state == int(ContactState.LANDED)
        crashed = contact_state == int(ContactState.CRASHED)

        fly = (kind == FLYPASS) & (segment_distance(goal, before, after) <= radius)
        # Read live: the curriculum may relax the hold gate in early stages.
        gate = self.config['task']['waypoint_flight']['hover_capture']
        stable = (((after - goal).norm(dim=-1) <= radius)
                  & (velocity_world.norm(dim=-1) <= float(gate['max_speed_m_s']))
                  & (rates_frd.norm(dim=-1) <= math.radians(float(gate['max_body_rate_deg_s']))))
        self.hold_elapsed = torch.where((kind == HOVER) & stable, self.hold_elapsed + self.dt,
                                        torch.zeros_like(self.hold_elapsed))
        hover = (kind == HOVER) & (self.hold_elapsed >= self.holds[ids, index] - 1e-6)
        pad_distance = (after[:, :2] - goal[:, :2]).norm(dim=-1)
        at_pad = (kind == LAND) & landed
        # Read live like the hover gate: the curriculum may relax the touchdown gate.
        touchdown_gate = float(self.config['task']['waypoint_flight']['landing_gates']['max_touchdown_speed_m_s'])
        land = at_pad & (pad_distance <= radius) & (touchdown_speed <= touchdown_gate)
        captured = fly | hover | land
        final = index >= self.count - 1

        outcome = torch.where(captured & final, torch.full_like(index, SUCCESS), torch.zeros_like(index))
        for mask, code in ((at_pad & ~land, BAD_LANDING),
                           (landed & (kind != LAND), PREMATURE_LANDING),
                           ((after[:, :2] < self.fence_low[:, :2]).any(-1)
                            | (after > self.fence_high).any(-1), GEOFENCE),
                           ((rates_frd.abs() > self.rate_limit).any(-1), SPIN),
                           (altitude_failed, ALTITUDE),
                           (crashed & ~tilt_failed, CRASH),
                           (tilt_failed, TILT)):
            outcome = torch.where(mask, torch.full_like(outcome, code), outcome)
        failed = (outcome != RUNNING) & (outcome != SUCCESS)
        advance = captured & ~final & ~failed
        current = self.potentials(after)
        for name, value in self.banked.items():
            self.banked[name] = torch.where(advance, current[name], value)
        self.index = index + advance.long()
        self.hold_elapsed = torch.where(advance, torch.zeros_like(self.hold_elapsed), self.hold_elapsed)
        self.captured_now = captured & ~failed
        self.success_now = outcome == SUCCESS
        self.failed_now = failed
        self.outcome = outcome
        # Graded touchdown quality at the pad (also for failed gates) so a
        # hard or off-centre landing still reports which way to improve.
        self.landing_quality_now = torch.where(
            at_pad, torch.exp(-touchdown_speed / touchdown_gate) * torch.exp(-pad_distance / radius),
            torch.zeros_like(pad_distance))
        ep = self.episode
        ep['energy_wh'] += energy_wh
        ep['time_s'] += self.dt
        ep['captured'] += self.captured_now.float()
        ep['flown_m'] += (after - before).norm(dim=-1)
        ep['touchdown_speed'] = torch.where(at_pad, touchdown_speed, ep['touchdown_speed'])
        ep['pad_distance'] = torch.where(at_pad, pad_distance, ep['pad_distance'])
        return outcome != RUNNING

    def reward(self, position: Tensor, rates_frd: Tensor, terminated: Tensor,
               action_delta: Tensor, energy_wh: Tensor) -> Tensor:
        """Weighted reward; call after ``step`` with the same terminated mask.

        Dense terms are Psi(s') - Psi(s) with no terminal refund, so each
        episode's dense total is Psi(s_T) - Psi(s_0).
        """
        del terminated  # terminal states keep their potential (no refund)
        shaped = {}
        for name, value in self.potentials(position).items():
            shaped[name] = value - self.last_potential[name]
            self.last_potential[name] = value
        values = dict(
            **shaped,
            # Per-step (not differenced) station-keeping bonus; see the YAML weight.
            hover_track=self.track_stillness(position) * self.dt,
            waypoint_capture=self.captured_now.float(),
            mission_success=self.success_now.float(),
            landing_quality=self.landing_quality_now,
            failure=self.failed_now.float(),
            time=torch.full_like(energy_wh, self.dt),
            energy=energy_wh,
            body_rate=(rates_frd / self.rate_soft).square().sum(-1) * self.dt,
            action_rate=action_delta.square().sum(-1),
        )
        self.last_terms = {key: self.weights[key] * values[key] for key in REWARD_TERMS}
        return torch.stack(tuple(self.last_terms.values())).sum(0)

    # ---- Observation ----

    def observation(self, position, quaternion_wxyz, linear_vel_frd, angular_vel_frd, height,
                    fin_angles, fin_rates, rotor_fraction, contact_state, battery_obs, previous_action,
                    max_fin_angle: float, max_fin_rate: float, noise: dict | None = None) -> Tensor:
        """51 channels, every vector in body FRD, all scaled to O(1).

        [0:3] active waypoint - position, [3:6] next - active, [6:9] next2 - next,
        [9:12] active kind (flypass/hover/land), [12:16] next kind (+none),
        [16:20] next2 kind (+none), [20] active radius, [21] hover hold left,
        [22:25] gravity direction, [25:28] velocity, [28:31] body rates,
        [31] height, [32:36] fin angles, [36:40] fin rates, [40] rotor speed,
        [41] contact state, [42:46] battery, [46:51] previous action,
        [51:54] active waypoint - position at a 2 m scale, saturated at 1.5, so
        sub-metre hover and landing errors are not 0.05-sized inputs.
        Heading is irrelevant for an axisymmetric vehicle whose targets are all
        body-relative, so attitude enters only through gravity and the vectors.
        """
        if noise and noise.get('enabled', False):
            n, dev = position.shape[0], position.device
            position_noise = torch.randn(n, 3, device=dev) * float(noise.get('position_std', 0.0))
            position = position + position_noise
            height = height + position_noise[:, 2]
            attitude_std = float(noise.get('attitude_std', 0.0))
            if attitude_std > 0:
                euler = torch.randn(n, 3, device=dev) * attitude_std
                quaternion_wxyz = normalize(multiply(quaternion_wxyz,
                                                     from_euler(euler[:, 0], euler[:, 1], euler[:, 2])))
            linear_vel_frd = linear_vel_frd + torch.randn(n, 3, device=dev) * float(noise.get('velocity_std', 0.0))
            angular_vel_frd = angular_vel_frd + torch.randn(n, 3, device=dev) * float(
                noise.get('angular_velocity_std', 0.0))
        # The state as the flight computer measured it (truth when noise is
        # off); telemetry records it beside the PhysX pose.
        self.measurement = dict(position=position, quaternion_wxyz=quaternion_wxyz,
                                linear_vel_frd=linear_vel_frd, angular_vel_frd=angular_vel_frd)
        q_inv = inverse(normalize(quaternion_wxyz))

        def body(vector):
            return isaac_position_to_frd(rotate_vector(q_inv, vector))

        ids, index, M = self.ids, self.index, MAX_WAYPOINTS
        has1, has2 = index + 1 < self.count, index + 2 < self.count
        i1, i2 = (index + 1).clamp(max=M - 1), (index + 2).clamp(max=M - 1)
        wp0 = self.positions[ids, index]
        wp1 = torch.where(has1[:, None], self.positions[ids, i1], wp0)
        wp2 = torch.where(has2[:, None], self.positions[ids, i2], wp1)
        none = torch.full_like(index, NONE)
        k0 = self.kinds[ids, index]
        k1 = torch.where(has1, self.kinds[ids, i1], none)
        k2 = torch.where(has2, self.kinds[ids, i2], none)
        one_hot = torch.nn.functional.one_hot
        scale = self.distance_scale
        down = torch.zeros_like(position)
        down[:, 2] = -1.0
        hold_left = (self.holds[ids, index] - self.hold_elapsed).clamp(min=0)
        return torch.cat((
            _clip_norm(body(wp0 - position) / scale, 3.0),
            _clip_norm(body(wp1 - wp0) / scale, 3.0),
            _clip_norm(body(wp2 - wp1) / scale, 3.0),
            one_hot(k0, 4)[:, :3].float(), one_hot(k1, 4).float(), one_hot(k2, 4).float(),
            (self.radii[ids, index] / self.radius_scale)[:, None],
            (hold_left / self.hold_scale)[:, None],
            isaac_to_frd(rotate_vector(q_inv, down)),
            linear_vel_frd / self.velocity_scale,
            angular_vel_frd / math.pi,
            (height / self.height_scale)[:, None],
            fin_angles / max_fin_angle,
            fin_rates / max_fin_rate,
            rotor_fraction[:, None],
            (contact_state.float() / 3.0)[:, None],
            battery_obs,
            previous_action,
            _clip_norm(body(wp0 - position) / 2.0, 1.5),
        ), dim=-1)

    # ---- Telemetry ----

    def snapshot(self) -> dict:
        """Per-env episode telemetry, captured before auto-reset."""
        result = {key: value.clone() for key, value in self.episode.items()}
        result.update(outcome=self.outcome.clone(), count=self.count.clone(), index=self.index.clone())
        return result

    def record(self, env_id: int = 0) -> dict:
        i, count = int(self.index[env_id]), int(self.count[env_id])
        origin = self.origins[env_id]
        return dict(waypoint_index=i, waypoint_count=count,
                    phase=KIND_NAMES[int(self.kinds[env_id, i])].upper(),
                    target_position=(self.positions[env_id, i] - origin).tolist(),
                    hold_elapsed_s=float(self.hold_elapsed[env_id]),
                    outcome=OUTCOMES[int(self.outcome[env_id])],
                    waypoints=[dict(position=(self.positions[env_id, j] - origin).tolist(),
                                    type=KIND_NAMES[int(self.kinds[env_id, j])],
                                    radius_m=float(self.radii[env_id, j]),
                                    hold_s=float(self.holds[env_id, j])) for j in range(count)],
                    episode={k: float(v[env_id]) for k, v in self.episode.items()})


def policy_to_env_action(raw: Tensor, max_fin_angle: float, throttle_center: float,
                         throttle_span: float) -> Tensor:
    """Decode the waypoint_flight_v1 action contract into physical commands.

    raw is the tanh-squashed actor output in [-1, 1]^5. Vanes: raw * max angle
    (+X, +Y, -X, -Y). Throttle: center + span * raw, clamped to [0, 1], with
    center = the level-hover duty. Diagnostic: run 20260923_152010 mapped
    throttle over [0, 1]; it learned "more throttle" from early sinking
    crashes, saturated at 0.98 duty after ~140M transitions and never learned
    altitude hold. Centring on hover spends the action resolution where
    altitude hold happens. At span 0.25 the range is ~0.53-1.0 duty, i.e.
    ~45-165% of hover thrust, still enough for descent, climb and flare.
    This is a fixed action definition, not feedback: the hardware applies
    the same affine map.
    """
    fins = raw[:, :4] * max_fin_angle
    throttle = (throttle_center + throttle_span * raw[:, 4:5]).clamp(0.0, 1.0)
    return torch.cat((fins, throttle), dim=-1)


class StartCohortOutcomes:
    """Release finished-episode outcomes in episode-start order, unbiased.

    Every env resets together when a stage is installed, so successes (short
    episodes) finish first and a window of the most recent finished episodes
    fills with them before the timeouts of the same cohort arrive. Runs
    20260924_120313..151720 advanced landing at 0.505, long_legs at 0.972 and
    routes at 0.898 this way, while one-episode-per-env evaluations measured
    0% land-mission success. Outcomes are released only once every episode
    that started at the same step or earlier must have ended (episodes last
    at most ``horizon_steps``), so each release is a complete start cohort.
    """

    def __init__(self, num_envs: int, device):
        self.device = device
        self.start = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.step = 0
        self._pending: list[tuple[Tensor, Tensor, Tensor]] = []

    def restart(self) -> None:
        """All envs were just reset (stage install, evaluation): drop in-flight episodes."""
        self.start.fill_(self.step)
        self._pending.clear()

    def observe(self, done: Tensor, success: Tensor, group: Tensor) -> None:
        """Call once after every env.step with that step's done/success/group masks."""
        ids = torch.nonzero(done).flatten()
        if len(ids):
            self._pending.append((self.start[ids].clone(), success[ids].clone(), group[ids].clone()))
            self.start[ids] = self.step + 1
        self.step += 1

    def release(self, horizon_steps: int) -> tuple[list[bool], list[int]]:
        if not self._pending:
            return [], []
        start, success, group = (torch.cat(parts) for parts in zip(*self._pending))
        ready = start <= self.step - int(horizon_steps)
        self._pending = [(start[~ready], success[~ready], group[~ready])] if bool((~ready).any()) else []
        order = torch.argsort(start[ready], stable=True)
        return success[ready][order].tolist(), group[ready][order].tolist()


def policy_to_env_rate_action(raw: Tensor, max_fin_angle: float) -> Tensor:
    """Decode the throttle-rate contract (``throttle_command.mode: rate``).

    Vanes as in :func:`policy_to_env_action`; channel 4 stays the normalized
    throttle-rate command in [-1, 1], which the env integrates with
    :func:`integrate_throttle` (the flight computer does the same).
    """
    return torch.cat((raw[:, :4] * max_fin_angle, raw[:, 4:5].clamp(-1.0, 1.0)), dim=-1)


def integrate_throttle(duty: Tensor, rate_command: Tensor, max_rate_per_s: float, dt: float) -> Tensor:
    """One policy step of the throttle-rate contract: duty += rate * max_rate * dt."""
    return (duty + rate_command.clamp(-1.0, 1.0) * max_rate_per_s * dt).clamp(0.0, 1.0)


def apply_yaw_damper(fins: Tensor, yaw_rate_frd: Tensor, gain: float, max_angle: float,
                     remove_policy_common_mode: bool = False) -> Tensor:
    """Fixed flight-computer yaw-rate damper (``yaw_damper`` in the task YAML).

    Adds the common-mode deflection -gain * r to all four vane commands;
    +deflection on every vane is +yaw torque (FRD), so this opposes the yaw
    rate. With ``remove_policy_common_mode`` the policy's own common mode
    (the mean of its four commands, i.e. its yaw torque request) is removed
    first, so the flight computer alone commands yaw; the differential part
    (roll/pitch authority) passes through unchanged.
    """
    if remove_policy_common_mode:
        fins = fins - fins.mean(dim=-1, keepdim=True)
    return (fins - gain * yaw_rate_frd[:, None]).clamp(-max_angle, max_angle)


def holding_duty(rotor_fraction: Tensor, bus_voltage: Tensor, reference_voltage: float) -> Tensor:
    """Duty whose voltage-scaled rotor target equals ``rotor_fraction``.

    Inverse of ``BatteryLiPo.update_motor``: rotor target = duty * bus / reference.
    """
    return (rotor_fraction * reference_voltage / bus_voltage.clamp(min=1e-3)).clamp(0.0, 1.0)


def final_task_snapshot(config: dict) -> dict:
    """Immutable copy of the un-curricularized task for ``apply_stage``."""
    task = config['task']
    return copy.deepcopy(dict(spawn=task['spawn'], waypoint_flight=task['waypoint_flight'],
                              episode_length_s=task['episode_length_s']))
