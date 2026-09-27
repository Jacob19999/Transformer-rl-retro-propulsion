"""Adjustable convex-guidance parameters: schema, validation and saved profiles.

The mission service exposes a whitelist of configs/controllers/convex_guidance.yaml
entries with physical bounds. A mission request may override any of them
(`convex_settings`); run_mission.py deep-merges the overrides over the YAML and
records the resolved parameters in the mission metadata, so every replay says
exactly which optimizer flew it. The attitude law (LQR weights, vane models)
is deliberately not exposed: it is identified against the plant, not tuned
per mission.
"""
from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / 'configs/controllers/convex_guidance.yaml'
PROFILES = Path(__file__).resolve().parent / 'library/convex'
PROFILE_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9 _.-]{0,47}')

# (section, key, group, label, unit, low, high, step, help). `low`/`high` are
# inclusive; a None `step` marks an integer. Keys absent here cannot be set.
PARAMETERS = [
    ('guidance', 'route_corridor_mode', 'Route corridor', 'Corridor enforcement', None, ('soft', 'strict'), None, None,
     'Strict rejects plans outside the corridor and holds if infeasible. Soft penalizes excess and permits emergency fallback. Neither guarantees actual vehicle tracking.'),
    ('guidance', 'solver_workers', 'Discretization', 'Parallel solver workers', '', 1, 8, None,
     'Independent time candidates solved concurrently. Physics and controller intervals are unchanged; CPU contention may affect solver time limits.'),
    ('guidance', 'objective', 'Objective', 'Cost', None, ('energy', 'delta_v'), None, None,
     'energy: electrical Wh with the momentum-theory power law; delta_v: integral of |T|/m.'),
    ('guidance', 'max_speed_m_s', 'Kinematic limits', 'Max speed', 'm/s', 0.5, 10.0, 0.1,
     'Speed cone on every plan node (paper eq. 10). Waypoint speeds may not exceed it.'),
    ('guidance', 'max_tilt_deg', 'Kinematic limits', 'Max planned tilt', 'deg', 5.0, 30.0, 0.5,
     'Thrust-pointing cone. Must stay below the tracking tilt limit.'),
    ('guidance', 'glide_slope_deg', 'Kinematic limits', 'Glide slope', 'deg', 10.0, 85.0, 1.0,
     'Approach cone half-angle from vertical on the landing leg (paper eq. 11).'),
    ('guidance', 'glide_slope_final_s', 'Kinematic limits', 'Cone narrowing time', 's', 0.5, 10.0, 0.1,
     'A widened cone is back to nominal this long before the landing gate.'),
    ('guidance', 'route_floor_m', 'Kinematic limits', 'Route floor', 'm', 0.5, 20.0, 0.1,
     'Minimum body altitude on waypoint legs.'),
    ('guidance', 'thrust_min_weight_fraction', 'Thrust', 'Min thrust', '× weight', 0.3, 0.95, 0.01,
     'rho1 floor: vanes lose authority without jet flow.'),
    ('guidance', 'thrust_excess_reserve_fraction', 'Thrust', 'Tracking reserve', 'share', 0.1, 0.9, 0.05,
     'Share of the thrust above weight withheld from the plan for feedback.'),
    ('guidance', 'descent_brake_ratio', 'Thrust', 'Descent/brake ratio', '', 0.25, 3.0, 0.05,
     'Planned downward acceleration never exceeds this times the braking acceleration.'),
    ('guidance', 'throttle_rate_per_s', 'Thrust', 'Duty rate', '/s', 0.05, 1.0, 0.01,
     'Upper bound of the duty slew (the rotor reaction yaws the body).'),
    ('guidance', 'throttle_rate_yaw_authority_fraction', 'Thrust', 'Duty-rate yaw share', 'share', 0.1, 1.0, 0.05,
     'Duty slew also capped so its rotor reaction uses this share of the yaw authority.'),
    ('guidance', 'tilt_rate_authority_fraction', 'Thrust', 'Tilt-rate share', 'share', 0.1, 1.0, 0.05,
     'Planned attitude slew as a share of what full vane deflection sustains.'),
    ('guidance', 'route_corridor_m', 'Route corridor', 'Default half-width', 'm', 0.2, 25.0, 0.1,
     'Corridor around the drawn route for steps without their own width.'),
    ('guidance', 'route_corridor_weight', 'Route corridor', 'Excess penalty', 's hover / m·s', 0.5, 1000.0, 0.5,
     'Price of flying outside the corridor (soft constraint). Higher follows the corridor harder.'),
    ('guidance', 'flypass_capture_fraction', 'Route corridor', 'Fly-through aim', '× radius', 0.1, 1.0, 0.05,
     'Plan fly-throughs inside this share of their capture radius.'),
    ('guidance', 'landing_nodes', 'Discretization', 'Landing nodes', '', 8, 60, None,
     'Intervals on the powered-descent leg.'),
    ('guidance', 'route_dt_s', 'Discretization', 'Route interval', 's', 0.15, 1.0, 0.05,
     'Target node spacing on waypoint legs.'),
    ('guidance', 'max_leg_nodes', 'Discretization', 'Max nodes per leg', '', 10, 80, None,
     'Longer legs get wider node spacing.'),
    ('guidance', 'max_solve_time_s', 'Discretization', 'Solver time limit', 's', 0.05, 2.0, 0.05,
     'Clarabel time limit per SOCP.'),
    ('guidance', 'replan_period_s', 'Re-planning', 'Re-plan period', 's', 0.2, 3.0, 0.05,
     'Closed-loop re-solve interval.'),
    ('guidance', 'replan_error_m', 'Re-planning', 'Re-plan position error', 'm', 0.2, 5.0, 0.05,
     'Beyond this tracking error the plan restarts from the measured state.'),
    ('guidance', 'replan_velocity_error_m_s', 'Re-planning', 'Re-plan velocity error', 'm/s', 0.2, 5.0, 0.05,
     'Beyond this velocity error the plan restarts from the measured state.'),
    ('guidance', 'freeze_time_s', 'Re-planning', 'Freeze before gate', 's', 0.0, 3.0, 0.1,
     'No re-solve this close to the landing gate.'),
    ('guidance', 'replan_blend_s', 'Re-planning', 'Feedforward blend', 's', 0.0, 1.0, 0.05,
     'Cross-fade of the thrust feedforward after a re-plan.'),
    ('guidance', 'gate_height_m', 'Landing', 'Gate height', 'm', 0.2, 3.0, 0.05,
     'Powered descent ends this far above touchdown height.'),
    ('guidance', 'terminal_max_descent_m_s', 'Landing', 'Terminal max descent', 'm/s', 0.05, 0.5, 0.01,
     'Rate clamp of the vertical terminal descent (a Landing step sets it from its touchdown speed).'),
    ('guidance', 'terminal_center_radius_m', 'Landing', 'Centering radius', 'm', 0.05, 1.0, 0.05,
     'Full descent rate only while this well centered over the pad.'),
    ('guidance', 'gate_capture_radius_m', 'Landing', 'Gate capture radius', 'm', 0.05, 1.5, 0.05,
     'Hand over to terminal descent when this close to the gate.'),
    ('tracking', 'kp_xy', 'Tracking feedback', 'kp horizontal', '1/s²', 0.0, 5.0, 0.05, 'Position feedback.'),
    ('tracking', 'kd_xy', 'Tracking feedback', 'kd horizontal', '1/s', 0.0, 5.0, 0.05, 'Velocity feedback.'),
    ('tracking', 'ki_xy', 'Tracking feedback', 'ki horizontal', '1/s³', 0.0, 2.0, 0.05, 'Integral (wind, CoM trim).'),
    ('tracking', 'kp_z', 'Tracking feedback', 'kp vertical', '1/s²', 0.0, 5.0, 0.05, 'Position feedback.'),
    ('tracking', 'kd_z', 'Tracking feedback', 'kd vertical', '1/s', 0.0, 6.0, 0.05, 'Velocity feedback.'),
    ('tracking', 'ki_z', 'Tracking feedback', 'ki vertical', '1/s³', 0.0, 2.0, 0.05, 'Integral (thrust bias).'),
    ('tracking', 'max_correction_m_s2', 'Tracking feedback', 'Max correction', 'm/s²', 0.5, 8.0, 0.1,
     'Feedback acceleration bound per axis group.'),
    ('tracking', 'max_tilt_deg', 'Tracking feedback', 'Max commanded tilt', 'deg', 5.0, 35.0, 0.5,
     'Tilt command limit; plans stay under the planned tilt.'),
]
_INDEX = {(section, key): row for section, key, *row in PARAMETERS}


def base_settings() -> dict:
    return yaml.safe_load(BASE_PATH.read_text(encoding='utf-8'))


def schema() -> list[dict]:
    """Parameter table for the console, with the repository defaults."""
    base = base_settings()
    rows = []
    for section, key, group, label, unit, low, high, step, text in PARAMETERS:
        row = dict(section=section, key=key, group=group, label=label, unit=unit, help=text,
                   default=base[section][key])
        if isinstance(low, tuple):
            row.update(kind='choice', choices=list(low))
        else:
            row.update(kind='integer' if step is None else 'number', min=low, max=high, step=step or 1)
        rows.append(row)
    return rows


def validate_settings(value) -> dict:
    """Bounded overrides, {section: {key: value}}; unknown keys are rejected."""
    if value is None:
        return {}
    if not isinstance(value, dict) or set(value) - {'guidance', 'tracking'}:
        raise ValueError('Convex settings may only override guidance and tracking parameters')
    result = {}
    for section, entries in value.items():
        if not isinstance(entries, dict):
            raise ValueError(f'Convex settings: {section} must be an object')
        for key, item in entries.items():
            row = _INDEX.get((section, key))
            if row is None:
                raise ValueError(f'Convex setting {section}.{key} is not adjustable')
            _, label, _, low, high, step, _ = row
            if isinstance(low, tuple):
                if item not in low:
                    raise ValueError(f'{label} must be one of {", ".join(low)}')
            else:
                if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) \
                        or not low <= item <= high:
                    raise ValueError(f'{label} must be a number between {low} and {high}')
                if step is None:
                    if item != int(item):
                        raise ValueError(f'{label} must be an integer')
                    item = int(item)
                else:
                    item = float(item)
            result.setdefault(section, {})[key] = item
    resolved = resolve(result)
    if resolved['tracking']['max_tilt_deg'] < resolved['guidance']['max_tilt_deg']:
        raise ValueError('Max commanded tilt must be at least the max planned tilt: feedback needs margin')
    return result


def resolve(overrides: dict | None) -> dict:
    """The YAML with the overrides applied (a new dict)."""
    settings = base_settings()
    for section, entries in (overrides or {}).items():
        settings[section].update(copy.deepcopy(entries))
    return settings


def bounds(section: str, key: str) -> tuple:
    """Inclusive (low, high) of one adjustable numeric parameter."""
    _, _, _, low, high, _, _ = _INDEX[(section, key)]
    return low, high


def max_speed(overrides: dict | None) -> float:
    return float(resolve(overrides)['guidance']['max_speed_m_s'])


# ---- saved profiles (mission_control/library/convex/<name>.json) ----

def _profile_path(name: str) -> Path:
    if not isinstance(name, str) or not PROFILE_NAME.fullmatch(name.strip()):
        raise ValueError('Profile names use 1-48 letters, digits, spaces, dots, dashes or underscores')
    slug = re.sub(r'[^a-z0-9]+', '-', name.strip().lower()).strip('-')
    if slug in {'con', 'prn', 'aux', 'nul', *(f'com{i}' for i in range(1,10)), *(f'lpt{i}' for i in range(1,10))}:
        raise ValueError('Profile name is reserved by Windows; choose a different name')
    return PROFILES / f'{slug}.json'


def list_profiles() -> list[dict]:
    if not PROFILES.is_dir():
        return []
    result = []
    for path in sorted(PROFILES.glob('*.json')):
        try:
            record = json.loads(path.read_text(encoding='utf-8'))
            result.append(dict(name=record['name'], settings=validate_settings(record.get('settings')),
                               note=record.get('note', ''), modified=path.stat().st_mtime))
        except (ValueError, KeyError, json.JSONDecodeError):
            continue  # a hand-edited file that no longer validates is skipped, never flown
    return result


def save_profile(name: str, settings, note: str = '') -> dict:
    path = _profile_path(name)
    if not isinstance(note, str) or len(note) > 200:
        raise ValueError('Profile notes are limited to 200 characters')
    record = dict(name=name.strip(), settings=validate_settings(settings), note=note.strip())
    PROFILES.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2), encoding='utf-8')
    temporary.replace(path)
    return record


def delete_profile(name: str) -> bool:
    path = _profile_path(name)
    if not path.is_file():
        return False
    path.unlink()
    return True
