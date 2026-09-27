"""Local, versioned flight-plan library. Import/export shares the UI contract.

A flight plan is the route only: its name and duration, the start state, the
landing pads and the steps. Guidance (convex optimizer) settings and the
environment (disturbances) are chosen separately for each run, so saving,
loading, importing or exporting a plan never reads or changes them. Files
written before this split may still carry those fields; they are dropped on read.
"""
import hashlib
import json
import re
from pathlib import Path

from .models import validate_mission

LIBRARY = Path(__file__).resolve().parent / 'library/plans'
ROUTE_FIELDS = ('name', 'duration_s', 'position', 'velocity', 'attitude_deg', 'angular_rate_deg_s',
                'initial_motor_fraction', 'pads', 'waypoints')


def validate_route(value):
    """The route fields of a plan or mission request, validated on their own.

    Step speeds are bounded by the planner speed limit of whichever guidance
    flies the route, so a route alone is checked against the widest limit a
    guidance profile may set; launching re-validates it with the chosen guidance.
    """
    if not isinstance(value, dict):
        raise ValueError('A flight plan must be an object')
    from .convex_parameters import bounds
    route = {key: value[key] for key in ROUTE_FIELDS if key in value}
    widest = {'guidance': {'max_speed_m_s': bounds('guidance', 'max_speed_m_s')[1]}}
    mission = validate_mission(dict(route, controller='convex', convex_settings=widest))
    return {key: mission[key] for key in ROUTE_FIELDS}


def record(route):
    return dict(format='edf-flight-plan', version=2, scope='route', mission=route)


def save_plan(value):
    route = validate_route(value)
    # Prefix avoids Windows device names; hash prevents different names that
    # slugify identically from accidentally replacing each other.
    slug = re.sub(r'[^a-z0-9]+', '-', route['name'].lower()).strip('-')[:40] or 'flight'
    key = 'plan-' + slug + '-' + hashlib.sha256(route['name'].encode()).hexdigest()[:10]
    LIBRARY.mkdir(parents=True, exist_ok=True)
    path = LIBRARY / f'{key}.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record(route), indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)
    return dict(id=key, name=route['name'])


def read_plan(key):
    if not re.fullmatch(r'plan-[a-z0-9-]+-[a-f0-9]{10}', key):
        raise FileNotFoundError(key)
    saved = json.loads((LIBRARY / f'{key}.json').read_text(encoding='utf-8'))
    return record(validate_route(saved['mission']))


def list_plans():
    result = []
    for path in sorted(LIBRARY.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            plan = read_plan(path.stem)
            result.append(dict(id=path.stem, name=plan['mission']['name'], modified=path.stat().st_mtime))
        except (ValueError, KeyError, FileNotFoundError):
            continue
    return result
