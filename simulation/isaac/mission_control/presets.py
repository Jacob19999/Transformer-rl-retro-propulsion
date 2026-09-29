"""Read-only built-in presets versioned with the repository.

presets/convex_profiles.json    convex guidance profiles per mission type
presets/disturbance_presets.json starting points for the Environment section
presets/flight_plans/*.json      sample routes (route-only edf-flight-plan v2) with
                                 a suggested guidance profile and environment

User-saved profiles and plans live in the git-ignored library/ folder; these
presets are never written by the service. Every preset is validated with the
same functions that validate a mission request, so a preset that drifts out
of bounds after a schema change fails loudly in the tests instead of being
flown.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

FOLDER = Path(__file__).resolve().parent / 'presets'
MISSION_TYPES = ('hop', 'hover', 'land')
SAMPLE_ID = re.compile(r'[0-9]{2}-[a-z0-9-]+')


def _read(name):
    return json.loads((FOLDER / name).read_text(encoding='utf-8'))


def convex_presets() -> list[dict]:
    from .convex_parameters import validate_settings
    result = []
    for item in _read('convex_profiles.json')['profiles']:
        if item['mission'] not in MISSION_TYPES or item['corridor'] not in ('soft', 'strict'):
            raise ValueError(f"Preset {item['id']}: unknown mission type or corridor")
        settings = validate_settings(item['settings'])
        if settings.get('guidance', {}).get('route_corridor_mode', 'soft') != item['corridor']:
            raise ValueError(f"Preset {item['id']}: corridor label does not match its settings")
        result.append(dict(item, settings=settings, builtin=True))
    return result


def disturbance_presets() -> list[dict]:
    from .disturbance_parameters import imu_profile_summary, imu_profiles, validate_settings
    result = []
    for item in _read('disturbance_presets.json')['presets']:
        if set(item['selected']) - {'wind', 'sensor_noise', 'com_shift'}:
            raise ValueError(f"Disturbance preset {item['id']}: unknown source")
        if item.get('group') == 'imu':
            # A hardware preset replaces the whole sensor model and nothing else.
            noise = item['settings'].get('sensor_noise', {})
            if item['selected'] != ['sensor_noise'] or set(item['settings']) != {'sensor_noise'} \
                    or set(noise) != {'position_std', 'velocity_std', 'attitude_std', 'angular_velocity_std'}:
                raise ValueError(f"IMU preset {item['id']} must set all four sensor_noise channels only")
            if not {'part', 'source', 'datasheet', 'mapping'} <= set(item.get('hardware', {})):
                raise ValueError(f"IMU preset {item['id']} needs its datasheet source and mapping")
            if item['hardware'].get('imu_profile') not in imu_profiles():
                raise ValueError(f"IMU preset {item['id']} needs an imu_profile from configs/sensors")
            item = dict(item, physical=imu_profile_summary(item['hardware']['imu_profile']))
        result.append(dict(item, selected=sorted(item['selected']), settings=validate_settings(item['settings'])))
    return result


def read_sample(key: str) -> dict:
    """One sample plan: a route, plus the guidance profile and environment it suggests.

    Plans are route-only (flight_plans.validate_route); the suggestions are
    applied only when the operator chooses to.
    """
    from .disturbance_parameters import validate_settings
    from .flight_plans import record, validate_route
    if not isinstance(key, str) or not SAMPLE_ID.fullmatch(key):
        raise FileNotFoundError(key)
    saved = json.loads((FOLDER / 'flight_plans' / f'{key}.json').read_text(encoding='utf-8'))
    if saved.get('format') != 'edf-flight-plan' or saved.get('version') != 2 or saved.get('scope') != 'route':
        raise ValueError(f'Sample {key} is not a route-only edf-flight-plan version 2 file')
    sample = dict(saved['sample'], id=key)
    profile = next((p for p in convex_presets() if p['id'] == sample.get('profile')), None)
    if profile is None:
        raise ValueError(f'Sample {key} suggests an unknown guidance profile')
    sample['profile_name'] = profile['name']
    if 'environment' in sample:
        environment = sample['environment']
        if set(environment['selected']) - {'wind', 'sensor_noise', 'com_shift'}:
            raise ValueError(f'Sample {key} suggests an unknown disturbance')
        sample['environment'] = dict(selected=sorted(environment['selected']),
                                     settings=validate_settings(environment['settings']))
    return dict(record(validate_route(saved['mission'])), sample=sample)


def list_samples() -> list[dict]:
    """Gallery entries: metadata plus the start-to-pad polyline for a thumbnail."""
    from .models import landing_pad
    result = []
    for path in sorted((FOLDER / 'flight_plans').glob('*.json')):
        plan = read_sample(path.stem)
        route = plan['mission']
        points = [route['position'], *(w['position'] for w in route['waypoints'])]
        if not any(w['type'] == 'land' for w in route['waypoints']):
            points.append(landing_pad(route)['position'])
        result.append(dict(plan['sample'], name=route['name'], max_altitude_m=max(p[2] for p in points),
                           steps=len(route['waypoints']), pads=len(route['pads']),
                           disturbance=plan['sample'].get('environment', {}).get('selected', []), route=points))
    return result
