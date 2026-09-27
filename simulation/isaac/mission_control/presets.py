"""Read-only built-in presets versioned with the repository.

presets/convex_profiles.json    convex guidance profiles per mission type
presets/disturbance_presets.json starting points for the Environment section
presets/flight_plans/*.json      sample flight plans (edf-flight-plan v2)

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
    from .disturbance_parameters import validate_settings
    result = []
    for item in _read('disturbance_presets.json')['presets']:
        if set(item['selected']) - {'wind', 'sensor_noise', 'com_shift'}:
            raise ValueError(f"Disturbance preset {item['id']}: unknown source")
        result.append(dict(item, selected=sorted(item['selected']), settings=validate_settings(item['settings'])))
    return result


def read_sample(key: str) -> dict:
    """One sample plan, re-validated like an imported file."""
    from .models import validate_mission
    if not isinstance(key, str) or not SAMPLE_ID.fullmatch(key):
        raise FileNotFoundError(key)
    record = json.loads((FOLDER / 'flight_plans' / f'{key}.json').read_text(encoding='utf-8'))
    if record.get('format') != 'edf-flight-plan' or record.get('version') != 2:
        raise ValueError(f'Sample {key} is not an edf-flight-plan version 2 file')
    return dict(format='edf-flight-plan', version=2, sample=dict(record['sample'], id=key),
                mission=validate_mission(record['mission']))


def list_samples() -> list[dict]:
    """Gallery entries: metadata plus the start-to-pad polyline for a thumbnail."""
    from .models import landing_pad
    result = []
    for path in sorted((FOLDER / 'flight_plans').glob('*.json')):
        record = read_sample(path.stem)
        mission = record['mission']
        points = [mission['position'], *(w['position'] for w in mission['waypoints'])]
        if not any(w['type'] == 'land' for w in mission['waypoints']):
            points.append(landing_pad(mission)['position'])
        result.append(dict(record['sample'], name=mission['name'], max_altitude_m=max(p[2] for p in points),
                           steps=len(mission['waypoints']), pads=len(mission['pads']),
                           disturbance=mission['disturbance'], route=points))
    return result
