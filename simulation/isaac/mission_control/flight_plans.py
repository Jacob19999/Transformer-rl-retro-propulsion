"""Local, versioned flight-plan library. Import/export shares the UI contract."""
import hashlib
import json
import re
from pathlib import Path

from .models import validate_mission

LIBRARY = Path(__file__).resolve().parent / 'library/plans'


def save_plan(value):
    mission = validate_mission(value)
    # Prefix avoids Windows device names; hash prevents different names that
    # slugify identically from accidentally replacing each other.
    slug = re.sub(r'[^a-z0-9]+', '-', mission['name'].lower()).strip('-')[:40] or 'flight'
    key = 'plan-' + slug + '-' + hashlib.sha256(mission['name'].encode()).hexdigest()[:10]
    record = dict(format='edf-flight-plan', version=2, mission=mission)
    LIBRARY.mkdir(parents=True, exist_ok=True)
    path = LIBRARY / f'{key}.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)
    return dict(id=key, name=mission['name'])


def read_plan(key):
    if not re.fullmatch(r'plan-[a-z0-9-]+-[a-f0-9]{10}', key):
        raise FileNotFoundError(key)
    record = json.loads((LIBRARY / f'{key}.json').read_text(encoding='utf-8'))
    return dict(format='edf-flight-plan', version=2, mission=validate_mission(record['mission']))


def list_plans():
    result = []
    for path in sorted(LIBRARY.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            record = read_plan(path.stem)
            result.append(dict(id=path.stem, name=record['mission']['name'], modified=path.stat().st_mtime))
        except (ValueError, KeyError, FileNotFoundError):
            continue
    return result
