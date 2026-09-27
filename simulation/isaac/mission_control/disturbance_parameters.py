"""Bounded mission overrides for the existing simulator disturbance models."""
import copy
import math
from pathlib import Path

import yaml

FOLDER = Path(__file__).resolve().parents[1] / 'configs/disturbances'
LIMITS = {
    'gust': {'magnitude': (0, 15), 'duration': (.05, 10)},
    'sensor_noise': {'position_std': (0, .5), 'velocity_std': (0, 2),
                     'attitude_std': (0, .1), 'angular_velocity_std': (0, 1)},
}


def defaults():
    result = {}
    for name, sections in [('wind', ('wind', 'gust')), ('sensor_noise', ('sensor_noise',)),
                           ('com_shift', ('com_offset',))]:
        source = yaml.safe_load((FOLDER / f'{name}.yaml').read_text())['disturbances']
        for section in sections:
            result[section] = {key: copy.deepcopy(value) for key, value in source[section].items()
                               if key != 'enabled'}
    return result


def validate_settings(value):
    if value is None:
        return {}
    allowed = {'wind': {'steady_vector'}, 'gust': {'magnitude', 'duration', 'interval'},
               'sensor_noise': set(LIMITS['sensor_noise']), 'com_offset': {'range'}}
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise ValueError('Unknown disturbance settings')

    def number(v, low, high):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not low <= v <= high:
            raise ValueError(f'Disturbance value must be finite and between {low} and {high}')
        return float(v)

    def vector(v, length, low, high):
        if not isinstance(v, list) or len(v) != length:
            raise ValueError(f'Disturbance vector requires {length} numbers')
        return [number(x, low, high) for x in v]

    result = {}
    for section, entries in value.items():
        if not isinstance(entries, dict) or set(entries) - allowed[section]:
            raise ValueError(f'Unknown {section} disturbance parameters')
        result[section] = {}
        for key, item in entries.items():
            if key == 'steady_vector':
                item = vector(item, 3, -15, 15)
                if math.hypot(*item[:2]) > 15 + 1e-9:
                    raise ValueError('Horizontal wind speed may not exceed 15 m/s')
            elif key == 'interval':
                item = vector(item, 2, .1, 120)
                if item[0] > item[1]:
                    raise ValueError('Minimum gust interval may not exceed maximum')
            elif key == 'range':
                if not isinstance(item, list) or len(item) != 2:
                    raise ValueError('COM range requires lower and upper XYZ bounds')
                item = [vector(row, 3, -.05, .05) for row in item]
                if any(low > high for low, high in zip(*item)):
                    raise ValueError('COM minimum may not exceed maximum on any axis')
            else:
                item = number(item, *LIMITS[section][key])
            result[section][key] = item
    return result
