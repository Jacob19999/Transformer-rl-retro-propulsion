"""Bounded mission overrides for the existing simulator disturbance models."""
import copy
import math
from pathlib import Path

import yaml

FOLDER = Path(__file__).resolve().parents[1] / 'configs/disturbances'
SENSORS = Path(__file__).resolve().parents[1] / 'configs/sensors'
LIMITS = {
    'gust': {'magnitude': (0, 15), 'duration': (.05, 10)},
    'sensor_noise': {'position_std': (0, .5), 'velocity_std': (0, 2),
                     'attitude_std': (0, .1), 'angular_velocity_std': (0, 1)},
}


def imu_profiles():
    """Names of the physical IMU chains in configs/sensors/imu_<name>.yaml."""
    return sorted(path.stem[4:] for path in SENSORS.glob('imu_*.yaml'))


def imu_profile_summary(name):
    """Headline timing figures of one physical IMU chain, for the preset cards."""
    imu = yaml.safe_load((SENSORS / f'imu_{name}.yaml').read_text(encoding='utf-8'))['imu']
    return dict(sample_rate_hz=imu['sample_rate_hz'], bandwidth_hz=imu['bandwidth_hz'],
                latency_ms=imu['latency_s'] * 1000, gyro_range_dps=imu['gyro']['range_dps'],
                yaw=imu['attitude']['yaw']['mode'])


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
               'sensor_noise': set(LIMITS['sensor_noise']) | {'imu_profile', 'imu_nav'}, 'com_offset': {'range'}}
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
            elif key == 'imu_profile':
                # '' keeps the white-noise attitude/rate model; a name selects a physical IMU chain.
                if item != '' and item not in imu_profiles():
                    raise ValueError(f'Unknown IMU profile; choose one of {imu_profiles()}')
            elif key == 'imu_nav':
                # 'external': position/velocity stay a white-noise reference. 'inertial': the physical
                # IMU integrates its own accelerometer and attitude (unaided strapdown navigation).
                # 'fused': EKF3-style filter over the IMU, TFmini Plus rangefinder, MTF-01P flow and baro.
                # 'fused_marker': the same plus a downward camera on a pad marker during the descent.
                if item not in ('external', 'inertial', 'fused', 'fused_marker'):
                    raise ValueError("imu_nav must be 'external', 'inertial', 'fused' or 'fused_marker'")
            else:
                item = number(item, *LIMITS[section][key])
            result[section][key] = item
    noise = result.get('sensor_noise', {})
    if noise.get('imu_nav') in ('inertial', 'fused', 'fused_marker') and not noise.get('imu_profile'):
        raise ValueError("Inertial and fused navigation need a physical IMU chain (imu_profile)")
    return result
