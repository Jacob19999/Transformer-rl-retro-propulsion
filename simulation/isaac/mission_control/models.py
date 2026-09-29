"""Validated mission inputs and fixed local controller choices."""
from pathlib import Path
import copy
import math
import yaml

ROOT = Path(__file__).resolve().parents[1]
# Explicit controllers (no checkpoint). 'convex' is the SOCP
# powered-descent guidance in tvc_env/controllers/convex_guidance.py.
CLASSICAL_CONTROLLERS = ('convex',)
# Missions fly the momentum-bounded coupled jet: each vane turns at most its quarter of the jet, so its side force
# cannot exceed (T/4) sin(angle). The pre-audit 'legacy' plant (independent
# q*S*CNa airfoils plus a 0.27 N m s/rad damper, 8.5x the momentum-bounded
# torque per degree, tvc_env/dynamics/coupled_jet.py) and the PID baseline,
# which only flew it (on momentum vanes it drifted 11 m off the pad, Isaac
# mission 6421f6ec55a4), were removed from mission control on 2026-09-26.
# Recorded legacy and PID missions still replay.
VANE_MODELS = ('momentum',)
# Landing pads are flat targets on the ground plane (the Isaac scene has no
# pad collider; the ground is the contact surface). The landing step names
# one; without a landing step the vehicle lands on the first pad.
HOME_PAD = dict(name='Home pad', position=[0., 0., 0.])
MAX_PADS = 4
PAD_SEPARATION_M = 3.   # pad markings are 2.5 m across
# 10 simulated minutes: the 8S 5 Ah pack hovers for roughly 6 minutes, so
# the battery, not the clock, bounds the longest flights.
MAX_DURATION_S = 600.
DEFAULTS = dict(name='Landing test', controller='convex', seed=2026, duration_s=120., fast_live=True, cpu_physics=False,
                hardware_profile='planned_8s', vane_model='momentum',
                position=[-.28, .82, 18.], velocity=[0., 0., -1.],
                attitude_deg=[0., 0., 0.], angular_rate_deg_s=[0., 0., 0.],
                initial_motor_fraction=0., disturbance=[], disturbance_settings={},
                pads=[HOME_PAD], waypoints=[], convex_settings={},
                battery=dict(enabled=True, capacity_ah=5., c_rating=45., initial_soc=1.,
                             cell_resistance_ohm=.003, max_current_a=120.))
WAYPOINT_FIELDS = {'position', 'type', 'hold_s', 'radius_m', 'speed_m_s', 'name', 'corridor_m', 'pad',
                   'approach_speed_m_s'}


def finite(value, low, high, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{name} must be a finite number between {low} and {high}')
    return float(value)


def validate_spline_clearance(start, waypoints):
    """Reject underground path references by checking cubic extrema.

    The final landing leg is unconstrained descent; preceding waypoint legs
    must leave body-origin clearance above the plane. This changes no actions.
    """
    heights = [start[2], *(wp['position'][2] for wp in waypoints), 0.]
    for leg in range(len(waypoints)):
        previous = start if leg == 0 else waypoints[leg-1]['position']
        if waypoints[leg]['position'] == previous:
            continue  # Consecutive equal positions denote a stationary leg.
        if waypoints[leg].get('type') in ('takeoff', 'descent'):
            continue  # These legs use straight vertical references, not cubics.
        p0, p1, p2, p3 = (heights[max(0,leg-1)], heights[leg], heights[leg+1], heights[leg+2])
        a = .5*(-p0+3*p1-3*p2+p3)
        b = .5*(2*p0-5*p1+4*p2-p3)
        c = .5*(-p0+p2)
        candidates = [0.,1.]
        if abs(a)<1e-12:
            if abs(b)>1e-12:
                candidates.append(-c/(2*b))
        else:
            discriminant = 4*b*b-12*a*c
            if discriminant>=0:
                candidates.extend(((-2*b+math.sqrt(discriminant))/(6*a),
                                   (-2*b-math.sqrt(discriminant))/(6*a)))
        minimum = min(((a*t+b)*t+c)*t+p1 for t in candidates if 0<=t<=1)
        if minimum < .34-1e-6:
            raise ValueError(f'Spline before waypoint {leg+1} drops below ground clearance; raise or reposition its neighboring waypoints')


def validate_pads(value):
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_PADS:
        raise ValueError(f'Provide 1-{MAX_PADS} landing pads')
    pads = []
    for i, item in enumerate(value):
        if not isinstance(item, dict) or set(item) - {'name', 'position'}:
            raise ValueError(f'Pad {i+1}: unknown fields')
        position = item.get('position')
        if not isinstance(position, list) or len(position) != 3:
            raise ValueError(f'Pad {i+1}: position requires three numbers')
        x, y = (finite(v, -100, 100, f'Pad {i+1} position') for v in position[:2])
        if finite(position[2], -1, 1, f'Pad {i+1} height') != 0:
            raise ValueError(f'Pad {i+1}: pads lie on the ground plane (Z = 0)')
        name = item.get('name') or f'Pad {i+1}'
        if not isinstance(name, str) or len(name.strip()) > 24:
            raise ValueError(f'Pad {i+1}: names are limited to 24 characters')
        for other in pads:
            if math.dist(other['position'][:2], (x, y)) < PAD_SEPARATION_M:
                raise ValueError(f'Pads must be at least {PAD_SEPARATION_M:.0f} m apart')
        pads.append(dict(name=name.strip(), position=[x, y, 0.]))
    return pads


def landing_pad(mission):
    """The pad the mission lands on: its landing step's, else the first."""
    step = next((w for w in mission.get('waypoints', []) if w['type'] == 'land'), None)
    pads = mission.get('pads') or [HOME_PAD]
    return pads[step.get('pad', 0) if step else 0]


def validate_mission(value):
    if not isinstance(value, dict) or set(value) - set(DEFAULTS):
        raise ValueError('Unknown mission fields')
    result = copy.deepcopy(DEFAULTS)
    result.update(value)
    from .convex_parameters import validate_settings, max_speed
    result['convex_settings'] = validate_settings(result['convex_settings'])
    if result['convex_settings'] and result['controller'] != 'convex':
        raise ValueError('Optimizer parameters apply to convex guidance only')
    result['pads'] = validate_pads(result['pads'])
    speed_limit = max_speed(result['convex_settings']) if result['controller'] == 'convex' else 15.
    if not isinstance(result['name'], str) or not 1 <= len(result['name'].strip()) <= 80:
        raise ValueError('Mission name must contain 1–80 characters')
    result['name'] = result['name'].strip()
    if result['controller'] == 'pid':
        raise ValueError('The PID baseline was removed from mission control (it only flew the legacy vanes); '
                         'fly convex guidance')
    if result['controller'] not in CLASSICAL_CONTROLLERS:
        raise ValueError('Unknown controller')
    selected = result['disturbance']
    if isinstance(selected, str):  # Existing recorded requests remain replayable.
        selected = [] if selected == 'nominal' else [selected]
    if not isinstance(selected, list) or any(x not in ('wind', 'sensor_noise', 'com_shift') for x in selected):
        raise ValueError('Unknown disturbance')
    result['disturbance'] = sorted(set(selected))
    from .disturbance_parameters import validate_settings as validate_disturbances
    result['disturbance_settings'] = validate_disturbances(result['disturbance_settings'])
    if result['hardware_profile'] not in ('planned_8s', 'legacy_6s'):
        raise ValueError('Unknown hardware profile')
    if result['vane_model'] is None:
        result['vane_model'] = VANE_MODELS[0]
    if result['vane_model'] == 'legacy':
        raise ValueError('The legacy vane physics was removed from mission control; missions fly the '
                         'momentum-bounded jet')
    if result['vane_model'] not in VANE_MODELS:
        raise ValueError('Unknown vane model')
    seed = finite(result['seed'], 0, 2**31 - 1, 'Seed')
    if seed != int(seed):
        raise ValueError('Seed must be an integer')
    result['seed'] = int(seed)
    result['duration_s'] = finite(result['duration_s'], 1, MAX_DURATION_S, 'Duration')
    if not isinstance(result['fast_live'], bool):
        raise ValueError('Fast live must be a boolean')
    if not isinstance(result['cpu_physics'], bool):
        raise ValueError('CPU physics must be a boolean')
    for key, limits in [('position', [(-100, 100), (-100, 100), (.34, 100)]),
                        ('velocity', [(-20, 20)] * 3), ('attitude_deg', [(-180, 180)] * 3),
                        ('angular_rate_deg_s', [(-720, 720)] * 3)]:
        a = result[key]
        if not isinstance(a, list) or len(a) != 3:
            raise ValueError(f'{key} requires three numbers')
        result[key] = [finite(v, lo, hi, key) for v, (lo, hi) in zip(a, limits)]
    if not isinstance(result['waypoints'],list) or len(result['waypoints'])>12:
        raise ValueError('Provide up to 12 waypoints')
    waypoints=[]
    for i, item in enumerate(result['waypoints']):
        if not isinstance(item,dict) or set(item)-WAYPOINT_FIELDS:
            raise ValueError(f'Waypoint {i+1}: unknown fields')
        kind=item.get('type','flypass')
        if kind not in ('hover','flypass','takeoff','descent','land'):
            raise ValueError('Waypoint type must be takeoff, hover, flypass, descent or land')
        pad = None
        if kind == 'land':
            # The landing step flies to its pad; its position is the pad's.
            pad = item.get('pad', 0)
            if isinstance(pad, bool) or not isinstance(pad, int) or not 0 <= pad < len(result['pads']):
                raise ValueError(f'Waypoint {i+1}: landing pad must be one of the {len(result["pads"])} pads')
            item = dict(item, position=list(result['pads'][pad]['position']))
        elif 'pad' in item or 'approach_speed_m_s' in item:
            raise ValueError(f'Waypoint {i+1}: only a landing step selects a pad and an approach speed')
        position=item.get('position')
        if not isinstance(position,list) or len(position)!=3:
            raise ValueError(f'Waypoint {i+1}: position requires three numbers')
        position=[finite(v,lo,hi,f'Waypoint {i+1} position') for v,(lo,hi) in zip(position,[(-100,100),(-100,100),(0 if kind=='land' else 1,100)])]
        name = item.get('name', '')
        if not isinstance(name, str) or len(name.strip()) > 40:
            raise ValueError('Waypoint name must contain at most 40 characters')
        previous = waypoints[-1]['position'] if waypoints else result['position']
        if kind == 'flypass' and position == previous:
            raise ValueError('Fly-through must differ from the preceding point; use Hover for a stationary step')
        if kind == 'takeoff' and (i != 0 or position[2] <= previous[2]):
            raise ValueError('Takeoff must be the first waypoint and above the start')
        if kind == 'descent' and position[2] >= previous[2]:
            raise ValueError('Descent must be below the preceding waypoint')
        if kind in ('takeoff','descent') and position[:2] != previous[:2]:
            raise ValueError('Takeoff and descent legs must be vertical (same X/Y as preceding point)')
        if kind == 'land' and i != len(result['waypoints'])-1:
            raise ValueError('Landing must be the final waypoint')
        if kind in ('takeoff','descent','land') and result['controller'] != 'convex':
            raise ValueError('Takeoff, descent and landing waypoints require convex guidance')
        corridor = item.get('corridor_m')
        if corridor is not None:
            if result['controller'] != 'convex':
                raise ValueError('Route corridors are convex-guidance constraints')
            corridor = finite(corridor, .2, 25, f'Waypoint {i+1} corridor half-width')
        waypoint = dict(position=position,type=kind,name=name.strip(),
            hold_s=finite(item.get('hold_s',2.),.1,60,'Hover hold'),
            radius_m=finite(item.get('radius_m',1.),.1,10,'Waypoint radius'),
            # A leg's speed limit (fly-through: also its arrival speed along
            # the route); a landing step's is the touchdown speed.
            speed_m_s=finite(item.get('speed_m_s',.15 if kind=='land' else 3.),.1,
                             .5 if kind=='land' else speed_limit,
                             'Waypoint speed'),
            corridor_m=corridor)
        if kind == 'land':
            approach = item.get('approach_speed_m_s')
            waypoint.update(pad=pad, approach_speed_m_s=None if approach is None else
                            finite(approach, .3, speed_limit, 'Landing approach speed'))
        waypoints.append(waypoint)
    result['waypoints']=waypoints
    validate_spline_clearance(result['position'], [w for w in waypoints if w['type'] != 'land'])
    result['initial_motor_fraction'] = finite(result['initial_motor_fraction'], 0, 1, 'Initial motor fraction')
    b = copy.deepcopy(DEFAULTS['battery'])
    if not isinstance(result['battery'], dict) or set(result['battery']) - set(b):
        raise ValueError('Unknown battery settings')
    b.update(result['battery'])
    if not isinstance(b['enabled'], bool):
        raise ValueError('Battery enabled must be a boolean')
    for k, lo, hi in [('capacity_ah', .5, 20), ('c_rating', 1, 150), ('initial_soc', 0, 1),
                      ('cell_resistance_ohm', .0001, .05), ('max_current_a', 5, 120)]:
        b[k] = finite(b[k], lo, hi, k)
    result['battery'] = b
    if result['controller'] == 'convex' and waypoints and not b['enabled']:
        raise ValueError('Convex waypoint missions use the battery-coupled mission sequencer; enable the LiPo model')
    return result


# Vehicle mass and net thrust at full rotor speed (neutral vane drag
# included) as the Isaac plant computes them (nominal_hover_throttle), from
# the vehicle_model records of convex missions 835c3de32185 (8S) and
# fdb559daede1 (6S). Only the Isaac asset can recompute them; re-record
# after changing the asset, the EDF model or the vane model.
PLANT_THRUST = {('planned_8s', 'momentum'): (3.104, 43.07), ('legacy_6s', 'momentum'): (3.104, 35.16)}


def braking_envelopes():
    """What the launch form's braking estimate needs, per 'profile/vanes'.

    The rotor spool is the EDF's first-order lag with the motor torque bound
    of the mission plant (vane_model_overrides); the rotor's aerodynamic drag,
    P_shaft / omega_max at full speed, then limits the spool. The tilt cone
    is the convex planner's.
    """
    edf = yaml.safe_load((ROOT / 'configs/params/edf_90mm.yaml').read_text(encoding='utf-8'))['edf']
    tilt = yaml.safe_load((ROOT / 'configs/controllers/convex_guidance.yaml').read_text(encoding='utf-8'))['guidance']['max_tilt_deg']
    source = yaml.safe_load((ROOT / 'configs/env/mission_plant.yaml').read_text(encoding='utf-8'))
    limit = source.get('dynamics', {}).get('motor_torque_limit') or {}
    battery = yaml.safe_load((ROOT / 'configs/params/battery_6s.yaml').read_text(encoding='utf-8'))['battery']
    result = {}
    for (profile, vanes), (mass, thrust) in PLANT_THRUST.items():
        overrides = hardware_overrides(dict(hardware_profile=profile))
        omega_max = float(overrides.get('edf', {}).get('omega_max', edf['omega_max']))
        shaft = float(overrides.get('battery', {}).get('shaft_power_at_max_w', battery['shaft_power_at_max_w']))
        limited = limit.get('enabled', False)
        result[f'{profile}/{vanes}'] = dict(
            mass_kg=mass, full_thrust_n=thrust, rotor_inertia=float(edf['rotor_inertia']), omega_max=omega_max,
            motor_time_constant_s=float(edf['tau_motor']),
            motor_torque_limit_nm=float(limit['max_torque_nm']) if limited else None,
            aero_torque_at_max_nm=shaft / omega_max, max_tilt_deg=float(tilt))
    return result


def convex_available():
    """Convex guidance needs the Clarabel conic solver in the Isaac Python."""
    import importlib.util
    return importlib.util.find_spec('clarabel') is not None


def battery_config(mission):
    c = yaml.safe_load((ROOT / 'configs/params/battery_6s.yaml').read_text())['battery']
    if mission['hardware_profile'] == 'planned_8s':
        c.update(hardware_overrides(mission)['battery'])
    c.update(mission['battery'])
    c['max_current_a'] = min(c['max_current_a'], c['capacity_ah'] * c['c_rating'], 120.)
    return c


def hardware_overrides(mission):
    if mission['hardware_profile'] == 'legacy_6s':
        return {}
    return yaml.safe_load((ROOT / 'configs/hardware/planned_8s.yaml').read_text())


def vane_model_overrides(mission):
    """Physics and dynamics sections for a convex mission.

    They come from configs/env/mission_plant.yaml: coupled-jet vanes, no
    artificial damper, the torque-limited motor, and the coupled Cayley gyro
    integration with PhysX external forces applied once per step (which that
    integration needs).
    """
    source = yaml.safe_load((ROOT / 'configs/env/mission_plant.yaml').read_text(encoding='utf-8'))
    return {key: copy.deepcopy(source[key]) for key in ('physics', 'dynamics')}


def disturbance_config(mission):
    """Compose sources, then explicitly enable each selected disturbance.

    wind.yaml also contains disabled COM/noise defaults. Those defaults must
    never turn off another selected disturbance because of merge ordering.
    """
    from tvc_env.envs.task_registry import deep_merge
    selected = mission['disturbance']
    if isinstance(selected, str):
        selected = [] if selected == 'nominal' else [selected]
    folder = ROOT / 'configs/disturbances'
    result = yaml.safe_load((folder / 'nominal.yaml').read_text())
    # Body drag is the vehicle's own geometry (configs/vehicle), not a disturbance.
    sections = {'wind': ('wind', 'gust'),
                'sensor_noise': ('sensor_noise',), 'com_shift': ('com_offset',)}
    for name in sorted(selected):
        source = yaml.safe_load((folder / f'{name}.yaml').read_text())['disturbances']
        # Wind's unrelated COM defaults must not reduce a selected 10 mm
        # COM disturbance to 5 mm just because wind is merged last.
        result = deep_merge(result, {'disturbances': {key: source[key] for key in sections[name]}})
    from .disturbance_parameters import validate_settings as validate_disturbances
    result = deep_merge(result, {'disturbances': validate_disturbances(mission.get('disturbance_settings', {}))})
    settings = result['disturbances']
    settings['enabled'] = bool(selected)
    for name in ('wind', 'sensor_noise', 'com_offset'):
        settings.setdefault(name, {})['enabled'] = ('com_shift' if name == 'com_offset' else name) in selected
    settings['gust']['enabled'] = 'wind' in selected
    # A named IMU profile swaps the white attitude/rate noise for the physical chain in
    # tvc_env/dynamics/imu_model.py; the request-only key never reaches the environment config.
    profile = settings['sensor_noise'].pop('imu_profile', '')
    navigation = settings['sensor_noise'].pop('imu_nav', 'external')
    if profile and 'sensor_noise' in selected:
        settings['sensor_noise']['imu'] = {'enabled': True, 'profile': profile}
        if navigation == 'inertial':
            settings['sensor_noise']['imu']['nav'] = {'enabled': True}
        elif navigation in ('fused', 'fused_marker'):
            settings['sensor_noise']['imu']['fusion'] = {'enabled': True, 'profile': 'tfmini_mtf01p'}
            if navigation == 'fused_marker':
                settings['sensor_noise']['imu']['fusion']['marker'] = {'enabled': True}
    return result


def default_mission():
    return copy.deepcopy(DEFAULTS)
