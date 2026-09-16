"""Validated mission inputs and fixed local policy choices."""
from pathlib import Path
import copy
import math
import json
import yaml

ROOT = Path(__file__).resolve().parents[1]
POLICIES = {
    'ppo': 'runs/ppo_exploration_anneal/ppo_landing_seed0_20260911_184222/ppo_step_26034176.pt',
    'ppo_deterministic': 'runs/ppo_kl_consolidation/ppo_landing_seed0_20260913_205810/ppo_best.pt',
}
DEFAULTS = dict(name='Landing test', controller='ppo', seed=2026, duration_s=30.,
                hardware_profile='planned_8s',
                position=[-.28, .82, 18.], velocity=[0., 0., -1.],
                attitude_deg=[0., 0., 0.], angular_rate_deg_s=[0., 0., 0.],
                initial_motor_fraction=0., disturbance=[],
                waypoints=[],
                battery=dict(enabled=True, capacity_ah=5., c_rating=45., initial_soc=1.,
                             cell_resistance_ohm=.003, max_current_a=120.))


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


def validate_mission(value):
    if not isinstance(value, dict) or set(value) - set(DEFAULTS):
        raise ValueError('Unknown mission fields')
    result = copy.deepcopy(DEFAULTS)
    result.update(value)
    if not isinstance(result['name'], str) or not 1 <= len(result['name'].strip()) <= 80:
        raise ValueError('Mission name must contain 1–80 characters')
    result['name'] = result['name'].strip()
    if result['controller'] not in (*policy_paths(), 'pid'):
        raise ValueError('Unknown controller')
    selected = result['disturbance']
    if isinstance(selected, str):  # Existing recorded requests remain replayable.
        selected = [] if selected == 'nominal' else [selected]
    if not isinstance(selected, list) or any(x not in ('wind', 'sensor_noise', 'com_shift') for x in selected):
        raise ValueError('Unknown disturbance')
    result['disturbance'] = sorted(set(selected))
    if result['hardware_profile'] not in ('planned_8s', 'legacy_6s'):
        raise ValueError('Unknown hardware profile')
    seed = finite(result['seed'], 0, 2**31 - 1, 'Seed')
    if seed != int(seed):
        raise ValueError('Seed must be an integer')
    result['seed'] = int(seed)
    result['duration_s'] = finite(result['duration_s'], 1, 180, 'Duration')
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
        if not isinstance(item,dict) or set(item)-{'position','type','hold_s','radius_m','speed_m_s'}:
            raise ValueError(f'Waypoint {i+1}: unknown fields')
        position=item.get('position')
        if not isinstance(position,list) or len(position)!=3:
            raise ValueError(f'Waypoint {i+1}: position requires three numbers')
        position=[finite(v,lo,hi,f'Waypoint {i+1} position') for v,(lo,hi) in zip(position,[(-100,100),(-100,100),(1,100)])]
        kind=item.get('type','flypass')
        if kind not in ('hover','flypass'):
            raise ValueError('Waypoint type must be hover or flypass')
        waypoints.append(dict(position=position,type=kind,
            hold_s=finite(item.get('hold_s',2.),.1,60,'Hover hold'),
            radius_m=finite(item.get('radius_m',1.),.1,10,'Waypoint radius'),
            speed_m_s=finite(item.get('speed_m_s',3.),.1,15,'Path reference speed')))
    result['waypoints']=waypoints
    validate_spline_clearance(result['position'], waypoints)
    if waypoints and result['controller'] != 'ppo_mission':
        raise ValueError('Waypoints require the experimental recovery + waypoints policy; legacy policies do not observe route targets')
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
    if result['controller'] in ('ppo_radial', 'ppo_mission') and (result['hardware_profile'] != 'planned_8s' or not b['enabled']):
        raise ValueError('The radial 8S policy requires the 8S hardware profile and coupled battery observations')
    return result


def battery_config(mission):
    c = yaml.safe_load((ROOT / 'configs/params/battery_6s.yaml').read_text())['battery']
    if mission['hardware_profile'] == 'planned_8s':
        c.update(hardware_overrides(mission)['battery'])
    c.update(mission['battery'])
    c['max_current_a'] = min(c['max_current_a'], c['capacity_ah'] * c['c_rating'], 120.)
    return c


def training_envelope_violations(mission, saved):
    """Compare against the checkpoint's current curriculum, not a fixed 18 m box.

    Being within these bounds is not proof the policy has mastered them.
    Explicit routes can differ from random training routes even with equal count.
    """
    task = saved.get('task_config', {}).get('task', {})
    spawn = dict(task.get('spawn', {}))
    curriculum = saved.get('curriculum') or {}
    stages = spawn.get('curriculum', {}).get('stages', [])
    stage = curriculum.get('stage_index')
    if stage is not None and 0 <= stage < len(stages):
        spawn.update(stages[stage])
    violations = []
    for field, key, scale in (('position', 'position_range', 1.),
            ('velocity', 'velocity_range', 1.), ('attitude_deg', 'attitude_range', math.pi/180),
            ('angular_rate_deg_s', 'angular_velocity_range', math.pi/180)):
        limits = spawn.get(key)
        if limits and any(not low-1e-5 <= value*scale <= high+1e-5
                          for value, low, high in zip(mission[field], *limits)):
            violations.append(field)
    rpm = spawn.get('initial_motor_omega_fraction')
    if rpm is not None and abs(mission['initial_motor_fraction']-rpm)>1e-5:
        violations.append('initial_motor_fraction')
    count = spawn.get('waypoint_count_range', task.get('navigation', {}).get('count_range', [0,0]))
    if not count[0] <= len(mission.get('waypoints', [])) <= count[1]:
        violations.append('waypoint_count')
    soc = saved.get('task_config', {}).get('battery', {}).get('initial_soc')
    if soc is not None and abs(mission['battery']['initial_soc']-soc)>1e-5:
        violations.append('initial_soc')
    return violations


def hardware_overrides(mission):
    if mission['hardware_profile'] == 'legacy_6s':
        return {}
    return yaml.safe_load((ROOT / 'configs/hardware/planned_8s.yaml').read_text())


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
    sections = {'wind': ('wind', 'gust', 'body_drag'),
                'sensor_noise': ('sensor_noise',), 'com_shift': ('com_offset',)}
    for name in sorted(selected):
        source = yaml.safe_load((folder / f'{name}.yaml').read_text())['disturbances']
        # Wind's unrelated COM defaults must not reduce a selected 10 mm
        # COM disturbance to 5 mm just because wind is merged last.
        result = deep_merge(result, {'disturbances': {key: source[key] for key in sections[name]}})
    settings = result['disturbances']
    settings['enabled'] = bool(selected)
    for name in ('wind', 'sensor_noise', 'com_offset'):
        settings.setdefault(name, {})['enabled'] = ('com_shift' if name == 'com_offset' else name) in selected
    settings['gust']['enabled'] = 'wind' in selected
    return result


def policy_paths():
    result = dict(POLICIES)
    registry = ROOT / 'mission_control/policy_registry.json'
    if registry.exists():
        record = json.loads(registry.read_text())
        path = (ROOT / record['checkpoint']).resolve()
        if path.is_relative_to((ROOT / 'runs').resolve()) and path.suffix == '.pt' and path.is_file():
            result['ppo_radial'] = str(path.relative_to(ROOT))
    mission_registry = ROOT / 'mission_control/mission_policy_registry.json'
    if mission_registry.exists():
        record = json.loads(mission_registry.read_text())
        path = (ROOT / record['checkpoint']).resolve()
        # Explicitly experimental, opt-in tracking of completed atomic saves.
        # Each mission still records the exact file and SHA256 it loaded.
        if record.get('follow_training_run') and record.get('training_run'):
            run = (ROOT / record['training_run']).resolve()
            if run.is_relative_to((ROOT/'runs').resolve()) and run.is_dir():
                candidates = [p for p in run.glob('ppo_step_*.pt')
                              if p.stem.removeprefix('ppo_step_').isdigit()]
                if (run/'ppo_final.pt').is_file():
                    candidates.append(run/'ppo_final.pt')
                if candidates:
                    path = max(candidates, key=lambda p: p.stat().st_mtime).resolve()
        if path.is_relative_to((ROOT/'runs').resolve()) and path.suffix=='.pt' and path.is_file():
            result['ppo_mission'] = str(path.relative_to(ROOT))
    return result


def default_mission():
    result = copy.deepcopy(DEFAULTS)
    if 'ppo_radial' in policy_paths():
        result['controller'] = 'ppo_radial'
    if 'ppo_mission' in policy_paths():
        result['controller'] = 'ppo_mission'
    return result
