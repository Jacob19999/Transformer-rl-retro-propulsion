"""Reject silent plant changes when evaluating a saved PPO controller."""
from copy import deepcopy
import hashlib
import math


def refine_physics_clock(config, physics_hz):
    """Refine only the numerical clock; retain policy and contact dwell times."""
    old_dt = config.physics_dt
    ratio = float(physics_hz) * old_dt
    if not math.isfinite(ratio):
        raise ValueError('Physics frequency must be finite')
    multiple = round(ratio)
    if multiple < 1 or not math.isclose(ratio, multiple):
        raise ValueError('Physics refinement must be an integer multiple of the training frequency')
    old_decimation = config.decimation
    dwell = int(config.config['task']['contact']['dwell_frames'])
    config.physics_dt = old_dt / multiple
    config.decimation = old_decimation * multiple
    config.config['env'].update(physics_dt=config.physics_dt, decimation=config.decimation)
    config.config['task']['contact']['dwell_frames'] = dwell * multiple
    return dict(training_hz=1/old_dt, evaluation_hz=1/config.physics_dt,
                policy_period_s=old_dt*old_decimation, contact_dwell_s=dwell*old_dt)


def physical_contract(config):
    result = {key: deepcopy(config.get(key, {})) for key in ('edf','servo','dynamics','battery')}
    # SOC is an explicitly identified sensitivity option, not a new pack.
    result['battery'].pop('initial_soc', None)
    result['env'] = {key: config.get('env', {}).get(key) for key in ('physics_dt','decimation','observe_battery','dispatch_mode')}
    result['physics'] = {key: config.get('physics', {}).get(key) for key in (
        'enable_gyroscopic_forces','max_angular_velocity_deg_s','contact_offset','rest_offset',
        'num_position_iterations','num_velocity_iterations','solver_type','enable_external_forces_every_iteration')}
    result['contact'] = deepcopy(config.get('task', {}).get('contact', {}))
    return result


def plant_mismatches(saved, config, root):
    reasons = []
    if physical_contract(saved.get('task_config', {})) != physical_contract(config):
        reasons.append('resolved physical configuration')
    manifest = saved.get('source_manifest', {})
    paths = [*root.glob('tvc_env/dynamics/*.py'), *root.glob('tvc_env/sim/*.py'),
             root/'tvc_env/envs/base_env.py', root/'tvc_env/envs/direct_rl_env.py',
             *root.glob('configs/vehicle/*.yaml'), root/'configs/params/edf_90mm.yaml',
             root/'configs/params/servo_mg996r.yaml']
    for path in paths:
        key = str(path.relative_to(root))
        # Windows manifests use native separators. JSON records preserve them.
        if manifest.get(key) != hashlib.sha256(path.read_bytes()).hexdigest():
            reasons.append(key)
    return reasons
