"""Built-in convex profiles, environment presets and sample flight plans."""
import json

from fastapi.testclient import TestClient

from mission_control import server
from mission_control.convex_parameters import resolve
from mission_control.models import validate_mission
from mission_control.presets import FOLDER, convex_presets, disturbance_presets, list_samples, read_sample


def test_convex_presets_cover_every_mission_type_and_corridor_mode():
    presets = convex_presets()
    assert len({p['id'] for p in presets}) == len(presets) == len({p['name'] for p in presets})
    assert {p['mission'] for p in presets} == {'hop', 'hover', 'land'}
    for mission in ('hop', 'hover', 'land'):
        assert {p['corridor'] for p in presets if p['mission'] == mission} == {'soft', 'strict'}
    assert len({resolve(p['settings'])['guidance']['max_tilt_deg'] for p in presets}) >= 4


def test_convex_presets_keep_identified_gains_and_feedback_margin():
    # The tracking gains and the tilt-rate share come from the damping analysis
    # and Isaac sweeps documented in convex_guidance.yaml; profiles may not retune them.
    identified = {'kp_xy', 'kd_xy', 'ki_xy', 'kp_z', 'kd_z', 'ki_z'}
    for preset in convex_presets():
        assert not identified & set(preset['settings'].get('tracking', {})), preset['id']
        assert 'tilt_rate_authority_fraction' not in preset['settings'].get('guidance', {})
        resolved = resolve(preset['settings'])
        assert resolved['tracking']['max_tilt_deg'] - resolved['guidance']['max_tilt_deg'] >= 7, preset['id']
        assert preset['summary'] and preset['rationale']


def test_disturbance_presets_validate_and_start_calm():
    presets = disturbance_presets()
    assert presets[0]['id'] == 'calm' and presets[0]['selected'] == []
    for preset in presets:
        mission = validate_mission(dict(disturbance=preset['selected'], disturbance_settings=preset['settings']))
        assert mission['disturbance'] == preset['selected']


def test_sample_plans_are_routes_with_suggested_guidance_and_environment():
    from mission_control.flight_plans import ROUTE_FIELDS
    samples = list_samples()
    assert len(samples) >= 10
    assert {s['category'] for s in samples} == {'hop', 'land', 'hover'}
    profiles = {p['id']: p for p in convex_presets()}
    for summary in samples:
        plan = read_sample(summary['id'])
        route = plan['mission']
        assert plan['scope'] == 'route' and set(route) == set(ROUTE_FIELDS)
        assert summary['profile_name'] == profiles[summary['profile']]['name']
        # Each route launches with repository-default guidance and with its suggested profile.
        validate_mission(dict(route, controller='convex'))
        mission = validate_mission(dict(route, controller='convex', convex_settings=profiles[summary['profile']]['settings']))
        if 'environment' in plan['sample']:
            environment = plan['sample']['environment']
            validate_mission(dict(mission, disturbance=environment['selected'], disturbance_settings=environment['settings']))
        if summary['category'] in ('hop', 'hover'):
            assert route['position'][2] < .5 and route['waypoints'][0]['type'] == 'takeoff'
        assert summary['route'][0] == route['position'] and summary['route'][-1][2] == 0
    hops = [s['max_altitude_m'] for s in samples if s['category'] == 'hop']
    assert min(hops) <= 3 and max(hops) >= 20


def test_sample_files_are_canonical_route_only_plans():
    from mission_control.flight_plans import validate_route
    for path in sorted((FOLDER / 'flight_plans').glob('*.json')):
        record = json.loads(path.read_text(encoding='utf-8'))
        assert record['format'] == 'edf-flight-plan' and record['version'] == 2 and record['scope'] == 'route'
        # Stored exactly as validated, so an export of a loaded sample is identical.
        assert validate_route(record['mission']) == record['mission']


def test_preset_endpoints_and_sample_path_validation():
    client = TestClient(server.app)
    assert len(client.get('/api/convex-presets').json()) == len(convex_presets())
    assert client.get('/api/disturbance-presets').json()[0]['id'] == 'calm'
    samples = client.get('/api/flight-plan-samples').json()
    plan = client.get(f"/api/flight-plan-samples/{samples[0]['id']}").json()
    assert plan['format'] == 'edf-flight-plan' and plan['mission']['name'] == samples[0]['name']
    for key in ('..%2Fconvex_profiles', 'missing-sample', '99-nope'):
        assert client.get(f'/api/flight-plan-samples/{key}').status_code == 404


def test_imu_hardware_presets_follow_their_documented_datasheet_mapping():
    import math
    imus = {p['id']: p for p in disturbance_presets() if p.get('group') == 'imu'}
    assert {'imu-bno085', 'imu-wtgahrs1', 'imu-vn110e'} <= set(imus)
    for preset in imus.values():
        noise = preset['settings']['sensor_noise']
        # IMU presets model attitude and rate only; position/velocity stay at the repository defaults.
        assert (noise['position_std'], noise['velocity_std']) == (0.01, 0.05)
        assert preset['selected'] == ['sensor_noise'] and preset['hardware']['source']
    sheet = lambda key: imus[key]['hardware']['datasheet']
    noise = lambda key: imus[key]['settings']['sensor_noise']
    close = lambda a, b: math.isclose(a, b, rel_tol=2e-3)
    # attitude_std: published pitch/roll error (dynamic when published) in rad.
    assert close(noise('imu-bno085')['attitude_std'], math.radians(sheet('imu-bno085')['game_rotation_vector_dynamic_error_deg']))
    assert close(noise('imu-vn110e')['attitude_std'], math.radians(sheet('imu-vn110e')['pitch_roll_dynamic_deg_rms']))
    assert close(noise('imu-wtgahrs1')['attitude_std'], math.radians(sheet('imu-wtgahrs1')['angle_accuracy_xy_deg']))
    # angular_velocity_std: noise density x sqrt(15 Hz), else the published rate accuracy.
    density = sheet('imu-vn110e')['gyro_noise_density_deg_hr_rthz'] / 3600
    assert close(noise('imu-vn110e')['angular_velocity_std'], math.radians(density * math.sqrt(15)))
    assert close(noise('imu-bno085')['angular_velocity_std'], math.radians(sheet('imu-bno085')['gyroscope_accuracy_deg_s']))
    assert close(noise('imu-wtgahrs1')['angular_velocity_std'], math.radians(sheet('imu-wtgahrs1')['gyro_stability_deg_s']))


def test_imu_presets_reach_the_simulator_noise_model():
    from mission_control.models import disturbance_config
    preset = next(p for p in disturbance_presets() if p['id'] == 'imu-bno085')
    mission = validate_mission(dict(disturbance=['sensor_noise', 'wind'], disturbance_settings=preset['settings']))
    resolved = disturbance_config(mission)['disturbances']['sensor_noise']
    assert resolved['enabled'] and resolved['attitude_std'] == preset['settings']['sensor_noise']['attitude_std']
