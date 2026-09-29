from fastapi.testclient import TestClient
from mission_control import server
from mission_control.models import validate_mission


def test_plan_validation_and_optimizer_profile_api(tmp_path, monkeypatch):
    from mission_control import convex_parameters
    monkeypatch.setattr(convex_parameters, 'PROFILES', tmp_path)
    client = TestClient(server.app)
    headers = {'X-Mission-Control': 'local'}
    request = dict(controller='convex', duration_s=600, fast_live=True,
                   pads=[dict(name='Remote pad', position=[8,3,0])],
                   waypoints=[dict(type='land', pad=0, corridor_m=2, approach_speed_m_s=1)],
                   convex_settings={'guidance': {'route_corridor_m': 2}})
    response = client.post('/api/flight-plan/validate', json=request, headers=headers)
    assert response.status_code == 200
    assert response.json()['waypoints'][0]['position'] == [8,3,0]
    assert client.post('/api/flight-plan/validate', json={'duration_s':601}, headers=headers).status_code == 422
    response = client.post('/api/convex-profiles', json={'name':'Test', 'settings':request['convex_settings']}, headers=headers)
    assert response.status_code == 200
    assert client.get('/api/convex-profiles').json()[0]['settings'] == request['convex_settings']
    assert client.post('/api/convex-profiles', json={'name':'../bad'}, headers=headers).status_code == 422
    assert client.post('/api/convex-profiles', json={'name':'Test'}).status_code == 403


def test_saved_flight_plans_are_route_only_and_independent_of_guidance(tmp_path, monkeypatch):
    from mission_control import flight_plans
    monkeypatch.setattr(flight_plans, 'LIBRARY', tmp_path)
    client = TestClient(server.app)
    request = dict(name='East pad plan', controller='convex', duration_s=600,
                   pads=[dict(name='East', position=[4,0,0])],
                   waypoints=[dict(type='hover', position=[2,0,3], speed_m_s=6), dict(type='land', pad=0, corridor_m=1)],
                   convex_settings={'guidance': {'route_corridor_mode':'strict', 'max_speed_m_s': 6}},
                   disturbance=['wind'], seed=7)
    headers = {'X-Mission-Control':'local'}
    response = client.post('/api/flight-plans', json=request, headers=headers)
    assert response.status_code == 200
    key = response.json()['id']
    assert client.get('/api/flight-plans').json()[0]['id'] == key
    saved = client.get(f'/api/flight-plans/{key}').json()
    assert saved['version'] == 2 and saved['scope'] == 'route'
    # Guidance, environment and run settings are chosen per run, never stored with the route.
    assert set(saved['mission']) == set(flight_plans.ROUTE_FIELDS)
    assert saved['mission']['waypoints'][1]['position'] == [4,0,0]
    # A 6 m/s step is a valid route; the chosen guidance's speed limit is checked at launch.
    assert saved['mission']['waypoints'][0]['speed_m_s'] == 6
    with __import__('pytest').raises(ValueError, match='speed'):
        validate_mission(dict(saved['mission'], convex_settings={}))
    assert client.post('/api/flight-plans', json={**request,'duration_s':120}, headers=headers).json()['id'] == key
    assert client.get(f'/api/flight-plans/{key}').json()['mission']['duration_s'] == 120
    assert client.get('/api/flight-plans/bad').status_code == 404
    assert client.post('/api/flight-plans', json={'duration_s':601}, headers=headers).status_code == 422
    assert client.post('/api/flight-plans', json=request).status_code == 403
    route = client.post('/api/flight-plan/route', json=request, headers=headers).json()
    assert route == saved['mission'] | {'duration_s': 600}


def test_plans_saved_before_the_route_split_load_without_their_guidance(tmp_path, monkeypatch):
    import json
    from mission_control import flight_plans
    monkeypatch.setattr(flight_plans, 'LIBRARY', tmp_path)
    legacy = validate_mission(dict(name='Legacy plan', convex_settings={'guidance': {'max_tilt_deg': 10}},
                                   disturbance=['sensor_noise'], waypoints=[dict(type='hover', position=[0,0,4])]))
    (tmp_path / 'plan-legacy-plan-0123456789.json').write_text(json.dumps(dict(format='edf-flight-plan', version=2, mission=legacy)))
    plan = flight_plans.read_plan('plan-legacy-plan-0123456789')
    assert set(plan['mission']) == set(flight_plans.ROUTE_FIELDS)
    assert plan['mission']['waypoints'][0]['position'] == [0,0,4]


def test_api_requires_local_write_header_and_rejects_cross_origin():
    client = TestClient(server.app)
    assert client.post('/api/missions', json={}).status_code == 403
    assert client.post('/api/missions', json={}, headers={'X-Mission-Control':'local','Origin':'https://external.example'}).status_code == 403
    assert client.post('/api/missions', json={'seed':-1}, headers={'X-Mission-Control':'local'}).status_code == 422


def test_read_only_api_and_path_validation():
    client = TestClient(server.app)
    assert client.get('/api/config').json()['defaults']['hardware_profile'] == 'planned_8s'
    assert client.get('/api/missions/not-a-mission').status_code == 404
    assert client.get('/', headers={'Host':'external.example'}).status_code == 400


def test_editor_html_versions_assets_and_revalidates_cache():
    client = TestClient(server.app)
    response = client.get('/')
    assert response.headers['cache-control'] == 'no-cache'
    style_version = (server.HERE / 'static/style.css').stat().st_mtime_ns
    assert f'/static/style.css?v={style_version}' in response.text


def test_convex_guidance_is_offered_and_may_fly_routes(monkeypatch):
    import pytest
    from mission_control import models
    monkeypatch.setattr(server, 'convex_available', lambda: True)
    client = TestClient(server.app)
    assert 'convex' in client.get('/api/config').json()['policies']
    route = [{'type': 'hover', 'position': [4, 2, 5]}]
    assert models.validate_mission({'controller': 'convex', 'waypoints': route})['waypoints'][0]['hold_s'] == 2
    with pytest.raises(ValueError, match='enable the LiPo model'):
        models.validate_mission({'controller': 'convex', 'waypoints': route, 'battery': {'enabled': False}})
    # Landing-only convex missions do not need the battery model.
    assert models.validate_mission({'controller': 'convex', 'battery': {'enabled': False},
                                    'hardware_profile': 'legacy_6s'})['controller'] == 'convex'


def test_stream_reader_ignores_incomplete_final_line(tmp_path, monkeypatch):
    monkeypatch.setattr(server, 'RUNS', tmp_path)
    p = tmp_path / '123456abcdef'
    p.mkdir()
    (p / 'frames.jsonl').write_text('{"t":0}\n{"t":1}\n{"t":')
    client = TestClient(server.app)
    data = client.get('/api/missions/123456abcdef/frames?after=1').json()
    assert data == {'frames':[{'t':1}], 'next':2}


def test_combinations_preserve_each_disturbance_and_legacy_requests():
    from mission_control.models import validate_mission, disturbance_config
    import itertools
    names = ['wind', 'sensor_noise', 'com_shift']
    for n in range(4):
        for selected in itertools.combinations(names, n):
            settings = disturbance_config(validate_mission({'disturbance':list(selected)}))['disturbances']
            assert settings['enabled'] == bool(selected)
            assert settings['wind']['enabled'] == ('wind' in selected)
            assert settings['gust']['enabled'] == ('wind' in selected)
            assert settings['sensor_noise']['enabled'] == ('sensor_noise' in selected)
            assert settings['com_offset']['enabled'] == ('com_shift' in selected)
            for name in selected:
                standalone = disturbance_config(validate_mission({'disturbance':[name]}))['disturbances']
                key = 'com_offset' if name == 'com_shift' else name
                assert settings[key] == standalone[key]
    assert validate_mission({'disturbance':'nominal'})['disturbance'] == []


def test_partial_replay_restores_recorded_settings_without_new_defaults(tmp_path):
    import json
    (tmp_path / 'metadata.json').write_text(json.dumps({'request': {
        'name': 'original', 'position': [1, 2, 8],
        'battery': {'enabled': True, 'capacity_ah': 7}}}))
    (tmp_path / 'request.json').write_text(json.dumps({'name': 'renamed'}))
    assert server.recorded_request(tmp_path) == {
        'name': 'renamed', 'position': [1, 2, 8],
        'battery': {'enabled': True, 'capacity_ah': 7}}


def test_adverse_mission_bounds_and_timed_hover_defaults(monkeypatch):
    import pytest
    from mission_control import models
    spec = dict(controller='convex', position=[-100, 100, 100],
                attitude_deg=[180, 0, 0], angular_rate_deg_s=[360, -360, 720],
                waypoints=[{'type': 'hover', 'position': [10, -10, 5]}])
    request = models.validate_mission(spec)
    assert request['waypoints'][0]['hold_s'] == 2
    for change in ({'position': [0, 0, -1]}, {'position': [101, 0, 10]},
                   {'controller': 'unknown'}, {'battery': {'enabled': False}}):
        with pytest.raises(ValueError):
            models.validate_mission({**spec, **change})


def test_spline_rejects_underground_overshoot_between_positive_waypoints():
    import pytest
    from mission_control.models import validate_spline_clearance
    with pytest.raises(ValueError, match='ground clearance'):
        validate_spline_clearance([0,0,100], [dict(position=[i,0,z]) for i,z in enumerate([1,1,100])])
    validate_spline_clearance([0,0,100], [dict(position=[i,0,z]) for i,z in enumerate([75,50,25])])


def test_mission_status_replace_retries_while_a_reader_holds_the_file(tmp_path, monkeypatch):
    # Windows refuses os.replace while another process (the service polling
    # status.json) has the target open; that aborted mission aea16ce345d4.
    import json
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(server.__file__).resolve().parents[1] / 'apps'))
    import run_mission
    real, calls = Path.replace, []

    def held_twice(self, target):
        calls.append(target)
        if len(calls) < 3:
            raise PermissionError(5, 'Access is denied')
        return real(self, target)

    monkeypatch.setattr(Path, 'replace', held_twice)
    monkeypatch.setattr(run_mission.time, 'sleep', lambda s: None)
    run_mission.atomic_json(tmp_path / 'status.json', {'state': 'running'})
    assert json.loads((tmp_path / 'status.json').read_text()) == {'state': 'running'} and len(calls) == 3


def test_missions_fly_the_mission_plant_without_pid_or_legacy_vanes(tmp_path, monkeypatch):
    import json
    import pytest
    from mission_control import models
    monkeypatch.setattr(server, 'convex_available', lambda: True)
    assert models.validate_mission({})['controller'] == 'convex'
    assert models.validate_mission({'controller': 'convex'})['vane_model'] == 'momentum'
    # The PID baseline and the legacy vanes were removed on 2026-09-26.
    with pytest.raises(ValueError, match='PID baseline was removed'):
        models.validate_mission({'controller': 'pid'})
    with pytest.raises(ValueError, match='legacy vane physics was removed'):
        models.validate_mission({'controller': 'convex', 'vane_model': 'legacy'})
    with pytest.raises(ValueError, match='Unknown vane model'):
        models.validate_mission({'controller': 'convex', 'vane_model': 'cfd'})
    assert 'pid' not in TestClient(server.app).get('/api/config').json()['policies']
    # The mission plant: configs/env/mission_plant.yaml.
    plant = models.vane_model_overrides({'vane_model': 'momentum'})
    assert plant['dynamics']['coupled_jet']['enabled'] and plant['dynamics']['body_angular_damping'] == 0.
    assert plant['dynamics']['gyro_integration'] == 'coupled_cayley'
    assert plant['dynamics']['motor_torque_limit'] == dict(enabled=True, max_torque_nm=.76, zero_throttle_brake=False)
    assert plant['physics']['enable_external_forces_every_iteration'] is False
    assert sorted(models.braking_envelopes()) == ['legacy_6s/momentum', 'planned_8s/momentum']
    # Recorded missions (legacy vanes and PID included) still replay and
    # report the plant they actually flew.
    for dynamics, expected in (({}, 'legacy'), (plant['dynamics'], 'momentum')):
        (tmp_path / 'metadata.json').write_text(json.dumps({'request': {'name': 'old', 'controller': 'pid'},
                                                            'dynamics': dynamics}))
        (tmp_path / 'request.json').write_text(json.dumps({'name': 'old'}))
        assert server.recorded_request(tmp_path)['vane_model'] == expected


def test_cpu_physics_is_an_explicit_opt_in():
    import pytest
    assert validate_mission({})['cpu_physics'] is False
    assert validate_mission({'cpu_physics': True})['cpu_physics'] is True
    with pytest.raises(ValueError, match='CPU physics'):
        validate_mission({'cpu_physics': 'yes'})
