from fastapi.testclient import TestClient
from mission_control import server


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


def test_saved_flight_plan_library_preserves_full_mission(tmp_path, monkeypatch):
    from mission_control import flight_plans
    monkeypatch.setattr(flight_plans, 'LIBRARY', tmp_path)
    client = TestClient(server.app)
    request = dict(name='East pad plan', controller='convex', duration_s=600,
                   pads=[dict(name='East', position=[4,0,0])],
                   waypoints=[dict(type='land', pad=0, corridor_m=1)],
                   convex_settings={'guidance': {'route_corridor_mode':'strict'}})
    headers = {'X-Mission-Control':'local'}
    response = client.post('/api/flight-plans', json=request, headers=headers)
    assert response.status_code == 200
    key = response.json()['id']
    assert client.get('/api/flight-plans').json()[0]['id'] == key
    saved = client.get(f'/api/flight-plans/{key}').json()
    assert saved['version'] == 2
    assert saved['mission']['convex_settings'] == request['convex_settings']
    assert saved['mission']['waypoints'][0]['position'] == [4,0,0]
    assert client.post('/api/flight-plans', json={**request,'duration_s':120}, headers=headers).json()['id'] == key
    assert client.get(f'/api/flight-plans/{key}').json()['mission']['duration_s'] == 120
    assert client.get('/api/flight-plans/bad').status_code == 404
    assert client.post('/api/flight-plans', json={'duration_s':601}, headers=headers).status_code == 422
    assert client.post('/api/flight-plans', json=request).status_code == 403


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
    monkeypatch.setattr(models, 'policy_paths', lambda: {})
    monkeypatch.setattr(server, 'policy_paths', lambda: {})
    monkeypatch.setattr(server, 'convex_available', lambda: True)
    client = TestClient(server.app)
    assert 'convex' in client.get('/api/config').json()['policies']
    assert any(p['key'] == 'convex' for p in client.get('/api/models').json()['flyable'])
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


def test_training_progress_ignores_partial_log_records_and_handles_nan(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(server, 'ROOT', tmp_path)
    run = tmp_path / 'runs' / 'experiment' / 'run'
    run.mkdir(parents=True)
    (run / 'args.json').write_text('{}')
    (run / 'train_log.jsonl').write_text(json.dumps({'global_step':1000,'spawn_stage_index':1,
                                                  'spawn_num_stages':4,'stage_success_fraction':.8})+'\n{"global_step":')
    (run / 'eval_log.jsonl').write_text(json.dumps({'success_fraction':0.,'success_mean_energy_wh':float('nan')})+'\n')
    result = server.training_snapshot(['python', 'run_train_ppo.py', '--output-dir', 'runs/experiment'])
    assert result['step'] == 1000 and result['stage_success'] == .8
    assert result['full_success'] == 0 and result['success_energy_wh'] is None
    assert server.training_snapshot(None) is None


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
    monkeypatch.setattr(models, 'policy_paths', lambda: {'ppo': 'legacy.pt', 'ppo_mission': 'new.pt'})
    spec = dict(controller='ppo_mission', position=[-100, 100, 100],
                attitude_deg=[180, 0, 0], angular_rate_deg_s=[360, -360, 720],
                waypoints=[{'type': 'hover', 'position': [10, -10, 5]}])
    request = models.validate_mission(spec)
    assert request['waypoints'][0]['hold_s'] == 2
    for change in ({'position': [0, 0, -1]}, {'position': [101, 0, 10]},
                   {'controller': 'ppo'}, {'battery': {'enabled': False}}):
        with pytest.raises(ValueError):
            models.validate_mission({**spec, **change})


def test_mission_envelope_uses_current_curriculum_not_only_final_bounds():
    from mission_control.models import validate_mission, training_envelope_violations
    request = validate_mission({'position': [0,0,8], 'initial_motor_fraction': .84})
    saved = {'curriculum': {'stage_index': 0}, 'task_config': {'task': {'spawn': {
        'position_range': [[-100,-100,25], [100,100,100]],
        'curriculum': {'stages': [{'position_range': [[-1,-1,6],[1,1,10]],
                                  'initial_motor_omega_fraction': .84}]}}}}}
    assert training_envelope_violations(request, saved) == []
    request['position'] = [0,0,90]
    assert training_envelope_violations(request, saved) == ['position']


def test_waypoint_flight_envelope_resolves_hover_rotor_and_counts_the_landing():
    from mission_control.models import validate_mission, training_envelope_violations
    saved = {'observation_contract': 'waypoint_flight_v1', 'reward_budget': {'hover_fraction': .84},
             'task_config': {'task': {
                 'spawn': {'position_range': [[-40,-40,3],[40,40,25]], 'initial_motor_omega_fraction': 'hover',
                           'initial_motor_omega_jitter': .05, 'initial_soc_range': [.5, 1.]},
                 'waypoint_flight': {'generator': {'count_range': [1, 2]}}}}}
    request = validate_mission({'position': [0,0,18], 'initial_motor_fraction': .86})
    assert training_envelope_violations(request, saved) == []
    request.update(initial_motor_fraction=0., waypoints=[{'position': [1,0,5]}] * 2)
    request['battery']['initial_soc'] = .2
    assert training_envelope_violations(request, saved) == ['initial_motor_fraction', 'waypoint_count', 'initial_soc']


def test_experimental_policy_follows_completed_saves_only(tmp_path, monkeypatch):
    import json
    import os
    from mission_control import models
    monkeypatch.setattr(models, 'ROOT', tmp_path)
    run = tmp_path/'runs'/'experiment'
    run.mkdir(parents=True)
    (tmp_path/'mission_control').mkdir()
    for step in (100,200):
        p = run/f'ppo_step_{step}.pt'
        p.write_bytes(b'checkpoint')
        os.utime(p, (step, step))
    (run/'ppo_step_300.pt.tmp').write_bytes(b'incomplete')
    registry = tmp_path/'mission_control'/'mission_policy_registry.json'
    registry.write_text(json.dumps({'checkpoint':'runs/experiment/ppo_step_100.pt',
        'training_run':'runs/experiment','follow_training_run':True}))
    assert models.policy_paths()['ppo_mission'] == str((run/'ppo_step_200.pt').relative_to(tmp_path))


def test_spline_rejects_underground_overshoot_between_positive_waypoints():
    import pytest
    from mission_control.models import validate_spline_clearance
    with pytest.raises(ValueError, match='ground clearance'):
        validate_spline_clearance([0,0,100], [dict(position=[i,0,z]) for i,z in enumerate([1,1,100])])
    validate_spline_clearance([0,0,100], [dict(position=[i,0,z]) for i,z in enumerate([75,50,25])])


def _waypoint_run(root, name='ppo_waypoint_flight_seed0_20260924_022647'):
    import json
    run = root / 'runs' / 'waypoint_flight' / name
    run.mkdir(parents=True)
    (run / 'args.json').write_text('{"num_envs": 8192}')
    (run / 'task_config.json').write_text(json.dumps({'task': {'waypoint_flight': {'curriculum': {
        'stages': [{'name': 'hold_position'}, {'name': 'hold_fine'}, {'name': 'full_task'}]}}}}))
    records = [dict(type='train_update', global_step=step, stage_index=1, stage_name='hold_fine',
                    stage_success_fraction=.25, rollout_success_fraction=.2, mean_peak_yaw_deg_s=250.,
                    explained_variance=.9, throttle_mean=.8, reward_mean=-.5, sps=15000.,
                    outcomes={'SUCCESS': 1, 'SPIN': 3, 'CRASH': 0}) for step in range(1000, 31000, 1000)]
    (run / 'train_log.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records) + '{"global_step":')
    (run / 'eval_latest.json').write_text('{"success_fraction": 0.0, "success_mean_energy_wh": NaN}')
    (run / 'ppo_step_20000.pt').write_bytes(b'x')
    return run


def test_waypoint_trainer_is_detected_and_summarised(tmp_path, monkeypatch):
    monkeypatch.setattr(server, 'ROOT', tmp_path)
    _waypoint_run(tmp_path)
    result = server.training_snapshot(['python', 'apps/run_train_waypoints.py', '--num-envs', '8192'])
    assert result['task'] == 'waypoint_flight' and result['step'] == 30000
    assert result['stage'] == 1 and result['stages'] == 3 and result['stage_name'] == 'hold_fine'
    assert result['full_success'] is None


def test_models_api_lists_runs_and_downsampled_history(tmp_path, monkeypatch):
    monkeypatch.setattr(server, 'MODEL_RUNS', tmp_path / 'runs' / 'waypoint_flight')
    monkeypatch.setattr(server, 'active_training_command', lambda: None)
    run = _waypoint_run(tmp_path)
    client = TestClient(server.app)
    data = client.get('/api/models').json()
    summary = data['runs'][0]
    assert summary['run'] == run.name and summary['stages'] == ['hold_position', 'hold_fine', 'full_task']
    assert summary['outcomes'] == {'SUCCESS': .25, 'SPIN': .75, 'CRASH': 0.}
    assert summary['evaluation']['success_mean_energy_wh'] is None
    assert [c['step'] for c in summary['checkpoints']] == [20000]
    history = client.get(f'/api/models/{run.name}/history?points=10').json()
    assert history['series'][0]['step'] == 1000 and history['series'][-1]['step'] == 30000
    assert len(history['series']) <= 11 and history['series'][0]['spin'] == .75
    assert client.get('/api/models/..%5C..%5Cetc/history').status_code == 404


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


def test_missions_fly_the_training_plant_without_pid_or_legacy_vanes(tmp_path, monkeypatch):
    import json
    import pytest
    from mission_control import models
    monkeypatch.setattr(models, 'policy_paths', lambda: {'ppo_mission': 'new.pt'})
    monkeypatch.setattr(server, 'convex_available', lambda: True)
    assert models.validate_mission({})['controller'] == 'convex'
    assert models.validate_mission({'controller': 'convex'})['vane_model'] == 'momentum'
    assert models.validate_mission({'controller': 'ppo_mission'})['vane_model'] == 'momentum'
    # The PID baseline and the legacy vanes were removed on 2026-09-26.
    with pytest.raises(ValueError, match='PID baseline was removed'):
        models.validate_mission({'controller': 'pid'})
    with pytest.raises(ValueError, match='legacy vane physics was removed'):
        models.validate_mission({'controller': 'convex', 'vane_model': 'legacy'})
    with pytest.raises(ValueError, match='Unknown vane model'):
        models.validate_mission({'controller': 'convex', 'vane_model': 'cfd'})
    assert 'pid' not in TestClient(server.app).get('/api/config').json()['policies']
    # The same physics and dynamics sections the waypoint_flight policies train on.
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
