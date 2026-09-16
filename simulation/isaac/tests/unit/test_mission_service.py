from fastapi.testclient import TestClient
from mission_control import server


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
