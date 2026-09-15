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
