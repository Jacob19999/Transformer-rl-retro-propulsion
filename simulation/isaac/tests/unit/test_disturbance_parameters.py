import pytest
from fastapi.testclient import TestClient

from mission_control import server
from mission_control.models import validate_mission, disturbance_config
from mission_control.disturbance_parameters import defaults, validate_settings


def test_parameters_reach_models_and_preserve_enable_flags():
    overrides = dict(wind=dict(steady_vector=[3, -4, .5]),
                     gust=dict(magnitude=2, duration=.8, interval=[3, 6]),
                     sensor_noise=dict(position_std=.03),
                     com_offset=dict(range=[[-.02, 0, -.005], [.01, .005, .005]]))
    mission = validate_mission(dict(disturbance=['wind', 'sensor_noise', 'com_shift'],
                                   disturbance_settings=overrides))
    assert validate_mission(mission) == mission
    resolved = disturbance_config(mission)['disturbances']
    for section, values in overrides.items():
        for key, value in values.items():
            assert resolved[section][key] == value
        assert resolved[section]['enabled'] is True
    mission['disturbance'] = []
    resolved = disturbance_config(mission)['disturbances']
    assert not resolved['enabled']
    assert all(not resolved[key]['enabled'] for key in overrides)


@pytest.mark.parametrize('value', [
    {'wind': {'enabled': True}}, {'wind': {'steady_vector': [12, 12, 0]}},
    {'wind': {'steady_vector': [0, 0]}}, {'wind': {'steady_vector': [0, 0, float('nan')]}},
    {'gust': {'interval': [8, 3]}}, {'gust': {'duration': 0}},
    {'sensor_noise': {'position_std': True}}, {'sensor_noise': {'attitude_std': -.1}},
    {'com_offset': {'range': [[.02, 0, 0], [.01, 0, 0]]}},
    {'com_offset': {'range': [[0, 0, 0], [.051, 0, 0]]}}, {'unknown': {}},
])
def test_invalid_settings_rejected(value):
    with pytest.raises(ValueError):
        validate_settings(value)


def test_legacy_requests_keep_repository_presets():
    settings = defaults()
    request = validate_mission(dict(disturbance=['wind', 'sensor_noise', 'com_shift']))
    assert request['disturbance_settings'] == {}
    resolved = disturbance_config(request)['disturbances']
    for section, values in settings.items():
        for key, value in values.items():
            assert resolved[section][key] == value


def test_validation_round_trips_disturbances_and_plans_leave_them_out(tmp_path, monkeypatch):
    from mission_control import flight_plans
    monkeypatch.setattr(flight_plans, 'LIBRARY', tmp_path)
    client = TestClient(server.app)
    headers = {'X-Mission-Control': 'local'}
    settings = defaults()
    settings['wind']['steady_vector'] = [0, 3, 0]
    request = dict(name='Custom disturbances', disturbance=['wind'], disturbance_settings=settings)
    response = client.post('/api/flight-plan/validate', json=request, headers=headers)
    assert response.status_code == 200
    assert response.json()['disturbance_settings'] == settings
    # Flight plans are route-only: the environment is chosen per run, never saved with a plan.
    saved = client.post('/api/flight-plans', json=request, headers=headers).json()
    mission = client.get(f'/api/flight-plans/{saved["id"]}').json()['mission']
    assert 'disturbance' not in mission and 'disturbance_settings' not in mission
    assert client.get('/api/config').json()['disturbance_defaults'] == defaults()
