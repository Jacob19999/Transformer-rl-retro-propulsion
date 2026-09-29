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


def test_imu_profile_selects_the_physical_chain_and_never_reaches_the_env_config():
    from mission_control.disturbance_parameters import imu_profiles
    assert {'wtgahrs1', 'bno085', 'vn110e'} <= set(imu_profiles())
    mission = validate_mission(dict(disturbance=['sensor_noise'],
                                    disturbance_settings=dict(sensor_noise=dict(imu_profile='vn110e', position_std=.02))))
    assert validate_mission(mission) == mission
    noise = disturbance_config(mission)['disturbances']['sensor_noise']
    assert noise['imu'] == {'enabled': True, 'profile': 'vn110e'} and 'imu_profile' not in noise
    assert noise['enabled'] and noise['position_std'] == .02
    # Unselected sensor noise, or an empty profile, leaves the white-noise model in place.
    mission['disturbance'] = []
    assert 'imu' not in disturbance_config(mission)['disturbances']['sensor_noise']
    mission = validate_mission(dict(disturbance=['sensor_noise'], disturbance_settings=dict(sensor_noise=dict(imu_profile=''))))
    assert 'imu' not in disturbance_config(mission)['disturbances']['sensor_noise']


def test_inertial_navigation_option_reaches_the_imu_config_and_needs_a_physical_chain():
    settings = dict(sensor_noise=dict(imu_profile='bno085', imu_nav='inertial'))
    mission = validate_mission(dict(disturbance=['sensor_noise'], disturbance_settings=settings))
    assert validate_mission(mission) == mission
    noise = disturbance_config(mission)['disturbances']['sensor_noise']
    assert noise['imu'] == {'enabled': True, 'profile': 'bno085', 'nav': {'enabled': True}}
    assert 'imu_nav' not in noise and 'imu_profile' not in noise
    # 'external' (and the default) leave position/velocity as the white-noise reference.
    external = validate_mission(dict(disturbance=['sensor_noise'],
                                     disturbance_settings=dict(sensor_noise=dict(imu_profile='bno085', imu_nav='external'))))
    assert 'nav' not in disturbance_config(external)['disturbances']['sensor_noise']['imu']
    for bad in ({'sensor_noise': {'imu_nav': 'inertial'}}, {'sensor_noise': {'imu_profile': '', 'imu_nav': 'inertial'}},
                {'sensor_noise': {'imu_profile': 'bno085', 'imu_nav': 'gps'}}):
        with pytest.raises(ValueError):
            validate_settings(bad)


def test_inertial_navigation_config_builds_a_model_with_nav_enabled():
    from tvc_env.dynamics.imu_model import imu_model_from_config
    mission = validate_mission(dict(disturbance=['sensor_noise'], disturbance_settings=dict(
        sensor_noise=dict(imu_profile='vn110e', imu_nav='inertial'))))
    noise = disturbance_config(mission)['disturbances']['sensor_noise']
    assert imu_model_from_config(2, 'cpu', 1 / 480, noise).nav_enabled


def test_imu_profile_reaches_a_buildable_simulator_model():
    from tvc_env.dynamics.imu_model import ImuModel, imu_model_from_config
    for name in ('wtgahrs1', 'bno085', 'vn110e'):
        mission = validate_mission(dict(disturbance=['sensor_noise'],
                                        disturbance_settings=dict(sensor_noise=dict(imu_profile=name))))
        noise = disturbance_config(mission)['disturbances']['sensor_noise']
        assert isinstance(imu_model_from_config(2, 'cpu', 1 / 480, noise), ImuModel)


@pytest.mark.parametrize('value', [{'sensor_noise': {'imu_profile': 'nonexistent'}},
                                   {'sensor_noise': {'imu_profile': '../etc'}}, {'sensor_noise': {'imu_profile': 3}},
                                   {'sensor_noise': {'imu_profile': None}}])
def test_bad_imu_profile_rejected(value):
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
