"""Built-in convex profiles, environment presets and sample flight plans."""
import json

import pytest
from fastapi.testclient import TestClient

from mission_control import server
from mission_control.convex_parameters import base_settings, resolve
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


def test_sample_plans_validate_and_carry_their_profile():
    samples = list_samples()
    assert len(samples) >= 10
    assert {s['category'] for s in samples} == {'hop', 'land', 'hover'}
    profiles = {p['id']: p for p in convex_presets()}
    for summary in samples:
        record = read_sample(summary['id'])
        mission = record['mission']
        assert validate_mission(mission) == mission
        assert mission['controller'] == 'convex'
        assert mission['convex_settings'] == profiles[summary['profile']]['settings']
        assert summary['profile_name'] == profiles[summary['profile']]['name']
        if summary['category'] in ('hop', 'hover'):
            assert mission['position'][2] < .5 and mission['waypoints'][0]['type'] == 'takeoff'
        assert summary['route'][0] == mission['position'] and summary['route'][-1][2] == 0
    hops = [s['max_altitude_m'] for s in samples if s['category'] == 'hop']
    assert min(hops) <= 3 and max(hops) >= 20


def test_sample_files_are_canonical_version_2_plans():
    for path in sorted((FOLDER / 'flight_plans').glob('*.json')):
        record = json.loads(path.read_text(encoding='utf-8'))
        assert record['format'] == 'edf-flight-plan' and record['version'] == 2
        # Stored exactly as validated, so an export of a loaded sample is identical.
        assert validate_mission(record['mission']) == record['mission']


def test_preset_endpoints_and_sample_path_validation():
    client = TestClient(server.app)
    assert len(client.get('/api/convex-presets').json()) == len(convex_presets())
    assert client.get('/api/disturbance-presets').json()[0]['id'] == 'calm'
    samples = client.get('/api/flight-plan-samples').json()
    plan = client.get(f"/api/flight-plan-samples/{samples[0]['id']}").json()
    assert plan['format'] == 'edf-flight-plan' and plan['mission']['name'] == samples[0]['name']
    for key in ('..%2Fconvex_profiles', 'missing-sample', '99-nope'):
        assert client.get(f'/api/flight-plan-samples/{key}').status_code == 404


def test_sample_settings_resolve_against_the_repository_yaml():
    base = base_settings()
    for summary in list_samples():
        settings = resolve(read_sample(summary['id'])['mission']['convex_settings'])
        assert set(settings) == set(base)
        with pytest.raises(KeyError):
            settings['guidance']['not_a_parameter']
