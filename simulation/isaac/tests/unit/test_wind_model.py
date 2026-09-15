"""Unit tests for vectorized wind/gust disturbance state."""

import torch

from tvc_env.dynamics.wind_model import WindModel


def test_wind_is_batched_per_environment():
    model = WindModel(steady_vector=[2.0, 0.5, 0.0], num_envs=3)
    wind = model.get_effective_wind_world()
    assert wind.shape == (3, 3)
    assert torch.allclose(wind, torch.tensor([[2.0, 0.5, 0.0]]).expand(3, -1))


def test_gusts_start_independently_and_reset_selected_envs():
    torch.manual_seed(3)
    model = WindModel(
        gust_enabled=True,
        gust_magnitude=5.0,
        gust_duration=0.5,
        gust_interval_min=0.0,
        gust_interval_max=0.0,
        num_envs=4,
    )
    model.update_gust(0.01)
    assert model._gust_active.all()
    assert model._gust_direction.shape == (4, 3)
    model.reset(torch.tensor([1, 3]))
    assert not model._gust_active[1]
    assert not model._gust_active[3]
    assert model._gust_active[0]
    assert model._gust_active[2]


def test_disabled_wind_preserves_drag_against_vehicle_motion():
    model = WindModel.from_disturbance_config({'disturbances': {
        'enabled': True, 'wind': {'enabled': False, 'steady_vector': [10., 0., 0.]},
        'gust': {'enabled': False}, 'body_drag': {'cd': 1., 'reference_area': .011},
    }})
    assert not model.get_effective_wind_world().any()
    velocity = torch.tensor([[2., 0., 0.]])
    force = model.compute_drag_force(velocity, torch.tensor([[1., 0., 0., 0.]]))
    assert force[0, 0] < 0
    assert force[0, 1:].abs().max() == 0
