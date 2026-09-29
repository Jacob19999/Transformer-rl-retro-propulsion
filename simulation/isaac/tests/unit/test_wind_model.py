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


BODY = dict(length=0.35, diameter=0.12, cd_body=1.0, reference_area=0.011)


def test_body_drag_uses_side_on_area_crosswise_and_end_on_area_axially():
    model = WindModel(**WindModel.body_drag_from_vehicle(BODY))
    level = torch.tensor([[1., 0., 0., 0.]])
    side = model.compute_drag_force(torch.tensor([[3., 0., 0.]]), level)
    down = model.compute_drag_force(torch.tensor([[0., 0., -3.]]), level)
    q = 0.5 * 1.225 * 9.0
    assert torch.allclose(side[0], torch.tensor([-q * 0.35 * 0.12, 0., 0.]), atol=1e-6)
    # Falling (world -z) is body +z in FRD at level attitude; drag pushes back up (FRD -z).
    assert torch.allclose(down[0], torch.tensor([0., 0., -q * 0.011]), atol=1e-6)


def test_body_drag_is_dissipative_at_any_attitude():
    torch.manual_seed(1)
    from tvc_env.common.quaternions import normalize, rotate_vector
    from tvc_env.common.frames import frd_to_isaac
    model = WindModel(**WindModel.body_drag_from_vehicle(BODY), num_envs=128)
    q = normalize(torch.randn(128, 4))
    v = torch.randn(128, 3) * 5.0
    force_world = rotate_vector(q, frd_to_isaac(model.compute_drag_force(v, q)))
    assert torch.all((force_world * v).sum(-1) <= 1e-9)


def test_wind_changes_the_air_not_the_airframe():
    """Enabling wind used to swap the vehicle's 0.011 m2 for an isotropic 0.02 m2."""
    calm = WindModel(**WindModel.body_drag_from_vehicle(BODY))
    windy = WindModel.from_disturbance_config({'disturbances': {
        'enabled': True, 'wind': {'enabled': True, 'steady_vector': [0., 0., 0.]},
        'gust': {'enabled': False}, 'body_drag': {'cd': 1., 'reference_area': .02}}}, body=BODY)
    velocity, level = torch.tensor([[2., -1., 0.5]]), torch.tensor([[1., 0., 0., 0.]])
    assert torch.equal(calm.compute_drag_force(velocity, level), windy.compute_drag_force(velocity, level))
