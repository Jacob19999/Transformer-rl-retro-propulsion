"""Unit tests for contact-sensor body classification."""

from types import SimpleNamespace

import torch

from tvc_env.sim.sensor_interface import SensorInterface


class _ContactSensor:
    body_names = ["AftFin", "Body", "FwdFin", "LeftFin", "RightFin"]

    def __init__(self, forces: torch.Tensor):
        self.data = SimpleNamespace(net_forces_w=forces)


_METADATA = {
    "body_link_name": "Body",
    "fin_link_names": ["FwdFin", "RightFin", "AftFin", "LeftFin"],
}


def test_body_force_is_landing_contact_force():
    forces = torch.zeros(2, 5, 3)
    forces[0, 1, 2] = 12.0
    iface = SensorInterface(_ContactSensor(forces), _METADATA)

    assert torch.equal(iface.get_landing_contact_force(), torch.tensor([12.0, 0.0]))
    assert torch.equal(iface.is_in_contact(), torch.tensor([True, False]))


def test_fin_force_is_unsafe_not_landing_contact():
    forces = torch.zeros(1, 5, 3)
    forces[0, 0, 2] = 8.0
    iface = SensorInterface(_ContactSensor(forces), _METADATA)

    assert not iface.is_in_contact()[0]
    assert iface.has_unsafe_contact(force_threshold=1.0)[0]
