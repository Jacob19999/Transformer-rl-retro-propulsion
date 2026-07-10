"""Unit tests for the Isaac link force bridge."""

import torch

from tvc_env.sim.link_force_interface import LinkForceInterface


class _Composer:
    def __init__(self):
        self.reset_env_ids = "not-called"

    def reset(self, env_ids=None):
        self.reset_env_ids = env_ids


class _Art:
    device = "cpu"

    def __init__(self):
        self.num_instances = 2
        self.permanent_wrench_composer = _Composer()


class _Map:
    fin_body_indices = [1, 2, 3, 4]


def test_clear_external_wrenches_resets_selected_env_slots():
    art = _Art()
    iface = LinkForceInterface(art, _Map(), torch.zeros(4, 3))
    env_ids = torch.tensor([1], dtype=torch.int64)

    iface.clear_external_wrenches(env_ids)

    assert torch.equal(art.permanent_wrench_composer.reset_env_ids, env_ids)
