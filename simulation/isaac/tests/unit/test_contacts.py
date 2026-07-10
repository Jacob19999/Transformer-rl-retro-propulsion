"""Unit tests for contact-state-machine config plumbing."""

import torch

from tvc_env.common.constants import ContactState
from tvc_env.sim.contacts import ContactStateMachine


def test_contact_state_machine_reads_nested_task_contact_config():
    machine = ContactStateMachine.from_task_config(
        {
            "task": {
                "contact": {
                    "dwell_frames": 15,
                    "min_contact_force": 2.5,
                }
            }
        },
        num_envs=3,
        device=torch.device("cpu"),
    )

    assert machine.dwell_frames == 15
    assert machine.min_contact_force == 2.5


def test_contact_state_machine_requires_minimum_contact_force_for_landing():
    machine = ContactStateMachine(
        num_envs=1,
        dwell_frames=2,
        min_contact_force=5.0,
        device=torch.device("cpu"),
    )

    for _ in range(3):
        state = machine.update(
            in_contact=torch.tensor([True]),
            is_crashed=torch.tensor([False]),
            contact_force=torch.tensor([1.0]),
        )

    assert state[0].item() == ContactState.AIRBORNE
