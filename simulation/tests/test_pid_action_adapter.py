from __future__ import annotations

import numpy as np

from simulation.isaac.pid_action_adapter import (
    map_hover_centered_thrust_to_isaac,
    map_pid_action_to_isaac,
)


def test_hover_centered_thrust_maps_endpoints_and_hover() -> None:
    hover = 0.68
    assert map_hover_centered_thrust_to_isaac(-1.0, hover_thrust_frac=hover) == 0.0
    assert map_hover_centered_thrust_to_isaac(0.0, hover_thrust_frac=hover) == hover
    assert map_hover_centered_thrust_to_isaac(1.0, hover_thrust_frac=hover) == 1.0


def test_hover_centered_thrust_maps_piecewise_linearly() -> None:
    hover = 0.68
    assert np.isclose(
        map_hover_centered_thrust_to_isaac(-0.5, hover_thrust_frac=hover),
        0.34,
    )
    assert np.isclose(
        map_hover_centered_thrust_to_isaac(0.5, hover_thrust_frac=hover),
        0.84,
    )


def test_pid_action_adapter_only_changes_thrust_channel() -> None:
    action = np.array([-0.25, 0.1, -0.2, 0.3, -0.4], dtype=np.float32)
    mapped = map_pid_action_to_isaac(action, hover_thrust_frac=0.7)
    assert mapped.shape == (5,)
    assert np.isclose(mapped[0], 0.525)
    np.testing.assert_allclose(mapped[1:], action[1:])
