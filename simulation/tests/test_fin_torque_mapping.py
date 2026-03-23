"""Unit tests for derived fin mapping utilities (no Isaac runtime required)."""

from __future__ import annotations

import numpy as np
from simulation.isaac.fin_mapping import (
    default_fin_mapping,
    derive_mapping_from_axis_response,
    mix_controls,
)


def test_default_mapping_keeps_expected_yaw_pattern() -> None:
    """Positive yaw command: pitch-pair positive, roll-pair negative.

    The 3° fin cant makes all pitch-pair fins produce positive yaw torque
    and all roll-pair fins produce negative yaw torque for positive deflection.
    Weights [+1,+1,-1,-1] reinforce this pattern.
    """
    mapping = default_fin_mapping()
    fins = np.array(mix_controls(pitch_cmd=0.0, roll_cmd=0.0, yaw_cmd=0.10, delta_max=0.20, mapping=mapping))
    assert fins[0] > 0.0   # right fin (pitch pair) positive
    assert fins[1] > 0.0   # left fin (pitch pair) positive
    assert fins[2] < 0.0   # fwd fin (roll pair) negative
    assert fins[3] < 0.0   # aft fin (roll pair) negative


def test_default_mapping_pitch_roll_groups() -> None:
    mapping = default_fin_mapping()
    pitch_fins = np.array(mix_controls(pitch_cmd=0.08, roll_cmd=0.0, yaw_cmd=0.0, delta_max=0.20, mapping=mapping))
    roll_fins = np.array(mix_controls(pitch_cmd=0.0, roll_cmd=0.08, yaw_cmd=0.0, delta_max=0.20, mapping=mapping))
    assert np.allclose(pitch_fins[0:2], [0.4, 0.4], atol=1e-6)
    assert np.allclose(pitch_fins[2:4], [0.0, 0.0], atol=1e-6)
    assert np.allclose(roll_fins[0:2], [0.0, 0.0], atol=1e-6)
    assert np.allclose(roll_fins[2:4], [0.4, 0.4], atol=1e-6)


def test_derive_mapping_from_axis_response() -> None:
    # Fin1/Fin2 pitch-dominant, Fin3/Fin4 roll-dominant.
    mapping = derive_mapping_from_axis_response(
        dominant_axis=["pitch", "pitch", "roll", "roll"],
        dominant_sign=[+1.0, +1.0, +1.0, +1.0],
    )
    pitch_fins = np.array(mix_controls(pitch_cmd=0.10, roll_cmd=0.0, yaw_cmd=0.0, delta_max=0.20, mapping=mapping))
    roll_fins = np.array(mix_controls(pitch_cmd=0.0, roll_cmd=0.10, yaw_cmd=0.0, delta_max=0.20, mapping=mapping))
    assert np.allclose(pitch_fins, [0.5, 0.5, 0.0, 0.0], atol=1e-6)
    assert np.allclose(roll_fins, [0.0, 0.0, 0.5, 0.5], atol=1e-6)


def test_derive_mapping_uses_yaw_torque_signs() -> None:
    """When yaw_torque_signs are provided, yaw weights match those signs."""
    mapping = derive_mapping_from_axis_response(
        dominant_axis=["pitch", "pitch", "roll", "roll"],
        dominant_sign=[+1.0, +1.0, -1.0, -1.0],
        yaw_torque_signs=[+1.0, +1.0, -1.0, -1.0],
    )
    yaw_fins = np.array(mix_controls(pitch_cmd=0.0, roll_cmd=0.0, yaw_cmd=0.10, delta_max=0.20, mapping=mapping))
    # Pitch-pair positive, roll-pair negative
    assert yaw_fins[0] > 0.0
    assert yaw_fins[1] > 0.0
    assert yaw_fins[2] < 0.0
    assert yaw_fins[3] < 0.0


def test_derive_mapping_inverted_yaw_torque_signs() -> None:
    """Reversed yaw signs produce reversed yaw weights."""
    mapping = derive_mapping_from_axis_response(
        dominant_axis=["pitch", "pitch", "roll", "roll"],
        dominant_sign=[+1.0, +1.0, -1.0, -1.0],
        yaw_torque_signs=[-1.0, -1.0, +1.0, +1.0],
    )
    yaw_fins = np.array(mix_controls(pitch_cmd=0.0, roll_cmd=0.0, yaw_cmd=0.10, delta_max=0.20, mapping=mapping))
    assert yaw_fins[0] < 0.0
    assert yaw_fins[1] < 0.0
    assert yaw_fins[2] > 0.0
    assert yaw_fins[3] > 0.0
