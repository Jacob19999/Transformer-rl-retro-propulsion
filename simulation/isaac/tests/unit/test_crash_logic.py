"""
Unit tests for crash detection criteria.

Tests each criterion independently, verifies below-threshold does not trigger,
and validates vectorized evaluation across multiple environments.
"""

import pytest
import torch
from tvc_env.sim.crash_logic import CrashDetector
from tvc_env.common.quaternions import from_euler


@pytest.fixture
def detector():
    return CrashDetector(
        max_impact_speed=3.0,
        max_tilt_at_contact=0.5,
        max_angular_rate_at_contact=3.0,
        max_tilt=1.57,
        max_altitude_error=10.0,
    )


@pytest.fixture
def identity_quat():
    return torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # Upright


@pytest.fixture
def tilted_quat():
    # 60° roll — exceeds 28° contact limit
    return from_euler(torch.tensor([1.05]), torch.tensor([0.0]), torch.tensor([0.0])).unsqueeze(0)


@pytest.fixture
def flipped_quat():
    # 100° roll — exceeds 90° absolute limit
    return from_euler(torch.tensor([1.75]), torch.tensor([0.0]), torch.tensor([0.0])).unsqueeze(0)


class TestImpactSpeed:
    def test_triggers_above_threshold(self, detector, identity_quat):
        """Crash detected when impact speed exceeds max_impact_speed."""
        impact = torch.tensor([4.0])
        in_contact = torch.tensor([True])
        result = detector.check_impact_speed(impact, in_contact)
        assert result[0].item() is True

    def test_no_trigger_below_threshold(self, detector):
        """No crash when impact speed is below threshold."""
        impact = torch.tensor([2.0])
        in_contact = torch.tensor([True])
        result = detector.check_impact_speed(impact, in_contact)
        assert result[0].item() is False

    def test_no_trigger_without_contact(self, detector):
        """No crash when not in contact, even at high speed."""
        impact = torch.tensor([10.0])
        in_contact = torch.tensor([False])
        result = detector.check_impact_speed(impact, in_contact)
        assert result[0].item() is False


class TestTiltAtContact:
    def test_triggers_for_tilted_vehicle_in_contact(self, detector, tilted_quat):
        """Crash detected when tilt at contact exceeds threshold."""
        in_contact = torch.tensor([True])
        result = detector.check_tilt_at_contact(tilted_quat, in_contact)
        assert result[0].item() is True

    def test_no_trigger_for_upright_vehicle(self, detector, identity_quat):
        """No crash when upright vehicle contacts ground."""
        in_contact = torch.tensor([True])
        result = detector.check_tilt_at_contact(identity_quat, in_contact)
        assert result[0].item() is False

    def test_no_trigger_without_contact(self, detector, tilted_quat):
        """No crash when tilted but not in contact."""
        in_contact = torch.tensor([False])
        result = detector.check_tilt_at_contact(tilted_quat, in_contact)
        assert result[0].item() is False


class TestAngularRateAtContact:
    def test_triggers_above_threshold(self, detector):
        """Crash detected when angular rate at contact exceeds threshold."""
        rate = torch.tensor([5.0])
        in_contact = torch.tensor([True])
        result = detector.check_angular_rate_at_contact(rate, in_contact)
        assert result[0].item() is True

    def test_no_trigger_below_threshold(self, detector):
        """No crash when angular rate is below threshold."""
        rate = torch.tensor([1.0])
        in_contact = torch.tensor([True])
        result = detector.check_angular_rate_at_contact(rate, in_contact)
        assert result[0].item() is False


class TestExcessiveTilt:
    def test_triggers_for_flipped_vehicle(self, detector, flipped_quat):
        """Crash detected when tilt exceeds 90°."""
        result = detector.check_excessive_tilt(flipped_quat)
        assert result[0].item() is True

    def test_no_trigger_for_upright(self, detector, identity_quat):
        """No crash when vehicle is upright."""
        result = detector.check_excessive_tilt(identity_quat)
        assert result[0].item() is False

    def test_no_trigger_for_small_tilt(self, detector):
        """No crash for small tilt angles."""
        q = from_euler(torch.tensor([0.2]), torch.tensor([0.1]), torch.tensor([0.0])).unsqueeze(0)
        result = detector.check_excessive_tilt(q)
        assert result[0].item() is False


class TestAltitudeError:
    def test_triggers_for_large_error(self, detector):
        """Crash detected when altitude error exceeds threshold."""
        err = torch.tensor([15.0])
        result = detector.check_altitude_error(err)
        assert result[0].item() is True

    def test_no_trigger_for_small_error(self, detector):
        """No crash for altitude error below threshold."""
        err = torch.tensor([5.0])
        result = detector.check_altitude_error(err)
        assert result[0].item() is False


class TestConfigLoading:
    def test_from_task_config_reads_nested_task_termination_values(self):
        detector = CrashDetector.from_task_config(
            {
                "task": {
                    "termination": {
                        "max_impact_speed": 1.25,
                        "max_tilt_at_contact": 0.25,
                        "max_angular_rate_at_contact": 2.5,
                        "max_tilt": 0.75,
                        "max_altitude_error": 30.0,
                    }
                }
            }
        )

        assert detector.max_impact_speed == 1.25
        assert detector.max_tilt_at_contact == 0.25
        assert detector.max_angular_rate_at_contact == 2.5
        assert detector.max_tilt == 0.75
        assert detector.max_altitude_error == 30.0


class TestVectorizedEvaluation:
    def test_evaluate_handles_batch(self, detector):
        """evaluate() should work across a batch of environments."""
        N = 32
        in_contact = torch.zeros(N, dtype=torch.bool)
        impact_speed = torch.zeros(N)
        q = torch.zeros(N, 4)
        q[:, 0] = 1.0  # identity
        angular_rate = torch.zeros(N)
        altitude_error = torch.zeros(N)

        result = detector.evaluate(in_contact, impact_speed, q, angular_rate, altitude_error)
        assert result.shape == (N,)
        assert not result.any()  # No crashes in clean scenario

    def test_each_criterion_independent(self, detector):
        """Each criterion should trigger independently of others."""
        N = 5
        # Env 0: impact speed crash
        # Env 1: tilt at contact crash
        # Env 2: angular rate crash
        # Env 3: excessive tilt crash (no contact needed)
        # Env 4: altitude crash

        in_contact = torch.tensor([True, True, True, False, False])
        impact_speed = torch.tensor([5.0, 0.0, 0.0, 0.0, 0.0])
        q = torch.zeros(N, 4)
        q[:, 0] = 1.0  # all upright except env 1 and 3
        # Set env 1 to tilted (large roll)
        q[1] = from_euler(torch.tensor([1.05]), torch.tensor([0.0]), torch.tensor([0.0]))
        # Set env 3 to flipped
        q[3] = from_euler(torch.tensor([1.75]), torch.tensor([0.0]), torch.tensor([0.0]))
        angular_rate = torch.tensor([0.0, 0.0, 5.0, 0.0, 0.0])
        altitude_error = torch.tensor([0.0, 0.0, 0.0, 0.0, 15.0])

        result = detector.evaluate(in_contact, impact_speed, q, angular_rate, altitude_error)
        assert result[0].item() is True   # impact crash
        assert result[1].item() is True   # tilt at contact
        assert result[2].item() is True   # angular rate at contact
        assert result[3].item() is True   # excessive tilt
        assert result[4].item() is True   # altitude error
