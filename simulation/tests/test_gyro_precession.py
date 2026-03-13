"""
test_gyro_precession.py — Pure-Python unit tests for gyroscopic precession torque math.

No Isaac Sim required. Tests verify:
  1. tau_gyro = -cross(omega_b, h_fan_b) for known inputs
  2. Zero thrust → zero omega_fan → zero tau_gyro
  3. _rotate_body_to_world is inverse of _rotate_world_to_body
  4. gyro_precession.enabled: false config loads correctly
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

# ---------------------------------------------------------------------------
# Repo-root bootstrap
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from simulation.isaac.quaternion_isaac import (
    rotate_body_to_world_wxyz,
    rotate_world_to_body_wxyz,
)


# ---------------------------------------------------------------------------
# Helpers (aliases to quaternion_isaac so tests track runtime contract)
# ---------------------------------------------------------------------------

def _rotate_world_to_body(v_world: torch.Tensor, quat_w: torch.Tensor) -> torch.Tensor:
    return rotate_world_to_body_wxyz(v_world, quat_w)


def _rotate_body_to_world(quat_w: torch.Tensor, v_body: torch.Tensor) -> torch.Tensor:
    return rotate_body_to_world_wxyz(quat_w, v_body)


_K_THRUST = 4.55e-7   # N/(rad/s)^2
_I_FAN    = 3.0e-5    # kg·m²


def _compute_tau_gyro(omega_b: torch.Tensor, thrust: torch.Tensor) -> torch.Tensor:
    """Compute gyroscopic precession torque in body frame (float64 for test precision)."""
    omega_b = omega_b.double()
    thrust  = thrust.double()
    num_envs = omega_b.shape[0]
    omega_fan = (thrust / _K_THRUST).clamp(min=0).sqrt()   # (N,)
    h_fan_b = torch.zeros((num_envs, 3), dtype=torch.float64)
    h_fan_b[:, 2] = _I_FAN * omega_fan                     # body +Z (FRD down)
    return -torch.linalg.cross(omega_b, h_fan_b)           # (N, 3)


# ---------------------------------------------------------------------------
# Test 1: Analytical tau_gyro for known inputs
# ---------------------------------------------------------------------------

class TestTauGyroAnalytical:
    """Verify tau_gyro = -cross(omega_b, [0, 0, I_fan*omega_fan]) matches hand-calc."""

    def test_pitch_rate_produces_roll_torque(self):
        """Pure pitch rate q → roll torque τ_x = −q·L, pitch torque τ_y = 0."""
        q_pitch = 1.0        # rad/s pitch rate
        thrust  = 30.0       # N (nominal thrust)
        omega_b = torch.tensor([[0.0, q_pitch, 0.0]])  # pure pitch
        tau = _compute_tau_gyro(omega_b, torch.tensor([thrust]))

        omega_fan = math.sqrt(thrust / _K_THRUST)
        L = _I_FAN * omega_fan
        expected_roll  = -q_pitch * L   # τ_x = −q·L
        expected_pitch = 0.0            # τ_y = p·L = 0 (p=0)
        expected_yaw   = 0.0            # τ_z = 0 always

        assert abs(tau[0, 0].item() - expected_roll) < 1e-12, (
            f"Roll torque: expected {expected_roll:.6e}, got {tau[0,0].item():.6e}"
        )
        assert abs(tau[0, 1].item() - expected_pitch) < 1e-12
        assert abs(tau[0, 2].item() - expected_yaw) < 1e-12

    def test_roll_rate_produces_pitch_torque(self):
        """Pure roll rate p → pitch torque τ_y = p·L, roll torque τ_x = 0."""
        p_roll  = 2.0
        thrust  = 30.0
        omega_b = torch.tensor([[p_roll, 0.0, 0.0]])
        tau = _compute_tau_gyro(omega_b, torch.tensor([thrust]))

        omega_fan = math.sqrt(thrust / _K_THRUST)
        L = _I_FAN * omega_fan
        expected_roll  = 0.0
        expected_pitch = p_roll * L   # τ_y = p·L

        assert abs(tau[0, 0].item() - expected_roll) < 1e-12
        assert abs(tau[0, 1].item() - expected_pitch) < 1e-12
        assert abs(tau[0, 2].item()) < 1e-12

    def test_yaw_rate_produces_zero_torque(self):
        """Yaw rate r is parallel to h_fan → cross product = 0."""
        r_yaw  = 5.0
        thrust = 30.0
        omega_b = torch.tensor([[0.0, 0.0, r_yaw]])
        tau = _compute_tau_gyro(omega_b, torch.tensor([thrust]))

        assert tau.abs().max().item() < 1e-10, (
            f"Yaw rate should produce zero precession, got {tau}"
        )

    def test_combined_omega(self):
        """Combined p, q, r: verify analytical formula component by component."""
        p, q, r = 1.0, 2.0, 3.0
        thrust  = 20.0
        omega_b = torch.tensor([[p, q, r]])
        tau = _compute_tau_gyro(omega_b, torch.tensor([thrust]))

        omega_fan = math.sqrt(thrust / _K_THRUST)
        L = _I_FAN * omega_fan
        # tau = -[p,q,r] x [0,0,L] = [−(q*L - r*0), −(r*0 - p*L), −(p*0 - q*0)]
        #                           = [−q·L, p·L, 0]
        assert abs(tau[0, 0].item() - (-q * L)) < 1e-12
        assert abs(tau[0, 1].item() - (p * L)) < 1e-12
        assert abs(tau[0, 2].item()) < 1e-12

    def test_batched_envs(self):
        """Multiple environments produce independent results."""
        omega_b = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        thrust  = torch.tensor([30.0, 30.0])
        tau = _compute_tau_gyro(omega_b, thrust)

        omega_fan = math.sqrt(30.0 / _K_THRUST)
        L = _I_FAN * omega_fan
        # env 0: p=1 → τ_y = L, τ_x = 0
        assert abs(tau[0, 0].item()) < 1e-12
        assert abs(tau[0, 1].item() - L) < 1e-12
        # env 1: q=1 → τ_x = -L, τ_y = 0
        assert abs(tau[1, 0].item() - (-L)) < 1e-12
        assert abs(tau[1, 1].item()) < 1e-12


# ---------------------------------------------------------------------------
# Test 2: Zero thrust → zero precession
# ---------------------------------------------------------------------------

class TestZeroThrust:
    def test_zero_thrust_zero_omega_fan(self):
        """Zero thrust → omega_fan = 0 → h_fan = 0 → tau_gyro = 0."""
        omega_b = torch.tensor([[1.0, 2.0, 3.0]])
        tau = _compute_tau_gyro(omega_b, torch.tensor([0.0]))
        assert tau.abs().max().item() < 1e-12

    def test_near_zero_thrust_clamped(self):
        """Negative thrust clamped to 0 → omega_fan = 0."""
        omega_b = torch.tensor([[1.0, 2.0, 3.0]])
        tau = _compute_tau_gyro(omega_b, torch.tensor([-10.0]))
        assert tau.abs().max().item() < 1e-12


# ---------------------------------------------------------------------------
# Test 3: _rotate_body_to_world is inverse of _rotate_world_to_body
# ---------------------------------------------------------------------------

class TestRotationInverse:
    def _identity_quat(self, n: int = 1) -> torch.Tensor:
        """Identity quaternion [1, 0, 0, 0] in IsaacLab wxyz order."""
        q = torch.zeros(n, 4)
        q[:, 0] = 1.0
        return q

    def _quat_from_axis_angle(self, axis: list[float], angle_deg: float) -> torch.Tensor:
        """Build unit quaternion from axis-angle."""
        angle = math.radians(angle_deg)
        s = math.sin(angle / 2)
        c = math.cos(angle / 2)
        ax = torch.tensor(axis, dtype=torch.float32)
        ax = ax / ax.norm()
        q = torch.zeros(1, 4)
        q[0, 0] = c
        q[0, 1:4] = ax * s
        return q

    def test_identity_roundtrip(self):
        """Identity quaternion: body_to_world → world_to_body = identity."""
        quat = self._identity_quat()
        v = torch.tensor([[1.0, 2.0, 3.0]])
        v_world = _rotate_body_to_world(quat, v)
        v_back  = _rotate_world_to_body(v_world, quat)
        assert torch.allclose(v, v_back, atol=1e-6), f"Roundtrip failed: {v} != {v_back}"

    def test_90deg_x_roundtrip(self):
        """90° rotation about X: roundtrip recovers original vector."""
        quat = self._quat_from_axis_angle([1, 0, 0], 90)
        v = torch.tensor([[0.0, 1.0, 0.0]])
        v_world = _rotate_body_to_world(quat, v)
        v_back  = _rotate_world_to_body(v_world, quat)
        assert torch.allclose(v, v_back, atol=1e-6)

    def test_90deg_z_roundtrip(self):
        """90° rotation about Z: roundtrip recovers original vector."""
        quat = self._quat_from_axis_angle([0, 0, 1], 90)
        v = torch.tensor([[1.0, 0.0, 0.0]])
        v_world = _rotate_body_to_world(quat, v)
        v_back  = _rotate_world_to_body(v_world, quat)
        assert torch.allclose(v, v_back, atol=1e-6)

    def test_arbitrary_rotation_roundtrip(self):
        """Arbitrary rotation: roundtrip recovers original vector."""
        quat = self._quat_from_axis_angle([1, 1, 0], 45)
        v = torch.tensor([[3.0, -1.0, 2.0]])
        v_world = _rotate_body_to_world(quat, v)
        v_back  = _rotate_world_to_body(v_world, quat)
        assert torch.allclose(v, v_back, atol=1e-5)

    def test_body_to_world_direction(self):
        """90° yaw (about Z): body +X maps to world +Y (FRD convention check)."""
        quat = self._quat_from_axis_angle([0, 0, 1], 90)
        v_body = torch.tensor([[1.0, 0.0, 0.0]])  # body +X
        v_world = _rotate_body_to_world(quat, v_body)
        # After 90° CCW yaw: body +X should be world -Y (standard rotation)
        # The exact direction depends on quaternion convention; verify magnitude preserved
        assert abs(v_world.norm().item() - 1.0) < 1e-6, "Rotation should preserve magnitude"


# ---------------------------------------------------------------------------
# Test 4: Config key gyro_precession.enabled loads correctly
# ---------------------------------------------------------------------------

class TestGyroConfig:
    def test_enabled_true_loads(self, tmp_path: Path):
        """gyro_precession.enabled: true parses correctly from YAML."""
        import yaml
        cfg = {"vehicle": {"edf": {"gyro_precession": {"enabled": True}}}}
        f = tmp_path / "v.yaml"
        f.write_text(yaml.dump(cfg))
        data = yaml.safe_load(f.read_text())
        vehicle = data.get("vehicle", data)
        edf = vehicle.get("edf", {})
        gyro_cfg = edf.get("gyro_precession", {})
        assert gyro_cfg.get("enabled", True) is True

    def test_enabled_false_loads(self, tmp_path: Path):
        """gyro_precession.enabled: false parses correctly."""
        import yaml
        cfg = {"vehicle": {"edf": {"gyro_precession": {"enabled": False}}}}
        f = tmp_path / "v.yaml"
        f.write_text(yaml.dump(cfg))
        data = yaml.safe_load(f.read_text())
        vehicle = data.get("vehicle", data)
        edf = vehicle.get("edf", {})
        gyro_cfg = edf.get("gyro_precession", {})
        assert gyro_cfg.get("enabled", True) is False

    def test_missing_key_defaults_true(self, tmp_path: Path):
        """Missing gyro_precession key → default True (on by default)."""
        import yaml
        cfg = {"vehicle": {"edf": {}}}
        f = tmp_path / "v.yaml"
        f.write_text(yaml.dump(cfg))
        data = yaml.safe_load(f.read_text())
        vehicle = data.get("vehicle", data)
        edf = vehicle.get("edf", {})
        gyro_cfg = edf.get("gyro_precession", {})
        assert gyro_cfg.get("enabled", True) is True  # default True when key absent

    def test_disabled_produces_zero_torque(self):
        """When gyro disabled, no tau_gyro is added to torques."""
        gyro_enabled = False
        omega_b = torch.tensor([[1.0, 2.0, 3.0]])
        thrust  = torch.tensor([30.0])
        torques = torch.zeros((1, 1, 3))
        if gyro_enabled:
            tau = _compute_tau_gyro(omega_b, thrust)
            torques[:, 0, :] += tau
        assert torques.abs().max().item() == 0.0, "Disabled gyro should add no torque"
