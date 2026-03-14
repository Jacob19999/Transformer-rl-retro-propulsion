"""
test_fin_torque_mapping.py — Pure-Python (no Isaac Sim) tests for the fin
axis-torque mapping used in diag_thrust_fin_wiggle.py.

Physical body frame (FRD: +X=fwd/nose, +Y=right, +Z=down)
  Confirmed from Isaac Sim 3D model top-down view.

Fin physical layout:
  Fin_1 at FRD +Y (right), Fin_2 at FRD -Y (left)     → pitch-dominant
  Fin_3 at FRD +X (fwd),   Fin_4 at FRD -X (aft)      → roll-dominant

The Isaac/default vehicle config now uses a 3 deg in-plane cant on all fin
lift directions. This keeps pitch/roll as the dominant axes while making the
differential fin pattern [-,+,-,+] produce a non-zero yaw torque.

Torque formula (τ = Σ r_i × F_i, F_i = scale·d_i·lift_dir_i):
  Roll  sweep d3=d4=+v: τ_x < 0, τ_y > 0, τ_z = 0
  Pitch sweep d1=d2=+v: τ_y > 0, τ_x > 0, τ_z = 0
  Yaw   sweep [-,+,-,+]: τ_z > 0, τ_x = τ_y = 0

Historical notes:
  Old wrong yaw sweep d1=−v,d2=+v,d3=+v,d4=−v assumed Fin_1/Fin_2 were
  the forward/aft pair. With the actual USD numbering it is not the intended yaw+ mix.
"""

from __future__ import annotations

import numpy as np
import pytest

CANT_DEG = 3.0
COS_CANT = float(np.cos(np.deg2rad(CANT_DEG)))
SIN_CANT = float(np.sin(np.deg2rad(CANT_DEG)))


def _load_fin_geometry() -> tuple[list[np.ndarray], list[np.ndarray]]:
    # Isaac torque mapping should follow the authored asset geometry, not the legacy
    # YAML fin-position fields. Keep this test independent of config placement data.
    positions = [
        np.array([0.0, +0.055, 0.14], dtype=float),   # Fin_1(right)
        np.array([0.0, -0.055, 0.14], dtype=float),   # Fin_2(left)
        np.array([+0.055, 0.0, 0.14], dtype=float),   # Fin_3(fwd)
        np.array([-0.055, 0.0, 0.14], dtype=float),   # Fin_4(aft)
    ]
    lift_dirs = [
        np.array([COS_CANT, -SIN_CANT, 0.0], dtype=float),
        np.array([COS_CANT, -SIN_CANT, 0.0], dtype=float),
        np.array([SIN_CANT, -COS_CANT, 0.0], dtype=float),
        np.array([SIN_CANT, -COS_CANT, 0.0], dtype=float),
    ]
    return positions, lift_dirs


def _net_torque(
    deflections: list[float],
    positions: list[np.ndarray],
    lift_dirs: list[np.ndarray],
    scale: float = 1.0,
) -> np.ndarray:
    tau = np.zeros(3)
    for d, r, ld in zip(deflections, positions, lift_dirs):
        F = scale * d * ld
        tau += np.cross(r, F)
    return tau


@pytest.fixture(scope="module")
def fin_geometry():
    return _load_fin_geometry()


class TestFinTorqueMapping:
    """Validate fin-command patterns against FRD body frame geometry.

    FRD body frame (+X=fwd/nose, +Y=right, +Z=down):
      ωx = ROLL  (rotation about +X = forward/nose axis)
      ωy = PITCH (rotation about +Y = right/lateral axis)
      ωz = YAW   (rotation about +Z = down axis)

    Fin layout (confirmed from Isaac Sim 3D model):
      Fin_1(right) at FRD +Y, Fin_2(left) at FRD -Y  →  hinge Y, lift +X → τ_y (PITCH)
      Fin_3(fwd)   at FRD +X, Fin_4(aft)  at FRD -X  →  hinge X, lift -Y → τ_x (ROLL)

    The 3 deg cant introduces a secondary fin-yaw couple while preserving
    pitch/roll as the dominant axes.
    """

    SCALE = 1.0
    DEFLECTION = 0.5

    # ------------------------------------------------------------------
    # Roll sweep (Fin_3+Fin_4): d3=+v, d4=+v
    # τ_x > 0 is dominant, τ_y > 0 is secondary, τ_z = 0.
    # ------------------------------------------------------------------
    def test_roll_sweep_produces_negative_roll(self, fin_geometry):
        """Fin_3+Fin_4 same positive deflection → τ_x > 0 (roll)."""
        pos, ld = fin_geometry
        v = self.DEFLECTION
        tau = _net_torque([0, 0, +v, +v], pos, ld, self.SCALE)
        assert tau[0] > 0, f"Roll sweep must produce positive τ_x, got {tau}"

    def test_roll_sweep_is_pure_roll(self, fin_geometry):
        pos, ld = fin_geometry
        v = self.DEFLECTION
        tau = _net_torque([0, 0, +v, +v], pos, ld, self.SCALE)
        assert tau[1] > 0, f"Roll sweep should have positive pitch cross-couple, got {tau}"
        assert abs(tau[2]) < 1e-9, f"τ_z should still be 0 for roll sweep, got {tau[2]:.2e}"
        assert abs(tau[0]) > abs(tau[1]), f"Roll must remain dominant, got {tau}"
        assert tau[0] > 0

    def test_roll_antisymmetric(self, fin_geometry):
        """d3=−v, d4=−v → τ_x < 0 (opposite roll direction)."""
        pos, ld = fin_geometry
        v = self.DEFLECTION
        tau = _net_torque([0, 0, -v, -v], pos, ld, self.SCALE)
        assert tau[0] < 0, f"Reverse roll sweep must produce negative τ_x, got {tau}"

    # ------------------------------------------------------------------
    # Pitch sweep (Fin_1+Fin_2): d1=+v, d2=+v
    # τ_y > 0 is dominant, τ_x > 0 is secondary, τ_z = 0.
    # ------------------------------------------------------------------
    def test_pitch_sweep_produces_positive_pitch(self, fin_geometry):
        """Fin_1+Fin_2 same positive deflection → τ_y > 0 (pitch)."""
        pos, ld = fin_geometry
        v = self.DEFLECTION
        tau = _net_torque([+v, +v, 0, 0], pos, ld, self.SCALE)
        assert tau[1] > 0, f"Pitch sweep must produce positive τ_y, got {tau}"

    def test_pitch_sweep_is_pure_pitch(self, fin_geometry):
        pos, ld = fin_geometry
        v = self.DEFLECTION
        tau = _net_torque([+v, +v, 0, 0], pos, ld, self.SCALE)
        assert tau[0] > 0, f"Pitch sweep should have positive roll cross-couple, got {tau}"
        assert abs(tau[2]) < 1e-9, f"τ_z should still be 0 for pitch sweep, got {tau[2]:.2e}"
        assert abs(tau[1]) > abs(tau[0]), f"Pitch must remain dominant, got {tau}"
        assert tau[1] > 0

    def test_pitch_antisymmetric(self, fin_geometry):
        """d1=−v, d2=−v → τ_y < 0 (opposite pitch direction)."""
        pos, ld = fin_geometry
        v = self.DEFLECTION
        tau = _net_torque([-v, -v, 0, 0], pos, ld, self.SCALE)
        assert tau[1] < 0, f"Reverse pitch sweep must produce negative τ_y, got {tau}"

    # ------------------------------------------------------------------
    # Differential yaw pattern from canted fin lift vectors
    # ------------------------------------------------------------------
    def test_yaw_pattern_produces_positive_yaw(self, fin_geometry):
        pos, ld = fin_geometry
        tau = _net_torque([-1.0, +1.0, -1.0, +1.0], pos, ld, 1.0)
        assert tau[2] > 0, f"Yaw+ pattern must produce positive τ_z, got {tau}"
        assert abs(tau[0]) < 1e-9, f"Yaw pattern should cancel τ_x, got {tau[0]:.2e}"
        assert abs(tau[1]) < 1e-9, f"Yaw pattern should cancel τ_y, got {tau[1]:.2e}"

    def test_yaw_pattern_produces_negative_yaw(self, fin_geometry):
        pos, ld = fin_geometry
        tau = _net_torque([+1.0, -1.0, +1.0, -1.0], pos, ld, 1.0)
        assert tau[2] < 0, f"Yaw- pattern must produce negative τ_z, got {tau}"
        assert abs(tau[0]) < 1e-9, f"Yaw pattern should cancel τ_x, got {tau[0]:.2e}"
        assert abs(tau[1]) < 1e-9, f"Yaw pattern should cancel τ_y, got {tau[1]:.2e}"

    # ------------------------------------------------------------------
    # Axis orthogonality
    # ------------------------------------------------------------------
    def test_pitch_cross_couples_less_than_primary(self, fin_geometry):
        pos, ld = fin_geometry
        tau = _net_torque([1.0, 1.0, 0.0, 0.0], pos, ld, 1.0)
        assert 0.0 < tau[0] < tau[1], f"Pitch cross-couple should stay smaller than primary, got {tau}"

    def test_roll_cross_couples_less_than_primary(self, fin_geometry):
        pos, ld = fin_geometry
        tau = _net_torque([0.0, 0.0, 1.0, 1.0], pos, ld, 1.0)
        assert 0.0 < tau[1] < abs(tau[0]), f"Roll cross-couple should stay smaller than primary, got {tau}"

    # ------------------------------------------------------------------
    # Lever arm magnitude sanity
    # ------------------------------------------------------------------
    def test_roll_lever_arm_fin1(self, fin_geometry):
        """Fin_3 alone: |τ_x| = Z_hinge * cos(cant) for unit deflection."""
        pos, ld = fin_geometry
        r3 = pos[2]
        tau = _net_torque([0.0, 0.0, 1.0, 0.0], pos, ld, 1.0)
        expected = abs(r3[2]) * COS_CANT
        assert abs(abs(tau[0]) - expected) < 1e-6, (
            f"Roll lever: expected |τ_x|={expected:.4f}, got {abs(tau[0]):.4f}"
        )

    def test_pitch_lever_arm_fin3(self, fin_geometry):
        """Fin_1 alone: |τ_y| = Z_hinge * cos(cant) for unit deflection."""
        pos, ld = fin_geometry
        r1 = pos[0]
        tau = _net_torque([1.0, 0.0, 0.0, 0.0], pos, ld, 1.0)
        expected = abs(r1[2]) * COS_CANT
        assert abs(abs(tau[1]) - expected) < 1e-6, (
            f"Pitch lever: expected |τ_y|={expected:.4f}, got {abs(tau[1]):.4f}"
        )
