"""
test_fin_torque_mapping.py — Pure-Python (no Isaac Sim) tests for the fin
axis-torque mapping used in diag_thrust_fin_wiggle.py.

Physical body frame (FRD: +X=fwd/nose, +Y=right, +Z=down)
  Confirmed from Isaac Sim 3D model top-down view.

Fin physical layout:
  Fin_1 at FRD +X (fwd/nose), Fin_2 at FRD -X (aft)   → τ_y → PITCH
  Fin_3 at FRD -Y (left),     Fin_4 at FRD +Y (right)  → τ_x → ROLL

Torque formula (τ = Σ r_i × F_i, F_i = scale·d_i·lift_dir_i):
  τ_y (pitch) = +Z·k·(d1+d2)   [Fin_1+Fin_2 at ±X, lift +X]
  τ_x (roll)  = −Z·k·(d3+d4)   [Fin_3+Fin_4 at ±Y, lift +Y]
  τ_z (yaw)   = 0               [no fin-based yaw with this geometry;
                                  yaw authority comes from EDF anti-torque]

Correct sweep patterns:
  Roll : d3=+v, d4=+v, d1=0,  d2=0   → τ_x = −0.28kv, τ_y=τ_z=0
  Pitch: d1=+v, d2=+v, d3=0,  d4=0   → τ_y = +0.28kv, τ_x=τ_z=0

Historical notes:
  Old wrong yaw sweep d1=−v,d2=+v,d3=+v,d4=−v → τ=0 with correct geometry.
  Old YAML had fin positions 90° rotated (fin_1_right at FRD +Y instead of FRD +X).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_fin_geometry() -> tuple[list[np.ndarray], list[np.ndarray]]:
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from simulation.config_loader import load_config
    from simulation.isaac.usd.parts_registry import load_fin_specs

    cfg = load_config(str(REPO_ROOT / "simulation" / "configs" / "default_vehicle.yaml"))
    specs = load_fin_specs(cfg.get("vehicle", cfg))
    positions  = [np.array(s.hinge_pos_frd, dtype=float) for s in specs]
    lift_dirs  = [np.array(s.lift_direction,  dtype=float) for s in specs]
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
      Fin_1(fwd) at FRD +X, Fin_2(aft) at FRD -X  →  hinge X, lift +X → τ_y (PITCH)
      Fin_3(left) at FRD -Y, Fin_4(right) at FRD +Y  →  hinge Y, lift +Y → τ_x (ROLL)

    Note: τ_z = 0 for all fin combinations with this geometry.
          Yaw authority comes from EDF anti-torque, not fins.
    """

    SCALE = 1.0
    DEFLECTION = 0.5

    # ------------------------------------------------------------------
    # Roll sweep (Fin_3+Fin_4): d3=+v, d4=+v
    # τ_x = −Z·k·(d3+d4) < 0  →  ROLL (rotation about nose/+X axis)
    # τ_y = 0, τ_z = 0
    # ------------------------------------------------------------------
    def test_roll_sweep_produces_negative_roll(self, fin_geometry):
        """Fin_3+Fin_4 same positive deflection → τ_x < 0 (roll)."""
        pos, ld = fin_geometry
        v = self.DEFLECTION
        tau = _net_torque([0, 0, +v, +v], pos, ld, self.SCALE)
        assert tau[0] < 0, f"Roll sweep must produce negative τ_x, got {tau}"

    def test_roll_sweep_is_pure_roll(self, fin_geometry):
        pos, ld = fin_geometry
        v = self.DEFLECTION
        tau = _net_torque([0, 0, +v, +v], pos, ld, self.SCALE)
        assert abs(tau[1]) < 1e-9, f"τ_y should be 0 for roll sweep, got {tau[1]:.2e}"
        assert abs(tau[2]) < 1e-9, f"τ_z should be 0 for roll sweep, got {tau[2]:.2e}"
        assert tau[0] < 0

    def test_roll_antisymmetric(self, fin_geometry):
        """d3=−v, d4=−v → τ_x > 0 (opposite roll direction)."""
        pos, ld = fin_geometry
        v = self.DEFLECTION
        tau = _net_torque([0, 0, -v, -v], pos, ld, self.SCALE)
        assert tau[0] > 0, f"Reverse roll sweep must produce positive τ_x, got {tau}"

    # ------------------------------------------------------------------
    # Pitch sweep (Fin_1+Fin_2): d1=+v, d2=+v
    # τ_y = +Z·k·(d1+d2) > 0  →  PITCH (rotation about right/+Y axis)
    # τ_x = 0, τ_z = 0
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
        assert abs(tau[0]) < 1e-9, f"τ_x should be 0 for pitch sweep, got {tau[0]:.2e}"
        assert abs(tau[2]) < 1e-9, f"τ_z should be 0 for pitch sweep, got {tau[2]:.2e}"
        assert tau[1] > 0

    def test_pitch_antisymmetric(self, fin_geometry):
        """d1=−v, d2=−v → τ_y < 0 (opposite pitch direction)."""
        pos, ld = fin_geometry
        v = self.DEFLECTION
        tau = _net_torque([-v, -v, 0, 0], pos, ld, self.SCALE)
        assert tau[1] < 0, f"Reverse pitch sweep must produce negative τ_y, got {tau}"

    # ------------------------------------------------------------------
    # No fin-based yaw torque with FRD geometry
    # ------------------------------------------------------------------
    def test_no_fin_yaw_single_fins(self, fin_geometry):
        """Each fin alone produces τ_z = 0."""
        pos, ld = fin_geometry
        for i, name in enumerate(["Fin_1", "Fin_2", "Fin_3", "Fin_4"]):
            deflections = [0.0, 0.0, 0.0, 0.0]
            deflections[i] = 1.0
            tau = _net_torque(deflections, pos, ld, 1.0)
            assert abs(tau[2]) < 1e-9, (
                f"{name} alone must not produce yaw: τ_z={tau[2]:.2e}"
            )

    def test_no_fin_yaw_all_combinations(self, fin_geometry):
        """Several fin combinations all give τ_z = 0."""
        pos, ld = fin_geometry
        combos = [
            [1.0, 1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0, -1.0],
        ]
        for combo in combos:
            tau = _net_torque(combo, pos, ld, 1.0)
            assert abs(tau[2]) < 1e-9, (
                f"Combo {combo} must give τ_z=0, got {tau[2]:.2e}"
            )

    # ------------------------------------------------------------------
    # Axis orthogonality
    # ------------------------------------------------------------------
    def test_pitch_does_not_affect_roll(self, fin_geometry):
        pos, ld = fin_geometry
        tau = _net_torque([1.0, 1.0, 0.0, 0.0], pos, ld, 1.0)
        assert abs(tau[0]) < 1e-9, f"Pitch must not produce roll: τ_x={tau[0]:.2e}"

    def test_roll_does_not_affect_pitch(self, fin_geometry):
        pos, ld = fin_geometry
        tau = _net_torque([0.0, 0.0, 1.0, 1.0], pos, ld, 1.0)
        assert abs(tau[1]) < 1e-9, f"Roll must not produce pitch: τ_y={tau[1]:.2e}"

    # ------------------------------------------------------------------
    # Lever arm magnitude sanity
    # ------------------------------------------------------------------
    def test_pitch_lever_arm_fin1(self, fin_geometry):
        """Fin_1 alone: τ_y = Z_hinge (r1[2]) × deflection."""
        pos, ld = fin_geometry
        r1 = pos[0]
        tau = _net_torque([1.0, 0.0, 0.0, 0.0], pos, ld, 1.0)
        expected = abs(r1[2])   # Z_hinge × scale=1 × deflection=1
        assert abs(abs(tau[1]) - expected) < 1e-6, (
            f"Pitch lever: expected τ_y={expected:.4f}, got {abs(tau[1]):.4f}"
        )

    def test_roll_lever_arm_fin3(self, fin_geometry):
        """Fin_3 alone: |τ_x| = Z_hinge (r3[2]) × deflection."""
        pos, ld = fin_geometry
        r3 = pos[2]
        tau = _net_torque([0.0, 0.0, 1.0, 0.0], pos, ld, 1.0)
        expected = abs(r3[2])   # Z_hinge × scale=1 × deflection=1
        assert abs(abs(tau[0]) - expected) < 1e-6, (
            f"Roll lever: expected |τ_x|={expected:.4f}, got {abs(tau[0]):.4f}"
        )
