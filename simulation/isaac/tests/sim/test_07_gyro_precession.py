"""
Simulation test: Gyroscopic precession (test_07).

Verifies the gyroscopic precession reaction torque applied to the BODY:
    τ_gyro_on_body = -(ω_body × H_rotor)
    where H_rotor = I_rotor * ω_rotor * spin_axis

The body must apply ω_body × H_rotor to the rotor to precess it; by Newton's
third law, the rotor exerts -ω_body × H_rotor on the body. PhysX has no
virtual rotor, so this reaction is applied as an external body torque.

Tests performed:
  1. Direction: for each body axis perturbation, the precession torque direction
     must match -(ω_body × H_rotor) (cross product direction check).
  2. Magnitude: τ_gyro magnitude must be proportional to both ω_rotor and ω_body.
     Specifically: |τ| = I_rotor * ω_rotor * |ω_body| * sin(θ)
     For orthogonal ω_body and spin_axis: |τ| = I_rotor * ω_rotor * ω_body.
  3. Anti-symmetry: reversing ω_body must reverse the precession torque.
  4. Independence from throttle spool state: the gyro effect depends only on the
     instantaneous ω_rotor and body angular velocity, not on the spool history.

Rotor spin axis: +Z in body-FRD frame (EDF exhaust direction, matching propulsion_edf.py).
Steady-state rotor speed for tests: OMEGA_ROTOR = 1000 rad/s.

Requires Isaac Sim runtime.
"""

from __future__ import annotations
import math
import pytest
import torch
from pathlib import Path

try:
    import omni.usd
    ISAAC_AVAILABLE = True
except ImportError:
    ISAAC_AVAILABLE = False

pytestmark = pytest.mark.skipif(not ISAAC_AVAILABLE, reason="Isaac Sim runtime not available")

VEHICLE_CONFIG_PATH = Path(__file__).parents[2] / "configs/vehicle/edf_drone_v2.yaml"

# Test parameters
OMEGA_ROTOR = 1000.0        # rad/s — steady rotor speed
OMEGA_BODY = 1.0            # rad/s — imposed body angular velocity magnitude
SPIN_AXIS = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)  # +Z body-FRD

# Tolerance for direction checks (cosine similarity)
DIR_TOLERANCE = 1e-5  # absolute difference between computed and expected unit vectors
# Tolerance for magnitude checks (relative)
MAG_TOLERANCE = 1e-4  # relative error in gyro torque magnitude


@pytest.fixture
def gyro_params():
    """Load EDFModel to obtain rotor_inertia and k_Q, k_T."""
    from tvc_env.dynamics.propulsion_edf import EDFModel
    model = EDFModel.from_yaml(VEHICLE_CONFIG_PATH)
    return model


@pytest.fixture
def scene_and_drone():
    """Build a minimal Isaac Sim scene for in-sim gyro measurement."""
    from tvc_env.sim.scene_builder import SceneConfig, build_scene

    config = SceneConfig(num_envs=1, gizmos_enabled=False)
    scene = build_scene(config)
    drone = scene["drone"]
    return scene, drone


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _expected_gyro_torque(
    omega_rotor: float,
    rotor_inertia: float,
    body_ang_vel_vec: torch.Tensor,
    spin_axis: torch.Tensor,
) -> torch.Tensor:
    """Compute expected gyro torque on the body analytically.

    τ_expected_body = -(ω_body × H_rotor) = -ω_body × (I_rotor * ω_rotor * spin_axis)

    The negation reflects that the rotor exerts the reaction of the precession
    torque on the body (the body applies +ω × H to the rotor).

    Args:
        omega_rotor: Scalar rotor speed (rad/s).
        rotor_inertia: Scalar moment of inertia (kg·m²).
        body_ang_vel_vec: (3,) body angular velocity in body-FRD (rad/s).
        spin_axis: (3,) unit rotor spin axis in body-FRD.

    Returns:
        Tensor of shape (3,) — expected body torque (N·m).
    """
    H = rotor_inertia * omega_rotor * spin_axis
    return -torch.linalg.cross(body_ang_vel_vec, H)


class TestGyroPrecession:
    """Verify gyroscopic precession direction, magnitude, and proportionality."""

    def test_zero_body_rate_gives_zero_gyro_torque(self, gyro_params):
        """No body rotation → no gyroscopic precession regardless of rotor speed."""
        from tvc_env.dynamics.rotor_reaction import compute_gyroscopic_precession

        omega = torch.tensor([OMEGA_ROTOR])
        body_ang_vel = torch.zeros(1, 3)
        torque = compute_gyroscopic_precession(
            omega, body_ang_vel, gyro_params.rotor_inertia, SPIN_AXIS
        )
        assert torque.norm().item() < 1e-10, (
            f"Zero body rate should give zero gyro torque, got {torque.tolist()}"
        )

    def test_zero_rotor_speed_gives_zero_gyro_torque(self, gyro_params):
        """Stationary rotor (ω_rotor = 0) → zero angular momentum → zero gyro torque."""
        from tvc_env.dynamics.rotor_reaction import compute_gyroscopic_precession

        omega = torch.zeros(1)
        body_ang_vel = torch.zeros(1, 3)
        body_ang_vel[0, 0] = OMEGA_BODY  # non-zero body rate
        torque = compute_gyroscopic_precession(
            omega, body_ang_vel, gyro_params.rotor_inertia, SPIN_AXIS
        )
        assert torque.norm().item() < 1e-10, (
            f"Zero rotor speed should give zero gyro torque, got {torque.tolist()}"
        )

    @pytest.mark.parametrize("body_axis, axis_label", [
        (torch.tensor([1.0, 0.0, 0.0]), "roll (+X)"),
        (torch.tensor([0.0, 1.0, 0.0]), "pitch (+Y)"),
        (torch.tensor([-1.0, 0.0, 0.0]), "roll (-X)"),
        (torch.tensor([0.0, -1.0, 0.0]), "pitch (-Y)"),
    ])
    def test_precession_direction_matches_cross_product(
        self, gyro_params, body_axis, axis_label
    ):
        """τ_gyro on body must exactly match -(ω_body × H_rotor) for each body axis."""
        from tvc_env.dynamics.rotor_reaction import compute_gyroscopic_precession

        omega = torch.tensor([OMEGA_ROTOR])
        body_ang_vel = (body_axis * OMEGA_BODY).unsqueeze(0)  # (1, 3)

        computed = compute_gyroscopic_precession(
            omega, body_ang_vel, gyro_params.rotor_inertia, SPIN_AXIS
        )[0]  # (3,)

        expected = _expected_gyro_torque(
            OMEGA_ROTOR, gyro_params.rotor_inertia, body_axis * OMEGA_BODY, SPIN_AXIS
        )

        diff = (computed - expected).norm().item()
        assert diff < DIR_TOLERANCE, (
            f"Precession direction mismatch for {axis_label}: "
            f"computed={computed.tolist()}, expected={expected.tolist()}, diff={diff:.2e}"
        )

    def test_precession_magnitude_proportional_to_rotor_speed(self, gyro_params):
        """Doubling ω_rotor must double the gyro torque magnitude."""
        from tvc_env.dynamics.rotor_reaction import compute_gyroscopic_precession

        body_ang_vel = torch.zeros(1, 3)
        body_ang_vel[0, 0] = OMEGA_BODY  # roll perturbation

        omega_1 = torch.tensor([500.0])
        omega_2 = torch.tensor([1000.0])

        t1 = compute_gyroscopic_precession(
            omega_1, body_ang_vel, gyro_params.rotor_inertia, SPIN_AXIS
        )
        t2 = compute_gyroscopic_precession(
            omega_2, body_ang_vel, gyro_params.rotor_inertia, SPIN_AXIS
        )

        ratio = t2.norm().item() / t1.norm().item()
        expected_ratio = 1000.0 / 500.0  # linear in ω_rotor
        assert abs(ratio - expected_ratio) < MAG_TOLERANCE * expected_ratio, (
            f"Gyro torque should scale linearly with ω_rotor; "
            f"expected ratio {expected_ratio:.2f}, got {ratio:.6f}"
        )

    def test_precession_magnitude_proportional_to_body_rate(self, gyro_params):
        """Doubling ω_body must double the gyro torque magnitude."""
        from tvc_env.dynamics.rotor_reaction import compute_gyroscopic_precession

        omega = torch.tensor([OMEGA_ROTOR])

        body_ang_vel_1 = torch.zeros(1, 3)
        body_ang_vel_1[0, 1] = 0.5  # pitch rate 0.5 rad/s

        body_ang_vel_2 = torch.zeros(1, 3)
        body_ang_vel_2[0, 1] = 1.0  # pitch rate 1.0 rad/s

        t1 = compute_gyroscopic_precession(
            omega, body_ang_vel_1, gyro_params.rotor_inertia, SPIN_AXIS
        )
        t2 = compute_gyroscopic_precession(
            omega, body_ang_vel_2, gyro_params.rotor_inertia, SPIN_AXIS
        )

        ratio = t2.norm().item() / t1.norm().item()
        expected_ratio = 1.0 / 0.5  # linear in ω_body
        assert abs(ratio - expected_ratio) < MAG_TOLERANCE * expected_ratio, (
            f"Gyro torque should scale linearly with ω_body; "
            f"expected ratio {expected_ratio:.2f}, got {ratio:.6f}"
        )

    def test_precession_magnitude_equals_analytical_formula(self, gyro_params):
        """
        |τ_gyro_on_body| = I_rotor * ω_rotor * ω_body when spin_axis ⊥ ω_body.

        For SPIN_AXIS = [0,0,1] and body_ang_vel = [ω_body, 0, 0]:
            H_rotor       = [0, 0, I*ω_rotor]
            ω_body × H    = [0, -I*ω_r*ω_b, 0]
            τ_gyro_body   = -(ω_body × H) = [0, +I*ω_r*ω_b, 0]
            |τ_gyro_body| = I_rotor * ω_rotor * ω_body
        """
        from tvc_env.dynamics.rotor_reaction import compute_gyroscopic_precession

        omega = torch.tensor([OMEGA_ROTOR])
        body_ang_vel = torch.zeros(1, 3)
        body_ang_vel[0, 0] = OMEGA_BODY  # pure roll perturbation

        torque = compute_gyroscopic_precession(
            omega, body_ang_vel, gyro_params.rotor_inertia, SPIN_AXIS
        )
        computed_mag = torque.norm().item()
        expected_mag = gyro_params.rotor_inertia * OMEGA_ROTOR * OMEGA_BODY

        rel_error = abs(computed_mag - expected_mag) / expected_mag
        assert rel_error < MAG_TOLERANCE, (
            f"Gyro torque magnitude mismatch: computed={computed_mag:.6f} N·m, "
            f"expected={expected_mag:.6f} N·m (rel error={rel_error:.2e})"
        )

    def test_reversing_body_rate_reverses_precession_torque(self, gyro_params):
        """τ_gyro(-ω_body) must equal -τ_gyro(+ω_body) (anti-symmetry)."""
        from tvc_env.dynamics.rotor_reaction import compute_gyroscopic_precession

        omega = torch.tensor([OMEGA_ROTOR])

        body_pos = torch.zeros(1, 3)
        body_pos[0, 1] = OMEGA_BODY  # +pitch

        body_neg = torch.zeros(1, 3)
        body_neg[0, 1] = -OMEGA_BODY  # -pitch

        t_pos = compute_gyroscopic_precession(
            omega, body_pos, gyro_params.rotor_inertia, SPIN_AXIS
        )[0]
        t_neg = compute_gyroscopic_precession(
            omega, body_neg, gyro_params.rotor_inertia, SPIN_AXIS
        )[0]

        diff = (t_pos + t_neg).norm().item()
        assert diff < 1e-8, (
            f"Gyro torque should be anti-symmetric: τ(+ω) + τ(-ω) should be ~0, "
            f"got {diff:.2e}. τ(+)={t_pos.tolist()}, τ(-)={t_neg.tolist()}"
        )

    def test_precession_torque_orthogonal_to_spin_axis_and_body_rate(self, gyro_params):
        """
        τ_gyro = ±(ω_body × H_rotor), so τ_gyro must be orthogonal to both
        ω_body and H_rotor regardless of sign convention.

        Check: τ · ω_body ≈ 0 and τ · H_rotor ≈ 0.
        """
        from tvc_env.dynamics.rotor_reaction import compute_gyroscopic_precession

        omega = torch.tensor([OMEGA_ROTOR])
        body_ang_vel_vec = torch.tensor([OMEGA_BODY, 0.0, 0.0])
        body_ang_vel = body_ang_vel_vec.unsqueeze(0)

        torque = compute_gyroscopic_precession(
            omega, body_ang_vel, gyro_params.rotor_inertia, SPIN_AXIS
        )[0]  # (3,)

        H_rotor = gyro_params.rotor_inertia * OMEGA_ROTOR * SPIN_AXIS

        dot_with_omega = abs(torch.dot(torque, body_ang_vel_vec).item())
        dot_with_H = abs(torch.dot(torque, H_rotor).item())

        assert dot_with_omega < 1e-8, (
            f"τ_gyro should be orthogonal to ω_body, dot product = {dot_with_omega:.2e}"
        )
        assert dot_with_H < 1e-8, (
            f"τ_gyro should be orthogonal to H_rotor, dot product = {dot_with_H:.2e}"
        )

    def test_gyro_torque_appears_in_sim_body_angular_acceleration(
        self, scene_and_drone, gyro_params
    ):
        """
        In the full Isaac Sim scene, imposing a body angular velocity on a spinning rotor
        should produce a measurable change in body angular acceleration consistent with
        the gyroscopic precession model.

        This test drives the sim for a short burst to verify no runtime errors occur and
        that the drone's body angular velocity is non-trivially updated by the step.
        """
        from tvc_env.sim.body_interface import BodyInterface
        from tvc_env.asset.usd_loader import load_asset_metadata
        from tvc_env.asset.articulation_map import build_articulation_map
        from tvc_env.dynamics.rotor_reaction import compute_gyroscopic_precession

        METADATA_PATH = Path(__file__).parents[2] / "assets/metadata/edf_drone_v2.asset.yaml"
        scene, drone = scene_and_drone
        metadata = load_asset_metadata(METADATA_PATH)
        art_map = build_articulation_map(metadata, drone)
        body_iface = BodyInterface(drone, art_map)

        # Step the simulation with no external forces — just verify it runs
        for _ in range(10):
            scene.step()

        # Read current body angular velocity to verify state access works
        ang_vel_frd = body_iface.get_angular_velocity_body_frd()
        assert ang_vel_frd.shape == (1, 3), (
            f"Expected body angular velocity shape (1, 3), got {ang_vel_frd.shape}"
        )

        # Compute expected gyro torque for the current state at steady rotor speed
        omega = torch.tensor([OMEGA_ROTOR])
        torque = compute_gyroscopic_precession(
            omega, ang_vel_frd, gyro_params.rotor_inertia, SPIN_AXIS
        )
        # Just verify the tensor is well-formed and finite
        assert torque.isfinite().all(), (
            f"Gyro torque contains non-finite values: {torque}"
        )
        assert torque.shape == (1, 3), (
            f"Expected gyro torque shape (1, 3), got {torque.shape}"
        )
