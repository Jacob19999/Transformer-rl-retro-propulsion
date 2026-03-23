"""Unit tests for the fin model (Stage 5)."""

import numpy as np

from simulation.dynamics.fin_model import FinModel, FinModelConfig


def _make_fin_config(
    Cl_alpha: float = 2 * np.pi,
    Cd0: float = 0.01,
    AR: float = 0.80,
    stall_angle: float = np.deg2rad(15),
    max_deflection: float = np.deg2rad(20),
    planform_area: float = 0.002,
    V_exhaust_nominal: float = 70.0,
    omega_fan_max: float = 9948.0,
) -> FinModelConfig:
    """Create a minimal FinModelConfig for testing."""
    fins_config = (
        {"position": [0, 0.04, 0.12], "lift_direction": [1, 0, 0], "drag_direction": [0, 0, 1]},
        {"position": [0, -0.04, 0.12], "lift_direction": [1, 0, 0], "drag_direction": [0, 0, 1]},
        {"position": [0.04, 0, 0.12], "lift_direction": [0, 1, 0], "drag_direction": [0, 0, 1]},
        {"position": [-0.04, 0, 0.12], "lift_direction": [0, 1, 0], "drag_direction": [0, 0, 1]},
    )
    return FinModelConfig(
        Cl_alpha=Cl_alpha,
        Cd0=Cd0,
        AR=AR,
        stall_angle=stall_angle,
        max_deflection=max_deflection,
        planform_area=planform_area,
        V_exhaust_nominal=V_exhaust_nominal,
        omega_fan_max=omega_fan_max,
        exhaust_velocity_ratio=True,
        fins_config=fins_config,
    )


def test_delta_zero_produces_zero_lift() -> None:
    """δ=0 on all fins → zero lift; only parasitic drag contributes to F_z."""
    cfg = _make_fin_config()
    com = np.array([0.0, 0.0, 0.05])
    model = FinModel(cfg, com)

    delta = np.zeros(4)
    omega_fan = 5000.0  # rad/s
    rho = 1.225

    F_fins, tau_fins = model.compute(delta, omega_fan, rho=rho)

    # With δ=0: alpha_eff=0, C_L=0. Lift component is zero for all fins.
    # Only Cd0 (parasitic) contributes to drag (along +z per config).
    # So F_x and F_y (lift directions) should be zero.
    assert np.allclose(F_fins[0], 0.0, atol=1e-14), "F_x should be zero (no lift)"
    assert np.allclose(F_fins[1], 0.0, atol=1e-14), "F_y should be zero (no lift)"
    # F_z has small parasitic drag from Cd0
    assert F_fins[2] > 0, "F_z should be positive (drag opposes exhaust, +z in body)"


def test_delta_10deg_cl_approximately_correct() -> None:
    """δ=10° → C_L in expected range (tanh soft-clamp gives ~0.96, linear would be ~1.097)."""
    cfg = _make_fin_config()
    com = np.array([0.0, 0.0, 0.05])
    model = FinModel(cfg, com)

    delta_10deg = np.deg2rad(10)
    delta = np.array([delta_10deg, 0.0, 0.0, 0.0])  # only fin 0 deflected
    omega_fan = 9948.0  # max RPM for full V_exhaust
    rho = 1.225

    F_fins, _ = model.compute(delta, omega_fan, rho=rho)

    # Fin 0 has lift_direction [1,0,0], so lift produces F_x.
    # C_L with tanh: stall * tanh(10°/15°) * Cl_alpha ≈ 0.96
    # q_dyn = 0.5 * 1.225 * 70^2 * 0.002 = 6.0025
    # F_lift_x = q_dyn * C_L * 1 (x-component of lift) ≈ 6 * 0.96 ≈ 5.76
    assert F_fins[0] > 0, "Fin 0 deflects +10°, should produce +x lift"
    # Sanity: C_L should be in [0.9, 1.2] (tanh gives ~0.96, linear 1.097)
    q_dyn = 0.5 * rho * (70.0 ** 2) * 0.002
    C_L_implied = F_fins[0] / q_dyn
    assert 0.9 <= C_L_implied <= 1.15, f"C_L should be ~0.96-1.1, got {C_L_implied}"


def test_symmetric_deflection_zero_net_lateral_force() -> None:
    """Symmetric deflection (delta = [d,-d,e,-e]) → zero net lateral (lift) force."""
    cfg = _make_fin_config()
    com = np.array([0.0, 0.0, 0.05])
    model = FinModel(cfg, com)

    # Fins 0,1 share lift_direction [1,0,0]; fins 2,3 share [0,1,0].
    # delta = [0.1, -0.1, 0.1, -0.1] → lift cancels in x and y.
    delta = np.array([0.1, -0.1, 0.1, -0.1])
    omega_fan = 5000.0
    rho = 1.225

    F_fins, tau_fins = model.compute(delta, omega_fan, rho=rho)

    assert np.allclose(F_fins[0], 0.0, atol=1e-10), "F_x should cancel (symmetric lift pair)"
    assert np.allclose(F_fins[1], 0.0, atol=1e-10), "F_y should cancel (symmetric lift pair)"
    # F_z from drag (all fins contribute) should be positive
    assert F_fins[2] > 0


def test_mechanical_clamp_limits_deflection() -> None:
    """Mechanical clamp clips delta to ±delta_max (±20°)."""
    cfg = _make_fin_config(max_deflection=np.deg2rad(20))
    com = np.array([0.0, 0.0, 0.05])
    model = FinModel(cfg, com)

    # Command 30° — should be clamped to 20°
    delta_over = np.array([np.deg2rad(30), 0.0, 0.0, 0.0])
    delta_clamped_ref = np.array([np.deg2rad(20), 0.0, 0.0, 0.0])

    omega_fan = 9948.0
    rho = 1.225

    F_over, _ = model.compute(delta_over, omega_fan, rho=rho)
    F_clamped, _ = model.compute(delta_clamped_ref, omega_fan, rho=rho)

    # Same result since 30° is clamped to 20°
    assert np.allclose(F_over, F_clamped, atol=1e-12)


def test_zero_omega_zero_exhaust_velocity() -> None:
    """omega_fan=0 with exhaust_velocity_ratio → V_e=0 → zero fin forces."""
    cfg = _make_fin_config()
    com = np.array([0.0, 0.0, 0.05])
    model = FinModel(cfg, com)

    delta = np.array([0.1, 0.1, 0.1, 0.1])
    omega_fan = 0.0
    rho = 1.225

    F_fins, tau_fins = model.compute(delta, omega_fan, rho=rho)

    assert np.allclose(F_fins, np.zeros(3), atol=1e-14)
    assert np.allclose(tau_fins, np.zeros(3), atol=1e-14)


def test_torque_arm_correct() -> None:
    """Torque should scale with (r_fin - com); offset CoM changes tau."""
    cfg = _make_fin_config()
    com1 = np.array([0.0, 0.0, 0.0])
    com2 = np.array([0.0, 0.0, 0.10])  # CoM closer to fins

    model1 = FinModel(cfg, com1)
    model2 = FinModel(cfg, com2)

    delta = np.array([0.1, 0.0, 0.0, 0.0])  # fin 0 only
    omega_fan = 5000.0
    rho = 1.225

    _, tau1 = model1.compute(delta, omega_fan, rho=rho)
    _, tau2 = model2.compute(delta, omega_fan, rho=rho)

    # Same force, different moment arms → different torques
    assert not np.allclose(tau1, tau2), "Different CoM should give different torques"


def test_from_config() -> None:
    """FinModel.from_config builds from vehicle YAML sections."""
    fins = {
        "Cl_alpha": 6.283,
        "Cd0": 0.01,
        "AR": 0.80,
        "stall_angle": 0.262,
        "max_deflection": 0.349,
        "planform_area": 0.002,
        "V_exhaust_nominal": 70.0,
        "exhaust_velocity_ratio": True,
        "fins_config": [
            {"position": [0, 0.04, 0.12], "lift_direction": [1, 0, 0], "drag_direction": [0, 0, 1]},
            {"position": [0, -0.04, 0.12], "lift_direction": [1, 0, 0], "drag_direction": [0, 0, 1]},
            {"position": [0.04, 0, 0.12], "lift_direction": [0, 1, 0], "drag_direction": [0, 0, 1]},
            {"position": [-0.04, 0, 0.12], "lift_direction": [0, 1, 0], "drag_direction": [0, 0, 1]},
        ],
    }
    edf = {"max_omega": 9948.0}
    com = np.array([0.0, 0.0, 0.05])

    model = FinModel.from_config(fins, edf, com)

    F_fins, tau_fins = model.compute(np.zeros(4), 5000.0, rho=1.225)
    assert F_fins.shape == (3,)
    assert tau_fins.shape == (3,)


# ── Deflection-rotated basis (cos/sin decomposition) tests ──────────────


def _make_fin_config_with_hinge(
    **kwargs: float,
) -> FinModelConfig:
    """Config with hinge_axis for each fin — enables force direction rotation."""
    defaults = dict(
        Cl_alpha=2 * np.pi,
        Cd0=0.01,
        AR=0.80,
        stall_angle=np.deg2rad(15),
        max_deflection=np.deg2rad(20),
        planform_area=0.002,
        V_exhaust_nominal=70.0,
        omega_fan_max=9948.0,
    )
    defaults.update(kwargs)
    fins_config = (
        {"position": [0, 0.04, 0.12], "lift_direction": [1, 0, 0], "drag_direction": [0, 0, 1], "hinge_axis": [0, 1, 0]},
        {"position": [0, -0.04, 0.12], "lift_direction": [1, 0, 0], "drag_direction": [0, 0, 1], "hinge_axis": [0, 1, 0]},
        {"position": [0.04, 0, 0.12], "lift_direction": [0, 1, 0], "drag_direction": [0, 0, 1], "hinge_axis": [1, 0, 0]},
        {"position": [-0.04, 0, 0.12], "lift_direction": [0, 1, 0], "drag_direction": [0, 0, 1], "hinge_axis": [1, 0, 0]},
    )
    return FinModelConfig(
        **defaults,
        exhaust_velocity_ratio=True,
        fins_config=fins_config,
    )


def test_rotation_zero_deflection_matches_fixed() -> None:
    """At δ=0, Rodrigues' rotation is identity — rotated model must match fixed."""
    com = np.array([0.0, 0.0, 0.05])
    fixed_model = FinModel(_make_fin_config(), com)
    rotated_model = FinModel(_make_fin_config_with_hinge(), com)

    delta = np.zeros(4)
    F_fixed, tau_fixed = fixed_model.compute(delta, 5000.0, rho=1.225)
    F_rot, tau_rot = rotated_model.compute(delta, 5000.0, rho=1.225)

    assert np.allclose(F_fixed, F_rot, atol=1e-12)
    assert np.allclose(tau_fixed, tau_rot, atol=1e-12)


def test_rotation_reduces_lateral_force_at_max_deflection() -> None:
    """At max deflection, cos(δ) < 1 reduces the useful lateral force component.

    Fin 0: lift_dir=[1,0,0], hinge=[0,1,0], drag_dir=[0,0,1].
    At δ=20°, the rotated lift direction has x-component = cos(20°) ≈ 0.94,
    so F_x (lateral) should be ~6% less than the fixed-direction model.
    """
    com = np.array([0.0, 0.0, 0.05])
    fixed_model = FinModel(_make_fin_config(), com)
    rotated_model = FinModel(_make_fin_config_with_hinge(), com)

    delta_20 = np.array([np.deg2rad(20), 0.0, 0.0, 0.0])
    omega = 9948.0
    rho = 1.225

    F_fixed, _ = fixed_model.compute(delta_20, omega, rho=rho)
    F_rot, _ = rotated_model.compute(delta_20, omega, rho=rho)

    # Lateral force (F_x from fin 0 lift) should be smaller with rotation
    assert F_rot[0] < F_fixed[0], "Rotation should reduce lateral component"
    # The ratio should be close to cos(20°) ≈ 0.94 (not exact because drag
    # direction also rotates, contributing a small +x component via sin(δ))
    ratio = F_rot[0] / F_fixed[0]
    assert 0.90 < ratio < 0.98, f"Expected ratio near cos(20°)=0.94, got {ratio}"


def test_rotation_increases_axial_force() -> None:
    """Rotation leaks lift into thrust-loss direction (body +z).

    With hinge rotation, F_z should increase compared to fixed-direction
    because lift·sin(δ) adds to the axial drag component.
    """
    com = np.array([0.0, 0.0, 0.05])
    fixed_model = FinModel(_make_fin_config(), com)
    rotated_model = FinModel(_make_fin_config_with_hinge(), com)

    delta_20 = np.array([np.deg2rad(20), 0.0, 0.0, 0.0])
    omega = 9948.0
    rho = 1.225

    F_fixed, _ = fixed_model.compute(delta_20, omega, rho=rho)
    F_rot, _ = rotated_model.compute(delta_20, omega, rho=rho)

    # F_z (axial / thrust loss) should be larger with rotation
    assert F_rot[2] > F_fixed[2], "Rotation should increase axial thrust loss"


def test_rotation_symmetric_deflection_still_cancels_lateral() -> None:
    """Symmetric opposite deflections with rotation still cancel lateral force.

    Fins 0 and 1 share hinge_axis [0,1,0] and lift_dir [1,0,0].
    delta = [+d, -d, 0, 0]: cos(d) and cos(-d) are equal, sin(d) and sin(-d)
    are opposite.  The x-components of the rotated lift are both scaled by
    cos(d) (same magnitude, same C_L sign ↔ opposite), so they still cancel.
    """
    com = np.array([0.0, 0.0, 0.05])
    model = FinModel(_make_fin_config_with_hinge(), com)

    delta = np.array([0.15, -0.15, 0.15, -0.15])
    F, _ = model.compute(delta, 5000.0, rho=1.225)

    assert np.allclose(F[0], 0.0, atol=1e-10), "F_x should cancel"
    assert np.allclose(F[1], 0.0, atol=1e-10), "F_y should cancel"
