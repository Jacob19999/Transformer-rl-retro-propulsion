"""Unit tests for the aerodynamic drag model (Stage 4)."""

import numpy as np

from simulation.dynamics.aero_model import AeroModel, AeroModelConfig
from simulation.dynamics.mass_properties import MassProperties, compute_mass_properties
from simulation.dynamics.quaternion_utils import quat_to_dcm, euler_to_quat


def _make_test_mass_props(
    projected_x: float = 0.01,
    projected_y: float = 0.01,
    projected_z: float = 0.01,
    com: np.ndarray | None = None,
) -> MassProperties:
    """Create MassProperties for tests with specified projected areas."""
    return MassProperties(
        total_mass=1.0,
        center_of_mass=com if com is not None else np.array([0.0, 0.0, 0.0]),
        inertia_tensor=np.eye(3),
        inertia_tensor_inv=np.eye(3),
        total_surface_area=0.1,
        projected_area_x=projected_x,
        projected_area_y=projected_y,
        projected_area_z=projected_z,
    )


def test_zero_velocity_produces_zero_drag() -> None:
    """Zero v_rel (v_b = wind in body frame) → F_aero = 0, tau_aero = 0."""
    cfg = AeroModelConfig(Cd=0.7, r_cp=np.array([0, 0, 0.05]), A_proj=0.01, compute_directional_drag=True)
    mass_props = _make_test_mass_props(0.01, 0.01, 0.01)
    model = AeroModel(cfg, mass_props)

    v_b = np.array([5.0, 0.0, 0.0])
    q = euler_to_quat(0.0, 0.0, 0.0)
    R = quat_to_dcm(q)
    v_wind_inertial = R @ v_b  # wind exactly cancels v_b in body frame
    rho = 1.225

    F_aero, tau_aero = model.compute(v_b, R, v_wind_inertial, rho=rho)

    assert np.allclose(F_aero, np.zeros(3), atol=1e-14)
    assert np.allclose(tau_aero, np.zeros(3), atol=1e-14)


def test_zero_speed_produces_zero_drag() -> None:
    """v_b = 0 and v_wind = 0 → v_rel = 0 → zero drag."""
    cfg = AeroModelConfig(Cd=0.7, r_cp=np.array([0, 0, 0.05]), A_proj=0.01, compute_directional_drag=False)
    mass_props = _make_test_mass_props(0.01, 0.01, 0.01)
    model = AeroModel(cfg, mass_props)

    v_b = np.array([0.0, 0.0, 0.0])
    R = np.eye(3)
    v_wind = np.array([0.0, 0.0, 0.0])
    rho = 1.225

    F_aero, tau_aero = model.compute(v_b, R, v_wind, rho=rho)

    assert np.allclose(F_aero, np.zeros(3), atol=1e-14)
    assert np.allclose(tau_aero, np.zeros(3), atol=1e-14)


def test_known_v_rel_analytical_drag_match() -> None:
    """Known v_rel → F_aero magnitude = 0.5 * rho * v^2 * Cd * A_eff, direction opposes v_rel."""
    Cd = 0.7
    A_proj = 0.02
    cfg = AeroModelConfig(Cd=Cd, r_cp=np.array([0, 0, 0]), A_proj=A_proj, compute_directional_drag=False)
    mass_props = _make_test_mass_props(A_proj, A_proj, A_proj)
    model = AeroModel(cfg, mass_props)

    v_rel = np.array([10.0, 0.0, 0.0])  # 10 m/s along body x
    R = np.eye(3)
    v_wind = np.zeros(3)
    v_b = v_rel
    rho = 1.225

    F_aero, tau_aero = model.compute(v_b, R, v_wind, rho=rho)

    expected_mag = 0.5 * rho * 10.0 * 10.0 * Cd * A_proj
    actual_mag = np.linalg.norm(F_aero)
    assert np.isclose(actual_mag, expected_mag, rtol=1e-10)

    F_dir = F_aero / actual_mag
    v_dir = v_rel / np.linalg.norm(v_rel)
    assert np.allclose(F_dir, -v_dir, atol=1e-10), "Force opposes velocity"

    # r_cp = com → zero torque
    assert np.allclose(tau_aero, np.zeros(3), atol=1e-14)


def test_directional_drag_different_areas() -> None:
    """Directional drag uses per-axis projected areas weighted by velocity direction."""
    proj_x, proj_y, proj_z = 0.01, 0.02, 0.03  # different per-axis areas
    cfg = AeroModelConfig(Cd=1.0, r_cp=np.array([0, 0, 0]), A_proj=0.01, compute_directional_drag=True)
    mass_props = _make_test_mass_props(proj_x, proj_y, proj_z)
    model = AeroModel(cfg, mass_props)

    v_rel = np.array([10.0, 0.0, 0.0])  # flow along x
    R = np.eye(3)
    v_wind = np.zeros(3)
    v_b = v_rel
    rho = 1.0

    F_aero, _ = model.compute(v_b, R, v_wind, rho=rho)

    # v_hat = [1,0,0] → A_eff = proj_x
    expected_mag = 0.5 * rho * 100.0 * 1.0 * proj_x
    assert np.isclose(np.linalg.norm(F_aero), expected_mag, rtol=1e-10)

    # Flow along z
    v_rel_z = np.array([0.0, 0.0, 10.0])
    F_aero_z, _ = model.compute(v_rel_z, R, v_wind, rho=rho)
    expected_mag_z = 0.5 * rho * 100.0 * 1.0 * proj_z
    assert np.isclose(np.linalg.norm(F_aero_z), expected_mag_z, rtol=1e-10)


def test_aero_torque_when_r_cp_offset_from_com() -> None:
    """When r_cp ≠ com, tau_aero = (r_cp - com) × F_aero is non-zero."""
    r_cp = np.array([0.0, 0.0, 0.05])
    com = np.array([0.0, 0.0, 0.0])
    cfg = AeroModelConfig(Cd=0.7, r_cp=r_cp, A_proj=0.02, compute_directional_drag=False)
    mass_props = _make_test_mass_props(0.02, 0.02, 0.02, com=com)
    model = AeroModel(cfg, mass_props)

    v_rel = np.array([5.0, 0.0, 0.0])
    R = np.eye(3)
    v_wind = np.zeros(3)
    rho = 1.225

    F_aero, tau_aero = model.compute(v_rel, R, v_wind, rho=rho)

    r_offset = r_cp - com
    expected_tau = np.cross(r_offset, F_aero)
    assert np.allclose(tau_aero, expected_tau, rtol=1e-12)


def test_wind_rotation_check() -> None:
    """Wind in inertial frame is correctly rotated to body frame: v_rel = v_b - R.T @ v_wind."""
    cfg = AeroModelConfig(Cd=0.7, r_cp=np.array([0, 0, 0]), A_proj=0.01, compute_directional_drag=False)
    mass_props = _make_test_mass_props(0.01, 0.01, 0.01)
    model = AeroModel(cfg, mass_props)

    v_b = np.array([0.0, 0.0, 0.0])
    q = euler_to_quat(0.0, np.pi / 2, 0.0)  # 90° pitch
    R = quat_to_dcm(q)
    # Wind in inertial: 10 m/s along inertial +x (North in NED)
    v_wind_inertial = np.array([10.0, 0.0, 0.0])
    # R.T @ v_wind = body-frame wind; for pitch 90°, inertial x → body -z
    v_wind_body = R.T @ v_wind_inertial
    rho = 1.225

    F_aero, _ = model.compute(v_b, R, v_wind_inertial, rho=rho)

    # v_rel = 0 - v_wind_body = -v_wind_body
    v_rel = -v_wind_body
    expected_mag = 0.5 * rho * np.linalg.norm(v_rel) ** 2 * cfg.Cd * cfg.A_proj
    actual_mag = np.linalg.norm(F_aero)
    assert np.isclose(actual_mag, expected_mag, rtol=1e-10)

    # Force opposes v_rel (i.e., along v_wind_body)
    F_dir = F_aero / actual_mag
    v_rel_dir = v_rel / np.linalg.norm(v_rel)
    assert np.allclose(F_dir, -v_rel_dir, atol=1e-10)


def test_integration_with_compute_mass_properties() -> None:
    """AeroModel works with MassProperties from compute_mass_properties."""
    primitives = [
        {
            "name": "test_cyl",
            "shape": "cylinder",
            "mass": 1.0,
            "radius": 0.05,
            "height": 0.2,
            "position": [0, 0, 0],
            "surface_area": 0.1,
            "drag_facing": {"x": 0.01, "y": 0.01, "z": 0.00636},
        }
    ]
    mass_props = compute_mass_properties(primitives)
    cfg = AeroModelConfig.from_config(
        {"Cd": 0.8, "r_cp": [0, 0, 0.02], "A_proj": 0.01, "compute_directional_drag": True}
    )
    model = AeroModel(cfg, mass_props)

    v_b = np.array([3.0, 2.0, 1.0])
    R = np.eye(3)
    v_wind = np.zeros(3)
    rho = 1.225

    F_aero, tau_aero = model.compute(v_b, R, v_wind, rho=rho)

    speed = np.linalg.norm(v_b)
    assert speed > 0
    assert np.linalg.norm(F_aero) > 0
    # Force opposes velocity
    assert np.dot(F_aero, v_b) < 0
