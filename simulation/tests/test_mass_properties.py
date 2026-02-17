import numpy as np

from simulation.dynamics.mass_properties import MassProperties, compute_mass_properties


def test_single_cylinder_matches_analytic_inertia() -> None:
    prims = [
        {
            "name": "cyl",
            "shape": "cylinder",
            "mass": 2.0,
            "radius": 0.5,
            "height": 1.0,
            "position": [1.0, 2.0, 3.0],
            "surface_area": 0.25,
            "drag_facing": {"x": 0.1, "y": 0.2, "z": 0.3},
        }
    ]
    mp = compute_mass_properties(prims)

    assert mp.total_mass == 2.0
    assert np.allclose(mp.center_of_mass, np.array([1.0, 2.0, 3.0]))

    m, r, h = 2.0, 0.5, 1.0
    Ixx = (1.0 / 12.0) * m * (3.0 * r * r + h * h)
    Izz = 0.5 * m * r * r
    I_expected = np.diag([Ixx, Ixx, Izz])
    assert np.allclose(mp.inertia_tensor, I_expected)
    assert np.allclose(mp.inertia_tensor_inv, np.linalg.inv(I_expected))

    assert mp.total_surface_area == 0.25
    assert mp.projected_area_x == 0.1
    assert mp.projected_area_y == 0.2
    assert mp.projected_area_z == 0.3


def test_two_boxes_parallel_axis_hand_calc() -> None:
    prims = [
        {
            "name": "b1",
            "shape": "box",
            "mass": 1.0,
            "dimensions": [2.0, 1.0, 1.0],
            "position": [0.0, 0.0, 0.0],
        },
        {
            "name": "b2",
            "shape": "box",
            "mass": 3.0,
            "dimensions": [1.0, 1.0, 3.0],
            "position": [1.0, 0.0, 0.0],
        },
    ]

    mp = compute_mass_properties(prims)

    m1, m2 = 1.0, 3.0
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.0, 0.0, 0.0])
    m_total = m1 + m2
    com = (m1 * p1 + m2 * p2) / m_total
    assert np.allclose(mp.center_of_mass, com)

    def box_I(m: float, a: float, b: float, c: float) -> np.ndarray:
        return np.diag(
            [
                (1.0 / 12.0) * m * (b * b + c * c),
                (1.0 / 12.0) * m * (a * a + c * c),
                (1.0 / 12.0) * m * (a * a + b * b),
            ]
        )

    I1 = box_I(m1, 2.0, 1.0, 1.0)
    I2 = box_I(m2, 1.0, 1.0, 3.0)

    d1 = p1 - com
    d2 = p2 - com
    I1_par = m1 * ((d1 @ d1) * np.eye(3) - np.outer(d1, d1))
    I2_par = m2 * ((d2 @ d2) * np.eye(3) - np.outer(d2, d2))
    I_expected = I1 + I2 + I1_par + I2_par
    I_expected = 0.5 * (I_expected + I_expected.T)

    assert np.allclose(mp.inertia_tensor, I_expected)


def test_inertia_is_symmetric_positive_definite_and_diagonal_dominantish() -> None:
    prims = [
        {
            "name": "a",
            "shape": "sphere",
            "mass": 0.5,
            "radius": 0.2,
            "position": [0.1, -0.2, 0.3],
        },
        {
            "name": "b",
            "shape": "box",
            "mass": 1.0,
            "dimensions": [0.3, 0.1, 0.2],
            "position": [-0.2, 0.0, 0.0],
        },
    ]
    mp = compute_mass_properties(prims)
    I = mp.inertia_tensor

    assert np.allclose(I, I.T)
    eig = np.linalg.eigvalsh(I)
    assert np.all(eig > 0.0)

    # "Diagonal dominance check" (soft): diagonals should exceed off-diagonal magnitude.
    off_diag_max = np.max(np.abs(I - np.diag(np.diag(I))))
    assert np.min(np.diag(I)) >= off_diag_max


def test_from_cad_override_sets_fields_and_inverts_inertia() -> None:
    cad = {
        "total_mass": 3.5,
        "center_of_mass": [0.01, -0.02, 0.03],
        "inertia_tensor": [[0.2, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.4]],
        "total_surface_area": 1.23,
    }
    mp = MassProperties.from_cad(cad)
    assert mp.total_mass == 3.5
    assert np.allclose(mp.center_of_mass, np.array([0.01, -0.02, 0.03]))
    assert np.allclose(mp.inertia_tensor_inv, np.linalg.inv(np.array(cad["inertia_tensor"])))
    assert mp.total_surface_area == 1.23
    assert mp.projected_area_x == 0.0
    assert mp.projected_area_y == 0.0
    assert mp.projected_area_z == 0.0


def test_mass_randomization_is_bounded_and_reproducible_with_seed() -> None:
    prims = [
        {
            "name": "fixed",
            "shape": "sphere",
            "mass": 1.0,
            "radius": 0.1,
            "position": [0.0, 0.0, 0.0],
        },
        {
            "name": "payload_variable",
            "shape": "sphere",
            "mass": 2.0,
            "radius": 0.1,
            "position": [0.1, 0.0, 0.0],
            "randomize_mass": 0.10,
        },
    ]

    mp1 = compute_mass_properties(prims, rng=np.random.default_rng(123))
    mp2 = compute_mass_properties(prims, rng=np.random.default_rng(123))
    assert np.isclose(mp1.total_mass, mp2.total_mass)
    assert np.allclose(mp1.center_of_mass, mp2.center_of_mass)
    assert np.allclose(mp1.inertia_tensor, mp2.inertia_tensor)

    # Total mass bounds: 1.0 + 2.0*(1±0.10)
    assert (1.0 + 2.0 * 0.9) <= mp1.total_mass <= (1.0 + 2.0 * 1.1)

