"""Mass-property frame validation tests."""

from tvc_env.asset.mass_properties import MassProperties, validate_mass_properties


def test_com_is_compared_after_frd_to_isaac_conversion():
    props = MassProperties(
        mass_kg=3.1,
        com_offset=[0.0, 0.0, 0.01],
        inertia={"Ixx": 0.05, "Iyy": 0.05, "Izz": 0.02},
    )
    config = {
        "total_mass": 3.1,
        "body_com_offset": [0.0, 0.0, -0.01],
        "inertia_tensor": {"Ixx": 0.05, "Iyy": 0.05, "Izz": 0.02},
    }
    assert validate_mass_properties(props, config) == []


def test_com_sign_mismatch_is_reported():
    props = MassProperties(
        mass_kg=3.1,
        com_offset=[0.0, 0.0, 0.01],
        inertia={"Ixx": 0.05, "Iyy": 0.05, "Izz": 0.02},
    )
    config = {
        "total_mass": 3.1,
        "body_com_offset": [0.0, 0.0, 0.01],
        "inertia_tensor": {"Ixx": 0.05, "Iyy": 0.05, "Izz": 0.02},
    }
    issues = validate_mass_properties(props, config)
    assert any("COM mismatch [z]" in issue for issue in issues)
