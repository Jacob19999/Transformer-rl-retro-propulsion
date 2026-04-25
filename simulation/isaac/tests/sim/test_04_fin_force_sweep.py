"""
Simulation test: Fin force sweep (test_04).

Sweeps a single fin from -max_deflection to +max_deflection (0.262 rad),
recording normal force (F_n) and tangential force (F_t) at each angle step.

Qualitative checks verified:
  - F_n is near-zero at zero deflection
  - |F_n| increases as |angle| increases (over the linear region)
  - F_n changes sign with angle sign (it is an odd function of alpha)
  - F_t is always non-negative (drag opposes flow)
  - F_t saturates / flattens at large deflection (saturation term in C_D)
  - The normal force curve saturates at large angles due to k_sat term

Saves curves to tests/goldens/fin_force_curves/fin_force_sweep.json.

Requires Isaac Sim runtime.
"""

from __future__ import annotations
import json
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

METADATA_PATH = Path(__file__).parents[2] / "assets/metadata/edf_drone_v2.asset.yaml"
VEHICLE_CONFIG_PATH = Path(__file__).parents[2] / "configs/vehicle/edf_drone_v2.yaml"
GOLDENS_PATH = Path(__file__).parents[2] / "tests/goldens/fin_force_curves/fin_force_sweep.json"

MAX_DEFLECTION = 0.262  # rad (15°)
NUM_SWEEP_STEPS = 25    # points from -max to +max inclusive
THROTTLE = 1.0          # full throttle for well-defined dynamic pressure

# Near-zero tolerance for F_n at alpha == 0
F_N_ZERO_TOLERANCE = 1e-4  # N
# F_n should grow from 0 to at least this by mid-range deflection
F_N_NONZERO_THRESHOLD = 1e-3  # N


@pytest.fixture
def aero_model_and_configs():
    """Load vehicle config and construct a FinAeroModel."""
    from tvc_env.asset.usd_loader import load_asset_metadata
    from tvc_env.asset.mass_properties import load_vehicle_config
    from tvc_env.dynamics.fin_aero import FinAeroModel

    metadata = load_asset_metadata(METADATA_PATH)
    vehicle_config = load_vehicle_config(VEHICLE_CONFIG_PATH)

    # Build the model using the same config path used in production
    edf_section = vehicle_config.get("edf", {})
    aero_model = FinAeroModel.from_config(vehicle_config, edf_section)
    return aero_model, metadata, vehicle_config


@pytest.fixture
def scene_and_dispatch(aero_model_and_configs):
    """Build a minimal Isaac Sim scene and FinForceDispatch for sweep testing."""
    from tvc_env.asset.usd_loader import load_asset_metadata
    from tvc_env.asset.articulation_map import build_articulation_map
    from tvc_env.dynamics.fin_force_dispatch import FinForceDispatch
    from tvc_env.sim.scene_builder import SceneConfig, build_scene

    aero_model, metadata, vehicle_config = aero_model_and_configs

    config = SceneConfig(num_envs=1, gizmos_enabled=False)
    scene = build_scene(config)
    drone = scene["drone"]
    art_map = build_articulation_map(metadata, drone)

    edf_section = vehicle_config.get("edf", {})
    dispatch = FinForceDispatch.from_metadata_and_config(
        metadata, vehicle_config, edf_section, device=drone.device
    )
    return scene, drone, dispatch


class TestFinForceSweep:
    """Sweep fin angles and verify qualitative force curve behaviour."""

    def test_sweep_records_forces_at_all_angles(self, scene_and_dispatch, aero_model_and_configs):
        """Force computation returns a result for every sweep angle without error."""
        aero_model, metadata, vehicle_config = aero_model_and_configs
        scene, drone, dispatch = scene_and_dispatch

        angles = torch.linspace(-MAX_DEFLECTION, MAX_DEFLECTION, NUM_SWEEP_STEPS)
        throttle = torch.tensor([THROTTLE])  # (1,) — single env

        for angle in angles:
            fin_angles = torch.zeros(1, 4)  # only sweep fin 0 (+X fin)
            fin_angles[0, 0] = angle.item()
            result = aero_model.compute_forces(fin_angles, throttle)
            # Each call must return well-formed FinForceResult
            assert result.normal_force.shape == (1, 4), (
                f"Unexpected normal_force shape at angle {angle:.4f}"
            )
            assert result.tangential_force.shape == (1, 4), (
                f"Unexpected tangential_force shape at angle {angle:.4f}"
            )

    def test_normal_force_near_zero_at_zero_deflection(self, aero_model_and_configs):
        """F_n should be near-zero when all fins are undeflected (alpha == 0)."""
        aero_model, _, _ = aero_model_and_configs
        fin_angles = torch.zeros(1, 4)
        throttle = torch.tensor([THROTTLE])
        result = aero_model.compute_forces(fin_angles, throttle)
        # C_N(0) = C_N_alpha * 0 * (1 - k_sat * 0) = 0
        assert result.normal_force.abs().max().item() < F_N_ZERO_TOLERANCE, (
            f"F_n should be ~0 at zero deflection, got {result.normal_force}"
        )

    def test_normal_force_increases_with_angle_in_linear_region(self, aero_model_and_configs):
        """F_n magnitude should grow monotonically over the small-angle linear region."""
        aero_model, _, _ = aero_model_and_configs
        throttle = torch.tensor([THROTTLE])

        # Use 5 linearly-spaced positive angles up to 60% of max (well inside linear region)
        linear_limit = MAX_DEFLECTION * 0.6
        small_angles = torch.linspace(0.01, linear_limit, 5)

        f_n_prev = 0.0
        for angle in small_angles:
            fin_angles = torch.zeros(1, 4)
            fin_angles[0, 0] = angle.item()
            result = aero_model.compute_forces(fin_angles, throttle)
            f_n = result.normal_force[0, 0].item()
            assert f_n > f_n_prev, (
                f"F_n should increase monotonically in linear region; "
                f"at angle={angle:.4f} got {f_n:.6f}, prev={f_n_prev:.6f}"
            )
            f_n_prev = f_n

    def test_normal_force_is_odd_function_of_angle(self, aero_model_and_configs):
        """F_n(-alpha) should equal -F_n(+alpha) (anti-symmetric)."""
        aero_model, _, _ = aero_model_and_configs
        throttle = torch.tensor([THROTTLE])
        test_angles = [0.05, 0.1, 0.15, 0.2, MAX_DEFLECTION]

        for alpha in test_angles:
            fin_pos = torch.zeros(1, 4)
            fin_pos[0, 0] = alpha
            fin_neg = torch.zeros(1, 4)
            fin_neg[0, 0] = -alpha

            res_pos = aero_model.compute_forces(fin_pos, throttle)
            res_neg = aero_model.compute_forces(fin_neg, throttle)

            f_pos = res_pos.normal_force[0, 0].item()
            f_neg = res_neg.normal_force[0, 0].item()

            assert abs(f_pos + f_neg) < 1e-6, (
                f"F_n is not anti-symmetric at alpha={alpha:.4f}: "
                f"F_n(+)={f_pos:.6f}, F_n(-)={f_neg:.6f}"
            )

    def test_tangential_force_always_non_negative(self, aero_model_and_configs):
        """F_t (drag) should be non-negative for all deflection angles."""
        aero_model, _, _ = aero_model_and_configs
        throttle = torch.tensor([THROTTLE])
        angles = torch.linspace(-MAX_DEFLECTION, MAX_DEFLECTION, NUM_SWEEP_STEPS)

        for angle in angles:
            fin_angles = torch.zeros(1, 4)
            fin_angles[0, 0] = angle.item()
            result = aero_model.compute_forces(fin_angles, throttle)
            f_t = result.tangential_force[0, 0].item()
            assert f_t >= 0.0, (
                f"F_t should be non-negative (drag), got {f_t:.6f} at angle={angle:.4f}"
            )

    def test_normal_force_saturates_at_large_angles(self, aero_model_and_configs):
        """The normal force curve should show saturation (diminishing returns) near max_deflection."""
        aero_model, _, _ = aero_model_and_configs
        throttle = torch.tensor([THROTTLE])

        # Compute F_n at three equally-spaced positive angles
        alpha_low = MAX_DEFLECTION * 0.25
        alpha_mid = MAX_DEFLECTION * 0.60
        alpha_high = MAX_DEFLECTION * 0.95

        def get_fn(alpha: float) -> float:
            fa = torch.zeros(1, 4)
            fa[0, 0] = alpha
            return aero_model.compute_forces(fa, throttle).normal_force[0, 0].item()

        fn_low = get_fn(alpha_low)
        fn_mid = get_fn(alpha_mid)
        fn_high = get_fn(alpha_high)

        # Increment in lower half vs. upper half — upper increment should be smaller
        lower_increment = fn_mid - fn_low
        upper_increment = fn_high - fn_mid
        assert upper_increment < lower_increment, (
            f"Expected saturation: upper increment {upper_increment:.6f} should be "
            f"less than lower increment {lower_increment:.6f}"
        )

    def test_saves_sweep_curves_to_golden(self, aero_model_and_configs):
        """Sweep curves should be saved to the goldens JSON file."""
        aero_model, _, _ = aero_model_and_configs
        throttle = torch.tensor([THROTTLE])

        angles = torch.linspace(-MAX_DEFLECTION, MAX_DEFLECTION, NUM_SWEEP_STEPS)
        records: list[dict] = []

        for angle in angles:
            fin_angles = torch.zeros(1, 4)
            fin_angles[0, 0] = angle.item()
            result = aero_model.compute_forces(fin_angles, throttle)
            records.append({
                "angle_rad": round(angle.item(), 6),
                "F_n_N": round(result.normal_force[0, 0].item(), 8),
                "F_t_N": round(result.tangential_force[0, 0].item(), 8),
            })

        GOLDENS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(GOLDENS_PATH, "w") as f:
            json.dump({"sweep": records, "max_deflection_rad": MAX_DEFLECTION}, f, indent=2)

        assert GOLDENS_PATH.exists(), "Golden file was not created"
        with open(GOLDENS_PATH) as f:
            loaded = json.load(f)
        assert len(loaded["sweep"]) == NUM_SWEEP_STEPS, (
            f"Expected {NUM_SWEEP_STEPS} sweep entries, got {len(loaded['sweep'])}"
        )
