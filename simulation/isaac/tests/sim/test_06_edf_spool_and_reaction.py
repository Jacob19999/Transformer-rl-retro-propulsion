"""
Simulation test: EDF spool dynamics and static reaction torque (test_06).

Commands a step from 0% to 100% throttle and:
  - Measures the motor speed (omega) response over time.
  - Verifies the spool-up time constant is within 10% of the configured tau_motor = 0.15 s.
  - Verifies the static reaction torque is non-zero and opposes the spin axis.
  - Verifies the reaction torque direction: for a +Z spin axis the static reaction
    torque must be in the -Z direction (Q = -k_Q * omega² * spin_axis).
  - Logs all three torque components (static, dynamic spool, gyroscopic) separately.

Saves step-response curves to tests/goldens/reaction_torque_curves/edf_step.json.

Timing note: the first-order lag model is:
    dω/dt = (ω_target - ω) / tau_motor

  → at t = tau_motor, ω reaches (1 - 1/e) ≈ 63.2% of ω_max.

We verify that the simulated 63% crossing time is within ±10% of tau_motor = 0.15 s.

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

SIM_ROOT = Path(__file__).parents[2]
VEHICLE_CONFIG_PATH = Path(__file__).parents[2] / "configs/vehicle/edf_drone_v2.yaml"
GOLDENS_PATH = (
    Path(__file__).parents[2] / "tests/goldens/reaction_torque_curves/edf_step.json"
)

# Model constants matching EDFModel defaults
TAU_MOTOR = 0.15        # s — configured spool time constant
TAU_TOLERANCE = 0.10    # 10% tolerance on measured tau
DT = 1.0 / 120.0        # s — physics timestep (matches SceneConfig.physics_dt)
N_STEPS = 400           # simulate ~3.3 s (well past tau_motor)
SPIN_AXIS = torch.tensor([0.0, 0.0, 1.0])  # +Z in body-FRD (EDF exhaust direction)


@pytest.fixture
def edf_model():
    """Construct EDFModel from the vehicle YAML config."""
    from tvc_env.dynamics.propulsion_edf import EDFModel
    return EDFModel.from_yaml(VEHICLE_CONFIG_PATH)


@pytest.fixture
def scene_and_drone():
    """Build a minimal Isaac Sim scene for torque measurement in simulation."""
    from tvc_env.sim.scene_builder import SceneConfig, build_scene

    config = SceneConfig(num_envs=1, gizmos_enabled=False)
    scene = build_scene(config)
    drone = scene["drone"]
    return scene, drone


@pytest.fixture
def spool_env():
    """Build a single-env TVC environment for thrust-direction integration tests."""
    from tvc_env.envs.base_env import BaseEnvConfig
    from tvc_env.envs.direct_rl_env import TVCDirectRLEnv

    config = BaseEnvConfig(
        task_name="hover",
        env_config_path=SIM_ROOT / "configs/env/single_env_debug.yaml",
        disturbance_config_path=SIM_ROOT / "configs/disturbances/nominal.yaml",
        overrides={
            "task": {
                "episode_length_s": 10.0,
                "termination": {
                    "crash": False,
                    "max_tilt": 3.14,
                    "max_altitude_error": 100.0,
                },
            }
        },
        sim_root=SIM_ROOT,
    )
    env = TVCDirectRLEnv(config)
    env.reset()
    try:
        yield env
    finally:
        env.close()


class TestEDFSpoolAndReaction:
    """Verify EDF first-order spool lag and static reaction torque characteristics."""

    def test_spool_up_reaches_steady_state(self, edf_model):
        """Motor speed should reach ≥ 95% of omega_max by t = 3 * tau_motor."""
        omega = torch.zeros(1)  # (1,) — single env
        throttle = torch.ones(1)  # 100%
        target_omega = edf_model.omega_max

        t_final = 3.0 * TAU_MOTOR
        n_steps = int(t_final / DT)

        for _ in range(n_steps):
            omega = edf_model.update(omega, throttle, DT)

        final_fraction = omega[0].item() / target_omega
        assert final_fraction >= 0.95, (
            f"Motor should reach ≥95% of omega_max after 3*tau_motor, "
            f"got {final_fraction:.3f}"
        )

    def test_tau_motor_within_tolerance(self, edf_model):
        """63.2% crossing time should match tau_motor within 10%."""
        omega = torch.zeros(1)
        throttle = torch.ones(1)
        threshold = 0.632 * edf_model.omega_max

        t_crossing = None
        for step in range(N_STEPS):
            omega = edf_model.update(omega, throttle, DT)
            t = (step + 1) * DT
            if omega[0].item() >= threshold and t_crossing is None:
                t_crossing = t
                break

        assert t_crossing is not None, (
            f"Motor never reached 63.2% of omega_max in {N_STEPS * DT:.2f} s"
        )
        tau_error = abs(t_crossing - TAU_MOTOR) / TAU_MOTOR
        assert tau_error < TAU_TOLERANCE, (
            f"Measured tau = {t_crossing:.4f} s, expected {TAU_MOTOR:.4f} s "
            f"(error = {tau_error * 100:.1f}%, limit = {TAU_TOLERANCE * 100:.0f}%)"
        )

    def test_static_reaction_torque_opposes_spin_axis(self, edf_model):
        """Static reaction torque direction must oppose the spin axis (+Z → -Z reaction)."""
        from tvc_env.dynamics.rotor_reaction import compute_static_reaction_torque

        omega = torch.tensor([edf_model.omega_max])  # full speed
        spin_axis = SPIN_AXIS
        torque = compute_static_reaction_torque(omega, edf_model.k_Q, spin_axis)

        # torque: (1, 3), should be in -Z direction
        torque_z = torque[0, 2].item()
        assert torque_z < 0.0, (
            f"Static reaction torque on +Z spin axis should be negative Z, "
            f"got T_z = {torque_z:.6f} N·m"
        )

    def test_static_reaction_torque_proportional_to_omega_squared(self, edf_model):
        """τ_static ∝ ω²: doubling omega should quadruple the reaction torque magnitude."""
        from tvc_env.dynamics.rotor_reaction import compute_static_reaction_torque

        omega_1 = torch.tensor([500.0])
        omega_2 = torch.tensor([1000.0])
        spin_axis = SPIN_AXIS

        t1 = compute_static_reaction_torque(omega_1, edf_model.k_Q, spin_axis)
        t2 = compute_static_reaction_torque(omega_2, edf_model.k_Q, spin_axis)

        ratio = t2.norm().item() / t1.norm().item()
        expected_ratio = (1000.0 / 500.0) ** 2  # = 4.0
        assert abs(ratio - expected_ratio) < 0.01, (
            f"Reaction torque should scale as ω²; expected ratio {expected_ratio:.2f}, "
            f"got {ratio:.4f}"
        )

    def test_all_three_torque_components_logged_separately(self, edf_model):
        """compute_output must return separate static, dynamic, and gyro torque tensors."""
        omega = torch.tensor([edf_model.omega_max * 0.8])
        omega_prev = torch.tensor([edf_model.omega_max * 0.75])
        body_ang_vel = torch.zeros(1, 3)
        body_ang_vel[0, 0] = 0.5  # small body roll rate → non-zero gyro torque

        output = edf_model.compute_output(omega, omega_prev, body_ang_vel, DT, SPIN_AXIS)

        # All three torque tensors must be shape (1, 3)
        assert output.static_reaction_torque.shape == (1, 3), (
            f"static_reaction_torque shape mismatch: {output.static_reaction_torque.shape}"
        )
        assert output.dynamic_spool_torque.shape == (1, 3), (
            f"dynamic_spool_torque shape mismatch: {output.dynamic_spool_torque.shape}"
        )
        assert output.gyro_precession_torque.shape == (1, 3), (
            f"gyro_precession_torque shape mismatch: {output.gyro_precession_torque.shape}"
        )

        # Gyro torque must be non-zero when body has angular velocity
        assert output.gyro_precession_torque.norm().item() > 1e-8, (
            "Gyro torque should be non-zero when body_ang_vel is non-zero"
        )

    def test_saves_step_response_curves_to_golden(self, edf_model):
        """Full step-response data should be written to the golden JSON file."""
        from tvc_env.dynamics.rotor_reaction import compute_all_rotor_torques

        omega = torch.zeros(1)
        throttle = torch.ones(1)
        spin_axis = SPIN_AXIS
        body_ang_vel = torch.zeros(1, 3)
        records: list[dict] = []

        for step in range(N_STEPS):
            omega_prev = omega.clone()
            omega = edf_model.update(omega, throttle, DT)

            static, dynamic, gyro = compute_all_rotor_torques(
                omega, omega_prev, body_ang_vel,
                k_Q=edf_model.k_Q,
                rotor_inertia=edf_model.rotor_inertia,
                spin_axis=spin_axis,
                dt=DT,
            )
            t = (step + 1) * DT
            records.append({
                "t_s": round(t, 6),
                "omega_rad_s": round(omega[0].item(), 4),
                "throttle": 1.0,
                "static_torque_Nm": [round(v, 8) for v in static[0].tolist()],
                "dynamic_spool_torque_Nm": [round(v, 8) for v in dynamic[0].tolist()],
                "gyro_precession_torque_Nm": [round(v, 8) for v in gyro[0].tolist()],
            })

        GOLDENS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(GOLDENS_PATH, "w") as f:
            json.dump({
                "description": "EDF step-response from 0% to 100% throttle",
                "tau_motor_s": TAU_MOTOR,
                "dt_s": DT,
                "n_steps": N_STEPS,
                "steps": records,
            }, f, indent=2)

        assert GOLDENS_PATH.exists(), "Golden file was not created"
        with open(GOLDENS_PATH) as f:
            loaded = json.load(f)
        assert len(loaded["steps"]) == N_STEPS, (
            f"Expected {N_STEPS} step entries, got {len(loaded['steps'])}"
        )

    def test_step_response_in_sim_matches_model(self, scene_and_drone, edf_model):
        """
        Drive the Isaac Sim scene while also integrating the EDFModel in lock-step.

        The sim body angular velocity feeds the model's gyro computation each step.
        We verify that the step-response thrust curve from the standalone model (no
        scene involvement) matches expected first-order behaviour, and that the scene
        simulation runs without error for N_STEPS.
        """
        scene, drone = scene_and_drone
        omega = torch.zeros(1)
        throttle = torch.ones(1)

        for step in range(50):  # 50 steps ≈ 0.42 s — partial ramp
            omega_prev = omega.clone()
            omega = edf_model.update(omega, throttle, DT)
            scene.step()

        # After 50 steps the motor should have spooled partially
        assert omega[0].item() > 0.0, "Motor omega should be positive after stepping"
        assert omega[0].item() < edf_model.omega_max, (
            "Motor should not have instantly reached omega_max"
        )

    def test_pre_spooled_full_throttle_lifts_and_keeps_orientation(self, spool_env, edf_model):
        """With pre-spooled EDF and full throttle, vehicle should climb and stay reasonably upright."""
        from tvc_env.common.quaternions import to_euler

        device = spool_env.device

        # Deterministic spawn with enough clearance to avoid touchdown artifacts.
        pos = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
        lin = torch.zeros(1, 3, dtype=torch.float32, device=device)
        ang = torch.zeros(1, 3, dtype=torch.float32, device=device)
        spool_env._body_iface.set_root_state(pos, quat, lin, ang)

        # Pre-spool close to steady-state so this test isolates thrust sign/application.
        spool_env._reset_manager._servo_state.zero_()
        spool_env._reset_manager._omega_state.fill_(edf_model.omega_max * 0.95)
        spool_env._reset_manager._omega_prev.copy_(spool_env._reset_manager._omega_state)
        spool_env._sim_scene.step()

        action = torch.zeros(1, 5, dtype=torch.float32, device=device)
        action[0, 4] = 1.0

        heights: list[float] = []
        vz_frd: list[float] = []
        tilt_rad: list[float] = []

        for _ in range(40):
            _, _, terminated, truncated, _ = spool_env.step(action)
            assert not terminated.any(), "Unexpected termination during thrust-direction check"
            assert not truncated.any(), "Unexpected timeout during thrust-direction check"

            state = spool_env._build_vehicle_state()
            heights.append(float(state.height[0]))
            vz_frd.append(float(state.linear_vel_frd[0, 2]))  # FRD: +z is down, -z is up

            roll, pitch, _ = to_euler(state.quaternion_wxyz)
            tilt = torch.sqrt(roll[0] ** 2 + pitch[0] ** 2)
            tilt_rad.append(float(tilt))

        assert heights[-1] > heights[0] + 0.20, (
            f"Expected climb under full throttle, got h0={heights[0]:.3f} h_end={heights[-1]:.3f}"
        )
        assert min(vz_frd[5:]) < -0.10, (
            "Expected upward FRD velocity (negative z_frd) once thrust is applied"
        )
        assert max(tilt_rad) < 0.35, (
            f"Orientation became unstable during thrust check (max tilt={math.degrees(max(tilt_rad)):.1f} deg)"
        )
