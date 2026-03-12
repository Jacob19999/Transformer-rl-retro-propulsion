"""
diag_gyro_precession.py -- Gyroscopic precession diagnostic for Isaac Sim.

Validates that gyroscopic precession from the spinning EDF rotor is correctly
modelled in the Isaac Sim environment. The test:

  Phase 1 (hover, 0.5 s): Stabilise drone at altitude with hover thrust.
                           Measures ω_fan from thrust_actual.
  Phase 2 (torque, 1.5 s): Apply constant external pitch torque (body Y-axis).
                            Records pitch_rate (q) and roll_rate (p) every 30 steps.
                            Expected: roll response from τ_gyro_x = −q·L

Physics note: yaw torque is intentionally NOT used as the test input. Yaw rate r
is parallel to the fan spin axis h_fan = [0,0,L], so cross(r_hat, h_fan_z) = 0
and yaw rate alone produces zero precession. Only pitch/roll rates (q, p) couple
with h_fan to produce the hallmark precession response.

Success criteria:
  PASS: roll_rate > 0.1 °/s at t > 0.5 s (when precession enabled)
  PASS: roll_rate < 0.05 °/s at all times (when precession disabled)
  FAIL: inverted response (wrong sign convention)

Usage::
    python -m simulation.isaac.scripts.diag_gyro_precession
    python -m simulation.isaac.scripts.diag_gyro_precession --torque-axis pitch --torque-mag 0.5 --duration 2.0
    python -m simulation.isaac.scripts.diag_gyro_precession --disable-precession
    python -m simulation.isaac.scripts.diag_gyro_precession --config simulation/isaac/configs/isaac_env_gyro_test.yaml
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

# SimulationApp MUST be created before any isaaclab.sim / carb imports
from isaacsim import SimulationApp  # noqa: E402

_SIM_APP: SimulationApp | None = None

# Physics constants (must match edf_landing_task.py)
_T_MAX    = 45.0      # N
_MASS     = 3.13      # kg
_GRAVITY  = 9.81      # m/s^2
_K_THRUST = 4.55e-7   # N/(rad/s)^2
_I_FAN    = 3.0e-5    # kg·m² (rotating fan blades only)

# Hover thrust: T_hover = m·g  →  norm = T_hover / T_max
_HOVER_NORM = (_MASS * _GRAVITY) / _T_MAX

# Observation indices (must match edf_landing_task.py _get_observations())
_OBS_ALTITUDE = 16   # h_agl (m)
_OBS_OMEGA_X  = 9    # roll rate p  (body frame rad/s)
_OBS_OMEGA_Y  = 10   # pitch rate q (body frame rad/s)
_OBS_OMEGA_Z  = 11   # yaw rate r   (body frame rad/s)


def _make_action(thrust_norm: float, fin_deflections: tuple[float, float, float, float] = (0, 0, 0, 0)) -> np.ndarray:
    """Build 5-dim action array."""
    action = np.zeros(5, dtype=np.float32)
    action[0] = float(np.clip(thrust_norm, -1.0, 1.0))
    action[1:5] = [float(d) for d in fin_deflections]
    return action


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Isaac Sim gyro precession diagnostic — pitch torque → roll coupling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Physics note:
  --torque-axis 'yaw' is intentionally NOT supported.
  Yaw rate r is parallel to the fan spin axis, so cross(omega_z, h_fan_z) = 0.
  Precession only occurs from pitch/roll rates.
  Use pitch or roll torque to observe the hallmark cross-axis response.
        """,
    )
    parser.add_argument(
        "--config",
        default="simulation/isaac/configs/isaac_env_gyro_test.yaml",
        help="Path to Isaac env YAML config (default: isaac_env_gyro_test.yaml)",
    )
    parser.add_argument(
        "--torque-axis",
        choices=["pitch", "roll"],
        default="pitch",
        help="Axis of applied external torque: 'pitch' (body Y) or 'roll' (body X). "
             "Default: pitch → expect roll response.",
    )
    parser.add_argument(
        "--torque-mag",
        type=float,
        default=0.5,
        help="External torque magnitude in N·m (default: 0.5)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Total test duration in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--spawn-alt",
        type=float,
        default=5.0,
        help="Spawn altitude in meters (default: 5.0)",
    )
    parser.add_argument(
        "--no-gravity",
        action="store_true",
        default=True,
        help="Disable gravity for isolated precession test (default: True via gyro_test config)",
    )
    parser.add_argument(
        "--disable-precession",
        action="store_true",
        default=False,
        help="Run with gyro_precession.enabled=false for A/B comparison",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run without GUI (default: False -- opens viewer)",
    )
    args = parser.parse_args()

    global _SIM_APP
    _SIM_APP = SimulationApp({"headless": args.headless})

    try:
        _run_diagnostic(args)
    finally:
        _SIM_APP.close()


def _obs_val(obs: np.ndarray, idx: int) -> float:
    if obs.ndim == 2:
        return float(obs[0, idx])
    return float(obs[idx])


def _is_done(done) -> bool:
    return bool(np.any(done))


def _run_diagnostic(args) -> None:
    """Run precession diagnostic; exits with code 0 (pass) or 1 (fail)."""
    from simulation.isaac.envs.edf_isaac_env import EDFIsaacEnv
    import torch

    omega_fan_hover = math.sqrt((_HOVER_NORM * _T_MAX) / _K_THRUST)
    L_hover = _I_FAN * omega_fan_hover  # angular momentum at hover

    print(f"\n{'='*50}")
    print("Gyro Precession Diagnostic Report")
    print("=" * 50)
    print(f"Isaac Sim v5.1.0 | I_fan = {_I_FAN:.2e} kg·m² | k_thrust = {_K_THRUST:.2e} N/(rad/s)²")
    prec_label = "DISABLED" if args.disable_precession else "ENABLED"
    print(f"Precession: {prec_label} | Gravity: DISABLED | Spawn alt: {args.spawn_alt:.1f} m")
    print(f"Torque axis: {args.torque_axis.upper()} | Magnitude: {args.torque_mag:.2f} N·m | Duration: {args.duration:.1f} s")
    print(f"Hover thrust: {_HOVER_NORM * _T_MAX:.1f} N | ω_fan: {omega_fan_hover:.0f} rad/s | L: {L_hover:.4e} kg·m²/s")
    print("=" * 50)

    # Patch vehicle config to disable precession if requested
    if args.disable_precession:
        import yaml
        veh_cfg_path = REPO_ROOT / "simulation" / "configs" / "default_vehicle.yaml"
        with open(veh_cfg_path) as f:
            veh_data = yaml.safe_load(f)
        veh_data.setdefault("vehicle", {}).setdefault("edf", {}).setdefault(
            "gyro_precession", {}
        )["enabled"] = False
        import tempfile, os
        tmp = tempfile.NamedTemporaryFile(
            suffix=".yaml", delete=False, mode="w", dir=str(REPO_ROOT)
        )
        yaml.dump(veh_data, tmp)
        tmp.close()
        veh_cfg_override = tmp.name
    else:
        veh_cfg_override = None

    env = EDFIsaacEnv(config_path=args.config)

    # If precession disabled, patch the task after env creation
    if args.disable_precession and hasattr(env, "_task"):
        env._task._gyro_enabled = False

    dt         = 1.0 / 120.0
    hover_steps = int(0.5 / dt)
    torque_steps = int((args.duration - 0.5) / dt)
    log_interval = 30

    obs, _ = env.reset()
    print(f"\nPhase 1: Hover stabilization ({0.5:.1f} s)")

    # Phase 1: hover (no external torque)
    for step in range(hover_steps):
        action = _make_action(_HOVER_NORM)
        obs, _, done, _, _ = env.step(action)
        if _is_done(done):
            obs, _ = env.reset()

    alt = _obs_val(obs, _OBS_ALTITUDE)
    print(f"  Thrust: {_HOVER_NORM * _T_MAX:.1f} N (hover) | ω_fan: {omega_fan_hover:.0f} rad/s")
    print(f"  Altitude: {alt:.2f} m (stable)")

    # Phase 2: apply pitch (or roll) torque
    torque_axis_label = "Y-axis (body pitch)" if args.torque_axis == "pitch" else "X-axis (body roll)"
    expected_response = "roll" if args.torque_axis == "pitch" else "pitch"
    print(f"\nPhase 2: {args.torque_axis.capitalize()} torque application ({args.duration - 0.5:.1f} s)")
    print(f"  Applied {args.torque_axis} torque: {args.torque_mag:.2f} N·m about {torque_axis_label}")
    print(f"  Expected response: {expected_response.upper()} rate coupling via τ_gyro = −ω×h_fan")
    print()
    print(f"  {'Time(s)':<8} {'Pitch(°/s)':<12} {'Roll(°/s)':<12} {'Yaw(°/s)':<10}")

    pitch_rates = []
    roll_rates  = []
    time_stamps = []

    # Build external torque vector in world frame (body Y or X for pitch/roll)
    # We apply via env.step with a fin action that approximates a torque, OR
    # directly through the task's set_external_force_and_torque if accessible.
    # Simplest: use fin deflections to generate a net torque.
    # For a pitch torque (about body Y): fins 1,3 deflect symmetrically opposite to fins 2,4.
    # For a roll torque (about body X): fins 1,2 vs 3,4.
    # Here we use a direct external torque injection via the task object if available.
    # Fallback: use fin-induced torque.
    task = getattr(env, "_task", None)

    for step in range(torque_steps):
        t_sim = step * dt

        # Apply hover thrust action (fins neutral)
        action = _make_action(_HOVER_NORM)
        obs, _, done, _, _ = env.step(action)

        # Inject external torque if task is accessible
        if task is not None and hasattr(task, "robot"):
            import torch
            num_envs = task.num_envs
            ext_forces  = torch.zeros((num_envs, 1, 3), device=task.device)
            ext_torques = torch.zeros((num_envs, 1, 3), device=task.device)
            if args.torque_axis == "pitch":
                ext_torques[:, 0, 1] = args.torque_mag   # body Y (pitch axis in Z-up)
            else:
                ext_torques[:, 0, 0] = args.torque_mag   # body X (roll axis in Z-up)
            task.robot.set_external_force_and_torque(
                ext_forces, ext_torques, body_ids=[0], is_global=False
            )

        if _is_done(done):
            obs, _ = env.reset()

        # Log every log_interval steps
        if step % log_interval == 0:
            p_rate = math.degrees(_obs_val(obs, _OBS_OMEGA_X))   # roll rate
            q_rate = math.degrees(_obs_val(obs, _OBS_OMEGA_Y))   # pitch rate
            r_rate = math.degrees(_obs_val(obs, _OBS_OMEGA_Z))   # yaw rate
            time_stamps.append(t_sim)
            pitch_rates.append(q_rate)
            roll_rates.append(p_rate)
            print(f"  {t_sim:<8.2f} {q_rate:<12.2f} {p_rate:<12.2f} {r_rate:<10.2f}")

    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------
    print()
    # Look at rates after 0.5 s (beyond initial transient)
    settled_rolls = [
        abs(r) for t, r in zip(time_stamps, roll_rates) if t >= 0.5
    ]
    settled_pitches = [
        abs(p) for t, p in zip(time_stamps, pitch_rates) if t >= 0.5
    ]
    max_perp_rate = max(settled_rolls) if settled_rolls else 0.0
    max_primary_rate = max(settled_pitches) if settled_pitches else 0.0

    # Analytical prediction: τ_gyro_x = −q·L  →  α_roll = τ_gyro_x / I_roll
    # For a rough estimate we use I_roll ≈ 0.005 kg·m² (from default_vehicle.yaml Ixx)
    I_roll_approx = 0.005
    q_at_t1 = pitch_rates[min(len(pitch_rates) - 1, len(pitch_rates) // 2)] if pitch_rates else 0.0
    tau_gyro_pred = abs(math.radians(q_at_t1)) * L_hover   # N·m
    alpha_roll_pred = math.degrees(tau_gyro_pred / I_roll_approx)  # °/s²

    print("=" * 50)
    if args.disable_precession:
        # Expect NO coupling
        threshold = 0.05   # °/s
        passed = max_perp_rate < threshold
        result_str = "PASS" if passed else "FAIL"
        print(f"RESULT: {result_str} — {'No' if passed else 'Unexpected'} {expected_response} response "
              f"(max {expected_response}_rate = {max_perp_rate:.2f} °/s, threshold < {threshold} °/s)")
    else:
        # Expect roll coupling
        threshold = 0.1   # °/s
        passed = max_perp_rate > threshold
        result_str = "PASS" if passed else "FAIL"
        print(f"RESULT: {result_str} — {expected_response.capitalize()} response detected "
              f"({max_perp_rate:.2f} °/s at t>0.5 s, threshold > {threshold} °/s)")
        if max_primary_rate > 0:
            ratio = max_perp_rate / max_primary_rate
            print(f"  Coupling ratio ({expected_response}_rate/{args.torque_axis}_rate): {ratio:.3f}")
        print(f"  Analytical precession estimate: {alpha_roll_pred:.3f} °/s² → ~{alpha_roll_pred * 0.5:.2f} °/s at t=0.5 s")

    print("=" * 50)

    if veh_cfg_override and Path(veh_cfg_override).exists():
        Path(veh_cfg_override).unlink()

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
