"""Single-episode deep diagnostic: trace omega, actions, and state per step."""
from __future__ import annotations
import numpy as np
from pathlib import Path

from simulation.training.edf_landing_env import EDFLandingEnv
from simulation.training.controllers.pid_controller import PIDController
from simulation.config_loader import load_config


def main() -> None:
    configs = Path("simulation/configs")
    pid_yaml = load_config(configs / "pid.yaml")

    v = load_config(configs / "default_vehicle.yaml")
    e = load_config(configs / "default_environment.yaml")
    r = load_config(configs / "reward.yaml")
    cfg: dict = {
        "vehicle": v.get("vehicle", v),
        "environment": e.get("environment", e),
        "reward": r.get("reward", r),
    }
    env_cfg = dict(cfg.get("environment", {}))
    atm = dict(env_cfg.get("atmosphere", {}))
    atm["randomize_T"] = 0.0
    atm["randomize_P"] = 0.0
    env_cfg["atmosphere"] = atm
    wind = dict(env_cfg.get("wind", {}))
    wind["mean_vector_range_lo"] = [0.0, 0.0, 0.0]
    wind["mean_vector_range_hi"] = [0.0, 0.0, 0.0]
    wind["turbulence_intensity"] = 0.0
    wind["gust_prob"] = 0.0
    env_cfg["wind"] = wind
    cfg["environment"] = env_cfg
    cfg["actuator_delay"] = {"enabled": False}
    cfg["obs_latency"] = {"enabled": False}
    cfg["observation"] = {"noise_std": 0.0}

    env = EDFLandingEnv(cfg)

    SEED = 0
    obs, info = env.reset(seed=SEED)
    ctrl = PIDController(pid_yaml)
    ctrl.reset()

    print(f"{'step':>4}  {'alt':>6}  {'vz':>6}  {'ox':>7}  {'oy':>7}  {'oz':>7}  "
          f"{'|w|':>7}  {'tilt':>6}  "
          f"{'a0':>6}  {'f1':>6}  {'f2':>6}  {'f3':>6}  {'f4':>6}")
    print("-" * 110)

    terminated = truncated = False
    ep_info: dict = {}
    step = 0
    while not (terminated or truncated):
        action = ctrl.get_action(obs)

        o = obs
        target_body = o[0:3]
        v_b = o[3:6]
        g_body = o[6:9]
        omega = o[9:12]
        h_agl = float(o[16])

        g2 = float(g_body[2]) if abs(float(g_body[2])) > 1e-9 else 1e-9
        roll_est = float(np.rad2deg(np.arctan2(float(g_body[1]), g2)))
        pitch_est = float(np.rad2deg(np.arctan2(-float(g_body[0]), g2)))
        tilt_deg = float(np.rad2deg(np.arccos(np.clip(g_body[2], -1, 1))))

        omega_norm = float(np.linalg.norm(omega))

        if step < 10 or step % 10 == 0 or step > 55:
            print(
                f"{step:4d}  {h_agl:6.2f}  {v_b[2]:6.2f}  "
                f"{omega[0]:7.2f}  {omega[1]:7.2f}  {omega[2]:7.2f}  "
                f"{omega_norm:7.2f}  {tilt_deg:6.1f}  "
                f"{action[0]:6.3f}  {action[1]:6.3f}  {action[2]:6.3f}  "
                f"{action[3]:6.3f}  {action[4]:6.3f}"
            )

        obs, _reward, terminated, truncated, ep_info = env.step(action)
        step += 1

    # Print final state
    o = obs
    omega_final = o[9:12]
    h_final = float(o[16])
    v_b_final = o[3:6]
    omega_norm_final = float(np.linalg.norm(omega_final))
    reason = ep_info.get("termination_reason", "truncated")
    print("-" * 110)
    print(f"TERMINATED: {reason}  step={step}  "
          f"alt={h_final:.3f}  v_z={v_b_final[2]:.2f}  "
          f"omega=[{omega_final[0]:.2f}, {omega_final[1]:.2f}, {omega_final[2]:.2f}]  "
          f"|w|={omega_norm_final:.2f}")


if __name__ == "__main__":
    main()
