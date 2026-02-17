"""Quick diagnostic: run N episodes with default PID gains, report results."""
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
    results: dict[str, int] = {"landed": 0, "crashed": 0, "oob": 0, "truncated": 0}
    N = 30

    for seed in range(N):
        obs, _info = env.reset(seed=seed)
        ctrl = PIDController(pid_yaml)
        ctrl.reset()
        terminated = truncated = False
        ep_info: dict = {}
        steps = 0
        while not (terminated or truncated):
            action = ctrl.get_action(obs)
            obs, _reward, terminated, truncated, ep_info = env.step(action)
            steps += 1

        reason = ep_info.get("termination_reason", "truncated")
        landed = ep_info.get("landed", False)
        crashed = ep_info.get("crashed", False)
        oob = ep_info.get("out_of_bounds", False)
        cep = ep_info.get("cep", float("inf"))
        alt = ep_info.get("altitude", float("nan"))
        tilt_deg = float(np.rad2deg(ep_info.get("tilt_angle", 0.0)))
        v_touch = ep_info.get("touchdown_velocity", None)
        omega_norm = ep_info.get("angular_rate", float("nan"))
        v_i = ep_info.get("velocity_inertial", np.zeros(3))
        v_mag = float(np.linalg.norm(v_i))

        if landed:
            results["landed"] += 1
        elif crashed:
            results["crashed"] += 1
        elif oob:
            results["oob"] += 1
        else:
            results["truncated"] += 1
        print(
            f"seed={seed:2d}  steps={steps:4d}  {reason:22s}  "
            f"v={v_mag:5.2f}  tilt={tilt_deg:5.1f}deg  w={omega_norm:5.2f}  "
            f"alt={alt:5.3f}  cep={cep:6.2f}"
        )

    print()
    pct = 100 * results["landed"] / N
    print(f"Results over {N} episodes:")
    print(f"  Landed:    {results['landed']}/{N}  ({pct:.0f}%)")
    print(f"  Crashed:   {results['crashed']}/{N}")
    print(f"  OOB:       {results['oob']}/{N}")
    print(f"  Truncated: {results['truncated']}/{N}")


if __name__ == "__main__":
    main()
