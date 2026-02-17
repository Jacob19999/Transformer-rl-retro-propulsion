"""Diagnose yaw spin-up: test with zero fins, hover thrust only."""
from __future__ import annotations
import numpy as np
from pathlib import Path

from simulation.training.edf_landing_env import EDFLandingEnv
from simulation.config_loader import load_config


def run_ep(env, action_fn, label, seed=0, steps=40):
    obs, _ = env.reset(seed=seed)
    print(f"\n=== {label} (seed={seed}) ===")
    print(f"{'step':>4}  {'alt':>6}  {'ox':>7}  {'oy':>7}  {'oz':>7}  {'|w|':>7}  {'a0':>6}  {'f1':>5}  {'f2':>5}  {'f3':>5}  {'f4':>5}")
    for s in range(steps):
        action = action_fn(obs)
        omega = obs[9:12]
        h = float(obs[16])
        if s < 15 or s % 5 == 0:
            print(f"{s:4d}  {h:6.2f}  {omega[0]:7.3f}  {omega[1]:7.3f}  {omega[2]:7.3f}  "
                  f"{np.linalg.norm(omega):7.3f}  "
                  f"{action[0]:6.3f}  {action[1]:5.2f}  {action[2]:5.2f}  {action[3]:5.2f}  {action[4]:5.2f}")
        obs, _, term, trunc, info = env.step(action)
        if term or trunc:
            print(f"  -> Terminated at step {s+1}: {info.get('termination_reason', 'trunc')}")
            break


def main():
    configs = Path("simulation/configs")
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

    # Test 1: zero fins, hover thrust
    run_ep(env, lambda obs: np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
           "ZERO FINS, hover thrust (a0=0)")

    # Test 2: zero fins, zero thrust
    run_ep(env, lambda obs: np.array([-1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
           "ZERO FINS, min thrust (a0=-1)")

    # Test 3: full positive pitch (common mode fins 1/2)
    run_ep(env, lambda obs: np.array([0.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32),
           "PITCH ONLY: fins 1/2 = +1")

    # Test 4: full negative roll (common mode fins 3/4)
    run_ep(env, lambda obs: np.array([0.0, 0.0, 0.0, -1.0, -1.0], dtype=np.float32),
           "ROLL ONLY: fins 3/4 = -1")

    # Test 5: old differential yaw pattern [+1, -1, +1, -1] (zero net yaw)
    run_ep(env, lambda obs: np.array([0.0, 1.0, -1.0, 1.0, -1.0], dtype=np.float32),
           "OLD YAW: [+1,-1,+1,-1] (zero net yaw)")

    # Test 6: corrected yaw pattern [+1, -1, -1, +1] (should produce yaw)
    run_ep(env, lambda obs: np.array([0.0, 1.0, -1.0, -1.0, 1.0], dtype=np.float32),
           "NEW YAW: [+1,-1,-1,+1] (should create yaw torque)")

    # Test 7: combined pitch+roll [+1,+1,-1,-1] (what PID does at step 0)
    run_ep(env, lambda obs: np.array([1.0, 1.0, 1.0, -1.0, -1.0], dtype=np.float32),
           "PITCH+ROLL: [+1,+1,-1,-1], a0=+1")


if __name__ == "__main__":
    main()
