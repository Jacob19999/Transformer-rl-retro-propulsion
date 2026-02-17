"""Print vehicle mass properties and yaw dynamics numbers."""
from __future__ import annotations
import numpy as np
from pathlib import Path

from simulation.training.edf_landing_env import EDFLandingEnv
from simulation.config_loader import load_config


def main() -> None:
    configs = Path("simulation/configs")
    v = load_config(configs / "default_vehicle.yaml")
    e = load_config(configs / "default_environment.yaml")
    r = load_config(configs / "reward.yaml")
    cfg: dict = {
        "vehicle": v.get("vehicle", v),
        "environment": e.get("environment", e),
        "reward": r.get("reward", r),
    }
    env = EDFLandingEnv(cfg)
    veh = env.vehicle

    print(f"Mass:  {veh.mass:.3f} kg")
    print(f"CoM:   {veh.com}")
    print(f"I_fan: {veh.I_fan}")
    print(f"Izz:   {veh.I[2,2]:.6f} kg*m^2")
    print(f"I tensor:")
    for row in veh.I:
        print(f"  {row}")

    k_thrust = veh.thrust_model.config.k_thrust
    T_hover = veh.mass * veh.g
    T_max = veh.thrust_model.config.T_max
    omega_hover = np.sqrt(T_hover / k_thrust)
    omega_max = np.sqrt(T_max / k_thrust)
    delta_omega = omega_max - omega_hover

    print(f"\nk_thrust:  {k_thrust}")
    print(f"T_hover:   {T_hover:.2f} N")
    print(f"T_max:     {T_max} N")
    print(f"omega_fan at hover: {omega_hover:.1f} rad/s")
    print(f"omega_fan at T_max: {omega_max:.1f} rad/s")
    print(f"Delta omega_fan: {delta_omega:.1f} rad/s")

    angular_impulse = veh.I_fan * delta_omega
    delta_omega_z = angular_impulse / veh.I[2,2]
    print(f"\nMotor reaction angular impulse: {angular_impulse:.5f} N*m*s")
    print(f"Total yaw velocity change (hover->max): {delta_omega_z:.2f} rad/s")

    # What I_fan would give delta_omega_z = 0.5 rad/s?
    I_fan_target = 0.5 * veh.I[2,2] / delta_omega
    print(f"\nI_fan needed for delta_omega_z < 0.5: {I_fan_target:.6f} kg*m^2")


if __name__ == "__main__":
    main()
