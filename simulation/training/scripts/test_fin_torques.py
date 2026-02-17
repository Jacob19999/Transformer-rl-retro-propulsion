"""Diagnostic: check fin-to-torque mapping to verify pitch/roll control."""
from __future__ import annotations

import numpy as np
from pathlib import Path
from simulation.config_loader import load_config
from simulation.dynamics.vehicle import VehicleDynamics
from simulation.environment.environment_model import EnvironmentModel


def main() -> None:
    configs = Path("simulation/configs")
    v = load_config(configs / "default_vehicle.yaml")
    e = load_config(configs / "default_environment.yaml")
    env_model = EnvironmentModel(e.get("environment", e))
    vehicle = VehicleDynamics(v.get("vehicle", v), env_model)

    print("CoM:", vehicle.com)
    print("Mass:", vehicle.mass)
    print()
    print("Fin positions:")
    for i, fc in enumerate(vehicle.fin_model._config.fins_config):
        pos = fc["position"]
        lift = fc["lift_direction"]
        drag = fc["drag_direction"]
        print(f"  fin_{i+1}: pos={pos}, lift={lift}, drag={drag}")
    print()

    hover_T = vehicle.mass * vehicle.g
    omega_fan = vehicle.thrust_model.omega_from_thrust(hover_T)
    print(f"Hover thrust: {hover_T:.1f} N, omega_fan: {omega_fan:.0f} rad/s")
    print()

    # Test fin-by-fin torques
    for i in range(4):
        delta = np.zeros(4)
        delta[i] = 0.1  # 0.1 rad positive deflection
        F, tau = vehicle.fin_model.compute(delta, omega_fan, rho=1.225)
        print(f"fin_{i+1} (+0.1 rad): F=[{F[0]:.4f}, {F[1]:.4f}, {F[2]:.4f}]  "
              f"tau=[{tau[0]:.4f}, {tau[1]:.4f}, {tau[2]:.4f}]")
    print()

    # Test mappings: differential vs common-mode
    tests = [
        ("fins1/2 DIFF  (current PID for pitch)", [0.1, -0.1, 0.0, 0.0]),
        ("fins1/2 COMMON                       ", [0.1, 0.1, 0.0, 0.0]),
        ("fins3/4 DIFF  (current PID for roll) ", [0.0, 0.0, 0.1, -0.1]),
        ("fins3/4 COMMON                       ", [0.0, 0.0, 0.1, 0.1]),
    ]
    for label, delta_list in tests:
        delta = np.array(delta_list)
        F, tau = vehicle.fin_model.compute(delta, omega_fan, rho=1.225)
        print(f"{label}: tau_roll={tau[0]:+.5f}  tau_pitch={tau[1]:+.5f}  tau_yaw={tau[2]:+.5f}")


if __name__ == "__main__":
    main()
