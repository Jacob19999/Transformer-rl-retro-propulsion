# Simulation File Map

> **Purpose**: Single place to track all files created under `simulation/`, their roles, and any implementation notes, so we can avoid repeatedly re-discovering the structure.

---

## 1. Package Layout

- **Root package**
  - `simulation/__init__.py` — package marker for the overall simulation library.
  - `simulation/config_loader.py` — shared YAML config loader (`load_config(path) -> dict`).

- **Dynamics**
  - `simulation/dynamics/__init__.py` — package marker for dynamics modules (vehicle, thrust, aero, fins, servo, integrator, quaternions).

- **Environment**
  - `simulation/environment/__init__.py` — package marker for environment modules (atmosphere, wind, environment assembly).
  - `simulation/environment/atmosphere_model.py` — ISA baseline + per-episode randomization; provides `(T, P, rho)` for force models. **Stage 8**.

- **Training**
  - `simulation/training/__init__.py` — package marker for Gym env + training utilities.
  - `simulation/training/controllers/__init__.py` — controller package (PID, PPO-MLP, GTrXL-PPO, SCP).
  - `simulation/training/scripts/__init__.py` — importable entry points for training/eval scripts.
  - `simulation/training/configs/__init__.py` — controller-specific config package.

- **Configs**
  - `simulation/configs/__init__.py` — top-level simulation config package.
  - `simulation/configs/default_vehicle.yaml` — **main vehicle config** copied from `vehicle.md §9.1`.
  - `simulation/configs/default_environment.yaml` — **main environment config** copied from `env.md §7.1`.
  - `simulation/configs/test_vehicle.yaml` — simplified, zero-drag, no-DR vehicle config for unit tests.
  - `simulation/configs/test_environment.yaml` — zero-wind, zero-randomization environment config for deterministic tests.

- **Tests**
  - `simulation/tests/__init__.py` — tests package marker (Stage 1+ will add concrete test modules here).
  - `simulation/tests/test_quaternion_utils.py` — unit tests for quaternion utilities (DCM, products, normalization, Euler conversions). **Stage 1**.
  - `simulation/tests/test_mass_properties.py` — unit tests for primitive inertia + aggregation + CAD override + optional mass randomization. **Stage 2**.
  - `simulation/tests/test_thrust_model.py` — unit tests for EDF thrust curve, motor lag, ground effect, density correction, force/torque outputs. **Stage 3**.
  - `simulation/tests/test_aero_model.py` — unit tests for relative velocity, directional drag, drag force, aero torque, wind rotation. **Stage 4**.
  - `simulation/tests/test_fin_model.py` — unit tests for thin-airfoil lift, induced drag, per-fin force, exhaust velocity scaling, mechanical clamp, symmetric deflection. **Stage 5**.
  - `simulation/tests/test_servo_model.py` — unit tests for servo rate-limited first-order lag, rate limiting, reset randomization. **Stage 6**.
  - `simulation/tests/test_integrator.py` — unit tests for RK4 integrator accuracy and quaternion re-normalization behavior. **Stage 7**.
  - `simulation/tests/test_atmosphere_model.py` — unit tests for ISA sea-level density, randomization bounds, and ideal gas consistency. **Stage 8**.

- **Isaac (Phase 2, optional)**
  - `simulation/isaac/__init__.py` — Isaac Sim integration package marker.
  - `simulation/isaac/configs/__init__.py` — Isaac-specific configs package marker.
  - `simulation/isaac/usd/` — placeholder directory for USD assets (no files yet).

---

## 2. Notes / TODOs by Area

- **Dynamics**
  - `simulation/dynamics/quaternion_utils.py` — quaternion math utilities (DCM, Hamilton product, normalization, Euler conversions). **Stage 1**.
  - `simulation/dynamics/mass_properties.py` — primitive-based mass property aggregation (mass, CoM, inertia via parallel axis theorem) + aerodynamic area aggregation + CAD override + optional mass randomization. **Stage 2**.
  - `simulation/dynamics/thrust_model.py` — EDF thrust curve + 1st-order lag + ground effect + density correction + thrust force/torque + motor reaction torque. **Stage 3**.
  - `simulation/dynamics/aero_model.py` — combined-shape aerodynamic drag (relative velocity, directional drag, drag force, aero torque). **Stage 4**.
  - `simulation/dynamics/fin_model.py` — 4× NACA 0012 fins in exhaust (thin-airfoil lift, induced drag, per-fin force, exhaust velocity scaling, mechanical clamp, total force/torque). **Stage 5**.
  - `simulation/dynamics/servo_model.py` — rate-limited first-order lag actuator model for fin servos, with per-episode tau + derating randomization. **Stage 6**.
  - `simulation/dynamics/integrator.py` — generic fixed-step RK4 integrator (`rk4_step`) plus optional periodic quaternion re-normalization (`RK4Integrator`). **Stage 7**.
  - TODO: Add file entries here as we implement `vehicle.py` (Stage 11).

- **Environment**
  - `simulation/environment/atmosphere_model.py` — ISA lapse + barometric pressure + ideal gas density; `reset()` randomizes base \(T_{base}, P_{base}\). **Stage 8**.
  - TODO: Add file entries here as we implement `wind_model.py`, `environment_model.py` (Stages 9–10).

- **Training / Gym**
  - TODO: Add file entries here as we implement `edf_landing_env.py`, `observation.py`, `reward.py`, `curriculum.py`, controllers, and training/eval scripts (Stages 12–23).

- **Configs**
  - Keep `default_*.yaml` and `test_*.yaml` in sync with the plan docs (`vehicle.md`, `env.md`, `training.md`). Any intentional deviations should be documented inline in the YAML and briefly summarized here.

