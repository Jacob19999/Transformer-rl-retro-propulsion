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

- **Isaac (Phase 2, optional)**
  - `simulation/isaac/__init__.py` — Isaac Sim integration package marker.
  - `simulation/isaac/configs/__init__.py` — Isaac-specific configs package marker.
  - `simulation/isaac/usd/` — placeholder directory for USD assets (no files yet).

---

## 2. Notes / TODOs by Area

- **Dynamics**
  - `simulation/dynamics/quaternion_utils.py` — quaternion math utilities (DCM, Hamilton product, normalization, Euler conversions). **Stage 1**.
  - `simulation/dynamics/mass_properties.py` — primitive-based mass property aggregation (mass, CoM, inertia via parallel axis theorem) + aerodynamic area aggregation + CAD override + optional mass randomization. **Stage 2**.
  - TODO: Add file entries here as we implement `vehicle.py`, `thrust_model.py`, `aero_model.py`, `fin_model.py`, `servo_model.py`, `integrator.py` (Stages 3–7).

- **Environment**
  - TODO: Add file entries here as we implement `atmosphere_model.py`, `wind_model.py`, `environment_model.py` (Stages 8–10).

- **Training / Gym**
  - TODO: Add file entries here as we implement `edf_landing_env.py`, `observation.py`, `reward.py`, `curriculum.py`, controllers, and training/eval scripts (Stages 12–23).

- **Configs**
  - Keep `default_*.yaml` and `test_*.yaml` in sync with the plan docs (`vehicle.md`, `env.md`, `training.md`). Any intentional deviations should be documented inline in the YAML and briefly summarized here.

