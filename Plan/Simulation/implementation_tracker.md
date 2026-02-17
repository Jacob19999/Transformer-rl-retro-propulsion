# Implementation Tracker — Simulation Pipeline

> **Purpose**: Step-by-step checklist for implementing the simulation pipeline. Each task is scoped small enough for a single coding session and ordered by dependency. Baseline controllers (PPO-MLP, PID, SCP) are completed and validated **before** GTrXL-PPO.
>
> **Source plans**: [master_plan.md](master_plan.md) | [vehicle.md](Vehicle%20Dynamics/vehicle.md) | [env.md](Enviornment/env.md) | [training.md](Training%20Plan/training.md)

---

## Status Legend

| Symbol  | Meaning                   |
| ------- | ------------------------- |
| `[ ]` | Not started               |
| `[~]` | In progress               |
| `[x]` | Complete                  |
| `[!]` | Blocked / needs attention |
| `[-]` | Skipped / deleted         |

---

## Stage 0 — Project Scaffolding

> Set up the directory structure, dependency management, and configuration loading before any physics code.

| #   | Task                                                                                                          | Status  | Files                                | Depends On | Notes                                                                                   |
| --- | ------------------------------------------------------------------------------------------------------------- | ------- | ------------------------------------ | ---------- | --------------------------------------------------------------------------------------- |
| 0.1 | Create `simulation/` directory tree matching [vehicle.md §12.1](Vehicle%20Dynamics/vehicle.md) file map       | `[x]` | dirs only                            | —         | `dynamics/`, `environment/`, `training/`, `configs/`, `tests/`                |
| 0.2 | Create `requirements.txt` with pinned core deps                                                             | `[x]` | `requirements.txt`                 | —         | `numpy>=1.24`, `scipy>=1.10`, `pyyaml>=6.0`, `pytest>=7.0`, `gymnasium>=0.29` |
| 0.3 | Create `configs/default_vehicle.yaml` — copy full YAML from [vehicle.md §9.1](Vehicle%20Dynamics/vehicle.md) | `[x]` | `configs/default_vehicle.yaml`     | 0.1        | All primitives, EDF, aero, fins, servo sections                                         |
| 0.4 | Create `configs/default_environment.yaml` — copy full YAML from [env.md §7.1](Enviornment/env.md)            | `[x]` | `configs/default_environment.yaml` | 0.1        | Wind, atmosphere, curriculum sections                                                   |
| 0.5 | Create `configs/test_vehicle.yaml` — simplified zero-drag, no-DR config for unit tests                     | `[x]` | `configs/test_vehicle.yaml`        | 0.3        | Single primitive, zero randomization                                                    |
| 0.6 | Create `configs/test_environment.yaml` — zero wind, zero randomization                                     | `[x]` | `configs/test_environment.yaml`    | 0.4        | Deterministic ISA, no gusts, no turbulence                                              |
| 0.7 | Write shared YAML config loader utility                                                                       | `[x]` | `simulation/config_loader.py`      | 0.1        | `load_config(path) -> dict`                                                           |
| 0.8 | Add `__init__.py` files to all packages                                                                     | `[x]` | `simulation/**/__init__.py`        | 0.1        | Empty or minimal exports                                                                |

---

## Stage 1 — Quaternion Utilities

> Pure math. No dependencies. Used by every dynamics module. Ref: [vehicle.md §2](Vehicle%20Dynamics/vehicle.md)

| #   | Task                                                                               | Status  | Files                              | Depends On | Notes                                                                                                                        |
| --- | ---------------------------------------------------------------------------------- | ------- | ---------------------------------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------- |
| 1.1 | Implement `quat_to_dcm(q)` — quaternion to 3x3 rotation matrix (body→inertial) | `[x]` | `dynamics/quaternion_utils.py`   | —         | Eq from vehicle.md §2.3. Scalar-first convention `[q0,q1,q2,q3]`                                                          |
| 1.2 | Implement `quat_mult(q1, q2)` — Hamilton product                                | `[x]` | `dynamics/quaternion_utils.py`   | —         | Used in quaternion kinematics `q_dot = 0.5 * q ⊗ [0, ω]`                                                                 |
| 1.3 | Implement `quat_normalize(q)` — unit norm re-normalization                      | `[x]` | `dynamics/quaternion_utils.py`   | —         | `q / np.linalg.norm(q)`                                                                                                    |
| 1.4 | Implement `euler_to_quat(roll, pitch, yaw)` — Euler angles to quaternion        | `[x]` | `dynamics/quaternion_utils.py`   | —         | Needed for initial condition sampling                                                                                        |
| 1.5 | Implement `quat_to_euler(q)` — quaternion to Euler (for logging/debug only)     | `[x]` | `dynamics/quaternion_utils.py`   | —         | Optional but useful for human-readable output                                                                                |
| 1.6 | Write unit tests for all quaternion functions                                      | `[x]` | `tests/test_quaternion_utils.py` | 1.1–1.5   | Compare `quat_to_dcm` against `scipy.spatial.transform.Rotation`; test quaternion identity, 90° rotations, double-cover |

---

## Stage 2 — Mass Properties

> Primitive-based aggregation. Init-only computation. Ref: [vehicle.md §5](Vehicle%20Dynamics/vehicle.md)

| #   | Task                                                                                                                        | Status  | Files                             | Depends On | Notes                                                                      |
| --- | --------------------------------------------------------------------------------------------------------------------------- | ------- | --------------------------------- | ---------- | -------------------------------------------------------------------------- |
| 2.1 | Implement `_primitive_inertia(prim)` — per-shape inertia formulas (cylinder, box, sphere)                                | `[x]` | `dynamics/mass_properties.py`   | —         | Formulas from vehicle.md §5.3                                             |
| 2.2 | Implement `compute_mass_properties(primitives)` — aggregate total mass, CoM, composite inertia via parallel axis theorem | `[x]` | `dynamics/mass_properties.py`   | 2.1        | vehicle.md §5.4. Include orientation rotation `R_i @ I_local @ R_i.T`   |
| 2.3 | Implement aerodynamic area aggregation — total surface area, per-axis projected areas `(A_x, A_y, A_z)`                  | `[x]` | `dynamics/mass_properties.py`   | 2.2        | Sum `surface_area`, `drag_facing.{x,y,z}` from primitives              |
| 2.4 | Implement `MassProperties.from_cad(cad_config)` — CAD override class method                                              | `[x]` | `dynamics/mass_properties.py`   | 2.2        | Bypass aggregation, use CAD-exported values directly                       |
| 2.5 | Implement optional per-episode mass randomization (`randomize_mass` field)                                                | `[x]` | `dynamics/mass_properties.py`   | 2.2        | ±10% on `payload_variable` primitive mass; recomputes CoM/I             |
| 2.6 | Write unit tests for mass properties                                                                                        | `[x]` | `tests/test_mass_properties.py` | 2.1–2.5   | Single primitive vs. analytical; 2-box hand-calc; diagonal dominance check |

---

## Stage 3 — Thrust Model

> EDF thrust curve, first-order motor lag, ground effect, density correction. Ref: [vehicle.md §6.1](Vehicle%20Dynamics/vehicle.md)

| #   | Task                                                                                             | Status  | Files                          | Depends On | Notes                                                                                     |
| --- | ------------------------------------------------------------------------------------------------ | ------- | ------------------------------ | ---------- | ----------------------------------------------------------------------------------------- |
| 3.1 | Implement thrust magnitude:`T = k * omega_fan^2` and inverse `omega_fan = sqrt(T/k)`         | `[x]` | `dynamics/thrust_model.py`   | —         | vehicle.md §6.1.1                                                                        |
| 3.2 | Implement first-order motor lag:`T_dot = (T_cmd - T) / tau_motor`                              | `[x]` | `dynamics/thrust_model.py`   | 3.1        | vehicle.md §6.1.2. Returns `T_dot` for RK4 integration. `tau_motor ≈ 0.1 s`         |
| 3.3 | Implement ground effect:`T_eff = T * (1 + 0.5*(r_duct/h)^2)` with altitude clamp `h >= 0.01` | `[x]` | `dynamics/thrust_model.py`   | 3.1        | vehicle.md §6.1.3                                                                        |
| 3.4 | Add density correction:`T_eff *= rho / rho_ref` — `rho` passed as parameter                 | `[x]` | `dynamics/thrust_model.py`   | 3.3        | env.md §5.3.`rho_ref = 1.225`                                                          |
| 3.5 | Implement thrust force vector:`F_thrust = [0, 0, T_eff]` and torque: `r_offset × F_thrust`  | `[x]` | `dynamics/thrust_model.py`   | 3.3        | vehicle.md §6.1.4                                                                        |
| 3.6 | Implement motor reaction torque:`tau_motor = -I_fan * omega_fan_dot * [0,0,1]`                 | `[x]` | `dynamics/thrust_model.py`   | 3.2        | vehicle.md §6.4. Derive `omega_fan_dot` from `T_dot`                                 |
| 3.7 | Implement `ThrustModel.reset()` for episode init                                               | `[x]` | `dynamics/thrust_model.py`   | 3.1        | Reset internal thrust state                                                               |
| 3.8 | Write unit tests for thrust model                                                                | `[x]` | `tests/test_thrust_model.py` | 3.1–3.7   | Step response: 63% at t=tau; ground effect at h=r_duct → 1.5x; zero-thrust → zero force |

---

## Stage 4 — Aerodynamic Drag Model

> Combined-shape drag on the vehicle body. Ref: [vehicle.md §6.2](Vehicle%20Dynamics/vehicle.md)

| #   | Task                                                                                | Status  | Files                        | Depends On | Notes                                                                                 |
| --- | ----------------------------------------------------------------------------------- | ------- | ---------------------------- | ---------- | ------------------------------------------------------------------------------------- |
| 4.1 | Implement relative velocity computation:`v_rel = v_b - R.T @ v_wind`              | `[x]` | `dynamics/aero_model.py`   | 1.1        | vehicle.md §6.2.1.`v_wind` comes from EnvironmentModel                             |
| 4.2 | Implement directional drag: per-axis projected areas weighted by velocity direction | `[x]` | `dynamics/aero_model.py`   | 4.1        | vehicle.md Appendix A pseudocode; fallback to scalar `A_proj` if disabled           |
| 4.3 | Implement drag force: `F_aero = -0.5 * rho * \|v_rel\| * v_rel * Cd * A_eff`        | `[x]` | `dynamics/aero_model.py`   | 4.1        | vehicle.md §6.2.2                                                                    |
| 4.4 | Implement aero torque:`tau_aero = (r_cp - com) × F_aero`                         | `[x]` | `dynamics/aero_model.py`   | 4.3        | vehicle.md §6.2.3                                                                    |
| 4.5 | Implement `AeroModel.__init__` accepting `MassProperties` for projected areas   | `[x]` | `dynamics/aero_model.py`   | 2.3, 4.1   | Reads `projected_area_{x,y,z}` from mass props                                      |
| 4.6 | Write unit tests for aero model                                                     | `[x]` | `tests/test_aero_model.py` | 4.1–4.5   | Zero velocity → zero drag; known v_rel → analytical drag match; wind rotation check |

---

## Stage 5 — Fin Model

> 4 NACA 0012 fins in exhaust, thin-airfoil coefficients, stall soft-clamp. Ref: [vehicle.md §6.3](Vehicle%20Dynamics/vehicle.md)

| #   | Task                                                                                                | Status  | Files                       | Depends On | Notes                                                                                      |
| --- | --------------------------------------------------------------------------------------------------- | ------- | --------------------------- | ---------- | ------------------------------------------------------------------------------------------ |
| 5.1 | Implement thin-airfoil lift:`C_L = Cl_alpha * alpha_eff` with `tanh` stall soft-clamp at ±15° | `[x]` | `dynamics/fin_model.py`   | —         | vehicle.md §6.3.3.`alpha_eff = stall_angle * tanh(alpha / stall_angle)`                 |
| 5.2 | Implement induced drag:`C_D = Cd0 + C_L^2 / (pi * AR)`                                            | `[x]` | `dynamics/fin_model.py`   | 5.1        | vehicle.md §6.3.3                                                                         |
| 5.3 | Implement per-fin force:`F_k = 0.5 * rho * V_e^2 * A_fin * (C_L * n_L + C_D * n_D)`               | `[x]` | `dynamics/fin_model.py`   | 5.1–5.2   | vehicle.md §6.3.4. Use per-fin config for `lift_direction`, `drag_direction`          |
| 5.4 | Implement exhaust velocity scaling:`V_exhaust = V_exhaust_nominal * (omega_fan / omega_fan_max)`  | `[x]` | `dynamics/fin_model.py`   | —         | vehicle.md §9.1 fins config                                                               |
| 5.5 | Implement mechanical clamp:`delta = clip(delta, -delta_max, +delta_max)` at ±20°                | `[x]` | `dynamics/fin_model.py`   | —         | vehicle.md §6.3.3 two-stage protection                                                    |
| 5.6 | Implement total fin force/torque: sum over 4 fins,`tau_fins = Σ (r_fin_k - com) × F_k`          | `[x]` | `dynamics/fin_model.py`   | 5.3        | vehicle.md §6.3.5. Vectorize over 4 fins (avoid Python loop)                              |
| 5.7 | Write unit tests for fin model                                                                      | `[x]` | `tests/test_fin_model.py` | 5.1–5.6   | δ=0 → zero lift; δ=10° → C_L ≈ 1.097; symmetric deflection → zero net lateral force |

---

## Stage 6 — Servo Model

> Rate-limited first-order lag for 4 fin servos. Ref: [vehicle.md §6.3.6](Vehicle%20Dynamics/vehicle.md)

| #   | Task                                                                                           | Status  | Files                         | Depends On | Notes                                                                                     |
| --- | ---------------------------------------------------------------------------------------------- | ------- | ----------------------------- | ---------- | ----------------------------------------------------------------------------------------- |
| 6.1 | Implement `ServoModel.__init__` — load tau, rate_max, derating from config                  | `[x]` | `dynamics/servo_model.py`   | —         | vehicle.md §6.3.6.4. Freewing 9 g servo params                                           |
| 6.2 | Implement `compute_rate(delta_cmd, delta_actual)` — returns `delta_dot` (4,) array        | `[x]` | `dynamics/servo_model.py`   | 6.1        | Rate-limited first-order lag:`clip((cmd - actual) / tau, -rate_max_eff, +rate_max_eff)` |
| 6.3 | Implement `ServoModel.step(delta_cmd, dt)` — Euler step for standalone testing              | `[x]` | `dynamics/servo_model.py`   | 6.2        | `delta_actual += delta_dot * dt`                                                        |
| 6.4 | Implement `ServoModel.reset(seed)` — zero positions, randomize tau for domain randomization | `[x]` | `dynamics/servo_model.py`   | 6.1        | tau_range from config, derating uniform [0.2, 0.5]                                        |
| 6.5 | Write unit tests for servo model                                                               | `[x]` | `tests/test_servo_model.py` | 6.1–6.4   | Small step → first-order response; large step → rate-limited; reset zeros positions     |

---

## Stage 7 — RK4 Integrator

> Fixed-step 4th-order Runge-Kutta. Ref: [vehicle.md §7](Vehicle%20Dynamics/vehicle.md)

| #   | Task                                                                                            | Status  | Files                        | Depends On | Notes                                                                                       |
| --- | ----------------------------------------------------------------------------------------------- | ------- | ---------------------------- | ---------- | ------------------------------------------------------------------------------------------- |
| 7.1 | Implement generic `rk4_step(f, y, u, t, dt)` — takes derivative function, returns `y_next` | `[x]` | `dynamics/integrator.py`   | —         | vehicle.md §7.1. Standard k1,k2,k3,k4 stages                                               |
| 7.2 | Add quaternion normalization every N steps (default 10)                                         | `[x]` | `dynamics/integrator.py`   | —         | vehicle.md §7.3. Step counter modulo check                                                 |
| 7.3 | Write unit tests for integrator                                                                 | `[x]` | `tests/test_integrator.py` | 7.1–7.2   | Simple ODE `dy/dt = -y` → exponential decay; error < O(dt^5); quaternion norm stays ~1.0 |

---

## Stage 8 — Atmosphere Model

> ISA baseline + per-episode randomization. Ref: [env.md §4](Enviornment/env.md)

| #   | Task                                                                                  | Status  | Files                               | Depends On | Notes                                                                                    |
| --- | ------------------------------------------------------------------------------------- | ------- | ----------------------------------- | ---------- | ---------------------------------------------------------------------------------------- |
| 8.1 | Implement `AtmosphereModel.__init__` — load ISA constants and randomization params | `[x]` | `environment/atmosphere_model.py` | —         | env.md §4.6. Precompute exponent `-g / (R * lapse)`                                   |
| 8.2 | Implement `get_conditions(h)` → `(T, P, rho)` with altitude lapse                | `[x]` | `environment/atmosphere_model.py` | 8.1        | env.md §4.3 equations                                                                   |
| 8.3 | Implement `reset()` — randomize T_base, P_base per episode                         | `[x]` | `environment/atmosphere_model.py` | 8.1        | Uniform ±10 K, ±2000 Pa                                                                |
| 8.4 | Write unit tests for atmosphere model                                                 | `[x]` | `tests/test_atmosphere_model.py`  | 8.1–8.3   | ISA sea-level → ρ=1.225; randomization bounds; ideal gas consistency `P/(R*T) = rho` |

---

## Stage 9 — Wind Model

> Mean + Dryden turbulence + discrete gusts. Ref: [env.md §3](Enviornment/env.md)

| #    | Task                                                                                              | Status  | Files                         | Depends On    | Notes                                                                                                                 |
| ---- | ------------------------------------------------------------------------------------------------- | ------- | ----------------------------- | ------------- | --------------------------------------------------------------------------------------------------------------------- |
| 9.1  | Implement `DrydenFilter.__init__` — filter state arrays, dt, V_ref                             | `[x]` | `environment/wind_model.py` | —            | env.md §3.3.4. States:`state_u` (scalar), `state_v` (2,), `state_w` (2,)                                       |
| 9.2  | Implement `DrydenFilter._compute_params(h)` — altitude-dependent scale lengths and intensities | `[x]` | `environment/wind_model.py` | 9.1           | env.md §3.3.3. Clamp `h >= 0.5` m                                                                                  |
| 9.3  | Implement `DrydenFilter.step(h, white_noise)` — advance filter one step                        | `[x]` | `environment/wind_model.py` | 9.1–9.2      | env.md §3.3.4. Returns `(u_g, v_g, w_g)`                                                                           |
| 9.4  | Implement `DrydenFilter.reset()` — zero all filter states                                      | `[x]` | `environment/wind_model.py` | 9.1           | —                                                                                                                    |
| 9.5  | Implement `WindModel.__init__` — compose DrydenFilter, store config                            | `[x]` | `environment/wind_model.py` | 9.3           | env.md §3.5                                                                                                          |
| 9.6  | Implement `WindModel._sample_mean_wind()` — uniform sample per episode                         | `[x]` | `environment/wind_model.py` | 9.5           | Horiz ±10 m/s, vert ±2 m/s                                                                                          |
| 9.7  | Implement `WindModel._setup_gust()` — Bernoulli event, random amplitude/timing                 | `[x]` | `environment/wind_model.py` | 9.5           | env.md §3.4. 10% prob, 3–8 m/s, horizontal bias                                                                     |
| 9.8  | Implement `WindModel.sample(t, h)` — return total wind `v_mean + v_turb + v_gust`            | `[x]` | `environment/wind_model.py` | 9.3, 9.6, 9.7 | env.md §3.1                                                                                                          |
| 9.9  | Implement `WindModel.reset()` — resample mean wind + gust params, reset Dryden                 | `[x]` | `environment/wind_model.py` | 9.5–9.7      | —                                                                                                                    |
| 9.10 | Write unit tests for wind model                                                                   | `[x]` | `tests/test_wind_model.py`  | 9.1–9.9      | Seeded reproducibility; zero config → zero wind; gust timing bounds; Dryden variance scales with intensity² |

---

## Stage 10 — Environment Model (Assembly)

> Top-level class composing Wind + Atmosphere. Ref: [env.md §2](Enviornment/env.md)

| #    | Task                                                                                            | Status  | Files                                | Depends On | Notes                                                        |
| ---- | ----------------------------------------------------------------------------------------------- | ------- | ------------------------------------ | ---------- | ------------------------------------------------------------ |
| 10.1 | Implement `EnvironmentModel.__init__` — compose WindModel + AtmosphereModel                  | `[x]` | `environment/environment_model.py` | 8.1, 9.5   | env.md §2.2                                                 |
| 10.2 | Implement `sample_at_state(t, p)` → `{'wind': (3,), 'rho': float, 'T': float, 'P': float}` | `[x]` | `environment/environment_model.py` | 10.1       | NED altitude:`h = -p[2]`                                   |
| 10.3 | Implement `reset(seed)` — reseed RNG, reset sub-models                                       | `[x]` | `environment/environment_model.py` | 10.1       | Ensures episode-level reproducibility                        |
| 10.4 | Write unit tests for environment model                                                          | `[x]` | `tests/test_environment_model.py`  | 10.1–10.3 | Correct dict keys/shapes; NED conversion; seeded determinism |

---

## Stage 11 — Vehicle Dynamics (Assembly)

> Top-level class: owns state, integrator, all sub-models. Ref: [vehicle.md §8](Vehicle%20Dynamics/vehicle.md)

| #     | Task                                                                                                    | Status  | Files                              | Depends On                         | Notes                                                                                         |
| ----- | ------------------------------------------------------------------------------------------------------- | ------- | ---------------------------------- | ---------------------------------- | --------------------------------------------------------------------------------------------- |
| 11.1  | Implement `VehicleDynamics.__init__` — load config, init all sub-models, accept `EnvironmentModel` | `[x]` | `dynamics/vehicle.py`            | 2.2, 3.1, 4.1, 5.1, 6.1, 7.1, 10.1 | vehicle.md §8.2. State: 18 scalars `[p(3), v_b(3), q(4), omega(3), T(1), delta_actual(4)]` |
| 11.2  | Implement `_unpack(y)` — extract state components from flat array                                    | `[x]` | `dynamics/vehicle.py`            | 11.1                               | `p, v_b, q, omega, T, delta_actual`                                                         |
| 11.3  | Implement `derivs(y, u, t)` — complete derivative function                                           | `[x]` | `dynamics/vehicle.py`            | 11.1–11.2                         | vehicle.md Appendix A. Query env once, compute all forces/torques, assemble 18-dim derivative |
| 11.4  | Implement `step(u)` — RK4 integration with quaternion normalization                                  | `[x]` | `dynamics/vehicle.py`            | 11.3, 7.1                          | vehicle.md §7.5. 5 substeps per policy step at 40 Hz / 200 Hz physics                        |
| 11.5  | Implement `reset(initial_state, seed)` — set state, reset sub-models                                 | `[x]` | `dynamics/vehicle.py`            | 11.1                               | vehicle.md §8.2 reset method                                                                 |
| 11.6  | Write integration tests: free fall                                                                      | `[x]` | `tests/test_vehicle_dynamics.py` | 11.1–11.5                         | T=0, q=[1,0,0,0] → z increases at g; v_b[2] = g*t                                            |
| 11.7  | Write integration tests: hover equilibrium                                                              | `[x]` | `tests/test_vehicle_dynamics.py` | 11.1–11.5                         | T=m*g → derivatives ≈ 0, state stationary                                                   |
| 11.8  | Write integration tests: gyroscopic precession                                                          | `[x]` | `tests/test_vehicle_dynamics.py` | 11.1–11.5                         | High RPM + small pitch rate → roll torque matches ω × h_fan                                |
| 11.9  | Write integration tests: wind step response                                                             | `[x]` | `tests/test_vehicle_dynamics.py` | 11.1–11.5                         | Sudden 10 m/s crosswind → lateral drift onset                                                |
| 11.10 | Write energy conservation test                                                                          | `[x]` | `tests/test_vehicle_dynamics.py` | 11.1–11.5                         | Torque-free, drag-free: E(t) ≈ E(0) to <1e-6 relative error over 100 s                       |

---

## Stage 12 — Gymnasium Environment Wrapper

> `EDFLandingEnv` wrapping Vehicle + Environment for RL. Ref: [training.md §2](Training%20Plan/training.md)

| #    | Task                                                                                              | Status  | Files                             | Depends On | Notes                                                                                                 |
| ---- | ------------------------------------------------------------------------------------------------- | ------- | --------------------------------- | ---------- | ----------------------------------------------------------------------------------------------------- |
| 12.1 | Implement `EDFLandingEnv.__init__` — define obs/action spaces, load configs                    | `[x]` | `training/edf_landing_env.py`   | 11.1, 10.1 | Obs: Box(20,), Action: Box(5,). 40 Hz policy, 200 Hz physics (5 substeps)                             |
| 12.2 | Implement `_scale_action(action)` — normalized [-1,1] → physical `[T_cmd, delta_1..4]`      | `[x]` | `training/edf_landing_env.py`   | 12.1       | training.md §4.2–4.3. Thrust centered at hover                                                      |
| 12.3 | Implement `_sample_initial_conditions()` — randomized IC per episode                           | `[x]` | `training/edf_landing_env.py`   | 12.1       | training.md §6.1. Alt 5–10 m, lateral ±2 m, descent 0–3 m/s, tilt ±5°                           |
| 12.4 | Implement `reset(seed)` — reset vehicle, env, sample ICs                                       | `[x]` | `training/edf_landing_env.py`   | 12.1–12.3 | Seed both vehicle and env for reproducibility                                                         |
| 12.5 | Implement `step(action)` — scale action, run substeps, compute obs/reward/terminated/truncated | `[x]` | `training/edf_landing_env.py`   | 12.2       | training.md §2.1. Clip action to [-1,1]                                                              |
| 12.6 | Implement `_check_terminated()` — landing, crash, extreme tilt, OOB conditions                 | `[x]` | `training/edf_landing_env.py`   | 12.5       | training.md §6.3. Landed: h<0.05, v<0.5, θ<15°, ω<0.5                                             |
| 12.7 | Implement `_get_info()` — logging dict with position, velocity, CEP, etc.                      | `[x]` | `training/edf_landing_env.py`   | 12.5       | training.md §6.5                                                                                     |
| 12.8 | Implement ground contact spring-damper model                                                      | `[x]` | `training/edf_landing_env.py`   | 12.5       | training.md §6.4. k=10000 N/m, c=500 N·s/m, 5 settling substeps                                     |
| 12.9 | Write env wrapper tests                                                                           | `[x]` | `tests/test_edf_landing_env.py` | 12.1–12.8 | Space shapes correct; reset produces valid obs; step runs without error; terminated on ground contact |

---

## Stage 13 — Observation Pipeline

> Sensor noise injection, body-frame observations. Ref: [training.md §3](Training%20Plan/training.md)

| #    | Task                                                                   | Status  | Files                           | Depends On | Notes                                                                                                                           |
| ---- | ---------------------------------------------------------------------- | ------- | ------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------- |
| 13.1 | Implement `_get_obs()` — compute 20-dim observation from true state | `[x]` | `training/observation.py`     | 12.1       | training.md §3.2. Target offset in body frame, gravity direction, angular vel, TWR, wind EMA, alt, speed, ang_speed, time_frac |
| 13.2 | Implement sensor noise injection: Gaussian per component               | `[x]` | `training/observation.py`     | 13.1       | training.md §3.5. Configurable σ values from BNO085 datasheet                                                                 |
| 13.3 | Implement wind EMA estimate:`α=0.05` exponential moving average     | `[x]` | `training/observation.py`     | 13.1       | training.md §3.3 wind estimate rationale                                                                                       |
| 13.4 | Integrate observation module into `EDFLandingEnv._get_obs()`         | `[x]` | `training/edf_landing_env.py` | 13.1–13.3 | Replace placeholder with actual observation code                                                                                |
| 13.5 | Write observation unit tests                                           | `[x]` | `tests/test_observation.py`   | 13.1–13.3 | Correct shape (20,); noise=0 → matches true state; noise stdev ≈ configured σ over N samples                                 |

---

## Stage 14 — Reward Function

> Multi-objective, potential-based shaping + terminal rewards. Ref: [training.md §5](Training%20Plan/training.md)

| #     | Task                                                                                       | Status  | Files                    | Depends On  | Notes                                                                                                |
| ----- | ------------------------------------------------------------------------------------------ | ------- | ------------------------ | ----------- | ---------------------------------------------------------------------------------------------------- |
| 14.1  | Implement `RewardFunction.__init__` — load all weights from config                      | `[x]` | `training/reward.py`   | —          | training.md §5.5 weight table                                                                       |
| 14.2  | Implement alive bonus:`+0.1` per step                                                    | `[x]` | `training/reward.py`   | 14.1        | training.md §5.3.1                                                                                  |
| 14.3  | Implement potential-based shaping: `Φ(s) = -c_d*||e_p|| - c_v*||v_b||`, `r_shape = γΦ(s_{t+1}) - Φ(s_t)` | `[x]` | `training/reward.py`   | 14.1        | training.md §5.3.2                                                                                  |
| 14.4  | Implement orientation penalty:`-w_θ * (1 - g_body_z)`                                   | `[x]` | `training/reward.py`   | 14.1        | training.md §5.3.3                                                                                  |
| 14.5  | Implement jerk penalty: finite-difference `a_dot`, normalized by j_ref=10                | `[x]` | `training/reward.py`   | 14.1        | training.md §5.3.4. Skip first 2 steps                                                              |
| 14.6  | Implement fuel penalty: `-w_f * |T_cmd|/T_max * dt_policy`                              | `[x]` | `training/reward.py`   | 14.1        | training.md §5.3.5                                                                                  |
| 14.7  | Implement action smoothness penalty: `-w_a * ||a_t - a_{t-1}||`                          | `[x]` | `training/reward.py`   | 14.1        | training.md §5.3.6                                                                                  |
| 14.8  | Implement terminal rewards: success (+100), precision Gaussian (+50), soft touchdown (+20) | `[x]` | `training/reward.py`   | 14.1        | training.md §5.4.1–5.4.3                                                                           |
| 14.9  | Implement terminal penalties: crash (-100), OOB (-50)                                      | `[x]` | `training/reward.py`   | 14.1        | training.md §5.4.4–5.4.5                                                                           |
| 14.10 | Implement `RewardFunction.reset()` — clear prev_potential, prev_velocity, prev_accel    | `[x]` | `training/reward.py`   | 14.1        | New episode state                                                                                    |
| 14.11 | Create `configs/reward.yaml` — all reward weights                                       | `[x]` | `configs/reward.yaml`  | —          | From training.md §15.2                                                                              |
| 14.12 | Write reward function unit tests                                                           | `[x]` | `tests/test_reward.py` | 14.1–14.10 | Hovering at target → positive shaping; crash → large penalty; perfect landing → high total reward |

---

## Stage 15 — Domain Randomization Additions

> Actuator delay DR, observation latency augmentation. Ref: [training.md §6.2](Training%20Plan/training.md)

| #    | Task                                                 | Status  | Files                             | Depends On | Notes                                                                                 |
| ---- | ---------------------------------------------------- | ------- | --------------------------------- | ---------- | ------------------------------------------------------------------------------------- |
| 15.1 | Implement actuator delay buffer in `EDFLandingEnv` | `[x]` | `training/edf_landing_env.py`   | 12.5       | training.md §6.2.1. Buffer action for N policy steps. ESC 10–40 ms, servo 5–20 ms  |
| 15.2 | Implement observation latency augmentation           | `[x]` | `training/edf_landing_env.py`   | 13.4       | training.md §6.2.2. Stale obs 0–3 policy steps, randomized per episode              |
| 15.3 | Make DR features toggleable via config flags         | `[x]` | `training/edf_landing_env.py`   | 15.1–15.2 | `actuator_delay.enabled`, `obs_latency.enabled`                                   |
| 15.4 | Write tests for DR features                          | `[x]` | `tests/test_edf_landing_env.py` | 15.1–15.3 | Delay buffer delays action by correct steps; latency returns stale obs                |

---

## Stage 16 — Controller Base Class

> Shared interface for all controller variants. Ref: [training.md §2.3](Training%20Plan/training.md)

| #    | Task                                                                                  | Status  | Files                            | Depends On | Notes             |
| ---- | ------------------------------------------------------------------------------------- | ------- | -------------------------------- | ---------- | ----------------- |
| 16.1 | Implement `Controller` ABC — `get_action(obs)`, `reset()`, `update_memory()` | `[x]` | `training/controllers/base.py` | —         | training.md §2.3 |

---

## Stage 17 — PID Controller (Baseline)

> Classical control baseline. Ref: [training.md §7.3](Training%20Plan/training.md)

| #    | Task                                                                      | Status  | Files                                      | Depends On | Notes                                                                     |
| ---- | ------------------------------------------------------------------------- | ------- | ------------------------------------------ | ---------- | ------------------------------------------------------------------------- |
| 17.1 | Implement outer loop: altitude PID → throttle command                    | `[x]` | `training/controllers/pid_controller.py` | 16.1       | training.md §7.3.2. PID with anti-windup                                 |
| 17.2 | Implement outer loop: lateral PD → desired roll/pitch angles             | `[x]` | `training/controllers/pid_controller.py` | 17.1       | Clamp desired angles ±20°                                               |
| 17.3 | Implement inner loop: attitude PD → fin deflections                      | `[x]` | `training/controllers/pid_controller.py` | 17.2       | training.md §7.3.3. Map pitch→fins 1/2, roll→fins 3/4                  |
| 17.4 | Implement `PIDController.reset()` — zero integral + derivative states  | `[x]` | `training/controllers/pid_controller.py` | 17.1–17.3 | —                                                                        |
| 17.5 | Implement `PIDController.get_action(obs)` — output normalized [-1,1]^5 | `[x]` | `training/controllers/pid_controller.py` | 17.1–17.3 | Appendix B full implementation                                            |
| 17.6 | Create `configs/pid.yaml` — initial gain values                        | `[x]` | `configs/pid.yaml`                       | —         | training.md §15.5                                                        |
| 17.7 | Write PID controller tests                                                | `[x]` | `tests/test_pid_controller.py`           | 17.1–17.6 | Hover → near-zero fin commands; position error → correct tilt direction |

---

## Stage 18 — PID Gain Tuning

> Systematic tuning via linearization + grid search. Ref: [training.md §8.4](Training%20Plan/training.md)

| #    | Task                                                                       | Status  | Files                            | Depends On | Notes                                                   |
| ---- | -------------------------------------------------------------------------- | ------- | -------------------------------- | ---------- | ------------------------------------------------------- |
| 18.1 | Implement linearization script: numerical Jacobians A, B at hover trim     | `[x]` | `training/scripts/tune_pid.py` | 11.1, 17.1 | training.md §8.4 Step 1. Finite-difference Jacobians   |
| 18.2 | Implement Ziegler-Nichols tuning per axis (altitude, lateral, attitude)    | `[x]` | `training/scripts/tune_pid.py` | 18.1       | Increase K_p → find K_u, T_u → compute PID gains      |
| 18.3 | Implement grid search: run N episodes per gain set, pick best success rate | `[x]` | `training/scripts/tune_pid.py` | 18.2, 12.1 | ±30% of Z-N gains, 5 steps per axis, 100 episodes each |
| 18.4 | Run PID tuning, save best gains to `configs/pid.yaml`                    | `[x]` | `configs/pid.yaml`             | 18.1–18.3 | Target: >50% success rate                               |

---

## Stage 19 — PPO-MLP Training (Primary Baseline)

> Prove RL can land. This is the critical first RL milestone. Ref: [training.md §7.1, §8.2](Training%20Plan/training.md)

| #    | Task                                                                       | Status  | Files                                 | Depends On | Notes                                                                     |
| ---- | -------------------------------------------------------------------------- | ------- | ------------------------------------- | ---------- | ------------------------------------------------------------------------- |
| 19.1 | Implement `PPOMlpController` wrapper for SB3                             | `[x]` | `training/controllers/ppo_mlp.py`   | 16.1       | Wraps `stable_baselines3.PPO` with `MlpPolicy`                        |
| 19.2 | Create `configs/ppo_mlp.yaml` — hyperparameters, architecture, schedule | `[x]` | `configs/ppo_mlp.yaml`              | —         | training.md §15.3. 2x256 MLP, tanh, ortho init                           |
| 19.3 | Install RL training deps:`stable-baselines3`, `torch`, `tensorboard` | `[x]` | `requirements.txt`                  | —         | `stable-baselines3>=2.0`, `torch>=2.0`                                |
| 19.4 | Write training script: vectorized envs + SB3 PPO + logging                 | `[x]` | `training/scripts/train_ppo_mlp.py` | 19.1, 12.1 | training.md §13.3.1.`SubprocVecEnv`, `VecNormalize`, `TensorBoard` |
| 19.5 | Add checkpointing: save every 500K steps, save best model                  | `[x]` | `training/scripts/train_ppo_mlp.py` | 19.4       | training.md §13.3.2                                                      |
| 19.6 | Add evaluation callback: 50 episodes every 100K steps                      | `[x]` | `training/scripts/train_ppo_mlp.py` | 19.4       | Log success rate, CEP                                                     |

### 19.M — PPO-MLP Milestones (Training Checkpoints)

| #     | Milestone                | Status  | Criterion                                                 | Action If Not Met                             |
| ----- | ------------------------ | ------- | --------------------------------------------------------- | --------------------------------------------- |
| 19.M1 | Agent learns to hover    | `[ ]` | Mean episode length > 5 s at ~1M steps                    | Debug reward shaping, check obs normalization |
| 19.M2 | Agent lands consistently | `[ ]` | Success rate > 50% at ~3M steps                           | Tune hyperparameters, check action scaling    |
| 19.M3 | Agent lands precisely    | `[ ]` | CEP < 0.5 m, success > 90% at ~7M steps                   | Add curriculum (Stage 22), increase DR        |
| 19.M4 | Baseline converged       | `[ ]` | CEP < 0.1 m, success > 99%, jerk < 10 m/s³ at ~10M steps | Proceed to Stage 20                           |

---

## Stage 20 — SCP Controller (Optimization Baseline)

> Trajectory optimization via successive convexification. Ref: [training.md §7.4](Training%20Plan/training.md)

| #    | Task                                                                                         | Status  | Files                                      | Depends On | Notes                                                              |
| ---- | -------------------------------------------------------------------------------------------- | ------- | ------------------------------------------ | ---------- | ------------------------------------------------------------------ |
| 20.1 | Install CVXPY + ECOS:`cvxpy>=1.4`, `ecos>=2.0`                                           | `[ ]` | `requirements.txt`                       | —         | SOCP solver                                                        |
| 20.2 | Implement SCP problem formulation: minimize total impulse, subject to dynamics + constraints | `[ ]` | `training/controllers/scp_controller.py` | 16.1       | training.md §7.4.1–7.4.2. Lossless thrust relaxation             |
| 20.3 | Implement trajectory discretization: N=50 nodes, trust region                                | `[ ]` | `training/controllers/scp_controller.py` | 20.2       | training.md §7.4.3                                                |
| 20.4 | Implement Mode A (open-loop): solve once at episode start, execute plan                      | `[ ]` | `training/controllers/scp_controller.py` | 20.3       | Tests best-case optimization                                       |
| 20.5 | Implement Mode B (MPC): re-solve at 2 Hz with updated state                                  | `[ ]` | `training/controllers/scp_controller.py` | 20.4       | More robust to disturbances                                        |
| 20.6 | Implement `SCPController.get_action(obs)` → [-1,1]^5 output                               | `[ ]` | `training/controllers/scp_controller.py` | 20.4–20.5 | Normalize SCP output to match env action space                     |
| 20.7 | Create `configs/scp.yaml` — solver params, trust region, constraints                      | `[ ]` | `configs/scp.yaml`                       | —         | training.md §15.6                                                 |
| 20.8 | Write SCP configuration/validation script                                                    | `[ ]` | `training/scripts/configure_scp.py`      | 20.2–20.6 | Test convergence on nominal case; profile solve time               |
| 20.9 | Write SCP tests                                                                              | `[ ]` | `tests/test_scp_controller.py`           | 20.2–20.6 | Nominal case converges; constraints satisfied; solve time < 100 ms |

---

## Stage 21 — Evaluation Framework

> Shared evaluation protocol for all controllers. Ref: [training.md §11–12](Training%20Plan/training.md)

| #    | Task                                                                                                    | Status  | Files                            | Depends On | Notes                                              |
| ---- | ------------------------------------------------------------------------------------------------------- | ------- | -------------------------------- | ---------- | -------------------------------------------------- |
| 21.1 | Implement evaluation runner: run N episodes with fixed seeds, collect metrics                           | `[ ]` | `training/scripts/evaluate.py` | 12.1, 16.1 | training.md §11.2.1. 100 episodes × 4 conditions |
| 21.2 | Implement metric computation: CEP, touchdown velocity, success rate, jerk 99th pctl, ΔV, recovery time | `[ ]` | `training/scripts/evaluate.py` | 21.1       | training.md §11.1                                 |
| 21.3 | Implement statistical comparison: paired t-tests (Bonferroni), ANOVA, Cohen's d, 95% CI                 | `[ ]` | `training/scripts/compare.py`  | 21.2       | training.md §11.3. Paired by episode seed         |
| 21.4 | Implement comparison table output (markdown + CSV)                                                      | `[ ]` | `training/scripts/compare.py`  | 21.3       | training.md §12.1 template                        |
| 21.5 | Implement visualization: trajectory plots, landing scatter, reward curves                               | `[ ]` | `training/scripts/compare.py`  | 21.2       | master_plan.md §3.6. Matplotlib                   |
| 21.6 | Create `configs/evaluation.yaml` — conditions, seeds, n_episodes                                     | `[ ]` | `configs/evaluation.yaml`      | —         | 4 conditions: nominal, windy, full DR, easy        |

---

## Stage 22 — Curriculum Learning (Optional)

> Enable only if uniform DR fails at 5M steps. Ref: [training.md §10](Training%20Plan/training.md)

| #    | Task                                                                                               | Status  | Files                           | Depends On | Notes                                                               |
| ---- | -------------------------------------------------------------------------------------------------- | ------- | ------------------------------- | ---------- | ------------------------------------------------------------------- |
| 22.1 | Implement `CurriculumScheduler` — linear difficulty ramp over training progress                 | `[ ]` | `training/curriculum.py`      | —         | training.md §10.3. Easy→Medium→Hard over 0→30%→70%→100% steps |
| 22.2 | Integrate curriculum with `EDFLandingEnv.reset()` — adjust IC ranges, wind, noise by difficulty | `[ ]` | `training/edf_landing_env.py` | 22.1       | Optional flag:`curriculum.enabled`                                |
| 22.3 | Implement automatic difficulty adjustment alternative                                              | `[ ]` | `training/curriculum.py`      | 22.1       | training.md §10.4. Based on rolling success rate                   |

---

## Stage 23 — Hyperparameter Search (PPO-MLP)

> Ray Tune ASHA for automated optimization. Ref: [training.md §9](Training%20Plan/training.md)

| #    | Task                                                              | Status  | Files                                 | Depends On | Notes                                       |
| ---- | ----------------------------------------------------------------- | ------- | ------------------------------------- | ---------- | ------------------------------------------- |
| 23.1 | Install Ray Tune:`ray[tune]>=2.9`                               | `[ ]` | `requirements.txt`                  | —         | —                                          |
| 23.2 | Implement sweep script: define search space per training.md §9.2 | `[ ]` | `training/scripts/sweep_hparams.py` | 19.4       | LR, batch, GAE λ, entropy coef, clip range |
| 23.3 | Configure ASHA scheduler: 50 trials, 3M steps/trial, grace 500K   | `[ ]` | `training/scripts/sweep_hparams.py` | 23.2       | training.md §9.4                           |
| 23.4 | Run PPO-MLP hyperparameter sweep                                  | `[ ]` | —                                    | 23.2–23.3 | ~72 GPU-hours budget                        |
| 23.5 | Update `configs/ppo_mlp.yaml` with best hyperparameters         | `[ ]` | `configs/ppo_mlp.yaml`              | 23.4       | —                                          |
| 23.6 | Retrain PPO-MLP with best hparams for full 10M steps              | `[ ]` | —                                    | 23.5       | Final baseline checkpoint                   |

---

## Stage 24 — Baseline Evaluation & Comparison

> Evaluate all baselines before proceeding to GTrXL. Ref: [training.md §12](Training%20Plan/training.md)

| #    | Task                                                                                             | Status  | Files | Depends On       | Notes                                                                                                              |
| ---- | ------------------------------------------------------------------------------------------------ | ------- | ----- | ---------------- | ------------------------------------------------------------------------------------------------------------------ |
| 24.1 | Evaluate PPO-MLP: 400 episodes (4 conditions × 100)                                             | `[ ]` | —    | 19.M4, 21.1      | Record all metrics                                                                                                 |
| 24.2 | Evaluate PID: 400 episodes (same seeds)                                                          | `[ ]` | —    | 18.4, 21.1       | —                                                                                                                 |
| 24.3 | Evaluate SCP (open-loop + MPC): 400 episodes each                                                | `[ ]` | —    | 20.8, 21.1       | —                                                                                                                 |
| 24.4 | Generate baseline comparison table                                                               | `[ ]` | —    | 24.1–24.3, 21.4 | PPO-MLP vs PID vs SCP                                                                                              |
| 24.5 | **Decision gate**: Does PPO-MLP pass deletion criterion? (>95% success, CEP<0.15, jerk<10) | `[ ]` | —    | 24.4             | If YES → GTrXL may not be needed (training.md §7.1.3). Proceed to Stage 25 regardless to test thesis hypothesis. |

---

## Stage 25 — GTrXL-PPO Implementation

> Only after baselines work. Attention-based temporal RL. Ref: [training.md §7.2](Training%20Plan/training.md)

| #     | Task                                                                                         | Status  | Files                                   | Depends On | Notes                                                                           |
| ----- | -------------------------------------------------------------------------------------------- | ------- | --------------------------------------- | ---------- | ------------------------------------------------------------------------------- |
| 25.1  | Install RLlib:`ray[rllib]>=2.9`                                                            | `[ ]` | `requirements.txt`                    | —         | Native attention support                                                        |
| 25.2  | Implement GTrXL encoder: linear projection → L transformer layers with gated self-attention | `[ ]` | `training/controllers/gtrxl_ppo.py`   | 16.1       | training.md §7.2.1. d_model=128, L=2, H=4, gate_bias=-2                        |
| 25.3  | Implement relative positional encoding (Dai et al., 2019)                                    | `[ ]` | `training/controllers/gtrxl_ppo.py`   | 25.2       | No absolute positions                                                           |
| 25.4  | Implement segment-level recurrence: cache hidden states, carry forward with stop-gradient    | `[ ]` | `training/controllers/gtrxl_ppo.py`   | 25.2       | training.md §7.2.3. Seg_len=64, mem_len=64                                     |
| 25.5  | Implement gating mechanism per layer (Parisotto et al., 2020)                                | `[ ]` | `training/controllers/gtrxl_ppo.py`   | 25.2       | `σ(W_g·[x, attn(x)]) ⊙ attn(x) + (1-σ) ⊙ x`, gate_bias=-2 (start closed) |
| 25.6  | Implement policy head + value head on top of GTrXL encoder                                   | `[ ]` | `training/controllers/gtrxl_ppo.py`   | 25.2       | d_model → 256 → 5 mean + 5 log_std (policy); d_model → 256 → 1 (value)      |
| 25.7  | Implement memory burn-in: skip gradients for first 20 steps of each episode                  | `[ ]` | `training/controllers/gtrxl_ppo.py`   | 25.4       | training.md §8.3 note 2. Mandatory per Parisotto et al.                        |
| 25.8  | Implement LR warm-up schedule: linear warm-up for first 50 iterations then linear decay      | `[ ]` | `training/controllers/gtrxl_ppo.py`   | 25.2       | training.md §8.3                                                               |
| 25.9  | Create `configs/gtrxl_ppo.yaml` — full config per training.md §15.4                      | `[ ]` | `configs/gtrxl_ppo.yaml`              | —         | —                                                                              |
| 25.10 | Write training script: RLlib PPO with attention model config                                 | `[ ]` | `training/scripts/train_gtrxl_ppo.py` | 25.2–25.8 | training.md §13.3.1 RLlib pattern                                              |
| 25.11 | Add checkpointing + evaluation callbacks (same as PPO-MLP)                                   | `[ ]` | `training/scripts/train_gtrxl_ppo.py` | 25.10      | Save every 500K, eval every 100K                                                |

---

## Stage 26 — GTrXL-PPO Training & Tuning

> Train and optimize GTrXL-PPO after baselines are validated. Ref: [training.md §8.3, §9.3](Training%20Plan/training.md)

| #    | Task                                                                      | Status  | Files                                 | Depends On  | Notes                                        |
| ---- | ------------------------------------------------------------------------- | ------- | ------------------------------------- | ----------- | -------------------------------------------- |
| 26.1 | Run initial GTrXL-PPO training: 3M steps with default hparams             | `[ ]` | —                                    | 25.10       | Sanity check: does it learn at all?          |
| 26.2 | Implement GTrXL hyperparameter sweep: d_model, n_layers, n_heads, seg_len | `[ ]` | `training/scripts/sweep_hparams.py` | 23.2, 25.10 | training.md §9.3. 30 trials, 3M steps/trial |
| 26.3 | Run GTrXL-PPO hyperparameter sweep                                        | `[ ]` | —                                    | 26.2        | ~48 GPU-hours                                |
| 26.4 | Update `configs/gtrxl_ppo.yaml` with best hyperparameters               | `[ ]` | `configs/gtrxl_ppo.yaml`            | 26.3        | —                                           |
| 26.5 | Run full GTrXL-PPO training: 15M steps with best hparams                  | `[ ]` | —                                    | 26.4        | Final checkpoint                             |

---

## Stage 27 — Multi-Controller Evaluation & Comparison

> Full comparative analysis: all 4 controllers × 4 conditions. Ref: [training.md §11–12](Training%20Plan/training.md)

| #    | Task                                                                               | Status  | Files | Depends On             | Notes                              |
| ---- | ---------------------------------------------------------------------------------- | ------- | ----- | ---------------------- | ---------------------------------- |
| 27.1 | Evaluate GTrXL-PPO: 400 episodes (4 conditions × 100, same seeds as Stage 24)     | `[ ]` | —    | 26.5, 21.1             | —                                 |
| 27.2 | Generate full 4-controller comparison table                                        | `[ ]` | —    | 24.1–24.3, 27.1, 21.4 | training.md §12.1 template        |
| 27.3 | Run pairwise statistical tests: GTrXL vs each baseline (paired t-test, Bonferroni) | `[ ]` | —    | 27.2, 21.3             | —                                 |
| 27.4 | Run ANOVA across all 4 controllers                                                 | `[ ]` | —    | 27.2, 21.3             | Tukey HSD follow-up if significant |
| 27.5 | Compute effect sizes (Cohen's d) and 95% bootstrap CIs                             | `[ ]` | —    | 27.2                   | —                                 |
| 27.6 | Generate comparison plots: radar chart, box plots per metric per condition         | `[ ]` | —    | 27.2, 21.5             | —                                 |

---

## Stage 28 — Robustness Characterization

> Wind sweeps, disturbance rejection, sensor sensitivity. Ref: [training.md §11.4](Training%20Plan/training.md)

| #    | Task                                                                          | Status  | Files                            | Depends On | Notes                                                      |
| ---- | ----------------------------------------------------------------------------- | ------- | -------------------------------- | ---------- | ---------------------------------------------------------- |
| 28.1 | Wind robustness sweep: 0–15 m/s in 1 m/s steps × 50 episodes per level      | `[ ]` | `training/scripts/evaluate.py` | 27.1, 21.1 | Plot success rate vs. wind speed. Define robustness margin |
| 28.2 | Disturbance rejection: inject 5 m/s step gust at t=3 s, measure recovery time | `[ ]` | `training/scripts/evaluate.py` | 27.1       | Compare across all controllers                             |
| 28.3 | Sensor noise sensitivity: sweep noise scale 0×–3× nominal × 50 episodes   | `[ ]` | `training/scripts/evaluate.py` | 27.1       | Plot CEP vs. noise scale                                   |

---

## Stage 29 — Ablation Studies

> Systematic feature importance analysis. Ref: [training.md §12.3](Training%20Plan/training.md)

| #    | Task                                                               | Status  | Files | Depends On | Notes                                                    |
| ---- | ------------------------------------------------------------------ | ------- | ----- | ---------- | -------------------------------------------------------- |
| 29.1 | Ablation: remove wind estimate from obs → retrain PPO-MLP + GTrXL | `[ ]` | —    | 27.1       | Does explicit wind info help MLP more than GTrXL?        |
| 29.2 | Ablation: remove jerk penalty → retrain both RL controllers       | `[ ]` | —    | 27.1       | Does jerk penalty hurt success rate?                     |
| 29.3 | Ablation: reduce GTrXL segment length (32 vs 64 vs 128)            | `[ ]` | —    | 27.1       | Minimum useful context window                            |
| 29.4 | Ablation: remove gating (GTrXL → TrXL)                            | `[ ]` | —    | 27.1       | Does gating matter for RL stability?                     |
| 29.5 | Ablation: double sensor noise                                      | `[ ]` | —    | 27.1       | Robustness to degraded sensors                           |
| 29.6 | Ablation: train without wind, test with wind                       | `[ ]` | —    | 27.1       | DR value-add measurement                                 |
| 29.7 | Ablation: remove actuator delay DR                                 | `[ ]` | —    | 27.1       | Hwangbo sim-to-real value check                          |
| 29.8 | Ablation: Dryden turbulence vs. white noise                        | `[ ]` | —    | 27.1       | Delete Dryden if white noise matches ±1% success        |
| 29.9 | Ablation: curriculum vs. uniform DR                                | `[ ]` | —    | 27.1       | Delete curriculum if uniform achieves M3 within 7M steps |

---

## Stage 30 — Model Export Pipeline

> ONNX/TensorRT for Jetson Nano deployment. Ref: [training.md §14](Training%20Plan/training.md)

| #    | Task                                                                                 | Status  | Files                               | Depends On  | Notes                                                |
| ---- | ------------------------------------------------------------------------------------ | ------- | ----------------------------------- | ----------- | ---------------------------------------------------- |
| 30.1 | Freeze observation normalization stats from training                                 | `[ ]` | —                                  | 19.M4, 26.5 | VecNormalize saved stats                             |
| 30.2 | Export PPO-MLP to ONNX (opset 17+) via `torch.onnx.export`                         | `[ ]` | `training/scripts/export_onnx.py` | 30.1        | Straightforward — no dynamic state                  |
| 30.3 | Export GTrXL-PPO to ONNX: memory state as explicit input/output tensors              | `[ ]` | `training/scripts/export_onnx.py` | 30.1        | training.md §14.3. Pad attention mask               |
| 30.4 | Validate ONNX outputs vs. PyTorch: max error < 1e-4 per action dim over 100 episodes | `[ ]` | `training/scripts/export_onnx.py` | 30.2–30.3  | training.md §14.5                                   |
| 30.5 | Profile inference latency: PPO-MLP < 5 ms, GTrXL < 20 ms                             | `[ ]` | —                                  | 30.2–30.3  | On target hardware (Jetson Nano) or desktop estimate |

---

## Stage 31 — Visualization & Reporting

> Post-hoc analysis for thesis. Ref: [master_plan.md §3.6](master_plan.md)

| #    | Task                                                                        | Status  | Files | Depends On | Notes                                  |
| ---- | --------------------------------------------------------------------------- | ------- | ----- | ---------- | -------------------------------------- |
| 31.1 | Generate trajectory plots: top-down + side view for representative episodes | `[ ]` | —    | 27.1       | All 4 controllers on same episode seed |
| 31.2 | Generate landing scatter plots with CEP circle overlay                      | `[ ]` | —    | 27.1       | Per condition, per controller          |
| 31.3 | Generate reward/training curves from TensorBoard logs                       | `[ ]` | —    | 27.1       | PPO-MLP + GTrXL learning curves        |
| 31.4 | Generate wind realization vs. trajectory overlay                            | `[ ]` | —    | 27.1       | Disturbance analysis                   |
| 31.5 | Compile final comparison report for thesis                                  | `[ ]` | —    | 27.2–29.9 | master_plan.md §5.6 deliverables      |

---

## Decision Gates

> Key decision points that determine whether to proceed, pivot, or delete features.

| Gate                             | Location          | Trigger                                                    | Decision                                                                                                           |
| -------------------------------- | ----------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **G1: Phase 1 viable?**    | After Stage 19.M2 | PPO-MLP < 50% success after 1e7 steps                      | STOP. Audit dynamics, reward, obs space (master_plan.md §6.2)                                                     |
| **G2: GTrXL needed?**      | After Stage 24.5  | PPO-MLP > 95% success, CEP < 0.15 m, jerk < 10             | GTrXL may be unnecessary; temporal memory not needed (training.md §7.1.3). Still implement for thesis comparison. |
| **G3: Curriculum needed?** | After Stage 19.M3 | Uniform DR achieves M3 within 7M steps                     | Delete curriculum (training.md §10.5)                                                                             |
| **G4: Phase 2 (Isaac)?**   | After Stage 27    | Sample bottleneck OR contact fidelity gap OR HIL readiness | Enter Isaac integration (master_plan.md §4.2)                                                                     |
| **G5: Timeline check**     | After Stage 27    | > 4 weeks spent in Stages 0–27                            | Skip remaining ablations; ship Phase 1 results (master_plan.md §6.2)                                              |

---

## Critical Path Summary

```
Stage 0 (scaffolding, 0.5d)
    │
    ├── Stage 1 (quaternion, 0.5d)
    ├── Stage 8 (atmosphere, 0.5d)
    │
    ├── Stage 2 (mass props, 1d) ───┐
    ├── Stage 3 (thrust, 1d)        │
    ├── Stage 4 (aero, 0.5d) ◄─────┤
    ├── Stage 5 (fins, 1d)          │
    ├── Stage 6 (servo, 1d)         │
    ├── Stage 7 (integrator, 0.5d)  │
    ├── Stage 9 (wind, 1.5d)        │
    │                               │
    └── Stage 10 (env assembly, 0.5d) ◄── 8, 9
            │
            └── Stage 11 (vehicle assembly, 1.5d) ◄── 1–7, 10
                    │
                    ├── Stage 12 (gym env, 2d) ◄── 11
                    │       │
                    │       ├── Stage 13 (observation, 0.5d)
                    │       ├── Stage 14 (reward, 1d)
                    │       ├── Stage 15 (DR additions, 0.5d)
                    │       │
                    │       └── Stage 16 (controller base, 0.25d)
                    │               │
                    │               ├── Stage 17 (PID, 1d) ──► Stage 18 (tuning, 1d)
                    │               ├── Stage 19 (PPO-MLP, 3–5d) ─┐
                    │               ├── Stage 20 (SCP, 2d)        │
                    │               │                              │
                    │               └── Stage 21 (eval framework, 1.5d)
                    │                       │
                    │                       └── Stage 24 (baseline eval, 1d) ◄── 18, 19.M4, 20
                    │                               │
                    │                               ├── ★ Decision Gate G2 ★
                    │                               │
                    │                               └── Stage 25 (GTrXL impl, 3d)
                    │                                       │
                    │                                       └── Stage 26 (GTrXL train, 3–5d)
                    │                                               │
                    │                                               └── Stage 27 (full eval, 2d)
                    │                                                       │
                    │                                                       ├── Stage 28 (robustness, 1d)
                    │                                                       ├── Stage 29 (ablations, 3d)
                    │                                                       ├── Stage 30 (ONNX export, 1d)
                    │                                                       └── Stage 31 (viz/report, 1d)
```

**Estimated total**: ~4–6 weeks for Phase 1 + evaluation (Stages 0–31).

---

## Effort Estimates

| Stage Group              | Stages | Est. Days              | Notes                                                  |
| ------------------------ | ------ | ---------------------- | ------------------------------------------------------ |
| Scaffolding              | 0      | 0.5                    | Dirs + configs                                         |
| Core Physics Modules     | 1–7   | 5.5                    | Quaternion, mass, thrust, aero, fin, servo, integrator |
| Environment Modules      | 8–10  | 2.5                    | Atmosphere, wind, assembly                             |
| Vehicle Assembly + Tests | 11     | 1.5                    | Integration                                            |
| Gym Env + Training Infra | 12–16 | 4.25                   | Env wrapper, obs, reward, DR, controller base          |
| Baseline Controllers     | 17–20 | 4                      | PID + tuning, SCP                                      |
| PPO-MLP Training         | 19     | 3–5                   | Training wall-clock                                    |
| Evaluation Framework     | 21     | 1.5                    | Scripts + statistical tools                            |
| Baseline Evaluation      | 24     | 1                      | Run + compare                                          |
| GTrXL Implementation     | 25     | 3                      | Architecture + training script                         |
| GTrXL Training + Tuning  | 26     | 3–5                   | Sweep + full training                                  |
| Full Evaluation          | 27–29 | 4–6                   | 4-way comparison, robustness, ablations                |
| Export + Reporting       | 30–31 | 2                      | ONNX + plots                                           |
| **Total**          |        | **~35–43 days** | **~7–9 weeks**                                  |
