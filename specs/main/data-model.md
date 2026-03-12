# Data Model: Isaac Sim Mass Properties, Thrust Test & Environmental Forces

**Feature**: 002-isaac-mass-thrust-env
**Date**: 2026-03-11

## Entities

### MassPropertyReport

Represents the comparison result between USDC and YAML mass properties.

| Field | Type | Description |
|-------|------|-------------|
| yaml_mass | float | Total mass from YAML config (kg) |
| usd_mass | float | Total mass from USDC MassAPI (kg) |
| yaml_com_frd | tuple[3] | Center of mass in FRD body frame (m) |
| usd_com_zup | tuple[3] | Center of mass in Z-up prim frame (m) |
| usd_com_frd | tuple[3] | Center of mass converted to FRD (m) |
| yaml_inertia | float[3][3] | Inertia tensor from YAML (kg·m²) |
| usd_inertia | float[3][3] | Inertia tensor from USD (reconstructed from diagonal + principal axes) (kg·m²) |
| tolerance | float | Comparison tolerance (default 0.01 = 1%) |
| passed | bool | True if all comparisons within tolerance |
| discrepancies | list[Discrepancy] | Per-field comparison details |

### Discrepancy

| Field | Type | Description |
|-------|------|-------------|
| field_name | str | e.g., "total_mass", "com_x", "Ixx" |
| expected | float | YAML value |
| actual | float | USD value |
| relative_error | float | |expected - actual| / |expected| |
| within_tolerance | bool | relative_error <= tolerance |

### IsaacWindState

Per-environment wind state for Isaac Sim force application.

| Field | Type | Description |
|-------|------|-------------|
| wind_vector_world | Tensor[num_envs, 3] | Current wind velocity in world frame (m/s) |
| wind_ema | Tensor[num_envs, 3] | Exponential moving average for observation vector |
| gust_active | Tensor[num_envs] | Boolean, whether gust is currently active |
| gust_onset_step | Tensor[num_envs] | Step at which gust started |
| gust_duration_steps | Tensor[num_envs] | Duration of gust in steps |
| gust_vector | Tensor[num_envs, 3] | Gust velocity component (m/s) |

### IsaacWindConfig

Configuration for Isaac Sim wind model, loaded from `default_environment.yaml`.

| Field | Type | Description |
|-------|------|-------------|
| enabled | bool | Whether wind is active (default: false for backward compat) |
| mean_vector_range_lo | float[3] | Min mean wind per axis (m/s) |
| mean_vector_range_hi | float[3] | Max mean wind per axis (m/s) |
| gust_prob | float | Per-episode gust probability |
| gust_magnitude_range | float[2] | [min, max] gust speed (m/s) |
| drag_coefficient | float | Composite Cd for wind drag force |
| projected_area | float[3] | Projected area per axis (m²), from YAML _computed |

## State Transitions

### Episode Wind State

```
RESET → sample mean_wind from [lo, hi] uniform
      → sample gust_prob → if triggered: set gust_onset, duration, vector
      → wind_ema = zeros

STEP  → wind_vector = mean_wind + (gust_vector if gust_active else 0)
      → compute F_drag from relative wind
      → update wind_ema with exponential filter
      → obs[13:16] = wind_ema
```

### Mass Validation Workflow

```
LOAD → open USDC stage via pxr
     → read MassAPI from /Drone/Body prim
     → load YAML explicit mass properties

COMPARE → convert USD CoM from Z-up to FRD
        → reconstruct full inertia tensor from USD diagonal + principal axes
        → compare each field within tolerance

REPORT → print table of expected vs actual
       → exit 0 (pass) or 1 (fail)
```

---

## User Story 4: Gyro Precession Entities

### GyroPrecessionConfig

Configuration for gyroscopic precession, loaded from `default_vehicle.yaml` under `edf.gyro_precession`.

| Field | Type | Description |
|-------|------|-------------|
| enabled | bool | Whether precession torque is injected (default: true) |

**Source parameters** (already in `default_vehicle.yaml` under `edf`):

| Field | Type | Description |
|-------|------|-------------|
| I_fan | float | Fan rotor moment of inertia about spin axis (kg·m²) |
| k_thrust | float | Thrust coefficient N/(rad/s)² — used to derive ω_fan from thrust |

### GyroPrecessionState

Per-environment runtime state for gyro precession computation (GPU tensors, owned by `EdfLandingTask`).

| Field | Type | Description |
|-------|------|-------------|
| omega_fan | Tensor[num_envs] | Current fan angular velocity (rad/s), derived from `sqrt(thrust_actual / k_thrust)` |
| h_fan_body | Tensor[num_envs, 3] | Fan angular momentum in body frame: `[0, 0, I_fan * ω_fan]` |
| tau_gyro_body | Tensor[num_envs, 3] | Precession torque in body frame: `−ω_body × h_fan_body` |
| tau_gyro_world | Tensor[num_envs, 3] | Precession torque rotated to world frame for `set_external_force_and_torque` |

## State Transitions

### Precession Torque Per Step

```
_apply_action():
  IF gyro_precession.enabled:
    omega_fan = sqrt(thrust_actual / k_thrust)
    h_fan_b = [0, 0, I_fan * omega_fan]       # body frame
    omega_b = robot.data.root_ang_vel_b        # body frame
    tau_gyro_b = -cross(omega_b, h_fan_b)      # body frame
    tau_gyro_w = rotate_body_to_world(quat_w, tau_gyro_b)  # world frame
    torques[:, 0, :] += tau_gyro_w
```

## Relationships

- `MassPropertyReport` depends on `ExplicitMassProps` (from `parts_registry.py`) for YAML values
- `IsaacWindState` is owned by `EdfLandingTask` (one per task instance, batched over num_envs)
- `IsaacWindConfig` is loaded from `default_environment.yaml` via the existing `config_loader.py`
- `IsaacWindState.wind_ema` feeds directly into observation vector elements [13:16]
- `GyroPrecessionState` is owned by `EdfLandingTask`, computed per `_apply_action()` call
- `GyroPrecessionConfig.enabled` gates all precession computation — when false, zero overhead
- `GyroPrecessionConfig.I_fan` is read directly from `edf.I_fan` in `default_vehicle.yaml` — no USD prim required
