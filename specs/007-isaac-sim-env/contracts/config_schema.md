# Contract: Configuration Schema

**Feature**: 007-isaac-sim-env | **Date**: 2026-03-22

## Overview

All physics parameters, environment settings, reward weights, and disturbance configurations reside in YAML files under `simulation/isaac/configs/`. No magic numbers in source code. Each parameter is labeled with its source category.

## Parameter Source Labels

Every numerical parameter in config files MUST include a `_source` comment or field:

| Label | Meaning |
|-------|---------|
| `measured` | Value obtained from physical measurement of the hardware |
| `datasheet` | Value from manufacturer datasheet |
| `estimate` | Engineering estimate based on similar systems or calculations |
| `to-be-calibrated` | Placeholder value requiring bench test or system identification |
| `derived` | Computed from other parameters (formula documented) |

## Config File Inventory

### Vehicle Configuration

**File**: `configs/vehicle/edf_drone_v2.yaml`

```yaml
vehicle:
  total_mass: 2.5                # kg, source: measured
  body_com_offset: [0, 0, 0.01]  # m, source: estimate
  inertia_tensor:                 # kg·m², source: estimate
    Ixx: 0.015
    Iyy: 0.015
    Izz: 0.025

  fins:
    count: 4
    area: 0.002                  # m², source: measured
    max_deflection: 0.262        # rad (15°), source: measured
    cop_offset: [0, 0, -0.05]    # m from hinge, source: estimate
```

### Servo Parameters

**File**: `configs/params/servo_mg996r.yaml`

```yaml
servo:
  mass: 0.055                    # kg, source: datasheet
  stall_torque: 1.08             # N·m at 6V, source: datasheet
  transit_time_60deg: 0.14       # s at 6V, source: datasheet
  max_angular_velocity: 7.5      # rad/s, source: derived (π/3 / 0.14)
  tau_servo: 0.05                # s, source: estimate (tune from bench)
  deadband: 0.017                # rad (~1°), source: estimate
  max_command_angle: 0.262       # rad (15°), source: measured (linkage)
```

### EDF Parameters

**File**: `configs/params/edf_90mm.yaml`

```yaml
edf:
  max_thrust: 48.0               # N, source: estimate (artifact target)
  diameter: 0.09                 # m, source: measured
  k_T: null                      # source: to-be-calibrated
  k_Q: null                      # source: to-be-calibrated
  rotor_inertia: 0.0005          # kg·m², source: estimate
  tau_motor: 0.15                # s, source: estimate (90mm class)
  omega_max: null                # rad/s, source: to-be-calibrated
  d_omega_max: null              # rad/s², source: to-be-calibrated
```

### Task Configuration

**File**: `configs/tasks/hover.yaml`

```yaml
task:
  name: hover
  target_position: [0, 0, 5.0]  # m, world frame
  episode_length_s: 30.0

  spawn:
    position_range: [[−1, −1, 4], [1, 1, 6]]
    velocity_range: [[−0.5, −0.5, −0.5], [0.5, 0.5, 0.5]]
    attitude_range: [[−0.05, −0.05, −0.1], [0.05, 0.05, 0.1]]  # rad

  reward:
    alive_bonus: 1.0
    position_error: -2.0
    attitude_error: -1.0
    angular_velocity: -0.5
    control_effort: -0.1
    control_rate: -0.05
    hover_stability: 2.0
    drift_penalty: -1.5
    contact_penalty: -10.0

  success:
    max_position_error: 0.5      # m
    max_tilt: 0.26               # rad (15°)
    max_angular_rate: 1.0        # rad/s
    dwell_time: 3.0              # s

  termination:
    max_tilt: 1.57               # rad (90°)
    max_altitude_error: 10.0     # m
    crash: true
```

**File**: `configs/tasks/landing.yaml`

```yaml
task:
  name: landing
  target_position: [0, 0, 0.0]  # m, landing pad
  episode_length_s: 60.0

  spawn:
    position_range: [[−2, −2, 8], [2, 2, 12]]
    velocity_range: [[−1, −1, −2], [1, 1, 0]]
    attitude_range: [[−0.1, −0.1, −0.2], [0.1, 0.1, 0.2]]

  reward:
    alive_bonus: 0.5
    position_error: -1.0
    attitude_error: -1.5
    angular_velocity: -0.5
    control_effort: -0.1
    crash_penalty: -100.0
    touchdown_softness: 5.0
    landing_success: 50.0
    pad_accuracy: 10.0
    vertical_speed_shaping: -2.0

  success:
    state: LANDED
    max_pad_distance: 0.5        # m from pad center

  termination:
    max_tilt: 1.57
    max_altitude: 20.0
    crash: true
```

### Disturbance Configuration

**File**: `configs/disturbances/wind.yaml`

```yaml
disturbances:
  enabled: true

  wind:
    steady_vector: [2.0, 0.5, 0.0]  # m/s, world frame

  gust:
    enabled: true
    magnitude: 5.0               # m/s
    duration: 0.5                # s
    interval: [5.0, 15.0]       # s, min/max between gusts

  body_drag:
    cd: 1.0                      # source: estimate
    reference_area: 0.02         # m², source: estimate

  com_offset:
    enabled: false
    range: [[-0.005, -0.005, -0.005], [0.005, 0.005, 0.005]]  # m

  sensor_noise:
    enabled: false
    position_std: 0.01           # m
    velocity_std: 0.05           # m/s
    attitude_std: 0.005          # rad
    angular_velocity_std: 0.02   # rad/s
```

### Environment Configuration

**File**: `configs/env/single_env_debug.yaml`

```yaml
env:
  num_envs: 1
  env_spacing: 4.0
  dispatch_mode: per_link_force
  gizmos_enabled: true
  decimation: 4                  # physics substeps per RL step
  physics_dt: 0.00833           # 1/120 s
```

**File**: `configs/env/train_128.yaml`

```yaml
env:
  num_envs: 128
  env_spacing: 4.0
  dispatch_mode: per_link_force
  gizmos_enabled: false
  decimation: 4
  physics_dt: 0.00833
```

## Config Loading

All configs are loaded via a deep-merge loader that allows overriding individual fields:
1. Load base config (e.g., `edf_drone_v2.yaml`)
2. Overlay environment config (e.g., `train_128.yaml`)
3. Overlay task config (e.g., `hover.yaml`)
4. Overlay disturbance config (e.g., `wind.yaml`)
5. Command-line overrides (highest priority)

## Validation Rules

- All `to-be-calibrated` values that are `null` MUST be replaced before training runs
- Joint limits in config MUST match USD joint limits (validated by `asset_validator.py`)
- Mass/inertia in config MUST match USD values within 1% (validated by `mass_properties.py`)
- Reward weights MUST sum to a documented expected range per task
