# Data Model: Phase 1 Isaac Sim EDF TVC Simulation Environment

**Branch**: `007-isaac-sim-env` | **Date**: 2026-03-22

## Entities

### VehicleAsset

Represents the USD-defined articulated rigid body loaded into Isaac Sim.

| Field | Type | Description |
|-------|------|-------------|
| root_prim_path | string | USD prim path to articulation root |
| body_link_name | string | Name of the main body rigid body link |
| fin_link_names | string[4] | Names of four fin links (ordered: +X, +Y, -X, -Y) |
| fin_joint_names | string[4] | Names of four revolute joints |
| hinge_axes | vec3[4] | Local-frame hinge axis per joint |
| joint_lower_limits | float[4] | Lower deflection limits (radians) |
| joint_upper_limits | float[4] | Upper deflection limits (radians) |
| fin_cop_positions | vec3[4] | Center of pressure positions in fin-local frame |
| fin_chord_directions | vec3[4] | Chord direction vectors in fin-local frame |
| fin_normal_directions | vec3[4] | Normal direction at neutral angle, fin-local frame |
| edf_thrust_axis | vec3 | Thrust axis in body frame |
| rotor_spin_axis | vec3 | Rotor rotation axis in body frame |
| landing_contact_regions | string[] | Prim paths for landing-gear colliders |
| total_mass | float | Total vehicle mass (kg) |
| body_inertia | mat3x3 | Body inertia tensor (kg·m²) |
| body_com_offset | vec3 | Center-of-mass offset from body origin (m) |

**Source**: Loaded from USD asset + `edf_drone_v2.asset.yaml` metadata file.
**Validation**: `asset_validator.py` verifies all fields are present and physically valid at environment init.

---

### ServoActuator

Models MG996R-class servo dynamics per fin.

| Field | Type | Source Label | Description |
|-------|------|-------------|-------------|
| mass | float | datasheet | Mass per servo (kg), default 0.055 |
| stall_torque | float | datasheet | Stall torque at 6V (N·m), default 1.08 |
| transit_time_60deg | float | datasheet | 60° transit time at 6V (s), default 0.14 |
| max_angular_velocity | float | derived | Max angular velocity (rad/s), default 7.5 |
| tau_servo | float | estimate | First-order lag time constant (s), default 0.05 |
| deadband | float | estimate | Deadband half-width (rad), default 0.017 |
| max_command_angle | float | measured | Max deflection angle (rad), config-defined |

**State per fin per env**:
| Field | Type | Description |
|-------|------|-------------|
| actual_angle | float | Current servo position (rad) |
| actual_rate | float | Current angular velocity (rad/s) |
| commanded_angle | float | Target position from action (rad) |

**Dynamics**: First-order lag with rate limiting: `ẋ = clamp((x_cmd - x) / τ, -ω_max, ω_max)`

---

### EDFPropulsion

Models the 90mm EDF thrust source with spool dynamics.

| Field | Type | Source Label | Description |
|-------|------|-------------|-------------|
| max_thrust | float | estimate | Static thrust at full command (N), default 48 |
| diameter | float | measured | EDF diameter (m), default 0.09 |
| k_T | float | to-be-calibrated | Thrust coefficient |
| k_Q | float | to-be-calibrated | Torque coefficient |
| rotor_inertia | float | estimate | Fan moment of inertia (kg·m²) |
| tau_motor | float | estimate | Spool time constant (s), default 0.15 |
| omega_max | float | to-be-calibrated | Max rotor speed (rad/s) |
| d_omega_max | float | to-be-calibrated | Max rotor acceleration (rad/s²) |

**State per env**:
| Field | Type | Description |
|-------|------|-------------|
| current_rpm | float | Current rotor speed (RPM) |
| current_omega | float | Current rotor angular velocity (rad/s) |
| commanded_throttle | float | Target throttle from action [0, 1] |

**Outputs per step**:
| Field | Type | Description |
|-------|------|-------------|
| thrust_force | vec3 | Thrust vector along EDF axis (N) |
| static_reaction_torque | vec3 | Steady-state reaction torque (N·m) |
| dynamic_spool_torque | vec3 | Torque from rotor acceleration (N·m) |
| gyro_precession_torque | vec3 | ω_body × H_rotor (N·m) |

---

### FinAeroModel

Per-fin aerodynamic force computation (stateless).

**Inputs per fin**:
| Field | Type | Description |
|-------|------|-------------|
| actual_angle | float | Current fin deflection (rad) |
| hinge_axis | vec3 | Joint hinge axis (local frame) |
| chord_direction | vec3 | Chord direction (local frame) |
| fin_normal | vec3 | Normal at neutral (local frame) |
| exhaust_speed | float | Local exhaust velocity magnitude (m/s) |
| density | float | Air density (kg/m³) |
| fin_area | float | Fin planform area (m²) |

**Coefficient parameters** (from YAML):
| Field | Type | Source Label | Description |
|-------|------|-------------|-------------|
| C_N_alpha | float | estimate | Normal force slope (1/rad) |
| k_saturation | float | estimate | Finite-angle saturation factor |
| C_D0 | float | estimate | Zero-deflection drag coefficient |
| C_D_alpha_sq | float | estimate | Drag-vs-angle² coefficient |
| AR_correction | float | estimate | Aspect ratio correction |
| duct_confinement | float | estimate | Duct confinement correction |
| calibration_factor | float | to-be-calibrated | Empirical calibration multiplier |

**Outputs per fin**:
| Field | Type | Description |
|-------|------|-------------|
| force_vector | vec3 | Total force in fin-local frame (N) |
| normal_force | float | Normal (control) component magnitude (N) |
| tangential_force | float | Tangential (drag) component magnitude (N) |
| thrust_loss | float | Estimated thrust loss (N) |

---

### ContactStateMachine

Tracks vehicle-ground interaction state per environment.

| State | Value | Description |
|-------|-------|-------------|
| AIRBORNE | 0 | No ground contact |
| GROUND_CONTACT_CANDIDATE | 1 | Contact detected, dwell timer running |
| LANDED | 2 | All dwell criteria met for required interval |
| CRASHED | 3 | Crash criterion triggered |

**Dwell criteria** (all must hold simultaneously for `dwell_frames` consecutive steps):
| Criterion | Type | Description |
|-----------|------|-------------|
| contact_active | bool | Ground contact sensor reports contact |
| vertical_speed | float | Below `max_vz_landed` threshold |
| lateral_speed | float | Below `max_vxy_landed` threshold |
| tilt_angle | float | Below `max_tilt_landed` threshold |
| angular_rate | float | Below `max_omega_landed` threshold |

**Crash triggers** (any one sufficient):
| Trigger | Description |
|---------|-------------|
| impact_speed | Contact vertical speed exceeds `max_vz_crash` |
| excessive_tilt | Tilt on contact exceeds `max_tilt_crash` |
| excessive_rate | Angular rate on contact exceeds `max_omega_crash` |
| unsafe_contact | Non-landing-gear body part contacts ground |
| tip_over | Tilt exceeds threshold after initial contact |

**State per env**:
| Field | Type | Description |
|-------|------|-------------|
| state | int | Current state enum value |
| dwell_counter | int | Frames in GROUND_CONTACT_CANDIDATE meeting all criteria |

---

### TaskConfig

Configuration-driven operating mode selecting reward, success, termination, and spawn behavior.

| Field | Type | Description |
|-------|------|-------------|
| task_name | string | "hover" or "landing" |
| reward_terms | dict[str, float] | Term name → weight mapping |
| success_criteria | dict[str, float] | Criterion name → threshold mapping |
| termination_conditions | dict[str, float] | Condition name → threshold mapping |
| spawn_position_range | tuple[vec3, vec3] | Min/max spawn position (m) |
| spawn_velocity_range | tuple[vec3, vec3] | Min/max spawn velocity (m/s) |
| spawn_attitude_range | tuple[vec3, vec3] | Min/max spawn Euler angles (rad) |
| target_position | vec3 | Target position for the task (m) |
| episode_length_s | float | Maximum episode duration (s) |
| disturbance_config | string | Path to disturbance YAML |

---

### ObservationVector

Controller-agnostic observation exposed by the environment.

| Index Range | Dimension | Field | Frame |
|-------------|-----------|-------|-------|
| 0-2 | 3 | Position error to target (m) | world |
| 3-6 | 4 | Attitude quaternion (w,x,y,z) | world |
| 7-9 | 3 | Linear velocity (m/s) | body |
| 10-12 | 3 | Angular velocity (rad/s) | body |
| 13 | 1 | Height above ground (m) | world |
| 14-17 | 4 | Fin actual angles (rad) | — |
| 18-21 | 4 | Fin actual rates (rad/s) | — |
| 22 | 1 | Motor RPM (normalized) | — |
| 23 | 1 | Contact state (encoded) | — |
| 24-26 | 3 | Wind estimate (m/s) (optional) | world |

**Total dimension**: 24 (base) or 27 (with wind estimate)

---

### ActionVector

Controller-agnostic action accepted by the environment.

| Index | Dimension | Field | Range |
|-------|-----------|-------|-------|
| 0-3 | 4 | Fin target angles (rad) | [-max_angle, +max_angle] |
| 4 | 1 | Throttle/RPM target | [0, 1] |

**Total dimension**: 5

---

### DisturbanceConfig

Configurable wind and perturbation framework.

| Field | Type | Description |
|-------|------|-------------|
| enabled | bool | Master enable/disable |
| wind_vector | vec3 | Steady-state wind velocity (m/s), world frame |
| gust_enabled | bool | Enable gust events |
| gust_magnitude | float | Max gust speed (m/s) |
| gust_duration | float | Gust event duration (s) |
| gust_interval | tuple[float, float] | Min/max time between gusts (s) |
| body_drag_cd | float | Body drag coefficient |
| body_ref_area | float | Body reference area (m²) |
| com_offset_enabled | bool | Enable COM shift |
| com_offset_range | tuple[vec3, vec3] | Min/max COM offset (m) |
| sensor_noise_enabled | bool | Enable observation noise |
| sensor_noise_std | dict[str, float] | Noise std per observation field |

## State Transitions

### ContactStateMachine

```
AIRBORNE ──contact detected──> GROUND_CONTACT_CANDIDATE
                                    │
                     ┌──────────────┼──────────────┐
                     │              │              │
              dwell criteria   contact lost    crash trigger
              met for N frames  (bounce)       detected
                     │              │              │
                     ▼              ▼              ▼
                  LANDED        AIRBORNE        CRASHED
```

### Episode Lifecycle

```
RESET ──> RUNNING ──> TERMINATED (crashed | timed_out | success)
  │                       │
  │                       ▼
  └──────────── AUTO-RESET (vectorized)
```

## Relationships

```
VehicleAsset ──1:4──> ServoActuator (one per fin)
VehicleAsset ──1:1──> EDFPropulsion
VehicleAsset ──1:4──> FinAeroModel (one per fin, stateless)
VehicleAsset ──1:1──> ContactStateMachine

TaskConfig ──selects──> RewardTerms (from registry)
TaskConfig ──defines──> SuccessCriteria
TaskConfig ──defines──> TerminationConditions
TaskConfig ──references──> DisturbanceConfig

DirectRLEnv ──uses──> VehicleAsset
DirectRLEnv ──uses──> TaskConfig
DirectRLEnv ──produces──> ObservationVector
DirectRLEnv ──consumes──> ActionVector
```
