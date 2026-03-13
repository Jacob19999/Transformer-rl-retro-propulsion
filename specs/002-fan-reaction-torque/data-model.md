# Data Model: EDF Fan Reaction Torque

**Branch**: `002-fan-reaction-torque` | **Date**: 2026-03-12

## Entities

### 1. ThrustModelConfig (extended)

Existing dataclass at `simulation/dynamics/thrust_model.py:35-62`. Extended with new fields.

| Field | Type | Units | Source | Status |
|-------|------|-------|--------|--------|
| `k_thrust` | float | N/(rad/s)² | `edf.k_thrust` | Existing |
| `tau_motor` | float | s | `edf.tau_motor` | Existing |
| `r_thrust` | ndarray(3) | m | `edf.r_thrust` | Existing |
| `r_duct` | float | m | `edf.r_duct` | Existing |
| `I_fan` | float | kg·m² | `edf.I_fan` | Existing |
| `T_max` | float | N | `edf.max_static_thrust` | Existing |
| `rho_ref` | float | kg/m³ | hardcoded 1.225 | Existing |
| `k_torque` | float | N·m/(rad/s)² | `edf.k_torque` | **New — activate** |
| `anti_torque_enabled` | bool | — | `edf.anti_torque.enabled` | **New** |

**Relationships:**
- Used by `ThrustModel` to compute forces/torques
- `k_torque` feeds the new `steady_state_anti_torque()` method
- `I_fan` is shared between gyro precession and RPM-ramp torque (no duplication)

### 2. IsaacAntiTorqueState (per-env tensor state in edf_landing_task.py)

Runtime state tracked per environment in the Isaac Sim task.

| Field | Type/Shape | Units | Purpose |
|-------|-----------|-------|---------|
| `_anti_torque_enabled` | bool | — | Config toggle loaded at init |
| `_k_torque` | float | N·m/(rad/s)² | Loaded from vehicle YAML at init |
| `_I_fan` | float | kg·m² | Already exists (gyro precession) |
| `tau_anti_log` | Tensor(N,) | N·m | Per-step anti-torque magnitude for logging |
| `tau_ramp_log` | Tensor(N,) | N·m | Per-step ramp-torque magnitude for logging |

**Note:** This is not a separate class but state fields on `EDFLandingTask`. Listed here for clarity.

### 3. DiagnosticLogEntry

Per-step output row from the reaction torque diagnostic script.

| Field | Type | Units | Description |
|-------|------|-------|-------------|
| `step` | int | — | Simulation step index |
| `time` | float | s | Elapsed simulation time |
| `altitude` | float | m | Drone altitude (world Z) |
| `yaw_deg` | float | ° | Yaw angle (Euler decomposition) |
| `yaw_rate_dps` | float | °/s | Yaw angular velocity |
| `thrust_N` | float | N | Current actual thrust |
| `tau_anti_Nm` | float | N·m | Steady-state anti-torque applied |
| `tau_ramp_Nm` | float | N·m | RPM-ramp torque applied |

**Relationships:**
- Emitted by `diag_reaction_torque.py` at each simulation step
- Used for post-processing validation and pass/fail determination

### 4. YAML Config Extensions

New keys added to `simulation/configs/default_vehicle.yaml`:

```yaml
edf:
  # existing...
  k_torque: 1.0e-8           # N·m/(rad/s)² — ALREADY PRESENT, now loaded
  anti_torque:
    enabled: true             # NEW — toggle for A/B comparison
```

**Validation rules:**
- `k_torque >= 0` (zero means no anti-torque; negative is physically invalid)
- `anti_torque.enabled` must be boolean
- When `anti_torque.enabled: false`, both steady-state and RPM-ramp yaw torques are suppressed

## State Transitions

```
ThrustModel state per step:
  T_cmd (input) → T_actual (1st-order lag) → ω_fan (algebraic)
                                              ├─ τ_anti = -k_torque × ω² (steady-state)
                                              └─ dω/dt → τ_ramp = -I_fan × dω/dt (transient)

Isaac Sim per-step in _apply_action():
  thrust_actual → ω_fan ─┬─ τ_anti (if anti_torque.enabled)
                          ├─ τ_ramp (if anti_torque.enabled)
                          └─ τ_gyro (if gyro_precession.enabled, existing)
  All → rotate body→world → accumulate → set_external_force_and_torque()
```
