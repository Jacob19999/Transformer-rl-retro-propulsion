# Research: EDF Fan Reaction Torque

**Branch**: `002-fan-reaction-torque` | **Date**: 2026-03-12

## RQ-1: What torque effects are already modelled vs. missing?

### Decision: Three distinct yaw-axis effects; one is missing

| Effect | Custom Sim | Isaac Sim | Status |
|--------|-----------|-----------|--------|
| **Gyroscopic precession** `τ = −ω × h_fan` | ✅ `vehicle.py:181-185` | ✅ `edf_landing_task.py:542-552` | Complete |
| **RPM-ramp reaction torque** `τ = −I_fan × dω/dt` | ✅ `thrust_model.py:140-148` | ❌ Not implemented | Port to Isaac Sim |
| **Steady-state anti-torque** `τ = −k_torque × ω²` | ❌ `k_torque` in YAML but unused | ❌ Not implemented | **New implementation** |

### Rationale

- Gyro precession couples pitch↔roll via the spinning fan; yaw is decoupled (parallel to spin axis). This is already validated via `diag_gyro_precession` (RQ-7/RQ-11 in main spec).
- RPM-ramp torque (`motor_reaction_torque()`) exists in the custom sim's `ThrustModel` but was never ported to the Isaac Sim `_apply_action()` pipeline.
- Steady-state anti-torque is the aerodynamic reaction to the fan spinning air. `k_torque: 1.0e-8` exists in `default_vehicle.yaml:209` but is never loaded into `ThrustModelConfig` or used in any computation.

### Alternatives Considered

- **Single combined model**: Model both steady-state and transient as one equation. Rejected because the physics are distinct (aerodynamic reaction vs. inertial acceleration), and separating them allows independent validation and config toggling.
- **Torque as function of thrust (τ = k_Q × T)**: Algebraically equivalent to `τ = k_torque × ω²` since `T = k_thrust × ω²`, giving `k_Q = k_torque / k_thrust`. Using `k_torque × ω²` is preferred because it matches propeller aerodynamic theory directly and the YAML already stores `k_torque` in those units.

---

## RQ-2: What are the correct magnitudes?

### Decision: Use existing `k_torque: 1.0e-8 N·m/(rad/s)²` with validation note

**Steady-state anti-torque at representative conditions:**

| Condition | Thrust (N) | ω_fan (rad/s) | τ_anti (N·m) | α_yaw (rad/s²) |
|-----------|-----------|---------------|--------------|-----------------|
| Hover     | 30.5      | 8187          | 0.670        | 139.6           |
| Max       | 45.0      | 9948          | 0.990        | 206.2           |
| 30% cmd   | 13.5      | 5448          | 0.297        | 61.8            |

**Cross-check with motor power:**
- FMS 90mm EDF: ~5500W electrical input at max, η_motor ≈ 0.85 → P_mech ≈ 4675W
- Q = P/ω = 4675/9948 = 0.470 N·m → implies `k_torque ≈ 4.75e-9`
- The config value `1.0e-8` is ~2× the power-derived estimate, suggesting it may need calibration against bench data. **For now, use the config value; flag for hardware validation.**

**RPM-ramp torque during 0→hover in τ_motor=0.1s:**
- dω/dt ≈ 8187 / 0.1 = 81870 rad/s² (peak, via 1st-order lag)
- τ_ramp = I_fan × dω/dt = 3.0e-5 × 81870 = 2.456 N·m
- This is ~3.7× the steady-state torque at hover — confirming the spec's prediction that ramp peak exceeds steady-state by >10%.

### Rationale

Using the existing YAML value preserves configuration-driven discipline (Constitution Principle II). The power-based discrepancy is noted as a calibration TODO but does not block implementation.

### Alternatives Considered

- Derive `k_torque` from motor power curve. Rejected for initial implementation because power data is not yet measured on the bench; the existing YAML placeholder provides a physically reasonable starting point.

---

## RQ-3: How to parameterize in YAML?

### Decision: Flat keys under `edf` section, reuse `I_fan`

**Config additions to `default_vehicle.yaml`:**

```yaml
edf:
  # existing keys...
  k_torque: 1.0e-8              # N·m/(rad/s)² — steady-state anti-torque coeff (ACTIVATE, already present)
  I_fan: 3.0e-5                 # kg·m² — already used for gyro, now also for ramp torque
  anti_torque:
    enabled: true                # Toggle for A/B diagnostic comparison
```

**No new `I_rotor` parameter needed** — `I_fan` is identical to the spec's `I_rotor` and is already in config for gyro precession. Introducing a duplicate parameter would violate DRY.

**The spec's `propulsion.torque_coefficient_k_Q` maps to the existing `edf.k_torque`** — but the physical model uses `τ = k_torque × ω²` rather than `τ = k_Q × T`, which is more consistent with propeller aerodynamic conventions.

### Rationale

Flat structure under `edf` matches the existing config pattern (no subsections for `gyro_precession` toggle either — it's `edf.gyro_precession.enabled`). Reusing `I_fan` avoids redundancy.

### Alternatives Considered

- New `propulsion` subsection as spec suggests. Rejected because it breaks the existing flat-under-`edf` convention and would require refactoring `ThrustModel.from_edf_config()`.

---

## RQ-4: Isaac Sim integration pattern

### Decision: Insert into `_apply_action()` between gyro precession and wind blocks

**Insertion point:** `edf_landing_task.py`, after line 552 (gyro precession), before line 554 (wind).

**Two new torque components:**

1. **Steady-state anti-torque:**
   ```
   ω_fan = sqrt(T_actual / k_thrust)
   τ_anti_b = [0, 0, -k_torque × ω_fan²]   (body FRD)
   τ_anti_w = rotate_body_to_world(q, τ_anti_b)
   torques += τ_anti_w
   ```

2. **RPM-ramp reaction torque:**
   ```
   T_dot = (T_cmd_clipped - T_actual) / tau_motor
   ω_safe = max(ω_fan, 1e-6)
   dω/dt = T_dot / (2 × k_thrust × ω_safe)
   τ_ramp_b = [0, 0, -I_fan × dω/dt]   (body FRD)
   τ_ramp_w = rotate_body_to_world(q, τ_ramp_b)
   torques += τ_ramp_w
   ```

**Pattern matches existing gyro precession block:** compute in body frame → rotate to world → accumulate into torques tensor → single `set_external_force_and_torque()` call.

### Rationale

Following the established accumulation pattern ensures consistency and avoids API call ordering issues. Body-to-world rotation is necessary because `is_global=True` is used for the combined force/torque application.

### Alternatives Considered

- Separate `set_external_force_and_torque()` call for reaction torques. Rejected because IsaacLab accumulates external forces per-body and the existing pattern already combines all sources into one call.

---

## RQ-5: Custom sim consistency

### Decision: Activate `k_torque` in `ThrustModel.outputs()`, keep RPM-ramp as-is

**Custom sim changes needed:**

1. **Load `k_torque` into `ThrustModelConfig`** — currently not extracted from config dict.
2. **Add steady-state anti-torque computation** in `ThrustModel.outputs()`:
   ```python
   omega = self.omega_from_thrust(T)
   tau_anti = np.array([0.0, 0.0, -self.config.k_torque * omega**2])
   ```
3. **Add to total torque return:** `tau_offset + tau_reaction + tau_anti`
4. **RPM-ramp torque already works** via `motor_reaction_torque()` — no changes needed.
5. **Add `anti_torque.enabled` toggle** to `ThrustModelConfig` for diagnostic comparison.

### Rationale

Ensures identical physics between custom sim and Isaac Sim (Constitution Principle V: Sim-to-Real Integrity). The custom sim is the reference implementation that Isaac Sim must match.

---

## RQ-6: Sign conventions

### Decision: Anti-torque acts along body −Z (yaw) in FRD frame

**Convention check against existing gyro work (RQ-7/RQ-11):**

- Body frame: FRD (Forward-Right-Down), thrust along +Z
- Fan spins about body +Z axis (positive ω_fan by convention)
- Precession: `τ_gyro = −ω × [0, 0, I_fan × ω_fan]` — correct, validated
- Anti-torque: Fan accelerates air in one rotational direction → body reacts in opposite direction
- Motor torque acts in +Z to spin the fan → body experiences −Z reaction

**Therefore:** `τ_anti_z = −k_torque × ω²` (negative Z in body frame). The negative sign is consistent with the existing `motor_reaction_torque()` which also uses `−I_fan × dω/dt`.

### Rationale

Consistent sign convention with the already-validated gyro precession code prevents sign errors that would produce physically impossible behavior.

---

## RQ-7: Diagnostic design

### Decision: Single script `diag_reaction_torque.py` with `--mode` flag

**Modes:**
- `--mode constant`: Zero-gravity, constant thrust, measure yaw rate buildup. Validates steady-state anti-torque.
- `--mode ramp`: Zero-gravity, thrust ramp 0→100%, measure yaw spike. Validates transient + steady-state.
- `--mode liftoff`: Normal gravity, full thrust from ground, observe combined liftoff + yaw. End-to-end validation.

**Config:** Reuse `isaac_env_gyro_test.yaml` structure (zero-gravity, single env) for `constant` and `ramp` modes. Use `isaac_env_single.yaml` for `liftoff` mode.

**Output:** Structured per-step log: `[step, time, altitude, yaw_deg, yaw_rate_dps, thrust_N, tau_anti_Nm, tau_ramp_Nm]`

### Rationale

One script with modes avoids script proliferation while keeping each test case independently runnable.

### Alternatives Considered

- Separate scripts per mode (like existing `diag_thrust_test.py` and `diag_gyro_precession.py`). Rejected because reaction torque validation is a single physics concept with different test conditions, not fundamentally different features.
