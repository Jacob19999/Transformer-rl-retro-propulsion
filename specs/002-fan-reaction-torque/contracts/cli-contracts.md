# CLI Contracts: EDF Fan Reaction Torque

**Branch**: `002-fan-reaction-torque` | **Date**: 2026-03-12

## Diagnostic Script: `diag_reaction_torque`

### Invocation

```bash
python -m simulation.isaac.scripts.diag_reaction_torque [OPTIONS]
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mode` | `{constant, ramp, liftoff}` | `constant` | Test mode |
| `--thrust` | float (0.0–1.0) | `0.68` | Normalized thrust command (constant mode) |
| `--ramp-duration` | float (s) | `1.0` | Time to ramp 0→100% (ramp mode) |
| `--duration` | float (s) | `3.0` | Total test duration |
| `--config` | path | `simulation/isaac/configs/isaac_env_gyro_test.yaml` | Isaac Sim env config |
| `--vehicle-config` | path | `simulation/configs/default_vehicle.yaml` | Vehicle YAML (k_torque source) |
| `--disable-anti-torque` | flag | false | Run with anti-torque disabled for A/B comparison |
| `--output` | path | stdout | Write structured log to file (CSV) |

### Modes

**`constant`**: Zero-gravity, hold thrust at `--thrust` level for `--duration`. Measures steady-state yaw rate.
- Default config uses zero gravity (`isaac_env_gyro_test.yaml`)
- Pass criterion: yaw rate matches `k_torque × ω² / I_zz` within 10%

**`ramp`**: Zero-gravity, ramp from 0 to 100% thrust over `--ramp-duration`, then hold. Measures yaw spike.
- Pass criterion: peak yaw rate during ramp > 110% of steady-state yaw rate at full thrust

**`liftoff`**: Normal gravity, full thrust from ground (~0.4m). End-to-end pipeline validation.
- Uses `isaac_env_single.yaml` (or override via `--config`)
- Pass criterion: altitude > 5m AND yaw > 5° after `--duration` seconds

### Output Format

Per-step CSV (header + data rows):
```
step,time_s,altitude_m,yaw_deg,yaw_rate_dps,thrust_N,tau_anti_Nm,tau_ramp_Nm
0,0.000,5.000,0.00,0.00,0.00,0.000,0.000
1,0.008,5.000,0.00,0.12,2.54,0.003,0.245
...
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | PASS — all criteria met |
| 1 | FAIL — criteria not met (details printed to stderr) |
| 2 | ERROR — runtime error (missing config, Isaac Sim not available) |

---

## Unit Test Contract: `test_reaction_torque.py`

### Test Cases

| Test | Input | Expected Output |
|------|-------|-----------------|
| `test_anti_torque_at_hover` | T=30.5N, k_torque=1e-8 | τ_z ≈ −0.670 N·m (±1%) |
| `test_anti_torque_at_zero` | T=0 | τ = [0, 0, 0] |
| `test_anti_torque_proportional` | T=13.5N, T=30.5N, T=45N | τ ratio matches T ratio (quadratic via ω²) |
| `test_anti_torque_disabled` | enabled=false, T=30.5N | τ = [0, 0, 0] |
| `test_ramp_torque_sign` | T_dot > 0 (spin-up) | τ_z < 0 (opposes spin-up) |
| `test_ramp_torque_at_constant_thrust` | T_dot = 0 | τ_ramp = [0, 0, 0] |
| `test_combined_torque` | T=30.5N, T_dot=100 | τ_anti + τ_ramp, both non-zero |
