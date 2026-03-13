# Quickstart: EDF Fan Reaction Torque

**Branch**: `002-fan-reaction-torque` | **Date**: 2026-03-12

## What This Feature Adds

Two yaw torque effects from the EDF fan that were previously unmodelled:

1. **Steady-state anti-torque**: The fan spinning air creates a constant yaw reaction torque on the body (≈0.67 N·m at hover). Present whenever thrust > 0.
2. **RPM-ramp torque**: During thrust changes, the rotor's angular acceleration creates an additional transient yaw spike (≈2.5 N·m peak during fast ramp). Already in the custom sim; now added to Isaac Sim.

## Prerequisites

- Feature 001 Isaac Sim env operational
- Feature 002 mass/thrust validation passing
- Isaac Sim / IsaacLab Python environment active

## Quick Validation

### 1. Unit test (no Isaac Sim required)

```bash
pytest simulation/tests/test_reaction_torque.py -v
```

Confirms anti-torque math: `τ = -k_torque × ω²` for 5+ thrust levels.

### 2. Steady-state anti-torque diagnostic

```bash
python -m simulation.isaac.scripts.diag_reaction_torque --mode constant --thrust 0.68 --duration 3.0
```

Expected: drone yaws monotonically at ~139 °/s² (≈418 °/s after 3s). PASS if yaw rate matches predicted value within 10%.

### 3. RPM-ramp diagnostic

```bash
python -m simulation.isaac.scripts.diag_reaction_torque --mode ramp --ramp-duration 1.0 --duration 3.0
```

Expected: yaw rate spikes during ramp, then settles to steady-state. PASS if peak > 110% of steady-state.

### 4. End-to-end liftoff with yaw

```bash
python -m simulation.isaac.scripts.diag_reaction_torque --mode liftoff --duration 2.0
```

Expected: altitude > 5m AND yaw > 5°. Confirms anti-torque active in full pipeline.

### 5. A/B comparison (disabled baseline)

```bash
python -m simulation.isaac.scripts.diag_reaction_torque --mode constant --thrust 0.68 --disable-anti-torque
```

Expected: yaw rate < 0.01 rad/s (no reaction torque).

## Config

Anti-torque uses the existing `edf.k_torque` field in `default_vehicle.yaml` (value: `1.0e-8 N·m/(rad/s)²`). Toggle with:

```yaml
edf:
  anti_torque:
    enabled: true   # set false for A/B diagnostics
```

## Impact on Training

The anti-torque at hover (≈0.67 N·m) produces a yaw acceleration of ≈140 rad/s² — **the fins must actively compensate**. This means:
- RL policies trained without anti-torque will fail to maintain yaw on the real drone
- PID controllers need a yaw trim bias proportional to thrust
- Domain randomization should vary `k_torque` to build robustness
