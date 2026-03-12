# Quickstart: Isaac Sim Mass Properties, Thrust Test & Environmental Forces

**Feature**: 002-isaac-mass-thrust-env

## Prerequisites

- Feature 001 complete (stable Isaac Sim env with fins articulation)
- Python 3.10+ with `pxr` (OpenUSD) bindings
- Isaac Sim / IsaacLab installed (for thrust and wind diagnostics)
- GPU with ≥8 GB VRAM (RTX 5070 or equivalent)

## 1. Validate Mass Properties (no Isaac Sim required)

```bash
# Compare USDC scene mass properties against YAML config
python -m simulation.isaac.scripts.validate_mass_props \
  --usd simulation/isaac/usd/drone.usdc \
  --config simulation/configs/default_vehicle.yaml

# Expected output:
# Mass Property Validation Report
# ================================
# total_mass:  YAML=3.130 kg  USD=3.130 kg  err=0.00%  ✓
# com_x:       YAML=0.0045 m  USD=0.0045 m  err=0.00%  ✓
# ...
# RESULT: PASS (10/10 within 1% tolerance)
```

## 2. Thrust Application Test (requires Isaac Sim)

```bash
# Drone starts on ground, applies full thrust, should lift off
python -m simulation.isaac.scripts.diag_thrust_test \
  --config simulation/isaac/configs/isaac_env_single.yaml

# Expected: drone ascends from ~0.4 m to >5 m in 2 seconds
# Logs altitude every 30 steps
```

## 3. Wind Disturbance Test (requires Isaac Sim)

```bash
# Apply constant 5 m/s wind, observe lateral drift
python -m simulation.isaac.scripts.diag_wind \
  --config simulation/isaac/configs/isaac_env_single.yaml \
  --wind-x 5.0

# Expected: drone drifts laterally in +X with measurable velocity
```

## 4. Run All Isaac Sim Tests

```bash
pytest -m isaac simulation/tests/test_mass_validation.py -v
pytest -m isaac simulation/tests/test_isaac_env.py -v
```

## 5. Gyro Precession Diagnostic (requires Isaac Sim v5.1.0)

```bash
# Spawn in zero-g, apply yaw torque, observe pitch precession response
python -m simulation.isaac.scripts.diag_gyro_precession \
  --config simulation/isaac/configs/isaac_env_single.yaml \
  --yaw-torque 0.5 --duration 2.0

# Expected: pitch rate develops proportional to I_fan * omega_fan * yaw_rate

# Comparison run with precession disabled:
python -m simulation.isaac.scripts.diag_gyro_precession \
  --config simulation/isaac/configs/isaac_env_single.yaml \
  --disable-precession

# Expected: no pitch response from yaw torque
```

## 6. Validate Rotor Mass Properties (no Isaac Sim required)

```bash
# Extended validation now checks /Drone/Body/Rotor prim
python -m simulation.isaac.scripts.validate_mass_props \
  --usd simulation/isaac/usd/drone.usdc \
  --config simulation/configs/default_vehicle.yaml

# Expected output now includes:
# Rotor Prim Validation
# =====================
# Prim path:  /Drone/Body/Rotor  ✓ (exists)
# Radius:     YAML=0.040 m  USD=0.040 m  ✓
# ...
```

## Verification Checklist

- [ ] Mass validation script passes on current `drone.usdc`
- [ ] Drone lifts off under full thrust (T_cmd=1.0)
- [ ] Drone hovers at T_cmd ≈ 0.68 (weight/T_max)
- [ ] Wind produces lateral drift when enabled
- [ ] Observation [13:16] reflects wind values when wind active
- [ ] All existing Feature 001 diagnostics still pass
- [ ] Gyro precession enabled: yaw torque produces pitch response
- [ ] Gyro precession disabled: yaw torque produces only yaw response
- [ ] Rotor prim `/Drone/Body/Rotor` exists in USDC with correct geometry
- [ ] `validate_mass_props` passes with rotor prim validation
- [ ] No FPS regression from precession torque computation
