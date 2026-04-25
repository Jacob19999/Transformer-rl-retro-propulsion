# Quickstart: Phase 1 Isaac Sim EDF TVC Environment

**Branch**: `007-isaac-sim-env` | **Date**: 2026-03-22

## Prerequisites

- NVIDIA GPU with CUDA support (RTX 5070 recommended)
- Isaac Sim 5.1 installed
- Isaac Lab 2.3.2 installed (available as submodule at `IsaacLab/`)
- Python 3.10+ with PyTorch >= 2.0

## Setup

```bash
# From repo root
cd simulation/isaac

# Install tvc_env as editable package (into Isaac Lab's Python env)
pip install -e .

# Verify imports
python -c "from tvc_env.common.frames import frd_to_isaac; print('OK')"
```

## Running the Validation Ladder

The validation ladder runs tests in dependency order. Each test must pass before proceeding to the next.

### Unit tests (no Isaac Sim required)

```bash
pytest tests/unit/ -v
```

### Simulation tests (require Isaac Sim runtime)

```bash
# Step 1: Asset validation
python apps/run_single_test.py --test test_00_asset_validation

# Step 2: Joint axes
python apps/run_single_test.py --test test_01_joint_axes

# Step 3: Single fin articulation
python apps/run_single_test.py --test test_02_single_fin_articulation

# Step 4: Unit force on fin
python apps/run_single_test.py --test test_03_unit_force_on_fin

# Step 5: Fin force sweep
python apps/run_single_test.py --test test_04_fin_force_sweep

# Step 6: Four fin superposition
python apps/run_single_test.py --test test_05_four_fin_superposition

# Step 7: EDF spool and reaction
python apps/run_single_test.py --test test_06_edf_spool_and_reaction

# Step 8: Gyro precession
python apps/run_single_test.py --test test_07_gyro_precession

# Step 9: Wind disturbance
python apps/run_single_test.py --test test_08_wind_disturbance

# Step 10: Contact/landed/crash
python apps/run_single_test.py --test test_09_contact_landed_crash

# Step 11: PID hover smoke
python apps/run_single_test.py --test test_10_pid_hover_smoke

# Step 12: 128-env RL API smoke
python apps/run_single_test.py --test test_11_rl_api_128env_smoke

# Step 13: Steady hover all forces
python apps/run_single_test.py --test test_12_steady_hover_all_forces
```

## Common Workflows

### Single-environment debug (with gizmos)

```bash
python apps/run_single_env_debug.py \
  --task hover \
  --env-config configs/env/single_env_debug.yaml \
  --disturbance configs/disturbances/nominal.yaml
```

This opens an Isaac Sim viewport with the drone, all debug gizmos (force arrows, body axes, HUD telemetry), and a keyboard/gamepad interface for manual control.

### PID hover evaluation

```bash
python apps/run_eval_pid.py \
  --task hover \
  --env-config configs/env/single_env_debug.yaml \
  --disturbance configs/disturbances/wind.yaml \
  --duration 30
```

### 128-environment smoke test

```bash
python apps/run_smoke_128.py \
  --task hover \
  --env-config configs/env/train_128.yaml \
  --steps 1000
```

### PPO training

```bash
python apps/run_train_ppo.py \
  --task landing \
  --env-config configs/env/train_128.yaml \
  --disturbance configs/disturbances/wind.yaml \
  --seed 42 \
  --total-steps 10000000
```

## Configuration Override Examples

```bash
# Change number of environments
python apps/run_smoke_128.py --override env.num_envs=64

# Enable wind disturbances
python apps/run_eval_pid.py --disturbance configs/disturbances/wind.yaml

# Switch task from hover to landing
python apps/run_single_env_debug.py --task landing

# Use high-fidelity solver
python apps/run_single_test.py --physics configs/physics/solver_high_fidelity.yaml
```

## Key Files

| Purpose | Path |
|---------|------|
| Frame conversions | `tvc_env/common/frames.py` |
| Asset validation | `tvc_env/asset/asset_validator.py` |
| Fin aero model | `tvc_env/dynamics/fin_aero.py` |
| Servo model | `tvc_env/dynamics/actuator_servo.py` |
| EDF model | `tvc_env/dynamics/propulsion_edf.py` |
| Contact state machine | `tvc_env/sim/contacts.py` |
| Force dispatch | `tvc_env/sim/wrench_dispatch.py` |
| DirectRLEnv | `tvc_env/envs/direct_rl_env.py` |
| Reward registry | `tvc_env/envs/reward_registry.py` |
| Debug gizmos | `tvc_env/sim/gizmos.py` |

## Troubleshooting

- **"Fin link not found"**: Check that `edf_drone_v2.asset.yaml` link names match the USD asset
- **NaN in observations**: Check servo/EDF config for null `to-be-calibrated` values
- **Gizmos not visible**: Verify `gizmos_enabled: true` in env config and `num_envs: 1`
- **Tensor shape mismatch at 128 envs**: Check `replicate_physics: true` in scene config
- **Mass mismatch warning**: Run `validate_usd_mass_props.py` and update YAML or USD
