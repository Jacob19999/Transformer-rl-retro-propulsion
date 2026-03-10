# Quickstart: Isaac Sim Drone Simulation Environment

**Branch**: `001-isaac-sim-env` | **Date**: 2026-03-10

---

## Prerequisites

- NVIDIA Isaac Sim 4.x installed (IsaacLab 1.x)
- NVIDIA GPU with ≥8 GB VRAM (16 GB recommended for 512+ parallel envs)
- Python 3.10+ with Isaac Sim Python environment activated
- Existing project dependencies: `pip install -r requirements.txt`

---

## Step 1: Generate the USD Drone Asset

Run once after any change to `default_vehicle.yaml`:

```bash
python -m simulation.isaac.usd.drone_builder \
    --config simulation/configs/default_vehicle.yaml \
    --output simulation/isaac/usd/drone.usd
```

Verify output: the file `simulation/isaac/usd/drone.usd` should be created and should show a drone with 4 articulated fins when opened in Omniverse Composer.

---

## Step 2: Launch Single-Environment (Visual Inspection)

For PID tuning and visual validation:

```bash
python -m simulation.isaac.envs.edf_isaac_env \
    --config simulation/isaac/configs/isaac_env_single.yaml \
    --mode visual
```

Expected behavior:
- Isaac Sim UI opens showing the arena (10×10m) with the landing pad
- Drone spawns at randomized altitude (5-10m) and falls under gravity
- Drone settles on the landing pad without penetrating the surface

---

## Step 3: Verify Fin Articulation

```bash
python -m simulation.isaac.scripts.test_fins \
    --config simulation/isaac/configs/isaac_env_single.yaml
```

Expected output: Each fin rotates to ±15° on command and returns to neutral. Pass/fail printed to console.

---

## Step 4: Launch Vectorized Environments

For RL training (128 parallel envs):

```bash
python -m simulation.training.scripts.train_isaac_ppo \
    --config simulation/isaac/configs/isaac_env_128.yaml \
    --seed 0
```

TensorBoard logs saved to `runs/isaac_ppo_<timestamp>/`.

---

## Step 5: Run Tests

```bash
pytest simulation/tests/test_isaac_env.py -v
```

Key tests:
- `test_single_env_reset_stability` — 1000 reset cycles without crash
- `test_fin_deflection_limits` — fins clamp at ±15°
- `test_observation_dimensions` — obs shape is (20,)
- `test_action_scaling` — normalized actions map to correct physical values

---

## Config Files

| Config | Purpose |
|--------|---------|
| `simulation/isaac/configs/isaac_env_single.yaml` | Single env, visual mode, PID tuning |
| `simulation/isaac/configs/isaac_env_128.yaml` | 128 parallel envs, RL training |
| `simulation/isaac/configs/isaac_env_1024.yaml` | 1024 parallel envs, large-scale training |

---

## Key File Locations

| File | Description |
|------|-------------|
| `simulation/isaac/usd/drone_builder.py` | Generates USD asset from YAML config |
| `simulation/isaac/usd/drone.usd` | Generated drone asset (do not edit manually) |
| `simulation/isaac/envs/edf_isaac_env.py` | Gymnasium-compatible env wrapper |
| `simulation/isaac/tasks/edf_landing_task.py` | IsaacLab DirectRLEnv task definition |
| `simulation/isaac/configs/isaac_env_single.yaml` | Single-env configuration |
| `simulation/tests/test_isaac_env.py` | Pytest tests for Isaac Sim env |

---

## Troubleshooting

**Issue**: USD asset fails to load in Isaac Sim
**Fix**: Ensure `drone.usd` was generated with the Isaac Sim Python environment active (not the system Python)

**Issue**: Fin deflection exceeds limits
**Fix**: Check `fins.max_deflection` in YAML; the USD joint limits are set in degrees and should match 15.0

**Issue**: GPU out of memory with 1024 envs
**Fix**: Reduce `num_envs` to 512 or 256; or reduce physics substeps from 4 to 2

**Issue**: Observation shape mismatch
**Fix**: Ensure `OBS_DIM = 20` matches between `observation.py` and Isaac Sim env; run `pytest simulation/tests/test_isaac_env.py::test_observation_dimensions`
