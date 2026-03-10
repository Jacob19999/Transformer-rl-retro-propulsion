# Implementation Plan: Isaac Sim Vectorized Drone Simulation Environment

**Branch**: `001-isaac-sim-env` | **Date**: 2026-03-10 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-isaac-sim-env/spec.md`

---

## Summary

Build a vectorized simulation environment in NVIDIA Isaac Sim (IsaacLab 1.x / Isaac Sim 4.x) for training thrust-vectoring control policies on an EDF drone. PhysX handles rigid-body dynamics and ground contact; custom Python force injection per step supplies EDF thrust and fin aerodynamics. The drone USD asset is generated programmatically from existing `default_vehicle.yaml`. Supports 1–1024 parallel environments via a config flag, with single-env mode as the default for PID validation. Exposes a Gymnasium-compatible API compatible with the existing SB3 training pipeline.

---

## Technical Context

**Language/Version**: Python 3.10+ (Isaac Sim Python environment)
**Primary Dependencies**: IsaacLab 1.x, Isaac Sim 4.x (PhysX), USD Python bindings (`pxr`), Gymnasium, NumPy, PyTorch (for tensor operations in IsaacLab), Stable-Baselines3
**Storage**: YAML config files (`simulation/configs/`); generated USD asset (`simulation/isaac/usd/drone.usd`); training checkpoints under `runs/`
**Testing**: pytest (existing framework); test configs `test_vehicle.yaml` compatible; new `test_isaac_env.py` with lightweight single-env tests
**Target Platform**: Linux workstation with NVIDIA GPU (≥8 GB VRAM); Isaac Sim is Linux-native
**Project Type**: Simulation environment library + training integration
**Performance Goals**: ≥10× training throughput at 128 envs vs. single env; environment reset in <100 ms per env; 1024 envs on a single GPU
**Constraints**: Gymnasium API compliance; 20-dim obs / 5-dim action match; FRD body frame; scalar-last quaternion; YAML config unchanged

---

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### I. Physics Fidelity ✅ (with justified deviation)

PhysX TGS integrator with 4 substeps at 1/120s gives effective dt ≈ 2.08 ms, below the 5 ms threshold. EDF thrust model (k_thrust × ω², first-order lag) and fin aerodynamics (NACA0012 exhaust-stream forces) are applied as custom external forces each step — parameterized from YAML datasheet values. Ground effect and gyroscopic coupling must be included. Aero drag is also applied as custom forces using per-axis projected areas from YAML primitives.

**Justified deviation**: The constitution specifies "RK4 at dt=0.005s" for the custom Python sim. Isaac Sim uses PhysX TGS (implicit integrator), which provides better stability at larger dt than explicit methods. The 4-substep configuration achieves finer effective dt than the RK4 baseline.

### II. Configuration-Driven Design ✅

All physics parameters (mass, EDF, fins, servo) read from `simulation/configs/default_vehicle.yaml`. No magic numbers in source. New Isaac-specific parameters (num_envs, substeps) live in `simulation/isaac/configs/isaac_env_*.yaml`. Domain randomization from `domain_randomization.yaml`. Config format is unchanged.

### III. Test-Driven Validation ✅

`simulation/tests/test_isaac_env.py` MUST be created with tests for: reset stability (1000 cycles), fin deflection limits, observation dimensions and range, action scaling, and single-env gravity fall behavior. Uses `test_vehicle.yaml` for asset construction in tests.

### IV. Reproducibility ✅

`--seed` argument controls all random state. PhysX determinism enabled via `physics.setDeterminism(True)`. Per-env seeds derived as `seed + env_id`. Checkpoints saved to `runs/` with seed and commit hash in directory name.

### V. Sim-to-Real Integrity ✅

FRD body frame maintained. Scalar-last quaternion [qx, qy, qz, qw]. All coordinate transforms from USD (Y-up) to FRD are explicit and tested. Observation layout matches `observation.py` exactly.

---

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|--------------------------------------|
| Different integrator than RK4 | Isaac Sim uses PhysX TGS; cannot inject custom RK4 | Isaac Sim does not expose per-step integration hooks for custom integrators; substep approach achieves equivalent accuracy |
| Custom force injection per step | EDF + fin aero cannot be expressed in PhysX native joints/actuators | PhysX has no EDF thrust primitive; NACA0012 exhaust aerodynamics require per-step computation |

---

## Project Structure

### Documentation (this feature)

```text
specs/001-isaac-sim-env/
├── plan.md              # This file
├── research.md          # Phase 0: decisions and rationale
├── data-model.md        # Phase 1: entities and state transitions
├── quickstart.md        # Phase 1: developer onboarding
├── contracts/
│   ├── gymnasium-env-interface.md   # Gymnasium API contract
│   └── usd-asset-schema.md         # USD hierarchy and attribute contract
└── tasks.md             # Phase 2 output (/speckit.tasks — not yet created)
```

### Source Code (repository root)

```text
simulation/
├── isaac/
│   ├── __init__.py
│   ├── usd/
│   │   ├── __init__.py
│   │   └── drone_builder.py      # Generates drone.usd from default_vehicle.yaml
│   ├── tasks/
│   │   ├── __init__.py
│   │   └── edf_landing_task.py   # IsaacLab DirectRLEnv task definition
│   ├── envs/
│   │   ├── __init__.py
│   │   └── edf_isaac_env.py      # Gymnasium-compatible wrapper
│   ├── scripts/
│   │   ├── test_fins.py          # Visual fin articulation test script
│   │   └── diag_isaac_single.py  # Single-env diagnostic (gravity fall)
│   └── configs/
│       ├── isaac_env_single.yaml  # Single env, PID tuning / visual
│       ├── isaac_env_128.yaml     # 128 parallel envs
│       └── isaac_env_1024.yaml    # 1024 parallel envs
├── training/
│   └── scripts/
│       └── train_isaac_ppo.py    # Training entry point for Isaac Sim env
└── tests/
    └── test_isaac_env.py         # Pytest tests for Isaac Sim env
```

**Structure Decision**: Single project extension of existing `simulation/` tree. New code lives under `simulation/isaac/` to keep Isaac-specific code isolated from the custom Python sim. No new top-level directories needed. Config files follow existing pattern in `simulation/configs/` (Isaac-specific configs in `simulation/isaac/configs/`).

---

## Implementation Phases (for `/speckit.tasks`)

### Phase A — USD Asset Generation (P1, unblocked)

1. Implement `drone_builder.py`:
   - Load `default_vehicle.yaml` via existing `config_loader.py`
   - Compute composite mass, CoM, inertia from primitives (reuse `MassProperties` logic or call it directly)
   - Emit USD prims for body (compound collision shapes from YAML primitives)
   - Emit 4 `RevoluteJoint` prims for fins with ±15° limits and PD drive parameters
   - Write `simulation/isaac/usd/drone.usd`
2. Create `test_vehicle_yaml` → USD round-trip test (verify dimensions within 1% tolerance)

### Phase B — Single-Environment Physics (P1, depends on Phase A)

1. Implement `edf_landing_task.py` (`DirectRLEnv`):
   - Scene setup: load `drone.usd` × `num_envs`, add ground plane + landing pad with μ=0.5
   - Spawn: randomize initial position/velocity via `set_root_state_tensor()`
   - Pre-physics step: compute EDF force (k_thrust × ω², 1st-order lag state), fin aerodynamic forces (NACA0012 in exhaust stream), apply via `apply_forces_and_torques_at_pos()`
   - Post-physics step: read state tensors, compute 20-dim observation
   - Termination: crash (h_agl < 0 with |v| > threshold) or step limit
2. Implement `edf_isaac_env.py` Gymnasium wrapper
3. Diagnostic script: gravity fall test (zero actions, visual)

### Phase C — Test Suite (P1, depends on Phase B)

1. `test_single_env_reset_stability` — 1000 reset cycles headless
2. `test_fin_deflection_limits` — command ±15° and ±20°, verify clamp
3. `test_observation_dimensions` — shape=(20,), dtype=float32, all finite
4. `test_action_scaling` — thrust_cmd=1.0 → T_max N; delta_cmd=1.0 → δ_max rad
5. `test_gravity_fall` — zero actions, drone altitude decreases monotonically until contact

### Phase D — Parallel Environments (P2, depends on Phase B)

1. Verify `num_envs=128` launches and steps without errors
2. Verify environment independence (reset env 42, check others unaffected)
3. Throughput benchmark: log steps/s for 1, 128, 512, 1024 envs
4. Add `isaac_env_128.yaml` and `isaac_env_1024.yaml` configs

### Phase E — Training Integration (P2, depends on Phase D)

1. Implement `train_isaac_ppo.py` using existing SB3 PPO + `VecNormalize` pattern
2. Verify `VecNormalize` compatible with batched numpy output from `EDFIsaacEnv`
3. TensorBoard logging with seed + commit hash in run name
4. Save `VecNormalize` stats alongside PPO checkpoints
