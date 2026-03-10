# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project validating a **Gated Transformer-XL PPO (GTrXL-PPO)** agent for thrust-vectoring control (TVC) in retro-propulsive landings on a physical **Electric Ducted Fan (EDF) drone** testbed. The goal is sim-to-real transfer from a custom Python simulation to hardware.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest

# Run a single test file
pytest simulation/tests/test_vehicle_dynamics.py

# Run a single test by name
pytest simulation/tests/test_reward.py::TestRewardFunction::test_landing_success

# Train PPO-MLP baseline
python -m simulation.training.scripts.train_ppo_mlp --seed 0

# Tune PID controller
python -m simulation.training.scripts.tune_pid

# Diagnostic scripts
python -m simulation.training.scripts.diag_single_ep
python -m simulation.training.scripts.diag_inertia
python -m simulation.training.scripts.diag_yaw
```

TensorBoard logs and checkpoints are saved to `runs/`.

## Architecture

### Simulation Stack (`simulation/`)

The simulation is a custom 6-DOF rigid-body plant — **not** Isaac Sim at this stage. Isaac Sim is a future milestone.

**Data flow:**
```
configs/*.yaml
    └─> VehicleDynamics (dynamics/vehicle.py)
            ├─ MassProperties (mass_properties.py) — composite CoM + inertia from YAML primitives
            ├─ ThrustModel (thrust_model.py) — EDF with 1st-order lag + ground effect
            ├─ AeroModel (aero_model.py) — directional drag aggregated from primitives
            ├─ FinModel (fin_model.py) — NACA0012 fins in exhaust stream
            ├─ ServoModel (servo_model.py) — 1st-order actuator lag
            └─ RK4Integrator (integrator.py) — dt=0.005s, re-normalizes quaternion every 10 steps
    └─> EnvironmentModel (environment/environment_model.py)
            ├─ AtmosphereModel — ISA density vs altitude
            └─ WindModel — configurable gust profiles
    └─> EDFLandingEnv (training/edf_landing_env.py)  ← Gymnasium wrapper
            ├─ ObservationPipeline (training/observation.py) — OBS_DIM=20 with noise injection
            ├─ RewardFunction (training/reward.py) — shaped + terminal rewards
            └─ Controllers (training/controllers/)
                    ├─ PIDController (pid_controller.py)
                    └─ PPOMlpController (ppo_mlp.py) — wraps SB3 PPO(MlpPolicy)
```

### State & Action Spaces

- **State:** 18-dim `[p(3), v_body(3), q(4), ω(3), T(1), δ_actual(4)]`
- **Observation:** 20-dim normalized vector (see `ObservationPipeline`)
- **Action:** 5-dim `[T_cmd, δ_1, δ_2, δ_3, δ_4]` normalized to `[-1, 1]`
- **Body frame:** FRD (Forward-Right-Down); thrust along +z

### Config System

All physics parameters live in `simulation/configs/*.yaml`. `config_loader.py` merges configs with deep-update semantics. Domain randomization is specified in `domain_randomization.yaml` and applied per episode at env reset.

Key configs:
- `default_vehicle.yaml` — EDF geometry, mass primitives, aero, fins, servo specs
- `default_environment.yaml` — atmosphere, wind
- `reward.yaml` — reward weights
- `ppo_mlp.yaml` / `ppo_mlp_smoketest.yaml` — SB3 hyperparameters

### Training

PPO-MLP (SB3) is the current baseline (Stage 19). GTrXL-PPO is the target architecture (future stages). Training uses `SubprocVecEnv` + `VecNormalize`. Checkpoints saved every 500K steps; best model tracked by landing success rate evaluated every 100K steps.

## Key Conventions

- All angles in **radians** internally; configs may use degrees where noted (servo deflection limits, fin angular offsets)
- Quaternion convention: scalar-last `[qx, qy, qz, qw]`
- `pytest.ini` sets `cache_dir = simulation/test_cache`
- Tests use `test_vehicle.yaml` / `test_environment.yaml` (lightweight configs) — do not modify `default_*.yaml` for test overrides
