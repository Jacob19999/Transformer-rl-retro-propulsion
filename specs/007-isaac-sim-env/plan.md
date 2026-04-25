# Implementation Plan: Phase 1 Isaac Sim EDF TVC Simulation Environment

**Branch**: `007-isaac-sim-env` | **Date**: 2026-03-22 | **Spec**: [spec.md](../../.specify/features/007-isaac-sim-env/spec.md)
**Input**: Feature specification from `.specify/features/007-isaac-sim-env/spec.md`
**Technical Reference**: [Technical Plan.md](../../.specify/features/007-isaac-sim-env/Technical%20Plan.md)

## Summary

Build a from-scratch Phase 1 simulation environment for the EDF thrust-vectoring drone in Isaac Sim 5.1 + Isaac Lab 2.3.2 using PhysX. The environment uses a DirectRLEnv architecture with per-fin force application at centers of pressure, a body-FRD canonical frame with a single conversion boundary, realistic servo/EDF actuator dynamics, a 4-state contact state machine, composable task-configurable rewards, single-env debug gizmos, and 128-environment vectorized training support. All physics modules are shared across modes; only wrappers and task configs change. Validation follows a 13-step incremental test ladder proving correctness before RL training.

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**: Isaac Sim 5.1, Isaac Lab 2.3.2, PhysX, PyTorch >= 2.0, NumPy >= 1.24
**Storage**: YAML config files under `simulation/isaac/configs/`; USD assets under `simulation/isaac/assets/`
**Testing**: pytest (unit tests offline), Isaac Sim simulation tests (require GPU/Isaac runtime)
**Target Platform**: Linux/Windows with NVIDIA GPU (RTX 5070), Isaac Sim 5.1 runtime
**Project Type**: Simulation environment (Isaac Lab DirectRLEnv plugin)
**Performance Goals**: 128 parallel environments at physics rate; single-env debug at interactive framerate with gizmos
**Constraints**: Isaac Lab 2.3.2 API (no 3.0 migration); PhysX only (no Newton); (w,x,y,z) quaternion convention; body-FRD internal frame
**Scale/Scope**: ~45 Python source files, ~25 YAML configs, 13 simulation tests, 6 unit test files

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*


| Principle                       | Status | Evidence                                                                                                                                                                                                                                             |
| ------------------------------- | ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| I. Physics Fidelity             | PASS   | All sub-models (fin aero, servo, EDF, rotor reaction) parameterized from MG996R datasheet and EDF engineering estimates. All params labeled as measured/datasheet/estimate/to-be-calibrated. YAML config authoritative; USDC validated against YAML. |
| II. Configuration-Driven Design | PASS   | All physics parameters, reward weights, disturbance settings, and environment configs in YAML under `simulation/isaac/configs/`. No magic numbers in source. USDC physics attributes validated against YAML via dedicated script.                    |
| III. Test-Driven Validation     | PASS   | 13-step simulation validation ladder (test_00 through test_12) + 6 unit test files. Thrust liftoff test (test_06), fin articulation test (test_01, test_02), wind disturbance test (test_08) all included per constitution requirements.             |
| IV. Reproducibility             | PASS   | Vectorized env supports seed control. Episode telemetry logged for comparison. Training runs use timestamped directories under `runs/`.                                                                                                              |
| V. Sim-to-Real Integrity        | PASS   | Body-FRD frame with single conversion boundary in `common/frames.py`. Quaternion convention (w,x,y,z) consistent with Isaac Lab 2.3.2. Wind/atmosphere disturbances configurable via YAML. Domain randomization hooks provided.                      |


**Mass property validation**: YAML is authoritative source of truth. A `validate_usd_mass_props.py` script will compare USDC scene properties against YAML config with 1% tolerance, per constitution requirement.

**Development workflow gates**: Pre-merge validation sequence (mass prop validation, thrust diagnostic, fin articulation diagnostic, wind disturbance diagnostic) maps directly to validation ladder tests 00, 01-02, 06, and 08.

No violations. All gates pass.

## Project Structure

### Documentation (this feature)

```text
specs/007-isaac-sim-env/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   ├── observation_space.md
│   ├── action_space.md
│   └── config_schema.md
└── tasks.md             # Phase 2 output (/speckit.tasks)
```

### Source Code (repository root)

```text
simulation/isaac/
├── README.md
├── apps/
│   ├── run_single_env_debug.py       # Single-env debug with gizmos
│   ├── run_single_test.py            # Run individual validation tests
│   ├── run_eval_pid.py               # PID hover evaluation
│   ├── run_train_ppo.py              # PPO training entrypoint
│   ├── run_train_gtrxl.py            # GTrXL-PPO training entrypoint
│   └── run_smoke_128.py              # 128-env vectorized smoke test
│
├── assets/
│   ├── usd/
│   │   ├── edf_drone_v2.usd          # Articulated vehicle USD
│   │   └── landing_pad.usd           # Landing target USD
│   └── metadata/
│       └── edf_drone_v2.asset.yaml   # Asset metadata (link names, COPs, axes)
│
├── configs/
│   ├── physics/
│   │   ├── physx_single.yaml         # Single-env PhysX settings
│   │   ├── physx_train.yaml          # Training PhysX settings (GPU pipeline)
│   │   └── solver_high_fidelity.yaml # High-fidelity solver for validation
│   ├── vehicle/
│   │   └── edf_drone_v2.yaml         # Vehicle mass, geometry, inertia
│   ├── env/
│   │   ├── single_env_debug.yaml     # Single-env with gizmos
│   │   ├── train_128.yaml            # 128-env training config
│   │   └── hil_validation.yaml       # HIL-oriented validation config
│   ├── tasks/
│   │   ├── hover.yaml                # Hover task definition
│   │   └── landing.yaml              # Landing task definition
│   ├── reward/
│   │   ├── common_terms.yaml         # Shared reward term defaults
│   │   ├── hover_reward.yaml         # Hover reward weights
│   │   └── landing_reward.yaml       # Landing reward weights
│   ├── disturbances/
│   │   ├── nominal.yaml              # No disturbances
│   │   ├── wind.yaml                 # Wind model params
│   │   ├── sensor_noise.yaml         # Sensor noise params
│   │   └── com_shift.yaml            # COM offset params
│   ├── params/
│   │   ├── servo_mg996r.yaml         # Servo model params (labeled source)
│   │   ├── edf_90mm.yaml             # EDF model params (labeled source)
│   │   └── wind_model.yaml           # Wind drag coefficients
│   └── debug/
│       └── gizmos.yaml               # Gizmo enable/disable and styling
│
├── tvc_env/
│   ├── __init__.py
│   │
│   ├── common/                       # Layer 0: Shared utilities
│   │   ├── frames.py                 # FRD ↔ Isaac frame conversion (SINGLE boundary)
│   │   ├── quaternions.py            # Quaternion math (wxyz convention)
│   │   ├── transforms.py            # Rotation/translation utilities
│   │   ├── constants.py             # Physical constants, enums
│   │   └── datatypes.py             # Typed data structures
│   │
│   ├── asset/                        # Layer 1: Asset management
│   │   ├── usd_loader.py            # USD scene loading and prim access
│   │   ├── articulation_map.py      # Link/joint name ↔ index mapping
│   │   ├── hinge_axis_extractor.py  # Extract hinge axes from USD joints
│   │   ├── mass_properties.py       # Mass/inertia extraction and validation
│   │   └── asset_validator.py       # Fail-fast structural validation
│   │
│   ├── dynamics/                     # Layer 2: Physics models (pure math, no sim API)
│   │   ├── fin_geometry.py          # Fin spatial layout, COP positions
│   │   ├── fin_aero.py              # Semi-empirical vane aero model
│   │   ├── fin_force_dispatch.py    # Per-fin force computation pipeline
│   │   ├── actuator_servo.py        # MG996R servo lag/rate-limit model
│   │   ├── propulsion_edf.py        # EDF thrust + spool dynamics
│   │   ├── rotor_reaction.py        # Static/dynamic reaction torque + gyro
│   │   ├── wind_model.py            # Wind + gust + drag model
│   │   ├── com_model.py             # Center-of-mass offset model
│   │   └── state_deriv_helpers.py   # State derivative utilities
│   │
│   ├── sim/                          # Layer 3: Isaac Sim interface
│   │   ├── scene_builder.py         # InteractiveScene setup and cloning
│   │   ├── body_interface.py        # Articulation state read/write
│   │   ├── link_force_interface.py  # Per-link force application at COP
│   │   ├── wrench_dispatch.py       # Force dispatch mode switching
│   │   ├── sensor_interface.py      # Contact/IMU sensor access
│   │   ├── contacts.py              # Contact state machine
│   │   ├── reset_logic.py           # Episode reset with randomized IC
│   │   ├── crash_logic.py           # Crash detection criteria
│   │   └── gizmos.py                # Debug visualization manager
│   │
│   ├── envs/                         # Layer 4: Environment wrappers
│   │   ├── base_env.py              # Shared DirectRLEnv base
│   │   ├── single_env.py            # Single-env with gizmos enabled
│   │   ├── direct_rl_env.py         # DirectRLEnv implementation
│   │   ├── task_registry.py         # Task name → config resolver
│   │   ├── reward_registry.py       # Reward term name → function map
│   │   ├── observations.py          # Observation vector assembly
│   │   ├── rewards.py               # Reward term implementations
│   │   ├── terminations.py          # Termination conditions
│   │   ├── success_criteria.py      # Success condition checks
│   │   └── domain_randomization.py  # Per-reset randomization
│   │
│   ├── tasks/                        # Layer 4b: Task definitions
│   │   ├── hover_task.py            # Hover task config adapter
│   │   └── landing_task.py          # Landing task config adapter
│   │
│   ├── controllers/                  # Layer 5: Controller adapters
│   │   ├── base.py                  # Controller interface
│   │   ├── pid_adapter.py           # PID → action mapping
│   │   ├── pid_fin_mixer.py         # Roll/pitch/yaw → fin angle mixing
│   │   ├── ppo_adapter.py           # PPO action interpretation
│   │   └── gtrxl_adapter.py         # GTrXL-PPO action interpretation
│   │
│   └── telemetry/                    # Layer 6: Logging and export
│       ├── logger.py                # Per-step telemetry logger
│       ├── metrics.py               # Aggregate episode metrics
│       ├── plots.py                 # Diagnostic plot generation
│       └── episode_export.py        # Episode data export
│
└── tests/
    ├── unit/                         # Offline tests (no Isaac runtime)
    │   ├── test_frames.py           # Frame conversion correctness
    │   ├── test_quaternions.py      # Quaternion operations
    │   ├── test_fin_geometry.py     # Fin spatial layout
    │   ├── test_fin_aero.py         # Aero model force curves
    │   ├── test_rotor_reaction.py   # Rotor torque computations
    │   └── test_crash_logic.py      # Crash criteria logic
    │
    ├── sim/                          # Simulation tests (require Isaac Sim)
    │   ├── test_00_asset_validation.py
    │   ├── test_01_joint_axes.py
    │   ├── test_02_single_fin_articulation.py
    │   ├── test_03_unit_force_on_fin.py
    │   ├── test_04_fin_force_sweep.py
    │   ├── test_05_four_fin_superposition.py
    │   ├── test_06_edf_spool_and_reaction.py
    │   ├── test_07_gyro_precession.py
    │   ├── test_08_wind_disturbance.py
    │   ├── test_09_contact_landed_crash.py
    │   ├── test_10_pid_hover_smoke.py
    │   ├── test_11_rl_api_128env_smoke.py
    │   └── test_12_steady_hover_all_forces.py
    │
    └── goldens/                      # Reference data for regression
        ├── fin_force_curves/
        ├── reaction_torque_curves/
        └── touchdown_cases/
```

**Structure Decision**: Follows the Technical Plan's `simulation/isaac/` layout with five architectural layers (asset → dynamics → sim interface → env/task → controller), plus telemetry and a test suite split into offline unit tests and Isaac Sim simulation tests. All code lives under `simulation/isaac/tvc_env/` as a Python package.

## Complexity Tracking

No constitution violations to justify.