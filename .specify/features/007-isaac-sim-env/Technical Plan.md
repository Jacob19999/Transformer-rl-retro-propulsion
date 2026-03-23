# Phase 1 Sim Environment Implementation Plan (From Scratch, Revised)

## Purpose

This document defines the **Phase 1 simulation environment** for the EDF retro-propulsion / thrust-vectoring project.

Phase 1 is focused on building a **correct, debuggable, vectorizable simulation artifact** that can later support:

- PID evaluation,
- PPO training,
- GTrXL-PPO training,
- simulation-to-hardware validation.

The goal of Phase 1 is **not** to deliver the final flight controller.  
The goal is to produce a simulation environment that is physically consistent, modular, testable, and suitable for sim-to-real iteration.

This revision incorporates:
- current Isaac Sim / Isaac Lab version guidance,
- the JPL technical review,
- the thrust-vectoring literature caveat regarding **subsonic EDF flow** versus **supersonic rocket exhaust**,
- clearer quantitative starting assumptions for servos, EDF dynamics, wind, and HIL scope.

---

## Phase 1 Objectives

Phase 1 must establish the following:

1. A **6-DOF articulated rigid-body vehicle** in Isaac Sim.
2. A **correct fin articulation and force application model** using the actual hinge geometry.
3. A **single physics core** shared by:
   - single-env debug,
   - hover validation,
   - landing validation,
   - later RL training.
4. A **vectorized Isaac Lab environment** that can scale to 128 envs.
5. A **task-based reward system** so hover and landing reuse the same environment but use different reward profiles.
6. A **validation ladder** that isolates major failure sources:
   - asset issues,
   - frame/sign issues,
   - articulation issues,
   - force application issues,
   - contact/landing logic issues,
   - hover stability issues.
7. A debug workflow for **single-env runs** that always renders force/axis gizmos.
8. A Phase 1 configuration that is explicit about:
   - actuator lag,
   - motor spool dynamics,
   - wind assumptions,
   - subscale / subsonic modeling limitations.

---

## Recommended Stack

### Primary Stack
- **Isaac Sim 5.1**
- **Isaac Lab 2.3.2**
- **PhysX** as the baseline physics backend

### Why this stack
Use Isaac Sim for:
- USD asset authoring,
- rigid body + articulation physics,
- contacts,
- visual debugging,
- landing pad / terrain / sensors.

Use Isaac Lab for:
- scene cloning,
- vectorized environment management,
- task wiring,
- reward and termination composition,
- training integration.

Use PhysX first because:
- it is the stable and documented baseline,
- it is the practical choice for Phase 1 verification,
- it matches the current stable branch used by Isaac Lab 2.3.x,
- Newton remains a later comparative/experimental backend rather than the Phase 1 baseline.

### Version policy
Phase 1 should **not** move to Isaac Lab 3.0 / Isaac Sim 6.0 as the baseline.  
Those releases are useful to monitor, but they introduce additional churn and breaking changes that are unnecessary for the first validated environment.

### Newton policy
Newton should **not** be the Phase 1 baseline.  
If explored later, it should be introduced only behind a backend abstraction after the PhysX environment is already validated.

### API policy
Phase 1 should be implemented against the **new composable wrench / forces-and-torques path** in Isaac Lab rather than centering the design around the older `set_external_force_and_torque()` pattern.

Important nuance:
- older APIs may still exist for compatibility,
- but the architecture should target the newer wrench-composition workflow from the beginning.

### Architectural reference policy
The Isaac Lab **Multirotor / ThrusterCfg** pattern should be studied as a **reference pattern**, not adopted blindly.
This vehicle is not a standard multirotor:
- it has one EDF thrust source,
- four jet vanes,
- force redirection rather than thrust allocation across multiple independent rotors.

Still, the actuator abstraction, force composition pattern, and rise/fall dynamics ideas are useful reference material.

---

## Core Design Principles

### 1. One canonical control frame
All vehicle/control/aero math should be done in a **body-fixed aerospace frame**:

- **body_frd**
  - x = forward
  - y = right
  - z = down

This should be the only controller/aero frame used across the codebase.

### 2. One conversion boundary
All conversions between:
- body_frd,
- Isaac/USD/world,
- quaternion conventions

must live in one place only.

### 3. Physics first, controller second
The simulation environment should expose:
- state vectors,
- actuator interfaces,
- contact state,
- reward/task hooks.

It should **not** embed controller-specific assumptions.

### 4. Per-fin force before body torque
Default Phase 1 force application should be:

- compute aerodynamic force on each fin,
- apply it at the fin center of pressure,
- let body torque emerge through articulation reaction and geometry.

A fallback “collapsed body wrench” mode may be supported later for performance, but not as the primary validation mode.

### 5. Same physics core for all modes
The same physics modules must power:
- single-env debug,
- PID evaluation,
- hover test,
- landing test,
- vectorized RL envs.

Only the wrappers, reward profiles, and visualization settings should change.

### 6. Explicit modeling limitations
The Phase 1 environment must document what it is and is not modeling:
- it is a **subscale EDF proxy** for powered-descent control and disturbance rejection,
- it is **not** a high-fidelity rocket plume simulation,
- it uses thrust-vectoring references for **force decomposition and geometry reasoning**,
- it does **not** import supersonic rocket coefficient formulas directly into the EDF model.

---

## High-Level Architecture

The environment is split into five layers:

1. **Asset layer**
   - USD vehicle asset
   - metadata extraction
   - link/joint validation

2. **Physics layer**
   - fin aerodynamics
   - EDF thrust
   - rotor reaction torques
   - servo lag
   - motor lag
   - wind/disturbances

3. **Simulation interface layer**
   - scene creation
   - body/link handles
   - force/torque dispatch
   - contacts
   - resets
   - gizmos

4. **Task layer**
   - hover task
   - landing task
   - success/termination logic
   - reward profile selection

5. **Controller layer**
   - PID adapter
   - PPO adapter
   - GTrXL-PPO adapter

---

## Proposed Repository Structure

```text
simulation/isaac/
├── README.md
├── apps/
│   ├── run_single_env_debug.py
│   ├── run_single_test.py
│   ├── run_eval_pid.py
│   ├── run_train_ppo.py
│   ├── run_train_gtrxl.py
│   └── run_smoke_128.py
│
├── assets/
│   ├── usd/
│   │   ├── edf_drone_v2.usd
│   │   └── landing_pad.usd
│   └── metadata/
│       └── edf_drone_v2.asset.yaml
│
├── configs/
│   ├── physics/
│   │   ├── physx_single.yaml
│   │   ├── physx_train.yaml
│   │   └── solver_high_fidelity.yaml
│   ├── vehicle/
│   │   └── edf_drone_v2.yaml
│   ├── env/
│   │   ├── single_env_debug.yaml
│   │   ├── train_128.yaml
│   │   └── hil_validation.yaml
│   ├── tasks/
│   │   ├── hover.yaml
│   │   └── landing.yaml
│   ├── reward/
│   │   ├── common_terms.yaml
│   │   ├── hover_reward.yaml
│   │   └── landing_reward.yaml
│   ├── disturbances/
│   │   ├── nominal.yaml
│   │   ├── wind.yaml
│   │   ├── sensor_noise.yaml
│   │   └── com_shift.yaml
│   ├── params/
│   │   ├── servo_mg996r.yaml
│   │   ├── edf_90mm.yaml
│   │   └── wind_model.yaml
│   └── debug/
│       └── gizmos.yaml
│
├── tvc_env/
│   ├── __init__.py
│   │
│   ├── common/
│   │   ├── frames.py
│   │   ├── quaternions.py
│   │   ├── transforms.py
│   │   ├── constants.py
│   │   └── datatypes.py
│   │
│   ├── asset/
│   │   ├── usd_loader.py
│   │   ├── articulation_map.py
│   │   ├── hinge_axis_extractor.py
│   │   ├── mass_properties.py
│   │   └── asset_validator.py
│   │
│   ├── dynamics/
│   │   ├── fin_geometry.py
│   │   ├── fin_aero.py
│   │   ├── fin_force_dispatch.py
│   │   ├── actuator_servo.py
│   │   ├── propulsion_edf.py
│   │   ├── rotor_reaction.py
│   │   ├── wind_model.py
│   │   ├── com_model.py
│   │   └── state_deriv_helpers.py
│   │
│   ├── sim/
│   │   ├── scene_builder.py
│   │   ├── body_interface.py
│   │   ├── link_force_interface.py
│   │   ├── wrench_dispatch.py
│   │   ├── sensor_interface.py
│   │   ├── contacts.py
│   │   ├── reset_logic.py
│   │   ├── crash_logic.py
│   │   └── gizmos.py
│   │
│   ├── envs/
│   │   ├── base_env.py
│   │   ├── single_env.py
│   │   ├── direct_rl_env.py
│   │   ├── task_registry.py
│   │   ├── reward_registry.py
│   │   ├── observations.py
│   │   ├── rewards.py
│   │   ├── terminations.py
│   │   ├── success_criteria.py
│   │   └── domain_randomization.py
│   │
│   ├── tasks/
│   │   ├── hover_task.py
│   │   └── landing_task.py
│   │
│   ├── controllers/
│   │   ├── base.py
│   │   ├── pid_adapter.py
│   │   ├── pid_fin_mixer.py
│   │   ├── ppo_adapter.py
│   │   └── gtrxl_adapter.py
│   │
│   └── telemetry/
│       ├── logger.py
│       ├── metrics.py
│       ├── plots.py
│       └── episode_export.py
│
└── tests/
    ├── unit/
    │   ├── test_frames.py
    │   ├── test_quaternions.py
    │   ├── test_fin_geometry.py
    │   ├── test_fin_aero.py
    │   ├── test_rotor_reaction.py
    │   └── test_crash_logic.py
    │
    ├── sim/
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
    └── goldens/
        ├── fin_force_curves/
        ├── reaction_torque_curves/
        └── touchdown_cases/
```

---

## Frame Conventions and Transform Policy

### Internal control/aero frame
Use:
- `body_frd`
  - +x forward
  - +y right
  - +z down

All controller outputs, aero computations, and simplified rigid-body reasoning should use this frame.

### Isaac/world frame
Use Isaac Sim’s documented world convention:
- +X forward
- +Z up
- right-handed
- quaternion order `(w, x, y, z)`

### Conversion rule
Never scatter sign flips across the codebase.

All conversions must pass through:
- `common/frames.py`
- `common/quaternions.py`
- `common/transforms.py`

### Result
This avoids the earlier class of bugs where:
- fin angle sign is correct in one file,
- hinge axis sign is flipped in another,
- torque is applied in local frame but interpreted as world frame.

---

## USD Asset Requirements

The vehicle asset should contain:
- a main rigid body / root,
- four fin links,
- four revolute joints,
- appropriate colliders on body and landing structure,
- mass and inertia on all dynamic links,
- consistent parent-child link hierarchy.

### Required metadata to extract
From USD or companion metadata:
- body link name
- fin link names
- fin joint names
- hinge axes
- joint limits
- fin COP local position
- fin chord direction
- fin normal direction at neutral angle
- EDF thrust axis
- rotor spin axis
- landing-gear contact regions

### Asset validation
The environment must fail fast if:
- a fin link is missing,
- a joint axis is undefined,
- a joint is not revolute,
- mass/inertia are invalid,
- link hierarchy is inconsistent.

---

## Physics Model Scope for Phase 1

Phase 1 should include the following modeled effects.

### Always enabled
- gravity
- rigid body dynamics
- articulations
- colliders and contact
- EDF thrust
- fin articulation
- per-fin aerodynamic force model

### Enabled in integrated tests
- servo lag
- motor spool lag
- RPM slew/rate limit
- static rotor reaction torque
- dynamic spool torque
- gyro precession
- wind disturbance

### Optional toggles
- sensor noise
- center-of-mass shift
- aerodynamic drag at body level
- contact/noise stress scenarios

---

## Flow Regime and Modeling Boundaries

### What the EDF artifact represents
The Phase 1 EDF vehicle is a **subscale, subsonic proxy** for powered-descent control and disturbance rejection.

It is intended to validate:
- control authority structure,
- sign correctness,
- actuator dynamics,
- disturbance rejection,
- hover/landing task wiring,
- sim-to-hardware iteration on the artifact.

It is **not** intended to directly reproduce:
- supersonic rocket plume physics,
- combustion chemistry,
- oblique shock / expansion-wave interactions,
- full-scale reentry aerothermodynamics.

### How the jet-vane references are used
The thrust-vectoring papers are used for:
- normal/tangential force decomposition,
- geometric reasoning,
- moment-from-force structure,
- intuition for finite-angle behavior.

They are **not** used directly for:
- supersonic coefficient values,
- Mach>1 formula transfer,
- rocket thermal/plume modeling.

### Phase 1 aerodynamic policy
Phase 1 should use a **subsonic EDF-appropriate vane model**, then calibrate it later with bench data.

---

## Fin Aerodynamic Model

Phase 1 should use a **semi-empirical jet-vane engineering model**, not a free-stream aircraft wing model and not a direct copy of supersonic rocket-vane coefficients.

### Concept
Each fin in the exhaust stream generates:
- a **normal/control-producing force**
- a **tangential/thrust-loss force**

These arise from vane deflection relative to the jet flow.

### Per-fin inputs
- actual joint angle
- local hinge axis
- local chord direction
- local fin normal
- local exhaust speed
- density
- fin area
- coefficient parameters

### Per-fin outputs
- force vector in fin-local or body-local frame
- optional diagnostic terms:
  - normal force
  - tangential force
  - thrust loss estimate

### Initial model behavior
Desired qualitative characteristics:
- near-zero normal force at zero deflection
- nonzero tangential drag/thrust loss possible at zero deflection
- normal force approximately linear for small/moderate angles
- nonlinear saturation at larger angles
- drag increases with deflection magnitude

### Initial coefficient policy
Use a subsonic thin-airfoil / flat-plate style starting approximation with correction terms for:
- aspect ratio,
- duct confinement,
- finite-angle saturation,
- empirical calibration.

This is a starting point, not the final identified model.

### Important implementation rule
The default Phase 1 path is:
1. compute force on the fin,
2. transform it to world,
3. apply it to the **fin link** at the fin COP.

Do not directly synthesize a body pitch/yaw/roll torque from fin angle as the primary physics path.

### Known Phase 1 limitations
The initial model may ignore or simplify:
- vane-to-vane interference,
- nonuniform duct velocity profile,
- duct wall interaction at large deflection,
- full separated-flow behavior.

These are acceptable Phase 1 simplifications but must be logged as known limitations for later calibration.

---

## Force Dispatch Strategy

Two supported modes:

### 1. `per_link_force` (default)
- compute force per fin
- apply each force to the corresponding fin link at COP
- body reaction is handled by articulation physics

Use for:
- single-env debug
- validation tests
- PID evaluation
- hover validation
- HIL-oriented checks

### 2. `collapsed_body_wrench` (optional fallback)
- compute all fin forces
- sum them into:
  - total body force
  - total body torque = sum of `r × F`
- apply one net body wrench

Use only if later profiling shows a need for higher throughput.

### Dispatch implementation policy
Do not hard-wire the implementation around deprecated APIs.

Create a small dedicated dispatch layer:
- `sim/wrench_dispatch.py`
- `sim/link_force_interface.py`

This layer should:
- accept forces/torques/positions in an internal standardized format,
- convert to the current Isaac Lab wrench application API,
- isolate any future API migration.

### Policy
Phase 1 correctness should be proven in `per_link_force` mode first.

---

## Servo Model

Because the project now uses **MG996R** servos, Phase 1 should reflect:
- higher mass,
- higher torque,
- slower transient response than tiny digital micro servos,
- increased power/current demand,
- realistic command lag and rate limits.

### Servo model requirements
- first-order lag
- angular rate limit
- max deflection limit
- optional deadband/backlash approximation
- unit-to-unit variance hook for later domain randomization

### Candidate starting parameters
These are **starting simulation values**, to be confirmed and refined with hardware measurements and datasheet verification.

| Parameter | Candidate starting value | Notes |
|-----------|--------------------------|-------|
| Mass per servo | 0.055 kg | MG996R class |
| Stall torque @6V | 1.08 N·m | datasheet-class starting point |
| Transit time (60°) @6V | 0.14 s | datasheet-class starting point |
| Max angular velocity | 7.5 rad/s | derived from transit time |
| First-order lag τ_servo | 0.04–0.07 s | tune from bench step response |
| Deadband | 1–2° equivalent | initial approximation |
| Max command angle | config-defined | based on linkage clearance |

### Why this matters
If servo dynamics are omitted, hover and landing results will be overly optimistic and less useful for sim-to-real transfer.

---

## EDF / Propulsion Model

Phase 1 propulsion should include:

### 1. Thrust
- thrust along the EDF axis
- mapped from RPM or commanded throttle through a chosen coefficient model

### 2. Motor spool dynamics
- first-order lag
- max RPM change rate

### 3. Static rotor reaction torque
- present during steady rotor rotation

### 4. Dynamic spool torque
- appears during rotor acceleration or deceleration

### 5. Gyro precession
- computed from body angular velocity crossed with rotor angular momentum

This is important because the proposal explicitly targets sim-to-hardware transfer and disturbance-resistant control rather than an oversimplified academic simulation.

### Candidate starting EDF parameters
These are **initial engineering estimates** for Phase 1 configuration and must be refined with thrust-stand / logging data.

| Parameter | Candidate starting value | Notes |
|-----------|--------------------------|-------|
| Static thrust at full command | 48 N | from current artifact target |
| EDF diameter | 90 mm | current hardware class |
| k_T | derive from thrust and max RPM estimate | fill from bench data |
| k_Q | derive from k_T and efficiency estimate | refine from measurements |
| Rotor inertia I_fan | estimated / measured | must be bench-estimated |
| τ_motor | 0.10–0.30 s | 90 mm class initial range |
| ω_max | config-defined | from motor/ESC data |
| dω_max | config-defined | from ESC + logging |

### Modeling policy
The configuration file must clearly separate:
- **measured values**
- **datasheet values**
- **engineering estimates**
- **to-be-calibrated values**

---

## Rotor Reaction and Gyroscopic Terms

Phase 1 should compute:
- static rotor reaction torque,
- dynamic spool torque,
- gyro precession.

### Validation policy
These torques must be logged separately during the validation ladder so their relative magnitudes can be compared against:
- fin-generated torques,
- wind-induced moments,
- contact-induced moments.

This is important because gyro precession may be physically correct but small relative to other moment sources.

---

## Wind and Disturbances

Phase 1 should support a wind toggle and disturbance framework.

### Disturbances to support
- constant wind
- gust pulses
- randomized lateral gusts
- center-of-mass offset
- optional sensor noise

### Phase 1 wind model
Use a simple but explicit model:
- steady-state wind vector,
- optional gust events,
- body-force application based on relative airspeed and simple drag assumptions.

A reasonable initial force structure is:
- body drag aligned opposite relative airflow,
- configurable reference area and drag coefficient,
- optional gust injection on top of the steady vector.

### Why
The research proposal explicitly includes disturbance robustness evaluation and the sim environment must support those cases from the beginning.

---

## Contacts, Landed Detection, and Crash Detection

Phase 1 should use a **state machine**, not one-frame checks.

### States
- `AIRBORNE`
- `GROUND_CONTACT_CANDIDATE`
- `LANDED`
- `CRASHED`

### Landed logic
The vehicle becomes landed only if, for a dwell interval:
- contact exists,
- vertical speed is below threshold,
- lateral speed is below threshold,
- tilt is below threshold,
- angular rate is below threshold.

### Crash logic
The vehicle becomes crashed if:
- impact speed exceeds threshold,
- excessive tilt occurs on contact,
- excessive angular rate persists on contact,
- body/unsafe structure contacts the ground,
- bounce leads to tip-over / loss of recovery.

### Important note
A bounce must **not** be considered landed immediately.

---

## Observation Space

The shared environment should expose a controller-agnostic observation vector.

### Minimum observation set
- position error to target/pad
- attitude representation
- linear velocity
- angular velocity
- height above ground
- fin actual angles
- fin actual rates
- motor RPM
- contact state
- optional disturbance state / wind estimate

The exact encoding may vary later, but the environment should not be rewritten to support different controllers.

---

## Action Space

Recommended default:
- 4 fin target angle commands
- 1 motor/throttle/RPM target

### Controller interpretation
- PID adapter may use an external mixer
- PPO may output raw fin commands
- GTrXL-PPO may output raw fin commands

This avoids baking a PID worldview into the physics environment.

---

## Task Structure

The environment should support multiple tasks through configuration.

### Phase 1 tasks
- **Hover**
- **Landing**

The environment physics stays the same.  
Only task settings change:
- target conditions
- reward profile
- success criteria
- termination logic
- spawn ranges
- disturbance settings

---

## Reward System Refactor

Reward logic should be composable and selected by task profile.

### Structure
- `reward_registry.py` maps reward term names to functions
- `hover_reward.yaml` defines hover task weights
- `landing_reward.yaml` defines landing task weights
- `common_terms.yaml` stores shared defaults/tolerances

### Shared reward terms
Potential shared terms:
- alive bonus
- position error penalty
- attitude error penalty
- angular velocity penalty
- control effort penalty
- control rate penalty
- saturation penalty
- crash penalty

### Hover-only reward terms
- hover stability bonus
- drift penalty
- contact penalty

### Landing-only reward terms
- touchdown softness
- landing success bonus
- pad accuracy bonus
- vertical speed shaping near touchdown

### Design rule
Do not create separate reward code paths per algorithm.  
Use the same reward framework and different configs.

---

## Hover Task (Phase 1 Validation Task)

Hover is an important Phase 1 validation task because it tests the integrated environment before landing.

### Hover task purpose
Verify that the vehicle can maintain bounded, physically plausible, disturbance-aware hover using all relevant modeled forces and lags.

### Hover target behavior
- hold a target altitude
- minimize lateral drift
- remain near upright
- avoid oscillation/chatter
- avoid ground contact

### Forces enabled in hover validation
- gravity
- thrust
- fin aero
- static rotor torque
- dynamic spool torque
- gyro precession
- servo lag
- motor lag
- optional wind

### Hover success
Hover is considered stable only after:
- position error stays within tolerance,
- tilt stays within tolerance,
- angular velocity stays within tolerance,
- no contact occurs,
- all conditions persist for a dwell interval.

---

## Landing Task (Primary Research Task)

Landing remains the main downstream research task.

### Landing target behavior
- descend safely
- manage vertical and horizontal velocity
- align attitude
- touch down inside landing region
- avoid crash and tip-over

### Landing reward emphasis
- vertical speed control
- lateral accuracy
- attitude stability
- touchdown softness
- success bonus
- strong crash penalty

---

## Single-Env Debug Visualization (Gizmos)

For all **single-env** runs, Phase 1 should always render debug gizmos.

### Required gizmos
- body local axes
- body COM marker
- target point / landing pad marker
- thrust vector
- per-fin force arrows
- total aerodynamic force arrow
- reaction torque arrow
- contact normals
- optional hover tolerance volume

### Required HUD values
- altitude error
- XY position error
- tilt
- body-rate magnitude
- motor RPM
- per-fin angle
- per-fin force magnitude
- total reward
- task mode
- landed/crashed/stable-hover state

### Policy
Single-env debug is not optional in Phase 1.

---

## Vectorized Training Mode

The environment should support multi-env execution through Isaac Lab scene replication.

### Target
- 128 environments

### Expected usage
- single-env debug with gizmos on
- multi-env training with gizmos off

### Configuration direction
Use:
- shared assets,
- replicated physics,
- consistent scene cloning,
- no task-specific physics forks.

---

## HIL / External Integration Scope

The research program includes HIL and hardware-facing validation, but the Phase 1 environment itself is Python/Isaac-Lab centered.

### Phase 1 HIL scope
Phase 1 should prepare for later HIL by:
- keeping actuator/state interfaces clean,
- logging command/state histories,
- defining message schemas or adapter boundaries,
- avoiding controller-specific coupling inside the physics core.

### Not required in Phase 1
Phase 1 does **not** need to fully solve:
- Simulink bridge implementation,
- Jetson deployment pipeline,
- full real-time co-simulation synchronization.

Those should be scoped as follow-on integration tasks once the simulation artifact is validated.

---

## Incremental Validation Ladder

Phase 1 should be implemented and validated in the following order.

### `test_00_asset_validation.py`
Validate:
- link existence
- joints
- hinge axes
- masses/inertias
- metadata consistency

### `test_01_joint_axes.py`
Command fin motion with no aero force:
- verify axis direction
- verify positive/negative sign
- verify joint limits

### `test_02_single_fin_articulation.py`
One fin moves, body mostly isolated:
- verify articulation behavior
- verify no unintended parent transform corruption

### `test_03_unit_force_on_fin.py`
Apply a known synthetic force at a fin COP:
- verify observed reaction sign
- verify expected `r × F` body effect

### `test_04_fin_force_sweep.py`
Sweep one fin through deflections under exhaust flow:
- inspect force curves
- inspect normal/tangential trend
- inspect saturation behavior

### `test_05_four_fin_superposition.py`
Apply known command patterns:
- roll-only
- pitch-only
- yaw-only
- symmetric
- opposite-pair

Verify summed wrench behavior.

### `test_06_edf_spool_and_reaction.py`
Step motor command:
- verify thrust lag
- verify static reaction torque
- verify dynamic spool torque

### `test_07_gyro_precession.py`
Impose body angular motion with spinning rotor:
- verify precession direction and magnitude trend
- log relative magnitude compared with fin torque

### `test_08_wind_disturbance.py`
Inject constant/gust wind:
- verify response direction
- verify no frame sign errors when rotated

### `test_09_contact_landed_crash.py`
Run scripted touchdowns:
- soft landing
- bounce-and-recover
- hard impact
- tip-over

### `test_10_pid_hover_smoke.py`
Simple closed-loop hover:
- no NaNs
- no sign mistakes
- bounded hover possible

### `test_11_rl_api_128env_smoke.py`
Vectorized env API check:
- reset
- observe
- step
- terminate
- no tensor shape issues

### `test_12_steady_hover_all_forces.py`
Integrated hover validation:
- all modeled forces enabled
- all major lags enabled
- success metrics logged

---

## Acceptance Criteria for Phase 1

Phase 1 is complete when all of the following are true:

1. The vehicle asset loads cleanly and validates successfully.
2. Fin articulation signs and axes are verified.
3. Synthetic force tests confirm correct reaction direction.
4. Per-fin aerodynamic force behavior is sane and bounded.
5. EDF spool and rotor torque tests pass.
6. Contact logic distinguishes between:
   - airborne,
   - bounce,
   - landed,
   - crashed.
7. A PID controller can maintain bounded hover in the all-forces hover test.
8. The shared environment can run in vectorized mode at 128 envs.
9. Hover and landing tasks can be selected by configuration without forking the physics code.
10. Single-env debug renders all required gizmos and telemetry.
11. Servo and EDF parameter files explicitly separate:
    - measured values,
    - datasheet values,
    - engineering estimates,
    - to-be-calibrated values.
12. The proposal and implementation both explicitly identify the EDF artifact as a **subsonic proxy**, not a direct rocket-plume analog.

---

## Recommended Implementation Order

### Step 1
Build:
- `common/`
- `asset/`
- USD validator
- frame conversion utilities

### Step 2
Build articulation and joint inspection tools:
- joint axis extraction
- fin angle readback
- single-fin motion tests

### Step 3
Build force dispatch path:
- synthetic unit-force test
- per-link COP application
- wrench-dispatch adapter

### Step 4
Build fin aero model:
- force sweep
- coefficient tuning
- visualization

### Step 5
Build propulsion and rotor reaction modules

### Step 6
Build contacts and touchdown/crash logic

### Step 7
Build reward registry and task configs

### Step 8
Build PID hover and full-force hover test

### Step 9
Wrap into Isaac Lab vectorized env

### Step 10
Profile and optimize only after correctness is proven

---

## Out of Scope for Phase 1

Phase 1 does **not** include:
- full-scale vehicle fidelity,
- actual rocket exhaust chemistry,
- untethered hardware flight,
- transonic reentry aerothermodynamics,
- deployment-level certification,
- final GTrXL-PPO benchmarking campaign,
- Newton as the baseline backend,
- irreversible training optimization before force validation,
- a complete Simulink/HIL bridge.

---

## Risks and Mitigations

### Risk 1: Frame/sign bugs
**Mitigation:** one frame module only, synthetic unit-force tests, mandatory gizmos.

### Risk 2: Incorrect fin torque behavior
**Mitigation:** apply force on fin links at COP, not arbitrary body torques.

### Risk 3: Unrealistically optimistic control performance
**Mitigation:** include servo lag, motor lag, RPM slew, rotor torques from Phase 1.

### Risk 4: Landing logic falsely declares success after bounce
**Mitigation:** stateful landed/crashed state machine with dwell thresholds.

### Risk 5: Reward duplication and environment drift
**Mitigation:** task configs + reward registry, one shared environment.

### Risk 6: Training too early on a broken physics model
**Mitigation:** enforce validation ladder before large-scale RL runs.

### Risk 7: Over-claiming rocket fidelity from EDF results
**Mitigation:** explicitly frame the artifact as a subscale, subsonic control proxy and separate force-structure reasoning from rocket-specific coefficient physics.

---

## Final Phase 1 Summary

Phase 1 should deliver a **single, validated, from-scratch Isaac Sim + Isaac Lab environment** for the EDF thrust-vectoring vehicle with:

- articulated body and fins,
- correct frame handling,
- force-at-position fin modeling,
- realistic actuator and propulsion lag,
- hover and landing task support,
- single-env debug visualization,
- 128-env vectorized readiness,
- and an ordered validation ladder proving the environment before training.

This environment becomes the foundation for:
- PID comparisons,
- PPO training,
- GTrXL-PPO training,
- later HIL integration,
- and later simulation-to-hardware transfer work.

The revised Phase 1 baseline is:

- **Isaac Sim 5.1**
- **Isaac Lab 2.3.2**
- **PhysX**
- **FRD internal frame contract**
- **per-link force-at-COP default**
- **wrench-dispatch adapter targeting the modern composable force/torque path**
- **explicit subsonic EDF modeling assumptions**
- **task-configurable hover and landing rewards**
- **debug-first validation before RL scale-up**
