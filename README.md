# EDF Retro-Propulsion

**Simulation-to-Hardware Validation of Convex Guidance for Disturbance-Resistant Retro-Propulsive Landings**

> Thrust-vectoring control of an Electric Ducted Fan (EDF) drone, from a physics-corrected Isaac Sim plant to tethered flight tests.

---
Isaac Sim environment (parallel envs):
<img width="2517" height="1244" alt="image" src="https://github.com/user-attachments/assets/8a5a5e4d-327d-45d1-8080-6d50f27499b4" />

---

## Current Status (September 2026)

Phase 1 (simulation) is in progress. The controller in use is explicit convex guidance; learned-policy work was removed from the repository on 2026-09-28 and is recoverable from git history.

| Controller | Result on the corrected Isaac Sim plant |
|------------|------------------------------------------|
| **Convex (SOCP) guidance** | Landed all 7 benchmark missions inside the vehicle envelope (touchdown 0.146–0.184 m/s, 0.05–0.40 m from pad centre), including combined wind + sensor noise + CoM offset, a 50 m cold-rotor start and a two-waypoint route. Also flew a 100 s three-hover route and landed four ~100 m starts in wind at up to 30 m/s initial speed. |
| **PID baseline** | Touched down at 1.50 m/s on the default mission (above the 0.5 m/s target). |

The dominant physical obstacle is **rotor–body angular-momentum exchange**: every throttle change yaws the airframe (~46.5 rad/s body yaw per unit throttle change in the unmitigated plant), while the four jet vanes provide only ~0.3 N·m of yaw authority.

### Recent development

- **Physics corrections to the simulator.**
  - A finite-momentum coupled-jet vane model replaces independent per-vane lift forces. The legacy model gave ~8.5× the physically possible side force.
  - The EDF is coupled to a LiPo equivalent circuit.
  - Rotor spool and gyroscopic reactions are integrated with an energy-conserving implicit-midpoint (Cayley) scheme.
  - External forces are applied about the true centre of mass.
- **Torque-limited motor model** ([mission_plant.yaml](simulation/isaac/configs/env/mission_plant.yaml)). The motor applies at most K<sub>t</sub>·I<sub>max</sub> ≈ 0.76 N·m, and zero throttle coasts with no brake, as EDF ESCs do. The previously unbounded spool lag put 5.2 N·m on the body at disarm and spun landed vehicles at 430–710 °/s on the pad.
- **Convex guidance controller** ([convex_guidance_2026-09-25.md](simulation/isaac/docs/convex_guidance_2026-09-25.md)). It adapts Açıkmeşe & Ploen (2007) lossless convexification to a constant-mass, battery-powered vehicle, solved in closed loop with Clarabel.
  - It minimizes electrical energy with a power-cone objective.
  - Added constraints: a thrust-vector rate bound derived from the vanes' yaw authority, a gate approached from above, and waypoint nodes.
  - A servo-deadband inverse removed a 1 Hz gyroscopic coning limit cycle, cutting hover body-rate RMS from 15 to 1.5 °/s.
- **IMU measurement model** ([imu_model.md](simulation/isaac/docs/imu_model.md)). An opt-in chain models the WTGAHRS1: sensor bias and drift, 20 Hz low-pass, output rate, latency and an onboard attitude filter. Most parameters are placeholders until bench data exists (`tools/imu_allan_variance.py`).
- **Mission Control** ([mission_control.md](simulation/isaac/docs/mission_control.md)). A local web console (`simulation/isaac/mission_control/start.ps1`, http://127.0.0.1:8830) plans and launches Isaac missions with convex guidance. It replays telemetry on a 3D vehicle model and shows convex-guidance diagnostics.

### Next steps

- Measure EDF rotor inertia and servo deadband on hardware, then begin HIL integration
- Characterize the WTGAHRS1 on the bench and replace the IMU model's placeholder parameters
- Wire the IMU model into mission control and fly convex missions with it enabled

## Overview

This project investigates the **simulation-to-hardware transfer** of **convex (SOCP) powered-descent guidance** for thrust-vectoring control (TVC) in disturbance-resistant retro-propulsive landings. Motivated by the growing need for rocket booster recovery in reusable launch vehicles (e.g., SpaceX Falcon 9, Starship, Blue Origin New Glenn), the project uses a scaled EDF drone testbed to study a controller family that is transparent, deterministic and certifiable in ways learned controllers are not:

- **PID controllers** struggle with nonlinear dynamics, parameter variations, and external disturbances.
- **Convex guidance** plans a minimum-energy thrust trajectory under thrust, tilt, glide-slope, speed and thrust-rate constraints and is re-solved in closed loop; large perturbations can still push a plan infeasible, so its robustness envelope must be characterized.

### Key Goals

- Validate convex guidance in high-fidelity simulation (NVIDIA Isaac Sim)
- Transfer the controller to a physical Electric Ducted Fan (EDF) drone testbed
- Quantitatively compare convex guidance against a PID baseline
- Characterize robustness under realistic disturbances (wind, sensor noise, CoM shifts, varying initial conditions)
- Advance the technology from **TRL 3** (analytical proof-of-concept) to **TRL 5** (validated in relevant environment)

---

## Disturbance Envelope

The system is designed to handle the following perturbations:

| Disturbance | Range |
|-------------|-------|
| Wind gusts | Up to 10 m/s (simulated via fans in hardware) |
| Sensor noise | Gaussian, sigma = 0.1 - 0.5 m/s² |
| Center of mass shift | +/- 10% variation |
| Initial altitude | 5 - 10 m |
| Initial velocity | 0 - 5 m/s |
| Fuel slosh (hardware) | Variable water payload on EDF |

---

## Project Architecture -- High-Level Modules

The project is organized into the following major modules. Module paths below are the planned architecture; see [Repository Layout](#repository-layout) for where the implemented code lives today.

### 1. Simulation Environment (`simulation/`)

The high-fidelity simulation backbone built on **NVIDIA Isaac Sim**.

- **6-DOF Rigid Body Dynamics**: Full rotational and translational dynamics with realistic mass properties
- **Dynamic Center of Mass Model**: Simulates CoM variation due to fuel consumption and payload shifts
- **Disturbance Injection Framework**: Configurable wind field models, Gaussian sensor noise injection, CoM perturbation profiles
- **Landing Terrain**: Simulated landing pad with ground contact physics
- **Sensor Simulation**: Emulated IMU (bias, drift, filtering, latency), optical flow, and barometric sensor outputs matching hardware specifications
- **Data Logging**: Automated state vector and metric collection per run

### 2. Controllers (`baselines/`)

- **Convex Guidance (SOCP)**: Optimization-based powered descent guidance using lossless convexification of the thrust constraints (Açıkmeşe & Ploen, 2007), tracked by position feedback and a geometric attitude loop
- **PID Controller**: Ziegler-Nichols tuned proportional-integral-derivative controller for attitude and position control

### 3. Hardware Platform (`hardware/`)

The physical EDF drone testbed for real-world validation.

- **Airframe & Propulsion**
  - FMS 90 mm metal ducted fan (12-blade, ~45 N thrust)
  - Thrust-to-weight ratio throttled to ~1.3 via PWM for realistic rocket-like dynamics
  - 120 A ESC with 8S LiPo battery
- **Thrust Vector Control (TVC)**
  - 4x KST DS215MG servo-actuated control fins in the thrust stream
  - Emulates rocket engine gimbal for attitude control
- **Sensor Suite**
  - Primary IMU: BNO085 (static error 2.0 deg, dynamic error 3.5 deg)
  - Backup IMU: WitMotion WTGAHRS1 (10-axis, high-stability AHRS with GPS)
  - PX4 optical flow camera (0.1 m accuracy at 1 Hz) for position estimation
- **Compute**
  - NVIDIA Jetson Nano (128-core Maxwell GPU, 4-core ARM CPU, 4 GB LPDDR4)
  - Real-time target: < 50 ms latency
- **Frame**: Carbon fibre rods with 3D-printed joints and mounts
- **Bill of Materials**: Estimated total < $1,000 USD

### 4. Hardware-in-the-Loop (HIL) Integration (`hil/`)

Bridging simulation and hardware before physical flight.

- **MATLAB Simulink Integration**: Real-time HIL pipeline connecting Isaac Sim dynamics to physical hardware I/O
- **Synthetic Sensor Feed**: Simulink feeds synthetic sensor data to the Jetson Nano running the controller
- **Latency Profiling**: End-to-end measurement of sensor-input to control-output delay
- **Transfer Fidelity Assessment**: Pearson correlation (target r > 0.9) between simulation and hardware metrics
- **Trial Volume**: ~500 HIL trials per controller with controlled disturbance injection

### 5. Flight Test Framework (`flight_tests/`)

Controlled tethered flight validation of the final system.

- **Test Environment**: Indoor controlled space (~10 x 10 m), tethered flights for safety
- **Test Protocol**: Autonomous descent from 5-10 m altitude with precision landing on marked pad
- **Disturbance Hardware**: External fans (wind injection), variable water payloads (CoM shift and fuel slosh emulation), added weights
- **Data Collection**: Onboard sensor logs at 100 Hz, optical flow ground-truth tracking via ground markers
- **Trial Volume**: 50-100+ tethered flights for the convex guidance controller
- **Safety Systems**: Manual kill switch, software-defined flight envelope limits, tether constraint, pre-flight checklists

### 6. Evaluation & Analysis (`evaluation/`)

Statistical analysis and visualization pipeline for all experimental phases.

- **Statistical Methods**
  - Paired t-tests and ANOVA (alpha = 0.05, power = 0.8) for inter-controller comparisons
  - Monte Carlo analysis (n = 100) for robustness characterization
  - Pearson's r for sim-to-hardware transfer correlation
  - 95% confidence intervals on all reported metrics
- **Metrics Suite**
  - Landing dispersion (CEP), jerk, touchdown velocity, success rate
  - Trajectory RMSE, recovery time, control effort, robustness margin
  - Delta-V (trajectory efficiency), simulated fuel remaining
  - Controller latency (sensor-to-actuator)
- **Visualization**: Trajectory plots, landing scatter maps, metric comparison tables (via Matplotlib)

### 7. Deployment & Artifacts (`artifacts/`)

Open-source deliverables and reproducibility assets.

- **Source Code**: Isaac Sim environment, controllers and mission tooling (GitHub)
- **Hardware Documentation**: Full bill of materials, CAD files for 3D-printed components, wiring diagrams
- **Datasets**: 100+ flight test logs with full state vectors for community benchmarking
- **Reproducibility**: Configuration files and random seeds

---

## Repository Layout

| Path | Contents |
|------|----------|
| [simulation/isaac/tvc_env/](simulation/isaac/tvc_env/) | Isaac Lab environment package: dynamics (EDF, coupled jet, LiPo, IMU), envs (`direct_rl_env`, waypoints), controllers (PID, convex) |
| [simulation/isaac/configs/](simulation/isaac/configs/) | Plant/env configs, task definitions (`hover`, `landing`), sensor profiles and controller settings |
| [simulation/isaac/apps/](simulation/isaac/apps/) | Entry points: `run_mission.py`, PID evaluation and sweeps, smoke tests |
| [simulation/isaac/mission_control/](simulation/isaac/mission_control/) | Local mission-control web console (server + frontend) |
| [simulation/isaac/docs/](simulation/isaac/docs/) | Dated design notes and investigation reports (physics review, convex guidance, IMU model) |
| [simulation/isaac/tests/](simulation/isaac/tests/) | Unit tests (`pytest`) |
| [Paper/](Paper/) | Draft paper (LaTeX) and references |
| [CAD/](CAD/) | EDF drone CAD, Blender/USD models and FEA |

---

## Methodology

The research follows a **design science** framework with three sequential experimental phases:

```
Phase 1: Simulation          Phase 2: HIL Testing          Phase 3: Flight Tests
┌─────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│ Isaac Sim 6-DOF     │     │ MATLAB Simulink +    │     │ Tethered EDF drone   │
│ environment setup   │────>│ Jetson Nano HIL      │────>│ indoor flight tests  │
│                     │     │                      │     │                      │
│ - Convex guidance   │     │ - Transfer fidelity  │     │ - 50-100+ landings   │
│ - Tune PID (Z-N)    │     │ - Latency profiling  │     │ - Disturbance inject │
│ - Disturbance models│     │ - 500 trials/variant │     │ - Ground truth via   │
│ - IMU model         │     │ - Correlation r>0.9  │     │   optical flow       │
└─────────────────────┘     └──────────────────────┘     └──────────────────────┘
```

### Phase 1 -- Simulation & Evaluation
- High-fidelity 6-DOF simulation in NVIDIA Isaac Sim with disturbance models
- Convex guidance and PID baseline implementation and tuning
- Initial metric evaluation and statistical comparison across controllers

### Phase 2 -- Hardware-in-the-Loop (HIL)
- Deploy the controller on Jetson Nano embedded compute
- Real-time HIL integration via MATLAB Simulink feeding synthetic sensor data
- Characterize latency, transfer fidelity, and controller behavior under simulated hardware constraints
- ~500 trials per variant with disturbance injection

### Phase 3 -- Controlled Flight Tests
- Tethered EDF drone flights in indoor environment
- Autonomous descent and precision landing from 5-10 m
- Physical disturbance injection (fans, variable payloads, water for slosh)
- ~100 flights with full state-vector logging at 100 Hz
- Statistical analysis against simulation predictions

---

## Baselines & Comparison Strategy

| Controller | Description | Tuning Method |
|------------|-------------|---------------|
| **Convex guidance** | SOCP powered-descent guidance, re-solved in closed loop | Solver and constraint configuration |
| **PID** | Classical proportional-integral-derivative controller | Ziegler-Nichols |

All controllers are evaluated on identical scenarios with statistical comparison via t-tests/ANOVA (alpha = 0.05).

---

## Timeline

| Phase | Target Date |
|-------|-------------|
| Literature review & research questions | Jan 2026 |
| Research design finalization | Feb 2026 |
| Isaac Sim environment construction | Mar 2026 |
| Simulation experiments | Mar - Jul 2026 |
| Hardware build & HIL integration | Jul - Nov 2026 |
| Flight test data collection | Nov 2026 |
| Data analysis | Dec 2026 |
| Draft thesis | Jan 2027 |
| Defense | Mar 2027 |
| Final submission | Apr 2027 |

---

## Scope & Limitations

### In Scope
- Convex guidance and PID for 6-DOF landing control
- NVIDIA Isaac Sim simulation with disturbance models
- EDF drone hardware testbed with TVC emulation
- HIL testing and tethered flight tests (100+ landings)
- Statistical validation and open-source artifact release

### Out of Scope
- Full-scale rocket hardware or boosters
- Untethered / outdoor flight tests
- Atmospheric reentry phases (hypersonic/supersonic regimes)
- Deployment on crewed or commercial systems
- Military applications

### Known Limitations
- EDF dynamics differ from full-scale rockets (higher thrust-to-weight, no fuel mass depletion, different Reynolds number)
- Consumer-grade sensors (BNO085 IMU: ~3.5 deg dynamic error) vs. aerospace-grade
- Indoor testing caps wind disturbance at ~10 m/s
- Embedded compute (Jetson Nano) representative of small spacecraft only
- Results demonstrate feasibility for small unmanned vehicles; extrapolation to larger vehicles requires additional validation

---

## Expected Deliverables

- Isaac Sim environment and controller source code (GitHub)
- EDF drone testbed design: bill of materials, CAD files, wiring documentation
- Dataset of 100+ hardware flight logs with full state vectors
- Statistical benchmarks comparing convex guidance and PID
- Graduate thesis documenting methodology, results, and analysis

---

## Key References

- Açıkmeşe & Ploen (2007). *Convex Programming Approach to Powered Descent Guidance for Mars Landing*. JGCD
- Mahony, Hamel & Pflimlin (2008). *Nonlinear Complementary Filters on the Special Orthogonal Group*. IEEE TAC

---

## License

This project is for academic research purposes. Dual-use considerations under ITAR/EAR apply. Not intended for military use. See the research proposal for full ethical considerations and compliance details.
