# Transformer-RL Retro-Propulsion

**Simulation-to-Hardware Validation of Gated Transformer-XL PPO for Disturbance-Resistant Retro-Propulsive Landings**

> Bridging the sim-to-real gap for attention-enhanced reinforcement learning in thrust-vectoring control systems.

---

## Overview

This research project investigates the **simulation-to-hardware transfer** of a **Gated Transformer-XL (GTrXL) enhanced Proximal Policy Optimization (PPO)** algorithm for thrust-vectoring control (TVC) in disturbance-resistant retro-propulsive landings. Motivated by the growing need for rocket booster recovery in reusable launch vehicles (e.g., SpaceX Falcon 9, Starship, Blue Origin New Glenn), the project addresses fundamental limitations in traditional controllers:

- **PID controllers** struggle with nonlinear dynamics, parameter variations, and external disturbances.
- **Sequential Convex Programming (SCP)** can fail when large perturbations push solutions outside trust regions, leading to infeasibility.
- **Vanilla PPO** handles uncertainty well but suffers from short-term memory constraints, limiting performance on long-horizon trajectories like entry, descent, and landing (EDL).

**GTrXL-PPO** integrates gated attention mechanisms into the PPO framework, enabling the agent to leverage long-term temporal context for improved decision-making across extended trajectories. Prior work (Federici et al., 2024; Carradori et al., 2025) has demonstrated GTrXL-PPO's effectiveness in simulation, achieving up to 98.7% landing success under uncertainties. However, **no prior study has validated transformer-augmented PPO on physical hardware for TVC landing systems** -- this project fills that critical gap.

### Key Goals

- Train and validate GTrXL-PPO in high-fidelity simulation (NVIDIA Isaac Sim)
- Transfer learned policies to a physical Electric Ducted Fan (EDF) drone testbed
- Quantitatively compare GTrXL-PPO against baseline controllers (vanilla PPO, PID, SCP)
- Characterize robustness under realistic disturbances (wind, sensor noise, CoM shifts, varying initial conditions)
- Advance the technology from **TRL 3** (analytical proof-of-concept) to **TRL 5** (validated in relevant environment)

---

## Research Questions

| # | Question | Key Metrics |
|---|----------|-------------|
| **RQ1** | What is the fidelity of simulation-to-hardware transfer for GTrXL-PPO on the EDF testbed? | Landing dispersion (CEP < 0.1 m), jerk (< 10 m/s³), touchdown velocity (< 0.5 m/s), success rate (> 99%, n=100), controller latency (< 50 ms) |
| **RQ2** | How robust is GTrXL-PPO to disturbances (wind, CoM shift, sensor noise, varying ICs) in trajectory-following and landing tasks? | RMSE deviation (< 0.1 m), recovery time, control effort (below saturation), robustness margin (max disturbance before failure) |
| **RQ3** | How does GTrXL-PPO compare to baselines (PPO, PID, SCP) in touchdown accuracy, trajectory efficiency, and safety margins? | Touchdown accuracy, delta-V magnitude, simulated fuel remaining (> 20%), success rate across controllers |

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

The project is organized into the following major modules, each addressing a distinct aspect of the research pipeline:

### 1. Simulation Environment (`simulation/`)

The high-fidelity simulation backbone built on **NVIDIA Isaac Sim**.

- **6-DOF Rigid Body Dynamics**: Full rotational and translational dynamics with realistic mass properties
- **Dynamic Center of Mass Model**: Simulates CoM variation due to fuel consumption and payload shifts
- **Disturbance Injection Framework**: Configurable wind field models, Gaussian sensor noise injection, CoM perturbation profiles
- **Landing Terrain**: Simulated landing pad with ground contact physics
- **Sensor Simulation**: Emulated IMU, optical flow, and barometric sensor outputs matching hardware specifications
- **Data Logging**: Automated state vector, reward, and metric collection per episode via Isaac Sim API

### 2. RL Training Pipeline (`training/`)

Training infrastructure for all controller variants using reinforcement learning.

- **GTrXL-PPO Agent**: Custom policy and value networks using Gated Transformer-XL architecture (Parisotto et al., 2020) with PPO optimization (Schulman et al., 2017)
  - Segment-level recurrence for long-term memory
  - Relative positional encoding
  - Gating mechanisms for gradient stability
- **Vanilla PPO Baseline**: Standard PPO without transformer memory (MLP-based policy)
- **Reward Shaping**: Custom reward function encoding landing precision, fuel efficiency, jerk minimization, and safety constraints
- **Hyperparameter Tuning**: Automated search via Ray Tune for RL agents; Ziegler-Nichols method for PID tuning
- **Meta-RL Training**: Training over distributions of scenarios (varied initial conditions, disturbance profiles) for generalization

### 3. Baseline Controllers (`baselines/`)

Traditional and classical control baselines for comparative evaluation.

- **PID Controller**: Ziegler-Nichols tuned proportional-integral-derivative controller for attitude and position control
- **Sequential Convex Programming (SCP)**: Optimization-based powered descent guidance using convex relaxation of nonlinear constraints (Acikme et al., 2007)
- **Vanilla PPO**: Standard PPO agent without transformer augmentation

### 4. Hardware Platform (`hardware/`)

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
  - Real-time inference target: < 50 ms latency
- **Frame**: Carbon fibre rods with 3D-printed joints and mounts
- **Bill of Materials**: Estimated total < $1,000 USD

### 5. Hardware-in-the-Loop (HIL) Integration (`hil/`)

Bridging simulation and hardware before physical flight.

- **MATLAB Simulink Integration**: Real-time HIL pipeline connecting Isaac Sim dynamics to physical hardware I/O
- **Synthetic Sensor Feed**: Simulink feeds synthetic sensor data to the Jetson Nano running the trained policy
- **Latency Profiling**: End-to-end measurement of sensor-input to control-output delay
- **Transfer Fidelity Assessment**: Pearson correlation (target r > 0.9) between simulation and hardware metrics
- **Trial Volume**: ~500 HIL trials per controller variant with controlled disturbance injection

### 6. Flight Test Framework (`flight_tests/`)

Controlled tethered flight validation of the final system.

- **Test Environment**: Indoor controlled space (~10 x 10 m), tethered flights for safety
- **Test Protocol**: Autonomous descent from 5-10 m altitude with precision landing on marked pad
- **Disturbance Hardware**: External fans (wind injection), variable water payloads (CoM shift and fuel slosh emulation), added weights
- **Data Collection**: Onboard sensor logs at 100 Hz, optical flow ground-truth tracking via ground markers
- **Trial Volume**: 50-100+ tethered flights for GTrXL-PPO controller
- **Safety Systems**: Manual kill switch, software-defined flight envelope limits, tether constraint, pre-flight checklists

### 7. Evaluation & Analysis (`evaluation/`)

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
- **Visualization**: Trajectory plots, landing scatter maps, metric comparison tables, training curves (via Matplotlib)

### 8. Deployment & Artifacts (`artifacts/`)

Open-source deliverables and reproducibility assets.

- **Trained Policies**: Verifiable GTrXL-PPO weights and checkpoints (Hugging Face)
- **Source Code**: Isaac Sim environment, training scripts, baseline implementations (GitHub)
- **Hardware Documentation**: Full bill of materials, CAD files for 3D-printed components, wiring diagrams
- **Datasets**: 100+ flight test logs with full state vectors for community benchmarking
- **Reproducibility**: Containerized training environment, configuration files, random seeds

---

## Methodology

The research follows a **design science** framework with three sequential experimental phases:

```
Phase 1: Simulation          Phase 2: HIL Testing          Phase 3: Flight Tests
┌─────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│ Isaac Sim 6-DOF     │     │ MATLAB Simulink +    │     │ Tethered EDF drone   │
│ environment setup    │────>│ Jetson Nano HIL      │────>│ indoor flight tests   │
│                     │     │                      │     │                      │
│ - Train GTrXL-PPO   │     │ - Transfer fidelity  │     │ - 50-100+ landings   │
│ - Train PPO baseline│     │ - Latency profiling  │     │ - Disturbance inject │
│ - Tune PID (Z-N)    │     │ - 500 trials/variant │     │ - Ground truth via   │
│ - Implement SCP     │     │ - Correlation r>0.9  │     │   optical flow       │
│ - Disturbance models│     │                      │     │                      │
└─────────────────────┘     └──────────────────────┘     └──────────────────────┘
```

### Phase 1 -- Simulation Training & Evaluation
- High-fidelity 6-DOF simulation in NVIDIA Isaac Sim with disturbance models
- Custom GTrXL-PPO policy and value function training with meta-RL over scenario distributions
- Baseline controller implementation and tuning
- Initial metric evaluation and statistical comparison across all controllers

### Phase 2 -- Hardware-in-the-Loop (HIL)
- Deploy trained policies on Jetson Nano embedded compute
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
| **GTrXL-PPO** | Gated Transformer-XL augmented PPO with long-term memory | Ray Tune (automated) |
| **Vanilla PPO** | Standard PPO without transformer memory (MLP backbone) | Ray Tune (automated) |
| **PID** | Classical proportional-integral-derivative controller | Ziegler-Nichols |
| **SCP** | Sequential convex programming optimization-based guidance | Convex solver config |

All controllers are evaluated on identical scenarios with statistical comparison via t-tests/ANOVA (alpha = 0.05).

---

## Timeline

| Phase | Target Date |
|-------|-------------|
| Literature review & research questions | Jan 2026 |
| Research design finalization | Feb 2026 |
| Isaac Sim environment construction | Mar 2026 |
| Training & simulation experiments | Mar - Jul 2026 |
| Hardware build & HIL integration | Jul - Nov 2026 |
| Flight test data collection | Nov 2026 |
| Data analysis | Dec 2026 |
| Draft thesis | Jan 2027 |
| Defense | Mar 2027 |
| Final submission | Apr 2027 |

---

## Scope & Limitations

### In Scope
- GTrXL-PPO, PPO, PID, and SCP for 6-DOF landing control
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

- Verifiable GTrXL-PPO trained policies and weights (Hugging Face)
- Isaac Sim environment and training source code (GitHub)
- EDF drone testbed design: bill of materials, CAD files, wiring documentation
- Dataset of 100+ hardware flight logs with full state vectors
- Statistical benchmarks comparing GTrXL-PPO vs. PPO, PID, and SCP
- Graduate thesis documenting methodology, results, and analysis

---

## Key References

- Schulman et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347
- Dai et al. (2019). *Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context*. ACL 2019
- Parisotto et al. (2020). *Stabilizing Transformers for Reinforcement Learning*. ICML 2020
- Federici et al. (2024). *Meta-Reinforcement Learning with Transformer for Lunar Landing*. AIAA SciTech 2024
- Carradori et al. (2025). *Transformer-Based Robust Feedback Guidance for Atmospheric Powered Landing*. AIAA SciTech 2025
- Acikme & Ploen (2007). *Convex Programming Approach to Powered Descent Guidance for Mars Landing*. JGCD
- Hwangbo et al. (2017). *Control of a Quadrotor with Reinforcement Learning*. IEEE RA-L
- Zhang & Li (2020). *Testing and Verification of Neural-Network-Based Safety-Critical Control Software*. IST

---

## License

This project is for academic research purposes. Dual-use considerations under ITAR/EAR apply. Not intended for military use. See the research proposal for full ethical considerations and compliance details.
