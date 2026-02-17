# Simulation Master Plan — Phased Training Strategy

> **Scope**: End-to-end simulation strategy for training, validating, and deploying GTrXL-PPO for EDF drone retro-propulsive landing.
> Three phases: Python-only baseline, conditional Isaac integration, evaluation and pivots.
> Integrates with [vehicle.md](Vehicle%20Dynamics/vehicle.md) and [env.md](Enviornment/env.md).

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Isaac Sim Value Assessment](#2-isaac-sim-value-assessment)
3. [Phase 1 — Baseline Python-Only Training](#3-phase-1--baseline-python-only-training)
4. [Phase 2 — Isaac Integration for Validation and Parallel Boost](#4-phase-2--isaac-integration-for-validation-and-parallel-boost)
5. [Phase 3 — Evaluation and Pivots](#5-phase-3--evaluation-and-pivots)
6. [Decision Framework](#6-decision-framework)
7. [Total Effort and Dependencies](#7-total-effort-and-dependencies)
8. [Risk Register](#8-risk-register)

---

## 1. Executive Summary

### 1.1 Core Question

Should we train in Python only, Isaac for viz? Or full Isaac training?

**Answer**: Hybrid. Fundamentals favor Python for speed and precision in early phases, Isaac for final validation, visualization, and HIL. The approach mirrors SpaceX-style iteration: start lean, add complexity only when data demands it.

### 1.2 Strategy

| Phase | Duration | Platform | Purpose |
|---|---|---|---|
| **Phase 1** | 1–2 weeks | Python (NumPy/JAX) | Fast baseline. Maximize iterations. Prove the agent can land. |
| **Phase 2** | 2–3 weeks | Isaac Sim (conditional) | GPU-parallel scaling, contact physics, sensor fidelity. Only if Phase 1 hits bottlenecks. |
| **Phase 3** | 1 week | Both | Evaluate, compare, decide what to keep. Prepare for HIL. |
| **Total** | **4–6 weeks** | | |

### 1.3 Guiding Principle

**Train in Python, fine-tune/validate in Isaac to close sim-to-real gaps (e.g., dynamics residuals).** Delete rigid adherence to full Isaac training if Python achieves >95% success rate faster. Adapt — don't commit to overhead without evidence it pays off.

**Multi-controller comparison**: All training uses 4 controller variants (PPO-MLP, GTrXL-PPO, PID, SCP) sharing the same `EDFLandingEnv`, observation/action spaces, and evaluation protocol. See [training.md](Training%20Plan/training.md) for the authoritative training pipeline specification.

---

## 2. Isaac Sim Value Assessment

### 2.1 GPU-Parallel Vectorized Environments

**What Isaac offers**: OmniIsaacGymEnvs (OIGE) runs 1024–4096 environments in parallel on GPU, accelerating sample collection by 50–100x compared to serial Python loops (~1–10 envs/sec on CPU). For PPO, which needs diverse samples, this is a pure compute win — no multi-body overhead for a single rigid body.

**Fundamentals**: PPO sample efficiency scales with batch diversity. Serial Python loops yield ~10k steps/hour; Isaac parallel can yield ~1M steps/hour. For 1e7 total steps (Phase 1 convergence target), that's 10 hours in Python vs. ~10 minutes in Isaac.

**Workstation spec (Mankato, Feb 2026)**:

| Component | Spec | Training Relevance |
|---|---|---|
| GPU | NVIDIA RTX 5070 — 12 GB GDDR7, 672 GB/s, 6144 CUDA cores (Blackwell) | JAX `vmap`, Isaac OIGE, policy inference |
| CPU | AMD Ryzen 9 9900X — 12C/24T, 5.6 GHz boost (Zen 5) | SB3/Ray CPU-parallel envs (12–24 workers) |
| RAM | 32 GB DDR5 6000 MT/s | Comfortable for any training config; no swap risk |
| Storage | M.2 NVMe SSD | Fast checkpoint I/O, log writes |

At 4096 envs with full state tensors (14 scalars × 4096 × float32), pure state memory is negligible (~230 KB), but PhysX scene overhead, USD stage, and renderer context add up. The RTX 5070's 12 GB GDDR7 has the same capacity ceiling as the previous-gen 3060 but ~1.8× the memory bandwidth (672 vs. 360 GB/s) and ~1.7× the CUDA core count, meaning higher throughput per env at equivalent `num_envs`.

| `num_envs` | Est. VRAM (PhysX + OIGE) | RTX 5070 (12 GB GDDR7) | Notes |
|---|---|---|---|
| 256 | ~2–3 GB | Comfortable | Room for renderer + debug |
| 512 | ~4–6 GB | Safe (headless) | Recommended Isaac ceiling |
| 1024 | ~5–8 GB | Feasible headless, monitor usage | GDDR7 bandwidth helps tensor throughput |
| 2048 | ~8–12 GB | Tight — profile before committing | Risk of fragmentation-induced OOM |
| 4096 | ~12–18 GB | OOM likely | Delete this ambition on 12 GB |

**Decision**:
- **Phase 1**: Delete Isaac parallelism. Use JAX `vmap` (RTX 5070 Blackwell compute is strong) or Ray/SB3 vec envs (Ryzen 9 9900X 12C/24T handles 12–24 workers easily). Good enough for baseline convergence.
- **Phase 2**: If sample bottleneck confirmed, enable Isaac at `num_envs=512` (safe headless on RTX 5070). Scale to 1024 headless if profiling shows headroom. Delete 4096 ambition — 12 GB VRAM is the hard constraint regardless of generation.
- **Fallback**: JAX-vectorized Python on GPU (`jax.vmap` over `VehicleDynamics.step()`) — lighter than Isaac, no PhysX overhead. RTX 5070's Blackwell compute should push JAX throughput to ~1–4M steps/sec at 256 envs. Benchmark both.

### 2.2 Built-in Physics Fidelity

**What Isaac offers**: PhysX handles ground contact (friction, normal force, bounce on landing pad), collision detection, and basic rigid-body integration. Python's custom contact (spring-damper hack in `derivs()`) is approximate, risking sim-to-real gaps — e.g., 0.2–0.5 m touchdown error from incorrect contact stiffness.

**What Isaac doesn't offer for this project**:
- No unique aero for a single rigid body (no CFD coupling, no vortex lattice — those require external solvers)
- PhysX uses implicit Euler integration, which is less accurate than RK4 for smooth trajectories. For precision metrics (jerk < 10 m/s^3, CEP < 0.1 m), RK4 may produce tighter residuals
- Subsonic EDF (~60 m/s) has no hypersonic shocks — PhysX adds no value over analytical drag

**Quantitative question**: Does PhysX's implicit Euler beat RK4 accuracy for jerk and position metrics? Expected answer: No — RK4 (4th-order) has $O(dt^5)$ local truncation vs. implicit Euler's $O(dt^2)$. PhysX wins on contact events, loses on trajectory smoothness.

| Physics Aspect | Python (RK4) | Isaac (PhysX) | Winner |
|---|---|---|---|
| Trajectory integration accuracy | $O(dt^5)$ per step | $O(dt^2)$ per step | **Python** |
| Contact/ground physics | Spring-damper hack | Built-in friction/bounce | **Isaac** |
| Jerk computation fidelity | Exact from $\dot{a}$ via RK4 stages | Numerical diff of PhysX state | **Python** |
| Wind injection | Custom Dryden filter (accurate) | Force callback (same accuracy) | **Tie** |
| Ground effect | Analytical $1 + 0.5(r/h)^2$ | Same (via callback) | **Tie** |
| Setup complexity | None (pure Python) | USD scene + OIGE boilerplate | **Python** |

**Decision**:
- **Phase 1**: Python RK4 for all dynamics. Better accuracy for precision metrics.
- **Phase 2**: Isaac for contact-phase validation only (last 0.5 m of descent). If PhysX contact model improves touchdown velocity prediction by >0.1 m/s vs. spring-damper, keep. Otherwise delete and use Python contact.
- **Deletion criterion**: If Python spring-damper matches PhysX touchdown within 0.1 m/s over 100 episodes (r > 0.95), delete PhysX contact.

### 2.3 Sensor Simulation Realism

**What Isaac offers**: Built-in IMU emulation (bias, drift, scale factor), optical flow (camera raytracing for ground truth with lighting/occlusion effects), barometer. These tie to the renderer for realistic noise profiles affected by shadows, reflections, and camera angle.

**What Python offers**: Gaussian additive noise (per [vehicle.md §11.3](Vehicle%20Dynamics/vehicle.md)). Simple, fast, and sufficient if the Gaussian $\sigma$ values match real sensor noise profiles.

**Impact analysis**:

| Sensor | Python Gaussian $\sigma$ | Isaac Emulation | Impact on Success Rate |
|---|---|---|---|
| IMU accel | 0.1 m/s^2 | Bias drift + scale + noise | Likely <2% difference at test altitudes (<10 m) |
| IMU gyro | 0.01 rad/s | Bias drift + temperature | <1% — short episodes (10 s) limit drift accumulation |
| Optical flow | 0.1 m position | Raytraced with occlusion | Could matter if shadows on landing pad confuse flow — but indoor tests have controlled lighting |
| Barometer | 0.5 m | Pressure-coupled with physics | <1% — altitude from optical flow dominates |

**Decision**:
- **Phase 1**: Gaussian noise is sufficient. Delete Isaac sensor emulation overhead.
- **Phase 2**: Enable Isaac sensor simulation for robustness ablation (RQ2). If success rate delta > 2% between Gaussian and Isaac sensors, keep. Otherwise delete.
- **Key insight**: The proposal's Gaussian $\sigma$ values (from [vehicle.md §11.3](Vehicle%20Dynamics/vehicle.md)) are calibrated to the BNO085 IMU datasheet. For indoor controlled tests, these are accurate enough.

### 2.4 HIL Pipeline Bridge

**What Isaac offers**: MATLAB Simulink integration (per [README.md](../../README.md)) for seamless synthetic sensor feeds to the Jetson Nano. This is the **strongest value proposition** for Isaac in this project. HIL testing (~500 trials, RQ1/RQ2) requires real-time-capable sensor feeds — Isaac's renderer can produce camera streams and IMU packets directly.

**What Python requires**: Custom wrappers (socket-based UDP/ROS bridge) to pipe synthetic sensor data to Jetson. Adds 1–2 weeks dev time and introduces transfer risk (timing jitter, format mismatches).

**Decision**:
- **Phase 2**: Isaac for HIL prep is **non-negotiable** if TRL 5 is the target. Even if Python trains better, Isaac provides the bridge.
- **Implementation**: Headless Isaac (no rendering overhead) as a sensor data generator. The trained Python policy runs on Jetson; Isaac provides the sensor environment.
- **Effort**: 1–2 weeks for OIGE task setup + Simulink bridge. Budget this into Phase 2.

### 2.5 Deletions for Isaac — What's Not Worth It

| Isaac Feature | Reason to Delete | When to Reconsider |
|---|---|---|
| Full Isaac training (all phases) | Setup overhead 1–2 weeks. Consumes VRAM budget on 12 GB card. No accuracy win for trajectory integration. | Only if JAX vectorization OOMs or Python throughput < 1e5 steps/hour |
| Isaac aero models | No built-in aero for single RB. Custom callbacks are identical to Python. | Never — delete permanently |
| Isaac domain randomization API | Python's per-episode `reset()` with `np.random` is simpler and equally capable | If >100 randomized parameters need GPU-parallel sampling |
| Isaac terrain/scene complexity | Flat landing pad. No obstacle avoidance. Scene adds USD overhead. | If future work adds terrain variety |
| Real-time rendering during training | 50+ ms latency per frame. Destroys throughput. | Debug only (enable for 10 episodes, then disable) |

---

## 3. Phase 1 — Baseline Python-Only Training

### 3.1 Objective

Prove the RL agent can land. Maximize iterations per wall-clock hour. Establish baseline metrics for comparison.

### 3.2 Why Python First

**Fundamentals**: NumPy/JAX with RK4 ([vehicle.md §7](Vehicle%20Dynamics/vehicle.md)) runs ~10k steps/sec on CPU (single core), scaling well on the Ryzen 9 9900X (12C/24T). JAX `vmap` on the RTX 5070 (Blackwell, 6144 CUDA cores) pushes throughput to ~1–4M steps/sec. No viz lag, no OIGE boilerplate, no VRAM contention.

**Parallel options** (on Ryzen 9 9900X + RTX 5070):

| Method | Envs | Steps/sec (est.) | Hardware Used |
|---|---|---|---|
| Serial NumPy (1 env) | 1 | ~10k | 1 CPU core |
| SB3 `SubprocVecEnv` (CPU) | 12–24 | ~100–500k | Ryzen 9 9900X 12C/24T |
| Ray RLlib (CPU workers) | 12–48 | ~200–800k | Ryzen 9 9900X + Ray cluster-ready |
| JAX `vmap` (GPU) | 64–512 | ~1–4M | RTX 5070 (Blackwell, 12 GB GDDR7) |

**Delete Isaac if Phase 1 tests show < 10% metric difference vs. Phase 2 Isaac**: Run 100 episodes in both platforms, compute landing CEP correlation. If Pearson r > 0.9, Python is sufficient for training; Isaac is only needed for viz and HIL.

### 3.3 Setup

**Gym environment**: Wrap `VehicleDynamics` + `EnvironmentModel` ([vehicle.md §8](Vehicle%20Dynamics/vehicle.md), [env.md §2](Enviornment/env.md)) as a standard Gymnasium env:

```python
class EDFLandingEnv(gymnasium.Env):
    """Python-only 6-DOF EDF landing environment for RL training."""

    def __init__(self, config: dict):
        self.vehicle = VehicleDynamics(config['vehicle'], EnvironmentModel(config['environment']))
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(20,))  # state + wind est
        self.action_space = spaces.Box(-1, 1, shape=(5,))  # T_cmd + 4 fin deltas (normalized)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        ic = self._sample_initial_conditions()
        self.vehicle.reset(ic, seed=seed)
        return self._get_obs(), {}

    def step(self, action):
        u = self._scale_action(action)  # denormalize to physical units
        for _ in range(self.substeps):   # e.g., 10 physics steps per policy step
            self.vehicle.step(u)
        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._check_landing() or self._check_crash()
        truncated = self.vehicle.time > self.max_time
        return obs, reward, terminated, truncated, {}
```

**Domain randomization**: Per-episode via `EnvironmentModel.reset()` ([env.md §6](Enviornment/env.md)):
- Wind: mean ±10 m/s, Dryden turbulence, 10% gust probability
- Atmosphere: $T_{base} \pm 10$ K, $P_{base} \pm 2000$ Pa (→ ~11% $\rho$ swing)
- Vehicle: CoM ±10% mass perturbation on `payload_variable` primitive

**Reward function** (multi-objective, potential-based shaping per Ng et al., 1999):

> **Authoritative specification**: See [training.md §5](Training%20Plan/training.md) for the complete reward function with all weights, terminal rewards, and tuning protocol.

$$
r_t = r_{alive} + r_{shape} + r_{orient} + r_{jerk} + r_{fuel} + r_{action} + \mathbb{1}_{terminal} \cdot r_{terminal}
$$

| Component | Weight | Type | Purpose |
|---|---|---|---|
| Alive bonus | 0.1 (fixed) | Step | Survive |
| Distance + velocity shaping | $c_d = 1.0$, $c_v = 0.2$ | Step (potential-based) | Approach target, slow down |
| Orientation | $w_\theta = 0.5$ | Step | Stay upright |
| Jerk | $w_j = 0.05$ | Step | Smooth control (RQ1: < 10 m/s³) |
| Fuel | $w_f = 0.01$ | Step | $\Delta V$ minimization (RQ3) |
| Action smoothness | $w_a = 0.02$ | Step | Actuator smoothing |
| Landing success | $R_{land} = 100$ | Terminal | Main objective |
| Precision bonus | $R_{prec} = 50$ | Terminal | Gaussian CEP bonus (σ = 0.1 m) |
| Soft touchdown | $R_{soft} = 20$ | Terminal | Reward softer landings |
| Crash penalty | $R_{crash} = 100$ | Terminal | Safety constraint |
| Out-of-bounds | $R_{oob} = 50$ | Terminal | Safety constraint |

### 3.4 Models

> **Authoritative specification**: See [training.md §7](Training%20Plan/training.md) for complete model architectures and [training.md §9](Training%20Plan/training.md) for hyperparameter search configuration.

**Four controller variants** (fair comparison — all share the same `EDFLandingEnv`, observation/action spaces, and evaluation protocol):

| Controller | Type | Architecture | Training Budget | Research Question |
|---|---|---|---|---|
| **PPO-MLP** (RL baseline) | RL (reactive) | 2×256 MLP, ~275K params | 10M steps (SB3) | RQ3 baseline |
| **GTrXL-PPO** (RL temporal) | RL (temporal) | GTrXL ($d_{model}$=128, L=2, H=4, seg=64), ~370K params | 15M steps (RLlib) | RQ1, RQ2 |
| **PID** (classical baseline) | Classical | Cascaded position→attitude→fins, ~12 gains | ~5 hrs (Ziegler-Nichols + grid search) | RQ3 baseline |
| **SCP** (optimization baseline) | Trajectory opt. | CVXPY SOCP, $N$=50 nodes, open-loop + MPC modes | ~2 days (config + trust region tuning) | RQ3 $\Delta V$ benchmark |

**Why four controllers**: The thesis contribution is comparative evaluation. PID and SCP quantify *how much* RL adds. Without baselines, we can't claim RL superiority.

**Hyperparameter search** (RL controllers only, via Ray Tune ASHA):

| Hyperparameter | PPO-MLP Range | GTrXL-PPO Additional | Priority |
|---|---|---|---|
| Learning rate | $[5 \times 10^{-5}, 3 \times 10^{-4}]$ (log) | Same | High |
| Batch size ($T \times N_{env}$) | $\{8192, 16384, 32768, 65536\}$ | Same | High |
| GAE $\lambda$ | $[0.9, 0.99]$ | Same | Medium |
| Entropy coefficient | $[0.001, 0.05]$ (log) | Same | Medium |
| $d_{model}$ | — | $\{64, 128, 256\}$ | High |
| $n_{layers}$ | — | $\{1, 2, 3\}$ | High |
| Segment length | — | $\{32, 64, 128\}$ | Medium |

Search budget: 50 trials (PPO-MLP), 30 trials (GTrXL-PPO), 3M steps per trial, ASHA early stopping.

### 3.5 Milestones

> **Evaluation targets**: See [training.md §11](Training%20Plan/training.md) for complete evaluation protocol (400 episodes per controller, 4 conditions, statistical analysis).

| Milestone | Criterion | Est. Steps | Action if Not Met |
|---|---|---|---|
| M1: Agent learns to hover | Mean episode length > 5 s (max 15 s per training.md §2.2) | ~1e6 | Debug reward shaping |
| M2: Agent lands consistently | Success rate > 50% (10 m start) | ~3e6 | Tune hyperparameters |
| M3: Agent lands precisely | CEP < 0.5 m, success > 90% | ~7e6 | Add curriculum (training.md §10), increase DR |
| M4: Baseline converged | CEP < 0.1 m, success > 99%, jerk < 10 m/s³, latency < 50 ms | ~1e7 (MLP) / ~1.5e7 (GTrXL) | Proceed to Phase 2 |
| M5: Multi-controller comparison | All 4 controllers evaluated (PPO-MLP, GTrXL-PPO, PID, SCP) | After M4 | Paired t-tests, ANOVA per training.md §11.3 |

### 3.6 Visualization

**Post-hoc only**. Delete real-time visualization during training.
- Matplotlib trajectory plots (top-down, side view)
- Landing scatter plots (CEP circle overlay)
- Reward curves (TensorBoard/Weights & Biases)
- Wind realization vs. trajectory overlay (for disturbance analysis)

### 3.7 Deletion Criteria

> **Aligned with [training.md §7](Training%20Plan/training.md) deletion criteria and [training.md §12.3](Training%20Plan/training.md) ablation studies.**

| Feature | Delete If | Replace With |
|---|---|---|
| GTrXL | Doesn't improve robustness (RQ2 metrics) by >10% over MLP across disturbance envelope; 35% parameter overhead not justified | Vanilla PPO-MLP for all training |
| PPO-MLP | PPO-MLP achieves >95% success with CEP <0.15 m and jerk <10 m/s³ (per training.md §7.1.3) | Delete GTrXL — temporal memory not needed |
| Curriculum learning | Vanilla uniform DR achieves M4 within 7×10⁶ steps (per training.md §10.5) | Uniform randomization only |
| JAX vectorization | NumPy + SB3 vec envs reaches 1e7 steps in < 48 hours | Keep NumPy (simpler codebase) |
| Dryden turbulence | Ablation shows white noise achieves same CEP (±0.01 m) and success rate (±1%) | Gaussian white noise |
| Actuator delay DR | Hardware latency profiling shows <10 ms end-to-end (per training.md §6.2.2) | Remove latency augmentation |

### 3.8 Deliverables

- [ ] `EDFLandingEnv` Gymnasium wrapper + `RewardFunction` + observation pipeline (Python)
- [ ] PPO-MLP trained checkpoint (>90% success, 10M steps)
- [ ] GTrXL-PPO trained checkpoint (conditional, 15M steps)
- [ ] PID controller with tuned gains (Ziegler-Nichols + grid search)
- [ ] SCP controller configured (CVXPY, open-loop + MPC modes)
- [ ] Hyperparameter sweep results (Ray Tune ASHA, 50 MLP trials + 30 GTrXL trials)
- [ ] Multi-controller comparison table (per [training.md §12](Training%20Plan/training.md): all 4 controllers × 4 conditions × 100 episodes)
- [ ] Statistical analysis: paired t-tests, ANOVA, effect sizes (per training.md §11.3)
- [ ] Ablation results: DR components, turbulence model, actuator delay, obs latency, GTrXL vs. MLP
- [ ] ONNX export pipeline validation (per training.md §14)
- [ ] Decision: proceed to Phase 2 (if bottlenecked) or skip to Phase 3

---

## 4. Phase 2 — Isaac Integration for Validation and Parallel Boost

### 4.1 Objective

Close sim-to-real gaps that Python can't address. Leverage GPU parallelism for meta-RL over disturbance distributions. Prepare the HIL pipeline.

### 4.2 Entry Criteria — Only Proceed If

Phase 2 is **conditional**. Enter only if one or more of:

| Trigger | Evidence | Phase 2 Response |
|---|---|---|
| Sample bottleneck | Python throughput < 1e5 steps/hour after JAX optimization | Isaac parallel envs (num_envs=256–512) |
| Contact fidelity gap | Touchdown velocity mismatch > 0.1 m/s between Python spring-damper and analytical prediction | Isaac PhysX for terminal phase |
| Sensor fidelity gap | Gaussian noise ablation shows > 2% success rate difference vs. structured noise | Isaac sensor emulation |
| HIL readiness | Timeline demands HIL prep (Jul 2026 per README.md) | Isaac as sensor bridge to Jetson |

If **none** of these triggers fire, skip to Phase 3. Use Python for all training, Isaac for viz only.

### 4.3 Setup

**Port to OmniIsaacGymEnvs (OIGE)**:
1. Create USD asset for EDF drone (single rigid body, visual mesh, collision mesh)
2. Define OIGE task class with `pre_physics_step()` callbacks for custom forces (thrust lag, fins, wind via Dryden)
3. Configure `num_envs` (start at 512, scale to 1024 headless if RTX 5070 VRAM allows)
4. Headless mode for training; enable renderer for debug batches only

**Force callback pattern** (same physics as Python, different execution context):

```python
class EDFLandingTask(RLTask):
    """OIGE task for EDF landing. Forces computed identically to Python env."""

    def pre_physics_step(self, actions):
        # Identical force models as Python, but batched over num_envs
        T_cmd = actions[:, 0]
        fin_deltas = actions[:, 1:5]

        # Wind from Dryden filter (per-env, GPU-batched)
        v_wind = self.wind_model.sample_batched(self.time, self.altitudes)

        # Custom forces (not from PhysX — applied via API)
        F_thrust = self.thrust_model.compute_batched(self.thrust_state, T_cmd, rho=self.rho)
        F_aero = self.aero_model.compute_batched(self.velocities, v_wind, rho=self.rho)
        F_fins = self.fin_model.compute_batched(fin_deltas, self.omega_fan, rho=self.rho)

        self._drones.apply_forces(F_thrust + F_aero + F_fins, is_global=False)
        self._drones.apply_torques(tau_total, is_global=False)
```

**Sensors** (Isaac built-in):
- IMU: Enable bias drift model (PhysX IMU sensor)
- Optical flow: Camera-based with raytracing (if GPU budget allows)
- Barometer: Pressure-coupled to AtmosphereModel

### 4.4 Training Strategy

**Hybrid workflow** — do not retrain from scratch in Isaac:

1. **Initialize from Python weights**: Load Phase 1 PPO-MLP checkpoint into Isaac OIGE training loop
2. **Fine-tune for 1e6 steps**: Adapt to PhysX integration differences (implicit Euler vs. RK4 residuals)
3. **Meta-RL sweep**: Use Isaac parallelism to train over a wider disturbance distribution (1024 envs × varied DR)
4. **Compare**: Log identical metrics as Phase 1. If Isaac fine-tuning improves CEP by > 0.05 m or success by > 5%, keep. Otherwise, revert to Python weights.

**RLlib integration**: Use the `rl_games` library (default for OIGE) or wrap OIGE task for RLlib. Init from Phase 1 checkpoint.

### 4.5 Visualization Role

| Mode | When | Purpose |
|---|---|---|
| Headless (no render) | Training (100% of steps) | Maximize throughput |
| Render 10 episodes | After each 1e5 steps | Debug landing trajectories, spot visual anomalies |
| Full render session | Post-training | Generate demo videos, landing animations for thesis |

**Delete full visualization if render latency > 50 ms per step** (per proposal metrics). This is the threshold where rendering measurably slows training.

### 4.6 HIL Preparation

> **Model export pipeline**: See [training.md §14](Training%20Plan/training.md) for the complete ONNX → TensorRT export specification, GTrXL stateful inference handling, and latency validation protocol.

If Phase 2 is triggered by HIL readiness:

1. **Simulink bridge**: Set up Isaac → Simulink → Jetson Nano pipeline
2. **Synthetic sensor feed**: Isaac generates IMU/optical flow/baro packets at 100 Hz
3. **Policy deployment**: Export Phase 1 Python policy to ONNX (opset 17+) → TensorRT (FP16) on Jetson. GTrXL requires explicit memory state as model input/output (training.md §14.3).
4. **Latency profiling**: Measure end-to-end sensor-to-actuator delay. Budget: sensor read (5 ms) + preprocessing (5 ms) + policy inference (PPO-MLP <5 ms, GTrXL <20 ms) + actuator command (5 ms) + margin (15 ms) = <50 ms total.
5. **Trial volume**: 500 HIL trials per controller variant (GTrXL-PPO, PPO-MLP, PID, SCP)
6. **Export validation**: Compare ONNX outputs vs. PyTorch outputs over 100 episodes (max absolute error < 10⁻⁴ per action dimension, per training.md §14.5)
7. **Transfer fidelity check**: Pearson r > 0.9 between Isaac sensor data and Python state predictions

### 4.7 Deliverables

- [ ] OIGE task class for EDF landing
- [ ] USD rigid body asset (EDF drone)
- [ ] Fine-tuned checkpoint (Isaac, initialized from Python)
- [ ] Comparative metrics: Python vs. Isaac (CEP, success, jerk, $\Delta V$)
- [ ] HIL pipeline (if triggered): Simulink bridge + Jetson deployment
- [ ] Decision: keep Isaac for remaining training, or revert to Python + Isaac viz only

---

## 5. Phase 3 — Evaluation and Pivots

### 5.1 Objective

Compare Python vs. Isaac training outcomes. Make final platform decision for thesis experiments. Prepare for hardware build (Jul 2026).

### 5.2 Metrics Comparison

> **Multi-controller evaluation**: See [training.md §11–12](Training%20Plan/training.md) for the authoritative evaluation protocol. All 4 controllers (PPO-MLP, GTrXL-PPO, PID, SCP) are evaluated on 400 episodes each (100 per condition: nominal, windy, full DR, easy) with paired statistical comparisons.

**Platform comparison** (Python vs. Isaac) — run identical evaluation episodes (n=100, same seeds):

| Metric | Phase 1 (Python) | Phase 2 (Isaac) | Threshold for Isaac Win |
|---|---|---|---|
| Landing CEP | _measured_ | _measured_ | Isaac < Python by > 0.05 m |
| Touchdown velocity | _measured_ | _measured_ | Isaac < Python by > 0.1 m/s |
| Success rate | _measured_ | _measured_ | Isaac > Python by > 5% |
| Jerk (99th pctl) | _measured_ | _measured_ | Isaac < Python by > 2 m/s³ |
| Training wall-clock | _measured_ | _measured_ | Isaac < Python by > 20% |
| $\Delta V$ efficiency | _measured_ | _measured_ | Isaac < Python by > 5% |

**Controller comparison** (per [training.md §12.1](Training%20Plan/training.md)):

| Metric | PPO-MLP | GTrXL-PPO | PID | SCP (OL) | SCP (MPC) |
|---|---|---|---|---|---|
| Success rate | — | — | — | — | — |
| CEP [m] | — | — | — | — | — |
| Touchdown vel [m/s] | — | — | — | — | — |
| Jerk 99th [m/s³] | — | — | — | — | — |
| $\Delta V$ [m/s] | — | — | — | — | — |
| Robustness margin [m/s] | — | — | — | — | — |
| Inference latency [ms] | — | — | — | — | — |

Statistical methods: paired t-tests (Bonferroni-corrected), one-way ANOVA, Cohen's $d$ effect sizes, 95% bootstrap CIs (per training.md §11.3).

### 5.3 Decision Matrix

| Outcome | Action |
|---|---|
| Isaac improves all metrics by > thresholds | Adopt Isaac for remaining training + HIL |
| Isaac improves contact fidelity but slows training 2x | Use Python for training, Isaac for terminal-phase validation + HIL only |
| Isaac adds > 5% noise realism but negligible CEP improvement | Delete Isaac sensor emulation; keep Python Gaussian noise |
| Python and Isaac are equivalent (r > 0.95 on all metrics) | Delete Isaac training entirely. Use Python for all training, Isaac for viz + HIL bridge only |
| Isaac training diverges or OOMs on RTX 5070 (12 GB) | Delete Isaac training. Fall back to JAX-vectorized Python. |

### 5.4 Pivot Criteria — When to Delete Isaac Entirely (Post-Viz)

If single rigid body simplicity makes Isaac redundant — no contact complexity beyond spring-damper, no multi-body interactions, no scene complexity — delete Isaac for everything except:
1. **Visualization**: Thesis figures and demo videos
2. **HIL bridge**: Sensor feed to Jetson (if no custom alternative is built)

Focus remaining effort on **hardware** (EDF testbed, sensor integration, flight tests).

### 5.5 Academic Efficiency Check

Thesis timeline: Mar–Jul 2026 for simulation experiments. If Isaac integration consumes 3+ weeks of the 4-month window without proportional metric improvement, it's academically inefficient. The thesis contribution is **sim-to-real transfer of GTrXL-PPO**, not simulation fidelity per se. A Python-trained policy that transfers well to hardware is a stronger result than an Isaac-trained policy that took twice as long to develop.

### 5.6 Deliverables

- [ ] Comparison report: Python vs. Isaac metrics (table + plots)
- [ ] Multi-controller comparison report: PPO-MLP vs. GTrXL-PPO vs. PID vs. SCP (per [training.md §12](Training%20Plan/training.md))
- [ ] Statistical analysis: paired t-tests, ANOVA, effect sizes, 95% CIs (per training.md §11.3)
- [ ] Final platform decision document
- [ ] Converged policy checkpoints (best of Python or Isaac, all controller variants)
- [ ] ONNX/TensorRT exported models validated against PyTorch (per training.md §14.5)
- [ ] Training logs, ablation results, and robustness characterization for thesis Appendix
- [ ] HIL readiness assessment

---

## 6. Decision Framework

### 6.1 Phase Gate Criteria

```
Phase 1 ──[M4 met?]──> YES ──[bottleneck?]──> YES ──> Phase 2
    │                                    │
    │                                    └──> NO ──> Phase 3 (Python-only)
    │
    └──> NO (after 2 weeks) ──> Debug. Re-examine reward, DR, hyperparams.
                                 Do NOT proceed to Phase 2 if Phase 1 fails.
```

### 6.2 Kill Switches

| Trigger | Action | Rationale |
|---|---|---|
| Phase 1 fails to reach 50% success after 1e7 steps | Stop. Audit dynamics model, reward function, observation space. | Agent fundamentals are broken; Isaac won't fix this. |
| Isaac OOMs at num_envs=128 | Delete Isaac training. Use JAX-vectorized Python. | Hardware limitation — don't fight it. |
| Isaac fine-tuning degrades Phase 1 metrics | Revert to Python checkpoint. Isaac adds no value. | PhysX integration mismatch. |
| Timeline pressure (> 4 weeks spent in Phases 1–2) | Skip Phase 2, go to Phase 3 with Python-only results. | Thesis deadline > sim perfection. |

### 6.3 Feature Retention Heuristic

For every simulation feature (Dryden turbulence, density correction, ground effect, etc.):

$$
\text{Keep if: } \Delta \text{CEP} > 0.01 \text{ m} \quad \text{OR} \quad \Delta \text{success rate} > 2\%
$$

$$
\text{Delete if: neither threshold met after 100-episode ablation}
$$

This is the same heuristic from [env.md §8.5](Enviornment/env.md) and [vehicle.md §10](Vehicle%20Dynamics/vehicle.md), applied globally.

---

## 7. Total Effort and Dependencies

### 7.1 Effort Breakdown

> **Training pipeline details**: See [training.md §8](Training%20Plan/training.md) for per-controller training procedures and [training.md §8.6](Training%20Plan/training.md) for resource budgets.

| Task | Phase | Est. Effort | Dependency |
|---|---|---|---|
| `VehicleDynamics` + `EnvironmentModel` implementation | Pre-Phase 1 | 10–12 days | [vehicle.md](Vehicle%20Dynamics/vehicle.md) + [env.md](Enviornment/env.md) |
| `EDFLandingEnv` + `RewardFunction` + obs pipeline | Phase 1 | 2 days | Vehicle + Env modules |
| PPO-MLP training (10M steps) + Ray Tune sweep | Phase 1 | 3–5 days | Gym env |
| GTrXL-PPO training (15M steps, conditional) | Phase 1 | 2–3 days | Gym env |
| PID tuning (Ziegler-Nichols + grid search) | Phase 1 | ~1 day | Gym env + linearization |
| SCP configuration (CVXPY setup + trust region) | Phase 1 | ~2 days | Gym env |
| Multi-controller evaluation + ablation studies | Phase 1 | 2–3 days | All controllers trained |
| ONNX export + validation (per training.md §14) | Phase 1 | 1–2 days | Trained checkpoints |
| OIGE task + USD asset (conditional) | Phase 2 | 5–7 days | Phase 1 decision |
| Isaac fine-tuning + meta-RL (conditional) | Phase 2 | 3–5 days | OIGE task |
| HIL bridge setup (conditional) | Phase 2 | 5–7 days | OIGE task |
| Evaluation + comparison | Phase 3 | 3–5 days | Phases 1–2 |
| **Total (if all phases)** | | **~6–8 weeks** | |
| **Total (Phase 1 only + eval)** | | **~4–5 weeks** | |

### 7.2 Critical Path

```
vehicle.md + env.md implementation (10-12 days)
    │
    ├──> Gym env wrapper + reward + obs pipeline (2 days)
    │       │
    │       ├──> PPO-MLP training (3-5 days) ─────────────────────┐
    │       ├──> [parallel] PID tuning (1 day) ───────────────────┤
    │       ├──> [parallel] SCP config (2 days) ──────────────────┤
    │       └──> [after MLP] GTrXL-PPO training (2-3 days) ──────┤
    │                                                              │
    │       Multi-controller eval + ablation (2-3 days) ──────────┤
    │       ONNX export + validation (1-2 days) ──────────────────┤
    │                                                              │
    │                                                  Phase 1 decision
    │                                                        │
    │       ┌──────────────────── [conditional] ────────────┘
    │       │
    │       └──> OIGE task (5-7 days) ──> Fine-tune (3-5 days) ──> Evaluation (3-5 days)
    │
    └──> [parallel] Config files, test suites
```

### 7.3 Hardware Dependencies

| Resource | Spec / Requirement | Fallback |
|---|---|---|
| GPU | RTX 5070 (12 GB GDDR7, 672 GB/s, 6144 CUDA cores) | CPU-only with SB3 vec envs on Ryzen 9 9900X (slower but functional) |
| CPU | Ryzen 9 9900X (12C/24T, 5.6 GHz boost, Zen 5) | Fewer SB3/Ray workers; JAX GPU compensates |
| RAM | 32 GB DDR5 6000 MT/s | Not a constraint — any training config fits |
| Storage | M.2 NVMe SSD | Fast checkpoint saves; not a bottleneck |
| Isaac Sim license | Free for research (Omniverse) | Python-only (no Isaac) |
| Jetson Nano (HIL) | Available by Jul 2026 | Defer HIL to hardware phase |
| MATLAB/Simulink (HIL) | University license | Custom Python socket bridge (add 1 week) |

---

## 8. Risk Register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Phase 1 agent fails to converge | Medium | High | Simplify: reduce DR, use curriculum, tune rewards. Consult Federici et al. (2024) hyperparams. |
| Isaac OOMs on RTX 5070 (12 GB) | Medium | Medium | Cap num_envs=512. Use JAX fallback. 12 GB GDDR7 is the hard limit; Blackwell efficiency helps but doesn't change capacity. |
| PhysX integration mismatch degrades policy | Medium | Medium | Fine-tune for 1e6 steps; revert to Python if metrics worsen. |
| GTrXL overhead not justified for 10 s episodes | Medium | Low | Delete GTrXL; stick to PPO-MLP. Short episodes may not need temporal memory. |
| Timeline slip (> 6 weeks on simulation) | Medium | High | Enforce kill switches (§6.2). Ship Python-only if needed. |
| Sim-to-real gap larger than expected | Medium | High | Increase DR range, add dynamics residual learning (Phase 2/3). |
| JAX vectorization harder than expected | Low | Medium | Fall back to NumPy + SB3. Slower but proven. |

---

## Appendix A: Platform Comparison Summary

| Criterion | Python (NumPy/JAX) | Isaac Sim (OIGE) | Winner for This Project |
|---|---|---|---|
| Setup time | 0 (already built) | 1–2 weeks | **Python** |
| Training throughput (RTX 5070) | ~1–4M steps/hr (JAX) | ~2–5M steps/hr (512 envs) | **Isaac** (marginal; JAX is competitive) |
| Integration accuracy | RK4, $O(dt^5)$ | PhysX implicit Euler, $O(dt^2)$ | **Python** |
| Contact physics | Spring-damper (approximate) | PhysX (accurate) | **Isaac** |
| Sensor simulation | Gaussian noise (approximate) | Raytraced + bias/drift (realistic) | **Isaac** (marginal for indoor) |
| HIL bridge | Custom sockets (1–2 weeks) | Simulink integration (built-in) | **Isaac** |
| Debugging ease | Print + Matplotlib | USD viewer + timeline | **Python** (faster iteration) |
| VRAM usage (RTX 5070, 12 GB) | ~1–2 GB (JAX, 256 envs) | ~4–6 GB (OIGE, 512 envs) | **Python** |
| Reproducibility | Seeded RNG, deterministic | PhysX can have non-determinism | **Python** |
| Thesis value | Proves method works | Proves method transfers | **Both** (different contributions) |

---

## Appendix B: Relationship to Existing Plans

This master plan supersedes the implementation phase sections in:
- [vehicle.md §12.3](Vehicle%20Dynamics/vehicle.md) — updated to reference master plan phases
- [env.md §10](Enviornment/env.md) — updated to reference master plan phases

The physics specifications in vehicle.md and env.md remain authoritative for their respective domains. This document governs the **training and evaluation workflow** that consumes those physics models.

---

## Appendix C: Decisions Log

| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| Training platform (Phase 1) | Python (NumPy/JAX on RTX 5070, SB3 on Ryzen 9 9900X) | Isaac Sim | Speed, accuracy, simplicity. RTX 5070 Blackwell + Ryzen 12C/24T makes Python-side training fast enough; Isaac overhead not justified for initial training. |
| Isaac role | Validation + HIL + viz | Full training platform | Hybrid maximizes each platform's strengths. Full Isaac is overkill for single RB. |
| Parallel strategy (Phase 1) | JAX `vmap` (RTX 5070) or SB3 vec envs (Ryzen 9 9900X, 12–24 workers) | Isaac OIGE | Lower VRAM, faster setup, sufficient throughput for 1e7 steps. Ryzen 12C makes CPU-parallel highly viable. |
| Parallel strategy (Phase 2) | OIGE at num_envs=512–1024 | num_envs=4096 | RTX 5070 12 GB GDDR7 constraint. Scale to 1024 headless if profiling shows headroom. |
| Contact physics | Python spring-damper (Phase 1) → PhysX (Phase 2 validation) | PhysX throughout | Python is more accurate for trajectory; PhysX only wins at touchdown contact. |
| Sensor noise | Gaussian (Phase 1) → Isaac emulation (Phase 2 ablation) | Isaac throughout | Gaussian is sufficient for indoor tests. Isaac sensors are a nice-to-have, not a need-to-have. |
| GTrXL vs. MLP | Start MLP, add GTrXL if disturbance recovery needs memory | GTrXL from start | MLP is cheaper to train. GTrXL adds value only if temporal context proves necessary. |
| Multi-controller comparison | 4 controllers (PPO-MLP, GTrXL-PPO, PID, SCP) sharing identical env/obs/action | RL-only comparison | PID and SCP quantify RL's value-add; fair comparison demands shared environment (training.md §1) |
| Evaluation protocol | 400 episodes/controller, 4 conditions, paired stats | Fewer episodes, unpaired | n=100 gives 80% power at Cohen's d=0.5; paired by seed enables stronger statistical claims |
| Model export | ONNX → TensorRT pipeline specified (training.md §14) | Unspecified deployment | Master plan §4.6 mandates Jetson Nano deployment; pipeline must be defined early |
| Visualization | Post-hoc Matplotlib (Phase 1), Isaac render (Phase 2 debug) | Real-time rendering | Training speed > visual feedback. Post-hoc is sufficient for analysis. |
