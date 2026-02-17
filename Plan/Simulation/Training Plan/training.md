# Training Plan — Implementation Specification

> **Scope**: Training, tuning, and evaluation specifications for all controller variants (GTrXL-PPO, PPO-MLP, PID, SCP) on a shared 6-DOF EDF landing environment.
> All controllers share the same `EDFLandingEnv`, `VehicleDynamics`, and `EnvironmentModel`.
> This document is the authoritative specification for the training pipeline; [master_plan.md](../master_plan.md) governs the phased strategy.

---

## Table of Contents

1. [Design Rationale](#1-design-rationale)
2. [Shared Infrastructure](#2-shared-infrastructure)
3. [Observation Space](#3-observation-space)
4. [Action Space](#4-action-space)
5. [Reward Function](#5-reward-function)
6. [Episode Lifecycle](#6-episode-lifecycle)
7. [Model Architectures](#7-model-architectures)
8. [Training Procedures](#8-training-procedures)
9. [Hyperparameter Search](#9-hyperparameter-search)
10. [Curriculum Learning](#10-curriculum-learning)
11. [Evaluation Protocol](#11-evaluation-protocol)
12. [Multi-Model Comparison Framework](#12-multi-model-comparison-framework)
13. [Implementation Architecture](#13-implementation-architecture)
14. [Model Export and Deployment Readiness](#14-model-export-and-deployment-readiness)
15. [Configuration Schema](#15-configuration-schema)

---

## 1. Design Rationale

### 1.1 Core Principle: Shared Environment, Varied Controllers

The thesis contribution is **comparative evaluation** of GTrXL-PPO against baselines (PPO-MLP, PID, SCP) for retro-propulsive landing. Fair comparison demands:

- **Identical dynamics**: All controllers see the same `VehicleDynamics` plant ([vehicle.md](../Vehicle%20Dynamics/vehicle.md)).
- **Identical disturbances**: All controllers face the same `EnvironmentModel` wind/atmosphere ([env.md](../Enviornment/env.md)).
- **Identical observation pipeline**: Same sensor noise, same observation space, same action interface.
- **Identical evaluation**: Same test episodes (seeded), same metrics, same statistical tests.

The controllers differ only in **how they map observations to actions** — this is the independent variable.

### 1.2 Meta-RL Interpretation: Domain Randomization as Task Distribution

A critical insight from Federici et al. (2024) and Carradori et al. (2025) is that training across randomized scenarios with a memory-equipped policy constitutes **meta-reinforcement learning**. This is not merely domain randomization for robustness — it is a structured learning-to-adapt framework:

1. **Each episode is a distinct task**: The combination of wind profile, initial conditions, atmospheric state, and actuator characteristics defines a unique landing scenario (task $\tau_i$ sampled from distribution $p(\tau)$).
2. **The transformer memory serves as a task identifier**: During each episode, the GTrXL attends over the observation history to implicitly infer the current task parameters (wind speed, density, thrust lag behavior). This is analogous to a Bayesian posterior update — more history yields better task identification.
3. **Online adaptation emerges from meta-training**: Because the policy was trained across diverse tasks, it has learned general strategies that it can specialize in real-time using its temporal context. The MLP baseline lacks this adaptation capability — it must use the same fixed mapping for all scenarios.

**Why this distinction matters**: Simple domain randomization produces a robust-but-fixed policy. Meta-RL produces an **adaptive** policy. Federici et al. achieved 98.7% success precisely because the GTrXL-PPO agent adapted online to each Monte Carlo realization. Carradori et al. confirmed this for atmospheric landings with varying wind profiles. The training plan's domain randomization scheme (wind, atmosphere, ICs, sensor noise) already provides the task distribution $p(\tau)$ — the GTrXL architecture provides the adaptation mechanism. This framing should guide how we interpret GTrXL's advantages over MLP in the evaluation: **not just robustness, but adaptation**.

| Aspect | Standard DR (PPO-MLP) | Meta-RL (GTrXL-PPO) |
|---|---|---|
| Task distribution | Same $p(\tau)$ | Same $p(\tau)$ |
| Within-episode adaptation | None (fixed mapping) | Attends over history → implicit task inference |
| Expected advantage | Robust to *average* conditions | Adapts to *specific* conditions |
| Literature support | Hwangbo et al. (2017) | Federici et al. (2024), Carradori et al. (2025) |

### 1.3 Why Multiple Controller Types

| Controller | Type | What It Tests | Research Question |
|---|---|---|---|
| **PPO-MLP** | RL (reactive) | Can RL learn to land without temporal memory? | RQ3 baseline |
| **GTrXL-PPO** | RL (temporal) | Does attention over history improve disturbance rejection? | RQ1, RQ2 |
| **PID** | Classical | Does model-free RL outperform model-based classical control? | RQ3 baseline |
| **SCP** | Optimization | Does RL match optimal trajectory planning under nominal conditions? | RQ3 baseline |

**First-principles justification**: Including PID and SCP isn't about expecting them to win — it's about quantifying *how much* RL adds. If PID achieves 95% success, the RL contribution is incremental. If PID achieves 60%, the RL contribution is substantial. Without baselines, we can't make this claim.

### 1.4 What This Document Does Not Cover

- **Physics models**: See [vehicle.md](../Vehicle%20Dynamics/vehicle.md) for dynamics, [env.md](../Enviornment/env.md) for environment.
- **Phase strategy**: See [master_plan.md](../master_plan.md) for when to use Python vs. Isaac, phase gates, and kill switches.
- **Hardware deployment**: See README.md for HIL and flight test plans.

---

## 2. Shared Infrastructure

### 2.1 Gymnasium Environment Wrapper

All controllers interface with the simulation through a single `EDFLandingEnv` class that wraps `VehicleDynamics` + `EnvironmentModel`:

```python
class EDFLandingEnv(gymnasium.Env):
    """Shared Gymnasium environment for all controller variants.

    Wraps VehicleDynamics and EnvironmentModel with:
    - Standardized observation/action spaces
    - Sensor noise injection
    - Reward computation
    - Episode lifecycle management
    """

    def __init__(self, config: dict):
        super().__init__()
        self.env_model = EnvironmentModel(config['environment'])
        self.vehicle = VehicleDynamics(config['vehicle'], self.env_model)
        self.reward_fn = RewardFunction(config['reward'])

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(20,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)

        # Episode config (40 Hz policy rate — see §2.2 for rationale)
        self.dt_physics = config['vehicle'].get('dt', 0.005)
        self.dt_policy = config.get('dt_policy', 0.025)     # 40 Hz (was 0.05 s / 20 Hz)
        self.substeps = int(self.dt_policy / self.dt_physics)  # 5 physics steps per policy step
        self.max_time = config.get('max_episode_time', 15.0)
        self.max_steps = int(self.max_time / self.dt_policy)   # 600 steps (was 300)

        # Sensor noise config
        self.noise_cfg = config.get('sensor_noise', {})

        # Target
        self.p_target = np.array(config.get('target_position', [0.0, 0.0, 0.0]))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        ic = self._sample_initial_conditions()
        self.vehicle.reset(ic, seed=seed)
        self.env_model.reset(seed=seed)
        self.step_count = 0
        self.prev_action = np.zeros(5)
        self.wind_ema = np.zeros(3)
        self.prev_accel = np.zeros(3)
        return self._get_obs(), self._get_info()

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        u = self._scale_action(action)

        # Servo dynamics are integrated inside vehicle.step() via the
        # rate-limited first-order lag model (vehicle.md §6.3.6).
        # Commands u[1:5] are fin deflection COMMANDS; the vehicle's
        # ServoModel filters them into actual servo positions each substep.
        for _ in range(self.substeps):  # 5 substeps at 40 Hz policy / 200 Hz physics
            self.vehicle.step(u)

        obs = self._get_obs()
        reward = self.reward_fn.compute(self, action)
        terminated = self._check_terminated()
        truncated = self.step_count >= self.max_steps
        info = self._get_info()

        if terminated and info.get('landed', False):
            reward += self.reward_fn.terminal_reward(self)

        self.prev_action = action.copy()
        self.step_count += 1

        return obs, reward, terminated, truncated, info
```

### 2.2 Timing Architecture

| Parameter | Value | Rationale |
|---|---|---|
| Physics timestep ($dt_{physics}$) | 0.005 s | RK4 stability for gyroscopic dynamics ([vehicle.md §7.2](../Vehicle%20Dynamics/vehicle.md)) |
| Policy timestep ($dt_{policy}$) | 0.025 s (40 Hz) | ~3× Nyquist for 12–15 Hz Freewing servo bandwidth; 5 substeps per policy step |
| Max episode time | 15.0 s | Descent from 10 m + settling; generous margin over ~8 s nominal |
| Max policy steps | 600 | 15.0 / 0.025 = 600 steps per episode |

**Why 40 Hz policy** (deleted prior 20 Hz — see [vehicle.md §6.3.6](../Vehicle%20Dynamics/vehicle.md)):

The prior 20 Hz rate was based on a generic "~50 Hz actuator bandwidth" assumption that conflated PWM frame rate compatibility with mechanical bandwidth. First-principles analysis of the actual Freewing 9 g digital servo reveals:

- **Servo mechanical bandwidth**: ~12–15 Hz (from $f_{bw} \approx 1/(\pi \times 0.025\text{ s}) \approx 13$ Hz; literature for similar 9 g servos: 8–20 Hz).
- **Nyquist requirement**: Sampling rate $> 4$–$10 \times f_{bw}$ for 30–60° phase margin → 50–150 Hz ideal, 40 Hz is $\sim 3 \times f_{bw}$ (marginally adequate, but a major improvement over 20 Hz which is $<2 \times f_{bw}$).
- **Disturbance spectrum**: Wind gusts peak at ~1–10 Hz (Dryden model); at 20 Hz, a gust demanding rapid fin reversal encounters phase lag $>90°$ at 5–10 Hz → poor rejection, higher jerk/oscillations (violates RQ1 jerk $<10$ m/s³).
- **GTrXL granularity**: 40 Hz → 600 steps/episode provides ~2× finer temporal resolution for the transformer attention window, improving disturbance inference from history and reducing aliasing of 3–5 Hz attitude dynamics.
- **Jetson latency headroom**: At 40 Hz ($dt_{policy} = 0.025$ s), ~20 ms is available for inference + PWM output. GTrXL inference target is $<20$ ms on Jetson Nano (FP16/TensorRT). Tight but feasible; delete 40 Hz only if profiling shows $>20$ ms.
- **Servo dynamics are now modeled**: The mandatory servo model ([vehicle.md §6.3.6](../Vehicle%20Dynamics/vehicle.md)) captures physical slew lag via a rate-limited first-order lag. At 40 Hz, commands arrive during servo slew → better tracking of small corrections ($<10°$). At the prior 20 Hz, the servo often completed its slew before the next command arrived → coarse quantization of control → amplified gyro cross-coupling.

**Verdict**: 20 Hz was marginally adequate but risked 10–30% success drop in hardware (based on Hwangbo et al. drone sim-to-real gaps). 40 Hz with mandatory servo modeling aligns physics exactly with the chosen Freewing servo, boosts RQ2 robustness, and prevents HIL surprises.

### 2.3 Controller Interface

All controllers implement a common interface:

```python
class Controller(ABC):
    """Abstract base class for all landing controllers."""

    @abstractmethod
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Map observation to action in [-1, 1]^5."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state for new episode."""
        ...

    def update_memory(self, obs: np.ndarray, action: np.ndarray,
                      reward: float, done: bool) -> None:
        """Optional: update internal memory (for recurrent policies)."""
        pass
```

RL controllers (PPO-MLP, GTrXL-PPO) implement `get_action` via their learned policy networks. PID implements it via feedback gains. SCP implements it via trajectory optimization + tracking.

---

## 3. Observation Space

### 3.1 Design Principles

The observation must be:
1. **Body-frame centric** — actions are in body frame (thrust along body z, fins deflect in body-frame planes), so observations should be too. This makes the policy yaw-invariant.
2. **Sensor-realizable** — every component maps to a real sensor output on the hardware platform.
3. **Sufficient for Markov property** — a memoryless policy (MLP) should have enough information to act optimally given the observation. Temporal context (for GTrXL) provides *additional* benefit but shouldn't be *required*.
4. **Normalized scale** — components should have comparable magnitudes to avoid gradient imbalance during training.

### 3.2 Observation Vector (20 Scalars)

| Index | Dim | Name | Formula | Units | Sensor Source | Noise $\sigma$ |
|---|---|---|---|---|---|---|
| 0:3 | 3 | Target offset (body) | $\mathbf{R}^T (\mathbf{p}_{target} - \mathbf{p})$ | m | Optical flow + IMU | 0.1 m |
| 3:6 | 3 | Body velocity | $\mathbf{v}_b$ | m/s | IMU integration | 0.05 m/s |
| 6:9 | 3 | Gravity direction (body) | $\mathbf{R}^T [0, 0, 1]^T$ | — | AHRS (BNO085) | 0.02 |
| 9:12 | 3 | Angular velocity | $\boldsymbol{\omega}$ | rad/s | Gyroscope | 0.01 rad/s |
| 12 | 1 | Thrust-to-weight ratio | $T / (m g)$ | — | ESC feedback | 0.02 |
| 13:16 | 3 | Wind estimate (body) | $\hat{\mathbf{v}}_{wind,b}$ (EMA) | m/s | Derived | 0.5 m/s |
| 16 | 1 | Altitude AGL | $h = -p_z$ | m | Barometer + optical flow | 0.3 m |
| 17 | 1 | Total speed | $\|\mathbf{v}_b\|$ | m/s | Derived from IMU | 0.05 m/s |
| 18 | 1 | Angular speed | $\|\boldsymbol{\omega}\|$ | rad/s | Derived from gyro | 0.01 rad/s |
| 19 | 1 | Time fraction | $t / t_{max}$ | — | Clock | 0 |

### 3.3 Component Rationale

**Target offset in body frame** (0:3): The agent needs to know where the target is relative to itself. Expressing this in body frame means the policy doesn't need to learn the rotation mapping — "target is 2 m to my right" directly maps to "deflect right fins." This makes the policy yaw-invariant: the same network output applies regardless of heading.

**Gravity direction** (6:9): Encodes pitch and roll orientation without quaternion double-cover ($\mathbf{q}$ and $-\mathbf{q}$ produce the same $\mathbf{R}$, but different quaternion values, confusing the network). When upright, $\mathbf{g}_{body} = [0, 0, 1]^T$. Tilt shows as nonzero x/y components. This is what an accelerometer measures at low acceleration (quasi-static approximation).

**Wind estimate** (13:16): Exponential moving average of the apparent wind in body frame:

$$
\hat{\mathbf{v}}_{wind,b}^{(t)} = \alpha \cdot (\mathbf{v}_b^{(t)} - \mathbf{v}_{b,prev}^{(t)}) + (1 - \alpha) \cdot \hat{\mathbf{v}}_{wind,b}^{(t-1)}
$$

with $\alpha = 0.05$ (slow adaptation). This is a crude wind estimate that helps the MLP baseline. The GTrXL can learn a better wind estimate from its temporal context — if it does, this feature becomes redundant (an expected GTrXL advantage).

**Altitude AGL** (16): Critical for landing — ground effect activates below $2 r_{duct}$ (~0.09 m), and terminal touchdown logic depends on altitude. Redundant with target_offset_z when the target is on the ground, but kept as an explicit feature because (a) the barometer measures it directly and (b) it emphasizes the most safety-critical dimension.

**Speed scalars** (17, 18): Magnitude summaries of translational and rotational motion. Help the agent quickly assess "am I going too fast?" without computing vector norms internally. Cheap to include, valuable for safety awareness.

### 3.4 Observation Normalization

For RL training, normalize observations to approximately zero mean and unit variance using running statistics:

$$
\hat{o}_i = \frac{o_i - \mu_i}{\sigma_i + \epsilon}
$$

where $\mu_i$ and $\sigma_i$ are computed from a running buffer of observations, $\epsilon = 10^{-8}$.

**Implementation**: Use SB3's `VecNormalize` wrapper or equivalent. Freeze normalization statistics after training for evaluation and deployment.

| Component | Expected Range | Expected $\mu$ | Expected $\sigma$ |
|---|---|---|---|
| Target offset | [-10, 10] m | ~0 | ~3 m |
| Body velocity | [-5, 5] m/s | ~-1 m/s (descending) | ~2 m/s |
| Gravity direction | [-1, 1] | ~[0, 0, 0.95] | ~0.1 |
| Angular velocity | [-2, 2] rad/s | ~0 | ~0.3 rad/s |
| Thrust ratio | [0, 2] | ~1 (hover) | ~0.3 |
| Wind estimate | [-5, 5] m/s | ~0 | ~2 m/s |
| Altitude | [0, 10] m | ~5 m | ~3 m |
| Speed | [0, 10] m/s | ~2 m/s | ~1.5 m/s |
| Angular speed | [0, 3] rad/s | ~0.3 | ~0.3 rad/s |
| Time fraction | [0, 1] | 0.5 | 0.29 |

### 3.5 Sensor Noise Injection

Noise is applied **after** computing the true observation from the physics state. This matches real hardware where sensors corrupt the true state.

```python
def _get_obs(self) -> np.ndarray:
    """Compute noisy observation from true state."""
    y = self.vehicle.state
    p, v_b, q, omega, T = self.vehicle._unpack(y)
    R = quat_to_dcm(q)

    # True values
    target_offset_body = R.T @ (self.p_target - p)
    gravity_body = R.T @ np.array([0.0, 0.0, 1.0])
    h = -p[2]
    twr = T / (self.vehicle.mass * self.vehicle.g)
    speed = np.linalg.norm(v_b)
    ang_speed = np.linalg.norm(omega)
    time_frac = self.vehicle.time / self.max_time

    # Wind estimate (EMA)
    v_wind_body = R.T @ self.env_model.wind_model.sample(
        self.vehicle.time, h)
    self.wind_ema = 0.05 * v_wind_body + 0.95 * self.wind_ema

    # Inject sensor noise
    obs = np.array([
        *self._add_noise(target_offset_body, 'position', sigma=0.1),
        *self._add_noise(v_b, 'velocity', sigma=0.05),
        *self._add_noise(gravity_body, 'attitude', sigma=0.02),
        *self._add_noise(omega, 'gyro', sigma=0.01),
        self._add_noise_scalar(twr, 'thrust', sigma=0.02),
        *self._add_noise(self.wind_ema, 'wind', sigma=0.5),
        self._add_noise_scalar(h, 'altitude', sigma=0.3),
        self._add_noise_scalar(speed, 'speed', sigma=0.05),
        self._add_noise_scalar(ang_speed, 'ang_speed', sigma=0.01),
        time_frac,  # exact (from clock)
    ], dtype=np.float32)

    return obs
```

**Noise calibration**: $\sigma$ values are derived from the BNO085 IMU datasheet and PX4 optical flow specs ([vehicle.md §11.5](../Vehicle%20Dynamics/vehicle.md)). For ablation, noise can be zeroed via config (`sensor_noise.enabled: false`).

### 3.6 Observation Design Decisions

| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| Position error frame | Body frame | Inertial (NED) | Yaw-invariant policy; direct mapping to body-frame actions |
| Orientation encoding | Gravity direction (3D) | Quaternion (4D), Euler angles (3D) | No sign ambiguity, sensor-realizable (accelerometer), captures pitch/roll directly |
| Yaw information | Omitted | Include heading angle | 4-fold symmetric fins mean yaw is irrelevant for body-frame control; saves 1 obs dim |
| Wind estimate | EMA of apparent wind | Raw wind (unavailable), no wind info | Gives MLP a fighting chance; GTrXL can learn better from history |
| Previous action | Omitted | Include 5D previous action | Thrust ratio (obs[12]) partially encodes lag; adding 5 dims is expensive for marginal gain. Revisit if MLP struggles with lag. |
| Observation size | 20 | 14 (raw state), 30+ (kitchen sink) | 20 is enough for Markov property; larger obs slows GTrXL attention without benefit |

---

## 4. Action Space

### 4.1 Action Vector (5 Continuous)

| Index | Name | Range | Physical Mapping | Units |
|---|---|---|---|---|
| 0 | Thrust command | [-1, 1] | $T_{cmd} = T_{hover} \cdot (1 + a_0 \cdot \Delta_{throttle})$ | N |
| 1 | Fin 1 deflection (right) | [-1, 1] | $\delta_1 = a_1 \cdot \delta_{max}$ | rad |
| 2 | Fin 2 deflection (left) | [-1, 1] | $\delta_2 = a_2 \cdot \delta_{max}$ | rad |
| 3 | Fin 3 deflection (forward) | [-1, 1] | $\delta_3 = a_3 \cdot \delta_{max}$ | rad |
| 4 | Fin 4 deflection (aft) | [-1, 1] | $\delta_4 = a_4 \cdot \delta_{max}$ | rad |

### 4.2 Action Scaling

**Thrust** is centered at hover to provide a natural equilibrium:

$$
T_{cmd} = T_{hover} \cdot (1 + a_0 \cdot \Delta_{throttle})
$$

| Parameter | Value | Notes |
|---|---|---|
| $T_{hover}$ | $m \cdot g \approx 2.84 \cdot 9.81 \approx 27.9$ N | Hover thrust (mass from vehicle config) |
| $\Delta_{throttle}$ | 0.5 | At $a_0 = +1$: $T_{cmd} = 1.5 \cdot T_{hover}$; at $a_0 = -1$: $T_{cmd} = 0.5 \cdot T_{hover}$ |

This means the throttle envelope is $[0.5, 1.5] \times T_{hover}$. The upper bound matches the vehicle's ~1.3 TWR at full throttle. The lower bound (50% hover thrust) allows controlled descent. Clamp $T_{cmd} \geq 0$ to prevent negative thrust.

**Rationale for centering at hover**: The landing problem is a regulation task around hover. Most of the time, the agent needs small adjustments around $T_{hover}$. Centering the action at this operating point means the initial random policy (mean ≈ 0) produces near-hover thrust, which is safe. If centered at zero, the initial policy would command zero thrust — immediate crash.

**Fins** use simple symmetric scaling:

$$
\delta_k = a_k \cdot \delta_{max}, \quad \delta_{max} = 0.26 \ \text{rad} \approx 15°
$$

At $a_k = 0$, fins are neutral (no deflection). Symmetric about zero is natural since fins provide bidirectional control authority.

### 4.3 Action Scaling Implementation

```python
def _scale_action(self, action: np.ndarray) -> np.ndarray:
    """Convert normalized [-1,1] action to physical units [T_cmd, delta_1..4]."""
    T_hover = self.vehicle.mass * self.vehicle.g
    T_cmd = T_hover * (1.0 + action[0] * self.throttle_range)
    T_cmd = np.clip(T_cmd, 0.0, self.T_max)

    deltas = action[1:5] * self.delta_max
    return np.concatenate([[T_cmd], deltas])
```

### 4.4 Action Rate Limiting (Largely Superseded by Servo Model)

To prevent physically unrealistic control transients, optionally limit the rate of change between consecutive actions:

$$
a_t^{limited} = a_{t-1} + \text{clip}(a_t - a_{t-1}, -\Delta a_{max}, +\Delta a_{max})
$$

| Parameter | Value | Notes |
|---|---|---|
| $\Delta a_{max}$ (thrust) | 0.2 per step | Limits throttle to ±20% per 25 ms — realistic ESC bandwidth |
| $\Delta a_{max}$ (fins) | 0.5 per step | Limits fin commands to ±50% per 25 ms — the servo model (vehicle.md §6.3.6) already enforces physical rate limits internally |

**Decision**: Start without action-level rate limiting. The mandatory servo dynamics model ([vehicle.md §6.3.6](../Vehicle%20Dynamics/vehicle.md)) already enforces physical rate limits on fin deflections via the rate-limited first-order lag ($\dot{\delta}_{max} = 10.5$ rad/s at 6 V, derated under aero load). The jerk penalty handles remaining smoothness. Add explicit action rate limiting only if the agent learns oscillatory thrust commands that the motor lag ($\tau_{motor} = 0.1$ s) cannot smooth.

### 4.5 Action Space Design Decisions

| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| Thrust centering | Hover-centered ($a_0 = 0 \rightarrow T_{hover}$) | Zero-centered ($a_0 = 0 \rightarrow 0$) | Safe initial policy; natural operating point for regulation |
| Fin parameterization | Independent per-fin | Roll/pitch/yaw moments | Per-fin is more general; let the network learn force mixing |
| Action range | [-1, 1] | [0, 1] | Symmetric range, standard for PPO with Gaussian policy |
| Rate limiting | Off (default), optional | Always on | Jerk penalty should handle this; rate limiting constrains exploration |

---

## 5. Reward Function

### 5.1 Design Philosophy

The reward function must:
1. **Guide toward landing** — dense signal pointing toward the target.
2. **Enforce safety** — hard penalties for crashes and dangerous states.
3. **Encourage smoothness** — jerk penalty for structural safety (RQ1: < 10 m/s³).
4. **Reward efficiency** — fuel penalty for $\Delta V$ minimization (RQ3).
5. **Not over-constrain** — the RL agent should discover good strategies, not follow a prescribed trajectory.

**First-principles critique**: Many RL landing papers use sparse rewards (success/fail at terminal state). This works for simple problems but is sample-inefficient for 6-DOF landing with continuous control. Dense shaping is necessary but must not create degenerate equilibria (e.g., hovering forever to collect alive bonus). Potential-based shaping (Ng et al., 1999) is theoretically clean but hard to tune in practice. We use **weighted component rewards** with careful normalization.

### 5.2 Reward Structure

$$
r_t = r_{alive} + r_{shape} + r_{orient} + r_{jerk} + r_{fuel} + r_{action} + \mathbb{1}_{terminal} \cdot r_{terminal}
$$

### 5.3 Step Reward Components

#### 5.3.1 Alive Bonus

$$
r_{alive} = +0.1
$$

Constant per step. Encourages the agent to stay alive (not crash) and to maintain control authority. Capped by episode time limit to prevent infinite-hover exploitation.

**Deletion criterion**: If the agent exploits this by hovering at high altitude without descending, reduce to 0.01 or delete.

#### 5.3.2 Shaping Reward (Distance + Velocity)

Potential-based shaping using a distance-velocity potential:

$$
\Phi(s) = -c_d \cdot \|\mathbf{e}_p\| - c_v \cdot \|\mathbf{v}_b\|
$$

$$
r_{shape} = \gamma \cdot \Phi(s_{t+1}) - \Phi(s_t)
$$

where $\mathbf{e}_p = \mathbf{p}_{target} - \mathbf{p}$ is the position error, $\gamma = 0.99$ is the discount factor.

| Parameter | Value | Effect |
|---|---|---|
| $c_d$ | 1.0 | Reward for reducing distance to target |
| $c_v$ | 0.2 | Reward for reducing speed (approach slowly) |

**Why potential-based**: Ng et al. (1999) proved that potential-based shaping preserves the optimal policy. The agent gets rewarded for making progress (reducing distance and speed) without being locked into a specific trajectory. At the target ($\mathbf{e}_p = 0, \mathbf{v}_b = 0$), the potential is maximized (zero, since both terms vanish).

#### 5.3.3 Orientation Reward

$$
r_{orient} = -w_\theta \cdot (1 - g_{body,z})
$$

where $g_{body,z} = (\mathbf{R}^T [0,0,1])_z$ is the z-component of the gravity direction in body frame. When perfectly upright, $g_{body,z} = 1$ and $r_{orient} = 0$. When tilted by angle $\theta$, $g_{body,z} = \cos\theta$ and $r_{orient} \approx -w_\theta \cdot \theta^2 / 2$ for small $\theta$.

| Parameter | Value | Effect |
|---|---|---|
| $w_\theta$ | 0.5 | Moderate tilt penalty — not so strong that the agent refuses to tilt for lateral correction |

#### 5.3.4 Jerk Penalty

$$
r_{jerk} = -w_j \cdot \|\dot{\mathbf{a}}\| / j_{ref}
$$

where $\dot{\mathbf{a}}$ is the jerk (time derivative of acceleration), estimated via finite differences:

$$
\dot{\mathbf{a}} \approx \frac{\mathbf{a}_t - \mathbf{a}_{t-1}}{dt_{policy}}
$$

and $\mathbf{a}_t = (\mathbf{v}_{b,t} - \mathbf{v}_{b,t-1}) / dt_{policy}$ is the acceleration estimate.

| Parameter | Value | Effect |
|---|---|---|
| $w_j$ | 0.05 | Light penalty — don't over-suppress necessary maneuvers |
| $j_{ref}$ | 10.0 m/s³ | Normalization reference (RQ1 threshold) |

**Implementation note**: Jerk estimation requires storing two previous velocity samples. Initialize with zeros at episode start; skip jerk penalty for the first 2 policy steps.

#### 5.3.5 Fuel Penalty

$$
r_{fuel} = -w_f \cdot \frac{|T_{cmd}|}{T_{max}} \cdot dt_{policy}
$$

Penalizes total impulse (thrust × time), which is the $\Delta V$ proxy for fuel consumption (RQ3).

| Parameter | Value | Effect |
|---|---|---|
| $w_f$ | 0.01 | Very light — landing accuracy >> fuel efficiency |

**Rationale for low weight**: The primary objective is landing safely (RQ1, RQ2). Fuel efficiency (RQ3) is secondary. Over-weighting fuel causes the agent to under-thrust and crash. Start with 0.01; increase if the agent wastes fuel hovering.

#### 5.3.6 Action Smoothness Penalty

$$
r_{action} = -w_a \cdot \|\mathbf{a}_t - \mathbf{a}_{t-1}\|
$$

Penalizes rapid action changes (distinct from jerk, which penalizes acceleration changes). Encourages smooth actuator commands.

| Parameter | Value | Effect |
|---|---|---|
| $w_a$ | 0.02 | Light smoothing on actuator commands |

### 5.4 Terminal Reward

Applied once at episode end, on top of the final step reward.

#### 5.4.1 Landing Success Bonus

$$
r_{success} = +R_{land} \cdot \mathbb{1}[\text{landed safely}]
$$

Landed safely requires ALL of:
- Altitude $h < h_{land}$ (0.05 m)
- Touchdown velocity $\|v_{touchdown}\| < v_{max}$ (0.5 m/s)
- Tilt angle $\theta < \theta_{max}$ (15°)
- Angular rate $\|\omega\| < \omega_{max}$ (0.5 rad/s)

| Parameter | Value | Notes |
|---|---|---|
| $R_{land}$ | 100.0 | Large positive reward for successful landing |
| $h_{land}$ | 0.05 m | Proximity threshold for "on the pad" |
| $v_{max}$ | 0.5 m/s | RQ1 touchdown velocity threshold |
| $\theta_{max}$ | 15° (0.26 rad) | Max tilt at touchdown |
| $\omega_{max}$ | 0.5 rad/s | Max angular rate at touchdown |

#### 5.4.2 Precision Bonus

$$
r_{precision} = R_{prec} \cdot \exp\left(-\frac{\|\mathbf{e}_{p,xy}\|^2}{2 \sigma_{prec}^2}\right) \cdot \mathbb{1}[\text{landed safely}]
$$

Gaussian bonus centered on target. Only awarded if landed safely.

| Parameter | Value | Notes |
|---|---|---|
| $R_{prec}$ | 50.0 | Max precision bonus (on top of success) |
| $\sigma_{prec}$ | 0.1 m | Width — matches RQ1 CEP target |

#### 5.4.3 Soft Touchdown Bonus

$$
r_{soft} = R_{soft} \cdot \left(1 - \frac{\|v_{touchdown}\|}{v_{max}}\right) \cdot \mathbb{1}[\text{landed safely}]
$$

Linear bonus for landing softer than the maximum allowed velocity.

| Parameter | Value |
|---|---|
| $R_{soft}$ | 20.0 |

#### 5.4.4 Crash Penalty

$$
r_{crash} = -R_{crash} \cdot \mathbb{1}[\text{crashed}]
$$

Crashed means ground contact with violated safety thresholds (too fast, too tilted).

| Parameter | Value |
|---|---|
| $R_{crash}$ | 100.0 |

#### 5.4.5 Out-of-Bounds Penalty

$$
r_{oob} = -R_{oob} \cdot \mathbb{1}[\text{out of bounds}]
$$

Triggered if the vehicle drifts beyond a safety envelope (e.g., > 20 m from target horizontally, or gains altitude above starting height + 5 m).

| Parameter | Value |
|---|---|
| $R_{oob}$ | 50.0 |

### 5.5 Reward Weight Summary

| Component | Weight | Type | Purpose |
|---|---|---|---|
| Alive bonus | 0.1 (fixed) | Step | Survive |
| Distance shaping | $c_d = 1.0$ | Step (potential) | Approach target |
| Velocity shaping | $c_v = 0.2$ | Step (potential) | Slow down |
| Orientation | $w_\theta = 0.5$ | Step | Stay upright |
| Jerk | $w_j = 0.05$ | Step | Smooth control (RQ1) |
| Fuel | $w_f = 0.01$ | Step | Efficiency (RQ3) |
| Action smoothness | $w_a = 0.02$ | Step | Actuator smoothing |
| Landing success | $R_{land} = 100$ | Terminal | Main objective |
| Precision | $R_{prec} = 50$ | Terminal | Accuracy (RQ1) |
| Soft touchdown | $R_{soft} = 20$ | Terminal | Safety margin |
| Crash | $R_{crash} = 100$ | Terminal | Safety constraint |
| Out-of-bounds | $R_{oob} = 50$ | Terminal | Safety constraint |

### 5.6 Reward Tuning Protocol

Reward weights are hyperparameters. Tune via:

1. **Phase 1**: Train with default weights. If the agent converges to an unwanted behavior (e.g., hovering, crashing deliberately for early termination), adjust the offending weight.
2. **Ablation**: Train with each component zeroed; measure impact on landing metrics. Delete components that don't improve any metric by > 1%.
3. **Sensitivity sweep**: For critical weights ($c_d$, $R_{land}$, $R_{crash}$), sweep over ±50% range and measure success rate. Ensure the chosen value is not on a cliff edge.

### 5.7 Reward Design Decisions

| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| Shaping type | Potential-based (Ng et al.) | Dense distance-to-target | Potential-based preserves optimal policy; raw distance creates degenerate solutions (hover at fixed distance to maximize alive bonus) |
| Terminal reward | Large binary + continuous precision | Sparse binary only | Binary provides clear success signal; continuous precision gradient guides the agent to the exact target |
| Jerk penalty | Finite-difference $\dot{a}$ | Action penalty, bang-bang penalty | Jerk is the physically meaningful quantity (RQ1); action penalty doesn't directly penalize physical jerk |
| Fuel penalty weight | 0.01 (very low) | 0.1 (moderate) | Landing success >> fuel efficiency. Higher weight causes under-thrusting and crashes. |
| Alive bonus | +0.1 per step | +1.0, 0.0 | Low enough to not dominate reward but high enough to discourage crashing for early termination |

---

## 6. Episode Lifecycle

### 6.1 Initial Conditions Sampling

At each `reset()`, sample initial conditions from a configurable distribution:

| State Variable | Distribution | Default Range | Notes |
|---|---|---|---|
| Position $x, y$ (NED) | Uniform | $[-2, 2]$ m | Lateral offset from target |
| Altitude $h$ | Uniform | $[5, 10]$ m | Starting height (README spec) |
| Velocity $v_x, v_y$ (inertial) | Uniform | $[-2, 2]$ m/s | Initial lateral drift |
| Descent rate $v_z$ (inertial) | Uniform | $[0, 3]$ m/s (downward) | Initial descent speed |
| Pitch, roll | Uniform | $[-5°, 5°]$ | Small initial tilt |
| Yaw | Uniform | $[0°, 360°]$ | Random heading (yaw-invariant policy) |
| Angular rates | Uniform | $[-0.2, 0.2]$ rad/s | Small initial tumble |
| Thrust | $T_{hover}$ | Exact | Start at hover thrust |

```python
def _sample_initial_conditions(self) -> np.ndarray:
    """Sample initial state vector for new episode."""
    rng = self.np_random  # Gymnasium-provided seeded RNG

    # Position (NED: z is down, so altitude h → p_z = -h)
    px = rng.uniform(-2.0, 2.0)
    py = rng.uniform(-2.0, 2.0)
    h0 = rng.uniform(5.0, 10.0)
    pz = -h0

    # Inertial velocity → body velocity via initial rotation
    vx_i = rng.uniform(-2.0, 2.0)
    vy_i = rng.uniform(-2.0, 2.0)
    vz_i = rng.uniform(0.0, 3.0)  # positive = downward in NED

    # Orientation (small random tilt + random yaw)
    roll = rng.uniform(-np.radians(5), np.radians(5))
    pitch = rng.uniform(-np.radians(5), np.radians(5))
    yaw = rng.uniform(0, 2 * np.pi)
    q = euler_to_quat(roll, pitch, yaw)
    R = quat_to_dcm(q)

    # Transform inertial velocity to body frame
    v_b = R.T @ np.array([vx_i, vy_i, vz_i])

    # Angular rates
    omega = rng.uniform(-0.2, 0.2, size=3)

    # Thrust at hover
    T_init = self.vehicle.mass * self.vehicle.g

    return np.concatenate([[px, py, pz], v_b, q, omega, [T_init]])
```

### 6.2 Sim-to-Real Domain Randomization: Actuator Delays and Action Latency

Hwangbo et al. (2017) identified **actuator delay randomization** as a critical factor for successful sim-to-real transfer of PPO policies to physical drones. The current DR scheme covers wind, atmosphere, and initial conditions, but the real hardware introduces additional latency sources that must be randomized during training:

#### 6.2.1 Actuator Delay Randomization

Real ESCs and servos introduce variable delays between commanded and applied actions. Randomize per-episode:

$$
\tau_{act} \sim \mathcal{U}(\tau_{min}, \tau_{max})
$$

| Actuator | $\tau_{min}$ | $\tau_{max}$ | Source |
|---|---|---|---|
| ESC (thrust) | 10 ms | 40 ms | ESC response time (BLHeli_32) |
| Servos (fins) | 5 ms | 20 ms | Freewing digital servo response time (transit 0.10 s/60° at 6 V) |

**Implementation**: Buffer the action for $\lfloor \tau_{act} / dt_{physics} \rfloor$ physics steps before applying. At 40 Hz, the policy sees its action take effect with a delay of 1–2 policy steps.

> **Note**: This actuator *delay* DR is distinct from the servo *dynamics* model ([vehicle.md §6.3.6](../Vehicle%20Dynamics/vehicle.md)). The delay represents the latency before the servo begins responding to a new command (communication + internal PID settling). The dynamics model represents the physical slew rate once the servo begins moving. Both are active simultaneously during training.

```python
def step(self, action):
    action = np.clip(action, -1.0, 1.0)
    u = self._scale_action(action)

    # Actuator delay: buffer action (1–2 policy steps at 40 Hz)
    self.action_buffer.append(u)
    u_delayed = self.action_buffer[0] if len(self.action_buffer) > self.delay_steps else self.prev_applied_action

    # Servo dynamics filter fin commands inside vehicle.step() (vehicle.md §6.3.6)
    for _ in range(self.substeps):  # 5 substeps at 40 Hz
        self.vehicle.step(u_delayed)
    # ...
```

#### 6.2.2 Observation-to-Action Latency Augmentation

Beyond actuator delays, the compute pipeline (sensor → policy inference → actuator command) introduces end-to-end latency. On the Jetson Nano, this may be 10–30 ms (per RQ1 target of <50 ms). Following Hwangbo et al., randomize this during training:

$$
d_{obs} \sim \{0, 1, 2\} \text{ policy steps}
$$

When $d_{obs} > 0$, the policy receives an observation that is $d_{obs}$ steps stale. This forces the policy to be robust to information delay.

| Parameter | Value | Notes |
|---|---|---|
| $d_{obs}$ range | 0–3 steps (0–75 ms at 40 Hz) | Covers Jetson inference + sensor pipeline |
| Default | 1 step (25 ms) | Nominal expected latency at 40 Hz |
| Enabled | True (default for Phase 1) | Per Hwangbo et al. — essential for sim-to-real |

**Deletion criterion**: If hardware latency profiling shows <10 ms end-to-end (unlikely on Jetson), delete latency augmentation. Otherwise, keep — Hwangbo et al. showed this is the most impactful DR dimension for sim-to-real transfer of drone control policies.

### 6.3 Termination Conditions

An episode terminates (`terminated = True`) when:

| Condition | Logic | Outcome |
|---|---|---|
| **Landed safely** | $h < 0.05$ m AND $\|v\| < 0.5$ m/s AND $\theta < 15°$ AND $\|\omega\| < 0.5$ rad/s | Success |
| **Crashed** | $h < 0.05$ m AND (any safety threshold violated) | Failure |
| **Ground contact (fast)** | $h \leq 0$ AND $\|v_z\| > 2.0$ m/s | Hard crash (abort) |
| **Extreme tilt** | $\theta > 60°$ | Unrecoverable attitude (abort) |
| **Out of bounds** | $\|p_{xy}\| > 20$ m OR $h > h_0 + 5$ m | Drifted too far |

An episode truncates (`truncated = True`) when:

| Condition | Logic | Outcome |
|---|---|---|
| **Time limit** | $t > t_{max}$ (15.0 s) | Ran out of time |

### 6.4 Ground Contact Model

Simple spring-damper contact for Python training ([vehicle.md §6.1.3](../Vehicle%20Dynamics/vehicle.md)):

$$
F_{ground} = \begin{cases}
k_{ground} \cdot (-h) + c_{ground} \cdot (-\dot{h}) & \text{if } h \leq 0 \\
0 & \text{if } h > 0
\end{cases}
$$

| Parameter | Value | Notes |
|---|---|---|
| $k_{ground}$ | 10000 N/m | Stiff ground — settles quickly |
| $c_{ground}$ | 500 N·s/m | Critical damping ≈ $2\sqrt{k \cdot m}$ |

**When ground contact is detected**: Run 5 additional physics substeps to allow settling, then check termination conditions on the settled state.

### 6.5 Info Dictionary

Every `step()` returns an `info` dict for logging:

```python
def _get_info(self) -> dict:
    p, v_b, q, omega, T = self.vehicle._unpack(self.vehicle.state)
    R = quat_to_dcm(q)
    v_inertial = R @ v_b
    h = -p[2]
    tilt = np.arccos(np.clip((R.T @ np.array([0,0,1]))[2], -1, 1))

    return {
        'position': p.copy(),
        'velocity_body': v_b.copy(),
        'velocity_inertial': v_inertial,
        'altitude': h,
        'tilt_angle': tilt,
        'angular_rate': np.linalg.norm(omega),
        'thrust': T,
        'time': self.vehicle.time,
        'landed': self._check_landed(),
        'crashed': self._check_crashed(),
        'cep': np.linalg.norm(p[:2] - self.p_target[:2]),
        'touchdown_velocity': np.linalg.norm(v_inertial) if h < 0.1 else None,
    }
```

---

## 7. Model Architectures

### 7.1 PPO-MLP (RL Baseline)

**Purpose**: Prove that RL can learn to land without temporal memory. Baseline for GTrXL comparison.

#### 7.1.1 Architecture

| Component | Specification |
|---|---|
| Policy network | MLP: [20] → [256] → [256] → [5 mean + 5 log_std] |
| Value network | MLP: [20] → [256] → [256] → [1] |
| Activation | Tanh (hidden layers) |
| Weight init | Orthogonal (gain=√2 for hidden, gain=0.01 for output) |
| Log-std | Learnable, per-action-dim, initialized at $-0.5$ |

**Total parameters**: ~140K (policy) + ~135K (value) ≈ **275K parameters**.

#### 7.1.2 Why These Hyperparameters

- **2 hidden layers, 256 units**: Standard for continuous control (SB3 defaults). Federici et al. (2024) use 2×256. Sufficient capacity for the 20-dim → 5-dim mapping.
- **Tanh activation**: Bounded output prevents gradient explosion; standard for PPO continuous control.
- **Orthogonal init**: Better gradient flow than Xavier/He for policy gradient methods (Andrychowicz et al., 2021).
- **Separate policy/value networks**: Avoids interference between policy gradient and value regression objectives. Shared networks can work but add a failure mode.

#### 7.1.3 Deletion Criterion

If PPO-MLP achieves >95% success rate with CEP <0.15 m and jerk <10 m/s³, delete GTrXL — temporal memory is not needed for this task.

### 7.2 GTrXL-PPO (Transformer RL)

**Purpose**: Test whether attention over temporal history improves disturbance rejection, wind adaptation, and precision landing.

#### 7.2.1 Architecture

```
Input (20-dim obs per step)
    │
    ├──> Linear projection: 20 → d_model (128)
    │
    ├──> GTrXL Encoder (L layers, H heads)
    │    ├── Multi-head gated self-attention (relative positional encoding)
    │    ├── Segment-level recurrence (memory from previous segment)
    │    ├── Gating mechanism per layer (Parisotto et al., 2020)
    │    └── Layer normalization + residual connections
    │
    ├──> Policy head: d_model → 256 → [5 mean + 5 log_std]
    └──> Value head: d_model → 256 → [1]
```

| Hyperparameter | Value | Notes |
|---|---|---|
| $d_{model}$ | 128 | Embedding dimension |
| $n_{layers}$ (L) | 2 | Transformer depth |
| $n_{heads}$ (H) | 4 | Attention heads ($d_{head} = 32$) |
| Segment length | 64 | Steps per attention window |
| Memory length | 64 | Cached steps from previous segment |
| FFN dimension | 512 | Feed-forward hidden dim ($4 \times d_{model}$) |
| Dropout | 0.0 | Disable for RL (no overfitting risk at this scale) |
| Gating bias | $b_{init} = -2$ | Per Parisotto et al.: bias toward residual (stability) |

**Total parameters**: ~300K (encoder) + ~70K (heads) ≈ **370K parameters**.

#### 7.2.2 Key GTrXL Components

**Gated self-attention** (Parisotto et al., 2020): Each layer applies a gating mechanism that can bypass the attention output and fall back to the residual. This stabilizes training — without gating, transformer RL often diverges.

$$
\text{output} = \sigma(\mathbf{W}_g \cdot [\mathbf{x}, \text{attn}(\mathbf{x})] + b_g) \odot \text{attn}(\mathbf{x}) + (1 - \sigma(\cdot)) \odot \mathbf{x}
$$

With $b_g$ initialized to $-2$, the gate starts "closed" (output ≈ residual), allowing the network to gradually learn when attention helps.

**Relative positional encoding** (Dai et al., 2019): Instead of absolute position embeddings, use relative position biases in the attention scores. This allows the model to generalize to different episode lengths and segment positions.

**Segment-level recurrence**: After processing a segment of 64 steps, the hidden states are cached and used as context for the next segment. This extends the effective context window beyond the segment length, enabling the agent to "remember" early-episode disturbances.

#### 7.2.3 Memory Management

For PPO training with GTrXL:

| Aspect | Approach |
|---|---|
| Rollout collection | Collect full episodes; segment into 64-step chunks for training |
| Memory initialization | Zero at episode start |
| Memory across segments | Carry forward (no gradient through memory — stop-gradient) |
| Batch training | Process segments sequentially within each episode, reset memory between episodes |

#### 7.2.4 Expected GTrXL Advantages

| Scenario | MLP Limitation | GTrXL Advantage |
|---|---|---|
| Wind pattern adaptation | Cannot infer wind from history; relies on EMA estimate | Attends to recent velocity deviations → implicit wind estimation |
| Thrust lag compensation | Only sees current thrust ratio | Attends to command-response history → predicts future thrust |
| Multi-phase trajectory | No concept of "descent phase" vs. "landing phase" | Temporal context enables implicit phase detection |
| Disturbance recovery | Reacts only to current error | Maintains context of pre-disturbance state → faster recovery |

**Deletion criterion**: If GTrXL doesn't improve robustness (RQ2 metrics) by >10% over MLP across the disturbance envelope, delete it. The 35% parameter overhead and slower training aren't worth marginal gains.

### 7.3 PID Controller (Classical Baseline)

**Purpose**: Establish the classical control performance floor. If PID performs well, the RL contribution is incremental.

#### 7.3.1 Architecture — Cascaded Loop Structure

```
                 ┌───────────────────────────────────────┐
                 │         OUTER LOOP (40 Hz)            │
  Position error │                                       │
  ─────────────>│  Position PD → Desired attitude        │──> Desired (roll, pitch)
  Velocity       │  Altitude PID → Throttle command       │──> T_cmd
                 └───────────────────────────────────────┘
                              │
                              ▼
                 ┌───────────────────────────────────────┐
                 │         INNER LOOP (40 Hz)            │
  Attitude error │                                       │
  ─────────────>│  Attitude PD → Fin deflections          │──> δ_1..4
  Angular rates  │                                       │
                 └───────────────────────────────────────┘
```

#### 7.3.2 Outer Loop — Position Control

**Altitude (vertical axis)**:

$$
T_{cmd} = m \cdot (g + K_{p,z} \cdot e_z + K_{d,z} \cdot \dot{e}_z + K_{i,z} \cdot \int e_z \, dt)
$$

where $e_z = h_{target} - h$ is the altitude error, $h_{target} = 0$ for landing.

**Lateral (horizontal plane)**: Convert lateral position error to desired tilt:

$$
\phi_{des} = K_{p,y} \cdot e_y + K_{d,y} \cdot \dot{e}_y
$$
$$
\theta_{des} = K_{p,x} \cdot e_x + K_{d,x} \cdot \dot{e}_x
$$

Clamp desired angles to $|\phi_{des}|, |\theta_{des}| \leq 20°$ for safety.

#### 7.3.3 Inner Loop — Attitude Control

$$
\delta_{roll} = K_{p,\phi} \cdot e_\phi + K_{d,\phi} \cdot \dot{e}_\phi
$$
$$
\delta_{pitch} = K_{p,\theta} \cdot e_\theta + K_{d,\theta} \cdot \dot{e}_\theta
$$

Map attitude commands to individual fin deflections:

$$
\delta_1 = +\delta_{pitch}, \quad \delta_2 = -\delta_{pitch}, \quad \delta_3 = +\delta_{roll}, \quad \delta_4 = -\delta_{roll}
$$

(Assuming fin layout from [vehicle.md §9.1](../Vehicle%20Dynamics/vehicle.md): fins 1/2 control pitch, fins 3/4 control roll.)

#### 7.3.4 Gain Tuning — Ziegler-Nichols

1. **Linearize** the vehicle dynamics about the hover trim point (numerically compute Jacobians).
2. **Apply Ziegler-Nichols** ultimate gain method:
   - Increase $K_p$ until sustained oscillations → $K_u$ (ultimate gain), $T_u$ (oscillation period).
   - PID gains: $K_p = 0.6 K_u$, $K_i = 2 K_p / T_u$, $K_d = K_p T_u / 8$.
3. **Fine-tune** via grid search on evaluation metrics (success rate, CEP).

#### 7.3.5 PID Gain Schedule (Optional)

Different gains for different flight phases:

| Phase | Altitude Range | Gain Set | Priority |
|---|---|---|---|
| Descent | $h > 2$ m | Aggressive position tracking | Reach target area |
| Approach | $0.5 < h \leq 2$ m | Moderate, velocity-limited | Slow down |
| Terminal | $h \leq 0.5$ m | Conservative, tilt-limited | Land safely |

### 7.4 SCP Controller (Optimization Baseline)

**Purpose**: Establish the theoretical optimal performance under known dynamics. SCP computes fuel-optimal trajectories — if GTrXL matches SCP on $\Delta V$, it's learning near-optimal behavior.

#### 7.4.1 Problem Formulation

$$
\min_{\mathbf{u}(\cdot)} \int_0^{t_f} T(t) \, dt
$$

Subject to:
$$
\dot{\mathbf{p}} = \mathbf{v}, \quad m\dot{\mathbf{v}} = T\,\hat{\mathbf{e}}_z + m\mathbf{g} + \mathbf{F}_{aero}
$$
$$
T_{min} \leq T \leq T_{max}, \quad |\delta_k| \leq \delta_{max}
$$
$$
\mathbf{p}(t_f) = \mathbf{p}_{target}, \quad \mathbf{v}(t_f) = \mathbf{0}
$$
$$
\|\mathbf{v}(t)\| \leq v_{max}, \quad \theta(t) \leq \theta_{max}
$$

#### 7.4.2 Convex Relaxation

Per Acikmese & Ploen (2007), the thrust magnitude constraint is losslessly relaxed:

$$
T_{min} \leq T \leq T_{max} \quad \rightarrow \quad T \in [T_{min}, T_{max}]
$$

The aerodynamic drag term $\mathbf{F}_{aero}$ is linearized about the reference trajectory at each SCP iteration via successive linearization.

#### 7.4.3 Discretization and Solving

| Parameter | Value | Notes |
|---|---|---|
| Time horizon | $t_f$ (free or fixed) | Free final time is harder to convexify; start with fixed |
| Discretization | $N = 50$ nodes | 0.3 s per node for 15 s horizon |
| Solver | ECOS or SCS (via CVXPY) | SOCP-capable |
| Trust region | $\|x_k - x_{k-1}\| \leq \Delta_k$ | Shrinks as iterations converge |
| Max SCP iterations | 20 | Convergence tolerance $\epsilon = 10^{-4}$ |
| Solve frequency | Once per episode (open-loop) or 2 Hz (MPC-style) | MPC-style is more robust but slower |

#### 7.4.4 Two SCP Modes

**Mode A — Open-loop optimal**: Solve once at episode start with known initial conditions and zero wind. Execute the planned trajectory. Tests best-case optimization performance under nominal conditions.

**Mode B — Receding horizon (MPC-style)**: Re-solve at 2 Hz with updated state. Handles disturbances by replanning. More robust but computationally expensive (~50–100 ms per solve on Ryzen 9 9900X).

**Comparison logic**: Mode A shows SCP's theoretical optimality (RQ3 $\Delta V$ benchmark). Mode B shows SCP's practical robustness. Compare both against RL.

#### 7.4.5 SCP Limitations for Fair Comparison

| Limitation | Impact | Mitigation |
|---|---|---|
| SCP assumes known dynamics | Advantage over RL in nominal conditions | Fair — compare robustness under disturbances where model mismatch matters |
| SCP cannot handle sensor noise directly | Open-loop is noise-sensitive | Use MPC mode with state estimation for robustness comparison |
| SCP is slow to solve | Cannot run at 40 Hz on Jetson Nano | Not a hardware candidate — simulation-only baseline |
| SCP requires convex constraints | Fin aerodynamics are nonlinear | Linearize per SCP iteration; accept approximation error |

### 7.5 Architecture Comparison Summary

| Property | PPO-MLP | GTrXL-PPO | PID | SCP |
|---|---|---|---|---|
| Type | Learned (RL) | Learned (RL) | Designed (classical) | Optimized (trajectory) |
| Parameters | ~275K | ~370K | ~12 (gains) | 0 (solver config) |
| Temporal context | None (Markov) | 64–128 steps | None (PD) or integral (PID) | Full trajectory (open-loop) |
| Handles noise | Implicitly (trained on noisy obs) | Implicitly (temporal filtering) | Derivative noise amplification | State estimator needed |
| Handles wind | Via wind estimate obs | Via attention over history | Feed-forward correction (if measured) | Re-plan (MPC mode) |
| Training time | ~3–5 days | ~5–7 days | ~1 day (tuning) | ~0.5 day (config) |
| Inference latency | <5 ms | <10 ms | <1 ms | 50–100 ms (per solve) |
| GPU required | Training only | Training only | No | No |

---

## 8. Training Procedures

### 8.1 PPO Algorithm — Shared by MLP and GTrXL

Both RL controllers use Proximal Policy Optimization with Clipped Surrogate (Schulman et al., 2017).

#### 8.1.1 PPO Core Algorithm

```
for iteration = 1, 2, ..., N_iterations:
    1. Collect rollout of T steps across K parallel environments
    2. Compute advantages using GAE(λ)
    3. For epoch = 1, ..., N_epochs:
        a. Shuffle rollout into mini-batches of size B
        b. For each mini-batch:
            - Compute policy ratio: r(θ) = π_θ(a|s) / π_θ_old(a|s)
            - Clipped surrogate: L_clip = min(r(θ)·A, clip(r(θ), 1±ε)·A)
            - Value loss: L_vf = (V_θ(s) - V_target)²
            - Entropy bonus: L_ent = -H[π_θ(·|s)]
            - Total loss: L = -L_clip + c_vf · L_vf - c_ent · L_ent
            - Update θ via Adam
```

#### 8.1.2 Default Hyperparameters

| Hyperparameter | Symbol | Default Value | Notes |
|---|---|---|---|
| Learning rate | $\eta$ | $3 \times 10^{-4}$ | Adam optimizer |
| Discount factor | $\gamma$ | 0.99 | Standard for continuous control |
| GAE lambda | $\lambda$ | 0.95 | Bias-variance trade-off |
| Clip range | $\epsilon$ | 0.2 | PPO clipping |
| Value function coeff | $c_{vf}$ | 0.5 | Value loss weight |
| Entropy coeff | $c_{ent}$ | 0.01 | Exploration encouragement |
| Max gradient norm | — | 0.5 | Gradient clipping |
| Number of epochs | $K$ | 10 | Reuse per rollout batch |
| Mini-batch size | $B$ | 256 | Per gradient step |
| Rollout length | $T$ | 2048 | Steps per env per collection |
| Number of parallel envs | $N_{env}$ | 16 | SB3 SubprocVecEnv on Ryzen 9 9900X |
| Total batch size | $T \times N_{env}$ | 32768 | Steps per iteration |

#### 8.1.3 Learning Rate Schedule

Linear decay from $\eta_0$ to 0 over total training steps:

$$
\eta(t) = \eta_0 \cdot \left(1 - \frac{t}{T_{total}}\right)
$$

This prevents late-training instability when the policy is near-converged and large updates are harmful.

### 8.2 PPO-MLP Training Specifics

| Parameter | Value |
|---|---|
| Framework | Stable-Baselines3 (SB3) |
| Algorithm class | `PPO` with `MlpPolicy` |
| Total timesteps | $1 \times 10^7$ (10M) |
| Estimated wall-clock | ~24–48 hours (16 CPU envs, Ryzen 9 9900X) |
| Checkpoint interval | Every 500K steps |
| Evaluation interval | Every 100K steps (50 episodes) |

### 8.3 GTrXL-PPO Training Specifics

| Parameter | Value |
|---|---|
| Framework | Custom implementation or Ray RLlib (`attention_net`) |
| Sequence handling | Segment training: 64-step segments with 64-step memory |
| Total timesteps | $1.5 \times 10^7$ (15M — 50% more budget due to higher sample complexity) |
| Estimated wall-clock | ~48–96 hours (16 CPU envs, Ryzen 9 9900X) |
| Checkpoint interval | Every 500K steps |
| Batch processing | Full episodes → segmented into 64-step chunks → sequential processing within episode |

**GTrXL-specific training notes**:

1. **Sequence padding**: Episodes shorter than 64 steps are padded with zeros. Mask padded positions in attention.
2. **Memory burn-in (mandatory)**: At episode start, memory is zero-initialized. The first 20 steps of each episode are a **mandatory burn-in period** where hidden states are collected but **no policy gradients are computed**. This is essential for training stability per Parisotto et al. (2020) — without burn-in, the sparse attention context in early steps produces noisy gradient estimates that destabilize learning. Parisotto et al. found that burn-in "substantially reduces the frequency of catastrophic policy collapses and the need for training restarts." This is not optional for this project.
3. **Gradient truncation**: Do not backpropagate through the cached memory (stop-gradient at segment boundaries). This is standard for Transformer-XL (Dai et al., 2019) and prevents memory-length-dependent gradient computation.
4. **Learning rate warm-up**: Unlike MLP, the GTrXL encoder is sensitive to large early gradient updates (Li et al., 2023 — "transformers exhibit sensitivity to hyperparameter choices in RL settings"). Use a linear warm-up for the first $N_{warmup}$ training iterations before switching to linear decay:

$$
\eta(t) = \begin{cases}
\eta_0 \cdot \frac{t}{N_{warmup}} & \text{if } t < N_{warmup} \\
\eta_0 \cdot \left(1 - \frac{t - N_{warmup}}{T_{total} - N_{warmup}}\right) & \text{otherwise}
\end{cases}
$$

| Parameter | Value | Notes |
|---|---|---|
| $N_{warmup}$ | 50 iterations (~1.6M steps) | ~10% of total training |
| $\eta_0$ | $3 \times 10^{-4}$ | Peak learning rate |

**Rationale**: Warm-up prevents early catastrophic updates to attention weights when the policy is still random. Standard practice in transformer training (Vaswani et al., 2017), and particularly important in RL where the data distribution is non-stationary.

### 8.4 PID Tuning Procedure

No RL training — gains are tuned analytically and then refined via grid search.

**Step 1: Linearize at hover** (automated, ~10 min):

```python
A, B = linearize_dynamics(vehicle, hover_state, hover_action, dt=1e-5)
eigenvalues = np.linalg.eigvals(A)
# Expect unstable modes (positive real parts) in pitch/roll → confirms need for active control
```

**Step 2: Ziegler-Nichols on each axis** (~1 hour):

For each control loop (altitude, lateral-x, lateral-y, roll, pitch):
1. Set $K_i = K_d = 0$.
2. Increase $K_p$ until sustained oscillations in simulation.
3. Record ultimate gain $K_u$ and period $T_u$.
4. Compute PID gains: $K_p = 0.6 K_u$, $K_i = 1.2 K_u / T_u$, $K_d = 0.075 K_u \cdot T_u$.

**Step 3: Grid search refinement** (~4 hours):

Run 100 evaluation episodes per gain set. Search over ±30% of Ziegler-Nichols gains in 5 steps per axis. Select the gain set maximizing success rate, with CEP as tiebreaker.

**Step 4: Gain scheduling** (optional, ~2 hours):

If fixed gains produce >70% success but poor terminal precision, implement 3-phase gain schedule (descent / approach / terminal) and repeat grid search for each phase.

### 8.5 SCP Configuration Procedure

No training — configure the trajectory optimization solver.

**Step 1: Problem setup** (~1 day):

1. Implement the convex optimization problem in CVXPY.
2. Define constraints (thrust bounds, tilt limits, terminal conditions).
3. Test convergence on a nominal (zero-wind, centered IC) scenario.
4. Verify that the converged trajectory satisfies all constraints.

**Step 2: Trust region tuning** (~0.5 day):

Sweep trust region size $\Delta \in \{0.1, 0.5, 1.0, 2.0\}$. Select the smallest $\Delta$ that reliably converges in <20 iterations across 100 randomized ICs.

**Step 3: MPC mode setup** (optional, ~1 day):

Implement receding-horizon replanning at 2 Hz. Profile solve time on Ryzen 9 9900X (target: <50 ms per solve). If solve time exceeds budget, reduce discretization nodes $N$ or increase timestep.

### 8.6 Training Resource Budget

| Controller | Compute Resource | Wall-Clock | GPU | CPU |
|---|---|---|---|---|
| PPO-MLP | 16 CPU envs | ~24–48 hours | Optional (SB3 inference) | Ryzen 9 9900X (12C) |
| GTrXL-PPO | 16 CPU envs | ~48–96 hours | Required (attention compute) | Ryzen 9 9900X (12C) |
| PID | Grid search | ~5 hours | None | 1 core |
| SCP | Solver config | ~2 days (setup) | None | 1 core |
| **Total** | | **~6–10 days** | RTX 5070 (intermittent) | Ryzen 9 9900X |

**JAX acceleration** (optional): If CPU training is too slow, port `EDFLandingEnv` to JAX and use `jax.vmap` over 64–256 environments on the RTX 5070 (12 GB GDDR7). Expected speedup: 10–50× over SB3 CPU envs. See [master_plan.md §3.2](../master_plan.md).

---

## 9. Hyperparameter Search

### 9.1 Strategy

Use Ray Tune for automated hyperparameter optimization of RL controllers. PID and SCP are tuned manually (§8.4, §8.5).

### 9.2 Search Space — PPO-MLP

| Hyperparameter | Search Range | Scale | Priority |
|---|---|---|---|
| Learning rate | $[5 \times 10^{-5}, 3 \times 10^{-4}]$ | Log-uniform | High |
| Batch size (rollout × envs) | $\{8192, 16384, 32768, 65536\}$ | Categorical | High |
| GAE $\lambda$ | $[0.9, 0.99]$ | Uniform | Medium |
| Entropy coefficient | $[0.001, 0.05]$ | Log-uniform | Medium |
| Clip range $\epsilon$ | $\{0.1, 0.2, 0.3\}$ | Categorical | Low |
| Discount $\gamma$ | $\{0.99, 0.995, 0.999\}$ | Categorical | Low |
| Number of epochs | $\{5, 10, 15\}$ | Categorical | Low |
| Hidden layer size | $\{128, 256, 512\}$ | Categorical | Low |

### 9.3 Search Space — GTrXL-PPO

Includes MLP hyperparameters plus transformer-specific:

| Hyperparameter | Search Range | Scale | Priority |
|---|---|---|---|
| $d_{model}$ | $\{64, 128, 256\}$ | Categorical | High |
| $n_{layers}$ | $\{1, 2, 3\}$ | Categorical | High |
| $n_{heads}$ | $\{2, 4, 8\}$ | Categorical | Medium |
| Segment length | $\{32, 64, 128\}$ | Categorical | Medium |
| Memory length | $\{32, 64, 128\}$ | Categorical | Medium |
| Gate bias init | $\{-1, -2, -3\}$ | Categorical | Low |

### 9.4 Search Configuration

| Parameter | Value | Notes |
|---|---|---|
| Search algorithm | ASHA (Async Successive Halving) | Early-stops bad trials |
| Number of trials | 50 (PPO-MLP), 30 (GTrXL-PPO) | Budget-limited |
| Max training steps per trial | $3 \times 10^6$ (3M) | Enough to see convergence trends |
| Evaluation metric | Mean success rate over 50 episodes | Primary metric |
| Secondary metric | Mean CEP over successful landings | Tiebreaker |
| Grace period | $5 \times 10^5$ (500K steps) | Don't kill trials before they warm up |
| Reduction factor | 3 | Top 1/3 of trials survive each rung |
| Compute budget | ~72 GPU-hours (PPO-MLP), ~48 GPU-hours (GTrXL) | RTX 5070, ~16 concurrent envs |

### 9.5 Reward Weight Search (Optional)

If default reward weights underperform, add reward weights to the search space:

| Weight | Search Range |
|---|---|
| $c_d$ (distance shaping) | $[0.5, 2.0]$ |
| $w_\theta$ (orientation) | $[0.1, 1.0]$ |
| $R_{land}$ (success reward) | $[50, 200]$ |

**Caution**: Searching over reward weights significantly expands the search space. Only do this if default weights fail to produce >50% success rate after 5M steps.

---

## 10. Curriculum Learning

### 10.1 Overview

Optional progressive difficulty scheduling. Disabled by default — enable if vanilla uniform randomization fails to converge within 5M steps.

### 10.2 Difficulty Dimensions

| Dimension | Easy | Medium | Hard |
|---|---|---|---|
| Starting altitude | 3–5 m | 5–8 m | 5–10 m |
| Starting velocity | 0–1 m/s | 0–3 m/s | 0–5 m/s |
| Lateral offset | 0–0.5 m | 0–1.5 m | 0–2 m |
| Mean wind | 0–2 m/s | 0–5 m/s | 0–10 m/s |
| Gust probability | 0.0 | 0.05 | 0.1 |
| $\Delta T$ (atmo) | ±2 K | ±5 K | ±10 K |
| Sensor noise scale | 0.5× | 0.75× | 1.0× |

### 10.3 Scheduling

```
Progress:    0%          30%          70%         100%
             │           │            │            │
Difficulty:  Easy ──────> Medium ────> Hard ──────>
```

Transition between levels using a linear interpolation on the difficulty parameters:

$$
d(t) = d_{easy} + \frac{\min(t / T_{curriculum}, 1.0) \cdot (d_{hard} - d_{easy})}{1.0}
$$

where $T_{curriculum}$ is the step count at which full difficulty is reached (default: 70% of total training steps).

### 10.4 Automatic Difficulty Adjustment (Alternative)

Instead of fixed scheduling, adjust difficulty based on current success rate:

$$
d_{next} = \begin{cases}
d_{current} + 0.1 & \text{if success rate} > 80\% \text{ over last 100 episodes} \\
d_{current} - 0.05 & \text{if success rate} < 40\% \\
d_{current} & \text{otherwise}
\end{cases}
$$

Clamp $d \in [0, 1]$ where 0 = easy, 1 = hard.

### 10.5 Deletion Criterion

Delete curriculum learning if uniform randomization (full difficulty from step 0) achieves M3 milestone (CEP < 0.5 m, success > 90%) within $7 \times 10^6$ steps. Curriculum adds training pipeline complexity without proportional benefit if vanilla DR works.

---

## 11. Evaluation Protocol

### 11.1 Evaluation Metrics

All metrics directly map to research questions (RQ1–RQ3):

| Metric | Formula | Target (RQ) | Units |
|---|---|---|---|
| **Landing CEP** | Circular error probable: radius enclosing 50% of landings | < 0.1 m (RQ1) | m |
| **Touchdown velocity** | $\|v_{inertial}\|$ at ground contact | < 0.5 m/s (RQ1) | m/s |
| **Success rate** | Fraction of episodes with safe landing | > 99% (RQ1, n=100) | % |
| **Jerk (99th pctl)** | 99th percentile of $\|\dot{\mathbf{a}}\|$ over episode | < 10 m/s³ (RQ1) | m/s³ |
| **Controller latency** | Inference time per `get_action()` call | < 50 ms (RQ1) | ms |
| **Trajectory RMSE** | RMS position error from reference descent | < 0.1 m (RQ2) | m |
| **Recovery time** | Time to return within 0.5 m of target after disturbance | — (RQ2) | s |
| **Robustness margin** | Max wind speed before success rate drops below 90% | — (RQ2) | m/s |
| **$\Delta V$ (impulse proxy)** | $\int_0^{t_f} T(t) / m \, dt$ | minimize (RQ3) | m/s |
| **Fuel remaining** | $1 - \Delta V / \Delta V_{max}$ | > 20% (RQ3) | % |

### 11.2 Evaluation Protocol

#### 11.2.1 Standard Evaluation Suite

For each controller, run **n = 100 episodes** with fixed seeds (seeds 0–99):

| Condition | IC Distribution | Wind | Atmosphere DR | Purpose |
|---|---|---|---|---|
| **Nominal** | Default ranges | 0–5 m/s mean | ISA (no DR) | Baseline performance |
| **Windy** | Default ranges | 5–10 m/s mean + gusts | ISA (no DR) | Wind robustness (RQ2) |
| **Full DR** | Default ranges | 0–10 m/s + Dryden + gusts | Full atmo DR | Worst-case training conditions |
| **Easy** | 5 m, centered, 0 velocity | 0 wind | ISA | Lower bound / sanity check |

Total: **400 episodes per controller** (100 per condition × 4 conditions).

#### 11.2.2 Seed Matching

All controllers must be evaluated on **identical episode seeds**. The seed determines:
- Initial conditions (position, velocity, attitude)
- Wind realization (mean, turbulence sequence, gust parameters)
- Atmosphere randomization ($T_{base}$, $P_{base}$)

This enables **paired** statistical comparisons (same disturbance → different controller responses).

### 11.3 Statistical Analysis

#### 11.3.1 Pairwise Comparison

For each metric, compare controllers pairwise using **paired t-tests** (paired by episode seed):

$$
H_0: \mu_{GTrXL} = \mu_{baseline}, \quad H_1: \mu_{GTrXL} \neq \mu_{baseline}
$$

| Parameter | Value |
|---|---|
| Significance level $\alpha$ | 0.05 |
| Correction for multiple comparisons | Bonferroni ($\alpha / k$ where $k$ = number of comparisons) |
| Effect size reporting | Cohen's $d$ |
| Confidence intervals | 95% bootstrap CI (10,000 resamples) |

#### 11.3.2 Multi-Controller Comparison

For overall comparison across all 4 controllers:

$$
\text{One-way ANOVA: } H_0: \mu_1 = \mu_2 = \mu_3 = \mu_4
$$

If ANOVA rejects $H_0$, follow up with pairwise Tukey HSD tests.

#### 11.3.3 Power Analysis

With $n = 100$ episodes per condition:

| Effect size (Cohen's $d$) | Statistical power |
|---|---|
| 0.2 (small) | 0.29 |
| 0.3 (small-medium) | 0.48 |
| 0.5 (medium) | 0.80 |
| 0.8 (large) | 0.99 |

**Interpretation**: With $n = 100$, we can reliably detect medium effect sizes ($d \geq 0.5$) at 80% power. For smaller effects, increase to $n = 200$. For the key metrics (CEP, success rate), $d = 0.5$ corresponds to ~0.05 m CEP difference or ~5% success rate difference — meaningful for the research questions.

### 11.4 Robustness Characterization

#### 11.4.1 Wind Robustness Profile

Sweep mean wind speed from 0 to 15 m/s in 1 m/s increments. For each level, run 50 episodes. Plot success rate vs. wind speed. Define **robustness margin** as the wind speed where success drops below 90%.

#### 11.4.2 Disturbance Rejection Time

Inject a step gust of 5 m/s at $t = 3$ s. Measure time to return within 0.5 m of pre-disturbance trajectory. Compare across controllers.

#### 11.4.3 Sensor Noise Sensitivity

Sweep sensor noise scale from 0× to 3× nominal. For each level, run 50 episodes. Plot CEP vs. noise scale.

---

## 12. Multi-Model Comparison Framework

### 12.1 Comparison Table Template

Each evaluation condition produces a table like:

| Metric | PPO-MLP | GTrXL-PPO | PID | SCP (OL) | SCP (MPC) |
|---|---|---|---|---|---|
| Success rate | __%  ± __% | __%  ± __% | __%  ± __% | __%  ± __% | __%  ± __% |
| CEP [m] | __.__ ± __.__ | __.__ ± __.__ | __.__ ± __.__ | __.__ ± __.__ | __.__ ± __.__ |
| Touchdown vel [m/s] | __.__ ± __.__ | __.__ ± __.__ | __.__ ± __.__ | __.__ ± __.__ | __.__ ± __.__ |
| Jerk 99th [m/s³] | __.__ ± __.__ | __.__ ± __.__ | __.__ ± __.__ | __.__ ± __.__ | __.__ ± __.__ |
| $\Delta V$ [m/s] | __.__ ± __.__ | __.__ ± __.__ | __.__ ± __.__ | __.__ ± __.__ | __.__ ± __.__ |
| Robustness margin [m/s] | __ | __ | __ | __ | __ |
| Inference latency [ms] | __ | __ | __ | __ | __ |

Values are mean ± 95% CI.

### 12.2 Expected Outcomes (Hypotheses)

| Metric | Expected Ranking | Rationale |
|---|---|---|
| Success rate (nominal) | SCP-MPC ≥ GTrXL ≥ PPO-MLP > PID | SCP has full dynamics model; GTrXL has temporal context |
| Success rate (windy) | GTrXL > PPO-MLP > SCP-MPC > PID > SCP-OL | GTrXL adapts to wind patterns; SCP-OL can't replan |
| CEP | SCP-MPC ≈ GTrXL > PPO-MLP > PID | Optimization is precise; GTrXL + precision bonus helps |
| $\Delta V$ (fuel) | SCP-OL > SCP-MPC ≥ GTrXL > PPO-MLP > PID | SCP minimizes fuel by construction |
| Jerk | PPO-MLP ≈ GTrXL > SCP-MPC > PID | RL trained with jerk penalty; PID has derivative kick |
| Robustness margin | GTrXL > PPO-MLP > SCP-MPC > PID | Temporal context helps wind adaptation |

**If these hypotheses are wrong** — that's a research finding too. If PID outperforms GTrXL on disturbance rejection, it means temporal attention doesn't help for this task duration/dynamics. If PPO-MLP matches GTrXL, it means 10 s episodes don't benefit from long-horizon memory.

### 12.3 Ablation Studies

| Ablation | What It Tests | Controllers Affected |
|---|---|---|
| Remove wind estimate from obs | Does explicit wind info help MLP more than GTrXL? | PPO-MLP, GTrXL-PPO |
| Remove jerk penalty | Does jerk penalty hurt success rate? | PPO-MLP, GTrXL-PPO |
| Reduce segment length (GTrXL) | What's the minimum useful context window? | GTrXL-PPO |
| Remove gating (GTrXL → TrXL) | Does gating matter for RL stability? | GTrXL-PPO |
| Double sensor noise | How robust is each controller to degraded sensors? | All |
| Zero wind (ablation, not eval) | Train without wind, test with wind | PPO-MLP, GTrXL-PPO |
| Remove actuator delay DR | Does latency randomization improve sim-to-real transfer? | PPO-MLP, GTrXL-PPO |
| Remove observation latency | Does stale-observation training improve hardware robustness? | PPO-MLP, GTrXL-PPO |
| Remove GTrXL burn-in | How much does burn-in improve training stability? | GTrXL-PPO |
| Curriculum vs. uniform DR | Does curriculum improve final performance? | PPO-MLP, GTrXL-PPO |

---

## 13. Implementation Architecture

### 13.1 File Structure

```
simulation/
├── training/
│   ├── __init__.py
│   ├── edf_landing_env.py          # EDFLandingEnv (Gymnasium wrapper)
│   ├── reward.py                   # RewardFunction class
│   ├── observation.py              # Observation computation + noise
│   ├── curriculum.py               # CurriculumScheduler (optional)
│   ├── controllers/
│   │   ├── __init__.py
│   │   ├── base.py                 # Controller ABC
│   │   ├── ppo_mlp.py             # PPO-MLP wrapper
│   │   ├── gtrxl_ppo.py           # GTrXL-PPO wrapper
│   │   ├── pid_controller.py      # PID with gain scheduling
│   │   └── scp_controller.py      # SCP with CVXPY
│   ├── scripts/
│   │   ├── train_ppo_mlp.py       # Training entry point for PPO-MLP
│   │   ├── train_gtrxl_ppo.py     # Training entry point for GTrXL-PPO
│   │   ├── tune_pid.py            # PID gain tuning script
│   │   ├── configure_scp.py       # SCP setup and validation
│   │   ├── evaluate.py            # Multi-controller evaluation suite
│   │   ├── compare.py             # Statistical comparison and plotting
│   │   └── sweep_hparams.py       # Ray Tune hyperparameter search
│   └── configs/
│       ├── default_training.yaml  # Default training config
│       ├── ppo_mlp.yaml           # PPO-MLP specific config
│       ├── gtrxl_ppo.yaml         # GTrXL-PPO specific config
│       ├── pid.yaml               # PID gains config
│       ├── scp.yaml               # SCP solver config
│       ├── reward.yaml            # Reward weights config
│       └── evaluation.yaml        # Evaluation suite config
├── configs/
│   ├── default_vehicle.yaml       # Vehicle config (see vehicle.md)
│   └── default_environment.yaml   # Environment config (see env.md)
└── tests/
    ├── test_edf_landing_env.py    # Env wrapper tests
    ├── test_reward.py             # Reward function tests
    ├── test_observation.py        # Observation computation tests
    ├── test_pid_controller.py     # PID tests
    └── test_scp_controller.py     # SCP tests
```

### 13.2 Dependencies

```
# Core
numpy >= 1.24
gymnasium >= 0.29
pyyaml >= 6.0

# RL Training
stable-baselines3 >= 2.0       # PPO-MLP training
ray[rllib,tune] >= 2.9         # GTrXL-PPO + hyperparameter search
torch >= 2.0                    # Neural network backend

# Optimization (SCP)
cvxpy >= 1.4                    # Convex optimization
ecos >= 2.0                     # SOCP solver

# Evaluation
scipy >= 1.10                   # Statistical tests
matplotlib >= 3.7               # Plotting
pandas >= 2.0                   # Results tables
tensorboard >= 2.14             # Training curves

# Testing
pytest >= 7.0
```

### 13.3 Key Implementation Notes

#### 13.3.1 Vectorized Training

For SB3 (PPO-MLP):
```python
from stable_baselines3.common.vec_env import SubprocVecEnv

envs = SubprocVecEnv([
    lambda: EDFLandingEnv(config)
    for _ in range(n_envs)
])
model = PPO("MlpPolicy", envs, **ppo_hyperparams)
model.learn(total_timesteps=10_000_000)
```

For RLlib (GTrXL-PPO):
```python
from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
    .environment(EDFLandingEnv, env_config=env_config)
    .framework("torch")
    .training(
        model={"use_attention": True, "attention_num_transformer_units": 2,
               "attention_dim": 128, "attention_num_heads": 4,
               "attention_memory_inference": 64, "attention_memory_training": 64,
               "attention_use_n_prev_actions": 0, "attention_use_n_prev_rewards": 0}
    )
    .rollouts(num_rollout_workers=16)
)
algo = config.build()
for i in range(1000):
    result = algo.train()
```

#### 13.3.2 Checkpoint Strategy

| Event | Action |
|---|---|
| Every 500K steps | Save full checkpoint (policy + optimizer + normalization stats) |
| Best success rate so far | Save as `best_model.zip` |
| Training complete | Save final checkpoint + training logs |
| Evaluation | Load `best_model.zip` with frozen normalization |

#### 13.3.3 Logging

Log to TensorBoard and optionally Weights & Biases:

| Category | Metrics Logged | Frequency |
|---|---|---|
| Training | loss, entropy, learning rate, clip fraction, explained variance | Every update |
| Rollout | mean reward, mean episode length, success rate | Every rollout |
| Evaluation | CEP, touchdown velocity, success rate, jerk | Every 100K steps |
| System | steps/sec, GPU utilization, memory usage | Every 60 s |

---

## 14. Model Export and Deployment Readiness

### 14.1 Motivation

The master plan (§4.6) mandates ONNX export and TensorRT inference on the Jetson Nano for HIL and flight testing. Carradori et al. (2025) specifically designed their GTrXL policy for "compatibility with real-time flight computers" — all heavy computation is offline during training, and online inference reduces to efficient forward passes. This section specifies how to bridge training to deployment.

### 14.2 Export Pipeline

```
Trained Policy (PyTorch)
    │
    ├──> Freeze observation normalization (VecNormalize stats)
    │
    ├──> Trace/Script model (torch.jit.trace or torch.jit.script)
    │
    ├──> Export to ONNX (opset 17+)
    │    ├── PPO-MLP: straightforward (no dynamic state)
    │    └── GTrXL: must include memory state as model input/output
    │
    ├──> Optimize with TensorRT (FP16 precision on Jetson Nano)
    │
    └──> Validate: compare ONNX outputs vs. PyTorch outputs over 100 episodes
         (max absolute error < 1e-4 per action dimension)
```

### 14.3 GTrXL Export Considerations

The GTrXL model has **stateful inference** — the segment memory must persist between policy calls. For ONNX export:

| Component | Approach |
|---|---|
| Memory state | Export as explicit input/output tensors (not hidden internal state) |
| Segment boundary | Caller manages memory buffer; passes previous segment's hidden states as input |
| Attention mask | Fixed-size input; pad shorter sequences |
| Dynamic shapes | Use ONNX dynamic axes for batch dimension (batch=1 at inference) |

### 14.4 Latency Targets

| Controller | Target Latency | Hardware | Notes |
|---|---|---|---|
| PPO-MLP | < 5 ms | Jetson Nano (FP16) | TensorRT, batch=1 |
| GTrXL-PPO | < 20 ms | Jetson Nano (FP16) | TensorRT, batch=1, segment_len=1 |
| PID | < 1 ms | Jetson Nano (CPU) | Pure NumPy |
| End-to-end (sensor → actuator) | < 50 ms | Jetson Nano | RQ1 requirement |

**Latency budget**: 50 ms total = sensor read (5 ms) + preprocessing (5 ms) + policy inference (20 ms) + actuator command (5 ms) + margin (15 ms). At 40 Hz ($dt_{policy} = 25$ ms), the full pipeline spans ~2 policy steps — this is why observation latency augmentation (§6.2.2) trains with 0–3 steps of delay.

**If GTrXL exceeds 20 ms on Jetson**: Try (1) reducing $d_{model}$ from 128 to 64, (2) reducing $n_{layers}$ from 2 to 1, (3) INT8 quantization. If all fail, GTrXL is not deployable on Jetson — use PPO-MLP for hardware, keep GTrXL as simulation-only result.

### 14.5 Validation Protocol

After export, validate deployment-ready models by running the full evaluation suite (§11) using the ONNX/TensorRT model instead of the PyTorch model. Success criteria:

| Metric | Tolerance |
|---|---|
| Action divergence (max) | < $10^{-4}$ per dimension |
| Success rate difference | < 1% (vs. PyTorch reference) |
| CEP difference | < 0.01 m |
| Latency on target hardware | Within budget (table above) |

---

## 15. Configuration Schema

### 15.1 Master Training Config

```yaml
training:
  # Timing (40 Hz — see §2.2 for rationale)
  dt_policy: 0.025            # s, policy step (40 Hz; was 0.05 s / 20 Hz — deleted)
  max_episode_time: 15.0      # s, max episode duration
  max_steps: 600              # 15.0 / 0.025 = 600 (was 300 at 20 Hz)
  target_position: [0.0, 0.0, 0.0]  # landing target (NED origin)

  # Action scaling
  throttle_range: 0.5         # ±50% of hover thrust
  T_max: 45.0                 # N, absolute max thrust (EDF limit)
  delta_max: 0.26             # rad (~15°), max fin deflection
  action_rate_limit: false    # enable per-step rate limiting

  # Sensor noise (Phase 1: Gaussian; Phase 2: Isaac emulation)
  sensor_noise:
    enabled: true
    position_sigma: 0.1       # m
    velocity_sigma: 0.05      # m/s
    attitude_sigma: 0.02      # unitless (gravity direction)
    gyro_sigma: 0.01          # rad/s
    thrust_sigma: 0.02        # unitless (TWR)
    wind_sigma: 0.5           # m/s
    altitude_sigma: 0.3       # m

  # Initial conditions
  initial_conditions:
    altitude_range: [5.0, 10.0]       # m
    lateral_range: [-2.0, 2.0]        # m
    velocity_range: [-2.0, 2.0]       # m/s (lateral)
    descent_rate_range: [0.0, 3.0]    # m/s (downward)
    tilt_range_deg: [-5.0, 5.0]       # degrees
    angular_rate_range: [-0.2, 0.2]   # rad/s

  # Actuator delay DR (Hwangbo et al., 2017 — sim-to-real)
  # NOTE: This is communication/response delay, distinct from servo dynamics
  # (vehicle.md §6.3.6 handles physical slew rate).
  actuator_delay:
    enabled: true
    esc_delay_range: [0.010, 0.040]    # s, ESC response time
    servo_delay_range: [0.005, 0.020]  # s, Freewing digital servo response time

  # Observation latency augmentation (Hwangbo et al., 2017)
  obs_latency:
    enabled: true
    delay_steps_range: [0, 3]          # policy steps of stale observation (0–75 ms at 40 Hz)

  # Ground contact
  ground:
    k_spring: 10000.0         # N/m, contact stiffness
    c_damper: 500.0           # N·s/m, contact damping
    settling_substeps: 5      # extra physics steps on contact

  # Termination thresholds
  termination:
    landing_altitude: 0.05    # m, "on the pad"
    max_touchdown_velocity: 0.5   # m/s
    max_touchdown_tilt_deg: 15.0  # degrees
    max_touchdown_angular_rate: 0.5   # rad/s
    max_lateral_drift: 20.0   # m, out-of-bounds
    max_altitude_gain: 5.0    # m above starting height
    max_tilt_deg: 60.0        # degrees, unrecoverable

  # Curriculum (optional)
  curriculum:
    enabled: false
    ramp_fraction: 0.7        # fraction of training to reach full difficulty
```

### 15.2 Reward Config

```yaml
reward:
  # Step rewards
  alive_bonus: 0.1
  shaping:
    distance_coeff: 1.0       # c_d
    velocity_coeff: 0.2       # c_v
    gamma: 0.99               # discount for potential shaping
  orientation_weight: 0.5     # w_theta
  jerk_weight: 0.05           # w_j
  jerk_reference: 10.0        # m/s³, normalization
  fuel_weight: 0.01           # w_f
  action_smooth_weight: 0.02  # w_a

  # Terminal rewards
  landing_success: 100.0      # R_land
  precision_bonus: 50.0       # R_prec
  precision_sigma: 0.1        # m, Gaussian width
  soft_touchdown: 20.0        # R_soft
  crash_penalty: 100.0        # R_crash
  oob_penalty: 50.0           # R_oob
```

### 15.3 PPO-MLP Config

```yaml
ppo_mlp:
  algorithm: PPO
  framework: stable-baselines3
  total_timesteps: 10_000_000
  n_envs: 16

  hyperparameters:
    learning_rate: 3.0e-4
    n_steps: 2048              # rollout length per env
    batch_size: 256            # mini-batch size
    n_epochs: 10
    gamma: 0.99
    gae_lambda: 0.95
    clip_range: 0.2
    ent_coef: 0.01
    vf_coef: 0.5
    max_grad_norm: 0.5

  policy:
    net_arch: [256, 256]
    activation_fn: tanh
    ortho_init: true
    log_std_init: -0.5

  schedule:
    lr_schedule: linear        # linear decay to 0

  checkpointing:
    save_freq: 500_000         # steps
    eval_freq: 100_000         # steps
    eval_episodes: 50
```

### 15.4 GTrXL-PPO Config

```yaml
gtrxl_ppo:
  algorithm: PPO
  framework: ray-rllib
  total_timesteps: 15_000_000
  n_workers: 16

  hyperparameters:
    learning_rate: 3.0e-4
    train_batch_size: 32768
    sgd_minibatch_size: 256
    num_sgd_iter: 10
    gamma: 0.99
    lambda_: 0.95
    clip_param: 0.2
    entropy_coeff: 0.01
    vf_loss_coeff: 0.5
    grad_clip: 0.5

  attention:
    d_model: 128
    n_layers: 2
    n_heads: 4
    segment_length: 64
    memory_length: 64
    ffn_dim: 512
    dropout: 0.0
    gate_bias_init: -2.0

  memory:
    burn_in_steps: 20              # mandatory per Parisotto et al. (2020)
    zero_init: true                # zero memory at episode start

  policy_head:
    hidden_size: 256
    log_std_init: -0.5

  schedule:
    lr_schedule: warmup_linear     # warm-up then linear decay (Li et al., 2023)
    warmup_iterations: 50          # ~10% of training
```

### 15.5 PID Config

```yaml
pid:
  outer_loop:
    altitude:
      Kp: 2.0                 # N/m (initial, pre-Ziegler-Nichols)
      Ki: 0.5                 # N/(m·s)
      Kd: 1.0                 # N·s/m
      integral_limit: 5.0     # N, anti-windup
    lateral_x:
      Kp: 0.3                 # rad/m
      Kd: 0.15                # rad·s/m
    lateral_y:
      Kp: 0.3
      Kd: 0.15
    max_tilt_cmd_deg: 20.0    # max commanded tilt

  inner_loop:
    roll:
      Kp: 5.0                 # rad/(rad) → fin deflection per roll error
      Kd: 1.0                 # rad/(rad/s)
    pitch:
      Kp: 5.0
      Kd: 1.0

  gain_schedule:
    enabled: false
    phases:
      - name: descent
        altitude_range: [2.0, 100.0]
        gain_scale: 1.0
      - name: approach
        altitude_range: [0.5, 2.0]
        gain_scale: 0.7
      - name: terminal
        altitude_range: [0.0, 0.5]
        gain_scale: 0.5
```

### 15.6 SCP Config

```yaml
scp:
  mode: mpc                    # 'open_loop' or 'mpc'
  solver: ECOS
  N_nodes: 50                  # trajectory discretization
  max_iterations: 20           # SCP iterations
  convergence_tol: 1.0e-4
  trust_region: 1.0            # initial trust region radius
  trust_region_shrink: 0.5     # shrink factor on rejection
  replan_hz: 2.0               # replanning frequency (MPC mode)

  constraints:
    T_min: 5.0                 # N, minimum thrust
    T_max: 45.0                # N, maximum thrust
    delta_max_deg: 15.0        # max fin deflection
    tilt_max_deg: 30.0         # max tilt during trajectory
    v_max: 5.0                 # m/s, max speed
```

---

## Appendix A: Reward Function — Complete Implementation

```python
class RewardFunction:
    """Multi-objective reward for EDF landing task."""

    def __init__(self, config: dict):
        self.alive = config.get('alive_bonus', 0.1)
        self.c_d = config['shaping']['distance_coeff']
        self.c_v = config['shaping']['velocity_coeff']
        self.gamma = config['shaping']['gamma']
        self.w_theta = config.get('orientation_weight', 0.5)
        self.w_j = config.get('jerk_weight', 0.05)
        self.j_ref = config.get('jerk_reference', 10.0)
        self.w_f = config.get('fuel_weight', 0.01)
        self.w_a = config.get('action_smooth_weight', 0.02)

        self.R_land = config.get('landing_success', 100.0)
        self.R_prec = config.get('precision_bonus', 50.0)
        self.sigma_prec = config.get('precision_sigma', 0.1)
        self.R_soft = config.get('soft_touchdown', 20.0)
        self.R_crash = config.get('crash_penalty', 100.0)
        self.R_oob = config.get('oob_penalty', 50.0)

        self.prev_potential = None
        self.prev_velocity = None
        self.prev_accel = None

    def compute(self, env, action: np.ndarray) -> float:
        """Compute step reward."""
        p, v_b, q, omega, T = env.vehicle._unpack(env.vehicle.state)
        R = quat_to_dcm(q)
        e_p = env.p_target - p
        g_body_z = (R.T @ np.array([0, 0, 1]))[2]
        speed = np.linalg.norm(v_b)

        # Alive bonus
        r = self.alive

        # Potential-based shaping
        potential = -self.c_d * np.linalg.norm(e_p) - self.c_v * speed
        if self.prev_potential is not None:
            r += self.gamma * potential - self.prev_potential
        self.prev_potential = potential

        # Orientation penalty
        r -= self.w_theta * (1.0 - g_body_z)

        # Jerk penalty
        accel = (v_b - self.prev_velocity) / env.dt_policy if self.prev_velocity is not None else np.zeros(3)
        if self.prev_accel is not None:
            jerk = np.linalg.norm(accel - self.prev_accel) / env.dt_policy
            r -= self.w_j * jerk / self.j_ref
        self.prev_accel = accel
        self.prev_velocity = v_b.copy()

        # Fuel penalty
        T_max = env.T_max
        r -= self.w_f * (T / T_max) * env.dt_policy

        # Action smoothness penalty
        r -= self.w_a * np.linalg.norm(action - env.prev_action)

        return float(r)

    def terminal_reward(self, env) -> float:
        """Compute terminal reward on episode end."""
        info = env._get_info()
        r = 0.0

        if info.get('landed', False):
            r += self.R_land

            # Precision bonus (Gaussian)
            cep = info['cep']
            r += self.R_prec * np.exp(-cep**2 / (2 * self.sigma_prec**2))

            # Soft touchdown bonus
            v_touch = info.get('touchdown_velocity', 0.5)
            v_max = env.termination_cfg['max_touchdown_velocity']
            r += self.R_soft * max(0, 1.0 - v_touch / v_max)

        elif info.get('crashed', False):
            r -= self.R_crash

        # OOB check is handled in termination, penalty applied here
        p = info['position']
        if np.linalg.norm(p[:2]) > env.termination_cfg.get('max_lateral_drift', 20.0):
            r -= self.R_oob

        return float(r)

    def reset(self):
        """Reset internal state for new episode."""
        self.prev_potential = None
        self.prev_velocity = None
        self.prev_accel = None
```

---

## Appendix B: PID Controller — Complete Implementation

```python
class PIDController(Controller):
    """Cascaded PID controller for EDF landing."""

    def __init__(self, config: dict):
        self.outer = config['outer_loop']
        self.inner = config['inner_loop']
        self.max_tilt = np.radians(config['outer_loop']['max_tilt_cmd_deg'])

        # Integral states
        self.alt_integral = 0.0
        self.integral_limit = self.outer['altitude']['integral_limit']

        # Previous errors (for derivative)
        self.prev_alt_error = 0.0
        self.prev_roll_error = 0.0
        self.prev_pitch_error = 0.0
        self.dt = 0.025  # policy timestep (40 Hz)

    def reset(self):
        self.alt_integral = 0.0
        self.prev_alt_error = 0.0
        self.prev_roll_error = 0.0
        self.prev_pitch_error = 0.0

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Map observation to normalized action [-1, 1]^5."""
        # Unpack observation (see §3.2)
        target_body = obs[0:3]
        v_b = obs[3:6]
        g_body = obs[6:9]
        omega = obs[9:12]
        twr = obs[12]
        h = obs[16]

        # --- Outer loop: position → desired attitude + throttle ---
        # Altitude PID
        alt_error = target_body[2]  # body-z component of target offset
        self.alt_integral = np.clip(
            self.alt_integral + alt_error * self.dt,
            -self.integral_limit, self.integral_limit)
        alt_deriv = (alt_error - self.prev_alt_error) / self.dt
        self.prev_alt_error = alt_error

        K = self.outer['altitude']
        thrust_correction = K['Kp'] * alt_error + K['Ki'] * self.alt_integral + K['Kd'] * alt_deriv
        thrust_action = np.clip(thrust_correction, -1.0, 1.0)

        # Lateral PD → desired tilt
        Kx = self.outer['lateral_x']
        Ky = self.outer['lateral_y']
        pitch_des = np.clip(Kx['Kp'] * target_body[0] + Kx['Kd'] * v_b[0],
                            -self.max_tilt, self.max_tilt)
        roll_des = np.clip(Ky['Kp'] * target_body[1] + Ky['Kd'] * v_b[1],
                           -self.max_tilt, self.max_tilt)

        # --- Inner loop: attitude error → fin deflections ---
        # Current tilt from gravity direction
        roll_est = np.arctan2(g_body[1], g_body[2])
        pitch_est = np.arctan2(-g_body[0], g_body[2])

        roll_error = roll_des - roll_est
        pitch_error = pitch_des - pitch_est

        Kr = self.inner['roll']
        Kp_i = self.inner['pitch']

        roll_cmd = Kr['Kp'] * roll_error + Kr['Kd'] * (roll_error - self.prev_roll_error) / self.dt
        pitch_cmd = Kp_i['Kp'] * pitch_error + Kp_i['Kd'] * (pitch_error - self.prev_pitch_error) / self.dt

        self.prev_roll_error = roll_error
        self.prev_pitch_error = pitch_error

        # Map to fin actions (normalized to [-1, 1])
        delta_max = 0.26  # rad
        fin1 = np.clip(+pitch_cmd / delta_max, -1, 1)   # right
        fin2 = np.clip(-pitch_cmd / delta_max, -1, 1)   # left
        fin3 = np.clip(+roll_cmd / delta_max, -1, 1)    # forward
        fin4 = np.clip(-roll_cmd / delta_max, -1, 1)    # aft

        return np.array([thrust_action, fin1, fin2, fin3, fin4], dtype=np.float32)
```

---

## Appendix C: Decisions Log

| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| Observation frame | Body-frame centric | Inertial-frame | Yaw-invariant policy; direct action mapping; body sensors measure body quantities |
| Orientation encoding | Gravity direction (3D) | Quaternion (4D) | No double-cover ambiguity; sensor-realizable; captures pitch/roll |
| Observation size | 20 | 14 (state only), 30+ | 20 is sufficient for Markov; larger slows GTrXL without benefit |
| Thrust action centering | Hover-centered ($a_0 = 0 \rightarrow T_{hover}$) | Zero-centered | Safe initial policy; natural regulation operating point |
| Fin parameterization | Independent per-fin (4D) | Roll/pitch/yaw moments (3D) | More general; no assumption on fin-moment mapping |
| Reward shaping | Potential-based (Ng et al., 1999) | Dense distance, sparse only | Preserves optimal policy; encourages progress without degenerate equilibria |
| Fuel penalty weight | 0.01 (very low) | 0.1 | Landing safety >> fuel; high fuel weight causes under-thrusting |
| Terminal reward | Large success + precision Gaussian + soft bonus | Sparse binary | Multi-component gives gradient for precision beyond just success/fail |
| PPO-MLP architecture | 2×256, tanh | 3×128, ReLU | Standard for continuous control; matches Federici et al. (2024) |
| GTrXL config | $d_{model}$=128, L=2, H=4 | $d_{model}$=256, L=4 | Smaller model is faster; 370K params is sufficient for 20-dim obs |
| GTrXL segment length | 64 | 32, 128 | 64 steps × 0.025 s = 1.6 s context at 40 Hz — covers typical wind gust duration; extend to 128 if adaptation needs longer horizon |
| PID architecture | Cascaded (position → attitude → fins) | Single-loop | Standard for rocket/drone control; inner loop stabilizes attitude |
| PID tuning | Ziegler-Nichols + grid search | Manual tuning, LQR | Z-N is systematic and reproducible; grid search optimizes for landing metrics |
| SCP formulation | Lossless thrust relaxation (Acikmese) | Nonlinear NLP | Convex guarantees convergence; Acikmese relaxation is proven lossless for powered descent |
| SCP execution | MPC (2 Hz replan) + open-loop comparison | Open-loop only | MPC is more robust; open-loop provides theoretical fuel-optimal baseline |
| Evaluation sample size | n=100 per condition | n=50, n=200 | 80% power at Cohen's d=0.5; sufficient for thesis claims |
| Statistical tests | Paired t-test + ANOVA + Bonferroni | Unpaired, no correction | Paired by seed; multiple comparison correction prevents false positives |
| RL framework | SB3 (MLP) + RLlib (GTrXL) | All SB3, all RLlib | SB3 is simplest for MLP; RLlib has native attention support |
| Meta-RL framing | Explicit (DR = task distribution, GTrXL = adaptation) | Implicit (DR for robustness only) | Federici et al. (2024) and Carradori et al. (2025) show meta-RL interpretation drives better evaluation of GTrXL advantage |
| Policy rate | 40 Hz ($dt_{policy}=0.025$ s, 600 steps/ep) | 20 Hz (deleted) | 20 Hz was $<2\times$ servo bandwidth (12–15 Hz); 40 Hz gives ~$3\times$ Nyquist, finer GTrXL granularity, better gust rejection. Mandatory with servo dynamics model. |
| Servo dynamics model | Mandatory rate-limited first-order lag (vehicle.md §6.3.6) | Instant action (deleted) | Freewing servo has 0.10 sec/60° transit, τ≈0.04 s lag. Instant action causes 10–30% sim-to-real gap. Adds 4 states to integrator. |
| Action rate limiting | Off (servo model handles fin rates physically) | On (per-step clamp) | Servo dynamics enforce physical rate limits internally; action-level clamp is redundant and constrains exploration |
| Actuator delay DR | Enabled (randomized per-episode) | Disabled | Hwangbo et al. (2017) identified actuator delay as the most impactful DR dimension for sim-to-real drone control transfer. Distinct from servo dynamics (vehicle.md §6.3.6). |
| Observation latency augmentation | Enabled (0–3 step delay at 40 Hz = 0–75 ms) | Disabled | Hwangbo et al. (2017) — forces policy robustness to real-world compute latency; essential for Jetson Nano deployment |
| GTrXL memory burn-in | Mandatory (20 steps) | Optional | Parisotto et al. (2020) — "substantially reduces catastrophic policy collapses and training restarts" |
| GTrXL LR schedule | Warm-up + linear decay | Linear decay only | Li et al. (2023) — transformers sensitive to hyperparameters in RL; warm-up prevents early catastrophic updates to attention weights |
| Model export pipeline | ONNX → TensorRT (specified) | Unspecified | Master plan §4.6 mandates deployment on Jetson Nano; Carradori et al. (2025) designed for real-time flight computer compatibility |

---

## Appendix D: Notation Reference

| Symbol | Meaning | Units |
|---|---|---|
| $\mathbf{e}_p$ | Position error: $\mathbf{p}_{target} - \mathbf{p}$ | m |
| $\mathbf{g}_{body}$ | Gravity direction in body frame: $\mathbf{R}^T [0,0,1]^T$ | — |
| $g_{body,z}$ | z-component of $\mathbf{g}_{body}$ (=1 when upright) | — |
| $T_{hover}$ | Hover thrust: $m \cdot g$ | N |
| $\Delta_{throttle}$ | Throttle range fraction (±50% of hover) | — |
| $\delta_k$ | Fin $k$ deflection angle | rad |
| $\delta_{max}$ | Maximum fin deflection (0.26 rad ≈ 15°) | rad |
| $\Phi(s)$ | Reward potential function | — |
| $c_d, c_v$ | Shaping coefficients (distance, velocity) | — |
| $w_\theta, w_j, w_f, w_a$ | Reward penalty weights | — |
| $R_{land}, R_{prec}, R_{soft}$ | Terminal reward bonuses | — |
| $R_{crash}, R_{oob}$ | Terminal reward penalties | — |
| $d_{model}$ | GTrXL embedding dimension | — |
| $n_{heads}$ | Number of attention heads | — |
| $K_p, K_i, K_d$ | PID controller gains | various |
| CEP | Circular error probable (50th percentile radius) | m |
| $\Delta V$ | Total velocity change (impulse proxy) | m/s |
