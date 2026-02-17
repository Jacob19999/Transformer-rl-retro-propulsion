# Environment Model — Implementation Plan

> **Scope**: Atmospheric environment simulation for 6-DOF EDF drone retro-propulsive landing.
> Wind disturbances, thermodynamic atmosphere, and domain randomization for RL training.
> Integrates with [vehicle.md](../Vehicle%20Dynamics/vehicle.md) — the vehicle's `derivs()` consumes environment outputs.

---

## Table of Contents

1. [Design Rationale](#1-design-rationale)
2. [Core Architecture](#2-core-architecture)
3. [Wind Model](#3-wind-model)
4. [Atmosphere Model](#4-atmosphere-model)
5. [Coupling Interface](#5-coupling-interface)
6. [Domain Randomization Strategy](#6-domain-randomization-strategy)
7. [Configuration Schema](#7-configuration-schema)
8. [Validation and Testing](#8-validation-and-testing)
9. [File Map and Dependencies](#9-file-map-and-dependencies)
10. [Implementation Phases](#10-implementation-phases)

---

## 1. Design Rationale

### 1.1 First-Principles: What Does the Environment *Really* Need to Simulate?

The vehicle's motion is governed by Newton's laws: forces (thrust, drag, lift, gravity) and torques. Atmospheric variables influence these via aerodynamics and propulsion:

- **Drag force**: $\mathbf{F}_{drag} = -\frac{1}{2} \rho\, C_d\, A\, \|\mathbf{v}_{rel}\|\,\mathbf{v}_{rel}$ — depends on air density $\rho$ and relative velocity (wind-affected).
- **Fin lift/drag**: Dynamic pressure $q = \frac{1}{2} \rho\, V_e^2$ scales with $\rho$.
- **EDF thrust**: Fan curves scale with inlet air density — thrust $\propto \rho$ (mass flow rate through the duct).
- **Wind**: Adds a velocity perturbation to the vehicle's relative airspeed, causing unsteady forces and torques. The primary disturbance for robustness training (RQ2).

The environment model exists to **provide these atmospheric quantities to the vehicle dynamics** at each integration step: wind vector, air density, temperature, and pressure.

### 1.2 What to Keep — Physics Demands These

| Variable | Why It Matters | Impact on Forces |
|---|---|---|
| **Wind** $\mathbf{v}_w$ | Primary disturbance. Up to 10 m/s gusts (README disturbance envelope). Changes $\mathbf{v}_{rel}$ directly. | Aero drag, fin lift/drag, stability |
| **Temperature** $T$ | Directly in $\rho = P / (R\,T)$. Varying $T$ by ±5 K changes $\rho$ by ~1.7% — enough to shift drag and thrust. | Density → all aero forces, EDF thrust |
| **Pressure** $P$ | Directly in $\rho$. Coupled to altitude via barometric formula. Weather-front variation ~±1000 Pa shifts $\rho$ by ~1%. | Density → all aero forces, EDF thrust |

### 1.3 What to Delete — First-Principles Critique

Every deleted feature is justified by quantifying its impact:

| Deleted Feature | Quantified Impact | Reason |
|---|---|---|
| **Humidity** | ~1–2% effect on $\rho$ via water vapor partial pressure | Below noise floor of other uncertainties. Adds state complexity without proportional fidelity. Violates Occam's razor for indoor tests. |
| **Wind shear** (altitude gradient) | Over 5–10 m test altitude: $\Delta v_{wind} < 0.5$ m/s | Negligible for short vertical profiles. Dryden turbulence already captures stochastic variation. |
| **Coriolis force** | $2\,\Omega_{earth} \times v \approx 10^{-4}$ m/s² | Five orders of magnitude below gravity. Irrelevant at this scale. |
| **Radiation / thermal plumes** | Non-quantifiable for indoor tests | No evidence of >0.1 m RMSE impact. Delete unless proven otherwise. |
| **Magnetic interference** | Off-axis for aerodynamic sim | Not a force on the airframe. Sensor noise is handled in the observation pipeline (vehicle.md §11.3). |
| **Icing / condensation** | Indoor test environment, non-condensing | Zero probability in controlled test conditions. |

**Decision heuristic**: If a variable causes $<$0.1 m RMSE deviation in landing CEP (per RQ2 metrics), delete it. If it can't be quantified, delete until evidence proves otherwise.

### 1.4 Why a Separate Environment Module

Previously, wind and $\rho$ were embedded in vehicle.md's force models. This violates separation of concerns:

- **Modularity**: The atmosphere affects *multiple* force models (aero, fins, thrust). A single environment query per timestep avoids redundant computation and ensures consistent $\rho$ across all models.
- **Domain randomization**: Episode-level atmospheric variation (for meta-RL) is an environment concern, not a vehicle concern. The vehicle's physics are fixed; the world it operates in varies.
- **Testing**: Environment models can be unit-tested independently of vehicle dynamics.
- **Isaac Sim integration**: In Isaac Sim mode, environment parameters can be varied per-env in vectorized training without touching PhysX rigid-body config.

### 1.5 Non-Negotiable Terms

- **Seeded randomization**: All stochastic models (wind, atmosphere perturbations) must be reproducible via a seed per episode. RL training requires deterministic replay.
- **Vectorized interface**: Must support batched calls for N parallel environments (shape `(N, 3)` wind vectors, `(N,)` densities).
- **Altitude-dependent wind**: Dryden turbulence intensity scales with altitude — not optional, it's in the MIL-F-8785C spec.
- **Consistent $\rho$ per step**: All force models in a single `derivs()` call must see the same $\rho$. Compute once, pass to all.

---

## 2. Core Architecture

### 2.1 Class Hierarchy

```
EnvironmentModel                (top-level, owns sub-models and RNG)
├── WindModel                   (mean wind + Dryden turbulence + gusts)
└── AtmosphereModel             (ISA baseline + episode randomization)
```

### 2.2 Top-Level Class: `EnvironmentModel`

```python
class EnvironmentModel:
    """Atmospheric environment for 6-DOF EDF landing simulation.

    Provides wind vector, air density, temperature, and pressure
    at each integration step. Owns all stochastic atmosphere state.
    """

    def __init__(self, config: dict, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.wind_model = WindModel(config['wind'], rng=self.rng)
        self.atmo_model = AtmosphereModel(config['atmosphere'], rng=self.rng)

    def reset(self, seed: int | None = None) -> None:
        """Reset for new episode. Re-sample mean wind, atmosphere offsets."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.wind_model.rng = self.rng
            self.atmo_model.rng = self.rng
        self.wind_model.reset()
        self.atmo_model.reset()

    def sample_at_state(self, t: float, p: np.ndarray) -> dict:
        """Query environment at time t, inertial position p (NED).

        Returns:
            dict with keys:
                'wind':  (3,) inertial-frame wind velocity [m/s]
                'rho':   scalar air density [kg/m³]
                'T':     scalar temperature [K]
                'P':     scalar pressure [Pa]
        """
        h = -p[2]  # altitude (NED: z down, so h = -z)
        wind_vec = self.wind_model.sample(t, h)
        T, P, rho = self.atmo_model.get_conditions(h)
        return {'wind': wind_vec, 'rho': rho, 'T': T, 'P': P}
```

### 2.3 Data Flow

```
                    ┌──────────────────────┐
                    │  EnvironmentModel     │
                    │  ┌────────────────┐   │
  t, p ───────────>│  │   WindModel     │───│──> v_wind (3,)
  (time, position)  │  └────────────────┘   │
                    │  ┌────────────────┐   │
                    │  │ AtmosphereModel │───│──> rho, T, P
                    │  └────────────────┘   │
                    └──────────────────────┘
                              │
                              ▼
                    ┌──────────────────────┐
                    │  VehicleDynamics      │
                    │  derivs(y, u, t)      │
                    │                      │
                    │  v_rel = v_b - R'·v_w │  ← wind affects relative velocity
                    │  F_aero ∝ rho         │  ← density affects drag
                    │  F_fins ∝ rho         │  ← density affects fin forces
                    │  F_thrust ∝ rho/rho₀  │  ← density affects EDF mass flow
                    └──────────────────────┘
```

---

## 3. Wind Model

### 3.1 Overview

Wind is decomposed into three additive components:

$$
\mathbf{v}_{wind}(t, h) = \mathbf{v}_{mean} + \mathbf{v}_{turb}(t, h) + \mathbf{v}_{gust}(t)
$$

| Component | Nature | Purpose |
|---|---|---|
| $\mathbf{v}_{mean}$ | Constant per episode | Steady crosswind / headwind. Sampled at `reset()`. |
| $\mathbf{v}_{turb}$ | Stochastic, continuous | Dryden-spectrum turbulence. Tests controller bandwidth. |
| $\mathbf{v}_{gust}$ | Rare discrete events | Step-function gust fronts. Tests transient recovery. |

### 3.2 Mean Wind

Sampled uniformly per episode from a configurable bounding box:

$$
v_{mean,i} \sim \mathcal{U}(v_{min,i},\, v_{max,i}), \quad i \in \{x, y, z\}
$$

Default ranges (inertial NED frame):
- Horizontal (N, E): $[-10, +10]$ m/s
- Vertical (D): $[-2, +2]$ m/s (updrafts/downdrafts — smaller range, since vertical wind is rare near ground)

**Implementation**: Sampled once at `reset()`, held constant for the episode.

```python
def _sample_mean_wind(self) -> np.ndarray:
    lo = np.array(self.config['mean_vector_range_lo'])  # [v_x_min, v_y_min, v_z_min]
    hi = np.array(self.config['mean_vector_range_hi'])
    return self.rng.uniform(lo, hi)
```

### 3.3 Dryden Turbulence Model

#### 3.3.1 Background

The Dryden model (MIL-F-8785C / MIL-HDBK-1797) generates realistic turbulence by passing white noise through shaping filters. It produces three turbulence components $(u_g, v_g, w_g)$ in the body-aligned wind axis with specified power spectral densities.

#### 3.3.2 Transfer Functions

Longitudinal ($u_g$):
$$
H_u(s) = \sigma_u \sqrt{\frac{2\,L_u}{\pi\,V}} \cdot \frac{1}{1 + \frac{L_u}{V}\,s}
$$

Lateral ($v_g$):
$$
H_v(s) = \sigma_v \sqrt{\frac{L_v}{\pi\,V}} \cdot \frac{1 + \frac{\sqrt{3}\,L_v}{V}\,s}{\left(1 + \frac{L_v}{V}\,s\right)^2}
$$

Vertical ($w_g$):
$$
H_w(s) = \sigma_w \sqrt{\frac{L_w}{\pi\,V}} \cdot \frac{1 + \frac{\sqrt{3}\,L_w}{V}\,s}{\left(1 + \frac{L_w}{V}\,s\right)^2}
$$

#### 3.3.3 Parameters (Low Altitude, $h < 1000$ ft)

Scale lengths (altitude-dependent):
$$
L_u = L_v = \frac{h}{(0.177 + 0.000823\,h)^{1.2}}, \quad L_w = h
$$

Turbulence intensities (light-to-moderate):
$$
\sigma_w = 0.1 \cdot W_{20}
$$
$$
\sigma_u = \sigma_v = \frac{\sigma_w}{(0.177 + 0.000823\,h)^{0.4}}
$$

where $W_{20}$ is the wind speed at 20 ft (~6 m), estimated from the mean wind magnitude. For the sim, $W_{20}$ is scaled from the config turbulence intensity parameter.

**Altitude clamping**: For very low altitudes ($h < 0.5$ m), clamp $h = 0.5$ m to avoid singularities in scale length formulas.

#### 3.3.4 Discrete-Time Implementation

For fixed-step integration at $dt$, discretize the continuous transfer functions using a first-order filter approximation:

```python
class DrydenFilter:
    """Discrete-time Dryden turbulence filter (3-axis)."""

    def __init__(self, dt: float, V_ref: float, config: dict):
        self.dt = dt
        self.V_ref = V_ref  # reference airspeed for filter design
        self.intensity = config['turbulence_intensity']
        self.state_u = 0.0
        self.state_v = np.zeros(2)  # 2nd-order filter state
        self.state_w = np.zeros(2)

    def step(self, h: float, white_noise: np.ndarray) -> np.ndarray:
        """Advance filter one step. Returns (u_g, v_g, w_g) turbulence."""
        h_clamped = max(h, 0.5)
        L_u, L_v, L_w, sigma_u, sigma_v, sigma_w = self._compute_params(h_clamped)

        V = max(self.V_ref, 1.0)  # avoid division by zero

        # Longitudinal (1st-order)
        alpha_u = self.dt * V / L_u
        self.state_u = (1 - alpha_u) * self.state_u + \
                        sigma_u * np.sqrt(2 * alpha_u) * white_noise[0]

        # Lateral (2nd-order, simplified discrete approximation)
        alpha_v = self.dt * V / L_v
        self.state_v[0] = (1 - alpha_v) * self.state_v[0] + alpha_v * white_noise[1]
        self.state_v[1] = (1 - alpha_v) * self.state_v[1] + alpha_v * self.state_v[0]
        v_g = sigma_v * (self.state_v[0] + self.state_v[1]) * np.sqrt(L_v / (np.pi * V))

        # Vertical (2nd-order, same structure)
        alpha_w = self.dt * V / L_w
        self.state_w[0] = (1 - alpha_w) * self.state_w[0] + alpha_w * white_noise[2]
        self.state_w[1] = (1 - alpha_w) * self.state_w[1] + alpha_w * self.state_w[0]
        w_g = sigma_w * (self.state_w[0] + self.state_w[1]) * np.sqrt(L_w / (np.pi * V))

        return np.array([self.state_u, v_g, w_g])

    def _compute_params(self, h: float) -> tuple:
        """Compute altitude-dependent Dryden parameters."""
        denom = (0.177 + 0.000823 * h) ** 1.2
        L_u = L_v = h / denom
        L_w = h
        sigma_w = self.intensity
        sigma_u = sigma_v = sigma_w / (0.177 + 0.000823 * h) ** 0.4
        return L_u, L_v, L_w, sigma_u, sigma_v, sigma_w

    def reset(self) -> None:
        self.state_u = 0.0
        self.state_v[:] = 0.0
        self.state_w[:] = 0.0
```

### 3.4 Discrete Gusts

Rare step-function wind events to test transient recovery:

$$
\mathbf{v}_{gust}(t) = \begin{cases}
\mathbf{g}_{amp} & \text{if } t_{onset} \leq t < t_{onset} + \Delta t_{gust} \\
\mathbf{0} & \text{otherwise}
\end{cases}
$$

| Parameter | Value | Notes |
|---|---|---|
| Gust probability per episode | 0.1 (configurable) | 10% of episodes include a gust event |
| $\mathbf{g}_{amp}$ magnitude | 3–8 m/s | Sampled uniformly at reset |
| $\mathbf{g}_{amp}$ direction | Random unit vector (horizontal bias) | NED, horizontal components dominant |
| $\Delta t_{gust}$ | 0.5–2.0 s | Duration of gust event |
| $t_{onset}$ | Uniform in $[0.2 \cdot T_{ep},\, 0.8 \cdot T_{ep}]$ | Not at episode boundaries |

**Implementation**: At `reset()`, with probability `gust_prob`, sample gust parameters. During `sample()`, check time window.

```python
def _sample_gust(self, t: float) -> np.ndarray:
    if not self.gust_active:
        return np.zeros(3)
    if self.gust_onset <= t < self.gust_onset + self.gust_duration:
        return self.gust_amplitude
    return np.zeros(3)
```

### 3.5 Complete WindModel Class

```python
class WindModel:
    """Wind disturbance model: mean + Dryden turbulence + discrete gusts."""

    def __init__(self, config: dict, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self.dryden = DrydenFilter(
            dt=config.get('dt', 0.005),
            V_ref=config.get('V_ref', 10.0),
            config=config
        )
        self.mean_wind = np.zeros(3)
        self.gust_active = False
        self.gust_onset = 0.0
        self.gust_duration = 0.0
        self.gust_amplitude = np.zeros(3)

    def reset(self) -> None:
        """Re-sample mean wind and gust parameters for new episode."""
        self.mean_wind = self._sample_mean_wind()
        self.dryden.reset()
        self._setup_gust()

    def sample(self, t: float, h: float) -> np.ndarray:
        """Return inertial wind vector at time t, altitude h. Seeded per episode."""
        noise = self.rng.standard_normal(3)
        turb = self.dryden.step(h, noise)
        gust = self._sample_gust(t)
        return self.mean_wind + turb + gust

    def _sample_mean_wind(self) -> np.ndarray:
        lo = np.array(self.config['mean_vector_range_lo'])
        hi = np.array(self.config['mean_vector_range_hi'])
        return self.rng.uniform(lo, hi)

    def _setup_gust(self) -> None:
        self.gust_active = self.rng.random() < self.config.get('gust_prob', 0.1)
        if self.gust_active:
            mag = self.rng.uniform(
                self.config.get('gust_magnitude_range', [3.0, 8.0])[0],
                self.config.get('gust_magnitude_range', [3.0, 8.0])[1]
            )
            direction = self.rng.standard_normal(3)
            direction[2] *= 0.3  # horizontal bias
            direction /= np.linalg.norm(direction)
            self.gust_amplitude = mag * direction
            ep_duration = self.config.get('episode_duration', 15.0)
            self.gust_onset = self.rng.uniform(0.2 * ep_duration, 0.8 * ep_duration)
            self.gust_duration = self.rng.uniform(0.5, 2.0)

    def _sample_gust(self, t: float) -> np.ndarray:
        if not self.gust_active:
            return np.zeros(3)
        if self.gust_onset <= t < self.gust_onset + self.gust_duration:
            return self.gust_amplitude
        return np.zeros(3)
```

### 3.6 Wind Model Design Decisions

| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| Turbulence spectrum | Dryden (MIL-F-8785C) | von Kármán, white noise | Dryden is simpler to discretize (rational transfer functions) while capturing realistic PSD. von Kármán uses irrational functions requiring approximation. |
| Gust model | Rare step function | Continuous ramp, 1-cosine gust (MIL-F-8785C) | Step is worst-case for transient response. If controller handles step gusts, ramps are trivially easier. Delete complexity. |
| Wind direction | 3-axis (NED), horizontal-biased gusts | 2D horizontal only | Vertical component (downdraft) matters for descent — EDF approaching landing pad can encounter ground-induced updrafts. |
| Altitude dependence | Dryden $L_u, \sigma_u$ scale with $h$ | Constant turbulence | MIL-F-8785C mandates altitude scaling. Ignoring it overestimates turbulence at very low $h$, leading to over-conservative policies. |
| Shear model | Deleted | Linear wind gradient | $\Delta v < 0.5$ m/s over 10 m. Dryden captures stochastic equivalent. Delete deterministic shear. |

---

## 4. Atmosphere Model

### 4.1 Overview

The atmosphere model computes thermodynamic properties at a given altitude, providing density $\rho$ that scales all aerodynamic forces and EDF thrust. It uses the International Standard Atmosphere (ISA) as a baseline with per-episode randomization for domain randomization.

### 4.2 ISA Baseline

The International Standard Atmosphere defines sea-level conditions and a temperature lapse rate:

| Parameter | Symbol | ISA Value | Units |
|---|---|---|---|
| Sea-level temperature | $T_0$ | 288.15 | K |
| Temperature lapse rate | $\lambda$ | -0.0065 | K/m |
| Sea-level pressure | $P_0$ | 101325 | Pa |
| Specific gas constant (dry air) | $R$ | 287.058 | J/(kg·K) |
| Gravitational acceleration | $g$ | 9.81 | m/s² |
| Sea-level density | $\rho_0$ | 1.225 | kg/m³ |

### 4.3 Thermodynamic Equations

**Temperature** (troposphere, $h < 11$ km):
$$
T(h) = T_{base} + \lambda \cdot h
$$

**Pressure** (hydrostatic balance):
$$
P(h) = P_{base} \cdot \left(\frac{T(h)}{T_{base}}\right)^{-g / (R\,\lambda)}
$$

The exponent evaluates to $-g / (R\,\lambda) = -9.81 / (287.058 \times -0.0065) \approx 5.256$.

**Density** (ideal gas law):
$$
\rho(h) = \frac{P(h)}{R\,T(h)}
$$

### 4.4 Altitude Lapse — Keep or Delete?

For the operational altitude range (5–10 m above ground):

| Quantity | At $h = 0$ m | At $h = 10$ m | $\Delta$ | Relative Change |
|---|---|---|---|---|
| $T$ | 288.15 K | 288.085 K | -0.065 K | 0.023% |
| $P$ | 101325 Pa | 101207 Pa | -118 Pa | 0.117% |
| $\rho$ | 1.2250 kg/m³ | 1.2236 kg/m³ | -0.0014 | 0.114% |

**Verdict**: The altitude lapse over 10 m causes $<$0.12% change in $\rho$. This is **negligible** for force accuracy — well below sensor noise and $C_d$ uncertainty.

**Decision**: **Keep the lapse computation** (it's a single multiply-add, effectively free) but **do not expect it to matter**. The real value of the atmosphere model is **per-episode randomization** of $T_{base}$ and $P_{base}$.

### 4.5 Per-Episode Randomization

At `reset()`, sample base conditions from perturbed ranges. This is **domain randomization** for meta-RL training — the controller learns policies robust to atmospheric variation without modeling every detail.

$$
T_{base}^{ep} = T_0 + \delta T, \quad \delta T \sim \mathcal{U}(-\Delta T,\, +\Delta T)
$$
$$
P_{base}^{ep} = P_0 + \delta P, \quad \delta P \sim \mathcal{U}(-\Delta P,\, +\Delta P)
$$

| Randomization Parameter | Symbol | Default | Physical Interpretation |
|---|---|---|---|
| Temperature perturbation | $\Delta T$ | ±10 K | Hot summer day (+10 K) to cold morning (-10 K) |
| Pressure perturbation | $\Delta P$ | ±2000 Pa | Weather front passing (~±2 kPa) |

**Resulting $\rho$ range** at $h = 0$:

- Hot, low-pressure: $T = 298.15$ K, $P = 99325$ Pa → $\rho = 1.160$ kg/m³ (−5.3%)
- Cold, high-pressure: $T = 278.15$ K, $P = 103325$ Pa → $\rho = 1.294$ kg/m³ (+5.6%)

A ~11% total swing in $\rho$ produces a corresponding ~11% swing in all aerodynamic forces and ~5.5% in EDF thrust — significant enough to stress-test the controller.

### 4.6 AtmosphereModel Class

```python
class AtmosphereModel:
    """ISA-based atmosphere with per-episode randomization.

    Computes T, P, rho at altitude h. Base conditions are
    randomized at reset() for domain randomization.
    """

    # Physical constants
    R = 287.058       # J/(kg·K), specific gas constant for dry air
    g = 9.81          # m/s²
    LAPSE = -0.0065   # K/m, tropospheric lapse rate

    # ISA sea-level defaults
    T0 = 288.15       # K
    P0 = 101325.0     # Pa
    RHO0 = 1.225      # kg/m³ (reference for thrust scaling)

    def __init__(self, config: dict, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self.rand_T = config.get('randomize_T', 10.0)
        self.rand_P = config.get('randomize_P', 2000.0)
        self.T_base = self.T0
        self.P_base = self.P0
        self._exponent = -self.g / (self.R * self.LAPSE)  # ≈ 5.256, precomputed

    def reset(self) -> None:
        """Randomize base atmosphere for new episode."""
        self.T_base = self.T0 + self.rng.uniform(-self.rand_T, self.rand_T)
        self.P_base = self.P0 + self.rng.uniform(-self.rand_P, self.rand_P)

    def get_conditions(self, h: float) -> tuple[float, float, float]:
        """Compute (T, P, rho) at altitude h [m] above ground.

        Returns:
            T:   temperature [K]
            P:   pressure [Pa]
            rho: air density [kg/m³]
        """
        T = self.T_base + self.LAPSE * h
        P = self.P_base * (T / self.T_base) ** self._exponent
        rho = P / (self.R * T)
        return T, P, rho

    @property
    def rho_ref(self) -> float:
        """Reference sea-level ISA density for thrust normalization."""
        return self.RHO0
```

### 4.7 Atmosphere Design Decisions

| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| Baseline model | ISA (dry air, ideal gas) | USSA-76, NRLMSISE-00 | ISA is analytically tractable and standard. Complex models add compute for negligible $\rho$ difference at $h < 100$ m. |
| Humidity | Deleted | Virtual temperature correction | <2% $\rho$ effect. Indoor tests are dry. Delete. |
| Lapse rate | Kept (free compute) | Deleted | One multiply-add per call. Harmless to keep for correctness even though effect is <0.12%. |
| Randomization | Uniform $\delta T$, $\delta P$ per episode | Gaussian, time-varying, altitude-correlated | Uniform is sufficient for domain randomization. Time-varying weather within a 10 s episode is physically unrealistic. |
| Temperature range | ±10 K | ±5 K, ±20 K | ±10 K covers realistic seasonal variation (Mankato, MN: -20°C winter to +35°C summer) without extreme extrapolation. |
| Pressure range | ±2000 Pa | ±1000 Pa, ±5000 Pa | ±2 kPa covers typical weather-front variation. ±5 kPa would simulate hurricane conditions — overkill. |

---

## 5. Coupling Interface

### 5.1 How Vehicle Dynamics Consumes Environment Data

In `vehicle.py`'s `derivs()`, the environment is queried once per derivative evaluation:

```python
def derivs(self, y: np.ndarray, u: np.ndarray, t: float) -> np.ndarray:
    p, v_b, q, omega, T = self._unpack(y)
    R = quat_to_dcm(q)

    # --- Query environment (single call per derivs) ---
    env_vars = self.env.sample_at_state(t, p)
    v_wind = env_vars['wind']
    rho = env_vars['rho']

    # --- Aerodynamic drag (uses wind + rho) ---
    v_rel = v_b - R.T @ v_wind
    F_aero = -0.5 * rho * np.linalg.norm(v_rel) * v_rel * self.Cd * self.A_proj
    # ...

    # --- Fin forces (uses rho) ---
    q_dyn = 0.5 * rho * V_exhaust**2
    # ...

    # --- Thrust density correction (uses rho) ---
    rho_ratio = rho / self.env.atmo_model.rho_ref
    T_eff = T * ground_effect(h, r_duct) * rho_ratio
    # ...
```

### 5.2 Force Models Affected by Environment

| Force Model | Environment Input | Effect |
|---|---|---|
| **Aero drag** (`aero_model.py`) | `v_wind`, `rho` | $\mathbf{v}_{rel} = \mathbf{v}_b - \mathbf{R}^T \mathbf{v}_{wind}$; $F_{drag} \propto \rho$ |
| **Fin lift/drag** (`fin_model.py`) | `rho` | $q_{dyn} = \frac{1}{2} \rho V_e^2$; all fin forces scale with $\rho$ |
| **EDF thrust** (`thrust_model.py`) | `rho` | $T_{eff} \propto \rho / \rho_{ref}$ (mass flow correction) |
| **Aerodynamic torque** | `v_wind`, `rho` | Via $\mathbf{F}_{aero}$ cross product with moment arm |
| **Fin torques** | `rho` | Via $\mathbf{F}_{fin}$ cross product with moment arm |

### 5.3 EDF Thrust Density Correction

EDF fans are volumetric flow devices — thrust scales with inlet air density:

$$
T_{corrected} = T_{raw} \cdot \frac{\rho}{\rho_{ref}}
$$

where $\rho_{ref} = 1.225$ kg/m³ (ISA sea-level). This is a first-order correction; full fan-curve shifting (pressure ratio vs. corrected mass flow) is overkill for the ~5% $\rho$ variation in training.

**Implementation**: Apply the density ratio in `thrust_model.py`'s `compute()` method, receiving $\rho$ as a parameter from the vehicle's `derivs()`.

### 5.4 Consistent $\rho$ Guarantee

**Critical design constraint**: Within a single `derivs()` call, all force models must use the same $\rho$ value. The `EnvironmentModel.sample_at_state()` returns $\rho$ once, and the vehicle passes it to each sub-model. No sub-model should independently query the atmosphere.

```
derivs(y, u, t):
    env = self.env.sample_at_state(t, p)   ← ONE call
    rho = env['rho']                        ← ONE value
    ├── aero_model.compute(..., rho=rho)    ← same rho
    ├── fin_model.compute(..., rho=rho)     ← same rho
    └── thrust_model.compute(..., rho=rho)  ← same rho
```

---

## 6. Domain Randomization Strategy

### 6.1 Purpose

Meta-RL training (per README.md) requires varied environments across episodes so the GTrXL-PPO policy learns to generalize. The environment model provides two axes of variation:

1. **Atmospheric conditions**: $T_{base}$, $P_{base}$ → varied $\rho$ affecting all aero forces and thrust.
2. **Wind conditions**: Mean wind vector, turbulence realization, gust events.

### 6.2 Randomization Protocol

At each `env.reset(seed)`:

| Parameter | Distribution | Range | Resampled Each Episode |
|---|---|---|---|
| $T_{base}$ | Uniform | $T_0 \pm 10$ K | Yes |
| $P_{base}$ | Uniform | $P_0 \pm 2000$ Pa | Yes |
| $\mathbf{v}_{mean}$ | Uniform (per-axis) | $[-10, +10]$ m/s (horiz), $[-2, +2]$ m/s (vert) | Yes |
| Turbulence seed | From master RNG | — | Yes (new noise realization) |
| Gust event | Bernoulli(0.1) | — | Yes |
| Gust amplitude | Uniform | 3–8 m/s | Yes (if gust active) |
| Gust onset | Uniform | $[0.2, 0.8] \cdot T_{ep}$ | Yes (if gust active) |
| Gust duration | Uniform | 0.5–2.0 s | Yes (if gust active) |

### 6.3 Curriculum Strategy (Optional)

> **Alignment**: Curriculum is optional (disabled by default) per [training.md §10](../Training%20Plan/training.md). Enable only if uniform DR fails to converge within 5M steps. The difficulty dimensions below align with training.md §10.2.

For training stability, gradually increase disturbance difficulty:

| Training Phase | Wind Range | $\Delta T$ | $\Delta P$ | Gust Prob | Sensor Noise Scale |
|---|---|---|---|---|---|
| Easy (0–30% steps) | 0–2 m/s | ±2 K | ±500 Pa | 0.0 | 0.5× |
| Medium (30–70%) | 0–5 m/s | ±5 K | ±1000 Pa | 0.05 | 0.75× |
| Hard (70–100%) | 0–10 m/s | ±10 K | ±2000 Pa | 0.1 | 1.0× |

Additional domain randomization dimensions managed by the training pipeline (see [training.md §6.2](../Training%20Plan/training.md)):
- **Actuator delay DR** (Hwangbo et al., 2017): ESC delay 10–40 ms, servo delay 5–20 ms, randomized per-episode.
- **Observation latency augmentation**: 0–2 policy steps of stale observation, forcing robustness to real-world compute latency.

This is optional — vanilla uniform randomization may suffice if the RL algorithm handles it. Implement as a config flag: `curriculum: true/false`.

### 6.4 Reproducibility

All randomization flows from a single master seed:

```
master_seed ──> np.random.default_rng(master_seed)
                ├── env.reset(episode_seed)
                │   ├── wind_model.reset()    (uses env.rng)
                │   │   ├── mean_wind sample
                │   │   ├── gust sample
                │   │   └── dryden noise sequence
                │   └── atmo_model.reset()    (uses env.rng)
                │       ├── T_base sample
                │       └── P_base sample
                └── vehicle.reset(initial_state, episode_seed)
                    └── mass randomization (if enabled)
```

Given the same `episode_seed`, the environment produces identical wind and atmosphere sequences. This is **mandatory** for RL training reproducibility and debugging.

---

## 7. Configuration Schema

### 7.1 YAML Structure

```yaml
environment:
  seed: 42

  atmosphere:
    T_base: 288.15          # K, ISA sea-level temperature
    T_lapse: -0.0065        # K/m, tropospheric lapse rate
    P_base: 101325.0        # Pa, ISA sea-level pressure
    rho_ref: 1.225          # kg/m³, reference density for thrust scaling
    randomize_T: 10.0       # ±K, per-episode domain randomization
    randomize_P: 2000.0     # ±Pa, per-episode domain randomization

  wind:
    dt: 0.005               # s, integration timestep (must match vehicle dt)
    V_ref: 10.0             # m/s, reference airspeed for Dryden filter design
    mean_vector_range_lo: [-10.0, -10.0, -2.0]   # m/s, [N, E, D] min
    mean_vector_range_hi: [10.0, 10.0, 2.0]      # m/s, [N, E, D] max
    turbulence_intensity: 0.3     # Dryden sigma_w scaling
    gust_prob: 0.1                # probability of gust event per episode
    gust_magnitude_range: [3.0, 8.0]   # m/s, amplitude bounds
    episode_duration: 15.0        # s, for gust timing bounds (matches training.md max_episode_time)

  curriculum:
    enabled: false                # per training.md §10 — enable only if uniform DR fails at 5M steps
    ramp_fraction: 0.7            # fraction of training to reach full difficulty
    phases:
      - fraction: 0.3
        wind_range: [0, 2]        # aligned with training.md §10.2 "Easy"
        rand_T: 2.0
        rand_P: 500.0
        gust_prob: 0.0
        sensor_noise_scale: 0.5
      - fraction: 0.7
        wind_range: [0, 5]        # aligned with training.md §10.2 "Medium"
        rand_T: 5.0
        rand_P: 1000.0
        gust_prob: 0.05
        sensor_noise_scale: 0.75
      - fraction: 1.0
        wind_range: [0, 10]       # aligned with training.md §10.2 "Hard"
        rand_T: 10.0
        rand_P: 2000.0
        gust_prob: 0.1
        sensor_noise_scale: 1.0
```

### 7.2 Config Loading

```python
import yaml

def load_environment_config(path: str) -> dict:
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg['environment']
```

---

## 8. Validation and Testing

### 8.1 Unit Tests — AtmosphereModel

| Test | Method | Pass Criterion |
|---|---|---|
| ISA sea-level density | `get_conditions(h=0)` with no randomization ($\Delta T = \Delta P = 0$) | $\rho = 1.225$ kg/m³ (±0.001) |
| ISA at 1000 m | `get_conditions(h=1000)` | $T = 281.65$ K, $\rho \approx 1.112$ kg/m³ (match ISA table ±0.1%) |
| Pressure formula consistency | Verify $P(h) / (R \cdot T(h)) = \rho(h)$ | Identity holds to machine precision |
| Randomization bounds | Reset 10,000 times, check $T_{base} \in [T_0 - \Delta T,\, T_0 + \Delta T]$ | All samples within bounds |
| Density range | With $\Delta T = 10$, $\Delta P = 2000$: check $\rho \in [1.16, 1.30]$ | Matches analytical bounds from §4.5 |
| Zero randomization | Set $\Delta T = \Delta P = 0$: all resets produce identical output | $\rho = 1.225$ every time |

### 8.2 Unit Tests — WindModel

| Test | Method | Pass Criterion |
|---|---|---|
| Mean wind bounds | Reset 10,000 times, check each axis | All within configured range |
| Zero wind | Set all ranges to $[0, 0]$, turbulence = 0, gust_prob = 0 | $\|\mathbf{v}_{wind}\| = 0$ for all $t$ |
| Seeded reproducibility | Same seed → same wind sequence over 100 steps | Bit-exact match |
| Different seeds produce different sequences | Two seeds → different wind at $t = 1$ s | At least one component differs |
| Gust timing | With gust_prob = 1.0: gust appears within $[0.2, 0.8] \cdot T_{ep}$ | onset within bounds |
| Gust magnitude | With gust_prob = 1.0: $\|\mathbf{g}_{amp}\| \in [3, 8]$ | Within configured range |
| Dryden statistics | Run 100,000 steps, compute variance of turbulence | $\sigma^2 \approx$ configured intensity² (±20%) |
| Altitude scaling | Compare turbulence variance at $h = 1$ m vs $h = 10$ m | $h = 1$ m has higher $\sigma$ (inverse altitude scaling) |

### 8.3 Unit Tests — EnvironmentModel

| Test | Method | Pass Criterion |
|---|---|---|
| Output dict keys | `sample_at_state(t, p)` returns `wind`, `rho`, `T`, `P` | All keys present |
| Output shapes | `wind` is `(3,)`, `rho`/`T`/`P` are scalars | Correct types and shapes |
| NED altitude conversion | Pass $p = [0, 0, -5]$ (NED) → $h = 5$ m | $h$ correctly computed |
| Reset determinism | `reset(seed=42)` twice → identical next sample | Bit-exact |

### 8.4 Integration Tests (with Vehicle Dynamics)

| Test | Setup | Expected Behavior |
|---|---|---|
| Free fall + wind | $T_{thrust} = 0$, constant 5 m/s crosswind | Lateral drift matches analytical: $\Delta x = \frac{1}{2} \frac{F_{drag}}{m} t^2$ |
| Hover + density variation | Trim thrust for ISA; change $\rho$ by +5% | Vehicle accelerates upward (more thrust from density, drag changes) |
| Hover + gust | Trim at hover; inject step gust at $t = 3$ s | Immediate lateral force onset; measurable $v_b$ change within 0.1 s |
| Density on fins | Max fin deflection; compare fin force at $\rho = 1.16$ vs $\rho = 1.30$ | Force ratio = $1.30 / 1.16 = 1.121$ (±1%) |
| Thrust density correction | Fixed $\Omega_{fan}$; compare $T_{eff}$ at two densities | $T_{eff}$ ratio matches $\rho / \rho_{ref}$ ratio |
| Full episode reproducibility | Same master seed: run 2 episodes, compare full trajectories | Bit-exact state sequences |

### 8.5 Ablation Tests (Post-Training)

These determine whether atmosphere features earn their keep:

| Ablation | Method | Delete If |
|---|---|---|
| Temperature randomization | Train with $\Delta T = 0$ vs $\Delta T = 10$ | Landing CEP difference < 0.01 m |
| Pressure randomization | Train with $\Delta P = 0$ vs $\Delta P = 2000$ | Landing CEP difference < 0.01 m |
| Discrete gusts | Train with gust_prob = 0 vs 0.1 | Robustness margin unchanged within noise |
| Dryden turbulence | Replace with Gaussian white noise | Success rate identical (±1%) |
| EDF density correction | Remove $\rho / \rho_{ref}$ from thrust | Landing $\Delta v$ difference < 0.01 m/s |

**If an ablation shows <0.01 m CEP improvement, delete the feature.** Lean sim trains faster.

---

## 9. File Map and Dependencies

### 9.1 Planned File Structure

```
simulation/
├── environment/
│   ├── __init__.py
│   ├── environment_model.py      # EnvironmentModel (top-level, owns sub-models)
│   ├── wind_model.py             # WindModel + DrydenFilter (mean + turbulence + gusts)
│   └── atmosphere_model.py       # AtmosphereModel (ISA + randomization)
├── configs/
│   ├── default_environment.yaml  # Default atmosphere + wind config
│   └── test_environment.yaml     # Simplified config for unit tests
└── tests/
    ├── test_wind_model.py
    ├── test_atmosphere_model.py
    └── test_environment_model.py  # Integration tests
```

### 9.2 Dependencies

```
numpy >= 1.24       # Core numerics (RNG, array ops)
pyyaml >= 6.0       # Config loading
pytest >= 7.0       # Testing
```

Optional (for GPU-accelerated batch training):
```
jax >= 0.4          # vmap for vectorized env batching
jaxlib >= 0.4
```

### 9.3 External Interface

The environment module exports a single entry point consumed by the vehicle:

```python
# In simulation/environment/__init__.py
from .environment_model import EnvironmentModel
```

```python
# In simulation/dynamics/vehicle.py
from simulation.environment import EnvironmentModel

class VehicleDynamics:
    def __init__(self, vehicle_config: dict, env: EnvironmentModel):
        self.env = env
        # ... (vehicle init, no wind model here)
```

---

## 10. Implementation Phases

> **Alignment**: This section maps to the 3-phase simulation master plan. See [master_plan.md](../master_plan.md) for the full training strategy. The environment module is built during the **Pre-Phase 1 core module build** and consumed by the Gymnasium wrapper in **Phase 1** and the OIGE task in **Phase 2** (conditional).

### 10.1 Phase Overview

| Phase | Files | Dependency | Est. Effort | Master Plan Phase |
|---|---|---|---|---|
| 1 | `atmosphere_model.py` + tests | None | 0.5 day | Pre-Phase 1 |
| 2 | `wind_model.py` (DrydenFilter + WindModel) + tests | None | 1.5 days | Pre-Phase 1 |
| 3 | `environment_model.py` (assembly) + integration tests | Phases 1–2 | 0.5 day | Pre-Phase 1 |
| 4 | Config YAML files | Phase 3 | 0.5 day | Pre-Phase 1 |
| 5 | Vehicle integration | Phase 3 + vehicle.md | 1 day | Pre-Phase 1 |
| 6 | Vectorization (JAX `vmap` or batched NumPy) | Phase 3 | 1–2 days | Phase 1 (training) |
| 7 | Isaac OIGE environment adapter (conditional) | Phase 3 | 1–2 days | Phase 2 (conditional) |
| 8 | Ablation tests (post-training) | Full pipeline | 1–2 days | Phase 3 (evaluation) |
| **Total** | | | **~6–10 days** | |

### 10.2 Phase 1: AtmosphereModel (0.5 Day)

- Implement `AtmosphereModel` class per §4.6.
- Unit tests per §8.1: ISA validation, randomization bounds, consistency checks.
- **Exit criterion**: `get_conditions(0)` returns $\rho = 1.225$ kg/m³ with zero randomization.

### 10.3 Phase 2: WindModel (1.5 Days)

- Implement `DrydenFilter` class per §3.3.4.
- Implement `WindModel` class per §3.5 (mean + turbulence + gusts).
- Unit tests per §8.2: bounds, reproducibility, statistics, altitude scaling.
- **Exit criterion**: Seeded wind sequence is bit-exact across runs; turbulence variance matches configured intensity within 20%.

### 10.4 Phase 3: EnvironmentModel Assembly (0.5 Day)

- Implement `EnvironmentModel` per §2.2, composing WindModel and AtmosphereModel.
- Unit tests per §8.3: dict structure, NED conversion, reset determinism.
- **Exit criterion**: `sample_at_state(t, p)` returns correct dict with all keys.

### 10.5 Phase 4: Configuration Files (0.5 Day)

- Write `default_environment.yaml` per §7.1.
- Write `test_environment.yaml` (zero wind, zero randomization — for deterministic vehicle tests).
- Validate YAML loading with `load_environment_config()`.

### 10.6 Phase 5: Vehicle Integration (1 Day)

- **Migrate**: Remove `WindModel` from `vehicle.py`'s init and class hierarchy.
- **Inject**: `VehicleDynamics.__init__` accepts an `EnvironmentModel` instance.
- **Update `derivs()`**: Call `self.env.sample_at_state(t, p)` once; pass `rho` and `wind` to `aero_model`, `fin_model`, `thrust_model`.
- **Thrust model**: Add `rho` parameter to `compute()` for density correction.
- **Fin model**: Add `rho` parameter to `compute()` replacing hardcoded density.
- **Config**: Remove `wind` and `aero.rho` sections from `default_vehicle.yaml` (now in `default_environment.yaml`).
- **Tests**: Re-run all vehicle integration tests with a deterministic `EnvironmentModel` (zero wind, ISA density).
- **Exit criterion**: All existing vehicle tests pass unchanged when using deterministic env.

### 10.7 Phase 6: Vectorization for Parallel Training (1–2 Days)

> **Context**: Master Plan Phase 1 requires batched environment execution. The EnvironmentModel must support N parallel environments with shapes `(N, 3)` for wind and `(N,)` for scalars. See [master_plan.md §3](../master_plan.md).

#### 10.7.1 Batched Interface

```python
class EnvironmentModel:
    # ... existing interface ...

    def sample_at_state_batched(self, t: np.ndarray, p: np.ndarray) -> dict:
        """Query environment for N parallel envs.

        Args:
            t: (N,) time array
            p: (N, 3) inertial position array (NED)

        Returns:
            dict with keys:
                'wind':  (N, 3) inertial wind velocity [m/s]
                'rho':   (N,) air density [kg/m³]
                'T':     (N,) temperature [K]
                'P':     (N,) pressure [Pa]
        """
        h = -p[:, 2]  # altitude for all envs
        wind = self.wind_model.sample_batched(t, h)      # (N, 3)
        T, P, rho = self.atmo_model.get_conditions_batched(h)  # each (N,)
        return {'wind': wind, 'rho': rho, 'T': T, 'P': P}
```

#### 10.7.2 Vectorization Strategy by Platform

| Platform | Method | Per-Env RNG | Notes |
|---|---|---|---|
| **NumPy (SB3 vec envs)** | N separate `EnvironmentModel` instances in `SubprocVecEnv` | Each has own `np.random.Generator` | Simplest. Ryzen 9 9900X 12C/24T handles 12–24 workers comfortably. |
| **JAX `vmap`** | Single batched function, `jax.random.split` for per-env keys | `jax.random.split(key, N)` | RTX 5070 Blackwell (6144 CUDA cores, 672 GB/s) pushes ~1–4M steps/sec. Requires rewriting Dryden filter in JAX (no Python control flow). |
| **Isaac OIGE** | GPU tensor operations in `pre_physics_step` callback | Per-env seeds from OIGE task reset | See Phase 7 below. RTX 5070 12 GB GDDR7 supports num_envs up to ~512–1024 headless. |

#### 10.7.3 JAX Compatibility Notes

For JAX `vmap` over the wind model:
- Replace `np.random.Generator` with `jax.random.PRNGKey` + `jax.random.split`
- Replace `DrydenFilter.step()` state mutation with functional state passing: `new_state, turb = dryden_step(state, h, key)`
- Replace `if` guards (gust timing) with `jnp.where()` for JIT compatibility
- All array shapes must be static — no dynamic allocation in `sample()`

**Effort estimate**: 1–2 days for JAX port of `WindModel` + `AtmosphereModel`. Only do this if SB3 `SubprocVecEnv` throughput on Ryzen 9 9900X (12–24 workers) is insufficient (< 100k steps/hour). The RTX 5070's Blackwell compute makes JAX `vmap` highly attractive if the port is needed.

### 10.8 Phase 7: Isaac OIGE Environment Adapter (1–2 Days, Conditional)

> **Entry criteria**: Only if Master Plan Phase 2 is triggered. See [master_plan.md §4.2](../master_plan.md).

In Isaac mode, the EnvironmentModel runs identically but in GPU-batched tensor form:

```python
class IsaacEnvironmentAdapter:
    """Wraps EnvironmentModel for OIGE pre_physics_step callbacks.

    Runs wind and atmosphere models on GPU tensors for num_envs parallel environments.
    """

    def __init__(self, config: dict, num_envs: int, device: str = 'cuda'):
        self.num_envs = num_envs
        self.device = device
        # Per-env atmosphere base conditions (randomized at reset)
        self.T_base = torch.full((num_envs,), 288.15, device=device)
        self.P_base = torch.full((num_envs,), 101325.0, device=device)
        # Per-env mean wind (randomized at reset)
        self.mean_wind = torch.zeros((num_envs, 3), device=device)
        # Dryden filter states (per-env)
        self.dryden_state = torch.zeros((num_envs, 5), device=device)

    def reset(self, env_ids: torch.Tensor):
        """Reset atmosphere and wind for specified environment indices."""
        n = len(env_ids)
        self.T_base[env_ids] = 288.15 + (torch.rand(n, device=self.device) * 2 - 1) * 10.0
        self.P_base[env_ids] = 101325.0 + (torch.rand(n, device=self.device) * 2 - 1) * 2000.0
        self.mean_wind[env_ids] = torch.rand((n, 3), device=self.device) * 20 - 10  # ±10 m/s
        self.mean_wind[env_ids, 2] *= 0.2  # ±2 m/s vertical
        self.dryden_state[env_ids] = 0.0

    def sample(self, t: torch.Tensor, p: torch.Tensor) -> dict:
        """Batched environment query for all num_envs. Returns GPU tensors."""
        h = -p[:, 2]  # altitude
        T = self.T_base + (-0.0065) * h
        P = self.P_base * (T / self.T_base) ** 5.256
        rho = P / (287.058 * T)
        wind = self.mean_wind + self._dryden_step(h)
        return {'wind': wind, 'rho': rho, 'T': T, 'P': P}
```

**PhysX vs. Python wind injection**: In Isaac mode, wind forces are applied as external force callbacks, not through PhysX aerodynamics. This ensures the Dryden model is identical to Python — PhysX has no built-in Dryden turbulence.

**Deletion criterion**: If Isaac parallelism does not improve training throughput by > 20% over JAX `vmap`, delete the Isaac environment adapter and use JAX for all parallel training.

### 10.9 Phase 8: Ablation Tests (1–2 Days, Post-Training)

> **Context**: Master Plan Phase 3. See [master_plan.md §5](../master_plan.md).

- Train PPO with and without each atmosphere/wind feature.
- Measure CEP, success rate, robustness margin.
- **Delete any feature that doesn't improve CEP by > 0.01 m.**

| Ablation | Method | Delete If |
|---|---|---|
| Temperature randomization | Train with $\Delta T = 0$ vs $\Delta T = 10$ | Landing CEP difference < 0.01 m |
| Pressure randomization | Train with $\Delta P = 0$ vs $\Delta P = 2000$ | Landing CEP difference < 0.01 m |
| Discrete gusts | Train with gust_prob = 0 vs 0.1 | Robustness margin unchanged within noise |
| Dryden turbulence | Replace with Gaussian white noise | Success rate identical (±1%) |
| EDF density correction | Remove $\rho / \rho_{ref}$ from thrust | Landing $\Delta v$ difference < 0.01 m/s |
| Isaac sensor emulation (if Phase 2) | Compare Isaac structured noise vs. Python Gaussian | Success rate difference < 2% |
| Isaac contact physics (if Phase 2) | Compare Isaac PhysX touchdown vs. Python spring-damper | Touchdown velocity r < 0.95 over 100 episodes |

**If an ablation shows < 0.01 m CEP improvement, delete the feature.** Lean sim trains faster.

---

## Appendix A: Sensitivity Analysis — $\rho$ Impact on Forces

For reference, the impact of density variation on each force model:

| $\rho$ (kg/m³) | $\Delta\rho$ (%) | $\Delta F_{drag}$ (%) | $\Delta F_{fin}$ (%) | $\Delta T_{eff}$ (%) |
|---|---|---|---|---|
| 1.160 (hot, low-P) | -5.3% | -5.3% | -5.3% | -5.3% |
| 1.225 (ISA ref) | 0% | 0% | 0% | 0% |
| 1.294 (cold, high-P) | +5.6% | +5.6% | +5.6% | +5.6% |

All aero forces scale linearly with $\rho$. A 5% density swing at hover means the thrust-to-weight ratio shifts by 5% — the controller must compensate by adjusting throttle, or the vehicle drifts vertically by $\Delta a \approx 0.5$ m/s² (order-of-magnitude, since $T/W \approx 1.3$ at hover).

---

## Appendix B: Dryden Model — Quick Reference

MIL-F-8785C / MIL-HDBK-1797 Dryden turbulence parameters for low altitude ($h < 1000$ ft):

| Parameter | Formula | Notes |
|---|---|---|
| $L_u = L_v$ | $h / (0.177 + 0.000823\,h)^{1.2}$ | Scale lengths (m), altitude-dependent |
| $L_w$ | $h$ | Vertical scale length = altitude |
| $\sigma_w$ | $0.1 \cdot W_{20}$ | Vertical turbulence intensity |
| $\sigma_u = \sigma_v$ | $\sigma_w / (0.177 + 0.000823\,h)^{0.4}$ | Horizontal turbulence intensity |
| $W_{20}$ | Wind speed at 20 ft (~6 m) | Estimated from mean wind config |

---

## Appendix C: Decisions Log

| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| Environment vs. embedded in vehicle | Separate `EnvironmentModel` module | Wind/atmo inside vehicle.py | Separation of concerns: atmosphere affects multiple force models; domain randomization is env-level, not vehicle-level |
| Humidity | Deleted | Virtual temperature correction | <2% effect on $\rho$; indoor tests; adds state without proportional fidelity |
| Wind shear | Deleted | Linear gradient model | $\Delta v < 0.5$ m/s over 10 m altitude; Dryden captures stochastic equivalent |
| Coriolis | Deleted | Earth rotation correction | $\sim 10^{-4}$ m/s² — five orders of magnitude below $g$ |
| Turbulence model | Dryden spectrum | von Kármán, white noise, constant | Dryden: rational transfer functions (easy to discretize), MIL-F-8785C standard, captures realistic PSD |
| Gust model | Step function (rare) | 1-cosine, ramp | Step is worst-case; ramps are trivially easier for controller |
| Atmosphere randomization | Uniform $\delta T$, $\delta P$ per episode | Gaussian, time-varying | Uniform is sufficient; time-varying weather in 10 s is unrealistic |
| Altitude lapse | Kept (negligible cost) | Deleted | Free compute; correct by default even if <0.12% effect at test altitudes |
| EDF density correction | $T \propto \rho / \rho_{ref}$ | Ignored, full fan-curve shift | First-order correction captures dominant effect; full fan curves are overkill for ~5% $\rho$ range |
| Config format | YAML (separate from vehicle) | Merged into vehicle YAML | Independent iteration on env params; separate config files for separate concerns |
| Curriculum | Optional flag | Mandatory, deleted | Let training pipeline decide; env model supports it but doesn't enforce it |
| Phase 1 vectorization | SB3 `SubprocVecEnv` on Ryzen 9 9900X 12C/24T (default) or JAX `vmap` on RTX 5070 (if throughput-limited) | Isaac OIGE from start | Lower VRAM, faster setup; Ryzen 12C + Blackwell compute sufficient for 1e7 steps. See [master_plan.md §3](../master_plan.md). |
| Isaac wind injection | Force callback (identical Dryden model) | Isaac built-in aero | PhysX has no built-in Dryden turbulence; custom callback ensures parity with Python. |
| Isaac sensor noise | Conditional ablation (Phase 2) | Isaac sensors from start | Gaussian σ sufficient for indoor tests. Isaac adds occlusion/lighting but likely < 2% impact. |
| Isaac environment adapter | GPU tensor (torch), conditional Phase 2 | Full Isaac physics for atmosphere | Atmosphere is trivial compute; no need for PhysX. Custom adapter ensures identical DR to Python. |

---

## Appendix D: Notation Reference

| Symbol | Meaning | Units |
|---|---|---|
| $\mathbf{v}_{wind}$ | Total wind velocity (inertial NED) | m/s |
| $\mathbf{v}_{mean}$ | Episode-constant mean wind | m/s |
| $\mathbf{v}_{turb}$ | Dryden turbulence component | m/s |
| $\mathbf{v}_{gust}$ | Discrete gust event | m/s |
| $T$ | Air temperature | K |
| $P$ | Air pressure | Pa |
| $\rho$ | Air density | kg/m³ |
| $T_0, P_0, \rho_0$ | ISA sea-level reference values | K, Pa, kg/m³ |
| $\lambda$ | Temperature lapse rate | K/m |
| $R$ | Specific gas constant (dry air) | J/(kg·K) |
| $L_u, L_v, L_w$ | Dryden turbulence scale lengths | m |
| $\sigma_u, \sigma_v, \sigma_w$ | Dryden turbulence intensities | m/s |
| $W_{20}$ | Wind speed at 20 ft reference height | m/s |
| $h$ | Altitude above ground | m |
| $\delta T, \delta P$ | Per-episode randomization offsets | K, Pa |
| $\mathbf{g}_{amp}$ | Gust amplitude vector | m/s |
