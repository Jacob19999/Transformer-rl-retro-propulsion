# Vehicle Dynamics — Implementation Plan

> **Scope**: 6-DOF rigid-body plant model for an EDF drone emulating TVC rocket landings.
> Post-slosh deletion. Fixed mass properties (computed once at init). RK4 fixed-step integration.

---

## Table of Contents

1. [Design Rationale](#1-design-rationale)
2. [State Vector and Kinematics](#2-state-vector-and-kinematics)
3. [Translational Dynamics](#3-translational-dynamics)
4. [Rotational Dynamics](#4-rotational-dynamics)
5. [Primitive-Based Mass Properties](#5-primitive-based-mass-properties)
6. [Force and Torque Models](#6-force-and-torque-models)
7. [RK4 Fixed-Step Integrator](#7-rk4-fixed-step-integrator)
8. [Implementation Architecture](#8-implementation-architecture)
9. [Configuration Schema](#9-configuration-schema)
10. [Validation and Testing](#10-validation-and-testing)
11. [Isaac Sim Integration Notes](#11-isaac-sim-integration-notes)
12. [File Map and Dependencies](#12-file-map-and-dependencies)

---

## 1. Design Rationale

### 1.1 Why Delete Slosh

Fuel slosh is a secondary disturbance for an electric drone: there is no propellant burn-down, mass is fixed, and the ODE states required for slosh (pendulum/spring-mass models) bloat integration time by 20–50% without proportional fidelity gains for short ~10 s episodes. Deleting slosh removes those extra states and the coupling logic from the derivative function.

### 1.2 Why Fixed Mass Properties (Init-Only)

Since masses are fixed for the entire episode:
- **Delete** dynamic CoM/MoI updates from the sim loop entirely.
- **Precompute** total mass \(m\), center of mass \(\mathbf{c}_{cm}\), inertia tensor \(\mathbf{I}\), and its inverse \(\mathbf{I}^{-1}\) once at init.
- **Cache** these as immutable scalars/matrices for the episode.
- **Payoff**: Eliminates a 3×3 matrix inversion per step, accelerating RL training by ~10–15%.

Modularity is retained: mass properties are still defined via geometric primitives (cylinder for duct, box for Jetson, etc.) in a config file, so design iteration (e.g., tweak EDF cylinder dims) triggers a one-time recompute without touching dynamics code.

### 1.3 Why RK4 Fixed-Step

| Consideration | RK4 Fixed-Step | RK45 Adaptive | RK8 |
|---|---|---|---|
| Evaluations per step | 4 | 6+ (variable) | 8 |
| Determinism (RL) | Yes — fixed dt | No — variable dt breaks reproducibility | Yes |
| Accuracy (subsonic) | Sufficient | Overkill | Overkill |
| Stiffness handling | dt=0.005 s handles gyro ~10⁴ rpm | Adaptive overhead not justified | ~2× overhead for negligible gain |

**Decision**: RK4 with dt = 0.005–0.01 s. Fixed-step guarantees reproducibility for RL training. The small dt handles stiffness from high-RPM gyroscopic rates without instability.

### 1.4 Aerodynamics: Keep It Simple

- Combined-shape drag is sufficient for wind disturbance fidelity; full CFD would destroy training speed.
- Fin forces use thin-airfoil approximation (\(C_L = 2\pi\alpha\)) — valid for low-Mach jet (~60 m/s) and small deflection angles (<15°). Full NACA airfoil lookup tables are unnecessary.
- Wind turbulence via Dryden spectrum (seeded for reproducibility).

### 1.5 Non-Negotiable Terms

- **Gyroscopic precession**: Spinning fan at ~10⁴ rad/s produces significant precession torques. Omitting this causes unexplained yaw/pitch drift.
- **Motor thrust lag**: 1st-order dynamics (\(\tau_{motor} \approx 0.1\) s). Unmodeled lag causes 0.5–1 m touchdown errors.
- **Ground effect**: Thrust multiplier near the pad. Without it, the controller sees a phantom deceleration near touchdown.
- **Motor reaction torque**: Small but prevents yaw drift accumulation.

---

## 2. State Vector and Kinematics

### 2.1 State Vector (13 Scalars)

Minimal for 6-DOF with quaternion orientation. No slosh states, no dynamic mass states.

| Symbol | Dimension | Frame | Description |
|---|---|---|---|
| \(\mathbf{p} = [x, y, z]^T\) | 3 | Inertial (NED) | Position (m) |
| \(\mathbf{v}_b = [u, v, w]^T\) | 3 | Body | Velocity (m/s) |
| \(\mathbf{q} = [q_0, q_1, q_2, q_3]^T\) | 4 | — | Orientation quaternion (scalar-first, unit norm) |
| \(\boldsymbol{\omega} = [p, q, r]^T\) | 3 | Body | Angular velocity (rad/s) |

**Flat array layout** (for integrator):

```
y[0:3]   = p     (inertial position, NED)
y[3:6]   = v_b   (body velocity)
y[6:10]  = q     (quaternion, scalar-first)
y[10:13] = omega  (body angular rate)
```

### 2.2 Frame Conventions

- **Inertial frame**: NED (North-East-Down). Origin at landing pad center. z-positive is downward — natural for descent/landing sim.
- **Body frame**: Forward-Right-Down (FRD). Origin at CoM. x-forward along drone longitudinal axis, z-down along thrust axis.
- Quaternion convention: scalar-first \([q_0, q_1, q_2, q_3]\), Hamilton product. Chosen over Euler angles to eliminate gimbal lock. Cheaper than rotation matrices for composition.

### 2.3 Direction Cosine Matrix (Body-to-Inertial)

\[
\mathbf{R}(\mathbf{q}) = \begin{bmatrix}
q_0^2 + q_1^2 - q_2^2 - q_3^2 & 2(q_1 q_2 - q_0 q_3) & 2(q_1 q_3 + q_0 q_2) \\
2(q_1 q_2 + q_0 q_3) & q_0^2 - q_1^2 + q_2^2 - q_3^2 & 2(q_2 q_3 - q_0 q_1) \\
2(q_1 q_3 - q_0 q_2) & 2(q_2 q_3 + q_0 q_1) & q_0^2 - q_1^2 - q_2^2 + q_3^2
\end{bmatrix}
\]

**Implementation note**: Compute \(\mathbf{R}\) once per derivative evaluation, reuse across position kinematics, gravity rotation, and wind rotation. Store as a 3×3 NumPy array; do not recompute redundantly.

### 2.4 Position Kinematics

\[
\dot{\mathbf{p}} = \mathbf{R}(\mathbf{q})\,\mathbf{v}_b
\]

Transforms body-frame velocity to inertial-frame position rate. Exact, no approximation.

### 2.5 Quaternion Kinematics

\[
\dot{\mathbf{q}} = \frac{1}{2}\,\mathbf{q} \otimes \begin{bmatrix} 0 \\ \boldsymbol{\omega} \end{bmatrix}
= \frac{1}{2} \begin{bmatrix}
-q_1 p - q_2 q - q_3 r \\
\phantom{-}q_0 p + q_2 r - q_3 q \\
\phantom{-}q_0 q + q_3 p - q_1 r \\
\phantom{-}q_0 r + q_1 q - q_2 p
\end{bmatrix}
\]

**Normalization policy**: Re-normalize \(\mathbf{q}\) to unit norm every 10 integration steps (not every step — saves divisions; drift is negligible over 10 × 0.005 s = 0.05 s). Implementation: `q /= np.linalg.norm(q)` with a step counter modulo 10.

---

## 3. Translational Dynamics

### 3.1 Newton's Second Law in a Rotating Body Frame

From Newton's second law, accounting for the rotating reference frame (Coriolis term):

\[
m\,(\dot{\mathbf{v}}_b + \boldsymbol{\omega} \times \mathbf{v}_b) = \mathbf{F}_b + m\,\mathbf{g}_b
\]

Solving for the state derivative:

\[
\boxed{\dot{\mathbf{v}}_b = \frac{\mathbf{F}_b}{m} + \mathbf{g}_b - \boldsymbol{\omega} \times \mathbf{v}_b}
\]

### 3.2 Gravity in Body Frame

\[
\mathbf{g}_b = \mathbf{R}(\mathbf{q})^T \begin{bmatrix} 0 \\ 0 \\ g \end{bmatrix}, \quad g = 9.81\ \text{m/s}^2
\]

Flat Earth assumption (delete curvature for <2 km altitude). In NED, gravity points in +z (downward). The transpose \(\mathbf{R}^T\) rotates from inertial to body frame.

### 3.3 Total Body-Frame Force

\[
\mathbf{F}_b = \mathbf{F}_{thrust} + \mathbf{F}_{aero} + \mathbf{F}_{fins}
\]

Each term is detailed in [Section 6](#6-force-and-torque-models).

### 3.4 Implementation Pseudocode

```python
def compute_vb_dot(self, v_b, omega, F_b, q):
    R = self.quat_to_dcm(q)
    g_b = R.T @ np.array([0.0, 0.0, self.g])
    coriolis = np.cross(omega, v_b)
    return F_b / self.mass + g_b - coriolis
```

---

## 4. Rotational Dynamics

### 4.1 Euler's Equation with Gyroscopic Precession

\[
\mathbf{I}\,\dot{\boldsymbol{\omega}} + \boldsymbol{\omega} \times (\mathbf{I}\,\boldsymbol{\omega}) + \boldsymbol{\omega} \times \mathbf{h}_{fan} = \boldsymbol{\tau}_b
\]

Solving for the state derivative:

\[
\boxed{\dot{\boldsymbol{\omega}} = \mathbf{I}^{-1}\left(\boldsymbol{\tau}_b - \boldsymbol{\omega} \times (\mathbf{I}\,\boldsymbol{\omega}) - \boldsymbol{\omega} \times \mathbf{h}_{fan}\right)}
\]

### 4.2 Fan Angular Momentum

\[
\mathbf{h}_{fan} = I_{fan}\,\Omega_{fan}\begin{bmatrix}0\\0\\1\end{bmatrix}
\]

| Parameter | Typical Value | Notes |
|---|---|---|
| \(I_{fan}\) | ~0.001 kg·m² | Rotor moment of inertia for 90 mm EDF |
| \(\Omega_{fan}\) | up to 10⁴ rad/s | Full-throttle RPM (~95,000 RPM) |

The cross product \(\boldsymbol{\omega} \times \mathbf{h}_{fan}\) generates precession torques. At high RPM, this term dominates over the standard \(\boldsymbol{\omega} \times (\mathbf{I}\boldsymbol{\omega})\) cross-coupling and is **critical for stability prediction**.

### 4.3 Inertia Inverse — Precomputed

\(\mathbf{I}^{-1}\) is computed once at init and cached. If the inertia tensor is diagonal (principal axes aligned with body frame), the inverse is trivial: \(\text{diag}(1/I_{xx}, 1/I_{yy}, 1/I_{zz})\). If off-diagonal terms exist (asymmetric mass distribution), compute the full 3×3 inverse at init.

**Decision heuristic**: If all off-diagonal inertia terms are <5% of the diagonals, zero them and use the diagonal inverse. Otherwise, use `np.linalg.inv()` once at init.

### 4.4 Total Body-Frame Torque

\[
\boldsymbol{\tau}_b = \boldsymbol{\tau}_{thrust} + \boldsymbol{\tau}_{aero} + \boldsymbol{\tau}_{fins} + \boldsymbol{\tau}_{motor}
\]

Each term is detailed in [Section 6](#6-force-and-torque-models).

### 4.5 Implementation Pseudocode

```python
def compute_omega_dot(self, omega, tau_b, omega_fan):
    h_fan = np.array([0.0, 0.0, self.I_fan * omega_fan])
    gyro_body = np.cross(omega, self.I @ omega)
    gyro_fan = np.cross(omega, h_fan)
    return self.I_inv @ (tau_b - gyro_body - gyro_fan)
```

---

## 5. Primitive-Based Mass Properties

### 5.1 Overview

Mass properties are aggregated from N geometric primitives defined in a YAML config file. Each primitive specifies shape, mass, local CoM position, and orientation. Aggregation runs **once at episode init**; results are cached as immutable attributes.

### 5.2 Per-Primitive Parameters

Each primitive \(i\) provides:
- Mass: \(m_i\) (kg)
- Local CoM position: \(\mathbf{c}_i\) (m, in body frame)
- Local inertia about own CoM: \(\mathbf{I}_i\) (3×3 tensor, kg·m²)
- Optional orientation: \(\mathbf{R}_i\) (rotation matrix, for non-axis-aligned primitives)

### 5.3 Primitive Inertia Formulas

**Cylinder** (aligned along z-axis, radius \(r\), height \(h\)):
\[
I_{xx} = I_{yy} = \frac{1}{12}\,m\,(3r^2 + h^2), \quad I_{zz} = \frac{1}{2}\,m\,r^2
\]

**Box** (dimensions \(a \times b \times c\)):
\[
I_{xx} = \frac{1}{12}\,m\,(b^2 + c^2), \quad I_{yy} = \frac{1}{12}\,m\,(a^2 + c^2), \quad I_{zz} = \frac{1}{12}\,m\,(a^2 + b^2)
\]

**Sphere** (radius \(r\)):
\[
I_{xx} = I_{yy} = I_{zz} = \frac{2}{5}\,m\,r^2
\]

If a primitive is oriented (rotated by \(\mathbf{R}_i\) relative to body frame), apply the similarity transform:
\[
\mathbf{I}_i^{body} = \mathbf{R}_i\,\mathbf{I}_i^{local}\,\mathbf{R}_i^T
\]

### 5.4 Aggregation Equations

**Total mass:**
\[
m = \sum_{i=1}^{N} m_i
\]

**Global center of mass:**
\[
\mathbf{c}_{cm} = \frac{1}{m}\sum_{i=1}^{N} m_i\,\mathbf{c}_i
\]

**Inertia about CoM** (parallel axis theorem):
\[
\mathbf{I} = \sum_{i=1}^{N}\left[\mathbf{I}_i + m_i\left((\mathbf{d}_i \cdot \mathbf{d}_i)\,\mathbf{1}_3 - \mathbf{d}_i\,\mathbf{d}_i^T\right)\right], \quad \mathbf{d}_i = \mathbf{c}_i - \mathbf{c}_{cm}
\]

where \(\mathbf{1}_3\) is the 3×3 identity matrix.

### 5.5 Post-Aggregation Actions

1. **Shift reference point**: All force application points (thrust, fins, center of pressure) are re-expressed relative to \(\mathbf{c}_{cm}\). This eliminates torque artifacts from an off-center CoM.
2. **Cache**: Store \(m\), \(\mathbf{c}_{cm}\), \(\mathbf{I}\), \(\mathbf{I}^{-1}\) as immutable instance attributes.
3. **Randomization** (optional, for robustness training): At episode init, apply ±10% perturbation to individual primitive masses before aggregation. This produces varied CoM/MoI across episodes for domain randomization.

### 5.6 Implementation Pseudocode

```python
def compute_mass_properties(self, primitives: list[dict]) -> None:
    """Aggregate mass, CoM, and inertia from primitive list. Call once at init."""
    total_mass = sum(p['mass'] for p in primitives)
    com = sum(p['mass'] * np.array(p['position']) for p in primitives) / total_mass

    I_total = np.zeros((3, 3))
    for p in primitives:
        I_local = self._primitive_inertia(p)          # shape-specific formula
        if 'orientation' in p:
            R_i = self._euler_to_dcm(p['orientation'])
            I_local = R_i @ I_local @ R_i.T           # rotate to body frame
        d = np.array(p['position']) - com              # offset from global CoM
        I_parallel = p['mass'] * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
        I_total += I_local + I_parallel

    self.mass = total_mass
    self.com = com
    self.I = I_total
    self.I_inv = np.linalg.inv(I_total)
```

---

## 6. Force and Torque Models

### 6.1 Thrust (Main Propulsive)

#### 6.1.1 Thrust Magnitude

Quadratic in fan speed:
\[
T = k\,\Omega_{fan}^2
\]

| Parameter | Value | Notes |
|---|---|---|
| \(k\) | ~10⁻⁶ N/(rad/s)² | Empirical, calibrate from EDF static test data |
| \(\Omega_{fan}\) | 0–10⁴ rad/s | Commanded via ESC PWM |

#### 6.1.2 Motor Lag (1st-Order Thrust Dynamics)

Thrust does not respond instantaneously to commands. Modeled as a 1st-order lag:
\[
\dot{T} = \frac{1}{\tau_{motor}}\,(T_{cmd} - T), \quad \tau_{motor} \approx 0.1\ \text{s}
\]

This adds **1 auxiliary state** to the integration (total: 14 scalars integrated). The lag is critical — without it, the controller overcompensates near touchdown and the sim-to-real gap widens by 0.5–1 m.

**Implementation**: Integrate \(T\) alongside the main state vector \(\mathbf{y}\). The commanded thrust \(T_{cmd}\) comes from the RL agent's action.

#### 6.1.3 Ground Effect

When altitude \(h < 2\,r_{duct}\) (within ~2 duct diameters of the pad):
\[
T_{effective} = T \cdot \left(1 + 0.5\,\left(\frac{r_{duct}}{h}\right)^2\right)
\]

Simple empirical approximation. Prevents phantom deceleration artifacts near touchdown. Clamp \(h \geq 0.01\) m to avoid division-by-zero.

#### 6.1.4 Thrust Force and Torque

\[
\mathbf{F}_{thrust} = T_{effective}\begin{bmatrix}0\\0\\1\end{bmatrix}, \quad \boldsymbol{\tau}_{thrust} = (\mathbf{r}_{thrust} - \mathbf{c}_{cm}) \times \mathbf{F}_{thrust}
\]

Thrust acts along the body z-axis (downward in FRD = upward in vehicle orientation). The torque arises if the thrust application point \(\mathbf{r}_{thrust}\) is offset from \(\mathbf{c}_{cm}\).

### 6.2 Aerodynamic Drag (Combined Shape)

#### 6.2.1 Relative Velocity

\[
\mathbf{v}_{rel} = \mathbf{v}_b - \mathbf{R}(\mathbf{q})^T\,\mathbf{v}_{wind}
\]

where \(\mathbf{v}_{wind}\) is the wind velocity in the inertial frame (e.g., gusts up to 10 m/s).

#### 6.2.2 Drag Force

\[
\mathbf{F}_{aero} = -\frac{1}{2}\,\rho\,\|\mathbf{v}_{rel}\|\,\mathbf{v}_{rel}\,C_d\,A_{proj}
\]

| Parameter | Value | Notes |
|---|---|---|
| \(\rho\) | 1.225 kg/m³ | Sea-level standard atmosphere |
| \(C_d\) | 0.5–1.0 | Bluff body; tune from wind tunnel or literature |
| \(A_{proj}\) | ~0.01 m² | Projected frontal area from primitive bounding box |

#### 6.2.3 Aerodynamic Torque

\[
\boldsymbol{\tau}_{aero} = (\mathbf{r}_{cp} - \mathbf{c}_{cm}) \times \mathbf{F}_{aero}
\]

Center of pressure \(\mathbf{r}_{cp}\) is approximately 0.05 m below CoM (aft, toward the duct). This is a small correction term.

#### 6.2.4 Wind Model (Dryden Turbulence)

Wind is sampled as a stochastic field at episode init for consistency:
- **Mean wind**: Constant vector \(\mathbf{v}_{wind,0}\) sampled uniformly per episode (0–10 m/s, random direction).
- **Turbulence**: Dryden spectrum-based fluctuations (3-axis, parameterized by altitude and intensity). Implemented as a coloring filter on white noise, seeded per episode for RL reproducibility.

```python
def sample_wind(self, t: float) -> np.ndarray:
    """Return inertial wind vector at time t. Seeded per episode."""
    turbulence = self.dryden_filter.step(self.rng.standard_normal(3))
    return self.wind_mean + turbulence
```

### 6.3 Fin Forces (TVC Emulation via NACA Airfoils in Exhaust)

#### 6.3.1 Per-Fin Geometry

4 fins arranged at 90° intervals around the exhaust duct, each with independent deflection \(\delta_k\) (rad), commanded by the RL agent.

#### 6.3.2 Effective Velocity and Angle of Attack

\[
V_e \approx 60\text{–}80\ \text{m/s (EDF exhaust velocity, proportional to RPM)}
\]
\[
\alpha_k = \delta_k + \beta_k
\]

where \(\beta_k\) is the local sideslip contribution (small, from body angular rates and lateral velocity).

#### 6.3.3 Thin-Airfoil Coefficients

Valid for \(|\alpha| < 15°\) (pre-stall):

\[
C_L = 2\pi\,\alpha_k
\]
\[
C_D = C_{D0} + \frac{C_L^2}{\pi\,AR}
\]

| Parameter | Value | Notes |
|---|---|---|
| \(C_{D0}\) | ~0.01 | Parasitic drag coefficient |
| AR | ~2 | Aspect ratio for short fins |

**Stall protection**: Clamp \(\alpha_k\) to ±15° in the coefficient computation. Beyond this, \(C_L\) should plateau/drop, but the RL agent should learn to avoid stall. Optionally add a smooth sigmoid transition:

```python
alpha_eff = max_alpha * np.tanh(alpha_k / max_alpha)  # soft clamp
```

#### 6.3.4 Per-Fin Force

\[
\mathbf{F}_{fin,k} = \frac{1}{2}\,\rho\,V_e^2\,A_{fin}\,(C_L\,\hat{\mathbf{n}}_{L,k} + C_D\,\hat{\mathbf{n}}_{D,k})
\]

| Parameter | Value | Notes |
|---|---|---|
| \(A_{fin}\) | ~0.005 m² | Fin planform area |
| \(\hat{\mathbf{n}}_{L,k}\) | Perpendicular to flow | Lift direction, depends on fin orientation |
| \(\hat{\mathbf{n}}_{D,k}\) | Parallel to flow | Drag direction, opposes exhaust velocity |

#### 6.3.5 Total Fin Contribution

\[
\mathbf{F}_{fins} = \sum_{k=1}^{4} \mathbf{F}_{fin,k}, \quad \boldsymbol{\tau}_{fins} = \sum_{k=1}^{4} (\mathbf{r}_{fin,k} - \mathbf{c}_{cm}) \times \mathbf{F}_{fin,k}
\]

Fin positions \(\mathbf{r}_{fin,k}\) are ~0.1 m aft of CoM, defined in the config.

### 6.4 Motor Reaction Torque

\[
\boldsymbol{\tau}_{motor} = -I_{fan}\,\dot{\Omega}_{fan}\begin{bmatrix}0\\0\\1\end{bmatrix}
\]

The fan acceleration \(\dot{\Omega}_{fan}\) is derived from the thrust lag state (\(\dot{T} = k \cdot 2\Omega_{fan}\dot{\Omega}_{fan}\), solved for \(\dot{\Omega}_{fan}\)). This is small but prevents yaw drift accumulation over the episode.

### 6.5 Force/Torque Summary Table

| Source | Force | Torque | Notes |
|---|---|---|---|
| Thrust | \(T_{eff}\,[0,0,1]^T\) | \((\mathbf{r}_{thrust} - \mathbf{c}_{cm}) \times \mathbf{F}_{thrust}\) | With lag and ground effect |
| Aero drag | \(-\frac{1}{2}\rho\|v_{rel}\|v_{rel}C_dA\) | \((\mathbf{r}_{cp}-\mathbf{c}_{cm})\times\mathbf{F}_{aero}\) | On combined shape |
| Fins (×4) | \(\frac{1}{2}\rho V_e^2 A_{fin}(C_L\hat{n}_L+C_D\hat{n}_D)\) | \((\mathbf{r}_{fin}-\mathbf{c}_{cm})\times\mathbf{F}_{fin}\) | Thin-airfoil, per fin |
| Motor reaction | — | \(-I_{fan}\dot{\Omega}_{fan}[0,0,1]^T\) | Prevents yaw drift |

---

## 7. RK4 Fixed-Step Integrator

### 7.1 Standard 4th-Order Runge-Kutta

Let \(\mathbf{y} = [\mathbf{p},\,\mathbf{v}_b,\,\mathbf{q},\,\boldsymbol{\omega},\,T]^T\) (14 scalars: 13 state + 1 thrust lag).
Let \(\dot{\mathbf{y}} = f(\mathbf{y}, \mathbf{u}, t)\) where \(\mathbf{u}\) = controls (thrust command, 4 fin deflections).

\[
\mathbf{k}_1 = dt \cdot f(\mathbf{y}_n,\,\mathbf{u},\,t_n)
\]
\[
\mathbf{k}_2 = dt \cdot f(\mathbf{y}_n + 0.5\,\mathbf{k}_1,\,\mathbf{u},\,t_n + 0.5\,dt)
\]
\[
\mathbf{k}_3 = dt \cdot f(\mathbf{y}_n + 0.5\,\mathbf{k}_2,\,\mathbf{u},\,t_n + 0.5\,dt)
\]
\[
\mathbf{k}_4 = dt \cdot f(\mathbf{y}_n + \mathbf{k}_3,\,\mathbf{u},\,t_n + dt)
\]
\[
\mathbf{y}_{n+1} = \mathbf{y}_n + \frac{1}{6}(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4)
\]

### 7.2 Time Step Selection

\[
dt = 0.005\ \text{s}
\]

Rationale:
- Gyroscopic rates up to ~10⁴ rad/s → highest dynamics frequency ~1.6 kHz → Nyquist requires dt < 0.0003 s **for resolving the oscillation**, but RK4 only needs dt small enough for **stability of the ODE integration**, not signal reconstruction. With the precession coupling (not the raw RPM), the effective stiffness is much lower.
- Empirical rule: dt < 1/(10 × highest eigenvalue of linearized system). Linearize at hover → check → adjust if needed.
- Training speed target: ~200× realtime on GPU with vectorized envs.

### 7.3 Quaternion Normalization

After the RK4 update, every 10 steps:

```python
if self.step_count % 10 == 0:
    q = y[6:10]
    y[6:10] = q / np.linalg.norm(q)
```

### 7.4 Controls Held Constant Over dt

The RL agent outputs actions at the policy frequency (e.g., 20 Hz = every 0.05 s = 10 integration steps at dt=0.005 s). Between policy steps, controls \(\mathbf{u}\) are held constant (zero-order hold). This is standard for RL + physics sim and matches real actuator behavior.

### 7.5 Implementation Pseudocode

```python
def step(self, u: np.ndarray, dt: float = 0.005) -> np.ndarray:
    """Advance state by one integration step using RK4."""
    y = self.state
    t = self.time

    k1 = dt * self.derivs(y, u, t)
    k2 = dt * self.derivs(y + 0.5 * k1, u, t + 0.5 * dt)
    k3 = dt * self.derivs(y + 0.5 * k2, u, t + 0.5 * dt)
    k4 = dt * self.derivs(y + k3, u, t + dt)

    self.state = y + (k1 + 2*k2 + 2*k3 + k4) / 6.0
    self.time += dt
    self.step_count += 1

    if self.step_count % 10 == 0:
        q = self.state[6:10]
        self.state[6:10] = q / np.linalg.norm(q)

    return self.state
```

---

## 8. Implementation Architecture

### 8.1 Class Hierarchy

```
VehicleDynamics                 (top-level, owns state and integrator)
├── MassProperties              (primitive aggregation, init-only)
├── ThrustModel                 (EDF thrust curve, lag, ground effect)
├── AeroModel                   (drag on combined shape)
├── FinModel                    (4x fin forces, thin-airfoil)
├── WindModel                   (Dryden turbulence, seeded)
└── Integrator                  (RK4 fixed-step)
```

### 8.2 Core Class: `VehicleDynamics`

```python
class VehicleDynamics:
    """6-DOF rigid-body plant model for EDF drone TVC landing sim."""

    def __init__(self, config: dict, seed: int = 0):
        # Mass properties (init-only, cached)
        self.mass_props = MassProperties(config['primitives'])
        self.mass = self.mass_props.total_mass
        self.I = self.mass_props.inertia
        self.I_inv = self.mass_props.inertia_inv
        self.com = self.mass_props.com

        # Force/torque sub-models
        self.thrust_model = ThrustModel(config['thrust'])
        self.aero_model = AeroModel(config['aero'])
        self.fin_model = FinModel(config['fins'])
        self.wind_model = WindModel(config['wind'], seed=seed)

        # Constants
        self.g = config.get('gravity', 9.81)
        self.dt = config.get('dt', 0.005)
        self.I_fan = config['thrust']['I_fan']

        # State: [p(3), v_b(3), q(4), omega(3), T(1)] = 14
        self.state = np.zeros(14)
        self.time = 0.0
        self.step_count = 0

    def reset(self, initial_state: np.ndarray, seed: int | None = None) -> np.ndarray:
        """Reset to initial conditions for new episode."""
        self.state = initial_state.copy()
        self.time = 0.0
        self.step_count = 0
        self.thrust_model.reset()
        if seed is not None:
            self.wind_model.reseed(seed)
        return self.state

    def derivs(self, y: np.ndarray, u: np.ndarray, t: float) -> np.ndarray:
        """Compute state derivatives. u = [T_cmd, delta_1..4]."""
        p, v_b, q, omega, T = self._unpack(y)
        T_cmd, fin_deltas = u[0], u[1:5]

        R = quat_to_dcm(q)

        # Forces and torques
        F_thrust, tau_thrust, T_dot, omega_fan = self.thrust_model.compute(
            T, T_cmd, p[2], R)
        F_aero, tau_aero = self.aero_model.compute(
            v_b, R, self.wind_model.sample(t))
        F_fins, tau_fins = self.fin_model.compute(
            fin_deltas, omega_fan, v_b, omega)
        tau_motor = self.thrust_model.reaction_torque(omega_fan, T_dot)

        F_total = F_thrust + F_aero + F_fins
        tau_total = tau_thrust + tau_aero + tau_fins + tau_motor

        # Derivatives
        p_dot = R @ v_b
        v_dot = F_total / self.mass + R.T @ np.array([0, 0, self.g]) \
                - np.cross(omega, v_b)
        q_dot = 0.5 * quat_mult(q, np.array([0, *omega]))
        omega_dot = self.I_inv @ (tau_total
                                   - np.cross(omega, self.I @ omega)
                                   - np.cross(omega, np.array([0, 0, self.I_fan * omega_fan])))

        return np.concatenate([p_dot, v_dot, q_dot, omega_dot, [T_dot]])

    def step(self, u: np.ndarray) -> np.ndarray:
        """RK4 integration step."""
        # ... (as in Section 7.5)

    def _unpack(self, y):
        return y[0:3], y[3:6], y[6:10], y[10:13], y[13]
```

### 8.3 Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Language | Python (NumPy) | Fast prototyping, Isaac Sim API is Python, vectorizable with JAX later |
| Config format | YAML | Human-readable, easy to edit primitives/params, standard in ML pipelines |
| State packing | Flat np.ndarray | RK4 operates on a single vector; avoids dict overhead in inner loop |
| Sub-model pattern | Composition | Each force model is independently testable and swappable |
| Randomization | At `reset()` | Episode-level domain randomization for RL robustness training |

### 8.4 Performance Considerations

- **Inner loop** (`derivs`): Called 4× per RK4 step. Must be lean. No allocations (preallocate scratch arrays). No Python loops over fins — vectorize with shape (4, 3) arrays.
- **Vectorized envs**: For RL training, batch N environments. Replace all `np` calls with batched equivalents (shape `(N, 14)` state). Consider JAX `vmap` for GPU acceleration.
- **Profiling target**: Single `derivs` call < 10 μs (NumPy), < 1 μs (JAX/GPU).

---

## 9. Configuration Schema

### 9.1 YAML Structure

```yaml
vehicle:
  gravity: 9.81
  dt: 0.005
  quat_normalize_interval: 10

  primitives:
    - name: "edf_duct"
      shape: "cylinder"
      mass: 1.2            # kg
      radius: 0.045        # m (90 mm diameter)
      height: 0.15         # m
      position: [0, 0, 0.05]  # body frame, m
      orientation: [0, 0, 0]   # Euler angles (deg), optional

    - name: "battery_8s"
      shape: "box"
      mass: 0.8
      dimensions: [0.15, 0.05, 0.04]  # a, b, c in m
      position: [0.02, 0, -0.03]

    - name: "jetson_nano"
      shape: "box"
      mass: 0.14
      dimensions: [0.08, 0.08, 0.03]
      position: [-0.04, 0, -0.05]

    - name: "frame_structure"
      shape: "cylinder"
      mass: 0.5
      radius: 0.06
      height: 0.30
      position: [0, 0, 0]

    - name: "payload_variable"
      shape: "sphere"
      mass: 0.2            # ±10% randomized per episode
      radius: 0.03
      position: [0, 0, -0.08]
      randomize_mass: 0.10  # ±10% uniform

  thrust:
    k_thrust: 1.0e-6       # N/(rad/s)^2
    tau_motor: 0.1          # s, 1st-order lag time constant
    I_fan: 0.001            # kg·m^2, fan rotor inertia
    max_rpm: 95000          # RPM
    r_thrust: [0, 0, 0.08] # thrust application point, body frame
    r_duct: 0.045           # duct radius for ground effect

  aero:
    rho: 1.225              # kg/m^3
    Cd: 0.7
    A_proj: 0.01            # m^2 projected area
    r_cp: [0, 0, 0.05]     # center of pressure, body frame

  fins:
    count: 4
    A_fin: 0.005            # m^2 planform area per fin
    Cd0: 0.01               # parasitic drag coefficient
    AR: 2.0                 # aspect ratio
    max_deflection: 0.26    # rad (~15 deg)
    V_exhaust_nominal: 70   # m/s at full RPM
    positions:              # body frame, per fin
      - [0, 0.04, 0.12]    # fin 1 (right)
      - [0, -0.04, 0.12]   # fin 2 (left)
      - [0.04, 0, 0.12]    # fin 3 (forward)
      - [-0.04, 0, 0.12]   # fin 4 (aft)
    normals:                # lift direction unit vectors per fin
      - [1, 0, 0]
      - [1, 0, 0]
      - [0, 1, 0]
      - [0, 1, 0]

  wind:
    mean_speed_range: [0, 10]   # m/s, sampled uniformly per episode
    turbulence_intensity: 0.3   # Dryden parameter
    altitude_ref: 5.0           # m, reference altitude for Dryden model
```

### 9.2 Config Loading

```python
import yaml

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)
```

---

## 10. Validation and Testing

### 10.1 Unit Tests (Per Sub-Model)

| Test | Method | Pass Criterion |
|---|---|---|
| Mass properties: single primitive | Compare to analytical formulas | Exact match (floating-point tolerance) |
| Mass properties: multi-primitive | Hand-compute CoM/I for 2 boxes | Match to 6 decimal places |
| Quaternion DCM | Compare `quat_to_dcm(q)` to scipy `Rotation.from_quat()` | Max element error < 1e-12 |
| Quaternion kinematics | Constant ω → integrated q matches analytical rotation | Angle error < 0.01° over 10 s |
| Thrust lag | Step response of 1st-order ODE | 63% of final value at t = τ_motor |
| Ground effect | Thrust at h = r_duct | T_eff = 1.5 × T |
| Fin forces | δ = 0 → zero lift | F_fin < 1e-10 N |
| Thin airfoil | δ = 10° → C_L = 2π × 0.1745 ≈ 1.097 | Match to 3 significant figures |
| Drag force | Known v_rel → compute analytically | Exact match |
| RK4 | Integrate simple ODE (e.g., ẏ = -y) | Error < O(dt⁵) per step |

### 10.2 Integration Tests (Full Dynamics)

| Test | Setup | Expected Behavior |
|---|---|---|
| Free fall | T=0, q=[1,0,0,0], v=0 | z increases at g, v_b[2] = g·t |
| Hover equilibrium | T = m·g, q=[1,0,0,0] | All derivatives ≈ 0, state stationary |
| Pure yaw from motor torque | T constant, fins neutral, perturb yaw | Yaw rate matches τ_motor / I_zz |
| Gyroscopic precession | High RPM, small pitch rate → expect roll torque | Magnitude matches ω × h_fan |
| Wind step response | Sudden 10 m/s crosswind | Lateral force onset, body drifts, stabilizes if controlled |
| Fin authority | Max deflection on one fin pair | Pitch/yaw moment matches analytical prediction |

### 10.3 Linearization and Frequency Analysis

Linearize the full nonlinear system about the hover trim point:
1. Numerically compute Jacobians \(A = \partial f / \partial y\), \(B = \partial f / \partial u\) via finite differences.
2. Check eigenvalues of \(A\): all real parts negative (open-loop hover is unstable for rocket-like vehicle — expect positive real eigenvalues in pitch/roll, confirming need for active control).
3. Verify that the fastest mode has frequency < 1/(2·dt) to confirm the time step is adequate.
4. Optional: Compute controllability matrix rank to verify full controllability with 5 inputs (thrust + 4 fins).

### 10.4 Energy Conservation Check

For a torque-free, drag-free case (disable all forces except gravity):
- Total mechanical energy \(E = \frac{1}{2}m\|\mathbf{v}\|^2 + mgh\) should be conserved by RK4.
- Integrate for 100 s, check \(|E(t) - E(0)| / E(0) < 10^{-6}\) (RK4 is not symplectic but should conserve well for short durations at dt=0.005 s).

---

## 11. Isaac Sim Integration Notes

### 11.1 Two Deployment Modes

| Mode | Use Case | Integration Strategy |
|---|---|---|
| **Pure NumPy** | Fast prototyping, unit testing, batch RL training (non-visual) | `VehicleDynamics` class runs standalone; no Isaac Sim dependency |
| **Isaac Sim** | Visual validation, HIL pipeline, domain randomization with renderer | Map to USD rigid bodies; PhysX handles contact/ground; custom force callbacks apply thrust/fins/aero |

### 11.2 Isaac Sim Callback Pattern

```python
# In Isaac Sim extension / OmniIsaacGymEnvs task
def pre_physics_step(self, actions):
    """Called before each PhysX step. Apply custom forces."""
    T_cmd = actions[:, 0]
    fin_deltas = actions[:, 1:5]

    # Compute forces from our models
    F_thrust, tau_thrust = self.thrust_model.compute_batched(...)
    F_fins, tau_fins = self.fin_model.compute_batched(...)
    F_aero, tau_aero = self.aero_model.compute_batched(...)

    # Apply to rigid body via Isaac Sim API
    self._drone.apply_forces(F_thrust + F_fins + F_aero, is_global=False)
    self._drone.apply_torques(tau_thrust + tau_fins + tau_aero + tau_motor, is_global=False)
```

In Isaac Sim mode, PhysX handles integration (not our RK4), but the force/torque models remain identical. This ensures the sim logic is the same across both modes.

### 11.3 Sensor Noise (Post-Sim)

Sensor simulation is **not** part of the plant model. After each physics step, apply noise to observations:
- **IMU**: Gaussian noise on accelerometer (σ = 0.1 m/s²) and gyroscope (σ = 0.01 rad/s), plus bias drift.
- **Optical flow**: Position noise (σ = 0.1 m), 1 Hz update rate (sample-and-hold between updates).
- **Barometer**: Altitude noise (σ = 0.5 m).

This is handled by the observation pipeline, not the dynamics module.

---

## 12. File Map and Dependencies

### 12.1 Planned File Structure

```
simulation/
├── dynamics/
│   ├── __init__.py
│   ├── vehicle.py              # VehicleDynamics (top-level class)
│   ├── mass_properties.py      # MassProperties (primitive aggregation)
│   ├── thrust_model.py         # ThrustModel (EDF curve, lag, ground effect)
│   ├── aero_model.py           # AeroModel (drag on combined shape)
│   ├── fin_model.py            # FinModel (4x thin-airfoil fins)
│   ├── wind_model.py           # WindModel (Dryden turbulence)
│   ├── integrator.py           # RK4 fixed-step integrator
│   └── quaternion_utils.py     # quat_to_dcm, quat_mult, quat_normalize
├── configs/
│   ├── default_vehicle.yaml    # Default EDF drone config
│   └── test_vehicle.yaml       # Simplified config for unit tests
└── tests/
    ├── test_mass_properties.py
    ├── test_thrust_model.py
    ├── test_aero_model.py
    ├── test_fin_model.py
    ├── test_integrator.py
    ├── test_quaternion_utils.py
    └── test_vehicle_dynamics.py # Full integration tests
```

### 12.2 Dependencies

```
numpy >= 1.24
scipy >= 1.10      # only for validation (Rotation, odeint comparison)
pyyaml >= 6.0
pytest >= 7.0      # testing
```

Optional (for GPU-accelerated batch training):
```
jax >= 0.4
jaxlib >= 0.4
```

### 12.3 Implementation Order

| Phase | Files | Dependency | Est. Effort |
|---|---|---|---|
| 1 | `quaternion_utils.py` | None | 0.5 day |
| 2 | `mass_properties.py` + tests | Phase 1 | 1 day |
| 3 | `thrust_model.py` + tests | None | 1 day |
| 4 | `aero_model.py` + `wind_model.py` + tests | Phase 1 | 1 day |
| 5 | `fin_model.py` + tests | Phase 1 | 1 day |
| 6 | `integrator.py` | None | 0.5 day |
| 7 | `vehicle.py` (assembly) + integration tests | All above | 1.5 days |
| 8 | Config files + linearization analysis | Phase 7 | 1 day |
| 9 | Isaac Sim callback wrapper | Phase 7 | 1–2 days |
| **Total** | | | **~8–9 days** |

---

## Appendix A: Derivative Function — Complete Pseudocode

For reference, the complete derivative function assembling all terms:

```python
def derivs(y, u, t, params):
    """
    Complete state derivative for 6-DOF EDF drone.

    y:  [p(3), v_b(3), q(4), omega(3), T(1)]  -- 14 scalars
    u:  [T_cmd, delta_1, delta_2, delta_3, delta_4]  -- 5 scalars
    t:  current time (s)
    """
    # Unpack state
    p       = y[0:3]        # inertial position (NED)
    v_b     = y[3:6]        # body velocity
    q       = y[6:10]       # quaternion (scalar-first)
    omega   = y[10:13]      # body angular rate
    T       = y[13]         # current thrust (with lag)

    T_cmd       = u[0]
    fin_deltas  = u[1:5]

    # Rotation matrix (body -> inertial)
    R = quat_to_dcm(q)

    # --- Thrust ---
    omega_fan = np.sqrt(max(T, 0) / params.k_thrust)
    h_alt = -p[2]  # altitude (NED: z is down, so altitude = -z)
    T_eff = T * ground_effect(h_alt, params.r_duct)
    F_thrust = np.array([0, 0, T_eff])
    r_offset = params.r_thrust - params.com
    tau_thrust = np.cross(r_offset, F_thrust)

    # Thrust lag ODE
    T_dot = (T_cmd - T) / params.tau_motor

    # --- Aerodynamics ---
    v_wind_body = R.T @ wind_model.sample(t)
    v_rel = v_b - v_wind_body
    speed_rel = np.linalg.norm(v_rel)
    F_aero = -0.5 * params.rho * speed_rel * v_rel * params.Cd * params.A_proj
    tau_aero = np.cross(params.r_cp - params.com, F_aero)

    # --- Fins ---
    V_exhaust = params.V_exhaust_nom * (omega_fan / params.omega_fan_max)
    F_fins = np.zeros(3)
    tau_fins = np.zeros(3)
    for k in range(4):
        alpha_k = np.clip(fin_deltas[k], -params.max_defl, params.max_defl)
        C_L = 2 * np.pi * alpha_k
        C_D = params.Cd0 + C_L**2 / (np.pi * params.AR)
        q_dyn = 0.5 * params.rho * V_exhaust**2 * params.A_fin
        F_k = q_dyn * (C_L * params.fin_normals[k] + C_D * params.fin_drags[k])
        F_fins += F_k
        tau_fins += np.cross(params.fin_positions[k] - params.com, F_k)

    # --- Motor reaction torque ---
    omega_fan_dot = 0.5 * T_dot / (params.k_thrust * max(omega_fan, 1.0))
    tau_motor = np.array([0, 0, -params.I_fan * omega_fan_dot])

    # --- Assemble total force/torque ---
    F_total = F_thrust + F_aero + F_fins
    tau_total = tau_thrust + tau_aero + tau_fins + tau_motor

    # --- State derivatives ---
    p_dot = R @ v_b

    g_body = R.T @ np.array([0, 0, params.g])
    v_dot = F_total / params.mass + g_body - np.cross(omega, v_b)

    q_dot = 0.5 * quat_mult(q, np.array([0, omega[0], omega[1], omega[2]]))

    h_fan = np.array([0, 0, params.I_fan * omega_fan])
    omega_dot = params.I_inv @ (
        tau_total
        - np.cross(omega, params.I @ omega)
        - np.cross(omega, h_fan)
    )

    return np.concatenate([p_dot, v_dot, q_dot, omega_dot, [T_dot]])
```

---

## Appendix B: Decisions Log

| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| Slosh model | Deleted | Pendulum/spring-mass | Electric drone, no propellant burn-down, +20-50% ODE cost |
| Mass properties | Init-only, from primitives | Per-step dynamic update | Fixed mass, saves ~10-15% compute |
| Orientation | Quaternion (scalar-first) | Euler angles, rotation matrix | No gimbal lock, 4 scalars vs 9 |
| Integrator | RK4 fixed-step | RK45 adaptive, RK8 | Deterministic for RL, best accuracy/speed ratio |
| Time step | 0.005 s | 0.001 s, 0.01 s | Balances gyro stiffness vs. training speed |
| Fin aero | Thin-airfoil (\(C_L=2\pi\alpha\)) | Full NACA lookup, CFD | Low-Mach, small α; lookup tables are overkill |
| Wind model | Dryden spectrum (seeded) | Constant wind, von Kármán | Good turbulence fidelity, reproducible |
| Ground effect | Simple \(1 + 0.5(r/h)^2\) | None, full ground-plane mirror | Captures dominant effect without CFD |
| Config format | YAML | JSON, TOML, hardcoded | Human-readable, standard in ML, easy to edit |
| Earth model | Flat, g=9.81 | WGS-84, J2 gravity | Altitude <2 km, curvature irrelevant |
| Radiation/vacuum | Deleted | — | Earth-bound sim, not space |

---

## Appendix C: Notation Reference

| Symbol | Meaning | Units |
|---|---|---|
| \(\mathbf{p}\) | Inertial position (NED) | m |
| \(\mathbf{v}_b\) | Body-frame velocity | m/s |
| \(\mathbf{q}\) | Orientation quaternion | — |
| \(\boldsymbol{\omega}\) | Body angular velocity | rad/s |
| \(T\) | Current thrust magnitude | N |
| \(m\) | Total vehicle mass | kg |
| \(\mathbf{I}\) | Inertia tensor about CoM | kg·m² |
| \(\mathbf{R}\) | DCM (body → inertial) | — |
| \(\mathbf{F}_b\) | Total body-frame force | N |
| \(\boldsymbol{\tau}_b\) | Total body-frame torque | N·m |
| \(\mathbf{h}_{fan}\) | Fan angular momentum | kg·m²/s |
| \(\Omega_{fan}\) | Fan angular velocity | rad/s |
| \(\mathbf{g}_b\) | Gravity in body frame | m/s² |
| \(\rho\) | Air density | kg/m³ |
| \(C_L, C_D\) | Lift/drag coefficients | — |
| \(\delta_k\) | Fin deflection angle | rad |
| \(dt\) | Integration time step | s |
