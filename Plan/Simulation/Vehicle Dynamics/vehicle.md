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
- **Precompute** total mass $m$, center of mass $\mathbf{c}_{cm}$, inertia tensor $\mathbf{I}$, and its inverse $\mathbf{I}^{-1}$ once at init.
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
- Fin forces use thin-airfoil approximation ($C_L = 2\pi\alpha$) — valid for low-Mach jet (~60 m/s) and small deflection angles (<15°). Full NACA airfoil lookup tables are unnecessary.
- Wind turbulence via Dryden spectrum (seeded for reproducibility).

### 1.5 Non-Negotiable Terms

- **Gyroscopic precession**: Spinning fan at ~10⁴ rad/s produces significant precession torques. Omitting this causes unexplained yaw/pitch drift.
- **Motor thrust lag**: 1st-order dynamics ($\tau_{motor} \approx 0.1$ s). Unmodeled lag causes 0.5–1 m touchdown errors.
- **Ground effect**: Thrust multiplier near the pad. Without it, the controller sees a phantom deceleration near touchdown.
- **Motor reaction torque**: Small but prevents yaw drift accumulation.

---

## 2. State Vector and Kinematics

### 2.1 State Vector (13 Scalars)

Minimal for 6-DOF with quaternion orientation. No slosh states, no dynamic mass states.

| Symbol | Dimension | Frame | Description |
|---|---|---|---|
| $\mathbf{p} = [x, y, z]^T$ | 3 | Inertial (NED) | Position (m) |
| $\mathbf{v}_b = [u, v, w]^T$ | 3 | Body | Velocity (m/s) |
| $\mathbf{q} = [q_0, q_1, q_2, q_3]^T$ | 4 | — | Orientation quaternion (scalar-first, unit norm) |
| $\boldsymbol{\omega} = [p, q, r]^T$ | 3 | Body | Angular velocity (rad/s) |

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
- Quaternion convention: scalar-first $[q_0, q_1, q_2, q_3]$, Hamilton product. Chosen over Euler angles to eliminate gimbal lock. Cheaper than rotation matrices for composition.

### 2.3 Direction Cosine Matrix (Body-to-Inertial)

$$
\mathbf{R}(\mathbf{q}) = \begin{bmatrix}
q_0^2 + q_1^2 - q_2^2 - q_3^2 & 2(q_1 q_2 - q_0 q_3) & 2(q_1 q_3 + q_0 q_2) \\
2(q_1 q_2 + q_0 q_3) & q_0^2 - q_1^2 + q_2^2 - q_3^2 & 2(q_2 q_3 - q_0 q_1) \\
2(q_1 q_3 - q_0 q_2) & 2(q_2 q_3 + q_0 q_1) & q_0^2 - q_1^2 - q_2^2 + q_3^2
\end{bmatrix}
$$

**Implementation note**: Compute $\mathbf{R}$ once per derivative evaluation, reuse across position kinematics, gravity rotation, and wind rotation. Store as a 3×3 NumPy array; do not recompute redundantly.

### 2.4 Position Kinematics

$$
\dot{\mathbf{p}} = \mathbf{R}(\mathbf{q})\,\mathbf{v}_b
$$

Transforms body-frame velocity to inertial-frame position rate. Exact, no approximation.

### 2.5 Quaternion Kinematics

$$
\dot{\mathbf{q}} = \frac{1}{2}\,\mathbf{q} \otimes \begin{bmatrix} 0 \\ \boldsymbol{\omega} \end{bmatrix}
= \frac{1}{2} \begin{bmatrix}
-q_1 p - q_2 q - q_3 r \\
\phantom{-}q_0 p + q_2 r - q_3 q \\
\phantom{-}q_0 q + q_3 p - q_1 r \\
\phantom{-}q_0 r + q_1 q - q_2 p
\end{bmatrix}
$$

**Normalization policy**: Re-normalize $\mathbf{q}$ to unit norm every 10 integration steps (not every step — saves divisions; drift is negligible over 10 × 0.005 s = 0.05 s). Implementation: `q /= np.linalg.norm(q)` with a step counter modulo 10.

---

## 3. Translational Dynamics

### 3.1 Newton's Second Law in a Rotating Body Frame

From Newton's second law, accounting for the rotating reference frame (Coriolis term):

$$
m\,(\dot{\mathbf{v}}_b + \boldsymbol{\omega} \times \mathbf{v}_b) = \mathbf{F}_b + m\,\mathbf{g}_b
$$

Solving for the state derivative:

$$
\boxed{\dot{\mathbf{v}}_b = \frac{\mathbf{F}_b}{m} + \mathbf{g}_b - \boldsymbol{\omega} \times \mathbf{v}_b}
$$

### 3.2 Gravity in Body Frame

$$
\mathbf{g}_b = \mathbf{R}(\mathbf{q})^T \begin{bmatrix} 0 \\ 0 \\ g \end{bmatrix}, \quad g = 9.81\ \text{m/s}^2
$$

Flat Earth assumption (delete curvature for <2 km altitude). In NED, gravity points in +z (downward). The transpose $\mathbf{R}^T$ rotates from inertial to body frame.

### 3.3 Total Body-Frame Force

$$
\mathbf{F}_b = \mathbf{F}_{thrust} + \mathbf{F}_{aero} + \mathbf{F}_{fins}
$$

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

$$
\mathbf{I}\,\dot{\boldsymbol{\omega}} + \boldsymbol{\omega} \times (\mathbf{I}\,\boldsymbol{\omega}) + \boldsymbol{\omega} \times \mathbf{h}_{fan} = \boldsymbol{\tau}_b
$$

Solving for the state derivative:

$$
\boxed{\dot{\boldsymbol{\omega}} = \mathbf{I}^{-1}\left(\boldsymbol{\tau}_b - \boldsymbol{\omega} \times (\mathbf{I}\,\boldsymbol{\omega}) - \boldsymbol{\omega} \times \mathbf{h}_{fan}\right)}
$$

### 4.2 Fan Angular Momentum

$$
\mathbf{h}_{fan} = I_{fan}\,\Omega_{fan}\begin{bmatrix}0\\0\\1\end{bmatrix}
$$

| Parameter | Typical Value | Notes |
|---|---|---|
| $I_{fan}$ | ~0.001 kg·m² | Rotor moment of inertia for 90 mm EDF |
| $\Omega_{fan}$ | up to 10⁴ rad/s | Full-throttle RPM (~95,000 RPM) |

The cross product $\boldsymbol{\omega} \times \mathbf{h}_{fan}$ generates precession torques. At high RPM, this term dominates over the standard $\boldsymbol{\omega} \times (\mathbf{I}\boldsymbol{\omega})$ cross-coupling and is **critical for stability prediction**.

### 4.3 Inertia Inverse — Precomputed

$\mathbf{I}^{-1}$ is computed once at init and cached. If the inertia tensor is diagonal (principal axes aligned with body frame), the inverse is trivial: $\text{diag}(1/I_{xx}, 1/I_{yy}, 1/I_{zz})$. If off-diagonal terms exist (asymmetric mass distribution), compute the full 3×3 inverse at init.

**Decision heuristic**: If all off-diagonal inertia terms are <5% of the diagonals, zero them and use the diagonal inverse. Otherwise, use `np.linalg.inv()` once at init.

### 4.4 Total Body-Frame Torque

$$
\boldsymbol{\tau}_b = \boldsymbol{\tau}_{thrust} + \boldsymbol{\tau}_{aero} + \boldsymbol{\tau}_{fins} + \boldsymbol{\tau}_{motor}
$$

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

Each primitive $i$ provides:
- **Mass**: $m_i$ (kg)
- **Local CoM position**: $\mathbf{c}_i$ (m, in body frame — FRD, origin at geometric center)
- **Dimensions**: Shape-specific (radius/height for cylinder, $a \times b \times c$ for box, radius for sphere)
- **Local inertia about own CoM**: $\mathbf{I}_i$ (3×3 tensor, kg·m²) — computed from shape + mass
- **Optional orientation**: $\mathbf{R}_i$ (rotation matrix, for non-axis-aligned primitives)
- **Surface area**: $S_i$ (m²) — total wetted surface area for skin friction estimates
- **Drag-facing areas**: $\{A_{x,i},\, A_{y,i},\, A_{z,i}\}$ (m²) — projected area along each body axis, for directional drag computation
- **Local drag coefficient**: $C_{d,i}$ — shape-specific (0.47 sphere, ~0.8 cylinder, ~1.0 box, 0.0 if shielded/internal)
- **Optional mass randomization**: $\Delta m_i$ (fraction, e.g., 0.10 for ±10% per episode)

Aggregated aerodynamic properties (computed at init alongside mass properties):

$$
A_{proj,\hat{n}} = \sum_{i=1}^{N} A_{\hat{n},i}, \quad S_{total} = \sum_{i=1}^{N} S_i
$$

These are used by `AeroModel` for combined-shape drag. Per-axis projected areas enable directional drag (different drag when wind hits the vehicle from the side vs. head-on).

### 5.3 Primitive Inertia Formulas

**Cylinder** (aligned along z-axis, radius $r$, height $h$):
$$
I_{xx} = I_{yy} = \frac{1}{12}\,m\,(3r^2 + h^2), \quad I_{zz} = \frac{1}{2}\,m\,r^2
$$

**Box** (dimensions $a \times b \times c$):
$$
I_{xx} = \frac{1}{12}\,m\,(b^2 + c^2), \quad I_{yy} = \frac{1}{12}\,m\,(a^2 + c^2), \quad I_{zz} = \frac{1}{12}\,m\,(a^2 + b^2)
$$

**Sphere** (radius $r$):
$$
I_{xx} = I_{yy} = I_{zz} = \frac{2}{5}\,m\,r^2
$$

If a primitive is oriented (rotated by $\mathbf{R}_i$ relative to body frame), apply the similarity transform:
$$
\mathbf{I}_i^{body} = \mathbf{R}_i\,\mathbf{I}_i^{local}\,\mathbf{R}_i^T
$$

### 5.4 Aggregation Equations

**Total mass:**
$$
m = \sum_{i=1}^{N} m_i
$$

**Global center of mass:**
$$
\mathbf{c}_{cm} = \frac{1}{m}\sum_{i=1}^{N} m_i\,\mathbf{c}_i
$$

**Inertia about CoM** (parallel axis theorem):
$$
\mathbf{I} = \sum_{i=1}^{N}\left[\mathbf{I}_i + m_i\left((\mathbf{d}_i \cdot \mathbf{d}_i)\,\mathbf{1}_3 - \mathbf{d}_i\,\mathbf{d}_i^T\right)\right], \quad \mathbf{d}_i = \mathbf{c}_i - \mathbf{c}_{cm}
$$

where $\mathbf{1}_3$ is the 3×3 identity matrix.

### 5.5 Post-Aggregation Actions

1. **Shift reference point**: All force application points (thrust, fins, center of pressure) are re-expressed relative to $\mathbf{c}_{cm}$. This eliminates torque artifacts from an off-center CoM.
2. **Cache**: Store $m$, $\mathbf{c}_{cm}$, $\mathbf{I}$, $\mathbf{I}^{-1}$ as immutable instance attributes.
3. **Randomization** (optional, for robustness training): At episode init, apply ±10% perturbation to individual primitive masses before aggregation. This produces varied CoM/MoI across episodes for domain randomization.

### 5.6 Implementation Pseudocode

```python
def compute_mass_properties(self, primitives: list[dict]) -> None:
    """Aggregate mass, CoM, inertia, and surface areas from primitive list.

    Call once at init. Results are cached as immutable attributes.
    """
    total_mass = sum(p['mass'] for p in primitives)
    com = sum(p['mass'] * np.array(p['position']) for p in primitives) / total_mass

    I_total = np.zeros((3, 3))
    total_surface_area = 0.0
    proj_x, proj_y, proj_z = 0.0, 0.0, 0.0

    for p in primitives:
        I_local = self._primitive_inertia(p)          # shape-specific formula
        if 'orientation' in p:
            R_i = self._euler_to_dcm(p['orientation'])
            I_local = R_i @ I_local @ R_i.T           # rotate to body frame
        d = np.array(p['position']) - com              # offset from global CoM
        I_parallel = p['mass'] * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
        I_total += I_local + I_parallel

        # Aggregate aerodynamic surface areas
        total_surface_area += p.get('surface_area', 0.0)
        drag = p.get('drag_facing', {})
        proj_x += drag.get('x', 0.0)
        proj_y += drag.get('y', 0.0)
        proj_z += drag.get('z', 0.0)

    self.total_mass = total_mass
    self.com = com
    self.inertia = I_total
    self.inertia_inv = np.linalg.inv(I_total)
    self.total_surface_area = total_surface_area
    self.projected_area_x = proj_x
    self.projected_area_y = proj_y
    self.projected_area_z = proj_z

@classmethod
def from_cad(cls, cad_config: dict) -> 'MassProperties':
    """Create MassProperties from CAD-exported values (Inventor, SolidWorks).

    Bypasses primitive aggregation entirely — uses validated CAD data.
    """
    props = cls.__new__(cls)
    props.total_mass = cad_config['total_mass']
    props.com = np.array(cad_config['center_of_mass'])
    props.inertia = np.array(cad_config['inertia_tensor'])
    props.inertia_inv = np.linalg.inv(props.inertia)
    props.total_surface_area = cad_config.get('total_surface_area', 0.0)
    # Directional areas not available from CAD — fall back to scalar A_proj
    props.projected_area_x = 0.0
    props.projected_area_y = 0.0
    props.projected_area_z = 0.0
    return props
```

---

## 6. Force and Torque Models

### 6.1 Thrust (Main Propulsive)

#### 6.1.1 Thrust Magnitude

Quadratic in fan speed:
$$
T = k\,\Omega_{fan}^2
$$

| Parameter | Value | Notes |
|---|---|---|
| $k$ | ~10⁻⁶ N/(rad/s)² | Empirical, calibrate from EDF static test data |
| $\Omega_{fan}$ | 0–10⁴ rad/s | Commanded via ESC PWM |

#### 6.1.2 Motor Lag (1st-Order Thrust Dynamics)

Thrust does not respond instantaneously to commands. Modeled as a 1st-order lag:
$$
\dot{T} = \frac{1}{\tau_{motor}}\,(T_{cmd} - T), \quad \tau_{motor} \approx 0.1\ \text{s}
$$

This adds **1 auxiliary state** to the integration (total with servo positions: 18 scalars integrated — see §6.3.6 for the 4 additional servo states). The lag is critical — without it, the controller overcompensates near touchdown and the sim-to-real gap widens by 0.5–1 m.

**Implementation**: Integrate $T$ alongside the main state vector $\mathbf{y}$. The commanded thrust $T_{cmd}$ comes from the RL agent's action.

#### 6.1.3 Ground Effect

When altitude $h < 2\,r_{duct}$ (within ~2 duct diameters of the pad):
$$
T_{effective} = T \cdot \left(1 + 0.5\,\left(\frac{r_{duct}}{h}\right)^2\right)
$$

Simple empirical approximation. Prevents phantom deceleration artifacts near touchdown. Clamp $h \geq 0.01$ m to avoid division-by-zero.

#### 6.1.4 Thrust Force and Torque

$$
\mathbf{F}_{thrust} = T_{effective}\begin{bmatrix}0\\0\\1\end{bmatrix}, \quad \boldsymbol{\tau}_{thrust} = (\mathbf{r}_{thrust} - \mathbf{c}_{cm}) \times \mathbf{F}_{thrust}
$$

Thrust acts along the body z-axis (downward in FRD = upward in vehicle orientation). The torque arises if the thrust application point $\mathbf{r}_{thrust}$ is offset from $\mathbf{c}_{cm}$.

### 6.2 Aerodynamic Drag (Combined Shape)

#### 6.2.1 Relative Velocity

$$
\mathbf{v}_{rel} = \mathbf{v}_b - \mathbf{R}(\mathbf{q})^T\,\mathbf{v}_{wind}
$$

where $\mathbf{v}_{wind}$ is the wind velocity in the inertial frame (e.g., gusts up to 10 m/s).

#### 6.2.2 Drag Force

$$
\mathbf{F}_{aero} = -\frac{1}{2}\,\rho\,\|\mathbf{v}_{rel}\|\,\mathbf{v}_{rel}\,C_d\,A_{proj}
$$

| Parameter | Value | Notes |
|---|---|---|
| $\rho$ | From EnvironmentModel | Per-step air density (see [env.md §4](../Enviornment/env.md)); ISA ref = 1.225 kg/m³ |
| $C_d$ | 0.5–1.0 | Bluff body; tune from wind tunnel or literature |
| $A_{proj}$ | ~0.01 m² | Projected frontal area from primitive bounding box |

#### 6.2.3 Aerodynamic Torque

$$
\boldsymbol{\tau}_{aero} = (\mathbf{r}_{cp} - \mathbf{c}_{cm}) \times \mathbf{F}_{aero}
$$

Center of pressure $\mathbf{r}_{cp}$ is approximately 0.05 m below CoM (aft, toward the duct). This is a small correction term.

#### 6.2.4 Wind Model

> **Migrated to EnvironmentModel**: Wind is now provided by the `EnvironmentModel` class (see [env.md §3](../Enviornment/env.md)). The vehicle's `derivs()` receives the wind vector and air density via `env.sample_at_state(t, p)` — it does not own a wind model.

Wind components (mean + Dryden turbulence + discrete gusts) affect aerodynamic drag via the relative velocity $\mathbf{v}_{rel} = \mathbf{v}_b - \mathbf{R}^T \mathbf{v}_{wind}$. See [env.md §3](../Enviornment/env.md) for the complete wind model specification and [env.md §5](../Enviornment/env.md) for the coupling interface.

### 6.3 Fin Forces (TVC Emulation via NACA Airfoils in Exhaust)

#### 6.3.1 Per-Fin Geometry

4 fins arranged at 90° intervals around the exhaust duct, each with independent deflection $\delta_k$ (rad), commanded by the RL agent.

| Fin Parameter | Value | Notes |
|---|---|---|
| Airfoil profile | NACA 0012 | Symmetric, 12% thickness ratio |
| Chord $c$ | 0.050 m | Along exhaust flow direction |
| Span $b$ | 0.040 m | Radial extent into exhaust stream |
| Thickness $t_{max}$ | 0.006 m | $= 0.12 \times c$ |
| Planform area $A_{fin}$ | 0.002 m² | $= c \times b$ |
| Wetted area | ~0.0041 m² | Both surfaces + leading/trailing edges |
| Aspect ratio AR | 0.80 | $= b^2 / A_{fin}$ (short, low-AR fins) |
| Mechanical deflection limit | ±20° (0.349 rad) | Freewing 9 g digital servo travel |
| Servo rate limit | ~600°/s (10.5 rad/s) at 6 V | No-load angular speed (0.10 sec/60° transit) |
| Servo torque | 1.9 kg·cm (~0.186 N·m) at 6 V | Sufficient for ±15° fins in exhaust jet |
| Servo time constant | $\tau_{servo} \approx 0.03$–$0.05$ s | Effective first-order position lag |
| Servo bandwidth | ~12–15 Hz | From $f_{bw} \approx 1/(\pi \cdot t_{half\_range})$; literature for similar 9 g servos: 8–20 Hz |
| Servo type | Digital, full metal gear | Low backlash (~1–2 µs deadband), LiPo 5–6 V supply |
| Servo weight / dims | ~9–12 g, 23.5×12×27 mm | Matches hardware BoM |

#### 6.3.2 Effective Velocity and Angle of Attack

$$
V_e \approx 60\text{–}80\ \text{m/s (EDF exhaust velocity, proportional to RPM)}
$$
$$
\alpha_k = \delta_k + \beta_k
$$

where $\beta_k$ is the local sideslip contribution (small, from body angular rates and lateral velocity).

#### 6.3.3 Thin-Airfoil Coefficients

Valid for $|\alpha| < 15°$ (pre-stall). The mechanical servo limit is ±20° (0.349 rad), but aerodynamic coefficients use stall protection beyond ±15° (0.262 rad).

$$
C_L = 2\pi\,\alpha_k
$$
$$
C_D = C_{D0} + \frac{C_L^2}{\pi\,AR}
$$

| Parameter | Value | Notes |
|---|---|---|
| $C_{L\alpha}$ | $2\pi$ ≈ 6.283 /rad | Thin-airfoil lift slope (NACA 0012 at low Mach) |
| $C_{D0}$ | ~0.01 | Parasitic drag coefficient |
| AR | 0.80 | Aspect ratio $b^2 / A_{fin}$ for short fins |
| Stall angle $\alpha_{stall}$ | 15° (0.262 rad) | Onset of lift coefficient plateau |
| Mechanical limit $\delta_{max}$ | ±20° (0.349 rad) | Freewing servo travel — hard clamp on command |

**Two-stage protection**:
1. **Servo clamp**: $\delta_k \leftarrow \text{clip}(\delta_k,\, -\delta_{max},\, +\delta_{max})$ — enforces mechanical travel limit (±20°).
2. **Stall soft-clamp**: For aerodynamic coefficient computation, apply smooth transition above $\alpha_{stall}$:

```python
alpha_eff = stall_angle * np.tanh(alpha_k / stall_angle)  # soft clamp at ±15°
```

This allows the servo to command the full ±20° range while the aero model gracefully handles the post-stall regime. The RL agent should learn to avoid sustained deflections > 15° where control authority degrades.

#### 6.3.4 Per-Fin Force

$$
\mathbf{F}_{fin,k} = \frac{1}{2}\,\rho\,V_e^2\,A_{fin}\,(C_L\,\hat{\mathbf{n}}_{L,k} + C_D\,\hat{\mathbf{n}}_{D,k})
$$

| Parameter | Value | Notes |
|---|---|---|
| $A_{fin}$ | ~0.005 m² | Fin planform area |
| $\hat{\mathbf{n}}_{L,k}$ | Perpendicular to flow | Lift direction, depends on fin orientation |
| $\hat{\mathbf{n}}_{D,k}$ | Parallel to flow | Drag direction, opposes exhaust velocity |

#### 6.3.5 Total Fin Contribution

$$
\mathbf{F}_{fins} = \sum_{k=1}^{4} \mathbf{F}_{fin,k}, \quad \boldsymbol{\tau}_{fins} = \sum_{k=1}^{4} (\mathbf{r}_{fin,k} - \mathbf{c}_{cm}) \times \mathbf{F}_{fin,k}
$$

Fin positions $\mathbf{r}_{fin,k}$ are ~0.1 m aft of CoM, defined in the config.

### 6.3.6 Servo Actuator Dynamics (Mandatory — Deletes Instant Action Assumption)

> **Hardware**: Freewing 9 g digital metal gear servo (4.8–6.0 V).
> **Why mandatory**: The prior model applied fin commands instantaneously ($\delta_{actual} = \delta_{cmd}$). This is physically wrong — the servo has finite slew rate and position lag. Unmodeled servo dynamics cause a 10–30% sim-to-real success drop (Hwangbo et al., 2017 for drones). Deleting the instant-action assumption is prerequisite for trustworthy HIL and flight tests.

#### 6.3.6.1 Servo Physics — Grounded in Product Data

From Freewing product data (Motion RC and consistent OEM listings):

| Parameter | Value at 4.8 V | Value at 6.0 V | Units / Notes |
|---|---|---|---|
| Operating speed | 0.12 sec/60° | 0.10 sec/60° | Transit time (no load) |
| Max angular velocity | ~500 °/s | ~600 °/s | Derived: 60° / transit time |
| Torque | 1.6 kg·cm | 1.9 kg·cm | ~0.157–0.186 N·m; sufficient for ±15° fins |
| Operating voltage | 4.8–6.0 V | — | Run at ~5–6 V on LiPo |
| Gear train | Full metal | — | Low backlash, high stiffness |
| Type | Digital | — | ~1–2 µs deadband (vs. analog ~10 µs), faster internal PID |
| PWM signal | Standard (50 Hz) | — | 500–2500 µs pulse; tolerates up to 200–333 Hz unofficially |
| Weight / Dimensions | ~9–12 g | 23.5×12×27 mm | Lightweight, matches hardware BoM |

#### 6.3.6.2 First-Principles Bandwidth Derivation

**Mechanical slew rate limit**: Max deflection range in task = ±15° (30° total swing, per training.md $\delta_{max} = 0.26$ rad ≈ 15°). At 6 V: time for full swing $\approx 0.10 \times (30°/60°) = 0.05$ s.

$$
\dot{\delta}_{max} = 600°/\text{s} \approx 10.5 \ \text{rad/s}
$$

Time constant:
$$
\tau_{servo} \approx 0.03\text{–}0.05 \ \text{s}
$$
(effective first-order lag for position commands).

**Fundamental torque limit**: Acceleration $\alpha_{max} = \tau_{torque} / I_{arm}$ (fin inertia small), but the spec gives a velocity limit under no-load; under aerodynamic load (exhaust jet at 60–80 m/s), effective slew is 20–50% slower.

**Bandwidth estimate**: For a slew-limited actuator, closed-loop position bandwidth:

$$
f_{bw} \approx \frac{1}{\pi \cdot t_{half\_range}} \approx \frac{1}{\pi \times 0.025} \approx 12\text{–}15 \ \text{Hz}
$$

Literature for similar 9 g servos in drone applications confirms 8–20 Hz effective bandwidth.

**Nyquist for control**: Sampling rate $> 4$–$10 \times f_{bw}$ for 30–60° phase margin → 50–150 Hz ideal for tight tracking. At the chosen 40 Hz policy rate ($dt_{policy} = 0.025$ s), this provides $\sim 3 \times f_{bw}$, which is marginally adequate for disturbance rejection but avoids the severe aliasing of the prior 20 Hz rate.

**Disturbance spectrum**: Wind gusts peak at ~1–10 Hz (Dryden model); gyro precession effective $<50$ Hz. Control bandwidth $>20$ Hz required to reject without residual oscillation (per Euler's equations, uncompensated moments cause divergence).

#### 6.3.6.3 Actuator Model — Rate-Limited First-Order Lag

Model each servo as a **rate-limited first-order lag** (combined position tracking + slew saturation):

$$
\dot{\delta}_{actual,k} = \text{clip}\!\left(\frac{\delta_{cmd,k} - \delta_{actual,k}}{\tau_{servo}},\ -\dot{\delta}_{max},\ +\dot{\delta}_{max}\right)
$$

This captures two regimes:
1. **Small corrections** ($|\Delta\delta| < 10°$): behaves as first-order lag with time constant $\tau_{servo}$.
2. **Large slews** ($|\Delta\delta| > 10°$): rate-limited at $\dot{\delta}_{max} = 10.5$ rad/s — the servo moves as fast as it can.

Under aerodynamic load, apply derating:

$$
\dot{\delta}_{max,eff} = \dot{\delta}_{max} \cdot (1 - f_{derating}), \quad f_{derating} \in [0.2, 0.5]
$$

where $f_{derating}$ is randomized per-episode for domain randomization.

**State addition**: This adds **4 auxiliary states** $[\delta_{actual,1}, \ldots, \delta_{actual,4}]$ to the integration (total: 18 scalars = 13 original + 1 thrust lag + 4 servo positions).

#### 6.3.6.4 Implementation Pseudocode

```python
class ServoModel:
    """Rate-limited first-order lag for fin servo actuators.

    Models physical slew dynamics of the Freewing 9 g digital servo.
    Deletes the instant-action assumption from the prior model.
    """

    def __init__(self, config: dict):
        self.tau = config['servo']['tau_servo']              # s, position lag
        self.rate_max = config['servo']['max_angular_velocity']  # rad/s
        self.derating = config['servo']['aero_load_derating']
        self.n_fins = config['count']
        self.delta_actual = np.zeros(self.n_fins)            # current servo positions

    def reset(self, seed: int | None = None):
        """Reset servo positions and randomize tau for DR."""
        self.delta_actual = np.zeros(self.n_fins)
        if seed is not None:
            rng = np.random.default_rng(seed)
            tau_range = self.config['servo']['tau_servo_range']
            self.tau = rng.uniform(tau_range[0], tau_range[1])
            self.derating = rng.uniform(0.2, 0.5)

    def step(self, delta_cmd: np.ndarray, dt: float) -> np.ndarray:
        """Advance servo positions by one physics timestep.

        Args:
            delta_cmd: commanded fin deflections (4,) in rad
            dt: physics timestep (s)

        Returns:
            delta_actual: current physical fin positions (4,) in rad
        """
        rate_max_eff = self.rate_max * (1.0 - self.derating)
        error = delta_cmd - self.delta_actual
        rate_desired = error / self.tau
        rate_clipped = np.clip(rate_desired, -rate_max_eff, rate_max_eff)
        self.delta_actual += rate_clipped * dt
        return self.delta_actual.copy()
```

#### 6.3.6.5 Why Prior 20 Hz + Instant Action Was Wrong

The prior model applied actions directly to fin angles at 20 Hz ($dt_{policy} = 0.05$ s). Problems:

1. **Mechanical mismatch**: Servo can move ~30° in 0.05 s → at 20 Hz, commands are coarsely quantized; small corrections ($<10°$) complete within a single step → no smooth tracking, amplifies gyro cross-coupling.
2. **Sim-to-real gap**: Unmodeled slew causes over-optimistic policies in sim; hardware lags → crashes in HIL/flight (undermines RQ1 fidelity).
3. **GTrXL adaptation hurt**: Transformer needs dense temporal granularity to infer disturbances via history; 20 Hz → 300 steps/episode is sparse for 3–5 Hz attitude dynamics.

**Delete**: Fixed 20 Hz with instant action. **Replace**: 40 Hz ($dt_{policy} = 0.025$ s) with mandatory servo dynamics model. See [training.md §2.2](../Training%20Plan/training.md) for the updated timing architecture.

### 6.4 Motor Reaction Torque

$$
\boldsymbol{\tau}_{motor} = -I_{fan}\,\dot{\Omega}_{fan}\begin{bmatrix}0\\0\\1\end{bmatrix}
$$

The fan acceleration $\dot{\Omega}_{fan}$ is derived from the thrust lag state ($\dot{T} = k \cdot 2\Omega_{fan}\dot{\Omega}_{fan}$, solved for $\dot{\Omega}_{fan}$). This is small but prevents yaw drift accumulation over the episode.

### 6.5 Force/Torque Summary Table

| Source | Force | Torque | Notes |
|---|---|---|---|
| Thrust | $T_{eff}\,[0,0,1]^T$ | $(\mathbf{r}_{thrust} - \mathbf{c}_{cm}) \times \mathbf{F}_{thrust}$ | With lag and ground effect |
| Aero drag | $-\frac{1}{2}\rho\|v_{rel}\|v_{rel}C_dA$ | $(\mathbf{r}_{cp}-\mathbf{c}_{cm})\times\mathbf{F}_{aero}$ | On combined shape |
| Fins (×4) | $\frac{1}{2}\rho V_e^2 A_{fin}(C_L\hat{n}_L+C_D\hat{n}_D)$ | $(\mathbf{r}_{fin}-\mathbf{c}_{cm})\times\mathbf{F}_{fin}$ | Thin-airfoil, per fin |
| Motor reaction | — | $-I_{fan}\dot{\Omega}_{fan}[0,0,1]^T$ | Prevents yaw drift |

---

## 7. RK4 Fixed-Step Integrator

### 7.1 Standard 4th-Order Runge-Kutta

Let $\mathbf{y} = [\mathbf{p},\,\mathbf{v}_b,\,\mathbf{q},\,\boldsymbol{\omega},\,T,\,\boldsymbol{\delta}_{actual}]^T$ (18 scalars: 13 state + 1 thrust lag + 4 servo positions).
Let $\dot{\mathbf{y}} = f(\mathbf{y}, \mathbf{u}, t)$ where $\mathbf{u}$ = controls (thrust command, 4 fin deflection commands).

$$
\mathbf{k}_1 = dt \cdot f(\mathbf{y}_n,\,\mathbf{u},\,t_n)
$$
$$
\mathbf{k}_2 = dt \cdot f(\mathbf{y}_n + 0.5\,\mathbf{k}_1,\,\mathbf{u},\,t_n + 0.5\,dt)
$$
$$
\mathbf{k}_3 = dt \cdot f(\mathbf{y}_n + 0.5\,\mathbf{k}_2,\,\mathbf{u},\,t_n + 0.5\,dt)
$$
$$
\mathbf{k}_4 = dt \cdot f(\mathbf{y}_n + \mathbf{k}_3,\,\mathbf{u},\,t_n + dt)
$$
$$
\mathbf{y}_{n+1} = \mathbf{y}_n + \frac{1}{6}(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4)
$$

### 7.2 Time Step Selection

$$
dt = 0.005\ \text{s}
$$

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

The RL agent outputs actions at the policy frequency (40 Hz = every 0.025 s = 5 integration steps at dt=0.005 s). Between policy steps, controls $\mathbf{u}$ (thrust command + fin deflection commands) are held constant (zero-order hold). The servo dynamics model (§6.3.6) internally tracks physical fin positions using the rate-limited first-order lag — this means the actual fin deflections $\boldsymbol{\delta}_{actual}$ evolve continuously via the RK4 integrator even while the commanded deflections $\boldsymbol{\delta}_{cmd}$ are held constant. This correctly captures the servo's finite slew behavior within each policy step.

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
├── AeroModel                   (drag on combined shape, receives rho from EnvironmentModel)
├── FinModel                    (4x fin forces, thin-airfoil, receives rho from EnvironmentModel)
├── ServoModel                  (4x rate-limited first-order lag, Freewing 9 g digital servo)
└── Integrator                  (RK4 fixed-step)

External dependency (injected at init):
└── EnvironmentModel            (wind + atmosphere — see env.md)
```

### 8.2 Core Class: `VehicleDynamics`

```python
class VehicleDynamics:
    """6-DOF rigid-body plant model for EDF drone TVC landing sim.

    Wind and atmospheric conditions are provided by an external EnvironmentModel
    instance (see env.md). The vehicle does not own wind or atmosphere models.
    """

    def __init__(self, config: dict, env: 'EnvironmentModel'):
        # Mass properties (init-only, cached — from primitives or CAD override)
        if config.get('cad_override', {}).get('use_cad_override', False):
            self.mass_props = MassProperties.from_cad(config['cad_override'])
        else:
            self.mass_props = MassProperties(config['primitives'])
        self.mass = self.mass_props.total_mass
        self.I = self.mass_props.inertia
        self.I_inv = self.mass_props.inertia_inv
        self.com = self.mass_props.com

        # Aerodynamic surface areas (aggregated from primitives)
        self.projected_area_x = self.mass_props.projected_area_x
        self.projected_area_y = self.mass_props.projected_area_y
        self.projected_area_z = self.mass_props.projected_area_z
        self.total_surface_area = self.mass_props.total_surface_area

        # External environment (wind + atmosphere — see env.md)
        self.env = env

        # Force/torque sub-models (receive rho from EnvironmentModel per step)
        self.thrust_model = ThrustModel(config['edf'])
        self.aero_model = AeroModel(config['aero'], self.mass_props)
        self.fin_model = FinModel(config['fins'])
        self.servo_model = ServoModel(config['fins'])  # Freewing 9 g servo dynamics

        # Constants
        self.g = config.get('gravity', 9.81)
        self.dt = config.get('dt', 0.005)
        self.I_fan = config['edf']['I_fan']

        # State: [p(3), v_b(3), q(4), omega(3), T(1), delta_actual(4)] = 18
        self.state = np.zeros(18)
        self.time = 0.0
        self.step_count = 0

    def reset(self, initial_state: np.ndarray, seed: int | None = None) -> np.ndarray:
        """Reset to initial conditions for new episode."""
        self.state = np.zeros(18)
        self.state[:14] = initial_state[:14].copy()  # core state (13 dynamics + 1 thrust lag)
        self.state[14:18] = 0.0                       # servo positions start neutral
        self.time = 0.0
        self.step_count = 0
        self.thrust_model.reset()
        self.servo_model.reset(seed=seed)  # randomize tau_servo for DR
        # Environment (wind + atmosphere) is reset by EDFLandingEnv, not here
        return self.state

    def derivs(self, y: np.ndarray, u: np.ndarray, t: float) -> np.ndarray:
        """Compute state derivatives. u = [T_cmd, delta_1..4].

        Queries EnvironmentModel once per call for consistent rho + wind.
        See env.md §5 for the coupling interface.
        Servo dynamics filter commanded fin angles via rate-limited lag (§6.3.6).
        """
        p, v_b, q, omega, T, delta_actual = self._unpack(y)
        T_cmd, fin_deltas_cmd = u[0], u[1:5]

        R = quat_to_dcm(q)

        # --- Query environment ONCE (consistent rho + wind for all models) ---
        env_vars = self.env.sample_at_state(t, p)
        v_wind = env_vars['wind']       # inertial wind vector (m/s)
        rho = env_vars['rho']           # air density (kg/m³)

        # --- Servo dynamics: commanded → actual fin deflections (§6.3.6) ---
        delta_dot = self.servo_model.compute_rate(fin_deltas_cmd, delta_actual)

        # Forces and torques (use actual servo positions, not commands)
        F_thrust, tau_thrust, T_dot, omega_fan = self.thrust_model.compute(
            T, T_cmd, p[2], R, rho=rho)
        F_aero, tau_aero = self.aero_model.compute(
            v_b, R, v_wind, rho=rho)
        F_fins, tau_fins = self.fin_model.compute(
            delta_actual, omega_fan, v_b, omega, rho=rho)
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

        return np.concatenate([p_dot, v_dot, q_dot, omega_dot, [T_dot], delta_dot])

    def step(self, u: np.ndarray) -> np.ndarray:
        """RK4 integration step (18-state vector including servo positions)."""
        # ... (as in Section 7.5)

    def _unpack(self, y):
        return y[0:3], y[3:6], y[6:10], y[10:13], y[13], y[14:18]
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

> **Coordinate convention**: All positions are in the **body frame (FRD)** — x-forward, y-right, z-down. Origin is the geometric center of the airframe (pre-CoM shift). The aggregation code shifts all force application points relative to the computed CoM at init.

> **Units**: SI throughout — meters, kilograms, seconds, radians (except `orientation` fields which use degrees for human readability).

```yaml
vehicle:
  # ─── Global Simulation Parameters ─────────────────────────────────
  gravity: 9.81               # m/s²
  dt: 0.005                   # s, RK4 integration time step
  quat_normalize_interval: 10 # re-normalize quaternion every N steps

  # ─── Geometric Primitives ─────────────────────────────────────────
  # Each primitive defines a sub-component for:
  #   1. Mass property aggregation (mass, CoM, composite inertia via parallel axis theorem)
  #   2. Aerodynamic surface area aggregation (total wetted area, per-axis projected areas)
  # Composite inertia, CoM, and surface areas are computed ONCE at episode init.
  #
  # Fields per primitive:
  #   name:            human-readable identifier
  #   shape:           "cylinder" | "box" | "sphere"
  #   mass:            kg
  #   <dimensions>:    shape-specific (radius/height, dimensions [x,y,z], radius)
  #   position:        [x, y, z] m — local CoM in body frame (FRD)
  #   orientation:     [roll, pitch, yaw] deg — rotation relative to body frame (optional, default [0,0,0])
  #   surface_area:    m² — total wetted surface area (for skin friction / aero bookkeeping)
  #   drag_facing:     {x, y, z} m² — projected area along each body axis (for directional drag)
  #   Cd_local:        shape-specific drag coefficient (0.0 if shielded/internal)
  #   randomize_mass:  fractional perturbation per episode (optional, e.g., 0.10 = ±10%)

  primitives:
    # ── EDF Duct (outer shroud) ──
    - name: "edf_duct"
      shape: "cylinder"
      mass: 0.85               # kg (aluminium duct shroud)
      radius: 0.045            # m (90 mm outer diameter)
      height: 0.15             # m (duct axial length)
      position: [0, 0, 0.05]  # m, body frame — slightly below geometric center
      orientation: [0, 0, 0]   # aligned with body z-axis (thrust axis)
      surface_area: 0.0424     # m², 2πrh + 2πr² (lateral + end caps)
      drag_facing:
        x: 0.0135              # m², side profile (d × h)
        y: 0.0135              # m², side profile
        z: 0.00636             # m², top/bottom (πr²)
      Cd_local: 0.82           # bluff cylinder

    # ── EDF Motor + Rotor Assembly ──
    - name: "edf_motor_rotor"
      shape: "cylinder"
      mass: 0.35               # kg (motor stator + rotor + hub)
      radius: 0.040            # m
      height: 0.04             # m
      position: [0, 0, 0.06]  # m, centered inside duct
      orientation: [0, 0, 0]
      surface_area: 0.0        # internal — not exposed to external airstream
      drag_facing: { x: 0, y: 0, z: 0 }
      Cd_local: 0.0            # shielded by duct

    # ── 8S LiPo Battery Pack ──
    - name: "battery_8s_lipo"
      shape: "box"
      mass: 0.80               # kg
      dimensions: [0.15, 0.05, 0.04]  # m [length_x, width_y, height_z]
      position: [0.02, 0, -0.03]      # m, slightly forward and above CoM
      orientation: [0, 0, 0]
      surface_area: 0.031      # m², 2(lw + lh + wh)
      drag_facing:
        x: 0.002               # m², w × h
        y: 0.006               # m², l × h
        z: 0.0075              # m², l × w
      Cd_local: 1.05           # box/rectangular prism

    # ── Jetson Nano + Carrier Board ──
    - name: "jetson_nano"
      shape: "box"
      mass: 0.14               # kg
      dimensions: [0.08, 0.08, 0.03]  # m
      position: [-0.04, 0, -0.05]     # m, aft and above
      orientation: [0, 0, 0]
      surface_area: 0.0224     # m²
      drag_facing:
        x: 0.0024              # m²
        y: 0.0024              # m²
        z: 0.0064              # m²
      Cd_local: 1.05

    # ── Carbon Fibre Frame + 3D-Printed Joints ──
    - name: "frame_structure"
      shape: "cylinder"
      mass: 0.50               # kg
      radius: 0.06             # m (effective outer envelope)
      height: 0.30             # m (total airframe height)
      position: [0, 0, 0]     # m, centered at body frame origin
      orientation: [0, 0, 0]
      surface_area: 0.113      # m², lateral surface (2πrh)
      drag_facing:
        x: 0.036               # m², d × h
        y: 0.036               # m²
        z: 0.01131             # m², πr²
      Cd_local: 0.50           # semi-streamlined open truss

    # ── ESC (120 A) ──
    - name: "esc_120a"
      shape: "box"
      mass: 0.12               # kg
      dimensions: [0.07, 0.04, 0.02]  # m
      position: [0.03, 0.02, -0.02]   # m
      orientation: [0, 0, 0]
      surface_area: 0.010      # m²
      drag_facing:
        x: 0.0008              # m²
        y: 0.0014              # m²
        z: 0.0028              # m²
      Cd_local: 1.05

    # ── Wiring Harness + Connectors ──
    - name: "wiring_misc"
      shape: "sphere"
      mass: 0.10               # kg (aggregate)
      radius: 0.025            # m (effective bounding sphere)
      position: [0, 0, -0.02] # m
      surface_area: 0.00785    # m², 4πr²
      drag_facing:
        x: 0.00196             # m², πr²
        y: 0.00196             # m²
        z: 0.00196             # m²
      Cd_local: 0.47           # sphere

    # ── IMU (BNO085) ──
    - name: "imu_bno085"
      shape: "box"
      mass: 0.01               # kg
      dimensions: [0.02, 0.02, 0.005]  # m
      position: [0, 0, -0.06] # m, mounted high on frame (above CoM)
      orientation: [0, 0, 0]
      surface_area: 0.0012     # m²
      drag_facing: { x: 0, y: 0, z: 0 }
      Cd_local: 0.0            # negligible

    # ── Optical Flow Camera (PX4) ──
    - name: "optical_flow_camera"
      shape: "box"
      mass: 0.02               # kg
      dimensions: [0.03, 0.03, 0.01]  # m
      position: [0, 0, -0.10] # m, mounted on underside (top in FRD = above)
      orientation: [0, 0, 0]
      surface_area: 0.003      # m²
      drag_facing: { x: 0, y: 0, z: 0 }
      Cd_local: 0.0            # negligible

    # ── Servo Actuators (4× Freewing 9 g Digital Metal Gear for fin control) ──
    - name: "servo_fin_1"
      shape: "box"
      mass: 0.010              # kg (~9–12 g, Freewing 9 g servo)
      dimensions: [0.0235, 0.012, 0.027]  # m (23.5×12×27 mm)
      position: [0, 0.045, 0.10]   # m, right fin servo
      orientation: [0, 0, 0]
      surface_area: 0.0023     # m²
      Cd_local: 0.0            # inside duct shroud

    - name: "servo_fin_2"
      shape: "box"
      mass: 0.010
      dimensions: [0.0235, 0.012, 0.027]
      position: [0, -0.045, 0.10]  # m, left fin servo
      orientation: [0, 0, 0]
      surface_area: 0.0023
      Cd_local: 0.0

    - name: "servo_fin_3"
      shape: "box"
      mass: 0.010
      dimensions: [0.0235, 0.012, 0.027]
      position: [0.045, 0, 0.10]   # m, forward fin servo
      orientation: [0, 0, 0]
      surface_area: 0.0023
      Cd_local: 0.0

    - name: "servo_fin_4"
      shape: "box"
      mass: 0.010
      dimensions: [0.0235, 0.012, 0.027]
      position: [-0.045, 0, 0.10]  # m, aft fin servo
      orientation: [0, 0, 0]
      surface_area: 0.0023
      Cd_local: 0.0

    # ── Variable Payload (domain randomization target) ──
    - name: "payload_variable"
      shape: "sphere"
      mass: 0.20               # kg (water / ballast for CoM shift tests)
      radius: 0.03             # m
      position: [0, 0, -0.08] # m
      surface_area: 0.01131    # m², 4πr²
      drag_facing:
        x: 0.00283             # m², πr²
        y: 0.00283
        z: 0.00283
      Cd_local: 0.47
      randomize_mass: 0.10     # ±10% uniform perturbation per episode

  # ─── EDF Propulsion System ───────────────────────────────────────
  # Complete EDF spec for thrust model, ground effect, gyroscopic coupling.
  # Hardware: FMS 90 mm 12-Blade Metal Ducted Fan.
  edf:
    unit_name: "FMS 90mm 12-Blade Metal"
    fan_diameter: 0.090        # m
    duct_inner_diameter: 0.088 # m
    duct_outer_diameter: 0.090 # m
    blade_count: 12
    max_static_thrust: 45.0    # N (manufacturer rated, sea-level ISA)
    max_rpm: 95000             # RPM
    max_omega: 9948.0          # rad/s (= 95000 × 2π / 60)
    k_thrust: 4.55e-7          # N/(rad/s)², from T_max / ω_max² (calibrate from static test)
    k_torque: 1.0e-8           # N·m/(rad/s)², reaction torque coefficient (calibrate from test)
    I_fan: 0.001               # kg·m², fan rotor MoI about spin axis
    tau_motor: 0.10            # s, 1st-order thrust lag time constant
    r_thrust: [0, 0, 0.08]    # m, thrust application point (body frame)
    r_duct: 0.045              # m, duct radius (ground effect reference)
    exhaust_velocity_max: 80.0 # m/s, peak exhaust velocity at max RPM
    motor_kV: 1750             # RPM/V, motor velocity constant
    battery_cells: 8           # S count
    battery_voltage_nominal: 29.6  # V (8 × 3.7 V)
    battery_voltage_full: 33.6     # V (8 × 4.2 V)
    esc_max_current: 120       # A
    esc_type: "120A BLHeli_32"

  # ─── Aerodynamic Properties (Combined Shape) ────────────────────
  aero:
    # NOTE: rho is provided per-step by EnvironmentModel (see env.md §4).
    # The vehicle config no longer specifies a fixed rho.
    Cd: 0.7                    # overall bluff body drag coefficient (tunable)
    r_cp: [0, 0, 0.05]        # m, center of pressure (body frame)

    # Per-axis projected area and total surface area are auto-computed from
    # primitives at init. These fallback values are used if auto-compute is disabled.
    A_proj: 0.01               # m², fallback projected frontal area
    compute_directional_drag: true  # aggregate per-axis drag_facing from primitives
    total_surface_area: null   # m², auto-computed from Σ primitive surface_area_i

  # ─── Fin Control Surfaces ────────────────────────────────────────
  # 4 NACA 0012 fins at 90° intervals in the EDF exhaust stream.
  # Hardware: Freewing 9 g digital metal gear servo (4.8–6.0 V).
  fins:
    count: 4

    # ── Fin Airfoil & Planform Geometry ──
    airfoil_profile: "NACA0012"       # symmetric, 12% thickness
    chord: 0.050                       # m, along exhaust flow direction
    span: 0.040                        # m, radial extent into exhaust stream
    thickness_ratio: 0.12              # t/c (NACA 0012)
    max_thickness: 0.006               # m, = chord × thickness_ratio
    planform_area: 0.002               # m², = chord × span (per fin)
    wetted_area: 0.0041                # m², ≈ 2.06 × planform_area (both surfaces)

    # ── Aerodynamic Coefficients ──
    Cl_alpha: 6.283                    # /rad, thin-airfoil lift slope (2π)
    Cd0: 0.01                          # parasitic drag coefficient
    AR: 0.80                           # aspect ratio = span² / planform_area
    stall_angle: 0.262                 # rad (15°), onset of C_L plateau
    post_stall_model: "tanh"           # "tanh" soft clamp or "flat" plateau

    # ── Deflection Limits ──
    max_deflection: 0.349              # rad (±20°), mechanical servo travel limit
    rate_limit: 10.5                   # rad/s (~600°/s at 6 V), Freewing 9 g digital servo no-load speed

    # ── Servo Actuator Dynamics (Freewing 9 g Digital Metal Gear) ──
    # Mandatory — models physical slew lag. Delete instant-action assumption.
    # See §6.3.6 for derivation from product data.
    servo:
      type: "digital"                  # digital hobby servo (not programmable high-rate)
      gear_train: "full_metal"         # low backlash, high stiffness
      supply_voltage: 6.0              # V, assumed LiPo supply
      transit_time_60deg: 0.10         # s, at 6 V (0.12 s at 4.8 V)
      max_angular_velocity: 10.5       # rad/s (~600°/s at 6 V)
      torque: 0.186                    # N·m (~1.9 kg·cm at 6 V)
      tau_servo: 0.04                  # s, effective first-order position lag time constant
      tau_servo_range: [0.03, 0.05]    # s, DR randomization range (±20%)
      bandwidth_hz: 13.0               # Hz, closed-loop position bandwidth estimate
      pwm_frame_rate: 50               # Hz, standard PWM (50–333 Hz unofficially tolerated)
      deadband_us: 2                   # µs, digital servo deadband
      weight_kg: 0.010                 # kg (~9–12 g)
      dimensions_mm: [23.5, 12, 27]    # mm [length, width, height]
      aero_load_derating: 0.3          # fraction — effective slew 20–50% slower under exhaust load

    # ── Exhaust Interaction ──
    V_exhaust_nominal: 70              # m/s at full RPM
    exhaust_velocity_ratio: true       # V_exhaust scales linearly with ω_fan / ω_fan_max

    # ── Per-Fin Configuration (body frame) ──
    # Each fin defines its position, hinge axis, lift/drag direction vectors,
    # and azimuthal position around the duct for wind-relative sideslip computation.
    fins_config:
      - name: "fin_1_right"
        position: [0, 0.04, 0.12]     # m, body frame
        hinge_axis: [1, 0, 0]         # deflection rotates about body x-axis
        lift_direction: [1, 0, 0]     # lift acts in body x-direction
        drag_direction: [0, 0, 1]     # drag opposes exhaust (body z)
        angular_offset_deg: 0         # azimuthal position (0° = starboard)

      - name: "fin_2_left"
        position: [0, -0.04, 0.12]
        hinge_axis: [1, 0, 0]
        lift_direction: [1, 0, 0]
        drag_direction: [0, 0, 1]
        angular_offset_deg: 180

      - name: "fin_3_forward"
        position: [0.04, 0, 0.12]
        hinge_axis: [0, 1, 0]
        lift_direction: [0, 1, 0]
        drag_direction: [0, 0, 1]
        angular_offset_deg: 90

      - name: "fin_4_aft"
        position: [-0.04, 0, 0.12]
        hinge_axis: [0, 1, 0]
        lift_direction: [0, 1, 0]
        drag_direction: [0, 0, 1]
        angular_offset_deg: 270

  # ─── Computed Properties (auto-populated at init) ────────────────
  # These are computed from primitives via MassProperties aggregation.
  # Listed here for documentation and validation. Do NOT set manually
  # — they are overwritten by compute_mass_properties() at init.
  #
  # After CAD import (see §9.3), these can be cross-checked against
  # Autodesk Inventor's iProperties or Isaac Sim's rigid body inspector.
  _computed:
    total_mass: null           # kg, Σ m_i
    center_of_mass: null       # [x, y, z] m, mass-weighted centroid (body frame)
    inertia_tensor: null       # [[Ixx, Ixy, Ixz], [Iyx, Iyy, Iyz], [Izx, Izy, Izz]] kg·m²
    inertia_tensor_inv: null   # (I)^{-1}, precomputed for Euler's equation
    principal_inertias: null   # [Ixx, Iyy, Izz] kg·m² (if off-diag < 5% of diag)
    total_surface_area: null   # m², Σ surface_area_i
    projected_area_x: null     # m², Σ drag_facing.x_i (side profile)
    projected_area_y: null     # m², Σ drag_facing.y_i (side profile)
    projected_area_z: null     # m², Σ drag_facing.z_i (top/bottom profile)

  # ─── Mass Override (from CAD or measured) ────────────────────────
  # If a validated CAD model or physical measurement provides mass properties,
  # set use_cad_override: true and populate the fields below.
  # When enabled, these replace the primitive-based aggregation entirely.
  # See §9.3 for the Autodesk Inventor export workflow.
  cad_override:
    use_cad_override: false
    source: null               # "inventor" | "solidworks" | "measured" | null
    source_file: null          # path to CAD export (e.g., "cad/drone_assembly.step")
    total_mass: null           # kg
    center_of_mass: null       # [x, y, z] m, in body frame
    inertia_tensor: null       # 3×3 kg·m², about CoM in body frame axes
    total_surface_area: null   # m²

  # NOTE: Wind and atmosphere config has been migrated to default_environment.yaml
  # (see env.md §7). The vehicle no longer owns wind models — EnvironmentModel
  # provides wind vectors and air density per integration step.
```

### 9.1.1 Primitive Summary Table

For quick reference, the following table summarizes the mass budget from the primitives above. These values are **design estimates** pending hardware measurement or CAD validation (see §9.3).

| Primitive | Shape | Mass (kg) | Position [x, y, z] (m) | Key Dimension |
|---|---|---|---|---|
| edf_duct | cylinder | 0.85 | [0, 0, 0.05] | r=45 mm, h=150 mm |
| edf_motor_rotor | cylinder | 0.35 | [0, 0, 0.06] | r=40 mm, h=40 mm |
| battery_8s_lipo | box | 0.80 | [0.02, 0, −0.03] | 150×50×40 mm |
| jetson_nano | box | 0.14 | [−0.04, 0, −0.05] | 80×80×30 mm |
| frame_structure | cylinder | 0.50 | [0, 0, 0] | r=60 mm, h=300 mm |
| esc_120a | box | 0.12 | [0.03, 0.02, −0.02] | 70×40×20 mm |
| wiring_misc | sphere | 0.10 | [0, 0, −0.02] | r=25 mm |
| imu_bno085 | box | 0.01 | [0, 0, −0.06] | 20×20×5 mm |
| optical_flow_camera | box | 0.02 | [0, 0, −0.10] | 30×30×10 mm |
| servo_fin × 4 | box | 4 × 0.010 | ±45 mm from center | 23.5×12×27 mm (Freewing 9 g) |
| payload_variable | sphere | 0.20 ± 10% | [0, 0, −0.08] | r=30 mm |
| **Total** | | **~3.17 kg** | | |

### 9.2 Config Loading

```python
import yaml

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)
```

### 9.3 CAD Import Workflow — Autodesk Inventor

> **Premise**: A 1:1 CAD model of the drone exists in Autodesk Inventor. This model can provide validated mass properties (total mass, center of mass, inertia tensor, surface area) that either **cross-validate** the primitive-based aggregation or **replace** it entirely via `cad_override`.

#### 9.3.1 What Inventor Provides

Inventor's **Physical** iProperties (per component and at assembly level) export:

| Property | Inventor Location | Maps To |
|---|---|---|
| Mass | iProperties → Physical → Mass | `cad_override.total_mass` |
| Center of Gravity | iProperties → Physical → Center of Gravity (x, y, z) | `cad_override.center_of_mass` |
| Moments of Inertia | iProperties → Physical → Moments of Inertia (Ixx, Iyy, Izz, Ixy, Ixz, Iyz) | `cad_override.inertia_tensor` |
| Surface Area | iProperties → Physical → Surface Area | `cad_override.total_surface_area` |
| Volume | iProperties → Physical → Volume | (diagnostic only) |

**Critical**: Inventor reports MoI about the **assembly origin** by default. To get MoI about CoM, either:
1. Move the assembly origin to coincide with the computed CoG, then re-read MoI.
2. Use the parallel axis theorem to shift: $\mathbf{I}_{CoM} = \mathbf{I}_{origin} - m\left((\mathbf{d} \cdot \mathbf{d})\,\mathbf{1}_3 - \mathbf{d}\,\mathbf{d}^T\right)$ where $\mathbf{d}$ is the CoG position from the origin.

#### 9.3.2 Per-Component Export (for Primitive Validation)

For each sub-component (EDF duct, battery, Jetson, etc.):
1. Open the component in Inventor.
2. Assign the correct material/density (or override mass manually for non-standard parts).
3. Read iProperties → Physical → Mass, CoG, MoI.
4. Compare against the corresponding primitive entry in the YAML config.
5. Update the primitive's mass, position, and dimensions if they deviate by > 5%.

#### 9.3.3 Assembly-Level Export (for CAD Override)

1. Open the full drone assembly.
2. Ensure all component materials/masses are set correctly.
3. Align the assembly coordinate system to the body frame convention (FRD: x-forward, y-right, z-down).
4. Export iProperties:
   - **Mass** → `cad_override.total_mass`
   - **Center of Gravity** → `cad_override.center_of_mass` (transform to body frame if needed)
   - **Moments of Inertia** → `cad_override.inertia_tensor` (6 unique values → 3×3 symmetric)
   - **Surface Area** → `cad_override.total_surface_area`
5. Set `cad_override.use_cad_override: true` in the config.

#### 9.3.4 Automated Export Script

Inventor's iLogic or COM API can automate this. A Python script using `win32com` can extract properties programmatically:

```python
def export_inventor_mass_props(inventor_file: str) -> dict:
    """Extract mass properties from Inventor assembly via COM API.

    Requires: Autodesk Inventor installed, pywin32 (win32com).
    Coordinate system: must match body frame (FRD).
    """
    import win32com.client
    inv = win32com.client.Dispatch("Inventor.Application")
    doc = inv.Documents.Open(inventor_file)
    mass_props = doc.ComponentDefinition.MassProperties

    result = {
        'total_mass': mass_props.Mass,                   # kg
        'center_of_mass': [
            mass_props.CenterOfMass.X / 100,             # cm → m
            mass_props.CenterOfMass.Y / 100,
            mass_props.CenterOfMass.Z / 100,
        ],
        'inertia_tensor': [
            [mass_props.MomentsOfInertia.Ixx / 1e4,     # g·cm² → kg·m²
             mass_props.MomentsOfInertia.Ixy / 1e4,
             mass_props.MomentsOfInertia.Ixz / 1e4],
            [mass_props.MomentsOfInertia.Ixy / 1e4,
             mass_props.MomentsOfInertia.Iyy / 1e4,
             mass_props.MomentsOfInertia.Iyz / 1e4],
            [mass_props.MomentsOfInertia.Ixz / 1e4,
             mass_props.MomentsOfInertia.Iyz / 1e4,
             mass_props.MomentsOfInertia.Izz / 1e4],
        ],
        'total_surface_area': mass_props.Area / 1e4,     # cm² → m²
        'source': 'inventor',
        'source_file': inventor_file,
    }
    doc.Close()
    return result
```

#### 9.3.5 STEP Export for Isaac Sim

Inventor can export the assembly as STEP (.stp) for import into Isaac Sim:
1. File → Export → CAD Format → STEP (AP214 or AP203).
2. Import into Isaac Sim via Omniverse's CAD importer (STEP → USD).
3. The USD asset preserves geometry for visualization; mass properties come from the YAML config (not from the USD mesh).

#### 9.3.6 Validation Protocol

| Check | Primitive-Based | CAD (Inventor) | Pass If |
|---|---|---|---|
| Total mass | Σ m_i from YAML | iProperties.Mass | Δ < 5% |
| CoM position | Mass-weighted centroid | iProperties.CoG | Δ < 5 mm per axis |
| Ixx, Iyy, Izz | Parallel axis aggregation | iProperties.MoI | Δ < 10% |
| Off-diagonal Ixy, Ixz, Iyz | Parallel axis aggregation | iProperties.MoI | Same sign, Δ < 20% |
| Surface area | Σ surface_area_i | iProperties.Area | Δ < 15% (primitives are approximations) |

If mass and CoM match within 5%, the primitive model is adequate for training. If inertia deviates > 10%, use `cad_override` for more accurate MoI.

### 9.4 Isaac Sim Primitive Export Workflow

> **Use case**: Build the drone's primitive geometry directly in Isaac Sim's Composer, position shapes visually, then export transforms back to the YAML config. This is the reverse of loading the config — it lets you design the layout visually and generate the config automatically.

#### 9.4.1 Creating Primitives in Isaac Sim

1. Open Isaac Sim (Omniverse Composer).
2. For each vehicle component, create a primitive shape:
   - `Create → Shape → Cylinder` for ducts, frame tubes.
   - `Create → Shape → Cube` for battery, Jetson, ESC.
   - `Create → Shape → Sphere` for payload, wiring aggregate.
3. Name each prim to match the YAML `name` field (e.g., `/World/drone/edf_duct`).
4. Set transforms (position, rotation, scale) using the Property panel to match the physical layout.
5. Assign visual materials for rendering (optional, not used in physics).

#### 9.4.2 Exporting Transforms to YAML

Write a Python script that runs inside Isaac Sim's Script Editor to read the USD stage and generate the YAML config:

```python
from pxr import Usd, UsdGeom
import yaml

def export_prims_to_yaml(stage_path: str, drone_root: str = "/World/drone") -> dict:
    """Export primitive transforms from Isaac Sim USD stage to YAML config format.

    Run inside Isaac Sim's Script Editor or via standalone Python with USD libraries.
    """
    stage = Usd.Stage.Open(stage_path)
    root = stage.GetPrimAtPath(drone_root)
    primitives = []

    for child in root.GetChildren():
        xform = UsdGeom.Xformable(child)
        translate = xform.GetLocalTransformation().GetRow3(3)  # translation
        name = child.GetName()

        prim_data = {
            'name': name,
            'position': [
                float(translate[0]),    # x (forward)
                float(translate[1]),    # y (right)
                float(translate[2]),    # z (down)
            ],
        }

        # Detect shape type and extract dimensions
        if child.IsA(UsdGeom.Cylinder):
            cyl = UsdGeom.Cylinder(child)
            prim_data['shape'] = 'cylinder'
            prim_data['radius'] = float(cyl.GetRadiusAttr().Get())
            prim_data['height'] = float(cyl.GetHeightAttr().Get())
        elif child.IsA(UsdGeom.Cube):
            cube = UsdGeom.Cube(child)
            scale = xform.GetLocalTransformation().GetRow3(0)  # x-scale
            prim_data['shape'] = 'box'
            size = float(cube.GetSizeAttr().Get())
            prim_data['dimensions'] = [size, size, size]  # adjust with scale
        elif child.IsA(UsdGeom.Sphere):
            sphere = UsdGeom.Sphere(child)
            prim_data['shape'] = 'sphere'
            prim_data['radius'] = float(sphere.GetRadiusAttr().Get())

        # Mass, surface_area, drag_facing, Cd_local must be added manually
        # or from a lookup table keyed by prim name.
        prim_data['mass'] = 0.0          # FILL FROM BOM
        prim_data['surface_area'] = 0.0  # COMPUTE FROM DIMENSIONS
        prim_data['Cd_local'] = 0.0      # ASSIGN FROM SHAPE TYPE

        primitives.append(prim_data)

    return {'vehicle': {'primitives': primitives}}
```

#### 9.4.3 Round-Trip Workflow

The recommended workflow combines Isaac Sim (visual layout) with CAD (mass validation):

```
Autodesk Inventor (1:1 CAD)          Isaac Sim (Composer)
        │                                     │
        ├─ STEP export ──────────────────────> │ USD import (visual mesh)
        │                                     │
        ├─ iProperties ──> mass, CoM, MoI     ├─ Primitive layout (visual positioning)
        │       │                             │       │
        │       v                             │       v
        │  cad_override {}                    │  export_prims_to_yaml()
        │       │                             │       │
        │       └──────────> YAML config <────┘───────┘
        │                        │
        │                        v
        │               MassProperties.compute()
        │                    (at init)
        │                        │
        │                        v
        │              Cross-validate: primitives vs. CAD
        │              (see §9.3.6 Validation Protocol)
```

**Key principle**: The YAML config is the **single source of truth** for the simulation. CAD and Isaac Sim are authoring tools that feed into it. The simulation code reads only the YAML.

#### 9.4.4 When to Use Which Path

| Scenario | Recommended Path | Rationale |
|---|---|---|
| No CAD model yet | Primitives only (manual YAML) | Fast iteration, good enough for Phase 1 |
| CAD model exists, no Isaac | Inventor export → `cad_override` + primitives for aero | Best mass accuracy |
| CAD model + Isaac needed | STEP → USD (viz) + Inventor → YAML (mass) | Visual fidelity + accurate physics |
| Design iteration (moving parts around) | Isaac Sim layout → export transforms → merge into YAML | Visual feedback loop |
| Final validated config | Inventor iProperties as ground truth, primitives as backup | Production config |

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
1. Numerically compute Jacobians $A = \partial f / \partial y$, $B = \partial f / \partial u$ via finite differences.
2. Check eigenvalues of $A$: all real parts negative (open-loop hover is unstable for rocket-like vehicle — expect positive real eigenvalues in pitch/roll, confirming need for active control).
3. Verify that the fastest mode has frequency < 1/(2·dt) to confirm the time step is adequate.
4. Optional: Compute controllability matrix rank to verify full controllability with 5 inputs (thrust + 4 fins).

### 10.4 Energy Conservation Check

For a torque-free, drag-free case (disable all forces except gravity):
- Total mechanical energy $E = \frac{1}{2}m\|\mathbf{v}\|^2 + mgh$ should be conserved by RK4.
- Integrate for 100 s, check $|E(t) - E(0)| / E(0) < 10^{-6}$ (RK4 is not symplectic but should conserve well for short durations at dt=0.005 s).

---

## 11. Isaac Sim Integration — Phased Hybrid Workflow

> **Governing document**: See [master_plan.md](../master_plan.md) for the full 3-phase simulation strategy.
> This section specifies how the vehicle dynamics module operates in each phase.

### 11.1 Three Deployment Modes

| Mode | Phase | Use Case | Integration Strategy |
|---|---|---|---|
| **Pure NumPy** | Phase 1 | Fast prototyping, unit testing, baseline RL training | `VehicleDynamics` class runs standalone; RK4 integration; no Isaac dependency |
| **JAX-Vectorized** | Phase 1 (scaling) | GPU-batched parallel envs (32–256) without Isaac overhead | `jax.vmap` over `VehicleDynamics.step()`; same physics, GPU-parallel |
| **Isaac Sim (OIGE)** | Phase 2 (conditional) | GPU-parallel validation, contact physics, sensor fidelity, HIL bridge | Map to USD rigid body; PhysX handles contact; custom force callbacks for thrust/fins/aero |

### 11.2 Phase 1: Python-Only Vehicle Usage

In Phase 1, the vehicle dynamics module is consumed by a standard Gymnasium environment wrapper:

```python
class EDFLandingEnv(gymnasium.Env):
    def __init__(self, config):
        self.vehicle = VehicleDynamics(config['vehicle'], EnvironmentModel(config['environment']))
        # Gym spaces, reward config, etc.

    def step(self, action):
        u = self._scale_action(action)
        for _ in range(self.substeps):  # 5 physics steps per policy step at 40 Hz
            self.vehicle.step(u)
        # ... obs, reward, done
```

**Parallelization options** (all use identical physics):

| Method | Envs | Est. Steps/sec | Hardware Used |
|---|---|---|---|
| Serial NumPy | 1 | ~10k | 1 core (Ryzen 9 9900X) |
| SB3 `SubprocVecEnv` | 12–24 | ~100–500k | Ryzen 9 9900X 12C/24T |
| Ray RLlib workers | 12–48 | ~200–800k | Ryzen 9 9900X + Ray cluster-ready |
| JAX `vmap` | 64–512 | ~1–4M | RTX 5070 (Blackwell, 12 GB GDDR7) |

**JAX compatibility requirements**: If JAX vectorization is used, `derivs()` must avoid Python control flow and dynamic shapes. Refactor: replace `np.cross()` with explicit component math, replace `if` guards with `jnp.where()`.

### 11.3 Phase 2: Isaac Sim Callback Pattern (Conditional)

> **Entry criteria**: See [master_plan.md §4.2](../master_plan.md) — only proceed if sample bottleneck, contact fidelity gap, or HIL readiness triggers.

In Isaac Sim mode, PhysX handles rigid-body integration (implicit Euler, not our RK4), but all force/torque models remain **identical** to the Python implementation. This ensures the sim logic is consistent across platforms; only the integrator differs.

```python
class EDFLandingTask(RLTask):
    """OIGE task for EDF landing. Custom forces applied via callbacks."""

    def pre_physics_step(self, actions):
        T_cmd = actions[:, 0]
        fin_deltas = actions[:, 1:5]

        # Query environment (batched over num_envs)
        env_vars = self.env_model.sample_batched(self.time, self.positions)
        v_wind = env_vars['wind']   # (num_envs, 3)
        rho = env_vars['rho']       # (num_envs,)

        # Compute forces from identical models (batched)
        F_thrust, tau_thrust = self.thrust_model.compute_batched(
            self.thrust_state, T_cmd, self.altitudes, rho=rho)
        F_aero, tau_aero = self.aero_model.compute_batched(
            self.velocities, v_wind, rho=rho)
        F_fins, tau_fins = self.fin_model.compute_batched(
            fin_deltas, self.omega_fan, rho=rho)
        tau_motor = self.thrust_model.reaction_torque_batched(
            self.omega_fan, self.thrust_dot)

        # Apply to rigid body via Isaac Sim API
        self._drones.apply_forces(F_thrust + F_aero + F_fins, is_global=False)
        self._drones.apply_torques(
            tau_thrust + tau_aero + tau_fins + tau_motor, is_global=False)
```

**Key difference from Python mode**: PhysX implicit Euler is $O(dt^2)$ vs. RK4's $O(dt^5)$. For trajectory accuracy (jerk, CEP), Python is more precise. Isaac wins on contact physics (friction, bounce on landing pad). The hybrid approach uses Python for training and Isaac for terminal-phase contact validation.

**GPU budget (RTX 5070 — 12 GB GDDR7, 672 GB/s, 6144 CUDA cores)**:

| `num_envs` | Est. VRAM | RTX 5070 Feasibility |
|---|---|---|
| 256 | ~2–3 GB | Comfortable (room for renderer + debug) |
| 512 | ~4–6 GB | Safe (headless) — recommended Isaac ceiling |
| 1024 | ~5–8 GB | Feasible headless, profile before committing |
| 2048 | ~8–12 GB | Tight — risk of fragmentation-induced OOM |
| 4096 | ~12–18 GB | OOM likely — delete this ambition on 12 GB |

Blackwell architecture and GDDR7 bandwidth (672 GB/s) improve per-env throughput vs. previous-gen, but the 12 GB capacity ceiling is the hard constraint for `num_envs` scaling.

### 11.4 Integration Accuracy Comparison

| Aspect | Python (RK4) | Isaac (PhysX) | Implication |
|---|---|---|---|
| Local truncation error | $O(dt^5)$ | $O(dt^2)$ | Python more precise for smooth trajectories |
| Contact handling | Spring-damper approximation | Built-in friction + normal force | Isaac better for landing touchdown |
| Determinism | Fully deterministic (seeded RNG) | PhysX may have GPU non-determinism | Python better for reproducible RL |
| Jerk computation | Exact via RK4 intermediate stages | Numerical differentiation of PhysX state | Python more reliable for jerk metrics |
| Ground effect | Analytical $1 + 0.5(r/h)^2$ | Same (via callback) | Equivalent |

**Decision**: Train in Python for accuracy, validate in Isaac for contact fidelity. If touchdown velocity mismatch between platforms > 0.1 m/s over 100 episodes, use Isaac for terminal phase only.

### 11.5 Sensor Noise (Post-Sim, Both Modes)

Sensor simulation is **not** part of the plant model. After each physics step, apply noise to observations:

| Sensor | Phase 1 (Python) | Phase 2 (Isaac) | Notes |
|---|---|---|---|
| **IMU accel** | Gaussian, σ = 0.1 m/s² | PhysX IMU sensor (bias + drift + noise) | Gaussian calibrated to BNO085 datasheet |
| **IMU gyro** | Gaussian, σ = 0.01 rad/s | PhysX IMU sensor (bias + drift) | Short episodes (10 s) limit drift accumulation |
| **Optical flow** | Gaussian position, σ = 0.1 m, 1 Hz sample-and-hold | Camera raytraced ground truth + noise | Isaac adds occlusion/lighting effects |
| **Barometer** | Gaussian altitude, σ = 0.5 m | Pressure-coupled to AtmosphereModel | Minimal difference for indoor tests |

**Deletion criterion**: If ablation shows < 2% success rate difference between Python Gaussian noise and Isaac structured noise, delete Isaac sensor emulation — the Gaussian approximation is sufficient for indoor controlled tests.

### 11.6 HIL Pipeline (Phase 2, If Triggered)

Isaac's strongest value-add is the HIL bridge to the Jetson Nano:

```
Isaac Sim (headless) ──[synthetic sensors]──> Simulink ──[UDP/ROS]──> Jetson Nano
                                                                          │
                                                                   Trained policy
                                                                   (ONNX/TensorRT)
                                                                          │
                                                        Control outputs ──┘
```

- **Sensor feed**: IMU, optical flow, barometer at 100 Hz
- **Latency target**: < 50 ms end-to-end (sensor → policy → actuator command)
- **Trial volume**: ~500 HIL trials per controller variant
- **Transfer fidelity check**: Pearson r > 0.9 between Isaac predictions and hardware measurements
- **Fallback**: If Isaac is deleted, build custom Python → Jetson socket bridge (adds 1–2 weeks)

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
│   ├── servo_model.py          # ServoModel (4x rate-limited first-order lag, Freewing 9 g)
│   ├── integrator.py           # RK4 fixed-step integrator
│   └── quaternion_utils.py     # quat_to_dcm, quat_mult, quat_normalize
├── environment/                # ← See env.md for full specification
│   ├── __init__.py
│   ├── environment_model.py    # EnvironmentModel (top-level, owns sub-models)
│   ├── wind_model.py           # WindModel + DrydenFilter
│   └── atmosphere_model.py     # AtmosphereModel (ISA + randomization)
├── training/                   # ← Phase 1 (master_plan.md §3) — see training.md §13.1 for full structure
│   ├── edf_landing_env.py      # EDFLandingEnv (Gymnasium wrapper)
│   ├── reward.py               # RewardFunction class (potential-based shaping + terminal)
│   ├── observation.py          # Observation computation + sensor noise
│   ├── curriculum.py           # CurriculumScheduler (optional)
│   ├── controllers/            # All controller variants (PPO-MLP, GTrXL-PPO, PID, SCP)
│   │   ├── base.py             # Controller ABC
│   │   ├── ppo_mlp.py          # PPO-MLP wrapper (SB3)
│   │   ├── gtrxl_ppo.py        # GTrXL-PPO wrapper (RLlib)
│   │   ├── pid_controller.py   # PID with gain scheduling
│   │   └── scp_controller.py   # SCP with CVXPY
│   ├── scripts/                # Training, evaluation, and comparison entry points
│   └── configs/                # Per-controller YAML configs (see training.md §15)
├── isaac/                      # ← Phase 2, conditional (master_plan.md §4)
│   ├── edf_landing_task.py     # OIGE RLTask with force callbacks
│   ├── usd/                    # USD assets for EDF drone
│   └── configs/                # Isaac-specific training configs
├── configs/
│   ├── default_vehicle.yaml    # Vehicle-only config (no wind/atmo — see env config)
│   ├── default_environment.yaml # Wind + atmosphere config (see env.md)
│   ├── test_vehicle.yaml       # Simplified config for unit tests
│   └── test_environment.yaml   # Zero-wind, ISA-only config for deterministic tests
└── tests/
    ├── test_mass_properties.py
    ├── test_thrust_model.py
    ├── test_aero_model.py
    ├── test_fin_model.py
    ├── test_integrator.py
    ├── test_quaternion_utils.py
    ├── test_vehicle_dynamics.py # Full integration tests
    ├── test_wind_model.py       # Wind model unit tests (see env.md §8.2)
    ├── test_atmosphere_model.py # Atmosphere unit tests (see env.md §8.1)
    └── test_environment_model.py
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

> **Alignment**: This implementation order maps to the 3-phase simulation master plan. See [master_plan.md](../master_plan.md) for the full training strategy.

#### Pre-Phase 1: Core Module Build (10–12 Days)

| Step | Files | Dependency | Est. Effort |
|---|---|---|---|
| 1 | `quaternion_utils.py` | None | 0.5 day |
| 2 | `mass_properties.py` + tests | Step 1 | 1 day |
| 3 | `thrust_model.py` + tests (add `rho` param for density correction) | None | 1 day |
| 4 | `aero_model.py` + tests (accepts `rho`, `v_wind` from env) | Step 1 | 0.5 day |
| 5 | `fin_model.py` + tests (accepts `rho` from env) | Step 1 | 1 day |
| 5b | `servo_model.py` + tests (rate-limited first-order lag, Freewing 9 g) | None | 1 day |
| 6 | `integrator.py` (18-state vector: 13 + 1 thrust + 4 servo) | None | 0.5 day |
| 7 | **EnvironmentModel** (`environment/` module) — see [env.md](../Enviornment/env.md) | None | 2.5 days |
| 8 | `vehicle.py` (assembly + ServoModel integration, accepts `EnvironmentModel`) + integration tests | Steps 1–7 | 1.5 days |
| 9 | Config files (`default_vehicle.yaml`, `test_vehicle.yaml`) + linearization analysis | Step 8 | 1 day |
| | **Subtotal** | | **~10.5 days** |

#### Master Plan Phase 1: Python-Only Training (1–2 Weeks)

> **See [training.md](../Training%20Plan/training.md) for complete training pipeline specification** including all four controller variants, reward function, observation/action spaces, and evaluation protocol.

| Step | Files | Dependency | Est. Effort |
|---|---|---|---|
| 10 | `EDFLandingEnv` Gymnasium wrapper + `RewardFunction` + obs pipeline | Steps 7–8 | 2 days |
| 11 | PPO-MLP training (10M steps) + Ray Tune sweep | Step 10 | 3–5 days |
| 12 | GTrXL-PPO training (15M steps, conditional — only if MLP fails on disturbance recovery) | Step 10 | 2–3 days |
| 13 | PID tuning (Ziegler-Nichols + grid search) | Step 10 | ~1 day |
| 14 | SCP configuration (CVXPY solver setup + trust region tuning) | Step 10 | ~2 days |
| 15 | Multi-controller evaluation (400 episodes per controller, 4 conditions) + ablation studies | Steps 11–14 | 2–3 days |
| | **Subtotal** | | **~12–16 days** |

#### Master Plan Phase 2: Isaac Integration (2–3 Weeks, Conditional)

> **Entry criteria**: Sample bottleneck, contact fidelity gap, or HIL readiness. See [master_plan.md §4.2](../master_plan.md).

| Step | Files | Dependency | Est. Effort |
|---|---|---|---|
| 14 | OIGE task class + USD rigid body asset | Step 8 | 5–7 days |
| 15 | Force callback wrapper (batched `compute()` for thrust/aero/fins) | Steps 3–5, 14 | 2–3 days |
| 16 | Fine-tune from Python checkpoint in Isaac | Steps 11, 14 | 3–5 days |
| 17 | HIL bridge (Simulink + Jetson deployment, if triggered) | Step 14 | 5–7 days |
| | **Subtotal** | | **~15–22 days** |

#### Master Plan Phase 3: Evaluation (1 Week)

| Step | Files | Dependency | Est. Effort |
|---|---|---|---|
| 18 | Comparison report: Python vs. Isaac metrics | Steps 11–16 | 2–3 days |
| 19 | Final platform decision + converged checkpoint | Step 18 | 1 day |
| 20 | Training logs and ablation results for thesis Appendix | Steps 13, 18 | 1–2 days |
| | **Subtotal** | | **~4–6 days** |

#### Totals

| Scenario | Total Effort |
|---|---|
| Full pipeline (all phases) | ~38–53 days (~7–10 weeks) |
| Phase 1 only + eval (skip Isaac training) | ~23–31 days (~4–6 weeks) |
| Minimum viable (core build + Phase 1 baseline) | ~19–25 days (~4–5 weeks) |

---

## Appendix A: Derivative Function — Complete Pseudocode

For reference, the complete derivative function assembling all terms. Note: wind and $\rho$ are provided by the **EnvironmentModel** ([env.md](../Enviornment/env.md)), not computed internally.

```python
def derivs(y, u, t, params, env):
    """
    Complete state derivative for 6-DOF EDF drone with servo dynamics.

    y:      [p(3), v_b(3), q(4), omega(3), T(1), delta_actual(4)]  -- 18 scalars
    u:      [T_cmd, delta_cmd_1, delta_cmd_2, delta_cmd_3, delta_cmd_4]  -- 5 scalars
    t:      current time (s)
    params: vehicle parameters (mass, inertia, thrust config, servo config, etc.)
    env:    EnvironmentModel instance (provides wind, rho — see env.md)
    """
    # Unpack state
    p            = y[0:3]        # inertial position (NED)
    v_b          = y[3:6]        # body velocity
    q            = y[6:10]       # quaternion (scalar-first)
    omega        = y[10:13]      # body angular rate
    T            = y[13]         # current thrust (with lag)
    delta_actual = y[14:18]      # actual servo positions (4 fins)

    T_cmd          = u[0]
    fin_deltas_cmd = u[1:5]      # commanded fin deflections

    # Rotation matrix (body -> inertial)
    R = quat_to_dcm(q)

    # --- Query environment ONCE (consistent rho + wind for all models) ---
    env_vars = env.sample_at_state(t, p)
    v_wind = env_vars['wind']       # inertial wind vector (m/s)
    rho = env_vars['rho']           # air density (kg/m³)

    # --- Servo dynamics (§6.3.6): commanded → actual fin deflections ---
    # Rate-limited first-order lag for each servo
    servo = params.fins.servo
    rate_max_eff = servo.max_angular_velocity * (1.0 - servo.aero_load_derating)
    servo_error = fin_deltas_cmd - delta_actual
    rate_desired = servo_error / servo.tau_servo
    delta_dot = np.clip(rate_desired, -rate_max_eff, rate_max_eff)

    # --- Thrust (with density correction) ---
    omega_fan = np.sqrt(max(T, 0) / params.edf.k_thrust)
    h_alt = -p[2]  # altitude (NED: z is down, so altitude = -z)
    rho_ratio = rho / env.atmo_model.rho_ref
    T_eff = T * ground_effect(h_alt, params.edf.r_duct) * rho_ratio
    F_thrust = np.array([0, 0, T_eff])
    r_offset = params.edf.r_thrust - params.com
    tau_thrust = np.cross(r_offset, F_thrust)

    # Thrust lag ODE
    T_dot = (T_cmd - T) / params.edf.tau_motor

    # --- Aerodynamics (rho + wind from env, directional drag) ---
    v_wind_body = R.T @ v_wind
    v_rel = v_b - v_wind_body
    speed_rel = np.linalg.norm(v_rel)
    if params.aero.compute_directional_drag and speed_rel > 1e-6:
        v_hat = np.abs(v_rel) / speed_rel
        A_eff = (v_hat[0] * params.projected_area_x
               + v_hat[1] * params.projected_area_y
               + v_hat[2] * params.projected_area_z)
    else:
        A_eff = params.aero.A_proj
    F_aero = -0.5 * rho * speed_rel * v_rel * params.aero.Cd * A_eff
    tau_aero = np.cross(params.aero.r_cp - params.com, F_aero)

    # --- Fins (use ACTUAL servo positions, not commands) ---
    # ±20° mechanical clamp on actual position, stall soft-clamp at ±15°
    V_exhaust = params.fins.V_exhaust_nominal * (omega_fan / params.edf.max_omega)
    F_fins = np.zeros(3)
    tau_fins = np.zeros(3)
    for k in range(4):
        delta_k = np.clip(delta_actual[k], -params.fins.max_deflection,
                          params.fins.max_deflection)
        alpha_eff = params.fins.stall_angle * np.tanh(delta_k / params.fins.stall_angle)
        C_L = params.fins.Cl_alpha * alpha_eff
        C_D = params.fins.Cd0 + C_L**2 / (np.pi * params.fins.AR)
        q_dyn = 0.5 * rho * V_exhaust**2 * params.fins.planform_area
        fin_cfg = params.fins.fins_config[k]
        F_k = q_dyn * (C_L * np.array(fin_cfg['lift_direction'])
                      + C_D * np.array(fin_cfg['drag_direction']))
        F_fins += F_k
        tau_fins += np.cross(np.array(fin_cfg['position']) - params.com, F_k)

    # --- Motor reaction torque ---
    omega_fan_dot = 0.5 * T_dot / (params.edf.k_thrust * max(omega_fan, 1.0))
    tau_motor = np.array([0, 0, -params.edf.I_fan * omega_fan_dot])

    # --- Assemble total force/torque ---
    F_total = F_thrust + F_aero + F_fins
    tau_total = tau_thrust + tau_aero + tau_fins + tau_motor

    # --- State derivatives ---
    p_dot = R @ v_b

    g_body = R.T @ np.array([0, 0, params.g])
    v_dot = F_total / params.mass + g_body - np.cross(omega, v_b)

    q_dot = 0.5 * quat_mult(q, np.array([0, omega[0], omega[1], omega[2]]))

    h_fan = np.array([0, 0, params.edf.I_fan * omega_fan])
    omega_dot = params.I_inv @ (
        tau_total
        - np.cross(omega, params.I @ omega)
        - np.cross(omega, h_fan)
    )

    return np.concatenate([p_dot, v_dot, q_dot, omega_dot, [T_dot], delta_dot])
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
| Fin aero | Thin-airfoil ($C_L=2\pi\alpha$) | Full NACA lookup, CFD | Low-Mach, small α; lookup tables are overkill |
| Wind model | Dryden spectrum (seeded) — **migrated to EnvironmentModel** ([env.md](../Enviornment/env.md)) | Constant wind, von Kármán | Good turbulence fidelity, reproducible. Separated from vehicle for modularity and domain randomization. |
| Ground effect | Simple $1 + 0.5(r/h)^2$ | None, full ground-plane mirror | Captures dominant effect without CFD |
| Config format | YAML | JSON, TOML, hardcoded | Human-readable, standard in ML, easy to edit |
| Fin deflection limit | ±20° mechanical / ±15° stall soft-clamp | Single limit at ±15° | Full servo travel for control authority; aero model handles post-stall gracefully |
| Servo model | Rate-limited first-order lag (mandatory) | Instant action (deleted) | Freewing 9 g servo has 0.10 sec/60° transit, ~600°/s slew, τ≈0.04 s lag — instant action causes 10–30% sim-to-real success drop (Hwangbo et al., 2017) |
| Servo hardware | Freewing 9 g digital metal gear | KST DS215MG (prior) | Freewing matches hardware BoM; slower transit (0.10 vs ~0.07 sec/60°) but sufficient torque for ±15° fins. Digital type gives lower deadband. |
| Policy rate | 40 Hz ($dt_{policy}=0.025$ s) | 20 Hz (deleted) | 20 Hz was marginally adequate but too coarse for 12–15 Hz servo bandwidth; 40 Hz gives ~3× Nyquist, finer GTrXL granularity. See training.md §2.2 |
| Mass property source | Primitives (default) + CAD override option | Primitives only, or CAD only | Primitives enable fast iteration; CAD override provides validated accuracy when available |
| Aerodynamic drag | Directional (per-axis projected area) | Single scalar A_proj | Wind from different directions sees different frontal areas; ≤10% drag accuracy improvement |
| EDF config | Dedicated `edf:` section with full spec | Embedded in `thrust:` | Cleaner separation; EDF params (fan dia, blade count, kV) are hardware identity, not just thrust model params |
| Earth model | Flat, g=9.81 | WGS-84, J2 gravity | Altitude <2 km, curvature irrelevant |
| Radiation/vacuum | Deleted | — | Earth-bound sim, not space |
| Training platform (Phase 1) | Python (NumPy/JAX) | Isaac Sim (full training) | Speed, accuracy ($O(dt^5)$ vs $O(dt^2)$), simplicity. See [master_plan.md §3](../master_plan.md). |
| Isaac role | Validation + HIL + viz (conditional Phase 2) | Full training platform from start | Hybrid maximizes each platform's strengths. Full Isaac overkill for single RB. |
| Parallel strategy (Phase 1) | JAX `vmap` (RTX 5070) or SB3 vec envs (Ryzen 9 9900X, 12–24 workers) | Isaac OIGE (1024+) | Lower VRAM, faster setup, sufficient for 1e7 steps. Ryzen 12C and Blackwell compute make Python-side highly competitive. |
| Environment injection | `EnvironmentModel` passed to `VehicleDynamics.__init__` | Wind/atmo embedded in vehicle | Separation of concerns: atmosphere affects multiple force models; DR is env-level, not vehicle-level |
| EDF density correction | $T \propto \rho / \rho_{ref}$ (from EnvironmentModel) | Ignored, full fan-curve shift | First-order correction captures dominant effect; full fan curves overkill for ~5% $\rho$ range |

---

## Appendix C: Notation Reference

| Symbol | Meaning | Units |
|---|---|---|
| $\mathbf{p}$ | Inertial position (NED) | m |
| $\mathbf{v}_b$ | Body-frame velocity | m/s |
| $\mathbf{q}$ | Orientation quaternion | — |
| $\boldsymbol{\omega}$ | Body angular velocity | rad/s |
| $T$ | Current thrust magnitude | N |
| $m$ | Total vehicle mass | kg |
| $\mathbf{I}$ | Inertia tensor about CoM | kg·m² |
| $\mathbf{R}$ | DCM (body → inertial) | — |
| $\mathbf{F}_b$ | Total body-frame force | N |
| $\boldsymbol{\tau}_b$ | Total body-frame torque | N·m |
| $\mathbf{h}_{fan}$ | Fan angular momentum | kg·m²/s |
| $\Omega_{fan}$ | Fan angular velocity | rad/s |
| $\mathbf{g}_b$ | Gravity in body frame | m/s² |
| $\rho$ | Air density (from EnvironmentModel, see [env.md](../Enviornment/env.md)) | kg/m³ |
| $C_L, C_D$ | Lift/drag coefficients | — |
| $C_{L\alpha}$ | Lift slope ($2\pi$ for thin airfoil) | /rad |
| $\delta_{cmd,k}$ | Commanded fin deflection angle | rad |
| $\delta_{actual,k}$ | Actual (servo-filtered) fin deflection angle | rad |
| $\delta_{max}$ | Mechanical fin deflection limit (±20°) | rad |
| $\dot{\delta}_{max}$ | Servo max angular velocity (~10.5 rad/s at 6 V) | rad/s |
| $\tau_{servo}$ | Servo position lag time constant (~0.03–0.05 s) | s |
| $f_{bw}$ | Servo closed-loop position bandwidth (~12–15 Hz) | Hz |
| $\alpha_{stall}$ | Stall angle onset (±15°) | rad |
| $S_i$ | Primitive surface area | m² |
| $A_{x,i}, A_{y,i}, A_{z,i}$ | Per-axis projected drag-facing area | m² |
| $C_{d,i}$ | Per-primitive drag coefficient | — |
| $dt$ | Integration time step | s |
