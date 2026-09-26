# Convex (SOCP) guidance controller — 2026-09-25 (fourth pass 2026-09-26)

Mission control can now fly the drone with **convex optimization**: a
second-order cone program (SOCP) plans the thrust trajectory from the
measured state through the mission route to the pad, and is re-solved in
closed loop. It is an explicit classical controller, alongside PID and
separate from PPO, and nothing in PPO training uses it.

- Planner: `tvc_env/controllers/convex_guidance.py`
- Flight controller (tracking, attitude, phases): `tvc_env/controllers/convex_adapter.py`
- Attitude regulator (gyroscopic LQR over the vane actuators): `tvc_env/controllers/attitude_lqr.py`
- Settings, each with its evidence: `configs/controllers/convex_guidance.yaml`
- Mission runner branch: `apps/run_mission.py` (`controller == 'convex'`)
- Tests: `tests/unit/test_convex_guidance.py`, `tests/unit/test_mission_service.py`
- Solver: [Clarabel](https://github.com/oxfordcontrol/Clarabel.rs) 0.11.1, an
  interior-point conic solver (added to `mission_control/requirements.txt`).

## Formulation

The method is Açıkmeşe & Ploen, *Convex Programming Approach to Powered
Descent Guidance for Mars Landing*, JGCD 30(5), 2007
(`Paper/Reference/Convex Programming.pdf`). It is adapted to a
battery-powered, constant-mass vehicle.

- **Dynamics.** Point mass, thrust acceleration `u = T/m`, `r'' = u + g`.
  The thrust is linear between nodes (first-order hold), and the dynamics
  are integrated exactly. The paper's Problem 4 uses piecewise-constant
  thrust (zero-order hold). That stepped the thrust direction at every node,
  which the rate-limited rotor and attitude loop cannot follow (see
  [Second pass](#second-pass-wobble-and-approach-2026-09-25)).
- **Lossless convexification.** The thrust annulus
  `rho1 <= ||T|| <= rho2` is not convex. `rho1 > 0` because the vanes need
  jet flow for authority. It is relaxed to `||u_k|| <= sigma_k`,
  `rho1/m <= sigma_k <= rho2/m`. By the paper's Lemma 1 the relaxation is
  tight at the optimum. Every plan reports
  `convexification_gap = max (sigma - ||u||)/sigma`; it is 0.0 in the
  tests.
- **Constraints:**

  | Constraint | Form | Source |
  |---|---|---|
  | Thrust pointing | `u_z >= sigma cos 15°` | paper Sec. VI |
  | Glide slope | `\|\|r_xy - pad\|\| <= tan 45° (r_z - z_td)`. A start outside it gets a widened cone that narrows back to 45° by 3 s before the gate | paper eq. 11 |
  | Gate from above | landing-leg nodes stay at or above the gate altitude | added |
  | Speed | `\|\|v\|\| <= 4 m/s`. A faster start keeps its speed, then brakes back under the bound | paper eq. 10 |
  | Upright, steady arrival at the gate | thrust at the gate `u = [0, 0, g]` | paper eq. 37 |
  | Thrust-vector rate | `\|\|u_{k+1} - u_k\|\| <= rate · dt`, starting at the rotor's present thrust vector. With the first-order hold this bounds the continuous thrust | added for this vehicle |
  | Waypoints | fly-through node inside half the capture radius; hover node at rest on the point for the remaining hold | added for routes |

  The thrust-vector rate bound is added because a rotor speed change yaws
  this vehicle (rotor/body angular momentum), and the bound keeps that
  torque inside the vanes' yaw authority. It is the same 0.25 duty/s bound
  as `task.waypoint_flight.throttle_command`. Bounding the vector rather
  than the slack matters: with the slack form the lossless gap was 3.4%,
  and the executed thrust stepped faster than the rotor can follow.
- **Objective.** Minimum electrical energy. The simulator's power model is
  `P = P_ref (T/T_full)^1.5 + P_aux`, with a power cone per node,
  integrated with the trapezoid rule. The
  alternative `delta_v` objective is the paper's fuel analog, the project's
  propulsive delta-v.
- **Free final time.** A line search (golden section) over the landing-leg
  duration, following the paper's Algorithm 1. Re-plans search around the
  previous plan's remaining time.
- **Infeasible starts** (e.g. too fast and too low to stop). A soft-terminal
  problem minimizes the miss while keeping every actuator constraint and a
  heavily penalized ground constraint, so it degrades to maximum braking.
  It may use full thrust and the fast spool slew.

Solve time is typically 3 ms per SOCP and 30–110 ms per plan (10–30
solves), on the host CPU with no warm start.

## Flight controller

Flight phases:

- `SPOOL_UP` — cold rotor. Attitude hold; duty ramps at 4 /s.
- `POWERED_DESCENT` or `ROUTE` — track the plan, re-solved every 0.5 s.
  Plans continue from the previous plan's *reference* state while tracking
  error is small, and from the measured state beyond 0.75 m or 0.75 m/s.
- `TERMINAL_DESCENT` — velocity-commanded vertical descent from a gate
  0.5 m above touchdown height. The command is 0.15 m/s, clamped at
  0.18 m/s. Descent pauses while more than 0.2–0.5 m off centre, except
  below a 0.15 m commit height.
- A contact candidate latches a land detector: level attitude, rotor
  unloaded to 50% of weight. The existing flight executive then disarms
  after LANDED.

Inner loops:

- **Tracking.** The plan's thrust acceleration is the feedforward, faded
  from the replaced plan over 0.3 s after each re-plan. PID feedback wraps
  it, with integrators bounded by the acceleration they may command and
  conditional integration (anti-windup).
- **Attitude.** Geometric SO(3) error into a full-state discrete LQR
  (fourth pass below). It feeds back the attitude-error integral, the
  error, the body rates and both vane actuator states (servo and joint,
  measured from the vane angles and rates), scheduled on rotor speed.
  The turn rate of the plan's thrust direction, plus half the feedback
  correction's, is fed forward with the vane effort that sustains that
  precession. Yaw is a separate LQR channel on the yaw rate and its
  integral; heading is free for this axisymmetric vehicle. Roll/pitch have
  priority over yaw, except for a yaw reserve. The vane servos' deadband is
  pre-compensated from the servo state (a backlash inverse, see the second
  pass). `attitude.law: pd` restores the earlier PD law.
- **Throttle.** Inverts `T = T_full (duty V_bus/V_ref)^2` at the measured
  bus voltage. The slew limits act on the rotor command (duty × V_bus/V_ref),
  so a bus-voltage change is compensated at once.

The controller reads only the observation (sensor noise included), the
mission sequencer's remaining route and the measured bus voltage.

## Plant identification that set the gains

All from recorded Isaac missions on the landing plant, R² ≥ 0.99:

| Channel | Identified model | Consequence |
|---|---|---|
| Yaw | `Izz r' = -0.265 r + 15.0 f² δ_cm` | The waypoint-flight damper gain 0.15 limit-cycled at ±100°/s. 0.03 leaves a gain margin of ~2.4. |
| Roll/pitch | `I p' = 20.7 f² e_roll - 0.33 p - 1.05 H q`, where `H = I_rotor ω` | Rotor gyroscopic coupling is ≈16 rad/s. Exact cross-feed (0.051) coned *harder* than the PID's 0.03, because of the ~0.1 s sampling plus vane lag. |
| Lateral | Residual acceleration = **59 m/s² per rad of vane effort** (skewed ~30° from the commanded axis) | The vanes act as a strong direct side-force actuator. This set the lateral gains (0.5 / 0.8 / 0.15) and the integrator authority (1.5 m/s²): a CoM offset makes the vanes hold trim, and their side force pushes ~1 m/s² sideways. |
| Vanes | First order, τ ≈ 0.08 s, slew ≈ 2.5 rad/s, 1° deadband | The attitude loop limit-cycled at 4.7 Hz at kd_att 0.10. 0.08 cut body-rate RMS from 97 to 24°/s. |
| Vane servo deadband | The servo moves only while `\|command − position\| > 1°` and stops 1° short: backlash | At kd_att 0.08 it kept a real 1.0–1.1 Hz coning in every mission. Pre-compensated since the second pass. |

There are two distinct attitude limit cycles. At kd_att 0.10 the 4.7 Hz
mode dominated, and sparse sampling in the first analysis script aliased it
to ~1 Hz. At the committed 0.08, a *separate*, real 1.0–1.1 Hz coning
remained: it is visible in full-frame-rate spectra of every recorded
mission. The first pass mistook it for the same aliasing; the second pass
traced it to the servo deadband.

## Validation (Isaac Sim, 2026-09-25)

Default request unless noted: planned 8S profile, full LiPo, 30 s duration.
Each second-pass run re-flies the first-pass request exactly. Cells are
first pass → second pass; roll/pitch RMS covers the whole powered flight.

| Mission | Outcome | Impact (m/s) | Pad error (m) | Landed at (s) | Energy (Wh) | Roll/pitch RMS (°/s) | Runs |
|---|---|---|---|---|---|---|---|
| Default: 18 m, cold rotor, −1 m/s | **LANDED ✓** → **✓** | 0.135 → 0.146 | 0.17 → **0.08** | 8.2 → 8.3 | 5.31 → 5.37 | 14.1 → **1.7** | `f88b065dccdc` → `f073d3d75347` |
| 7 m offset + wind/gusts, cold, −2 m/s | **✓** → **✓** | 0.177 → 0.152 | 0.27 → 0.30 | 7.1 → 8.3 | 4.94 → 5.58 | 24.0 → **13.3** | `412efbd8a8f6` → `806dcfa76dc9` |
| Hot 3 m hop, 1 m offset | **✓** → **✓** | 0.215 → 0.172 | 0.20 → **0.10** | 4.9 → 6.1 | 2.71 → 3.48 | 19.7 → **7.4** | `85ace5ddef76` → `0aca3dabefe3` |
| Wind + sensor noise + CoM shift, warm rotor, −5 m/s | **✓** → **✓** | 0.181 → 0.153 | 0.15 → **0.07** | 20.6 → 20.1 | 12.65 → 12.12 | 19.5 → **11.4** | `59f0ae521be1` → `46cb23362484` |
| Route: fly-through + 2 s hover, then land | **✓** → **✓**, 2/2 captured | 0.156 → 0.150 | 0.30 → 0.40 | 25.7 → 23.5 | 15.37 → 13.82 | 18.2 → **8.3** | `fd97d2f786c3` → `63a3fe1dc2a8` |
| 50 m start | **✓** → **✓** | 0.133 → 0.146 | 0.10 → **0.05** | 12.7 → 16.1 | 8.08 → 9.90 | 14.7 → **1.5** | `aee67f1982ed` → `730ad74e6f18` |
| Legacy 6S, warm rotor, 8 m | **✓** → **✓** | 0.200 → 0.184 | 0.25 → 0.36 | 11.9 → 9.6 | 7.16 → 5.72 | 20.5 → **8.9** | `38037d9880a2` → `765cbba9bdc6` |
| Legacy 6S, default mission | ✗ 2.56 m/s touchdown | | | | | | `e2ac7c86f095` |
| Wind + noise + CoM, *cold* rotor, −5 m/s from 15 m | ✗ crash | | | | | | `06b4627ce778` |
| PID baseline, default mission (for comparison) | ✗ 1.50 m/s touchdown | | | | | | `7836ec0d4506` |

The 50 m start now respects the 4 m/s bound: the first pass reached
7.4 m/s through the speed-cap ratchet. The second pass pays 3.4 s and
1.8 Wh for that. Its remaining 5.9 m/s peak is the fall during the cold
spool-up, before the first plan. Three missions end a little further from
centre, all inside the 0.5 m pad. In each, the terminal descent coasts
across the pad centre and drifts out slowly (see Limits).

The two convex failures are beyond the vehicle's envelope rather than the
controller's:

- **Cold rotor, −5 m/s from 15 m.** Full thrust is commanded by t = 0.5 s,
  but the battery-coupled motor can only add rotor energy at ~3 kW. Thrust
  reaches 40 N at 3.5 m altitude, still falling at 6.8 m/s, which needs
  more than 5 m to stop.
- **Legacy 6S, default mission.** The 6S pack sags to T/W ≈ 1.1 under
  load, and cannot arrest the fall built up during a cold spool-up.

## Second pass: wobble and approach (2026-09-25)

A mission-control trial, `0289417f5d60`, wobbled for its whole flight. It
started at 50 m, flew three hover waypoints (10 s holds) and then landed,
within 100 s. It timed out 0.31 m from the pad, still in terminal descent.

### The wobble: a servo-deadband coning limit cycle

- **What it was.** A 1.0–1.1 Hz *retrograde* coning. Tilt held ≈ 3.2°
  while its direction rotated against the rotor spin. Roll and pitch rates
  were 15°/s RMS each, with ≈ 3° attitude error in every phase. Every
  recorded mission shows it at full frame rate (`f88b065dccdc`,
  `aee67f1982ed`, `fd97d2f786c3`: 12–16°/s at ~1 Hz).
- **What it cost.** Through the vane side force it drove ±0.5 m/s of
  lateral velocity. The terminal descent kept pausing off-centre, so the
  trial never touched down.
- **Mechanism.** In the recording the vanes freeze while their commands
  sweep ±1°, then trail them by 1°. The servo model moves only while
  `|command − position|` exceeds its 1° deadband, and stops 1° short: that
  is backlash. With the rotor's angular momentum (H ≈ 0.78 N·m·s at hover)
  the attitude loop's retrograde precession mode sits at ≈ 176° of loop
  phase. The loop is therefore conditionally stable, and the gain and phase
  that backlash removes sustain a limit cycle.
- **Reproduced offline.** A replica of the landing plant drove the real
  controller: servo model, 25 ms vane-joint lag, vane aero, rotor gyro and
  rigid body, at 120/30 Hz. It gave 14.1°/s, 3.0° tilt at 1.0 Hz (Isaac:
  15°/s, 3.2°, 1.07 Hz). Without the deadband: 0.06°/s. No gain set removes
  it (kd_att 0.08–0.12, gyro cross-feed 0–0.051, kp_att 0.35–0.7: 8–17°/s).
- **Fix: deadband inverse.** Each vane is commanded
  `desired + b · sat((desired − measured) / (0.25 b))`. `b` is the servo's
  deadband from the hardware profile; the measured vane angle comes from the
  observation (`attitude.deadband_compensation`, `deadband_smoothing`).
  Replica: 1.2°/s, 0.19° tilt. Over-compensation (true deadband 0) dithers
  at 1.7°/s. Under-compensation leaves proportionally more coning (true
  deadband 2×: 12.6 vs 27.8°/s). At the sensor-noise disturbance levels:
  2.0°/s, with 10% more vane travel.

### The approach: four planner defects

With the wobble fixed (`e7d7fbb81708`), the trial still reached the gate
0.6 m off-centre. Its re-plans exposed four defects, all fixed as convex
constraints:

| Defect | Evidence | Fix |
|---|---|---|
| Speed cap ratchet. Every node was capped at \|v0\| + 0.25 m/s, and the energy objective flies at the cap | 4.0 → 5.9 m/s over the re-plans; a 1.88 Hz tilt sawtooth at the re-plan period; no feasible plan to the gate 2.9 m up | A faster start keeps its speed, then brakes back under 4 m/s at half the slowest braking the limits allow |
| Widened glide cone for the whole leg | 60° cone from 30 m out; 2 m/s sideways 1 m above the pad | The widened cone narrows back to 45° by 3 s before the gate |
| Zero-order-hold thrust | A 12.6° thrust-direction step at t = 0 of the landing leg, then 8.6° steps every 0.46 s; an 11.6° tilt command in one frame, 48°/s | First-order hold: continuous thrust from the present thrust vector, exactly rate-bounded |
| The landing leg could sink below the gate | The plan dipped to 0.52 m (gate 0.81 m); the vehicle sank to 0.48 m, then climbed back (`614cb15ab14a`) | Landing-leg nodes stay at or above the gate altitude |

The first-order hold needed one more constraint. A hover point and its
hold now also pin the thrust to hover. With only zero velocity at the
nodes, the thrust alternated ±0.9 N node to node through a hold
(`f1b3c7b587da`: a 1.5 Hz, 6°/s wobble at a hover).

### Results on the trial

| Run | Controller | Roll/pitch RMS | Attitude error | Roll+pitch travel | Peak speed | Outcome |
|---|---|---|---|---|---|---|
| `0289417f5d60` | first pass | 15.8°/s | 3.12° | 2813° | 6.1 m/s | timeout, never settled; 0.31 m off |
| `e7d7fbb81708` | + deadband inverse | 4.4 | 0.47 | 389 | 6.2 | timeout; gate reached 0.6 m off |
| `614cb15ab14a` | + speed cap, glide taper | 3.9 | 0.41 | 330 | 4.1 | timeout 5 cm above the pad |
| `d211d01ee3e3` | all (final) | **3.3** | **0.39** | **313** | 4.1 | touched down at 99.4 s, 0.19 m off, 0.19 m/s; bounce left LANDED unconfirmed at 100 s |
| `f1b3c7b587da` | all, 120 s window | 3.9 | 0.50 | 389 | 4.1 | **LANDED** at 99.8 s, 0.16 m, 0.19 m/s |

At hover holds the rates fell from 15°/s to 1.2–3°/s. The route plus a
landing at the enforced 4 m/s now takes ~100 s, so the 100 s window is too
short by a fraction of a second. The first pass had flown at up to 6 m/s,
and still never settled.

## Third pass: momentum-bounded vanes (2026-09-25)

Mission control's landing plant used the legacy vane model: each vane an
independent airfoil (q·S·C_Nα at the ideal 128 m/s jet speed), plus a
0.27 N·m·s/rad artificial damper. The project's 2026-09-14 audit
(`tvc_env/dynamics/coupled_jet.py`) found those forces exceed the jet's
momentum. At hover they give 8.5× the torque per degree of the
momentum-bounded coupled jet that the waypoint_flight PPO trains on:
10° of vane made 21 N sideways, 62% of the thrust. Classical-controller
missions now fly the coupled jet by default (request field `vane_model`:
`momentum` | `legacy`, with physics copied from `train_waypoint_flight.yaml`).
Missions recorded before the field report the plant they flew.

**Probed authority** (`TVCDirectRLEnv.vane_authority()`, torque per rad at
full rotor, planned 8S): legacy 20.04 roll/pitch and 14.58 yaw (flight
identification 20.7 / 15.0); momentum-bounded 2.365 / 1.72. On momentum vanes,
full deflection makes ~0.33 N·m. Against the rotor's 0.78 N·m·s of angular
momentum, that tilts the vehicle at most ~24°/s.

**Controller changes.** Every one is a no-op on the legacy vanes, where a
re-fly reproduced the second pass: 0.146 m/s, 0.088 m (`0926e63a642b`).

| Change | Why (evidence) |
|---|---|
| Vane efforts × reference/actual authority (reference = legacy probe) | The legacy-tuned gains lost attitude on momentum vanes (offline replica and Isaac) |
| Planner tilt-rate bound: 40% of the vanes' full-deflection precession rate (~10°/s) | Plans slewing at the thrust-rate bound (~30°/s) saturated every vane |
| Duty slew capped so the rotor reaction uses ≤40% of the yaw authority (~0.09 /s) | 0.25 /s at the gate yawed the body to 34°/s with the vanes saturated in yaw, then a 2.5 Hz oscillation: 0.29 m/s touchdown (`8abd7233a4ec`) |
| Yaw keeps a reserve of vane travel (the reaction at that slew) | With a CoM trim saturating roll/pitch, the body spun to 400°/s (replica) |
| Cascaded attitude loop: body-rate command clamped to 60% of the precession rate (~15°/s) | The PD law saturates momentum vanes at ~1.5° of error; saturated descents oscillated (replica) |

**Fin mapping check.** The vanes rendered in mission control (PhysX link
poses) equal the joint angle about the metadata hinge axis, for every vane
and frame. An open-loop Isaac test (constant duty, no controller) gave:
- +5° common mode: +225°/s² of yaw acceleration on momentum vanes, +97°/s² on legacy vanes.
- −5° common mode: −221°/s² and −134°/s².

So positive common mode yaws the body clockwise seen from above
(counterclockwise from below), as the metadata states. Where a replay shows
the vanes pushing against the rotation, the yaw damper is losing to the
rotor's reaction torque.

**Isaac, momentum-bounded vanes** (warm rotor; cold-start requests were
re-flown with the rotor at hover):

| Mission | Outcome | Impact | Pad error | Run |
|---|---|---|---|---|
| Default 18 m, warm | **LANDED ✓** | 0.175 m/s | 0.44 m | `addd8752dc7c` |
| Hot 3 m hop | **LANDED ✓** | 0.147 | 0.23 | `e9c203457bba` |
| 7 m offset + crosswind, warm | **LANDED ✓** | 0.136 | 0.18 | `a12d5ef1196f` |
| Legacy 6S, 8 m | **LANDED ✓** | 0.208 | 0.25 | `1bb89b6a12bb` |
| 50 m start, warm | **LANDED ✓** | 0.145 | 0.33 | `fa66253416c8` |
| Route: fly-through + hover | ✗ touched down 0.74 m off | 0.178 | 0.74 | `3d413ec4468b` |
| Mission trial (50 m, 3 hovers), 120 s | ✗ touched down 0.73 m off | 0.205 | 0.73 | `ae7ec094fb49` |
| Wind + noise + CoM shift, −5 m/s | ✗ drifted away | | | `6cd362e3ff2a` |
| Default 18 m, *cold* rotor | ✗ crash | | | `760e6addc9b9` |
| PID baseline, warm | ✗ timeout, 11 m off | | | `6421f6ec55a4` |

**What the realistic plant shows**
- **Cold in-air starts are infeasible.** Spinning the rotor up moves 0.78 N·m·s
  into the body, and the vanes hold only ~0.25 N·m in yaw. The body spins
  (5,400° of yaw in `42046eaa3e85`). Spool the rotor up on the ground.
- **CoM offsets eat the vane travel.** A 1 cm lateral offset needs ~10° of
  trim out of 11.5°. The `com_shift` disturbance (±1 cm per axis) is mostly
  outside the envelope; the airframe needs balancing to a few millimetres.
- **Residual oscillations.**
  - ~1 Hz with 10–22% of samples saturated, in maneuvering and trimmed flight
    (hop, crosswind, route, 6S): saturation lowers loop gain, as the servo
    backlash did.
  - ~3.5 Hz in the terminal descent of the cleanest runs (default, 50 m):
    the closed-loop nutation mode, which the legacy plant's artificial damper
    used to damp.
- **Shutdown spin.** Cutting the throttle on the pad spins the body (438°/s):
  the rotor's spin-down reaction, with no damper to absorb it.
- **PID baseline.** Not plant-aware, so it cannot fly these vanes.

## Fourth pass: plant audit and a gyroscopic attitude regulator (2026-09-26)

Goal: remove the remaining wobble on the momentum-bounded vanes, make
waypoint flight land on the pad, and check that the plant physics is right.
PID stays the legacy-vane reference: mission validation now rejects PID on
momentum vanes, and the launch form locks its vane selector to legacy.

### Plant audit

Open-loop Isaac tests (no controller; scripted vanes and duty, from 40 m):

| Test | 120 Hz physics | 480 Hz physics | Theory |
|---|---|---|---|
| Nutation, neutral vanes | 2.489 Hz, decay 0.019 /s | 2.492 Hz, 0.018 /s | H/I = 2.48 Hz, undamped |
| Roll step (0.05 rad effort) → steady pitch rate | 3.70°/s | 3.72°/s | τ/H = 6.1°/s × (1 − 1° deadband / 0.05 rad) ≈ 3.9°/s |
| Yaw step (0.05 rad common mode) | 110.6°/s² | 111.1°/s² | 173°/s² × 0.66 (deadband) = 114°/s² |
| Throttle cut, in the air | body +2242°/s | +2242°/s | ΔH / I_zz = 2233°/s |

A regression of Isaac's measured angular acceleration on the model's terms
(missions `e9c203457bba`, `addd8752dc7c`) gave coefficients of 0.9–1.03 on
the vane torque and gyroscopic coupling, and 1.0 on the yaw vane and spool
reaction torques. The gyro, the coupled-jet vane forces and the momentum
bookkeeping are right, and 120 Hz is converged. Two defects:

- **Motor torque.** The first-order spool lag (0.15 s) was unbounded: a
  throttle cut stopped the rotor in ~0.5 s, which takes 5.2 N·m. The planned
  4075 KV1500 motor on its 120 A ESC can apply at most Kt·I = 0.76 N·m, and an
  EDF ESC with the brake off applies none at zero throttle, so the rotor
  coasts on its drag (0.47 N·m at hover, a ~1.7 s time constant). The
  unbounded spin-down spun every landed momentum-vane vehicle at
  430–710°/s on the pad after disarm. The torque is now bounded
  (`dynamics.motor_torque_limit` in `train_waypoint_flight.yaml`, which is
  also the PPO training plant), and the pad spin fell to 2–4°/s.
- **Warm-rotor spawns** started the battery at its open-circuit voltage
  while the rotor already drew hover power. The first substep dropped the
  bus 3% (33.6 → 32.6 V), and a flight computer reading 33.6 V under-drove
  the rotor. It spun down 1600 rpm, and the reaction yawed the body to 30°/s
  within 0.1 s of every warm start. Resets now put a spinning rotor on its
  loaded bus (`LiPoBattery.carry_load`).

The offline replica now matches these open-loop tests within a few
percent. Its forward-Euler gyro had been adding energy (nutation growing
at 0.24 /s), and is now implicit midpoint, as Isaac's coupled Cayley step
is.

### Why the wobble survived the third pass

Linearizing the roll/pitch loop (body, rotor gyro, servo 0.05 s, vane
joint 0.025 s, zero-order hold at 30 Hz) shows the third pass's PD law was
**unstable on momentum-bounded vanes at every rotor speed**:

| Mode | Growth rate (rotor 0.78 → 0.95) |
|---|---|
| ~4 Hz nutation | +1.1 → +2.8 /s |
| ~1.6 Hz precession | +0.6 → +1.9 /s |

Only saturation and the deadband kept the flights bounded, as limit cycles:
the 4 Hz, 25–30°/s terminal-descent oscillation (`addd8752dc7c`,
`fa66253416c8`) and the 1.0–1.4 Hz retrograde wobble with 16–38% of vane
commands saturated (`e9c203457bba`, `3d413ec4468b`, `ae7ec094fb49`). The
cause is the gyroscope. The nutation sits at H/I ≈ 2.5 Hz, and the vanes
lag ~0.075 s there, so rate feedback passes 90° of phase and stops damping.
On the legacy vanes the 0.27 N·m·s/rad artificial damper had covered this,
with 0.1 /s of margin at full rotor. No PD retune works: the best found
decays at 1.0 /s with a 10× slower attitude loop.

### The attitude regulator

`attitude_lqr.py` solves a discrete LQR on that model, scheduled on rotor
fraction (both K·f² and H change with it). It feeds back:

- the attitude-error integral, which holds a CoM trim;
- the attitude error and body rates;
- the servo and joint angles in effort space. The joint is the measured
  vane angle. The servo is `j + τ_j·j'` from the measured vane rate, which
  matches the servo model in Isaac to 0.009° RMS.

The actuator states supply the phase lead a PD law lacks, and the gain
rotates the torque about 43° toward the precession direction. Nominal
closed-loop damping is ζ ≥ 0.46 at hover and ≥ 0.40 from 30% to 100% rotor.
Any single model error keeps ζ ≥ 0.24: ±30% authority, +50% servo or joint
lag, 8 ms of extra delay, ±20% inertia or +20% rotor momentum.

Details that mattered, each found offline or in Isaac:

| Choice | Evidence |
|---|---|
| Input weight by Bryson's rule on vane *torque*, R = 1/τ², τ = 0.24 N·m (yaw 0.3) | Weighted per rad of effort, the legacy vanes' 8.5× authority made the design 72× more aggressive: a 7 Hz loop that limit-cycled on the servo slew limit |
| Yaw: LQR on rate and its integral; the yaw design torque also caps the duty slew | The jet's residual swirl is a steady ~0.05 N·m of yaw torque. Rate feedback alone left a 15–27°/s spin that rotated the body-frame trim integral, and the landing leg circled the pad for 20 s. On legacy vanes a 0.25 /s duty slew (8× the yaw design torque) wobbled roll/pitch at 13–23°/s |
| Rate feedforward: the plan's thrust direction plus *half* the feedback correction; optimal plans only, capped at the planner's tilt-rate bound | Differentiating the whole command closed a loop through the vanes' own non-minimum-phase side force: a 1.3 Hz mode at ζ 0.03, seen as 4–5°/s of wobble in Isaac terminal descents. The plan alone left the weak 6S vanes lagging every correction, and the vehicle circled the pad until timeout (`766270f3320d`). Half gives ζ 0.37 on both packs. Soft-terminal fallback plans swing between re-plans and are not fed forward |
| Attitude error clamped to 60% of the vane limit | A large error becomes a bounded-rate precession, leaving travel for damping |
| Integral on the clamped error, only while unsaturated | Wider windows trimmed a mid-air CoM offset faster but learned the long legs' tracking lag (up to 1.8 m off the pad) |
| Vane effort limit 0.245 rad (servo 0.262 minus the deadband headroom) | 22% more torque than the PID's 0.20. `validate_action` had also clipped every command at 0.20, discarding the deadband inverse exactly when the vanes saturated |
| Feedforward faded over 0.3 s after each re-plan | Each plan starts at the present thrust with its own slope: tilt steps of up to 1.6° at re-plans on the route. With the fade, 0.4° |
| Duty slew applied to the rotor command (duty × V_bus/V_ref) | The yaw-safe slew had held back the duty that compensates bus sag |

Lateral gains are now 1.0/1.2/0.3 (from 0.5/0.8/0.15). A linear hover
model of the whole loop (lateral + attitude + actuators + vane side force +
feedforward) damps the side-force/velocity loop at ζ 0.37 on 8S and 6S.
With the plan-only feedforward, ζ is 0.33 at kd 1.2, 0.18 at kd 1.6 and 0.08
at kd 2.0, because the side force opposes the tilt it commands, which caps
the velocity gain. Offline, the worst miss fell from 0.19 m to 0.08 m over
six 8S missions, and to 0.04 m over three 6S missions.

### Results (Isaac, final code)

All missions are momentum-bounded vanes, rotor already spinning, unless
noted. Body-rate RMS is roll/pitch in flight (manoeuvring included) and in
the terminal descent. "Third pass" is the same request under the PD law.

| Mission | Outcome | Impact | Pad error | Body rates flight / terminal | Vanes saturated | Third pass | Run |
|---|---|---|---|---|---|---|---|
| Default 18 m | **LANDED ✓** | 0.178 m/s | 0.017 m | 0.7 / 0.8°/s | 0.0% | 0.44 m; 4 Hz, 25°/s terminal | `408e6b926eee` |
| Hot 3 m hop | **LANDED ✓** | 0.154 | 0.045 | 5.6 / 3.9 | 0.1% | 0.23 m; 1.4 Hz, 16°/s | `e2ba15c45abb` |
| 7 m offset + crosswind | **LANDED ✓** | 0.159 | 0.052 | 6.8 / 2.7 | 0.3% | 0.18 m; 1 Hz, 17°/s | `49cc16aca7f5` |
| Legacy 6S pack, 8 m | **LANDED ✓** | 0.175 | 0.051 | 7.3 / 1.4 | 0.1% | 0.25 m; 1 Hz, 18°/s | `fdb559daede1` |
| 50 m start | **LANDED ✓** | 0.152 | 0.004 | 0.7 / 0.7 | 0.0% | 0.33 m; 4 Hz, 30°/s | `c8233615eabf` |
| Route: fly-through + hover | **LANDED ✓** | 0.152 | 0.098 | 7.3 / 3.3 | 0.2% | ✗ 0.74 m off | `0036c82ca31d` |
| Mission trial (50 m, 3 hovers), 120 s | **LANDED ✓** at 99 s | 0.148 | 0.063 | 3.8 / 3.2 | 0.1% | ✗ 0.73 m off | `1b9c6c33b85b` |
| Wind + sensor noise + CoM shift, −5 m/s | **LANDED ✓** | 0.136 | 0.146 | 9.3 / 4.6 | 6.8% | ✗ drifted away | `d4e2b8d8ceac` |
| Legacy vanes (regression) | **LANDED ✓** | 0.182 | 0.029 | 1.8 / 1.0 | 0.0% | 0.146 m/s, 0.088 m | `3c3f033b00a2` |
| PID, legacy vanes (reference) | **LANDED ✓** | 0.152 | 0.025 | | | ✗ 11 m off (momentum) | `7cf4e8c1506b` |

The 1–4 Hz limit cycles are gone: every spectrum peaks below 1 Hz
(manoeuvring), and saturation is ≤ 0.3% except with the ±1 cm CoM shift.
After landing, a momentum-vane vehicle now sits still (≤ 6°/s), where it
used to spin at 430–710°/s. Two runs carried yaw into the landing: the route
touched down yawing 29°/s, stopped within 0.2 s; the all-disturbances run
was disarmed at 79% rotor with the legs lightly loaded, and the coasting
rotor spun it briefly to 124°/s, stopped within 0.4 s. On the legacy plant
(no motor torque limit, kept as recorded) disarm still spins the vehicle,
to 153°/s.

### Remaining limits

- **Yaw transients of 20–40°/s.** Throttle changes accelerate the rotor,
  and its reaction yaws the body. The momentum vanes hold ~0.3 N·m of yaw
  against 0.78 N·m·s of rotor momentum. A 10× yaw-rate weight cut the peaks
  only from 26 to 22°/s; slower throttle changes would trade descent
  performance. They are heading excursions, not an oscillation.
- **CoM tolerance ~5 mm.** A 1 cm offset needs ~10° of trim out of 14°.
  From a mid-air start with an untrimmed 5.8 mm offset the vehicle tips
  ~13° while the integral builds (offline harness), and more when the jet
  swirl and a throttle slew also claim vane travel. A disturbance-torque
  observer (model-predicted against measured angular acceleration) would
  separate the trim from manoeuvre lag and trim much faster. It is the next
  step, together with balancing the airframe.
- **Cold in-air starts stay infeasible.** The torque-limited motor needs
  ~1.3 s to reach hover speed, and the body takes the rotor's momentum:
  1,781°/s of yaw and a 6.8 m/s crash (`ee3f973be19e`). Spool up on the
  pad. The launch form warns about cold in-air starts on momentum vanes.

## Fifth pass: following the drawn route (2026-09-26)

Goal: fly the route drawn on the launch form, not only its waypoints. In
`835c3de32185` the start was 94 m up, descending at 20 m/s and drifting
9 m/s sideways, with the rotor at 50%, and one fly-through waypoint at 32.5 m.
The vehicle sank to 6.6 m, climbed back 26 m to the waypoint, then landed,
up to 28.5 m from the blue route line. Tracking error stayed under 0.7 m: the
controller flew its plan, and the plan ignored the line. There were three
causes:

- **The planner only constrained the waypoints.** Between them it took the
  minimum-energy path. A fly-through must be passed, so after overshooting
  it climbs back.
- **The start could not reach the waypoint.** Half of the thrust above
  weight is reserved for feedback, so plans brake a descent at only
  2.0 m/s² (36.8 of 43.1 N). A 20.9 m/s descent needs ~107 m to stop at that
  rate; the waypoint was 62 m below.
- **Braking weakened mid-flight.** While no plan was feasible the fallback
  braked at full thrust, and its plans bottomed at 15–18 m. Once a plan was
  feasible again (3.5 s) it planned at the reserved ceiling and dipped to
  4.3 m.

### Route corridor

On a mission with waypoints each plan node now stays within 1 m of the
drawn route: the sequencer's Catmull-Rom curve through the start, the
waypoints and the pad, the same curve the launch planner draws.

| Choice | Evidence |
|---|---|
| Distance to a chord of the curve (an SOC per node, with a chord parameter) | A corridor measured across the curve's tangent left the along-route direction free: on the near-vertical first leg of `835c3de32185` the plan still sank 20 m below the waypoint. A chord ends at its waypoint, so running past it counts |
| Two passes: the whole leg's chord, then ±10% chords around each planned node | The whole chord leaves the timing free but is 1–3 m from the curve on the routes tried; local chords are 0.2–0.6 m from it. A second refinement changed nothing |
| Soft: the excess is priced at 10 s of hover energy per metre-second | A start the corridor cannot contain still gets a plan; the landing-time search ranks durations with the penalty included |
| Legs never run below their lower end | Catmull-Rom undershoots after a steep arrival. The trial's hover-to-hover leg (6 m, 4.5 m) dips to 1.5 m, and following it took the plan to the 0.8 m floor at 2.3 m/s: touchdown 17 m from the pad (offline replica). The sequencer's curve is unchanged (PPO observation contract) |
| On a corridor, an over-speed start may brake at full thrust until back under the speed bound | Without a corridor the energy optimum brakes no harder than the floor forces: offline, the same allowance turned `835c3de32185`'s maximum-braking fallback into a plan that skimmed the 0.8 m floor with no thrust left for tracking. With the corridor the plan from that mission's 3.5 s state bottoms at 18.5 m instead of 4.3 m |

A rotor above the planning ceiling (a braking plan or tracking margin) now
comes back down at the thrust rate instead of starting the next plan at the
ceiling in one step.

Offline (coupled-jet replica, no wind), largest distance from the drawn
route, corridor off → on; every mission landed within 0.07 m of centre:

| Route | Off | On |
|---|---|---|
| `835c3de32185` start | 33.7 m | 17.4 m |
| Launch-form draft (83 m, −18 m/s, hover at 17.9 m) | 11.4 m | 2.2 m |
| Fly-through + hover | 1.9 m | 1.2 m |
| L-turn fly-throughs | 4.1 m | 1.2 m |
| Zig-zag, three fly-throughs | 6.2 m | 2.1 m |
| 120 s trial, three hovers | 4.3 m | 3.5 m (1.0 m rms; 3 m of it is the dip it now skips) |

The `835c3de32185` remainder is physics: straight-line braking from that
start carries the vehicle ~33 m across the route (the tilt cone allows
2.6 m/s² sideways while the thrust brakes the descent). Solves take longer:
median 70–150 ms per re-plan (was 70–90), 380 ms on the three-hover trial.

### Launch-form route check

The launch board and route planner estimate where the start velocity
carries the vehicle: straight-line braking at full thrust inside the 15°
tilt cone, after the rotor spools up from its start speed (the EDF lag, and
the 0.76 N·m torque bound on momentum vanes). Mass and full-rotor thrust are
the Isaac plant's (`models.PLANT_THRUST`). The board warns when:

- the vehicle cannot stop above the ground;
- the stop point lies more than the waypoint radius off the first leg (for
  `835c3de32185`: ~33 m);
- the drawn route dips more than 1 m below both ends of a waypoint leg (the
  convex corridor holds the lower end).

The planner draws the braking path as a dashed line ending in ×.

### Results (Isaac)

Largest (RMS) distance from the drawn route, before → after; every
mission landed (0.15–0.16 m/s):

| Mission | Before | After | Pad error after | Run |
|---|---|---|---|---|
| `835c3de32185` start (wind, sensor noise) | 28.5 m (14.4), lowest 6.6 m before the 32.5 m waypoint | 17.3 m (7.5), lowest 17.5 m | 0.27 m | `96c4b40be2be` |
| Fly-through + hover | 1.9 m (0.69) | 1.2 m (0.49) | 0.12 m | `4c8c210998f3` |
| L-turn fly-throughs (new) | 4.1 m offline | 1.2 m (0.78) | 0.14 m | `f379934526eb` |
| 120 s trial, three hovers | 4.3 m (1.47) | 3.5 m (1.02); 1.2 m from the corridor | 0.13 m | `1d29169d440f` |

Peak roll/pitch rates on the short route fell from 42/41 to 21/23°/s. The
0.27 m pad error of `96c4b40be2be` came after the vehicle was centred to
0.03 m at touchdown height: it slid in the wind for 1.4 s before the landing
registered, a contact-phase effect this pass does not touch. The eight
missions without waypoints (default, hop, crosswind, 6S, 50 m, all
disturbances, legacy vanes) flew exactly as in the fourth pass: same
impact speeds and pad errors.

A user mission from the launch form (`60e1f757fa15`: 80.8 m, −15.9 m/s,
rotor 50%, four waypoints) held 2.7 m (0.86 m RMS) of the route, stopped at
19.7 m above an 18.5 m fly-through, and timed out 0.1 m above touchdown
height at its 60 s limit.

### Remaining limits

- **A start the tilt cone cannot turn stays off the route.** From
  `835c3de32185` the straight-line estimate is ~33 m across the first leg
  and the flight kept 17 m; only a slower or better-aimed start fixes it.
  The launch board says so before launch.
- **A half-spun rotor still yaws the body at ~690°/s.** A 50% start spools
  up at the full rate to brake, and the body takes the rotor's momentum.
  The launch board's cold-start warning only covers starts below 50%.
- **Near the pad the corridor competes with the glide slope and the
  vertical gate arrival**, so the last metres of a slanted final leg can sit
  1–2 m off the drawn curve.

## Mission control

- **Controller option.** **CONVEX · SOCP powered-descent guidance** is
  offered when Clarabel is importable. It flies landing and waypoint
  missions; waypoint missions need the LiPo model, because the mission
  sequencer shares the PPO observation contract.
- **Flight page.** Draws the optimized trajectory in force at the replay
  time as a green line in the camera views. It adds a guidance readout
  (phase, time to gate, tracking error, thrust command, solve time, lossless
  gap) and phase and fallback events on the timeline. The altitude and
  thrust charts gain PLAN and COMMAND traces.
- **Launch-form route check.** The board and route planner estimate
  full-thrust braking from the start velocity and warn when the vehicle
  cannot stop above the ground, cannot hold the first leg, or the drawn
  route dips below a leg's lower end (see the fifth pass).
- **Recording.** Frames record `guidance` telemetry; the full planned path
  is stored only on re-plan frames. Metadata records the settings, solver
  version and vehicle model. `--convex-settings` is a local diagnostic
  override, recorded in metadata and never used by the service.
- **Altitude fail-stop.** For convex missions this is raised to 105 m. The
  landing task's 30 m training geofence terminated every start above 30 m
  at t = 0; PID keeps the old behaviour.

## Limits and next steps

- **CoM offsets** are the tightest limit on momentum-bounded vanes (see the
  fourth pass). The next steps are a disturbance-torque observer for the
  trim and balancing the airframe to a few millimetres.
- **The deadband inverse needs the servo's real deadband** (1° is an
  estimate). Over-compensation only dithers; under-compensation brings the
  coning back in proportion. Measure it on the bench. Estimating it online
  from the measured vane motion is the robust follow-up.
- **The attitude LQR is only as good as its model.** Vane authority, servo
  and joint lags, inertia and rotor momentum come from the simulator. It
  tolerates any single ±30% authority, +50% lag or ±20% inertia error, but
  re-identify them on hardware (a step test per axis, as in the fourth
  pass's open-loop table) before flying it.
- **The servo state estimate uses the vane rate.** In Isaac that is the
  PhysX joint velocity. On hardware it would need a vane position sensor,
  or a model-based estimate from the commands.
- Real-time use would need a solve-latency state predictor. Solves take
  30–110 ms (70–150 ms with a route corridor, up to 1.1 s on the
  three-hover trial), against a 33 ms control period.
- The mission runner now retries its status-file replace. On Windows, the
  service reading `status.json` at the same instant had aborted a mission
  (`aea16ce345d4`).
