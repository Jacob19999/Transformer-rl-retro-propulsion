# Convex (SOCP) guidance controller — 2026-09-25

Mission control can now fly the drone with **convex optimization**: a
second-order cone program (SOCP) plans the thrust trajectory from the
measured state through the mission route to the pad, and is re-solved in
closed loop. It is an explicit classical controller, alongside PID and
separate from PPO, and nothing in PPO training uses it.

- Planner: `tvc_env/controllers/convex_guidance.py`
- Flight controller (tracking, attitude, phases): `tvc_env/controllers/convex_adapter.py`
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

- **Tracking.** The plan's thrust acceleration is the feedforward. PID
  feedback wraps it, with integrators bounded by the acceleration they may
  command and conditional integration (anti-windup).
- **Attitude.** Geometric SO(3) error. It uses the PID baseline's
  fin-effort structure and mixer, with roll/pitch priority over yaw. Yaw
  is rate-damped only. Heading does not matter for this axisymmetric
  vehicle. The vane servos' deadband is pre-compensated from the measured
  vane angles (a backlash inverse, see the second pass below).
- **Throttle.** Inverts `T = T_full (duty V_bus/V_ref)^2` at the measured
  bus voltage.

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
- **Recording.** Frames record `guidance` telemetry; the full planned path
  is stored only on re-plan frames. Metadata records the settings, solver
  version and vehicle model. `--convex-settings` is a local diagnostic
  override, recorded in metadata and never used by the service.
- **Altitude fail-stop.** For convex missions this is raised to 105 m. The
  landing task's 30 m training geofence terminated every start above 30 m
  at t = 0; PID keeps the old behaviour.

## Limits and next steps

- **Lateral precision (0.05–0.4 m) is now set by a slow lateral loop.**
  kp_xy/kd_xy 0.5/0.8 (~0.7 rad/s) were chosen while the coning was
  present, because higher gains excited it through the vane side force. In
  the terminal descent the vehicle now coasts across the pad centre and
  drifts out slowly. In the offline replica, 1.0/1.8 landed 0.01–0.04 m from
  centre where 0.5/0.8 missed by up to 0.43 m. That needs an Isaac sweep
  first, because of the next point.
- **CoM offsets.** The attitude loop is PD only. A CoM offset therefore
  leaves a steady attitude error (1.7° in `46cb23362484`), and a vane trim
  whose side force the lateral integrator must cancel. In the replica, a
  1 cm lateral offset (the edge of `com_shift`'s ±1 cm per axis) drifts off
  the pad during the terminal descent. At 1.4 cm with higher lateral gains
  it diverges. A body-frame attitude integrator (ki 0.5, anti-windup) landed
  the 1 cm case 0.02 m from centre offline. The terminal descent should also
  hand back to powered descent when the vehicle drifts far off the pad.
- **The deadband inverse needs the servo's real deadband** (1° is an
  estimate). Over-compensation only dithers; under-compensation brings the
  coning back in proportion. Measure it on the bench. Estimating it online
  from the measured vane motion is the robust follow-up.
- The identified numbers come from the simulator's estimated vane and
  motor models. Re-identify on hardware before trusting any gain.
- Real-time use would need a solve-latency state predictor. Solves take
  30–110 ms, against a 33 ms control period.
- The mission runner now retries its status-file replace. On Windows, the
  service reading `status.json` at the same instant had aborted a mission
  (`aea16ce345d4`).
