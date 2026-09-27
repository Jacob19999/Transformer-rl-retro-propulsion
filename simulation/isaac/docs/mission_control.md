# EDF mission control

Run `simulation/isaac/mission_control/start.ps1` in PowerShell, then open
http://127.0.0.1:8830. It uses this repository's Isaac Python environment.
For first-time dependencies run that Python with `-m pip install -r
simulation/isaac/mission_control/requirements.txt`. The launcher installs the
locked browser dependencies and builds the frontend. No cloud service is used.
It listens on the LAN by default (other devices use http://<this-PC-IP>:8830,
and Windows Firewall must allow inbound TCP 8830); pass `-Local` to serve only
this machine.

## Presets and Mission plan layout (2026-09-27)

The Mission plan page runs top to bottom in four numbered sections: **01 Route**,
**02 Guidance**, **03 Environment** and **04 Vehicle & launch**. A sticky launch
bar at the bottom shows the route, guidance profile and environment that
**Run Isaac simulation** will fly. Each section header shows live summary chips,
so a folded section still tells you what it is set to.

![Route section](presets-route-desktop.png)

- **Route.** *Sample flight plans* is a gallery of repository plans with an
  altitude thumbnail, filtered by Hops, Landings and Hover. **Load** replaces
  only the route (see *Flight plans are routes* below). The message then
  offers the sample's suggested guidance profile and environment as one-click
  buttons. **Undo** restores the previous route. *My plans* (saved plans) and JSON import/export
  sit in one bar. In the 3D scene, legs take the colour of the step they fly
  to. Dashed plumb lines and ground shadows show each marker's altitude, and
  the selected step shows its capture sphere. Labels include altitude, and
  Fit / 3D / Top / Side set the camera. The **altitude profile** under the
  scene plots altitude against distance along the drawn route, with the route
  floor. Click a marker to select its step. The flight sequence is a timeline
  from Start to the landing. An expanded step groups its fields into Step,
  Position, Arrival and Path, and fields that do not apply to the step type
  are left out.
- **Guidance.** *Mission profile* holds the built-in convex profiles below,
  plus your saved ones under *My profiles*. **Apply** replaces every optimizer
  override. *Why these values?* lists each override against its default, with
  the evidence behind it. The card for the profile that matches the current
  settings shows *Applied*. *Key constraints* shows corridor enforcement
  (soft/strict), corridor half-width, planned and feedback tilt, speed, glide
  slope and cost as sliders, next to a schematic of the envelope. A feedback
  tilt below the planned tilt is flagged in the form. *Parameter groups* is
  always visible: the full 38-parameter editor, one diagram per group. Saving
  and loading your own profiles sits under the profile cards.
  Section 02 is shown for every controller. When the selected controller is
  not convex guidance (for example the PPO policy, which the service picks by
  default when its checkpoint is present), the section is dimmed. A notice
  explains that its settings are not sent with that run, and a button
  switches to convex guidance.
- **Environment.** Presets: Calm, Light breeze, Moderate wind, Gusty
  crosswind, Noisy sensors, COM offset 1 cm, Combined stress. Each source tab
  shows ON/OFF and has an *Apply … to this flight* switch. When a source is
  off, its values stay editable but are dimmed and marked as not flown. Gusts
  are drawn on an illustrative 60 s timeline.

### Flight plans are routes

A flight plan holds only the route: mission name, duration, start state
(position, velocity, attitude, body rate, initial rotor), landing pads and
steps. Guidance (convex optimizer) settings and the environment (disturbances)
are chosen per run and are never saved with a plan. That covers **Save**,
**Open**, sample loading, and **Import / Export JSON**, all of which use
`flight_plans.validate_route` and the `/api/flight-plan/route` endpoint.
Plan files are `edf-flight-plan` version 2 with `"scope": "route"`. Plans saved
or exported before this change still load; their guidance and environment
fields are ignored, and an import says so.

A route is validated against the widest speed limit any guidance profile may
set (10 m/s). The guidance you pick is checked at launch. If a loaded route
does not fit the current guidance (for example a 5 m/s step with a 3 m/s
speed limit), the route still loads and the message names the conflict.
Recorded missions keep the complete resolved request, guidance and
environment included.

![Guidance section](presets-guidance-desktop.png)
![Environment section](presets-environment-desktop.png)

### IMU hardware presets

The Sensor noise tab lists three sensors as **IMU hardware** cards. **Apply**
sets all four noise channels and switches sensor noise on, leaving wind and
COM as they are. The Environment chips and the launch bar then name the
sensor, e.g. "BNO085 IMU". Each card's *Datasheet & mapping* shows the source
figures.

![IMU hardware presets](presets-imu-desktop.png)

The simulator draws independent Gaussian noise at every 30 Hz control step
(`tvc_env/envs/observations.apply_sensor_noise`). It has no bias, drift or
correlation model, so each datasheet figure becomes one white-noise sigma:

- `attitude_std`: the published pitch/roll error, the dynamic figure when the
  maker publishes one, else the static one.
- `angular_velocity_std`: the gyro noise density × √(15 Hz), the control
  loop's Nyquist band, when a density is published. Otherwise the published
  gyro accuracy or stability figure.
- `position_std` / `velocity_std`: none of these parts measures position or
  velocity to landing precision, so these stay at the repository defaults
  (0.01 m, 0.05 m/s). That stands for an external source such as motion
  capture or RTK.

| Preset | Class | Source figures | attitude_std | angular_velocity_std |
|---|---|---|---|---|
| BNO085 | hobby | game rotation vector dynamic error 2.5° (static 1.5°); gyroscope accuracy 3.1°/s | 2.5° (0.0436 rad) | 3.1°/s (0.0541 rad/s) |
| WITMOTION WTGAHRS1 | hobby | X/Y angle accuracy 0.05° (static only); gyro stability 0.05°/s; GPS 2.5 m (not modelled) | 0.05° (0.00087 rad) | 0.05°/s (0.00087 rad/s) |
| VectorNav VN-110E | tactical | pitch/roll dynamic 1.0° RMS (static 0.05°); gyro noise density 5°/hr/√Hz; bias 0.6°/hr | 1.0° (0.0175 rad) | 0.0054°/s (9.4e-5 rad/s) |

Sources: the WTGAHRS1 datasheet v20-0615, §3.1 (the WITMOTION PDF). CEVA's
BNO08X datasheet (rev 1.16/1.17). VectorNav's VN-110/VN-110E product brief and
datasheet. The vendor pages for the last two were not reachable from the
build environment, so their figures come from excerpts of those datasheets;
check them against the PDFs before relying on them.

Caveats:
- The BNO085 figures are for the game rotation vector, which uses no
  magnetometer. The EDF's high-current bus (limited at 120 A) would disturb a
  magnetometer.
- The WTGAHRS1 publishes only a static angle accuracy, so its preset is
  optimistic in flight; measure it on the bench. Its GPS (2.5 m) exceeds both
  the 0.5 m noise bound and the 0.5 m landing criterion.
- A datasheet error is mostly slow bias, while the simulator's noise is
  white. White noise at the spec magnitude shakes the controller harder but
  averages out, where a bias would not.

Offline replica check (same replica as the sample table below, with
`apply_sensor_noise` on the controller's observation; not Isaac):

| Noise applied | 5 m vertical hop, 4 seeds | 18 m cold-rotor landing, 4 seeds |
|---|---|---|
| none | 0.15 m/s every seed | 0.16 m/s every seed |
| position/velocity only (0.01 m, 0.05 m/s) | 2 of 4 land hard (0.68, 0.75 m/s) | all land (0.14-0.17 m/s) |
| VN-110E attitude + rate only | all land (0.15 m/s) | 1 of 4 lost (6.4 m/s impact) |
| BNO085 attitude + rate only | all land (0.14-0.15 m/s) | 4 of 4 lost during spool-up |

Two existing controller sensitivities show up here, and neither is caused by
the presets:
- **Hover landings and position/velocity noise.** In the failing hop runs the
  vehicle is still in powered descent at 2.5 m/s through 3 m, with throttle
  held to the 0.25/s duty rate, and never reaches the gate.
- **Cold-rotor descents and attitude/rate noise.** The vehicle is lost while
  the rotor spools up from zero. Vane authority scales with rotor speed
  squared, so attitude/rate noise has most effect then.

Flying the three presets in Isaac, and treating these two sensitivities
through the convex controller's estimation and robustness, is the next step.
Neither is patched in the preset values.

### Built-in convex profiles

Stored in `mission_control/presets/convex_profiles.json` and validated by
`convex_parameters.validate_settings`. Presets are served read-only at
`/api/convex-presets` and never written. No profile changes the identified
tracking gains (kp/kd/ki) or `tilt_rate_authority_fraction`. Every profile
keeps the feedback tilt limit at least 7° above the planned tilt.

| Profile | Mission | Corridor | Planned / feedback tilt | Speed | Notable overrides |
|---|---|---|---|---|---|
| Hop - gentle low tilt | hop | soft 1.0 m, weight 25 | 8° / 16° | 2.0 m/s | first hops |
| Hop - soft corridor | hop | soft 1.0 m, weight 25 | 15° / 22° | 3.0 m/s | |
| Hop - strict corridor | hop | strict 1.0 m | 12° / 20° | 2.5 m/s | re-plan triggers 0.6 m, 0.6 m/s |
| Hop - agile wide tilt | hop | soft 1.5 m | 20° / 28° | 5.0 m/s | route_dt 0.3 s, correction 3.5 m/s² |
| Hover - precision hold strict | hover | strict 0.75 m | 10° / 18° | 1.5 m/s | re-plan 0.4 s, 0.5 m, 0.5 m/s; fly-through aim 0.4 |
| Hover - gust tolerant soft | hover | soft 1.5 m | 15° / 26° | 3.0 m/s | thrust reserve 0.6, correction 4 m/s² |
| Hover - endurance efficient | hover | soft 2.0 m, weight 5 | 10° / 18° | 2.0 m/s | 2 solver workers, re-plan 0.75 s |
| Landing - precision strict | land | strict 0.75 m | 12° / 20° | 3.0 m/s | glide 40°, gate 0.6 m, capture 0.25 m, 30 nodes |
| Landing - soft touchdown | land | soft | 15° / 22° | 3.0 m/s | descent/brake 0.8, gate 0.8 m |
| Landing - crosswind soft | land | soft 1.5 m | 18° / 26° | 4.0 m/s | glide 40°, reserve 0.6, centring 0.3 m |
| Landing - high altitude fast | land | soft 1.5 m | 15° / 22° | 6.0 m/s | 32 landing nodes, solve 0.6 s, re-plan 0.6 s |
| Landing - delta-v objective | land | soft | 15° / 22° | 4.0 m/s | objective delta_v |

### Sample flight plans

Seventeen routes in `mission_control/presets/flight_plans/` (route-only
edf-flight-plan version 2, stored exactly as `validate_route` returns them),
served at `/api/flight-plan-samples`. Each names a suggested guidance profile,
and samples 10, 13 and 15 also suggest an environment. Neither is applied
unless you click it. Hops and
hover missions start on the ground with the rotor stopped. Airborne landing
starts use the 0.84 hover rotor fraction unless the sample tests a cold rotor.

Before commit, every sample was flown in an offline closed-loop replica. The
replica is the rigid-body plant from `tests/unit/test_convex_guidance.py`: rotor
gyro, servo model with deadband, 25 ms vane-joint lag and momentum-bounded
vanes. It adds a ground plane for launches, the mission `WaypointMission`
sequencer and the real `ConvexGuidanceController` with the suggested profile's
resolved settings. It is **not Isaac**. It models no contact dynamics, gust or
sensor-noise realisation, and wind only as body drag. It checks feasibility
and sequencing, not flight qualification.

| Sample | Profile | Replica touchdown | Pad error | Flight time |
|---|---|---|---|---|
| 01 Hop 2 m - pad to pad | hop-gentle | 0.160 m/s | 0.005 m | 16.6 s |
| 02 Hop 5 m - vertical | hop-soft | 0.149 m/s | 0.000 m | 15.3 s |
| 03 Hop 10 m - 8 m lateral transfer | hop-soft | 0.152 m/s | 0.000 m | 34.6 s |
| 04 Hop 20 m - high vertical | hop-strict | 0.166 m/s | 0.000 m | 43.5 s |
| 05 Hop 3 m - strict slalom | hop-strict | 0.171 m/s | 0.002 m | 29.3 s |
| 06 Hop 6 m - agile 16 m transfer | hop-agile | 0.155 m/s | 0.001 m | 23.1 s |
| 07 Landing 18 m - nominal descent | land-precision | 0.157 m/s | 0.052 m | 12.9 s |
| 08 Landing 30 m - offset high altitude | land-high-altitude | 0.151 m/s | 0.017 m | 10.5 s |
| 09 Landing 25 m - 6 m/s fast sink | land-high-altitude | 0.154 m/s | 0.001 m | 8.6 s |
| 10 Landing 15 m - crosswind and gusts | land-crosswind | 0.159 m/s | 0.043 m | 16.7 s |
| 11 Landing 15 m - divert to remote pad | land-precision | 0.156 m/s | 0.029 m | 19.4 s |
| 12 Landing 20 m - attitude upset recovery | land-crosswind | 0.155 m/s | 0.029 m | 10.2 s |
| 13 Landing 12 m - noisy sensors and COM offset | land-soft-touchdown | 0.123 m/s | 0.003 m | 35.5 s |
| 14 Landing 12 m - cold-rotor drop | land-soft-touchdown | 0.123 m/s | 0.000 m | 10.2 s |
| 15 Hover 10 m - station keeping in gusts | hover-gust | 0.156 m/s | 0.007 m | 34.4 s |
| 16 Hover 6 m - survey box | hover-precision | 0.152 m/s | 0.000 m | 54.7 s |
| 17 Hover 4 m - 60 s endurance hold | hover-endurance | 0.151 m/s | 0.000 m | 71.5 s |

Every run completed its route. The strict-corridor samples (04, 05, 11, 16)
each spent about 0.5 s in HOLD, 13-16 control steps at 30 Hz, while no strict
plan was feasible. They then planned and flew normally. Sample 13 carried a
lateral COM offset of (7, -5) mm. None of these has been flown in Isaac yet,
so treat them as starting points and confirm each in Isaac before relying on
it.

## Advanced flight plans

The route scene now sits beside a selectable flight sequence. Expand one step
to edit it and see its arrival conditions. Environment controls have a separate
section with Wind, Sensor noise and Center of mass tabs; saved optimizer
profiles are under **Guidance → All parameters & saved profiles** (see the
section above for the current layout). See the
[2026-09-27 verification](waypoint_planner_2026-09-27.md) for the fix to repeated
replans postponing waypoint arrival, two successful Isaac replays and screenshots.

Choose **Convex** on **Mission plan**, then add, name, drag, edit and reorder
up to 12 steps. Click a marker in the 3D editor to show its red **X**, green
**Y** and blue **Z** arrows; drag an arrow to move along that axis alone.
Takeoff/descent expose only Z because their X/Y follow the preceding point;
landing follows its pad (move the pad marker in X/Y). Right-click a waypoint
to edit its name, type, speed, hold, radius, corridor and landing parameters.
Right-click empty space to add a step at the selected waypoint's height, or
3 m when none is selected; new airborne steps are inserted before landing.
The editor also supports orbit, right-drag pan, zoom, **Fit route** and
**Full screen**, including the parameter menus. The editor uses the full
page width and a large responsive viewport. Expand **Precision views** for top/side
editing, or enter exact coordinates in the rows. Takeoff/descent remain vertical.

Create up to **four named ground pads**, at least 3 m apart. Drag their green
markers or edit their X/Y coordinates. The final Landing step chooses the pad;
without a Landing step, the first pad is the target. PPO remains origin-pad only.
The physics success check, controller, route, and replay pad markers use the
same selected pad. Pads are targets on the existing flat ground, not raised
platforms; Z is fixed at zero.

**Save plan** stores the mission locally in `mission_control/library/plans/`.
Choose it in **Saved plan** and click **Load saved** to restore it. Saving the
same mission name replaces the existing plan. **Export JSON** downloads a
portable version 2 `edf-flight-plan` file containing the entire mission: initial conditions and rotor state, steps, pad selection, corridor
widths, speeds, optimizer overrides, hardware, battery, disturbances, seed,
duration and Fast live setting. **Import JSON** validates the whole file before
changing the draft. Version 1 files still load, restoring the original single
origin pad and retaining the other launch settings. An example for
loading is [advanced_flight_plan.json](advanced_flight_plan.json); allow at
least 90 seconds for its complete route. Names, sequence numbers and phase
colors appear in both projections and the Flight camera; replay labels use
the recorded plan even while another draft is edited.

| Step | Parameters and completion |
| --- | --- |
| Takeoff | First step, above the start. Vertical climb at the selected speed limit, finishing within the capture radius at low speed. |
| Hover | Approach speed limit, capture radius and continuous hold duration. Leaving the radius or exceeding 0.4 m/s resets the hold. |
| Fly-through | Leg speed limit and arrival speed along the route tangent; swept capture permits passing through without stopping. |
| Descent | Vertical leg below the previous point, selected descent speed limit, low-speed capture at its endpoint. |
| Landing | Last step, selected pad, approach speed limit and terminal touchdown speed; physical contact and settling required. |

Takeoff/descent X/Y follows the preceding point (takeoff follows the start).
Add a hover before a vertical descent when a transit requires braking.
Convex route speeds are 0.1 m/s up to the configured optimizer speed cap (4 m/s by default); touchdown is 0.1–0.5 m/s. Speeds are
optimization constraints, with acceleration/deceleration at stop points and
an actuator-limited braking allowance when entering above the requested
speed. In soft corridor mode, emergency soft-terminal plans can relax these constraints
and remain identified as approximate in telemetry. A speed setting cannot
make a physically infeasible maneuver feasible.

Omitting Landing uses the first pad at the default 0.15 m/s touchdown speed.
Landing is not counted as a captured airborne waypoint. Advanced types are
restricted to convex missions; the PPO observation/action contract is unchanged.
For a ground start use an upright pose, zero velocity and Z = 0.34 m. An
explicit Takeoff keeps launch contact from completing the landing dwell
until the vehicle clears 0.8 m; contact telemetry and crash checks stay active.

Verification (2026-09-26): 83 targeted Python tests and 19 browser-module
tests pass. Mission `fa510617ea44` flew all five airborne waypoints from a
cold rotor with wind and sensor noise, then landed successfully and settled
8.8 cm from the pad center. Its requested 0.20 m/s touchdown reference
reached the controller; recorded impact was 0.229 m/s. Total mission time
including shutdown was 44.22 s. This is simulation validation, not hardware
qualification. The browser editor, parameter changes, import and generated
export payload were also checked.
See the [planner](advanced_plan_check.png) and
[labeled flight view](advanced_flight_check.png).

## Corridors, optimizer profiles and longer live runs

The blue tube is the **incoming leg's corridor half-width**, in meters; it is
separate from the waypoint capture radius. Blank widths inherit **Default
half-width** under Convex optimizer. Landing has separate approach and touchdown
speeds. The displayed convex spline uses the same vertical legs and lower-end
altitude clamp as guidance. An implicit direct landing has no route corridor;
add a Landing step to give it one.

**Convex optimizer** exposes validated guidance and tracking settings, including
speed/tilt/thrust bounds, objective, discretization, replanning, corridor width
and penalty. **Save profile** (under *All parameters & saved profiles*) stores a
named profile locally under `mission_control/library/convex/`; **Load** restores
it, it also appears under *Mission profile → My profiles*, and **Restore
repository defaults** removes the overrides. Saving an existing name replaces that profile.
Profiles apply to future runs; recorded requests and resolved parameters remain
with each mission. The full flight-plan export also carries the overrides.

Corridor enforcement is explicit:

- **soft** (the existing default) penalizes departures and reports maximum
  planned corridor excess. An infeasible route can use the visibly labelled
  soft-terminal emergency fallback, which relaxes speed/route constraints.
- **strict** fixes corridor slack to zero and additionally checks the sampled
  trajectory against the actual curved centerline, including eight samples per
  SOCP interval. An infeasible corridor does not receive an unconstrained
  fallback; guidance holds if it has no usable plan. This is a sampled planning
  check, not a continuous-time safety certificate or a tracking guarantee.

Waypoint speeds constrain SOCP node velocities; fly-through also fixes tangent
arrival velocity. Overspeed entries have an actuator-limited braking allowance.
Real tracking, wind and physical feasibility still matter. The Flight optimization
panel shows corridor excess, solver/fallback status and the selected pad.

Missions now allow **1–600 simulated seconds**, with a 120 s default. **Fast live**
reduces redundant headless Kit redraws to one every 0.25 wall seconds, retaining
all physics substeps, controller calls, contact checks, sensor processing and
recorded telemetry. The browser renders the same recorded poses. Independent
SOCP duration candidates can run on 1–8 solver workers (default 4); the equations,
node counts and replan period are unchanged. More workers need not be faster on
every machine, and wall-time solver limits can be affected by CPU contention.
Summaries record simulation wall time and real-time factor, excluding startup.
Replay playback speed is separate from live simulation throughput.

Verification: 116 targeted Python tests and 21 browser-module tests passed.
Browser checks covered 3D dragging, full-screen sizing, JSON import, local plan
save/load, and optimizer profile save/load.

A complete two-pad example is [multi_pad_flight_plan.json](multi_pad_flight_plan.json).
Isaac mission `0875cdb81ef5` completed its takeoff, hover and descent sequence,
landed on the second pad at **[4, 0, 0]**, and settled successfully: **0.154 m/s**
impact, **0.0031 m** final pad error, **24.59 s** simulated duration. Its recorded
waypoint widths and resolved optimizer overrides were checked end to end.
The full-redraw comparison `a37e1d87ac33` produced the same **739 telemetry
frames exactly**, excluding solver wall-time measurements. Both used 120 Hz
physics and the same control decimation. This route did not show a net redraw
speedup (full redraw: 108.3 s excluding startup); physics/optimization dominate.
An isolated duration-search benchmark of that route measured **0.416–0.425 s
with one worker versus 0.282–0.301 s with four**, with the same 24 candidate
solves and optimal result. Thus parallel planning reduced planner latency by
about 30% here; it does not make the complete simulation real-time.

## Console layout (2026-09-24)

The console is split into five pages, reached from the tab row or keys 1–5
(the URL hash, e.g. `#telemetry`, is bookmarkable):

1. **Flight.** Mission archive and export bar, the camera array with the
   webcast band, four key charts and the vehicle column.
2. **Mission plan.** Launch card (name, controller, hardware, seed, duration,
   run), the route planner, and initial-state, disturbance and LiPo cards.
   Launching switches to Flight.
3. **Telemetry.** All eight strip charts, flight rotation totals, fin motion
   rates, power system readouts and the event log.
4. **Flight software.** Registered policy, latest training runs, and the
   selected run's history, curriculum, evaluation and checkpoints.
5. **Checklists.** Physical testbed pre-flight checks, saved with timestamps
   in this browser. Open directly with `#checklists` or key 5. Checks use the
   current mission draft settings and update the shared GO/NO-GO board;
   they do not gate simulation launches.

The header, clock and GO/NO-GO board stay on every page. The panels are:

- **Header and GO/NO-GO board.** Mission clock; Isaac, trainer, next
  controller, telemetry, vehicle and power states. Each state shows a glyph
  and a word; colour is never the only signal.
- **Camera array** with a webcast band:
  - Speed, altitude, thrust and LiPo gauges.
  - A mission timeline of recorded milestones (start, waypoint captures,
    contact, landed/crashed, motor off, outcome).
- **Vehicle telemetry column:**
  - Attitude indicator: ZYX Euler from the recorded quaternion, plus tilt.
  - Altitude, vertical and ground speed, pad error, thrust and thrust-to-weight.
  - Fin deflection bars against the servo limit (bar is actual, tick is command).
  - Gyro bars spanning ±2× each soft limit, with dashed limit markers.
- **Telemetry strip charts.** Eight charts on one time axis: altitude, velocity,
  distance, thrust, body rates, fin angles, bus voltage and pack current. Hover
  to read values; click or drag to seek. Dashed lines are limits; vertical
  hairlines are the timeline milestones. Space plays or pauses the replay;
  ←/→ step 0.5 s (Shift: 5 s).
- **Flight software panel.**
  - Lists the registered mission-flyable policy.
  - Lists the latest `runs/waypoint_flight` training runs from their logs:
    curriculum stage, stage success, outcome mix, peak yaw, checkpoints, and
    a history plot. Nothing is loaded from the checkpoints.
  - These 54-channel `waypoint_flight_v1` checkpoints are listed for inspection
    only. `run_mission.py` does not yet build explicit waypoint missions for
    them.

Both `run_train_ppo.py` and `run_train_waypoints.py` are recognised as trainers
that own Isaac. Mission launches are refused while either runs.

## Draggable mission planner and experimental recovery policy

Use the top **X/Y** and side **X/Z** canvases to drag the cyan starting diamond
or numbered waypoints. Changes update the numeric form immediately. X/Y can
range from −100 to +100 m; Z is above-ground altitude, from 0.34 to 100 m for
the starting body origin. Waypoints allow 1–100 m altitude. Negative altitude
would start underground and is rejected. View range controls zoom, not the
allowed bounds. The amber velocity endpoint represents two seconds of initial
velocity; use numeric velocity fields when the zero-length arrow overlaps the
start marker. **Invert Start** toggles roll between 0° and 180°.

**+ HOVER** creates a timed hold, default **2 seconds**. Hold time accrues only
continuously within its radius at speed ≤0.4 m/s; leaving resets the timer.
**+ FLY-THROUGH** advances on a forward swept pass within the acceptance
radius. Edit position, radius, speed and hover duration in the waypoint rows;
reorder with ↑ or remove with ×. Up to twelve points can be edited; the
training task currently samples zero to three. Every route ends at the
selected landing pad (the origin for PPO). The dashed curve is a reference path, not a scripted controller.
Routes whose waypoint splines dip below ground clearance between control
points are rejected; raising the neighboring points can remove the overshoot.

Waypoint missions require the **EXPERIMENTAL PPO · latest recovery + waypoints**
choice registered in `mission_policy_registry.json`, which is pinned by path and
SHA256 to one checkpoint. Registering it does not qualify it or alter archived
recordings, and each new mission logs the exact checkpoint file and hash it
loaded.

The deployed save is the final one from the 200,278,016-transition staged
waypoint run (`runs/ppo_waypoints_staged_v5`). It trained only through
curriculum stage 1 — ±3 m XY, 8–15 m altitude, ±0.15 rad tilt and **zero
waypoints** — and never advanced, so route following, recovery and inverted
starts have had no training at all and are not capabilities of this model.
At that stage it reached 36.1% mission success, 60.0% landed, 0.412 m mean pad
distance and 0.197 m/s mean touchdown speed. The uncurricularised full-task
evaluation scored 0/8,192, which is the expected result for a stage-1 policy
rather than a measurement of the approach. Whole-flight peak yaw stays near
1,625°/s against a 180°/s soft limit and a 172°/s contact gate; that unresolved
spin drives the remaining 40% crash rate.

The legacy 26M and 34M landing policies were retired on 2026-09-17. They were
trained before the radial fin hinge correction, so their fin mapping does not
match the vehicle this simulation flies, and they never observed waypoint
targets. The PID baseline and the legacy vane physics were removed from
mission control on 2026-09-26 (below); their recorded missions still replay.
Training status shows current stage and independent full-task evaluation
separately. Simulation launch is disabled while the trainer owns Isaac; editing
plans and replay remain usable.

To test interactively before a long training run finishes, create a file named
`STOP` inside its run directory. The trainer finishes its current rollout or
evaluation, saves `ppo_final.pt`, and releases Isaac. Wait for the training
banner to clear before launching a mission. The browser's **STOP RUN** button
only stops a mission, not PPO training. The active run and exact launch options
are recorded in `runs/ppo_waypoints_staged_v5/*/args.json`.

The recorded mission phase displays the active waypoint, cross-track error
and hover timer. Editing a draft does not alter the reference route in replay
cameras or exported footage. Old recordings retain their original telemetry;
missing navigation or cumulative rotation is shown as unavailable.

The whole-flight rotation panel reports per-axis peak rate, angular travel,
excess rotation and seconds above the provisional 90/90/180°/s limits. Travel
counts turns and reversals, not wrapped Euler differences. Amber gyro values
and dashed plot lines identify limit exceedances. Flight totals freeze at
terminal contact; motor-off settling remains a separate procedure.

Set position, world velocity, attitude, body FRD angular rates, initial rotor
speed, seed, controller, disturbances, and battery parameters. Wind/gusts,
sensor noise and center-of-mass offset are independent checkboxes and can be
combined. **Run Isaac
Simulation** launches a real headless Isaac/PhysX process. Startup takes about
10 seconds on this machine; simulated time can advance slower than wall time.
The archive retains completed and failed missions. Stop preserves the samples
already recorded. Replay supports seeking, speed control and orbit-camera drag.

Four synchronized views show the actual USD body and fin poses: orbit, ground
tracking, overhead and an orthographic bottom view. The bottom view shows the
four measured fin angles beside their fins, with FWD at the top. Looking up
from below puts the vehicle's RIGHT fin on the left of the image. The body
and ground overlays are hidden in this view. The rendered vehicle is the
textured CAD/Blender model (`CAD/EDF Drone v1/Blender/usd_v2.blend`) with its
Inventor materials, reduced from 691,680 to about 152,000 triangles by
`mission_control/export_visual_model.py`. Poses still come from the physics USD
links; the export refuses to write if any link's mesh bounds differ from
`geometry.json` by more than 1 mm. Fin plates are coloured to match their
bottom-view labels. Regenerate after a CAD change with
`blender -b "CAD/EDF Drone v1/Blender/usd_v2.blend" --python simulation/isaac/mission_control/export_visual_model.py`.
Camera images use Three.js/WebGL; they are **not Isaac RTX renders**.
The CAD fan blades and spinner rotate together about the EDF shaft using
recorded `rotor_rpm`, including spool-up and motor-off coast-down. Blade
animation is slowed **200×** to keep rotation visible at display frame rates;
the RPM readout remains the actual recorded speed. Phase follows mission time,
so pausing freezes the fan and seeking, playback speed and video export stay
synchronized. This is a visual child of Body, not a new physics link.
The ground presentation and thrust arrow are visual overlays. Physics, contact,
fin articulation and trajectory all come from Isaac, not browser animation rules.

**Export Video** records all four cameras plus telemetry into a local WebM file.
Keep the tab visible while it records. Each run stores request, source hashes,
checkpoint hash, resolved physics parameters, frames, outcome and video in
`simulation/isaac/runs/mission_control/<mission id>/`. JSON telemetry is also
downloadable. `landing.webm` is a replay visualization, not camera sensor data.

## Convex (SOCP) guidance controller (2026-09-25)

**CONVEX · SOCP powered-descent guidance** is a third controller choice. It
is a deterministic classical controller with no checkpoint.

- **Planning.** A second-order cone program (Açıkmeşe & Ploen 2007, lossless
  convexification) plans a minimum-energy thrust trajectory to the pad. It
  flies waypoint routes too: fly-through and timed hover points.
- **Constraints.** Thrust bounds, a 15° tilt cone, a 45° glide slope,
  4 m/s speed and the flight computer's thrust-rate bound.
- **Route corridor.** Explicit routes use per-leg corridor widths, falling back
  to the configured default of 1 m (soft by default; strict is selectable), never below the lower
  end of a waypoint leg. Before, plans only met the waypoints: from a
  20 m/s descent one sank 26 m below its fly-through and climbed back,
  28 m off the line (`835c3de32185`). On a corridor, a fast start may brake
  at full thrust instead of the reserved ceiling.
- **Launch-form route check.** The board and route planner estimate
  full-thrust braking from the start velocity, after the rotor spools up
  (a dashed line to × on the planner). ROUTE CHECK warns when the vehicle
  cannot stop above the ground, when the start carries it off the first
  leg by more than the waypoint radius, or when the drawn route dips more
  than 1 m below both ends of a leg. The estimate uses the Isaac plant's
  mass and full-rotor thrust (`models.PLANT_THRUST`) and applies to every
  controller.
- **Closed loop.** The plan is re-solved every 0.5 s from the measured
  state. A tracking loop and a geometric attitude loop fly it, ending in a
  velocity-commanded terminal descent. The planned thrust is continuous
  (first-order hold). Attitude is a full-state LQR over the body, the rotor
  gyro and the vane actuators (`attitude_lqr.py`), scheduled on rotor speed.
  On momentum-bounded vanes every PD law is unstable: the rotor's nutation
  (~2.5 Hz) sits where the vanes lag. That PD instability was the 1–4 Hz
  wobble. The attitude loop also pre-compensates the vane servos' 1°
  deadband.
- **Flight page.** The Convex optimization panel compares the recorded plan
  with the flown ground track, altitude and thrust. It shows planned energy,
  time to gate, tracking error, solve time, relaxation gap and instantaneous
  thrust headroom (available minus commanded). Previous/next-plan buttons
  seek to recorded replans. The camera's PLAN OVERVIEW frames the trajectory
  with vehicle, reference and endpoint markers; VEHICLE restores the close-up.
  Green denotes the plan, white the flight so far, and blue the reference.
  Fallback plans appear amber, and terminal descent labels the retained plan
  as historical. The amber thrust line is the current available thrust, not
  an optimization bound; those bounds were not recorded. Planned thrust is
  the optimizer's slack sigma. Guidance events remain on the timeline, with
  PLAN and COMMAND traces on the key-channel charts. Older recordings without
  guidance hide the optimization panel.
  Four diagnostic cards add solve-time bars with fallback markers and recent
  plan seek buttons, tracking-error history with sample RMS and peak, battery
  energy consumed alongside discrete per-plan energy costs, and thrust reserve
  history with a command/available gauge. All histories and statistics stop at
  the selected replay time and reset on backward seeks. Plan energy costs cover
  different horizons and are neither added together nor treated as cumulative
  consumption. Missing battery data and delta-v objectives do not fabricate
  energy values. Solver bars represent recorded plans, not unrecorded failed
  attempts or internal convergence iterations.
- **Requirements.** The Clarabel solver from `requirements.txt`. Waypoint
  missions also need the LiPo model enabled.
- **Vane physics.** Convex missions fly the momentum-bounded jet: the
  coupled-jet model the waypoint_flight PPO trains on, with a torque-limited
  motor. The motor applies at most 0.76 N·m, and at zero throttle (ESC brake
  off) the rotor coasts. That removed a 430–710°/s pad spin after every
  landing. PPO missions always fly their training plant.

  On 2026-09-26 the *legacy airfoils + damper* plant (8.5× too much vane
  torque per degree) and the PID baseline, which only flew it (on
  momentum-bounded vanes it drifted 11 m off the pad, `6421f6ec55a4`), were
  removed from the launch form and the service; requests naming them are
  rejected. Missions recorded with them still replay, labelled LEGACY VANES.

  Start in the air with the rotor spinning, or spool up on the pad: a cold
  in-air spool-up spins the body, and the launch board warns about it. A
  spawned spinning rotor now starts on its loaded bus; the open-circuit
  voltage had made every warm start yaw 30°/s.

Convex missions raise the landing task's 30 m altitude fail-stop to 105 m,
so starts up to the planner's 100 m ceiling are flyable. On 2026-09-26, on
momentum-bounded vanes, it landed nine of nine validation missions
0.004–0.15 m from centre at 0.14–0.18 m/s. They include the waypoint route,
the 120 s mission trial, the 6S pack and wind + sensor noise + CoM shift.
Terminal-descent body rates were 0.7–4.6°/s, against 15–30°/s limit cycles
before. With the route corridor (2026-09-26) route flights hold the
drawn line to 1.2 m (fly-through + hover, L-turn) and 3.5 m on the 120 s
trial, where they had strayed 1.9–4.3 m; the 20 m/s dive of `835c3de32185`
now bottoms at 17.5 m instead of 6.6 m. Formulation, the plant audit, the plant identification behind every
gain, the wobble diagnosis and the full validation table are in
`docs/convex_guidance_2026-09-25.md`.

## Hardware provenance

The user selected the **planned 8S system**, with purchased/fitted details still
pending, on September 13, 2026. The master parts workbook contains only carbon
tubes (10×8, 12×10 and 16×14 mm) and an M3.5×25 screw entry. It does **not** confirm
an EDF, ESC or battery SKU. `Paper/Reference/parts.md` is the electronics plan;
the old Isaac parameter file instead describes a 6S system. These are separate
selectable profiles. No estimate is represented as a new hardware measurement.

| Component | Planned 8S model | Basis and remaining uncertainty |
|---|---|---|
| EDF | FMS 90 mm, 12 blades, 4075 KV1500 candidate | [FMS manufacturer](https://www.fmshobby.com/products/edf-system-90mm-12-blade-8s-power-system-with-4075-kv1500-motor-pro-metal) confirms this 8S product, SKU FMSEDF010. Installation is unconfirmed. 48 N is the project planning estimate, not a verified thrust-stand curve. |
| ESC | FLYFUN 120A V5 candidate; 120 A conservative ceiling | [Hobbywing manual](https://www.hobbywing.com/en/uploads/file/20221015/12f49cbe05185401b0773cfe8f019dce.pdf): 3–8S, 120 A continuous, 150 A peak. The model does not reproduce proprietary protection firmware. |
| Servo | Four MG996R, regulated 6 V; 1.079 N m stall; 0.15 s/60° | [TowerPro](https://towerpro.com.tw/product/mg996R/) specifies these values and 55 g each. Old 0.14 s assumption is replaced in the 8S profile. Loaded response and linkage deadband still need measurement. |
| Battery | 8S, 29.6 V nominal / 33.6 V full; editable 5 Ah, 45C default | Pack SKU unconfirmed. Capacity/C-rating are scenario assumptions consistent with [FMS's 8S application recommendation](https://www.fmshobby.com/products/fms-edf-jet-90mm-super-scorpion-v2-pnp), not an identified purchased pack. |
| Mass | Existing 3.104 kg Isaac assembly | 3.1 kg body plus four 1 g fins. Battery and servo masses are already lumped into the body; they are not added twice. Capacity changes do not silently rescale mass or inertia. Actual assembly weighing and inertia measurement are pending. |

`configs/hardware/planned_8s.yaml` holds the versioned overrides. The complete
electrical defaults are in `configs/params/battery_6s.yaml`, extended by the 8S
profile and validated form input. Every run records the resolved values.

## Battery physics and limits

The pack uses a generic one-RC Thevenin circuit: rested OCV versus SOC, ohmic
resistance, polarization voltage, coulomb counting, heat generation and cooling.
This circuit structure follows the standard [Thevenin equivalent-circuit
model](https://docs.pybamm.org/en/v25.12.0/source/api/models/equivalent_circuit/thevenin.html);
it is a local implementation, not a fitted PyBaMM chemistry model.

Motor target speed depends on loaded voltage. Estimated fan shaft power scales
with RPM cubed; rotor acceleration consumes kinetic energy. A current/power
constraint limits motor speed, and cutoff latches until the next reset. Charge
and energy telemetry are integrated at every physics substep. No regenerative
charging is assumed. The current ceiling is the minimum of the user setting,
capacity × C-rating, and the planned ESC's 120 A continuous limit.

Resistance (3 mΩ per cell), RC coefficients, 88% efficiency, shaft power and
thermal coefficients are estimates requiring pack/EDF bench identification.
Rated KV provides a no-load RPM bound, not measured loaded RPM. The existing
EDF thrust cap is not increased at full charge. Avionics/BEC/servo demand is a
lumped 10 W load; individual servo stall current, BEC faults and power loss in
PhysX servo drives are not modeled. This is a propulsion-coupled battery model,
not a complete electrical system or hardware flight qualification.

## Telemetry interpretation

- Position and linear velocity: world XYZ, Z up. Altitude is the body-link
  origin, not foot clearance; nominal contact occurs near 0.31 m body altitude.
- Quaternion: `w,x,y,z`, from the actual articulation. Fin poses are also
  recorded individually; the renderer does not guess hinge axes.
- Gyro: body FRD, radians/s in JSON, degrees/s in the display. The controller's
  noisy gyro observation is recorded separately where available.
- Fin commands and angles: radians in JSON, degrees on screen. Target motion
  rate is the change in the rate-limited servo target divided by the control
  interval. It is distinct from measured joint velocity.
- These PPO policies output fin angles and throttle directly. There is no
  commanded body-rate target. In recorded PID missions the internally named
  `rate_cmd` is a fin-mixer control signal, not a calibrated body-rate
  setpoint; it is not plotted as one.
- Applied thrust includes fin axial losses. Raw thrust, RPM, throttle, battery
  voltage/current/power/SOC/temperature/energy and contact force are logged.
- Propulsive delta-v is the time integral of the magnitude of EDF plus fin
  force divided by the current vehicle mass, in m/s. It excludes gravity and
  wind, and is neither net velocity change nor rocket-equation delta-v.
  Energy and delta-v are integrated at every physics substep.
- A settled contact state alone is not a pass. Success additionally requires
  worst recorded impact ≤0.25 m/s and pad error ≤0.50 m. Failures remain visible.

New mission replays add an explicit two-second motor-off procedure after the
LANDED event, beyond the active flight duration. Fin and throttle commands go
to zero; Isaac continues to simulate rotor coast-down, contact and battery
load. This is the flight executive's terminal procedure, outside the PPO
episode. It does not create or help reach the landing event. The final half
second must retain contact with speed below 0.05 m/s, body rate below 0.15 rad/s,
tilt below 0.2 rad and pad error within 0.5 m. A failed settling check is shown
as POST_LANDING_FAILURE. The original landing event must also pass its own
impact/pad gates. Summary fields separate flight energy/time from the added
shutdown recording; batched PPO evaluation reports the original episode gates.

## Radial hinge correction and retraining

On September 14 the user confirmed that hinges run along each fin's radial
span. Inspection of the USD showed the old joints perpendicular to that span,
rotating the thin vane in its own plane. Both the USD revolute frames and the
aerodynamic hinge/normal metadata now follow the radial span. The neutral CAD
pose is unchanged. This gives the four-vane mixer roll, pitch **and yaw**
authority. A common positive fin angle produces positive FRD yaw; FWD positive
deflection produces negative roll and positive yaw. The previous asset is
preserved as `drone_v2_tangential_hinges_legacy.usd` for historical reproduction.

The old 97.7% PPO result belongs to the previous 6S, ideal-voltage, incorrect-
hinge model. It does not establish performance with the corrected mechanism.
Old recordings preserve their recorded poses and are labeled as old-hinge
archives. Initial pre-correction 8S tests: legacy PPO timed
out near 0.53 m after 30 s; legacy PID settled on the pad with 1.459 m/s impact
and therefore failed the soft-landing criterion. The archive preserves both.
New radial training uses explicit `train_512_8s_radial.yaml`. It initializes
only the actor/distribution from the 8S transfer, with a documented initial
output-channel permutation for the changed actuator basis. Critic, optimizer
and step count start fresh. There is no runtime action permutation or scripted
landing controller. The 28 policy inputs include SOC, normalized voltage,
normalized current and polarization voltage. New input weights start at zero
and learn through PPO. Four explicit spawn stages progress from short,
pre-spooled approaches to the full 16–20 m cold-rotor task. Evaluations also
run against the full task independently of the training stage.

The reward no longer pays an alive bonus. Electrical work costs 0.5 per Wh,
and propulsive delta-v costs 0.02 per m/s. A 30 s example at 2500 W and 10 m/s²
costs 16.42 units, secondary to the +475 ideal terminal reward and −200 crash
penalty. This budget is checked offline without global reward scaling.
Evaluation reports energy, delta-v and time for **successful episodes**
separately so early crashes cannot appear efficient. Initial charge is currently
fixed at 100% for this curriculum; lower-charge performance needs a separate
evaluation and, if necessary, explicit battery-domain training.

The UI disables new GPU simulation runs while repository PPO training is
active; existing recordings remain available. A corrected-physics policy is
made available as `ppo_radial` only after validation, through the local
`mission_control/policy_registry.json` checkpoint entry. Legacy choices are
labeled as such; their results must not be confused with radial validation.

`tools/verify_mission_hinges.py <mission folder>` checks recorded Isaac link
quaternions against the four expected radial rotations, independently of the
renderer. The fresh run `8b0000000001` passed all four axes across 451 samples:
maximum angular disagreement below 0.000034°. Its FWD fin at −5.645° moved
an axial 78 mm chord vector sideways by 7.67 mm, with negligible radial
movement. This run verifies articulation; its PID landing timed out and is
explicitly marked as unsuccessful.

At 4.06M radial training steps, the fresh critic had near-zero explained
variance and an absolute output bound of only 20.8 versus +475 terminal
rewards. Training continued from the saved 6.029M checkpoint with actor Adam
LR 3e-5 and separate critic LR 1e-3. Both networks retain their learned weights
and Adam moments. Unit verification checks that migration from the old single
optimizer group preserves the next update exactly at equal learning rates.

Verification: unit checks cover circuit equations, charge/energy accounting,
low-charge thrust reduction, current and kinetic-energy bounds, cutoff, selective
reset, API input validation, origin protection and partial telemetry lines.
Browser checks cover actual mission submission, four cameras, replay, hardware
sources and video generation. The service binds to loopback only and launches
fixed local simulator commands without a shell.

### Visual disturbance design

Mission Plan → Design disturbances opens the editor below the route preview.
Drag the wind compass or use its speed/direction controls; the 3D route shows
the steady airflow vector in world XYZ (Z up). Flow direction describes where
the air travels, from +X toward +Y. Gusts add a random horizontal vector for
the specified duration, with a uniformly sampled wait between events.

Sensor sliders specify independent Gaussian standard deviations in the
simulator's observation units. COM bounds specify a uniform box sampled at
reset in body FRD, shown in millimeters and stored in meters. Equal lower and
upper bounds fix an axis. Diagrams show configured distributions, not a run's
sampled disturbance history. Enable each source separately; Reset parameters
restores the YAML values while retaining the source selections.

The bounded `disturbance_settings` mission field is preserved by saved plans,
JSON import/export and run metadata, then merged into the existing disturbance
models. Requests without it retain the repository presets. Restart an already
running mission-control service after updating the backend; until then the UI
continues to offer the original preset toggles.
