# EDF mission control

Run `simulation/isaac/mission_control/start.ps1` in PowerShell, then open
http://127.0.0.1:8830. It uses this repository's Isaac Python environment.
For first-time dependencies run that Python with `-m pip install -r
simulation/isaac/mission_control/requirements.txt`. The launcher installs the
locked browser dependencies and builds the frontend. No cloud service is used.

## Console layout (2026-09-24)

The console is split into four pages, reached from the tab row or keys 1–4
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
landing pad. The dashed curve is a reference path, not a scripted controller.
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
targets. **PID · radial hinges** remains available for non-waypoint missions.
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
- **Closed loop.** The plan is re-solved every 0.5 s from the measured
  state. A tracking loop and a geometric attitude loop fly it, ending in a
  velocity-commanded terminal descent. The planned thrust is continuous
  (first-order hold). The attitude loop pre-compensates the vane servos' 1°
  deadband, which had kept every flight in a ~1 Hz, 3° coning wobble.
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
- **Vane physics.** The launch form's VANE PHYSICS selector (request field
  `vane_model`) picks the plant for PID and convex missions:
  - *Momentum-bounded jet* is the default: the coupled-jet model the
    waypoint_flight PPO trains on.
  - *Legacy airfoils + damper* has 8.5× too much vane torque per degree; it
    is kept to reproduce missions flown before 2026-09-25.

  PPO missions always fly their training plant. On momentum vanes, start
  with the rotor at hover: a cold in-air spool-up spins the body.

Convex missions raise the landing task's 30 m altitude fail-stop to 105 m,
so starts up to the planner's 100 m ceiling are flyable. On 2026-09-25 it
landed the default mission (18 m, cold rotor) at 0.146 m/s, 0.08 m from
centre, with 1.7°/s roll/pitch RMS. The PID baseline hit at 1.50 m/s.
Formulation, the plant identification behind every gain, the wobble
diagnosis and the full validation table are in
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
  commanded body-rate target. PID's internally named `rate_cmd` is a fin-mixer
  control signal, not a calibrated body-rate setpoint; it is not plotted as one.
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
