# EDF mission control

Run `simulation/isaac/mission_control/start.ps1` in PowerShell, then open
http://127.0.0.1:8830. It uses this repository's Isaac Python environment.
For first-time dependencies run that Python with `-m pip install -r
simulation/isaac/mission_control/requirements.txt`. The launcher installs the
locked browser dependencies and builds the frontend. No cloud service is used.

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
choice registered in `mission_policy_registry.json`. This explicitly
experimental choice follows completed checkpoint saves in the designated run;
it does not qualify them or alter archived recordings. Each new mission logs
the exact checkpoint file and hash. The initial saved model has only easy-stage
training; the initial full adverse evaluation had 0/2,048 successes.
Old policies do not observe waypoint targets and are rejected for such
requests. The wider recovery policy is still training; inverted starts and
long routes are not qualified capabilities. Training status shows current
stage and independent full-task evaluation separately. Simulation launch is
disabled while the trainer owns Isaac; editing plans and replay remain usable.

To test interactively before a long training run finishes, create a file named
`STOP` inside its run directory. The trainer finishes its current rollout or
evaluation, saves `ppo_final.pt`, and releases Isaac. Wait for the training
banner to clear before launching a mission. The browser's **STOP RUN** button
only stops a mission, not PPO training. The active run and exact launch options
are recorded in `runs/ppo_8s_waypoint_recovery/*/args.json`.

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
and ground overlays are hidden in this view. The body CAD is reduced from 691,680 to
60,000 triangles for rendering. Fin meshes are the existing simplified Isaac
geometry. Camera images use Three.js/WebGL; they are **not Isaac RTX renders**.
The ground presentation and thrust arrow are visual overlays. Physics, contact,
fin articulation and trajectory all come from Isaac, not browser animation rules.

**Export Video** records all four cameras plus telemetry into a local WebM file.
Keep the tab visible while it records. Each run stores request, source hashes,
checkpoint hash, resolved physics parameters, frames, outcome and video in
`simulation/isaac/runs/mission_control/<mission id>/`. JSON telemetry is also
downloadable. `landing.webm` is a replay visualization, not camera sensor data.

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
