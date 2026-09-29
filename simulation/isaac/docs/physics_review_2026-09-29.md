# End-to-end physics review and flight-computer cost (2026-09-29)

Scope: every force, torque and state update the Isaac environment applies
(`tvc_env/dynamics`, `tvc_env/sim`, `envs/direct_rl_env.py`), the parameter
files the mission plant flies (`configs/env/mission_plant.yaml`, vehicle,
EDF, battery, servo), and the onboard stack that would run on the vehicle's
companion computer (convex guidance, tracking adapter, attitude LQR, IMU and
navigation filter). Earlier audits (2026-09-10, 2026-09-27) and the
conservation tests (Isaac test 18, gyro midpoint) were taken as the baseline
and not repeated.

## Physics findings

| # | Finding | Effect | Change |
| --- | --- | --- | --- |
| 1 | **No intake momentum ("ram") drag.** Thrust is `k_T omega^2` at any airspeed, and the only aerodynamic force on a translating vehicle was body form drag. The duct swallows `mdot` kg/s of air that arrives at the vehicle's air-relative velocity and is turned onto the duct axis, which costs `-mdot v_rel`. | At hover on the planned 8S vehicle `mdot` is ~0.30 kg/s: 0.30 N per m/s crosswise, ~10x the body's form drag at 1 m/s (they are equal near 12 m/s). In the 4 m/s steady wind of the gust missions that is 1.2 N the plant never saw, applied at the intake lip 0.10 m above the COM: 0.12 N m at 4 m/s, about a third of the ~0.35 N m the momentum vanes make at hover rotor speed. | `propulsion_edf.inlet_momentum_drag`; `dynamics.inlet_momentum_drag` (enabled in the mission plant, inlet at FRD z = -0.11 m, the top of the USD body mesh). Mass flow is the coupled jet's own (`T/u`). The force uses inlet velocity relative to the wind, including `w x (inlet - COM)`, so gusts load the intake as well as the body. Its power `-mdot v_rel^2` is never positive. |
| 2 | **Body drag area depended on whether wind was enabled.** Still air used the vehicle's `reference_area` 0.011 m² (the end-on, axial area); any disturbance file replaced it with an isotropic 0.02 m². The side-on area of the 0.35 x 0.12 m body is 0.042 m². | Crosswise form drag was 3.8x (calm) or 2.1x (wind) too low, and switching wind on changed the airframe. | `WindModel` splits drag into axial (`cd_body`, 0.011 m²) and crossflow (`cd_lateral`, `lateral_reference_area` 0.042 m²) components, each quadratic in its own body-frame velocity. Both calm and windy runs take it from `configs/vehicle` (`body`); the disturbance files' `body_drag` blocks are gone. The legacy isotropic path remains for callers without a vehicle. |
| 3 | **PhysX angular-speed cap defaulted to 100 deg/s.** `SceneConfig.max_angular_velocity_deg_s` feeds the rigid-body `max_angular_velocity` of every link. Only the mission plant and the 8S training configs raised it (3600 deg/s). | Any other config (single-env debug, HIL validation, the legacy training configs) silently clipped body rates at 100 deg/s and removed the energy above it; a warm-rotor spawn alone hands the body ~680 deg/s of yaw. | Default raised to 3600 deg/s (10x any physical rate the vehicle reaches). |

### Checked and consistent (no change)

* **Coupled jet vanes are passive.** Each vane removes a fraction `turn` in [0, 1) of the normal velocity and `drag` in [0, 1) of the along-chord velocity, so `|v_out| <= |v_in|` in the vane frame; the reported dissipation is never negative and the vanes cannot add jet energy.
* **Swirl shares shaft power** (`P = T u / 2 + Q^2 u / (2 T <r^2>)` at fixed power) and the body reaction is `-f Q` (motor reaction `-Q`, stator recovery `(1-f) Q`), equal and opposite to the swirl's angular-momentum flux. The motor's aerodynamic drag `shaft_power_at_max / omega_max (omega/omega_max)^2` and the jet's `residual_q` use the same 3072 W, so the torque-limited spool and the swirl reaction agree.
* **Rotor momentum exchange.** Spool reaction `-I_r domega/dt` uses the same `omega` step the battery energy cap bounds; gyroscopic coupling is the coupled implicit midpoint + Cayley orientation (energy- and momentum-conserving in the free-flight split, Isaac test 18).
* **Battery energy cap**: `omega_next^2 <= omega^2 + 2 dt (P_shaft - P_aero) / I_r`, no regeneration; SOC by coulomb counting.
* **Servo** first-order lag is explicit Euler at `dt / tau = 0.17` (stable, monotone); rate and position limits and the deadband act on the error, not the state.
* **Body rotational drag** (`cylinder_rotational_drag`) and the new translational drag and intake drag are all dissipative for any attitude (unit tests).
* **Wrench bookkeeping.** Fin forces at moving COPs, EDF thrust on the thrust line at the Body origin, torques moved to the link COM once; the coupled predictor now also carries the intake force and moment.

### Noted, not changed

* **IMU gravity.** The IMU chain adds `STANDARD_GRAVITY` (9.80665) to the kinematic acceleration while PhysX uses 9.81. The chain and the navigation filter both use 9.80665, so navigation is exact; only the absolute specific force reads 0.34 mg low (far below every modelled bias).
* **Gusts are horizontal steps** (0 to 5 m/s in one physics step, uniform azimuth). No vertical gusts or downdrafts reach Isaac; the 2026-09-28 hard-landing analysis needed 1.5 m/s² downdrafts offline.
* **No ground effect or jet impingement** below ~2 duct diameters, where the landing gate sits (0.8 m); the exhaust hits the pad in the terminal descent.
* **Geometry estimates.** The coupled jet's `exit_position_frd` z = 0.05 m while the vane COPs sit at the exit plane z = 0.10 m (it only sets the `w x r` inflow at the vanes). The USD body mesh is ~0.22 m across over its upper 0.16 m, against the configured 0.12 m diameter; the side-on area may be ~1.5x larger than the derived 0.042 m².
* **Axial intake drag is first-order.** `-mdot v_axial` is the leading thrust lapse in climb (gain in descent); the fan's own pressure-rise curve with inflow is not modelled and needs a thrust stand at airspeed.
* **Test interpreter.** The documented unit-test interpreter (Python 3.12) has no `clarabel`, so `test_convex_guidance.py` (44 tests) is skipped there; run it with `env_isaaclab`.

## Flight-computer cost (Raspberry Pi 5 / Jetson Orin Nano)

Timed on a Ryzen 9 9900X, one thread unless stated. The Pi 5 (4x Cortex-A76,
2.4 GHz) and Orin Nano (6x Cortex-A78AE, 1.5 GHz) CPUs are an estimated 4-5x
slower single-threaded; the Orin's GPU does not help a single-vehicle SOCP.

| Work | Desktop | Estimated Pi 5 / Orin Nano |
| --- | --- | --- |
| Tracking + attitude LQR, per 30 Hz step | 0.7 ms | ~3 ms (10% of the period) |
| Landing plan (15 SOCPs, 24 nodes) | 48 ms mean, 74 ms max | 0.2-0.4 s |
| 4-waypoint route plan (24 SOCPs, 112 nodes) | 0.52 s (4 workers), 0.69 s (1) | 2-3.5 s |
| Split of a route plan | Clarabel 40%, Python problem building 60% (`_Cones.assemble` 17%, `lil_matrix` cost updates ~10%) | |

**The blocker is architectural:** `ConvexGuidanceController.compute_action`
calls the planner synchronously every `replan_period_s` (0.5 s). On the
flight computer a route re-plan would stall the 30 Hz loop for seconds.

### Changes in this review

* `guidance.plan_latency_s` (default 0: unchanged behaviour): a re-plan
  requested at `t` replaces the tracked plan only at `t + latency`, its
  clock starting at `t` (the state it was solved from); the vehicle keeps
  tracking the old plan meanwhile. The first plan (solved on the pad) and
  HOLD re-plans stay immediate. This is the deployment model of an
  asynchronous planner, deterministic in simulation. Offline replica
  (momentum vanes, servo + joint lag, rotor gyro), identical landings:

  | Start | 0 s | 0.25 s | 0.5 s | 1.0 s |
  | --- | --- | --- | --- | --- |
  | (2, -1.5, 10) m | 0.159 m/s, 0.012 m | 0.159, 0.012 | 0.158, 0.012 | 0.155, 0.011 |
  | (8, 5, 18) m | 0.156 m/s, 0.014 m | 0.155, 0.017 | 0.155, 0.020 | 0.153, 0.036 |

  (impact speed, pad distance). Also a mission-control parameter (Re-planning).
* `guidance.async_replan` (flight computer; default off, simulation keeps
  the deterministic model above): re-plans are solved in the background and
  taken over when done, no earlier than `plan_latency_s`. Control-step time
  over five offline landings (Ryzen 9 9900X, replica paced near real time,
  pad plan excluded):

  | Mode | Worst step | p99 step | Landing |
  | --- | --- | --- | --- |
  | `thread` (GIL switch interval lowered to 1 ms) | 35-39 ms | 6 ms | identical |
  | `process` (own interpreter and GIL) | 1.0-1.4 ms | 0.8-1.1 ms | identical |

  A thread shares the GIL with the solve's Python problem building: at the
  default 5 ms switch interval with 4 solver workers a step waited up to
  34 ms, a whole 30 Hz period; lowering the interval fixes the typical case
  but C-level work still holds the GIL for ~35 ms at times. Use `process` on
  the Pi and call `planner_ready()` on the pad (the process takes ~1 s to
  start). `close()` stops it.
* `tools/bench_flight_computer.py`: run on the target to time the tracking
  step, landing re-plans and a route plan, and to get a `plan_latency_s` to
  validate in Isaac (route p95 x 1.25, rounded up to a control period).
  Desktop: tracking 0.39 ms median, landing re-plans 37 ms median / 72 ms
  p95, route plan 0.48 s, suggested 0.63 s.
* Cost matrix accumulated as COO triplets instead of `lil_matrix` item
  updates; constraint rows assembled with list extends. Plans are
  bit-identical to before (cost, every node) for landing, route and
  weighted-route problems; ~10% less planning time.
* LQR gain schedule: one lerp of the whole table per call instead of an
  `np.interp` per gain entry (called several times per control step);
  identical to 4e-18.

### Recommended pathways, in order

1. **Run the planner beside the control loop** (done: `async_replan:
   process`). Measure on the target with `tools/bench_flight_computer.py`,
   validate in Isaac with the suggested `plan_latency_s`, then fly with the
   planner process pinned to its own cores.
2. **Build each problem once, update values.** Durations change only the
   values of `A`, `b`, `q` and `P`, never the sparsity, across the ~20 solves
   of one line search. Building the sparsity once per segment structure and
   refreshing values (Clarabel's data-update API) removes most of the 60%
   Python share; a fixed-structure landing leg can also be code-generated.
3. **Fewer SOCPs per plan.** With a duration hint the line search solves 3
   samples plus 12 golden-section probes; 3-4 golden iterations from a
   hinted bracket would roughly halve it.
4. **Leave `solver_workers` below the core count** (2-3 on the Pi 5) and give
   the control loop its own core with real-time priority.
5. **Torch-free flight path.** The adapter uses torch only for the
   observation, one rotation matrix, the mixer and the action tensor; numpy
   equivalents avoid shipping and importing torch on the Pi.
6. **Precompute the LQR table.** The 15-point DARE schedule is solved at
   start-up with scipy; storing it (`.npz`) removes scipy from the boot path.
7. **Navigation.** `nav_ekf.py` is a batched torch stand-in for ArduPilot
   EKF3; on hardware the flight controller's own EKF3 should provide state,
   not a port of this filter.

## Isaac missions on the corrected plant

Two recorded missions re-flown from their saved requests (convex, WTGAHRS1
sensor noise, 4 m/s steady wind at 45 deg plus 3 m/s, 0.8 s gusts every 5-12 s):

| Mission | Before (recorded) | Corrected plant |
| --- | --- | --- |
| Landing challenge - high-energy capture (`a93eb786cb50`) | LANDED 41.3 s, 0.129 m/s, 0.036 m from pad, 23.5 Wh | LANDED 44.5 s, 0.142 m/s, 0.204 m from pad, 25.1 Wh |
| Hover 10 m - station keeping in gusts (`614281ca9910`) | LANDED 43.8 s, 0.154 m/s, 0.005 m | TIMEOUT at 120 s; with the relaxed hold rule below LANDED 44.6 s, 0.128 m/s, 0.116 m, 24.7 Wh |

The hover mission shows what the old plant hid. During the 20 s hold the
recorded vehicle stayed within 0.045 m and 0.064 m/s of the waypoint, with
0.4 deg of mean tilt, in a 4 m/s wind: the wind barely reached it. On the
corrected plant it leans 5.5 deg into the wind, holds within 0.40 m (p95
0.31 m, radius 1 m), and gusts push it to 0.34 m/s (p95) and 0.51 m/s (peak).
`WaypointMission.advance` resets the hover timer whenever speed exceeds
0.4 m/s, which happened on 2.5% of hover frames, about once per gust, so the
20 s hold never completed. The attitude and position loops stay stable; what
fails is the task's stillness criterion against realistic gust loading.
**Hold rule relaxed** (user decision): inside the capture radius, a speed
excursion over `task.navigation.hover_max_speed_m_s` (0.4 m/s) now pauses
the hover timer and resets it only if it lasts longer than
`hover_hold_grace_s` (1.0 s, longer than a 0.8 s gust); leaving the radius
still resets it at once, and takeoff/descent arrival is unchanged. Re-flown,
the mission's 20 s hold paused on 19 frames, never reset, and the vehicle
landed. Gust rejection in the tracking loop (a disturbance observer, or wind
feedforward from the estimated air-relative velocity) remains the way to
tighten station keeping itself. `dynamics.inlet_momentum_drag.enabled: false`
restores the previous plant for comparison.

## Validation

* Unit tests (Python 3.12): 371 passed, 7 skipped (Clarabel/pxr).
* Convex, flight-planning, mission-service and disturbance tests (`env_isaaclab`): 120 passed.
* Mission-control JavaScript tests: 46 passed.
* Planner and LQR speedups: plans and the offline closed-loop landing are
  bit-identical to the previous code at `plan_latency_s: 0`.
* Isaac missions: above.
