# Convex guidance: route following and objectives beyond energy (2026-09-28)

The SOCP planner (`tvc_env/controllers/convex_guidance.py`) minimised one
cost, electrical energy (or propulsive delta-v), subject to the route
corridor. This note adds secondary objective terms and a fly-through arrival
cone, and revises the mission profiles that follow drawn routes. The
repository defaults leave every new term off, so missions flown with the
default YAML are unchanged.

## Why the plan did not follow the drawn route

An offline sweep of the planner alone (planned 8S limits, 10 deg/s attitude
slew bound, 1 m soft corridor) found two causes:

1. **Inside the corridor the energy optimum is indifferent to position.** The
   corridor only prices distance *beyond* its half-width, so energy plans
   rode the edge: 0.98-0.99 m from the curve on every leg of a 1 m corridor.
   The tracking loop then followed a reference that was already a metre off
   the drawn route.
2. **Fly-through gates fixed the arrival velocity exactly** (leg speed along
   the route tangent). That equality decides the shape of every leg around a
   gate. A 10 m zigzag had no feasible plan at all under the slew bound
   (soft-terminal fallback), and in closed loop two of the samples added
   earlier today, 25 (weaving corridor) and 27 (descending dogleg), never got
   a strict-corridor plan with their suggested profiles: the controller held
   until the mission timed out with 0 waypoints done.

## What changed

### Secondary objective terms (`ObjectiveWeights`)

Each is priced in **seconds of hover cost**, the unit the corridor penalty
already used, so a weight reads "worth this many seconds of hovering"
under either primary objective, and the cost breakdown is reported in Wh
(energy) or m/s (delta-v).

| YAML key (`guidance`) | Term | Form |
| --- | --- | --- |
| `path_weight` | Cross-track distance from the drawn route, per m s | SOC epigraph per corridor node. First pass: distance to the leg chord; refined pass: distance to the curve's tangent line at the node's point (cross-track only, along-track timing stays free; the corridor chord still caps the leg). |
| `time_weight` | Plan duration, per s | Constant per solve, so it ranks the free-final-time line search. With a route it also tries 0.8x waypoint-leg timings and keeps the cheaper plan. |
| `smoothness_weight` | Thrust jerk, per s at the planned attitude-slew bound | Quadratic (Clarabel P matrix), `sum dt (|u_k+1 - u_k|/dt / J_ref)^2`, `J_ref = g * tilt_rate`. |
| `tilt_weight` | Horizontal thrust, per s at the planned tilt limit | Quadratic, `sum w_k (|u_xy| / g tan(max_tilt))^2`. |

The quadratic terms keep the problem convex (QP-SOCP). None involves the
thrust slack sigma, so the lossless-convexification argument is unchanged,
and every plan still reports its convexification gap (all tested plans:
< 1e-7, except 2e-2 with a zero heading tolerance, see below).

### Fly-through arrival cone

`flypass_min_speed_fraction` (default 1.0) and
`flypass_heading_tolerance_deg` (default 0) replace the exact gate velocity
with: along-route speed between that fraction of the leg speed and the leg
speed, and at most the tolerance angle off the tangent (one SOC). The
defaults reproduce the exact arrival. A zero tolerance with a lower speed
fraction uses two equalities across the tangent instead; that variant
showed a 2% relaxation gap with a path weight, so the profiles always set a
positive tolerance.

### Telemetry

`solver.cost_terms` (primary cost and each priced term),
`solver.route_deviation_m` (planned) and `cross_track_m` (the vehicle's
distance from the active leg as flown) are recorded per frame; the guidance
panel shows them.

## Evidence

### Offline closed loop (batch 1)

`fly_route.py` in the session scratchpad: the momentum-vane plant of
`tests/unit/test_convex_guidance._land_on_vanes` (rotor gyro, servo with
deadband, 25 ms vane joint, spool reaction), a simple ground contact for
take-offs, the real `WaypointMission` sequencer and `ConvexGuidanceController`,
flying each sample with its suggested profile (`profile`) and with the new
terms added on top. No wind or sensor-noise model. Cross-track is the
sequencer's own `cross_track_error_m` while airborne on a waypoint leg.

| Sample | Variant | Result | Cross-track RMS / p95 (m) | Touchdown (m/s, m) | Peak att. err. (deg) | Wh |
| --- | --- | --- | --- | --- | --- | --- |
| 05 slalom strict | profile | 4/4 | 0.305 / 0.575 | 0.155, 0.007 | 5.4 | 15.0 |
| | + path 10 | 4/4 | 0.167 / 0.389 | 0.158, 0.006 | 5.8 | 15.1 |
| | + path, cone | 4/4 | 0.162 / 0.441 | 0.157, 0.003 | 3.8 | 14.1 |
| | + path, cone, smooth 2 | 4/4 | 0.158 / 0.395 | 0.179, 0.005 | 2.9 | 14.2 |
| 06 agile transfer | profile | 3/3 | 0.220 / 0.389 | 0.154, 0.002 | 6.5 | 11.8 |
| | + path 10 | 3/3 | 0.154 / 0.228 | 0.164, 0.010 | 6.4 | 11.9 |
| | + path, cone | 3/3 | 0.140 / 0.338 | 0.152, 0.018 | 3.0 | 12.5 |
| 11 remote pad | profile | 1/1 | 0.324 / 0.470 | 0.154, 0.015 | 3.7 | 9.6 |
| | + path 10 | 1/1 | 0.135 / 0.296 | 0.149, 0.040 | 5.0 | 9.1 |
| | + path, cone, smooth 2 | 1/1 | 0.148 / 0.338 | 0.165, 0.017 | 2.7 | 9.7 |
| 16 survey box | profile | 5/5 | 0.317 / 0.616 | 0.152, 0.000 | 3.8 | 28.0 |
| | + path 10 | 5/5 | 0.055 / 0.094 | 0.153, 0.020 | 5.7 | 27.6 |
| | + path, cone, smooth 2 | 5/5 | 0.042 / 0.062 | 0.179, 0.005 | 3.4 | 26.9 |
| 19 precision compass | profile | 9/9 | 0.155 / 0.402 | 0.151, 0.000 | 3.8 | 70.4 |
| | + path 10 | 9/9 | 0.033 / 0.029 | 0.152, 0.015 | 4.0 | 69.5 |
| 22 figure eight (4 m/s) | profile | 11/11 | 0.390 / 0.738 | 0.148, 0.000 | 5.8 | 39.6 |
| | + path 10 | 11/11 | 0.385 / 0.739 | 0.159, 0.003 | 5.8 | 38.8 |
| | + path, cone | 11/11 | 0.212 / 0.438 | 0.154, 0.009 | 5.5 | 52.0 (101 s vs 77 s) |
| | + path, cone, smooth 2 | 11/11 | 0.181 / 0.442 | 0.179, 0.009 | 4.5 | 47.3 |
| 25 weaving corridor (strict) | profile | **timeout 0/11** | – | – | – | – |
| | + path 10 | **timeout 0/11** | – | – | – | – |
| | + path, cone | 11/11 | 0.163 / 0.443 | 0.153, 0.000 | 6.3 | 52.2 |
| | + path, cone, smooth 2 | 11/11 | 0.140 / 0.376 | 0.180, 0.000 | 1.5 | 53.4 |
| 27 descending dogleg (strict) | profile | **timeout 0/6** | – | – | – | – |
| | + path 10 | **timeout 0/6** | – | – | – | – |
| | + path, cone | 6/6 | 0.136 / 0.440 | 0.121, 0.000 | 3.3 | 38.2 |
| | + path, cone, smooth 2 | 6/6 | 0.103 / 0.142 | 0.125, 0.000 | 2.4 | 35.5 |

Cone = 50-100% of leg speed within 20 deg. Readings:

* **Path weight 10** cuts the cross-track RMS 1.4-6x wherever the
  exact gate velocity does not dictate the leg shape, with no loss in
  touchdown accuracy and energy within about 1% (often lower).
* **The cone** is what makes strict corridors with many gates plannable
  (25, 27), and it lets the path term work on the fast figure eight, but
  there the plan slows at every gate: +31% time and energy. The agile
  profile therefore keeps the exact arrival; the time-weighted profile uses
  a narrower cone (70%, 15 deg).
* **Smoothness 2** cuts the peak attitude error (6.3 -> 1.5 deg on 25), but
  raised touchdown speed from ~0.15 to ~0.18 m/s on every pad-centred
  landing except 27 (still under the 0.25 m/s gate). The profiles use 1.
* The mid-batch landing sink-rate envelope (another session, default 0.35)
  entered the later jobs; it acts on landing legs only.

Solve times (not real time in this harness: the plant waits for the solver)
rose 10-40% with the path term; p95 was already 1.8-2.3 s on the long
figure-eight and compass routes before any change, above the 0.4-0.5 s
re-plan period. That is a pre-existing onboard-feasibility concern for long
multi-gate routes, not introduced here.

### Offline closed loop (batch 2: revised profiles, new samples)

Same harness, every sample flown with its suggested (revised or new)
profile. Environments are **not** modelled offline, so 31, 33, 41 and 42
flew calm; their wind/noise/CoM stress needs Isaac.

| Sample | Result | Cross-track RMS / p95 / max (m) | Touchdown (m/s, m) | Peak att. err. (deg) | Vane sat. | Wh |
| --- | --- | --- | --- | --- | --- | --- |
| 22 figure eight (hop-agile) | 11/11 | 0.384 / 0.738 / 0.98 | 0.151, 0.006 | 5.8 | 1.6% | 39.7 |
| 25 weaving corridor (hop-strict) | 11/11 | 0.140 / 0.378 / 0.75 | 0.168, 0.000 | 3.0 | 0.1% | 54.4 |
| 27 descending dogleg (land-precision) | 6/6 | 0.115 / 0.291 / 0.75 | 0.124, 0.000 | 3.6 | 0% | 36.4 |
| 30 low-level square | 6/6 | 0.025 / 0.038 / 0.19 | 0.128, 0.019 | 1.7 | 0.1% | 33.0 |
| 31 35 m station (calm offline) | 6/6 | 0.022 / 0.024 / 0.34 | 0.179, 0.000 | 1.4 | 0.3% | 70.3 |
| 32 raster grid | 9/9 | 0.045 / 0.085 / 0.27 | 0.152, 0.070 | 4.5 | 0.1% | 52.5 |
| 33 micro-steps (no noise offline) | 9/9 | 0.028 / 0.047 / 0.21 | 0.125, 0.012 | 2.6 | 0.1% | 41.8 |
| 34 hairpins | 8/8 | 0.133 / 0.313 / 0.74 | 0.176, 0.031 | 6.4 | 0.5% | 31.2 |
| 35 square corners | 8/8 | 0.083 / 0.152 / 0.49 | 0.174, 0.007 | 5.5 | 0.1% | 35.3 |
| 36 terrain following | 7/7 | 0.067 / 0.151 / 0.40 | 0.179, 0.031 | 2.6 | 0.1% | 25.5 |
| 37 cross-field sprint | 5/5 | 0.137 / 0.197 / 0.98 | 0.163, 0.058 | 7.5 | 0.8% | 27.9 |
| 38 altitude sawtooth | 8/8 | 0.368 / 0.818 / 1.15 | 0.155, 0.005 | **9.2** | **12.9%** | 24.9 |
| 39 go-around | 5/5 | 0.182 / 0.331 / 0.99 | 0.166, 0.023 | 4.4 | 0.2% | 26.3 |
| 40 helix | 10/10 | 0.194 / 0.548 / 1.00 | 0.125, 0.000 | 2.6 | 0% | 46.1 |
| 41 low-altitude arrest (calm offline) | 2/2 | 0.174 / 0.296 / 0.94 | 0.155, 0.001 | 5.6 | 0% | 12.6 |
| 42 combined stress (upset only offline) | 4/4 | 1.037 / 2.971 / **3.81** | 0.153, 0.000 | 5.4 | 1.2% | 25.3 |

Every new sample completes offline in calm air; that is a floor, not a
pass. The sawtooth (agile, exact gate velocities) is the hardest actuator
case: 13% of commands saturate a vane and the attitude error peaks at 9 deg.
The upset in 42 leaves the 3 m arrest corridor by up to 3.8 m before the
route is rejoined.

Smoothness 0 / 1 / 2 (everything else as the revised profile):

| Sample | Cross-track RMS (m) | Peak att. err. (deg) | Touchdown (m/s) |
| --- | --- | --- | --- |
| 05 slalom | 0.166 / 0.159 / 0.158 | 4.6 / 2.9 / 2.9 | 0.159 / 0.174 / 0.179 |
| 11 remote pad | 0.133 / 0.134 / 0.148 | 4.3 / 2.8 / 2.7 | 0.148 / 0.155 / 0.165 |
| 16 survey box | 0.121 / 0.041 / 0.042 | 7.3 / 3.5 / 3.4 | 0.153 / 0.171 / 0.179 |

Smoothness 1 captures almost all of the attitude-error and tracking gain of
2. Touchdown speed rises monotonically with the weight (+0.01-0.025 m/s at
1), still well inside the 0.25 m/s success gate; the cause (the terminal
descent itself is not planned) is not yet diagnosed and is worth a look
before raising the weight further.

## Profiles

| Profile | Change |
| --- | --- |
| hop-gentle | + path 10, smoothness 1 |
| hop-soft | + path 10, smoothness 1, cone 50% / 20 deg |
| hop-strict | + path 10, smoothness 1, cone 50% / 20 deg (fixes sample 25) |
| hop-agile | + path 10 (exact arrival kept, see above) |
| hover-precision | + path 10, smoothness 1, cone 50% / 20 deg |
| land-precision | + path 10, smoothness 1, cone 50% / 20 deg (fixes sample 27) |
| **hop-time-weighted** (new) | agile limits + time 1.0, path 5, smoothness 1, cone 70% / 15 deg |
| **hover-attitude-reserve** (new) | hover-gust reserves + tilt 3, smoothness 2, path 5 |
| **land-routed-tracking** (new) | soft 1.5 m corridor, path 15, smoothness 1, cone 50% / 25 deg, 40 deg glide slope |

hover-gust, hover-endurance, land-soft-touchdown, land-crosswind,
land-high-altitude and land-delta-v are unchanged. Tracking gains and the
tilt-rate share are untouched in every profile.

## Limits and next steps

* All evidence is offline. Isaac runs of 05, 16, 22, 25, 27 and the new
  samples with the revised profiles are needed before relying on them.
* The tilt weight's value in gusts is argued, not measured: the offline
  plant has no wind. Compare hover-gust and hover-attitude-reserve on
  sample 31 in Isaac.
* The time weight changed little on the routes tried, because waypoint legs
  are timed from their speed caps; a proper leg-time search (e.g. a scale
  per leg) would give it more reach at higher solve cost.
* Tests: `tests/unit/test_convex_guidance.py` (path weight, arrival cone,
  exact default arrival, smoothness/tilt, time pricing, validation,
  controller telemetry), `mission_control/web/guidance.test.js` (cost
  breakdown), and the preset tests over all samples and profiles.
