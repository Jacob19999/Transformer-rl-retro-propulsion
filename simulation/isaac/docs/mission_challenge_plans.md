# Mission planner challenge routes

Twenty-five built-in routes (18–42) extend the original seventeen samples. Find
them under **Sample flight plans**, filtered by Hover, Hops or Landings;
each new name includes **challenge**. Load a route, then use the suggested
guidance and environment buttons to reproduce its intended configuration.
Loading remains route-only, so the current controller, hardware, battery,
guidance and disturbances are not silently changed.

These are simulation stress scenarios intended to expose tracking,
actuator, energy and capture limits, not demonstrated successful flights.
They target the existing planned 8S vehicle and coupled battery model.
They change neither controller gains, dynamics, capture
rules or safety limits. Speeds are requested limits, not guaranteed achieved
speeds. Flight performance still requires Isaac runs.

| ID | Route | Main demand | Suggested profile / environment |
| --- | --- | --- | --- |
| 18 | Hover altitude ladder | Five 8 s captures at 3, 12, 4, 18 and 6 m; alternating vertical acceleration and braking | Hover gust tolerant / current environment |
| 19 | Hover precision compass | Eight 5 s holds at the compass points and centre; 0.25 m capture, strict 0.75 m corridor | Hover precision / current environment |
| 20 | Hover gust staircase | Five 10 s holds around a 12 m box at 6–14 m; simultaneous tracking and disturbance rejection | Hover gust tolerant / combined stress |
| 21 | Hover endurance relocation | Four 45 s holds, 8 m transfers and altitude changes; declining voltage and sustained power demand | Hover endurance / current environment |
| 22 | Hop fast figure eight | Nine fly-through gates across 24 × 16 m, up to 4 m/s, followed by full-stop capture | Hop agile / current environment |
| 23 | Hop reversal shuttle | Four 24 m traverses, full-stop reversals, 6–12 m altitude changes, remote landing | Hop agile / current environment |
| 24 | Hop climbing spiral | Eight rising gates to 24 m, up to 3.5 m/s; remote capture and 19 m vertical descent | Hop agile / current environment |
| 25 | Hop weaving altitude corridor | Eight gates across 40 m, alternating lateral and altitude demand, strict 1 m corridor | Hop strict / current environment |
| 26 | Landing high-energy capture | 45 m start, 8 m/s sink, 6.7 m/s lateral speed; braking gate and staged captures | Landing high altitude / current environment |
| 27 | Landing descending dogleg | Three descending gates, two stop captures, strict 0.75 m corridor, 0.12 m/s touchdown | Landing precision / current environment |
| 28 | Landing crosswind diversion | Descending transfer past home to a remote pad; lateral braking in 6 m/s crosswind and 5 m/s gusts | Landing crosswind / gusty crosswind |
| 29 | Landing upset and recovery | 20/-15 deg initial tilt, 30/-20/15 deg/s body rates, 5 m/s sink; arrest, transfer and recapture | Landing crosswind / noisy sensors |

Offline (no Isaac), routes 25 and 27 got no strict-corridor plan with their
suggested profiles as first published and held until timeout: exact
fly-through velocities made the strict tube infeasible. The revised
hop-strict and land-precision profiles (fly-through arrival cone) fly both;
see `docs/convex_multi_objective_2026-09-28.md`.

### Second set (30–42, 2026-09-28)

These target failure modes the first set does not isolate: operating near
the 0.8 m route floor, tight turns inside narrow corridors, long exposure
to wind, rest-to-rest settling, and landings that do not start with a
direct approach. Routes with fly-through turns suggest the revised
route-following profiles (see `docs/convex_multi_objective_2026-09-28.md`).

| ID | Route | Main demand | Suggested profile / environment |
| --- | --- | --- | --- |
| 30 | Hover low-level precision square | 1.3–1.5 m holds on a 3 m square, 0.2 m capture, strict 0.5 m corridor just above the route floor | Hover precision / current environment |
| 31 | Hover 35 m station in strong gusts | 75 s of holds at 35, 35 and 20 m, then a 15 m column descent through the wind | Hover attitude reserve / gusty crosswind |
| 32 | Hover 3 × 3 raster survey | Serpentine 10 × 10 m grid, eight 4 s holds alternating 7.5 and 6 m, strict 0.75 m corridor | Hover precision / current environment |
| 33 | Hover micro-step settling | Seven 0.5 m steps, each held inside 0.15 m for 6 s, with BNO085-class IMU noise | Hover precision / IMU BNO085 |
| 34 | Hop hairpin switchbacks | Three 180° hairpins 4 m apart on 12 m legs at 3 m/s, 1 m corridor, remote pad | Hop soft / current environment |
| 35 | Hop square corners | 16 m square flown through its 90° corners at 2.5 m/s, strict 0.75 m corridor | Hop strict / current environment |
| 36 | Hop low-level terrain following | 36 m at 1.4–3.2 m through gully/ridge gates, strict 0.5 m corridor, remote pad | Hop strict / current environment |
| 37 | Hop 72 m cross-field sprint | 72 m at 4 m/s through three offset gates, full stop over a remote pad | Hop time weighted / current environment |
| 38 | Hop altitude sawtooth | Three 9 m climbs and dives every 6 m of ground track (~56° flight path) at 3 m/s | Hop agile / current environment |
| 39 | Landing overshoot and go-around | Arrive at 5 m/s, pass over the pad at 12 m, 14 m circuit, re-established hover | Landing routed tracking / current environment |
| 40 | Landing tightening helix | Two turns (8 → 4.5 m radius) from 30 to 9 m at 3 m/s, 1 m corridor, column descent | Landing routed tracking / current environment |
| 41 | Landing low-altitude lateral arrest | 4.5 m up, 5 m/s sideways, sinking 0.8 m/s; arrest above the floor | Landing crosswind / light breeze |
| 42 | Landing combined stress, remote pad | 15/-10° tilt, 20/-15/10 °/s rates, 4 m/s sink at 38 m; 40 m routed transfer | Landing crosswind / combined stress |

Route 37's gates are 4 m/s, the repository-default speed cap every sample
must also launch under. A waypoint speed caps its whole leg, so the time
weight shortens braking, capture and landing rather than raising cruise.

For a calm comparison run, explicitly select the Calm environment. The
three disturbance suggestions reuse the versioned environment presets;
they apply throughout the mission, not at scripted route events. Hold
times require continuous capture under the existing sequencer rules.

## Design basis

The modeled 8S plant in `mission_control/models.py` supplies 43.07 N at
full rotor speed for 3.104 kg: nominal T/W is about 1.41. The agile profile
plans at 20 degrees of tilt and retains 8 degrees of tracking margin,
without increasing the identified tilt-rate authority share. The new hop
routes use up to 4 m/s, within both the default 4 m/s and agile 5 m/s caps.
Longer legs, reversals, climbs and settling requirements provide the
challenge rather than relaxing the vehicle model or guidance bounds.

As a scale check, using half the nominal excess thrust gives about
2.03 m/s² vertical braking acceleration. Arresting an 8 m/s sink then
needs about 15.8 m even before rotor slew, tilt, voltage sag or tracking
error. Route 26 starts at 45 m with its rotor at 0.84 of full speed and
uses a soft corridor so initial braking can exceed the corridor when
necessary. This arithmetic motivates a high-altitude stress case; it is
not a feasibility certificate. The upset case similarly retains 40 m
of initial altitude rather than demanding recovery just above contact.

Ground starts have zero velocity and a stopped rotor; airborne starts
use the existing 0.84 rotor-speed fraction prior. Each hop is one launch
and one final landing: the sequencer only permits Takeoff first and
Landing last, so these are not unsupported repeated touch-and-go cycles.
Every route uses at most twelve waypoints. Time limits include room for
braking, capture, dwell and terminal descent rather than using timeout as
the primary challenge. The endurance route has 180 s of main holds and
a 420 s timeout; its battery remains coupled and can end flight sooner.

## Evaluation

The existing preset and service tests validate every file's canonical
route-only format, ground clearance, waypoint and pad rules, default and
suggested guidance compatibility, environment bounds and gallery API.
They do not simulate the flight dynamics.

For flight comparisons, retain the same hardware, battery and seed and
record completed waypoints, capture/dwell progress, tracking error,
corridor excess or strict infeasibility, vane saturation, current,
state of charge, touchdown speed and pad error. Compare calm and suggested
disturbances separately; a timeout, failed capture or crash is a research
result, not a reason to silently weaken the route or retune the controller.
