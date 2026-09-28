# Mission planner challenge routes

Twelve built-in routes (18–29) extend the original seventeen samples. Find
them under **Sample flight plans**, filtered by Hover, Hops or Landings;
each new name includes **challenge**. Load a route, then use the suggested
guidance and environment buttons to reproduce its intended configuration.
Loading remains route-only, so the current controller, hardware, battery,
guidance and disturbances are not silently changed.

These are simulation stress scenarios intended to expose tracking,
actuator, energy and capture limits, not demonstrated successful flights.
They target the existing planned 8S vehicle and coupled battery model.
They change neither PPO training nor controller gains, dynamics, capture
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
