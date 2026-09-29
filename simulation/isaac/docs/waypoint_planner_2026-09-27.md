# Waypoint progression and planner workspace — 2026-09-27

Mission `8243dfb69c0d` timed out after 120 s at its first hover waypoint.
At 30 s, tracking error was only 0.084 m, but the reference remained over
2 m above the requested hover. Every replan budgeted the entire original
incoming corridor arc, putting arrival roughly 36 s into the future again.
The hold timer correctly remained zero outside the capture sphere.

`ConvexGuidance` now times each leg from the remaining arc plus the distance
needed to rejoin it, bounded below by direct target distance. Landing horizon
search uses the same calculation. The complete corridor geometry remains in
the constraint checks. Capture radii, continuous dwell, speed gates,
physical dynamics and controller gains are unchanged.

## Simulation checks

Both missions used the Isaac plant with wind and sensor noise enabled.

| Recording | Check | Result | Duration including shutdown | Impact | Final pad error |
| --- | --- | --- | --- | --- | --- |
| `c9278243dfb6` | Exact request from `8243dfb69c0d`, including delta-v objective, long fast approach and 10 s hover | LANDED / success; settled | 39.25 s | 0.132 m/s | 0.097 m |
| `c92720910ef9` | Cold-rotor takeoff → hover → fly-through → hover → descent → landing, energy objective | 5/5 airborne waypoints; LANDED / success; settled | 44.55 s | 0.203 m/s | 0.112 m |

The previously stuck hover completed at 29.49 s. The full sequence advanced
at 9.40, 11.43, 20.13, 28.16 and 33.92 s. These checks validate simulation
progression, not hardware readiness: the original aggressive request still
has a large initial yaw transient (~606°/s), as did its pre-fix recording.
Recordings remain in `runs/mission_control/` and appear in the mission archive.

## Interface

- The 3D scene and selectable flight sequence share a workspace. Step cards
  expose arrival conditions beside their editable fields. Add, reorder,
  remove, pads, full-plan save/load and JSON exchange remain available.
- Camera controls provide fit, top, side and fullscreen views. Scene labels
  fit their text; pad labels sit below their markers.
- Disturbances have their own section with keyboard-accessible source tabs,
  explicit Enabled/Off status, diagrams and numeric/slider controls.
- Optimizer flight settings and solver/tracking tuning are visually grouped.
  Profile management is expandable; parameter defaults, diagrams, help and
  group resets remain visible. Hidden invalid fields reveal their section.
- Convex replay capture diagnostics distinguish an out-of-radius waypoint,
  excess speed and a counting hold using recorded physical states.

## Verification

71 targeted Python tests passed (`test_convex_guidance`, `test_waypoints`,
`test_mission_service`, `test_disturbance_parameters`). Regression coverage
includes repeated real SOCP replans feeding the actual waypoint sequencer
for both energy and delta-v objectives, and retention of the full corridor.
24 browser-module tests passed; the production frontend bundle builds.

Browser checks covered selection, editing, insertion before Landing,
reordering/removal, hidden invalid-field reveal, disturbance toggles and
parameters, optimizer tabs, camera presets and fullscreen. Desktop (1440 px)
and mobile (390 px) had no horizontal overflow and no browser errors.

Screenshots: [route](planner-workspace-desktop.png),
[environment](planner-environment-desktop.png),
[guidance](planner-guidance-desktop.png),
[mobile route](planner-workspace-mobile.png),
[mobile guidance](planner-guidance-mobile.png).
