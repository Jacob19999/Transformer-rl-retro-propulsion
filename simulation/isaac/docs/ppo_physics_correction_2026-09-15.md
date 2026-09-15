# PPO continuation and numerical physics corrections — 15 September 2026

## Why the previous training is not a validated landing policy

The finite-momentum run `runs/ppo_8s_momentum_2048/ppo_landing_seed0_20260914_191453`
completed 100,139,008 transitions. It reached the 6–10 m, spinning-fan stage's
70% rolling success gate at 84,410,368 transitions, but the final 16–20 m
cold-rotor evaluation remained **0% successful and 100% crashed**. That failure
is preserved in `eval_log.jsonl`; stage mastery is not full-task success.

The separate throttle-head hover-prior experiment
`runs/ppo_8s_momentum_prior/ppo_landing_seed0_20260914_192057` completed
10,223,616 transitions with zero soft successes even on its easiest stage.
Its 91.26% LANDED fraction hid a mean impact speed of 1.1955 m/s. It is a
failed initialization ablation, not a controller to deploy.

## Wrench application defects in the installed Isaac Lab checkout

The installed checkout reports VERSION 2.3.2 and git description
`v0.2.0-1318-g87608f062bb`. Its `utils/warp/kernels.py` position-aware wrench
kernel subtracts positions in world coordinates but crosses that lever arm
with a force already rotated into link coordinates. It also replaces a
supplied torque when a force application position is present. These are
observations of this checkout, not claims about every Isaac Lab release.

The local `LinkForceInterface` now explicitly computes world-frame moments
about each body's actual COM and supplies force plus torque **without** the
composer position argument. The body wrench applies thrust at the authored
body origin, preserving the thrust-line moment under COM offsets. Fin loads
act at their actual moving COP. The COM setter invalidates Isaac's lazy COM
state buffers so same-step calculations cannot use a previous reset's COM.
No vendor files were modified.

Actual Isaac regressions:

- Test 16: 48 N thrust with ±10 mm COM offset gives expected ±0.040 rad/s
  pitch impulse at 240 Hz; observed ±0.039898 rad/s.
- Test 17: identical fin commands at four world headings give the same body
  rates within 1e-7 rad/s. A simultaneous application-point force preserves
  the cold-spool yaw impulse, observed −0.645620 rad/s in one physics step.

Logs: `runs/mission_control/thrust_com_physics_test_v3.log` and
`runs/mission_control/wrench_covariance_test.log`.

## Rotor gyro integration

Applying `H × omega_body` explicitly at the old angular velocity is an
unstable forward-Euler step: the torque is perpendicular to the old velocity
but still adds discrete rotational energy. With the configured rotor inertia
and 90% RPM, Isaac test 18 measured **10.2035 times** the initial rotational
energy after two seconds at 240 Hz with no applied angular work.

`compute_midpoint_gyroscopic_torque` now solves
`(I - dt*[H]x/2) omega_mid = I*omega_old` and applies `[H]x*omega_mid`.
Here `I` comes from the actual PhysX body inertia. This Cayley/midpoint update
conserves kinetic energy for the isolated gyro operator, at the unchanged
rotor inertia and full gyro strength. It introduces neither a tuned damping
term nor reduced gyro torque. PhysX separately integrates the locked-body
Euler term; coupled-torque splitting error remains subject to refinement.

The original unforced Isaac case now ends at **0.99670–0.99675** of initial
rotational energy. Tiny aerodynamic damping and articulation/splitting errors
remain. Baseline and corrected results are recorded in
`runs/mission_control/gyro_conservation_explicit_baseline.json` and
`runs/mission_control/gyro_conservation_midpoint.log`.
Higher-rate and spinning-body stress checks are recorded separately.

After this correction, 217 unit/pure-tensor tests passed. Isaac test 15 also
passed the integrated battery/jet/gyro test, moving-fin flow test, and real
soft contact at 0.122625 m/s with 0.125 s dwell. Its log is
`runs/mission_control/momentum_midpoint_regression.log`.

## Active training definition

`configs/env/train_2048_8s_wrench_v2.yaml` introduces explicit rotor-state
curriculum stages: 6–10 m at 84% RPM, then the full 16–20 m spawn at 84%,
55%, 25%, and finally zero RPM. Earlier stages need 70% rolling soft, on-pad
success. Full evaluation always uses the cold-rotor task.

The corrected-wrench actor/critic reached 4,194,304 transitions and was
gracefully stopped for the gyro correction. That checkpoint resumed in
`runs/ppo_8s_gyro_midpoint/ppo_landing_seed0_20260915_030057`, targeting
120 million total transitions. The explicit `physics_correction.json`
records changed source hashes and the numerical diagnosis. Each new run
contains its configuration, source manifest and `source_snapshot.zip`.

This is direct-action, feedforward PPO (two 256-unit layers), not yet the
project's proposed GTrXL recurrent policy. All four fin targets and throttle
are learned. There is no PID wrapper, thrust floor, scripted braking burn,
or prescribed descent speed. Terminal soft-landing rewards dominate the
documented episode-cost envelope; energy, impulse and time costs discourage
hovering. The objective weights and their arithmetic are described in
`momentum_landing_2026-09-14.md`.

Neither a curriculum score nor training reward qualifies a checkpoint.
Acceptance requires independent full-task trials, combined disturbances,
battery SOC and residual-swirl sensitivity, timestep refinement, and a
mission replay that remains settled after touchdown. Report success and
crash rates alongside successful-episode Wh and propulsive impulse/mass.

The planned 8S hardware is still unpurchased/unspecified. Rotor inertia,
fan maps, residual swirl, airframe inertia and pack impedance remain estimates;
these numerical fixes do not turn the reduced-order model into calibrated
hardware or CFD.
