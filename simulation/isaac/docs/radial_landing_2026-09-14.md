# Radial hinge and 8S landing work — September 14, 2026

The user confirmed that each fin hinges along its radial span. The previous
USD joints were perpendicular to that span, so the bottom view correctly
exposed an error in the simulated mechanism. The authored joints and force
bases now agree with the radial mechanism. Neutral CAD geometry is preserved.

## Physical and visual evidence

- The USD/mesh test checks that each hinge is parallel to span and perpendicular
  to the vane normal. The four axes agree with the force metadata.
- Isaac test 14 passed net-lift equilibrium, radial fin torque direction,
  opposite body reaction during rotor acceleration, and isolated soft contact.
- Fresh recording `runs/mission_control/8b0000000001` contains 451 actual
  articulation samples. Independent quaternion comparison gives maximum
  radial-axis error below 0.000034° across all four fins. At −5.645° FWD
  deflection, a 78 mm chord vector sweeps 7.67 mm sideways and less than
  one nanometre along the radial axis (numerical residual).
- The camera uses those measured rigid-link poses. FWD is up and RIGHT is on
  the left when looking up from below. Measured angles sit beside each fin.
  Old recordings retain their old poses and have a camera-local warning.
- The fresh four-camera WebM decodes completely with no errors. This particular
  PID run timed out and is evidence of articulation, not landing convergence.

Wind/gusts, noise and center-of-mass offset can be combined. Composition retains
each standalone source's strength; in particular, adding wind cannot shrink the
COM offset from 10 mm to the wind file's unrelated 5 mm default.

## PPO experiment definition

`configs/env/train_512_8s_radial.yaml` specifies the 8S plant, 28 observations,
four feedback-driven spawn stages and physical efficiency costs. PPO directly
learns four fin angles plus throttle. The battery adds SOC, normalized loaded
voltage, current and polarization voltage to the original 24 observations.
Electrical work and the integral of propulsive acceleration are measured at
each physics substep; success still requires settled contact, impact ≤0.25 m/s
and pad error ≤0.5 m.

The first radial experiment initialized the previous 8S actor, permuting only
its initial fin output weights for the new physical basis, and started a fresh
critic/optimizer. It mastered the first stage but had only 33.1% recent stage-1
success by 6.029M transitions. At 4.06M the fresh critic's output was bounded
to ±20.8 approximately by its tanh features and output weights, far below
the +475 ideal terminal return; explained variance was near zero.

Continuation from `ppo_step_6029312.pt` retained actor, critic and Adam moments,
with actor LR 3e-5 and critic LR 1e-3. The former single-group optimizer migration
is tested for exact next-update equivalence when rates match. The controller,
reward scale, physical parameters and landing gates were unchanged. The 2–4 m
stage passed its 70% advancement threshold near 8.3M; the 6–10 m stage passed
near 11.4M. The final stage uses the full 16–20 m cold-rotor spawn distribution.

Training-window success is not independent validation. Full-task evaluations
and held-out batch results are retained separately. Efficiency comparisons
must report success/crash rates alongside Wh and propulsive delta-v, and use
successful episodes rather than treating low-energy crashes as efficient.

## Calibration limits

The 8S hardware is a plan, not a confirmed purchased assembly. EDF thrust/power,
pack resistance/RC/thermal parameters, rotor inertia, vane coefficients and
the inherited 0.27 N m s/rad linear body damping remain uncalibrated estimates.
Mass and inertia come from the authored USD rather than a new physical weighing
or inertia test. The simulation and training results do not establish hardware
flight performance. Manufacturer-supported candidate specifications and the
remaining gaps are listed in `mission_control.md` and `mission_control/hardware.json`.
