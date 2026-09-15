# PPO landing: momentum-model revision, 14 September 2026

**15 September update:** the 100.139M run failed the full cold-rotor task.
Subsequent force-frame and gyro-integration defects were found and corrected.
See [the continuation record](ppo_physics_correction_2026-09-15.md) for the
diagnosis, actual Isaac regressions and replacement training run. Earlier
scores below do not validate the corrected plant.

The objective is reliable soft touchdown followed by lower pack energy and
propulsive impulse per mass. A late braking burn is a behavior to evaluate;
the reward does not prescribe a Falcon 9 trajectory. Unlike a rocket with a
minimum stable engine throttle, this EDF can throttle below hover, but loses
vane authority as its jet weakens. Electrical energy and impulse are also
different objectives because EDF shaft power scales approximately with RPM³.

## Baseline retained

`runs/ppo_8s_radial_critic/ppo_landing_seed0_20260914_093426/ppo_final.pt`
completed 30,015,488 transitions. Its final stochastic 512-environment
evaluation reported 95.703% success, 0.391% crashes, zero timeouts, mean landed
impact 0.1746 m/s and mean landed pad error 0.1313 m. Successful episodes used
3.4871 Wh and 46.985 m/s of integrated propulsive acceleration over 4.6948 s.
These are evaluations under the **previous** plant, not validation of this
revision or physical hardware. The actor is an initialization prior only.

## Physics diagnosis and replacement

The old plant used a 0.27 N m s/rad linear angular damper tuned in simulation.
It also applied `q*S*CNa*alpha` independently to each vane, without accounting
for their shared finite jet mass flow. At 128 m/s, 0.002 m² area and 15°,
this could exceed 15 N of sideforce per vane, although a quarter of the
48 N jet carries only 12 N of axial momentum. Local vane/body motion and
residual fan swirl were absent from incidence.

`tvc_env/dynamics/coupled_jet.py` implements an explicit reduced-order model:

- Four equal streamtubes share `mdot = T/U`. Their representative radii are
  the measured-pose COP radii from the USD/metadata. Equal sharing is an
  approximation that needs flow measurements or CFD.
- Residual fan/stator torque is `Q = f_residual * P_shaft / omega`. The same
  `Q` generates swirl through `Q = mdot * <r²> * omega_swirl`, and its negative
  is applied to the airframe. Spool `-I*domega/dt` and rotor precession
  `-omega_body × H_rotor` remain separate, at full scale.
- The axial speed is reduced slightly so axial plus swirl kinetic power
  remains equal to the nominal wake power. Swirl does not receive free energy.
- Incoming velocity includes the translating/rotating nozzle and subtracts
  each actual moving COP velocity. COM velocities use COM lever arms. This
  is a convected frozen-wake approximation, not a resolved wake simulation.
- For each actual vane plane, normal velocity is attenuated by
  `1-exp(-q*S*CNa/(mdot_fin*U))`. It matches the old small-loading slope, but
  approaches alignment instead of reversing all incoming transverse flow.
  Profile drag attenuates streamwise velocity separately. Incidence-induced
  drag already follows from momentum turning and is not added a second time.
- Force on the vane is `mdot_fin*(v_in-v_out)`. The model is passive in the
  moving vane frame. Loss and moments are applied once, at the actual COP.
- The tuned angular damper is replaced by a still-air cylindrical crossflow
  integral, proportional to `rho*Cd*D*L^4/64 * |omega_xy|*omega_xy`. This is
  small. Axial skin-friction torque is omitted pending calibration. Vane-flow
  damping comes from local relative velocity, not a tuned body torque.
- PhysX rigid-body gyroscopic terms are explicitly enabled. These integrate
  locked-body inertia; the extra spinning rotor momentum is not in the USD
  and therefore still needs the separate rotor torque.
- Physics runs at 240 Hz, policy at 30 Hz, contact dwell 30 physics frames
  (the same 0.125 s used previously). The angular velocity limit is explicit
  in the installed Isaac Lab schema's degrees/second units.

The flow-velocity/angle and torque bookkeeping follows the control-volume
principles in [Drela's QPROP formulation](https://web.mit.edu/drela/Public/web/qprop/qprop_theory.pdf).
This implementation is **not** QPROP, a blade-element fan solution, or CFD.

The residual stator fraction 0.1, outlet position, rotor inertia, vane
coefficients, pack resistance and authored body inertia remain estimates.
Tests at residual fractions 0 and 0.2 are required separately; these values
are sensitivity cases, not a claimed measured uncertainty interval. Ground
effect, recirculation, motor/ESC maps and detailed servo electrical loads
remain calibration/model gaps. Purchased 8S EDF/ESC/pack details are pending.

## Training definition

`configs/env/train_512_8s_momentum.yaml` versions the new task. PPO controls all
four fin targets and throttle directly; no PID, minimum-thrust floor, descent
guidance or action replacement is used. The slow descent-reference terms are
disabled. True simulated Wh and integral `|Fprop|/m dt` are charged at physics
frequency. The latter is a propulsive impulse-per-mass measure, not the
rocket equation's delta-v.

The 30 s example at 2500 W and 10 m/s² costs 41.67 energy points, 30 impulse
points and 4.5 time points. Typical attitude/position/rate costs add ~84.6;
this remains below the 200 crash penalty and 475 ideal soft landing reward.
The success gates remain impact <=0.25 m/s, pad error <=0.5 m, real contact.
Discount gamma=0.999 retains about 86% of a 5 s terminal reward. There is no
global reward scaling. A fresh critic/optimizer avoids reusing values learned
under a different plant/reward/discount. Actor LR=3e-5, critic LR=1e-3,
KL limit=0.015, entropy coefficient=0.001 are recorded in each run's args.

Feedback-driven warm-start spawn stages precede the full 16–20 m cold-rotor
task. Stage success is not full-task success. Final acceptance must use
independent seeds, explicit action mode, combined disturbances, SOC and swirl
sensitivity, with failure rates beside success-only efficiency statistics.
Compare checkpoints on shared successful trials as well as all-trial success.
Neither a single landing nor a low-energy crash establishes optimal control.

## Validation so far

210 unit/pure-tensor tests passed, including angular momentum closure, wake
energy including swirl, passive vane energy loss, zero-flow loss of authority,
radial torque directions and geometry-derived damping. Isaac test 15 passed
actual battery/jet/gyro integration, measured moving-fin COP airflow, and a
real soft contact at 0.122625 m/s with environment isolation. Log:
`runs/mission_control/momentum_physics_test.log`.

Long PPO training is in progress. No new-model landing success or optimality
claim is made until its independent evaluation is complete.
