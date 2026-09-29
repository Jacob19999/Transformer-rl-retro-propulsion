# Isaac physics and landing validation — 10–11 September 2026

> September 14 correction: the original hinge interpretation in this report
> was wrong. The user confirmed radial-span hinges; the previous USD axes
> were tangential. Corrected radial hinges provide all three moment axes,
> including yaw. The historical controller results below used the old joints
> and must not be treated as validation of the corrected 8S mechanism. See
> `mission_control.md` for the correction and current evidence.

## Scope and status

Reviewed the Isaac Sim / IsaacLab environment, authored USD, EDF and vane forces,
contact state machine and PID baseline. It does not validate sim-to-real
performance.

Confirmed physics and evaluation defects have been corrected. PhysX momentum,
hover and contact checks pass. PID landing validation results are below.

**Result:** the tuned PID landed successfully in 13 of 16 holdout cases with no
crashes.

## Corrections

| Finding | Change and evidence |
| --- | --- |
| Vane force followed the deflected jet rather than its reaction | Positive joint rotation now produces side force opposite `hinge × flow`. The physical regression verifies negative FRD pitch torque for positive forward-vane deflection. |
| Mixer requested unavailable yaw torque | The radial vane geometry has roll/pitch authority only. The yaw allocation column is zero; nonzero legacy yaw coupling is rejected. Replaced a simulation test that incorrectly treated any nonzero torque norm as evidence of yaw authority. |
| Rotor momentum was suppressed | Enabled full `I_rotor * dω/dt` reaction and gyroscopic torque, previously scaled by 0 and 0.1 respectively. Cold spool-up now visibly produces the opposite body yaw response. |
| Vane drag omitted local moments | Apply downstream drag at each vane COP and raw EDF thrust at the body. Bound all vane drag contributions together so axial loss is counted once. |
| COPs stayed fixed while fins moved | Calibrate neutral metadata COPs into each fin's local coordinates at scene startup, then transform by measured link poses every substep. |
| Vehicle YAML nesting was ignored by the fin model | Accept the `vehicle.fins` schema, preventing silent fallback to default aerodynamic parameters. |
| Calm air disabled translational drag | Retain vehicle drag in calm conditions. Disabled wind fields no longer inject a configured steady wind. |
| PID mixed body vertical and world vertical | Horizontal position/velocity use a level heading frame; altitude uses world vertical velocity. Landing reference velocity is transformed consistently into the body observation. PID integration uses the configured control period. |
| Landing detector could override a crash | A crash on the last contact-dwell frame takes precedence over a new LANDED transition. |
| Bounces erased the arrival speed | Preserve the maximum downward arrival speed across contact/re-contact events. A later gentle contact can no longer turn an earlier hard impact into a soft success. |
| PID evaluation hid terminal contacts | Disable automatic reset for the landing evaluation, preserve first-impact speed, and default to the task's 0.25 m/s and 0.5 m success gates. |
| Episode costs could favor early failure | Rebalanced explicit YAML weights and added a bounded ground-relative descent tracking cost. The documented 30-second hover example costs about 152, versus a 200 crash penalty and up to 475 for a soft, centered terminal landing. This is an example budget, not a bound over all states. |
| Failed landings could harvest positive terminal reward | Add a 250 hard-landing penalty. It exceeds the maximum 225 partial terminal payout, so landings outside the speed gate have negative terminal reward. V3's reward rose to +9.92/step while success fell to 2.15%, exposing this incentive. |
| Curriculum restoration omitted reset fields | Versioned explicit height, velocity, attitude and rotor-spool stages. Restore every final spawn field for evaluation, including the cold rotor. |

Lift is resolved perpendicular to the incoming flow and drag along it; see
[NASA's aerodynamic force definitions](https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/aerodynamic-forces/).
The side-force sign additionally follows reaction to the jet's momentum change.
The current model remains a semi-empirical lift/drag approximation.

The wrench reference point was checked against installed IsaacLab sources and
the [PhysX tensor API](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.0/extensions/runtime/source/omni.physics.tensors/docs/api/python.html).
The tensor API applies forces at link transforms when no positions are supplied;
an extra COM torque correction would double-count the existing offset handling.

## Measured simulation checks

| Check | Result |
| --- | --- |
| Unit suite | 164 passed, including bounce preservation and reward budget regressions |
| USD mass/COM/inertia validator | PASS; Body 3.1 kg, four 0.001 kg fins, total 3.104 kg; diagonal Body inertia `[0.05, 0.05, 0.02]` kg m²; COM `[0,0,+0.01]` m in Isaac / `[0,0,-0.01]` in FRD |
| Level equilibrium with all rotor terms enabled | Throttle 0.933886; maximum speed 0.000592 m/s in the regression interval |
| Positive forward-vane command | Correct negative FRD pitch reaction |
| Cold rotor acceleration | Correct opposite body yaw response |
| Soft contact and environment isolation | LANDED at 0.081750 m/s first impact; three neighboring environments stayed AIRBORNE |
| Hard-drop contact pipeline | PASS (`test_13_physx_contact_pipeline`) |
| Initial PID hover gain sweep | Best tested gains `kp_att=.2`, `kd_att=.05`, `gyro_comp_rp=.06`: 0.08 m position RMS over the last five seconds, four sampled cases, no crashes |

Final contact checks are in [physics_review_contract_final.log](../logs/physics_review_contract_final.log).
Hover sweep data are in [physics_review_pid_sweep.json](../logs/physics_review_pid_sweep.json).
The lowest neutral body collision point is 0.3124945 m below the Body origin;
that offset informs the explicit descent-reward clearance calculation.

## Landing experiments

All final-task evaluations retain the 16–20 m cold-rotor spawn, ±2 m horizontal
spawn range, 30-second episode, 0.5 m pad radius and 0.25 m/s impact-speed gate.
The PID baseline explicitly uses `LandingGuidance`.

### PID holdout

The selected landing preset produced **13/16 successful landings (81.25%)**:
15 reached LANDED, two of those missed the pad, one timed out, and none crashed.
All 15 touchdown speeds were **0.161–0.202 m/s**, including earlier impacts
before any bounce. Maximum tilt was **0.131 rad (7.51 degrees)**. This is a small
nominal holdout, not a statistical reliability guarantee or disturbance test.

The preset is now selected automatically by `run_eval_pid.py --task landing`:

| Parameter | Value |
| --- | --- |
| Altitude P / I / D | 0.1 / 0.005 / 0.25 |
| Attitude P / D | 0.5 / 0.1 |
| Gyroscopic feedforward coefficient | 0.03 |
| Horizontal position / velocity gains | 0.125 / 0.3 |
| Minimum fin command floor | 0 (remove the PID's command floor; the physical servo deadband remains) |
| Descent / flare altitude / flare speed | 1.5 m/s / 2.5 m / 0.18 m/s |
| Hover throttle | Derived from the loaded vehicle, nominally 0.933886 |

Explicit CLI overrides still take precedence; hover evaluation keeps its own
defaults. Holdout data, including every initial pose/velocity and outcome:
[physics_review_pid_landing_holdout.json](../logs/physics_review_pid_landing_holdout.json).
The default-command check with seed 0 also **passed**: touchdown at 28.355 s,
0.178 m/s arrival speed and 0.208 m pad error. See
[physics_review_pid_landing_preset.log](../logs/physics_review_pid_landing_preset.log).
The earlier v1/v2 landing sweeps used the faulty bounce-speed accounting and
must not be cited as successful soft-landing evidence. The v3 sweep and holdout
use the corrected metric.

## Reproduction

Run these PowerShell commands from `C:\Transformer-rl-retro-propulsion\simulation\isaac`:

```powershell
$isaacPython = 'C:\Transformer-rl-retro-propulsion\env_isaaclab\Scripts\python.exe'
& $isaacPython -m pytest tests/unit -q
& $isaacPython tools/validate_usd_mass_props.py
& $isaacPython apps/run_single_test.py --test test_14_physics_review --headless
& $isaacPython apps/run_pid_sweep.py --task hover
& $isaacPython apps/run_eval_pid.py --task landing --seed 0 --headless
# Reproduce the 16-case PID holdout:
& $isaacPython apps/run_pid_sweep.py --task landing --seeds 16 --seed 123 --kp-att .5 --kd-att .1 --gyro-comp-rp .03 --kp-alt .1 --ki-alt .005 --kd-alt .25 --k-pos-xy .125 --k-vel-xy .3 --descent-rate 1.5 --flare-alt 2.5 --flare-descent-rate .18 --min-fin-cmd-xy 0 --output logs/pid_landing_holdout_repeat.json
```

## Limits that remain relevant to transfer

The EDF's loaded RPM, thrust curve, rotor inertia, motor lag, residual stator
torque, vane coefficients/COPs and servo response still include estimates.
Body angular damping remains the pre-existing, explicitly configured
0.27 N m s/rad simulation estimate; it has not been identified from hardware.
The force/rotation regressions establish implementation consistency with those
parameters, not their empirical accuracy. Thrust-stand, loaded servo and free-body
response measurements are needed before treating the gains as
hardware-ready. Yaw remains unactuated by the current radial vane layout.

The user's pre-existing `apps/isaac_launcher.py` changes were preserved.
