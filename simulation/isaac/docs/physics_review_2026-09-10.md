# Isaac physics and landing validation — 10–11 September 2026

> September 14 correction: the original hinge interpretation in this report
> was wrong. The user confirmed radial-span hinges; the previous USD axes
> were tangential. Corrected radial hinges provide all three moment axes,
> including yaw. The historical controller results below used the old joints
> and must not be treated as validation of the corrected 8S mechanism. See
> `mission_control.md` for the correction and current evidence.

## Scope and status

Reviewed the Isaac Sim / IsaacLab environment, authored USD, EDF and vane forces,
contact state machine, PID baseline, and Isaac feed-forward PPO trainer. This
runner uses an MLP actor/critic; it does not validate the project's separate
GTrXL-PPO implementation or sim-to-real performance.

Confirmed physics and evaluation defects have been corrected. PhysX momentum,
hover and contact checks pass. PID and PPO landing validation results are below;
a physics regression pass alone is not evidence of policy convergence.

**Result:** the tuned PID landed successfully in 13 of 16 holdout cases with no
crashes. PPO learned the easiest contact stage but did not solve the full landing
task in the authorized GPU budget. The final critic-loss adjustment has unit and
short runtime validation only; a convergence run with that adjustment remains.

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
| PPO evaluation mislabeled body speed as world descent | Record and use pre-reset world velocity. Count terminal failures and timeouts in evaluation/curriculum denominators. |
| PPO actor updates were suppressed by critic errors | Clip the independent actor and critic networks separately. July's failed run had critic losses around 2900–4300. |
| Critic clipping reused policy ratio units | Default to unclipped value regression; optional `--value-clip-range` is expressed in return units. A gradient regression reproduces how the old 0.2 limit creates flat gradients toward a 475-unit return. Record explained variance. This change was made after the long run. |
| PPO entropy did not describe bounded actions | Use the entropy of the tanh-transformed distribution and its stable Jacobian. Store sampled latent actions so saturated actions retain exact likelihoods during PPO updates. |
| Actor throttle prior ignored vane drag | Derive equilibrium from all USD link masses and net thrust. Nominal level hover requires 0.933886 throttle, versus the old 0.78 prior. |
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

Value clipping is optional and depends on reward scale; see the
[Stable-Baselines3 PPO documentation](https://stable-baselines3.readthedocs.io/en/v2.5.0/modules/ppo.html).
The policy ratio clip remains 0.2. The critic change does not rescale rewards.

## Measured simulation checks

| Check | Result |
| --- | --- |
| Unit suite | 164 passed, including bounce preservation, transformed likelihoods, reward budget and critic-gradient regressions |
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

## Landing and PPO experiments

All final-task evaluations retain the 16–20 m cold-rotor spawn, ±2 m horizontal
spawn range, 30-second episode, 0.5 m pad radius and 0.25 m/s impact-speed gate.
No behavior cloning, residual PID, or scripted guidance is enabled in the PPO
experiments. The PID baseline explicitly uses `LandingGuidance`.

The July reference run had zero successful landings and 100% crashes. The first
corrected-physics run completed 2,015,232 steps in about 18.5 minutes but remained
at curriculum stage 0 and failed its full-task evaluation. Its early few training
landings disappeared. That evidence motivated the transformed entropy correction
and explicit descent tracking in the second run.

First corrected run:
[eval_log.jsonl](../runs/physics_review/ppo_landing_seed0_20260910_162004/eval_log.jsonl),
[train_log.jsonl](../runs/physics_review/ppo_landing_seed0_20260910_162004/train_log.jsonl).
Its old descent telemetry used body-z speed; do not compare that field directly
with the corrected world-frame metric.

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

### PPO result and remaining work

The final long run completed **2,310,144 steps**. At step **442,368**, it crossed
the 60% rolling success gate for the 0.34–0.50 m initial stage and advanced to
the 0.7–1.2 m stage. It did not master that stage: its final rolling success was
0.1%. Full 16–20 m evaluations recorded **0% successful landings** and a final
100% terminal-failure fraction. The failure fraction includes physical crashes
and other terminal failures such as exceeding the altitude limit.

Final long-run artifacts:
[eval_final.json](../runs/physics_review_final/ppo_landing_seed0_20260910_214736/eval_final.json),
[train_log.jsonl](../runs/physics_review_final/ppo_landing_seed0_20260910_214736/train_log.jsonl),
[ppo_final.pt](../runs/physics_review_final/ppo_landing_seed0_20260910_214736/ppo_final.pt).
This checkpoint **is not a successful landing policy** and still reflects the
old 0.2 critic value clipping. The run directory retains the actual task and
physical-parameter snapshots used in that experiment.

After inspecting the critic losses, value clipping was decoupled from policy
clipping and disabled by default. A **65,536-step runtime check** completed with
finite losses, a 34.62% rolling success rate on the easiest training stage, and
the new explained-variance telemetry. Its two-second evaluation was only a
runtime check, not a landing evaluation. No full convergence claim follows from
it. See [physics_review_ppo_value_fix_smoke.log](../logs/physics_review_ppo_value_fix_smoke.log).

Logged PPO execution across the diagnostic and validation runs totals about
**56.3 minutes**, excluding brief Isaac startup overhead. At the end of the original one-hour pass, no training jobs remained
running. The user subsequently removed the GPU limit; see the follow-up log
[ppo_convergence_2026-09-11.md](ppo_convergence_2026-09-11.md). The remaining research work is a longer convergence run with the final
critic loss, assessing stage retention and full-task success rather than reward
or loss alone. Observation normalization and finer curriculum transitions are
additional hypotheses to evaluate if the next run again stalls; neither has
been silently introduced into the reported experiments.

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
# New training run, using the final implementation (the user subsequently authorized unrestricted local GPU time):
& $isaacPython apps/run_train_ppo.py --task landing --total-steps 2300000 --eval-interval 500000 --save-interval 250000 --headless --max-wall-time 1480 --output-dir runs/physics_review_followup
```

## Limits that remain relevant to transfer

The EDF's loaded RPM, thrust curve, rotor inertia, motor lag, residual stator
torque, vane coefficients/COPs and servo response still include estimates.
Body angular damping remains the pre-existing, explicitly configured
0.27 N m s/rad simulation estimate; it has not been identified from hardware.
The force/rotation regressions establish implementation consistency with those
parameters, not their empirical accuracy. Thrust-stand, loaded servo and free-body
response measurements are needed before treating the gains or learned policy as
hardware-ready. Yaw remains unactuated by the current radial vane layout.

The user's pre-existing `apps/isaac_launcher.py` changes were preserved.
