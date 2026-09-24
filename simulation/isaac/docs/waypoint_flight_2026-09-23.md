# Waypoint flight task (2026-09-23)

This replaces the landing-first mission task (`configs/env/train_2048_8s_waypoints.yaml`).
That run was abandoned at 200M transitions, still on curriculum stage 1 of 11.

The policy now learns a single skill: reach the next waypoint efficiently.
Missions are composed from waypoints by a planner. A landing is a route that
ends in a `land` waypoint; a station hold is a route that ends in `hover`.

| File | Role |
|---|---|
| `tvc_env/envs/waypoint_flight.py` | Mission generation, capture logic, reward, observation (pure torch) |
| `configs/tasks/waypoint_flight.yaml` | Task definition: gates, reward weights, randomization, curriculum |
| `configs/env/train_waypoint_flight.yaml` | Plant, clock and 8S hardware (unchanged physics) |
| `apps/run_train_waypoints.py` | PPO trainer |
| `apps/waypoint_eval.py` | Full-task and benchmark evaluation (library and CLI) |
| `tests/unit/test_waypoint_flight.py` | Budget, generator, capture, shaping and observation tests |

## Task

- **Waypoints.** `flypass` is captured when the swept segment passes within
  its radius. `hover` needs a continuous hold inside its radius at ≤0.5 m/s
  and ≤60°/s body rate. `land` is allowed only as the final waypoint and needs
  a real LANDED contact within the pad radius at ≤0.25 m/s. Every mission ends
  in `hover` or `land`.
- **Observation (54 channels, body FRD).**
  - Vector to the active waypoint, plus the next two legs.
  - Kinds of the active and next two waypoints.
  - Active radius and remaining hold time.
  - Gravity direction, velocity, body rates, height.
  - Vane angles and rates, rotor speed, contact state.
  - Battery state, the previous action, and a fine-scale (2 m) copy of the target vector.
- **Actions.** The four vane angles and the throttle, applied directly. There
  is no mixer, reference path, speed profile or guidance.
- **Outcomes.** SUCCESS, CRASH, TILT, ALTITUDE, SPIN (>360°/s on any axis),
  GEOFENCE, PREMATURE_LANDING, BAD_LANDING. TIMEOUT is a truncation.

## Efficiency objective (energy-weighted)

Per second of level hover (~2.09 kW electrical on the 8S pack), time costs
0.25 and energy costs 1.3/Wh × 0.58 Wh/s = 0.75, so energy is **75%** of the
efficiency cost. At full power (3.5 kW), energy rises to 83% of it.

Because hover power dominates, minimum-energy and minimum-time mostly agree:
both favour flying legs briskly. They diverge on climbs and hard braking,
where throttle bursts cost roughly f³ in power.

Dense guidance is potential-based: Φ = −(remaining route length), so the
shaping is γΦ(s′) − Φ(s) with Φ = 0 at absorbing states. The
`test_potential_shaping_telescopes_to_minus_initial_potential` test checks
that its discounted sum is exactly −Φ(s₀).

**Rule-2 budget** (`reward_budget()`, enforced when training starts). The
nominal worst case is a 90 s episode at full power, with all three body axes
at their soft rate limits and a jittery action stream. Its integrated step
cost is 274, against ±400 terminals. Measured on the plant at start-up: hover
fraction 0.841, hover power 2085 W, hover cost 1.00/s, energy share 75.1%.

## Yaw: diagnosis and fix

Evidence from `runs/ppo_waypoints_staged_v5/.../curriculum_eval.jsonl`: mean
peak yaw stayed at 1625–1651°/s from 32.5M to 200M transitions. Over the same
period success rose from 9% to 36% and crashes held at 40%. Roll and pitch
peaks stayed within 90°/s.

Offline torque budget, using this repo's `CoupledJet` model:

- **Steady swirl is not the cause.** At zero vane deflection the vanes cancel
  the residual swirl torque; the net is 0.000 N·m from 0.5 to 1.0 throttle.
- **Vane authority is adequate.** Common-mode vane deflection gives ±0.30 N·m
  of yaw at hover (±0.43 N·m at full throttle).
- **The cause is angular-momentum exchange between rotor and body.**
  I_rotor·ω_max/I_zz = 2e-4 × 4650 / 0.02 = **46.5 rad/s of body yaw per unit
  of throttle change**. A descent throttle swing from 0.84 to 0.23 gives
  exactly the observed 1625°/s.
- **Spool torque is transient.** It peaks at ~6.2 N·m per unit throttle step
  (20 times the vane authority). The vanes shed the momentum over about
  0.3–2 s, which is a learnable rate-feedback task. The old reward simply
  never paid for it: about 5 units per episode against ±1600 terminals.

What changed:

1. **Dense yaw-weighted body-rate cost.** Σ(rate/soft)² per second, with soft
   limits of 180/180/90°/s. Yaw at 90°/s costs half the hover cost.
2. **SPIN termination at 360°/s.** It carries the failure penalty.
3. **Throttle exploration prior of −2.0 log-std** (rule 4). An offline 60 s
   simulation of a hover-biased policy without yaw feedback gave:
   - at −1.0 (old default): 96% of episodes spin past 360°/s;
   - at −2.0: median peak 124°/s, p99 169°/s.
4. **Rotor spawns at hover speed** (`initial_motor_omega_fraction: hover`),
   as if the vehicle spooled up on the pad. The old recovery stages spun up
   0 → hover in the air, giving the body about 40 rad/s of yaw at spawn.
5. **Hover capture requires ≤60°/s body rate.** Previously a spinning vehicle
   could "hold" a hover waypoint.
6. **The previous action is observed**, so the policy knows the pending spool
   command.

**Hardware to-do.** `rotor_inertia: 0.0002` is an estimate. Yaw coupling
scales linearly with it, so measure it (spin-down test) before sim-to-real.

## Randomization and curriculum

The final (evaluation) task has 1–6 waypoints:

- **Legs.** 2–25 m horizontal and ±10 m vertical, with 10% pure vertical
  legs, at altitudes of 2–30 m inside a ±60 m arena.
- **Waypoint mix.** Final waypoint is `land` 40% of the time; 25% of
  intermediate waypoints are `hover`.
- **Tolerances.** Flypass radii 0.5–2 m, hover radii 0.3–1 m, pad radii
  0.3–0.5 m, holds 0.5–5 s.
- **Spawn.** Speeds up to 3 m/s, ±20° tilt, any heading, rotor at hover ±5%.
- **Environment.** Battery state of charge 50–100%, steady wind 0–4 m/s from
  any direction.
- **Disturbances.** Sensor noise, centre-of-mass shift and gusts come from
  `--disturbance`, e.g. `configs/disturbances/combined.yaml`.

Curriculum stages (advance on rolling mission success):

1. `hover_hold` at ≥0.6 success.
2. `short_legs` at 0.5.
3. `landing` at 0.5.
4. `routes` at 0.5.
5. `full_task`, which is identical to the evaluation task.

Stages change only spawn, generator and episode length.

## Commands

From `simulation/isaac`, using `..\..\env_isaaclab\Scripts\python.exe`:

```powershell
# Train (STOP file in the run directory stops cleanly)
python apps/run_train_waypoints.py --num-envs 8192 --rollout-steps 32 --total-steps 400000000
# Evaluate a checkpoint on the random task and the fixed benchmark routes
python apps/waypoint_eval.py --checkpoint runs/waypoint_flight/<run>/ppo_best.pt --num-envs 512
```

The benchmark routes are defined in `apps/waypoint_eval.py`: station_keeping,
straight_line, square, climb_descend, reversal, vertical_landing,
landing_approach and zigzag_to_land.

## Not yet done

- Mission control still serves the legacy 43-observation policy. Flying a
  `waypoint_flight_v1` checkpoint needs these changes:
  - **Planner UI:** a LAND waypoint anywhere, straight legs instead of splines,
    and no speed field.
  - **`run_mission.py`:** build explicit missions through
    `WaypointFlightTask.set_explicit_missions`.
- The policy is feed-forward (MLP). GTrXL needs a sequence-aware PPO optimizer
  (`apps/run_train_gtrxl.py` still refuses to train).

## First training attempts (2026-09-23, 8192 envs, ~16k env-steps/s)

| Run (runs/waypoint_flight/) | Change | Result |
|---|---|---|
| `..._143257` | initial | Critic never fit: explained variance 0.000 over 25 updates, v_loss ~2300 vs ±400 terminals. Updates 1–9 were single full-LR Adam steps (KL guard). |
| `..._144426` | PopArt value normalization + actor LR warmup | Explained variance 0.95 by update 12, but every stage-0 episode learned to climb into the 45 m ceiling. With γ=0.999, postponing −400 is worth ~12/s. |
| `..._145644` | Hover settle/hold potentials | 50% stage-0 success, then decayed; the settle potential was sign-inverted (it paid for leaving the target). |
| `..._150323` | Corrected settle potential | Same decay; 60°/s hold gate unreachable (2.3% rotor change gives 61°/s of momentum-exchange yaw). |
| `..._151130` | Hold-gate curriculum 240→60°/s; fine 2 m target vector (54 obs) | Stage 0 mastered at 63% in 2.6M steps; stage 1 went 100% geofence (climb). |
| `..._152010` | Mission-relative geofence; stage 0 needs translation | Stage 0 mastered at ~52M steps; stage 1 had zero success for 160M steps. After ~130M, throttle saturated at 0.98, KL guard cut every update to one minibatch, EV went negative. Stopped at 214M. |

Plant check (open loop, vanes at zero): measured yaw matches −46.5 × Δrotor
fraction within 5–10% with no drift; swirl and vane torque cancel. The
failures are learning failures, not physics.

Persistent failure mode across runs: the policy learns "more throttle" to
avoid early sinking crashes, never learns precise altitude hold, and climbs
away from targets. The climb drives rotor speed above spawn, so body yaw sits
at −100 to −200°/s. Exploration collapses (log-std for fins and throttle
falls to ~0.055) before translation is learned.
