# Feature Specification: Isaac Sim PID Controller Tuning, Evaluation & Episode Logging

**Feature Branch**: `003-pid-controller`
**Created**: 2026-03-12
**Status**: Draft
**Input**: User description: "Use the PID in `simulation` to train a PID controller in Isaac Sim. Use Ziegler-Nichols PID tuning in single- and multi-env Isaac Sim tests. Test with the best PID values. I should be able to see episode logging (action, state, reward, PID values). For training, monitor reward and print a warning if reward is not decreasing. Be able to disable specific forces, e.g. wind and gyro."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Single-Environment Isaac PID Tuning (Priority: P1)

A researcher runs the existing cascaded `PIDController` from `simulation/training/controllers/pid_controller.py` against the Isaac Sim landing environment in single-env mode. The system performs Ziegler-Nichols-based loop tuning, evaluates candidate gains over repeated episodes, and outputs a ranked result plus a best-gain YAML payload that can be reused directly for later tests.

**Why this priority**: This is the minimum path to prove the existing PID logic can control the Isaac Sim plant without inventing a second controller implementation. If single-env tuning is unstable or mismatched to Isaac observations/actions, multi-env rollout and later policy comparisons are meaningless.

**Independent Test**: Run `python -m simulation.isaac.scripts.tune_pid_isaac --config simulation/isaac/configs/isaac_env_single.yaml --pid-config simulation/configs/pid.yaml --method ziegler-nichols --episodes 20 --disable-wind --disable-gyro` and verify the run completes, prints tuned gains, and writes a best-gains output file.

**Acceptance Scenarios**:

1. **Given** the Isaac Sim single-env task exposes the same 20-D observation contract expected by `PIDController`, **When** the tuning script runs in single-env mode, **Then** it can roll out closed-loop episodes without modifying the controller law.
2. **Given** the tuning method is `ziegler-nichols`, **When** the script searches for ultimate gain / oscillation period per loop, **Then** it produces finite gains for the enabled loops and stores them in a PID-config-compatible structure.
3. **Given** a baseline `simulation/configs/pid.yaml`, **When** the tuning run finishes, **Then** the script reports the baseline score and the tuned score on the same evaluation seeds.
4. **Given** force toggles `--disable-wind` and `--disable-gyro`, **When** the single-env tuning run starts, **Then** those effects are disabled for every episode in that run.

---

### User Story 2 - Multi-Environment Isaac PID Batch Tuning (Priority: P1)

A researcher runs the same PID tuning workflow in a vectorized Isaac Sim environment so multiple candidate gain sets, seeds, or evaluation rollouts can be tested in parallel. The researcher uses the same controller and scoring logic as the single-env run, but now gets higher-throughput candidate evaluation on 128+ environments.

**Why this priority**: Single-env tuning is too slow once gain sweeps, robustness checks, and ablations are added. Multi-env batch evaluation is the only practical way to compare many candidate PID settings on the Isaac plant.

**Independent Test**: Run `python -m simulation.isaac.scripts.tune_pid_isaac --config simulation/isaac/configs/isaac_env_128.yaml --pid-config simulation/configs/pid.yaml --method ziegler-nichols --episodes 64 --num-candidates 16 --disable-wind --disable-gyro` and verify multiple envs run concurrently and the script returns a ranked candidate table.

**Acceptance Scenarios**:

1. **Given** a multi-env Isaac config, **When** the tuning script launches, **Then** it evaluates PID candidates across multiple environments without cross-env state leakage.
2. **Given** different envs terminate at different timesteps, **When** batch evaluation is active, **Then** per-env episode summaries are still correct and no finished env corrupts unfinished ones.
3. **Given** the same candidate gain set is evaluated in both single-env and multi-env modes on matched seeds, **When** results are compared, **Then** the aggregate score differs only within a small tolerance attributable to simulator nondeterminism.
4. **Given** multi-env tuning is enabled, **When** the script prints progress, **Then** it includes candidate id, env count, completed episodes, current best score, and best gains so far.

---

### User Story 3 - Best-Gain Validation with Episode Trace Logging (Priority: P1)

A researcher takes the best PID gains from the Isaac tuning run and validates them in dedicated test episodes. During each episode, the system logs state, action, reward, and PID internals so the researcher can inspect exactly why a run succeeded or failed.

**Why this priority**: A single scalar score is not enough for controller debugging. The user explicitly needs full episode traces including control outputs and reward evolution to diagnose oscillation, saturation, lag, or force-model issues.

**Independent Test**: Run `python -m simulation.isaac.scripts.test_pid_isaac --config simulation/isaac/configs/isaac_env_single.yaml --pid-config runs/pid_isaac/best_pid.yaml --episodes 5 --log-dir runs/pid_isaac_eval` and verify per-step logs contain action, state, reward, cumulative reward, and PID values.

**Acceptance Scenarios**:

1. **Given** a saved best-gains YAML file, **When** the evaluation script runs, **Then** it reuses that file without manual translation or editing.
2. **Given** episode logging is enabled, **When** a rollout executes, **Then** each logged step includes at least `episode_id`, `env_id`, `step`, `time_s`, observation/state fields, action vector, step reward, cumulative reward, and PID terms/gains used for that action.
3. **Given** a rollout terminates early due to crash, landing, out-of-bounds, or timeout, **When** the episode summary is written, **Then** the termination reason and aggregate metrics are preserved.
4. **Given** the best PID gains are tested in both single-env and multi-env configs, **When** evaluation completes, **Then** the summary reports per-config success rate, mean total reward, mean CEP, and failure counts.

---

### User Story 4 - Reward Trend Monitoring & Force Ablation Controls (Priority: P2)

A researcher runs long PID tuning / training jobs and wants the script to tell them when progress has stalled or moved in the wrong direction. The same run must also support force ablations so the researcher can isolate whether wind, gyro precession, anti-torque, or gravity is causing instability.

**Why this priority**: Long Isaac runs are expensive. Without trend warnings, the user can waste GPU time on a stalled search. Without force ablations, PID failures are hard to attribute to the controller versus the plant disturbance model.

**Independent Test**: Run `python -m simulation.isaac.scripts.tune_pid_isaac --config simulation/isaac/configs/isaac_env_single.yaml --pid-config simulation/configs/pid.yaml --monitor-window 10 --monitor-direction decreasing --disable-wind --disable-gyro` and verify the script emits a warning when the monitored reward trend stalls and that disabled forces do not contribute during rollout.

**Acceptance Scenarios**:

1. **Given** reward monitoring is enabled, **When** the rolling reward metric fails to move in the configured direction over the configured window, **Then** the script prints a warning with the current window values and the active best candidate.
2. **Given** the user requests `--monitor-direction decreasing`, **When** episode rewards are flat or increasing over the monitoring window, **Then** the warning triggers exactly on that condition.
3. **Given** force disable flags such as `--disable-wind`, `--disable-gyro`, and `--disable-anti-torque`, **When** the rollout starts, **Then** the corresponding physics contributions are disabled without editing YAML by hand.
4. **Given** multiple forces are disabled simultaneously, **When** an episode trace is logged, **Then** the trace metadata records which force toggles were active for that run.

---

### Edge Cases

- What happens if Ziegler-Nichols probing never reaches sustained oscillation for one loop?
- What happens if a candidate gain set immediately saturates thrust or fins and terminates every episode early?
- How is reward-trend monitoring defined when the optimization target is a cost-like metric rather than a reward-like metric?
- What happens when multi-env evaluation produces different termination reasons in the same batch?
- How large can per-step trace logs get before logging overhead distorts Isaac Sim throughput?
- What happens if wind is disabled but gyro / anti-torque remain active and still destabilize yaw?
- What happens if the PID dt does not match the effective Isaac control step?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST reuse the existing `simulation/training/controllers/pid_controller.py` controller implementation for Isaac Sim tuning and evaluation; it MUST NOT introduce a second, divergent PID law for Isaac-only use.
- **FR-002**: The Isaac PID workflow MUST support both single-environment and multi-environment evaluation using existing Isaac env config files.
- **FR-003**: The tuning workflow MUST implement Ziegler-Nichols-based gain initialization for the controllable loops exposed by `PIDController`, at minimum altitude, lateral-x, lateral-y, roll, and pitch; yaw damping MAY be tuned by a compatible derivative-only sweep.
- **FR-004**: The tuning workflow MUST evaluate PID candidates using Isaac episode outcomes and reward-derived scoring, not only open-loop heuristic metrics.
- **FR-005**: The workflow MUST support testing a baseline PID config and one or more tuned candidate configs on the same seed set for direct comparison.
- **FR-006**: The workflow MUST save the best-performing PID gains in a YAML structure compatible with `simulation/configs/pid.yaml`.
- **FR-007**: The evaluation workflow MUST provide a dedicated "best PID values" test path that runs the saved best gains without re-tuning.
- **FR-008**: Per-step logging MUST include action vector, observation/state values, step reward, cumulative reward, and PID values used to produce the action.
- **FR-009**: Per-episode logging MUST include at least total reward, termination reason, landed/crashed/out-of-bounds flags, episode length, and CEP or equivalent landing error metric.
- **FR-010**: The training / tuning CLI MUST print ongoing progress to the console, including completed episodes, current candidate score, and current best gains.
- **FR-011**: The workflow MUST support runtime force toggles for at least wind and gyro precession, and SHOULD also support anti-torque and gravity because those already exist as separately identifiable Isaac-side effects.
- **FR-012**: Force toggles MUST apply to both single-env and multi-env runs and MUST be recorded in the run metadata / episode logs.
- **FR-013**: The workflow MUST monitor a rolling reward metric and print a warning when the metric fails to move in the user-configured direction over a user-configured window.
- **FR-014**: The reward monitor MUST support `decreasing` as a valid direction because that is the requested user workflow, even if the implementation internally monitors an equivalent cost metric derived from reward.
- **FR-015**: The workflow MUST accept CLI paths for Isaac env config, vehicle config, reward config, PID config, output directory, and log directory.
- **FR-016**: The workflow MUST expose enough logged PID internals to explain each action, including loop errors and the corresponding loop outputs before final action clipping.
- **FR-017**: The workflow MUST function when disturbances are disabled, so deterministic ablation runs can be used for controller debugging and repeatable comparisons.
- **FR-018**: The workflow MUST preserve compatibility with the existing observation/action contract used by the custom Python simulation, so the same PID config can be ported between backends.

### Key Entities

- **IsaacPidTrainer**: Orchestrates PID tuning / training episodes in Isaac Sim, handles candidate evaluation, progress reporting, and best-gain selection.
- **ZieglerNicholsTuner**: Runs ultimate-gain / oscillation-period probing and converts measured `Ku`, `Tu` values into candidate gains for the cascaded PID loops.
- **PidCandidateEvaluator**: Scores one PID gain set over one or more Isaac episodes and returns reward, success, CEP, and failure metrics.
- **EpisodeTraceLogger**: Writes per-step and per-episode logs including state, action, reward, PID internals, and active force toggles.
- **ForceToggleConfig**: Runtime configuration describing which Isaac-side disturbances or physics effects are disabled for a run.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A single-env Isaac tuning run completes end-to-end and produces a finite best-gains YAML file compatible with `PIDController`.
- **SC-002**: A multi-env Isaac tuning run evaluates multiple PID candidates in parallel and returns a ranked result table without env-state corruption or runtime errors.
- **SC-003**: The best tuned PID gains achieve non-regressed performance versus the baseline `simulation/configs/pid.yaml` on the same evaluation seeds, measured by success rate first and mean total reward second.
- **SC-004**: The best-gain validation run writes per-step logs that explicitly contain action, state/observation, reward, cumulative reward, and PID values for every recorded step.
- **SC-005**: Reward-trend monitoring emits a warning when the rolling metric fails the configured direction check over the configured window.
- **SC-006**: Disabling wind and gyro measurably changes the run metadata and removes those contributions from the physics path used during rollout.
- **SC-007**: The same saved best-gains file can be evaluated in both `isaac_env_single.yaml` and `isaac_env_128.yaml` without manual format changes.

## Assumptions

- "Train a PID controller" in this feature means tuning / optimizing PID gains against Isaac Sim episodes, not replacing PID with PPO or another learned policy.
- `simulation/training/controllers/pid_controller.py` remains the source of truth for the control law; Isaac-specific code wraps the environment, logging, and tuning workflow around it.
- The Isaac environment exposes an observation layout compatible enough with `PIDController.get_action()` that no structural controller rewrite is required.
- Existing reward signals in `simulation/configs/reward.yaml` are meaningful enough to rank candidate PID settings, even if the warning monitor tracks an equivalent cost-like transform when the user wants a "decreasing" metric.
- Force toggles can be implemented by patching task-level runtime flags / models (for example `_wind_model`, `_gyro_enabled`, `_anti_torque_enabled`) rather than requiring separate YAML variants.
- Single-env runs are used for debugging and loop identification; multi-env runs are used for throughput and robustness evaluation.

## Dependencies

- Existing PID implementation in `simulation/training/controllers/pid_controller.py`
- Existing PID config in `simulation/configs/pid.yaml`
- Existing Isaac task / environment stack in `simulation/isaac/tasks/edf_landing_task.py`
- Existing Isaac configs including `simulation/isaac/configs/isaac_env_single.yaml` and multi-env variants such as `simulation/isaac/configs/isaac_env_128.yaml`
- Existing reward configuration in `simulation/configs/reward.yaml`
- Feature 001 Isaac Sim environment support
- Feature 002 reaction torque / force-model infrastructure, because force ablations are part of this feature
