# PPO convergence follow-up — September 11, 2026

The user removed the earlier one-hour GPU limit. This is an ongoing experiment
record. September 13 holdouts establish 3,002/3,072 successes (97.72%) for the
learned stochastic policy at 26,034,176 steps. Deterministic inference remains
below the convergence gate; the best September 11 checkpoint achieves 191/512
successes (37.30%) on an independent seed-456 deterministic holdout. These experiments
train the feed-forward PPO baseline; the GTrXL entrypoint still lacks a sequence
optimizer and is not a trained-policy result.

See [the September 13 validation report](ppo_landing_validation_2026-09-13.md)
for the selected checkpoint, inference mode, independent holdouts and replay.

## Algorithm and measurement corrections

- Critic clipping in policy-ratio units (0.2) is disabled by default. Its flat
  regression gradients were incompatible with terminal returns near 475.
- Curriculum outcomes now enter the rolling window in temporal order. Aggregated
  success-first insertion biased the window when it evicted older events.
- A stage transition finishes the old rollout/GAE update, then resets all
  environments to the new stage. Old-stage episodes cannot enter its statistics.
- Training evaluation measures one episode per environment. Repeated automatic
  resets no longer give fast failures extra weight. `curriculum_eval.jsonl`
  separates deterministic current-stage evaluation from the unchanged full task.
- Format-2 checkpoints retain optimizer state, curriculum state, update and
  cumulative environment-step counts. Resume starts fresh physical episodes;
  it is not a bitwise replay of RNG/scene state. Physical/task configuration is
  checked, except that the number of independent environments may change.
- Runs record source-file hashes. Creating `STOP` inside a current run directory
  requests a clean stop with final checkpoint/evaluation at a rollout boundary.

## Diagnosed safety-reward escape

`runs/ppo_convergence/ppo_landing_seed0_20260911_110123` reached stage 1 at
425,984 steps, but retained zero successes through 1,032,192 steps. Its critic
explained variance improved substantially; removing value clipping alone did
not solve the behavior.

The 507,904-step checkpoint was evaluated on three independent stage-1 episodes
(seed 123). All climbed through the 30 m altitude safety limit after 8.63–8.93 s,
with upward speeds around 8.5 m/s and AIRBORNE contact state. The old crash reward
checked contact state alone, so these terminal failures escaped its -200 penalty.
See `runs/ppo_diagnostics/stage1_507904/trajectory.jsonl`.

`check_failure_terminations` is now shared by the detector and crash reward;
altitude/tilt safety failures receive the same configured terminal penalty as
physical crashes. Ordinary timeouts remain truncations. This corrects a missing
failure signal; it does not change action control or loosen landing criteria.
The unit suite passes 167 tests, including this airborne-terminal regression.

## Evidence and run lineage

| Run/checkpoint | Result and role |
| --- | --- |
| `ppo_convergence/...110123/ppo_stage_0_mastered.pt` | 425,984 steps; 60% rolling stochastic stage-0 success gate passed |
| `ppo_diagnostics/stage0_mastered` | Four independent deterministic stage-0 episodes, seed 321: 4/4 successful; touchdown 0.1219–0.1487 m/s, pad error 0.0195–0.1087 m; small diagnostic, not full-task validation |
| `ppo_resume_smoke/...110441` | Resume 16,384→32,768 steps retained curriculum counts and advanced every Adam state from step 32→64 |
| `ppo_safety_reward_fix/...111004` | Resumed mastered-stage checkpoint under corrected safety reward; cleanly stopped at 737,280 steps to increase parallel throughput |
| `ppo_throughput_512/...111144` | 512-copy runtime/throughput smoke: 131,072 steps in 31.3 s including short evaluations; first rollout ~7,480 transitions/s despite concurrent training; not a convergence result |
| `ppo_safety_512/...111352` | Completed 20,004,864 cumulative steps in 2,124.8 s; stage 1→2 at 13,189,120, 2→3 at 14,761,984, 3→4 at 18,235,392; final training rolling success 15.62% |
| `ppo_holdouts/full20m_seed123` | Independent full-task deterministic holdout: 3/512 successful (0.59%); 360 crashes/safety failures, 106 timeouts, 23 soft off-pad, 20 hard landings (10 also off-pad) |
| `ppo_full_task_60m/...182342` | Resumed 20,004,864-step checkpoint for continued training on the full task, targeting 60 million cumulative steps |

The final task is still a cold-rotor spawn at 16–20 m with ±2 m horizontal
position and the original randomized velocities/attitudes. Success still requires
LANDED, pad distance ≤0.5 m and worst arrival speed across bounces ≤0.25 m/s.
All runs above use direct learned actions: no PID residuals, behavior cloning,
or scripted landing guidance.

The 20M training-seed full-task evaluation had 2/512 successes (0.39%). The
independent holdout agrees that this checkpoint is far below the convergence
gate (at least 80% success and at most 5% crashes). The higher rolling training
success fraction must not be reported as deterministic evaluation performance.

`apps/run_eval_ppo_batch.py` writes exact initial states and one record per
episode, plus a checkpoint hash and task configuration. Deterministic evaluation
is its default; optional stochastic evaluation is explicitly labeled for
diagnosing differences from training rollouts.

## Full-task continuation and discount experiment

The same 20M checkpoint achieves 139/512 successes (27.15%) when sampling its
learned distribution on the seed-123 initial states. This is a diagnostic result,
not a replacement for deterministic validation. The deterministic comparison was
3/512. Stochastic failures were 40 crashes, 7 timeouts, 134 soft off-pad landings,
and 192 hard landings (130 also off-pad).

At 22,036,480 steps, continued gamma=0.99 training evaluated at 0% deterministic
success and 98.05% timeouts. This supports checking the time horizon, alongside
retaining the unchanged baseline run. With a 0.03332 s control step, gamma=0.99
weights a terminal 10 s ahead by about 0.049 and one 30 s ahead by 0.000118.
Gamma=0.999 weights those events by about 0.741 and 0.406, respectively.

`ppo_long_horizon/...183143` starts from the 20,004,864-step learned actor and its
Adam state, with gamma=0.999. `--reset-critic` explicitly reinitializes its value
network and critic Adam state because changing the discount changes value targets.
Reward weights, actions, physics, observation representation, curriculum state,
entropy coefficient and learning rate are unchanged. No result is claimed for
this comparison until its full-task evaluation has run. Its 22,036,480-step
evaluation was 0% success and 99.80% crashes. The branch was stopped cleanly at
22,429,696 steps to prioritize the action-noise mismatch; this short branch does
not establish that a longer horizon cannot converge.

## Exploration-consolidation experiment

At 24,002,560 steps the unchanged continuation had roughly 65% rolling sampled
training success, but deterministic evaluation again had 0/512 successes and
85.74% crashes. Replacing `tanh(mean)` with the true bounded-action expectation
(20-point Gauss-Hermite quadrature) also failed: 0/512 successes on seed 123.
That diagnostic remains an explicitly labeled evaluator option; deployment and
training defaults were not changed to hide this failure.

`ppo_exploration_anneal/...184222` resumes the 24M actor, critic and optimizer
with gamma=0.99. It removes the entropy bonus and linearly lowers caps on the
learned log standard deviations over five million transitions. Fin caps move
from approximately -2.4 to -4 (physical angle sigma at most 0.0048 rad), and the
throttle latent cap from -1.058 to -2.5. These are explicit, checkpointed algorithm
settings. Every action mean remains learned by PPO, with full actuator range;
no PID, scripted guidance, actuator compensation or relaxed gate is introduced.
The schedule never raises a variance that PPO has already reduced below its cap.
The unchanged continuation was the comparison run. The unit suite then had
169 passing tests, including the distribution expectation and variance-cap checks.

The comparison was stopped at 28,000,256 steps with 0% deterministic success.
The annealing run finished 40,058,880 steps, but its final checkpoint regressed;
the best deterministic checkpoint is `ppo_step_28000256.pt` in
`runs/ppo_exploration_anneal/ppo_landing_seed0_20260911_184222`.

| Annealing checkpoint | Full-task deterministic success | Crashes/safety failures |
| --- | ---: | ---: |
| 26,034,176 | 26.17% | 31.25% |
| 28,000,256 | 36.33% | 27.93% |
| 30,031,872 | 32.23% | 34.18% |
| 32,063,488 | 28.32% | 31.84% |
| 34,029,568 | 8.79% | 49.80% |
| 36,061,184 | 4.88% | 76.95% |
| 38,027,264 | 10.74% | 68.36% |
| 40,058,880 | 3.52% | 54.30% |

Independent holdouts confirm the intermediate improvement:

- `ppo_holdouts/full26m_seed123_anneal`: 130/512 successes (25.39%), 176 crashes,
  116 hard landings (53 also off-pad), 90 soft off-pad landings, no timeouts.
- `ppo_holdouts/full28m_seed456_anneal`: 191/512 successes (37.30%), 130 crashes,
  49 hard landings (21 also off-pad), 142 soft off-pad landings, no timeouts.
  LANDED impact-speed mean 0.22285 m/s and pad-distance mean 0.48254 m.

## September 13: update stability

The annealing run's `train_log.jsonl` has median epoch-averaged approximate KL
0.04604 and maximum 0.88656 despite target 0.03. At 34M, clipping affected 43.67%
of sampled ratios. The old guard waited until every minibatch in an epoch had
already updated the policy. Narrowing variance makes small action-mean changes
large relative to the learned distribution, so the same learning rate becomes
more aggressive. This diagnosis does not establish the sole cause of regression.

The trainer now checks exact diagonal-Gaussian KL before each minibatch update
and stops when it exceeds the target. The shared tanh bijection preserves KL.
This follows the pre-update placement used by [Stable Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/ppo/ppo.html),
using analytic KL instead of a sampled estimate. A single optimizer step can
still overshoot; this is an early-stop guard, not a strict trust-region solver.
`policy_kl_final` measures the actual final policy against the complete rollout,
including scheduled variance changes; logs also record early stops and optimizer
minibatch counts. Analytic KL is checked against PyTorch's distribution reference.

`ppo_kl_consolidation` resumes the best 28M checkpoint, retaining its optimizer,
critic and saved variance schedule. Learning rate is 3e-5 (previously 3e-4), KL
target 0.015 (previously 0.03), entropy coefficient 0, and gamma 0.99. The schedule
still reaches its original caps at 29,002,560 transitions. Physics, task gates,
reward weights, observations and full action ranges are unchanged. The run is
configured for up to 60M cumulative transitions with evaluation every 2M;
continuation decisions use deterministic landing outcomes, not training success.

The consolidation comparison was stopped cleanly at **34,029,568** transitions,
after **6,029,312 new transitions / 92 updates** and 817.5 seconds including
evaluations. Deterministic full-task results progressed from 52.54% success /
18.75% crashes at 30M, to 54.69% / 11.72% at 32M, to **61.72% / 10.94%** at 34M.
The final and best checkpoints both retain the 34M state and optimizer. Final
full-rollout KL had median 0.006826 and maximum 0.021243; seven updates triggered
the new minibatch guard. The guard is functioning and regression is reduced,
but deterministic inference still does not pass the success/crash gate.

The selected stochastic 26M policy passes independent nominal, sensor-noise,
wind/gust, and COM-offset tests documented in the validation report. This is
an explicit stochastic-policy result; it does not relabel training-window
statistics or deterministic inference as passing. Both evaluators share model
inference code and record `action_mode`; single-environment evaluation also
retains full precision when applying impact-speed and pad-distance gates.
Final regression suite: **171 unit tests passed**.

Independent final-checkpoint evaluation (`ppo_holdouts/full34m_seed6543_kl_consolidation`):
319/512 deterministic successes (62.30%), 65 crashes, 3 timeouts, 60 hard
landings (18 also off-pad) and 65 soft off-pad landings. The stochastic 26M policy
remains the selected passing policy; deterministic 34M is retained as a resumable
improvement candidate. All training and evaluation processes finished cleanly.
