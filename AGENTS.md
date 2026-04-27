# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

Research project validating a **Gated Transformer-XL PPO (GTrXL-PPO)** agent for thrust-vectoring control (TVC) in retro-propulsive landings on a physical **Electric Ducted Fan (EDF) drone** testbed. The goal is sim-to-real transfer from a custom Python simulation to hardware

## PPO/RL Training Principle

For PPO and other RL work, do not solve training failures with ad hoc controller or environment workarounds that only make a smoke test pass. The point of PPO/RL in this project is to learn the control behavior over many epochs from a correct algorithm, well-defined observations/actions, and robust reward terms. When training behavior is poor, prefer fixing PPO implementation correctness, reward shaping, telemetry, normalization, curriculum, or training convergence criteria. Any scripted controller, guidance, or baseline wrapper used during training must be explicit, justified, and treated as part of the task definition rather than as a hidden patch for a specific failure.

### Permitted research-aligned levers

The following are explicitly **not** ad hoc workarounds and may be applied when justified by the diagnosis. They remain part of the algorithm/task definition (versioned in code or YAML), not hidden patches:

1. **Termination / contact-detection correctness.** A reward signal the agent never observes cannot shape its policy. If a terminal event (LANDED, CRASHED, success criterion) physically occurs but the state machine fails to register it, that is an *environment-correctness bug*, not a reward-shaping question. Fix the detector first, before re-tuning weights.

2. **Reward-magnitude balance between terminal and per-step terms.** Integrated per-step costs over an episode of length T can dwarf one-time terminal rewards. Whenever per-step `Σ |w_i · r_i| · T` is comparable to or larger than `|w_terminal|`, PPO will optimize per-step minimization (e.g. zero throttle) and never sample the terminal event. Re-balance weights in the task YAML so terminal magnitudes clearly dominate, document the magnitudes in comments, and verify with an offline arithmetic estimate before training. A global `--reward-scale` < 1.0 must not be used to compensate, since it shrinks all weights equally and so cannot fix a per-step-vs-terminal magnitude inversion.

3. **Potential-based reward shaping (Ng, Harada & Russell, 1999).** Dense shaping terms of the form `F(s, s') = γ Φ(s') − Φ(s)` are policy-invariant and may be added to accelerate exploration of sparse-terminal landing tasks. Non-PBS dense terms (e.g. position-error penalties) are also permitted, but their magnitude must respect rule (2).

4. **Initialization priors on the action distribution.** Biasing the actor's output (e.g. throttle channel near hover) so initial rollouts straddle the productive control basin is standard practice for control PPO and is not a scripted controller — the policy still learns the full action distribution from gradient updates. Document the chosen prior and the physical reasoning (mass, max thrust, hover throttle) in code comments.

5. **Curriculum on spawn / disturbance / episode length.** Staged difficulty (easier spawn box → full spawn box, calm → disturbance) is a research-aligned method, provided each stage's task definition is explicit in YAML and the final evaluation runs against the un-curricularized task.

Any change covered by (1)–(5) should cite, in a code comment or commit message, the diagnostic evidence that motivated it — typically a metric from `eval_log.jsonl` or `train_log.jsonl` — so the reasoning survives in the project record.
