# Speaking Notes — TVC RL Presentation
## "Reinforcement Learning for Thrust-Vectored Fin Control of an EDF VTOL Vehicle in Simulation"
### ~20 minutes, AI Conference, General Audience

---

## Slide 1 — Title Slide (~30 seconds)

Good morning/afternoon everyone. My name is Jacob Tang from Minnesota State University, Mankato. Today I'll be presenting our work on using reinforcement learning — specifically Proximal Policy Optimization — to control a thrust-vectored electric ducted fan drone in simulation. This is a vehicle that steers not by spinning multiple rotors at different speeds like a typical quadcopter, but by deflecting small fins inside its own exhaust stream. It's a fundamentally different and much harder control problem, and we'll show how deep RL can tackle it.

---

## Slide 2 — Presentation Outline (~30 seconds)

Here's our roadmap for the next 20 minutes. We'll start with why this problem matters and what makes it hard. Then we'll walk through the vehicle itself, the simulation environment we built in NVIDIA Isaac Sim, the physics models under the hood, our PPO training setup and reward design, and then the results for both hover and landing tasks. We'll wrap up with the open challenges and where this work is headed. The key question we're answering: can deep RL learn to control a vehicle that steers by deflecting fins in its own exhaust stream? Spoiler — yes, but with some interesting caveats.

---

## Slide 3 — Motivation & Problem Statement (~1.5 minutes)

So why are we doing this? Most drones you see today are multirotors — quadcopters, hexacopters. They control attitude by spinning each rotor at a different speed. It's elegant, it's well-understood, and there's a ton of RL work on it.

But there's another class of vehicles that use thrust vector control — a single engine, and you steer by redirecting the exhaust. Think SpaceX Falcon 9 grid fins, or old-school rocket jet vanes. Our vehicle does exactly this: one electric ducted fan, four flat-plate fins sitting in the exhaust stream.

The fundamental challenge is this coupling shown at the bottom right: fin force is proportional to throttle squared. So when you reduce throttle to descend, you simultaneously lose your ability to steer laterally. At 50% throttle, you only have 25% of your maximum control authority. This is a deeply nonlinear, tightly coupled control problem. Classical PID or LQR controllers work fine near hover, but they struggle across the full flight envelope. That's exactly where RL shines — learning the full nonlinear mapping end-to-end.

---

## Slide 4 — Related Work (~1 minute)

Briefly on related work. RL for aerial vehicles has made great progress — Hwangbo showed PPO can learn agile quadrotor control, Koch compared multiple RL algorithms for attitude control, and GPU-parallel simulation from Isaac Gym has been a game-changer for training speed.

On the thrust vector control side, there's a long history in rocketry, but applying RL to jet-vane TVC for VTOL vehicles is largely unexplored. Classical approaches linearize around hover and apply PID or LQR, which limits the operating envelope.

For sim-to-real, domain randomization and system identification are the standard tools. The fidelity of the dynamics model is critical, which is why we put significant effort into our semi-empirical models.

The research gap is clear: RL for thrust-vectored fin control of EDF vehicles, with the unique challenges of coupled throttle-authority, hobby-grade actuators, and gyroscopic effects.

---

## Slide 5 — Vehicle Platform (~1 minute)

Here's our testbed. It's a custom-built 3.1 kilogram EDF drone. Single FMS 90-millimeter 12-blade ducted fan, powered by a 6S LiPo battery. Four MG996R hobby servos actuate flat-plate jet vanes in the exhaust stream, arranged in a cruciform pattern.

The control architecture gives us full 6-DOF authority through just 5 actuator commands: four fin angles for pitch, roll, and yaw, plus one throttle for altitude. The forward and aft fins handle pitch, left and right handle roll, and differential deflection gives us yaw. It's mechanically simpler than a quadrotor in some ways, but the control problem is significantly harder.

---

## Slide 6 — Simulation Environment (~1 minute)

We built our simulation on NVIDIA Isaac Sim using the Isaac Lab framework. The key enabler is GPU-accelerated parallel simulation — we run 128 copies of the drone simultaneously on a single GPU. Physics runs at 120 Hz, and the RL agent makes decisions at 30 Hz with a decimation factor of 4, meaning four physics substeps per RL step.

The environment implements a standard Gymnasium interface, so it plugs directly into any RL training framework. All the aerodynamic forces, propulsion dynamics, and actuator models are updated at the full 120 Hz physics rate, not just at the RL decision rate. This matters for capturing the servo dynamics and spool-up behavior accurately.

---

## Slide 7 — Propulsion Model (~1 minute)

The EDF propulsion model has two key components. First, first-order spool dynamics — the motor doesn't instantly reach commanded speed, it has a time constant of 0.15 seconds. Second, quadratic thrust — thrust scales with the square of rotor speed.

What makes this interesting are the three torque components. Static reaction torque opposes the rotor spin. Dynamic spool torque is the reaction to rotor acceleration — when you throttle up quickly, the body wants to spin the other way. And gyroscopic precession couples body rotation to rotor angular momentum. This last one is particularly important: if the vehicle pitches while the rotor is spinning at 4300 rad/s, you get a yaw torque. These are real physical effects that make the control problem non-trivial and that a good simulation must capture.

---

## Slide 8 — Fin Aerodynamic Model (~1 minute)

The fin aerodynamic model is semi-empirical. Each fin sees a dynamic pressure that depends on exhaust velocity and throttle squared — there's that coupling again. We use a lift-curve slope of 3.5 per radian from thin-airfoil theory with empirical correction, plus a stall saturation term to capture the nonlinear behavior at large deflections.

The critical point I want to emphasize is at the bottom: this quadratic coupling between throttle and fin effectiveness is THE defining characteristic of jet-vane TVC. It's what makes this fundamentally different from a quadrotor, where each rotor provides independent, throttle-decoupled control authority. When our vehicle descends and reduces throttle, it simultaneously loses its ability to correct lateral errors. This is the core tension the RL agent must learn to navigate.

---

## Slide 9 — PPO Algorithm (~1 minute)

We use standard PPO with clipped surrogate objectives. For those less familiar, PPO is an on-policy actor-critic algorithm that constrains how much the policy can change in each update — preventing catastrophically large gradient steps while maintaining good sample efficiency. It's become the workhorse of continuous control RL.

Our setup uses GAE for advantage estimation with gamma 0.99 and lambda 0.95, plus an adaptive KL early stopping criterion as an additional safeguard. The hyperparameters on the right are fairly standard for continuous control tasks. We chose PPO specifically because prior work by Koch and Hwangbo showed it provides robust training stability for aerial vehicle control.

---

## Slide 10 — Observation & Action Space (~1 minute)

The observation space is 24-dimensional. Position error to the target, attitude as a quaternion to avoid gimbal lock, linear and angular velocities in the body frame, height above ground, all four fin angles and their rates, normalized motor RPM, and contact state.

Two design choices worth highlighting: we include fin angles and rates as proprioceptive feedback because the servo lag is significant — the policy needs to know where the fins actually are, not just where it commanded them. And we include normalized motor RPM so the policy knows how much fin authority it currently has, given that coupling.

The action space is just 5 dimensions — four fin deflections and one throttle. That's actually smaller than a quadrotor's 4 rotor speeds, which is a nice property of this architecture.

The network is straightforward: separate actor and critic, each with two hidden layers of 256 units and tanh activations.

---

## Slide 11 — Initialization Priors (~1.5 minutes)

This slide is arguably the most important practical insight from our work. We found that two initialization choices are absolutely critical for training to succeed.

First, throttle bias initialization. We set the actor's output bias for the throttle channel so the initial mean throttle maps to about 0.78 — near the hover throttle of 0.88. Without this, the default is 0.5, which means the vehicle immediately free-falls. It accumulates huge negative rewards, and PPO converges to a zero-throttle equilibrium — the agent literally learns that crashing fast minimizes cost. This is a degenerate local optimum that's very hard to escape.

Second, per-channel exploration noise. We use quieter exploration on the fins (log sigma = -2) and louder exploration on throttle (log sigma = -1). With uniform noise, the fins oscillate wildly, destabilizing the vehicle before it can learn anything useful about descent.

These aren't ad hoc hacks — they're physically motivated initialization priors. The throttle bias comes from the hover condition: square root of mg over T-max. The noise asymmetry reflects the different sensitivity scales of the actuators. But without them, training simply fails. This is an important lesson for anyone applying RL to thrust-vectored vehicles.

---

## Slide 12 — Reward Design (~1.5 minutes)

Reward design for landing is tricky, and we learned an important lesson about magnitude balance. The reward has two components: per-step shaping terms that provide dense gradients throughout the episode, and one-time terminal events for crash, touchdown quality, pad accuracy, and landing success.

The key insight is on the right. For a 30-second episode at 30 Hz, that's 900 steps. The integrated per-step costs can easily reach order 100. If your terminal rewards aren't substantially larger than that, PPO will optimize per-step cost minimization instead of actually landing. In practice, this means the agent learns to reduce throttle to zero — crashing quickly minimizes the accumulated per-step penalties. We set terminal magnitudes at 200 or above to clearly dominate the per-step budget.

The horizontal guidance shaping at weight -0.30 provides a dense lateral gradient throughout descent. This is important because the exponential pad accuracy bonus has negligible gradient when you're more than 2 meters away, which is most of the descent.

---

## Slide 13 — Training Procedure & Curriculum (~1 minute)

We train two tasks. Hover uses a residual-PID architecture — a PID controller provides the baseline, and PPO learns a small bounded correction. This converges in just 2,048 steps with a very conservative learning rate.

Landing is the main event: pure PPO, no PID baseline, 5 million steps. The policy must learn the entire descent-to-touchdown trajectory from scratch.

For landing, we use a spawn-position curriculum. Training starts with a narrow ±0.5 meter spawn box, linearly annealing to the full ±2.0 meter range over the first 3 million steps. This lets the policy first learn how to touch down softly before dealing with large lateral offsets. Critically, evaluation always uses the full range — we never report metrics on the easy version of the task.

---

## Slide 14 — Hover Results (~45 seconds)

Hover results are clean. Mean position error of 0.085 meters, well within the 0.5 meter threshold. Near-zero tilt and angular rates — the PID baseline handles attitude, and the RL residual corrects position drift. The remarkable thing is this only took 2,048 environment steps to converge. The residual architecture is very efficient when you have a reasonable baseline controller.

---

## Slide 15 — Landing Results (~1.5 minutes)

Landing results are more nuanced. The headline numbers: 100% landing rate with a mean touchdown speed of 0.126 meters per second. That's a very soft landing — well below the 0.5 m/s contact detection threshold. The policy has clearly learned controlled descent and gentle touchdown.

However, lateral accuracy is the open challenge. Mean pad distance of 1.89 meters, and only 2% of landings are actually on the pad. The policy reliably descends and lands softly, but it doesn't consistently steer toward the pad center. It's learned the "land safely anywhere" part of the task but not the "land precisely here" part.

This is actually a very informative result. It tells us that the vertical control problem — managing throttle for controlled descent — is well within PPO's capability. But the lateral guidance problem, complicated by that throttle-fin coupling, needs more work.

---

## Slide 16 — Training Curves (~45 seconds)

The training curve tells a clear story. Reward starts around -3 in the first million steps — per-step costs dominate because the policy hasn't learned to land yet. Around 2 million steps, we see the crossover to positive rewards, meaning terminal landing bonuses now outweigh the per-step costs. By 5 million steps, mean reward reaches about +1.75 with a monotonic upward trend.

That negative-to-positive crossover is the key moment: it marks when the majority of episodes end in successful landings rather than timeouts or crashes. The 100% landing rate is achieved relatively early and maintained throughout, while the policy continues improving touchdown softness and descent control.

---

## Slide 17 — Lateral Accuracy Challenge (~1 minute)

Let me dig into why lateral accuracy is hard. Four factors contribute.

First, the throttle-fin coupling we've discussed — descending means less lateral authority. Second, the servo bandwidth — hobby-grade MG996R servos can only move so fast, limiting correction rate during final approach. Third, the reward gradient — the exponential pad bonus has essentially zero gradient beyond 2 meters, and most of the descent happens at larger distances. Fourth, the curriculum may not provide enough training signal for large lateral corrections.

Compared to quadrotor RL, this is a fundamentally harder lateral control problem because attitude authority is coupled to thrust state. Standard quadrotor techniques like separate attitude and position loops don't directly transfer.

---

## Slide 18 — Sim-to-Real Considerations (~1 minute)

For sim-to-real transfer, we've identified four main gap sources. Aerodynamic model fidelity — our flat-plate model may miss flow separation and wake interactions. Servo nonlinearities — real servos have backlash and load-dependent behavior. Battery voltage sag under load. And sensor noise — simulation gives clean states, reality gives noisy IMU data.

Each has a mitigation path, and the overarching strategy is domain randomization — randomizing these parameters during training so the policy learns to be robust to the uncertainty. This is a well-established approach, and our simulation architecture is designed to support it.

---

## Slide 19 — Future Work (~1 minute)

Five directions going forward. First, replacing the feed-forward MLP with a Gated Transformer-XL architecture to give the policy temporal context — this should help with wind disturbance rejection and trajectory planning. Recent aerospace results from Federici and Carradori show transformers improve landing under uncertainty.

Second, domain randomization for sim-to-real robustness. Third, improving lateral guidance through potential-based reward shaping and hierarchical policies. Fourth, the ultimate goal — deploying on the physical drone testbed. And fifth, training under stochastic wind conditions.

The near-term priority is GTrXL integration plus domain randomization, leading to hardware flight tests.

---

## Slide 20 — Conclusion & Thank You (~30 seconds)

To summarize: we've shown that PPO can learn thrust-vectored fin control for an EDF VTOL vehicle in simulation. The hover policy achieves 8.5 centimeter accuracy in just 2,048 steps. The landing policy achieves 100% success rate with soft touchdowns over 5 million steps. We've identified critical initialization priors and reward design principles specific to TVC vehicles. And we've characterized the lateral accuracy challenge as the key open problem.

The code and data are publicly available on GitHub. Thank you for your attention — I'm happy to take questions.

---

## Timing Summary

| Slide | Topic | Time |
|-------|-------|------|
| 1 | Title | 0:30 |
| 2 | Outline | 0:30 |
| 3 | Motivation | 1:30 |
| 4 | Related Work | 1:00 |
| 5 | Vehicle Platform | 1:00 |
| 6 | Simulation Environment | 1:00 |
| 7 | Propulsion Model | 1:00 |
| 8 | Fin Aerodynamic Model | 1:00 |
| 9 | PPO Algorithm | 1:00 |
| 10 | Obs/Action Space | 1:00 |
| 11 | Initialization Priors | 1:30 |
| 12 | Reward Design | 1:30 |
| 13 | Training & Curriculum | 1:00 |
| 14 | Hover Results | 0:45 |
| 15 | Landing Results | 1:30 |
| 16 | Training Curves | 0:45 |
| 17 | Lateral Challenge | 1:00 |
| 18 | Sim-to-Real | 1:00 |
| 19 | Future Work | 1:00 |
| 20 | Conclusion | 0:30 |
| **Total** | | **~20:00** |
