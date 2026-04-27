
## SLIDE 1 — Title  [~0:30]

Good morning, and thank you for being here.

I'm Jacob Tang, a graduate student in Applied Data Science at Minnesota State University, Mankato.

Today I want to show you how PPO can learn to fly and land an unusual little vehicle: an electric ducted fan drone that steers by deflecting fins inside its own exhaust.

The same idea that flew on the V-2 and that SpaceX uses on Falcon 9 during powered descent — jet vanes — but at hobby scale.

---

## SLIDE 2 — Outline  [~0:30]

Here is the roadmap.

Motivation, then the vehicle and the simulator we built in NVIDIA Isaac Sim.

Then the physics: propulsion, fin aerodynamics, and the actuators.

Then the learning side — PPO, observations and actions, two critical initialization tricks, and the reward design.

Finally results for hover and landing, the open challenges, and where this work is headed.

---

## SLIDE 3 — Motivation: Why TVC EDF Drones?  [~1:30]

Let me start with the core challenge.

A quadcopter steers by changing each rotor's speed independently — every rotor gives you decoupled control authority.

Our vehicle has one fan, and that fan only pushes air in one direction.

To steer, you deflect four small fins immersed in the exhaust stream.

The force those fins generate scales with the dynamic pressure of the exhaust — and dynamic pressure scales with the square of the throttle.

So when you reduce throttle to descend, you simultaneously lose your ability to steer laterally.

At fifty percent throttle, fin authority collapses to twenty-five percent.

This is not a small perturbation — it's a fundamental coupling that makes the control problem qualitatively harder than a quadrotor.

Classical PID or LQR can be tuned at the hover point but degrade rapidly off it, and that is the gap we use reinforcement learning to fill.

---

## SLIDE 4 — Related Work & Research Gap  [~1:00]

Briefly situating the work.

Hwangbo, Koch, and others showed PPO learns agile quadrotor and attitude control policies that can transfer to hardware.

Rudin and colleagues demonstrated that GPU-parallel simulation in Isaac Gym trains policies orders of magnitude faster than sequential rollouts.

On the TVC side, jet vanes are a classical idea — they flew on the V-2 and remain in use today.

More recent work by Federici and Carradori applied transformer policies to powered landing in simulation, but neither closes the loop on hardware.

The gap we target is specifically RL for thrust-vectored fin control of EDF vehicles, with the quadratic throttle coupling, hobby-grade servos, and gyroscopic effects all present.

---

## SLIDE 5 — Vehicle Platform  [~1:00]

Here is the testbed.

It's 3.1 kilograms, with an FMS 90-millimeter twelve-blade EDF on a 6S battery, producing up to 39.2 newtons — a thrust-to-weight ratio of about 1.29.

The four control fins are flat aluminum plates driven by MG996R hobby servos in a cruciform pattern, each deflecting up to plus or minus 15 degrees.

Forward and aft fins control pitch, left and right control roll, and differential deflection across all four generates yaw.

So five actuator commands — four fin angles and one throttle — give us full six-degree-of-freedom control.

Inertia values were taken from CAD via Isaac Sim's MassAPI; fin geometry and deflection limits were measured directly on the hardware.

---

## SLIDE 6 — Simulation Environment  [~1:00]

The simulation runs on Isaac Lab v2.3.2, using PhysX 5 for GPU-accelerated rigid-body physics.

We run 128 parallel drone instances on a single GPU, each independently spawned and stepped together.

Physics runs at 120 hertz; the RL policy decides at 30 hertz, with four physics substeps between decisions.

That ratio matters because servo lag and motor spool-up happen on timescales shorter than the RL decision period.

The environment exposes a standard Gymnasium interface, and we support two force dispatch modes — per-link forces at each fin's center of pressure, and a collapsed body wrench for debugging.

The per-link mode is the default, because it lets the physics engine compute torques from geometry rather than us synthesizing them by hand.

---

## SLIDE 7 — Propulsion Model  [~1:15]

The EDF is modeled with first-order spool dynamics: when you command a new throttle, the rotor approaches the target with a 0.15-second time constant.

The policy can't rely on instantaneous thrust changes — it has to anticipate.

Thrust follows a quadratic law in rotor speed, calibrated so the maximum 4300 radians per second produces 39.2 newtons.

Beyond thrust, three torque components act on the body.

A static reaction torque opposes the rotor's spin direction.

A dynamic spool torque is the body's reaction to rotor angular acceleration.

And a gyroscopic precession torque couples body rotation to rotor angular momentum — pitch the vehicle while the rotor spins, and you get a yaw torque.

That last effect is what distinguishes this simulation from a simplified point-mass model and what makes yaw control non-trivial.

---

## SLIDE 8 — Fin Aerodynamic Model  [~1:15]

Each fin is modeled as a flat plate immersed in the exhaust, using a semi-empirical force model.

Dynamic pressure scales as throttle squared — and that is where the coupling comes from.

The normal-force coefficient uses a linear lift-curve slope of 3.5 per radian, derived from thin-airfoil theory with a duct correction, plus a cubic saturation term to model stall at large deflections.

Drag has a zero-deflection baseline plus a quadratic term for increasing drag with angle.

Normal force gives lateral control authority; tangential force opposes the exhaust and shows up as a thrust loss subtracted from the EDF thrust.

The servo dynamics are also part of this physics: a 50-millisecond first-order lag with a 7.54 radian-per-second slew limit and a roughly one-degree deadband.

So between commanded fin angle and actual fin angle, there is a real lag — and we'll see in a moment why that drives the observation-space design.

The takeaway is the coupling: at full throttle you have full fin authority; as you cut throttle to descend, fin authority falls quadratically, and the policy has to manage that trade-off.

---

## SLIDE 9 — PPO Algorithm  [~1:00]

For learning we use Proximal Policy Optimization.

PPO is an on-policy actor-critic method whose key innovation is the clipped surrogate objective.

If the new policy's action probability moves too far from the old policy, the objective is clipped — preventing destructively large gradient steps.

That clipping is what makes PPO stable on noisy on-policy rollouts.

Advantages use Generalized Advantage Estimation with gamma 0.99 and lambda 0.95.

We add an adaptive KL early-stopping criterion at target 0.03 as an extra safeguard, and the loss combines the clipped policy term with a value MSE at coefficient 0.5 and an entropy bonus at 0.005.

---

## SLIDE 10 — Observation & Action Space  [~1:15]

The agent gets a 24-dimensional observation at each step.

Position error to the target, attitude as a quaternion to avoid gimbal lock, and linear and angular velocities in the body frame.

Height above ground is included explicitly because it's critical for landing.

Crucially, we include the four fin angles and their rates, not just the commanded angles — because of the servo lag, the policy needs to know where the fins actually are, not where it asked them to be.

Normalized motor RPM tells the policy how much fin authority it currently has, and a contact flag tells it when it has touched down.

The action space is five-dimensional: four fin commands and a throttle command, passed through tanh and linearly scaled to plus or minus 0.262 radians for the fins and zero to one for the throttle.

Both networks are two-layer MLPs with 256 hidden units and tanh activations.

---

## SLIDE 11 — Initialization Priors  [~1:30]

This is, in my view, the most practically important insight from the project.

Two initialization choices were the difference between training succeeding and failing entirely.

First, throttle bias.

By default, the actor's tanh output puts mean throttle at 0.5 — well below the hover throttle of 0.88.

Initial rollouts free-fall, accumulate large negative rewards, and PPO converges on a degenerate optimum: zero throttle, crash fast, end the episode early.

We fix this by setting the output-layer bias on the throttle channel so the initial mean throttle is about 0.78 — just below hover.

Now early rollouts straddle hover and descent, giving the policy something useful to learn from.

The hover throttle itself is just the square root of m-g over T-max — physics, not a hyperparameter.

Second, per-channel exploration noise.

We initialize fin channels with log-sigma minus 2 — standard deviation about 0.14 — and the throttle channel with log-sigma minus 1 — standard deviation about 0.37.

The asymmetry is deliberate: louder fin noise destabilizes the vehicle before it can learn anything; quieter fins and louder throttle keep early trajectories stable enough to learn from.

These aren't ad hoc patches — they encode what we already know about the vehicle.

---

## SLIDE 12 — Reward Design  [~1:30]

Reward design for landing is one of the hardest parts, and the lesson here generalizes.

The reward has per-step shaping terms — alive bonus, position and attitude errors, angular velocity, control effort and rate, vertical-speed shaping, and a delta-v cost — and one-time terminal events at episode end.

The terminal events are a large crash penalty, a touchdown softness bonus, an exponential pad-accuracy bonus, and a flat landing-success bonus.

Here is the critical insight.

A 30-second episode at 30 hertz is 900 steps; the integrated per-step costs can easily reach order 100 in magnitude.

If your terminal rewards aren't substantially larger than that, PPO will optimize per-step minimization rather than landing — and the cheapest way to minimize per-step cost is to crash quickly and end the episode early.

So we set terminal magnitudes at 200 or above: minus 200 for crash, up to plus 200 for pad accuracy, plus 250 for landing success — clearly dominating the per-step budget.

You can verify the arithmetic before training: estimate per-step cost magnitude, multiply by episode length, and make terminals two to three times larger.

The other important detail is the linear horizontal-position penalty at weight minus 0.30 — the exponential pad bonus has essentially no gradient beyond two meters, so we need a dense linear term to point toward the pad through most of the descent.

---

## SLIDE 13 — Training & Spawn-Position Curriculum  [~1:00]

We train two tasks.

For hover, we use a residual-PID architecture: a classical PID provides the baseline action and the PPO policy adds a small bounded correction at scale 0.05.

This converges in just 2,048 environment steps with a very conservative 10-to-the-minus-6 learning rate.

For landing, we use pure PPO with no PID baseline — the policy learns the entire descent-to-touchdown from scratch over five million steps at 3-times-10-to-the-minus-4.

Because precision landing from large initial offsets is a hard exploration problem, we use a spawn-position curriculum.

Training begins with a narrow plus-or-minus 0.5-meter spawn box that anneals linearly to the full plus-or-minus 2.0-meter range over the first three million steps.

Altitude spawn stays constant from 8 to 12 meters, and evaluation is always against the full un-curricularized range.

---

## SLIDE 14 — Hover Results  [~0:45]

Hover first.

After only 2,048 steps, the policy achieves 0.085 meters mean position error, with a maximum of 0.215 meters across all 128 evaluation episodes — both well inside the 0.5-meter success threshold.

Mean tilt and angular rate are essentially zero, telling us the PID is doing its job on attitude while the RL residual handles position drift.

The headline is the speed: 2,048 steps over 128 environments is only about 260,000 transitions — the residual architecture is very sample-efficient when you have a reasonable baseline.

---

## SLIDE 15 — Landing Results  [~1:30]

Now landing, which is more nuanced.

The headline is a 100% landing rate across all 128 evaluation episodes.

Mean touchdown speed is 0.126 meters per second, maximum 0.347 — both well below the 0.5 threshold, so every landing is classified as a soft touchdown.

Mean throttle during evaluation is 0.759, below the 0.88 hover throttle, confirming the policy has learned a controlled descent rather than just hovering in place.

Maximum downward speed reaches 9.75 meters per second, so the policy is willing to descend aggressively when it can.

Now the open challenge.

Mean pad distance is 1.89 meters, and the on-pad fraction — landings within 0.5 meters of pad center — is only 2 percent.

The policy has clearly solved vertical control: it descends and touches down softly every time.

It has not solved lateral guidance: it doesn't consistently steer to the pad center.

This isn't an algorithmic failure — it reflects the genuine physical trade-off, where reducing throttle to descend simultaneously kills lateral authority.

---

## SLIDE 16 — Training Progression  [~0:45]

The training curve tells the story.

For the first million steps, mean reward sits around minus 3 — per-step costs dominate, and the policy hasn't learned to land reliably.

Around two million steps, reward crosses zero and starts climbing.

That crossover is the key diagnostic moment: it marks when the majority of episodes start ending in successful landings rather than timeouts or crashes.

From two to five million steps, reward climbs steadily to about plus 1.75 — softer touchdowns, smoother descent profiles, better throttle management.

The 100% landing rate is actually achieved relatively early; what continues to improve is landing quality.

---

## SLIDE 17 — Discussion: Lateral Accuracy Challenge  [~1:00]

Four factors contribute to the lateral-accuracy gap.

First, the throttle-fin coupling — lateral authority scales as throttle squared, so descent and steering compete.

Second, actuator bandwidth: the MG996R servos have a 50-millisecond time constant and a 7.54 radian-per-second slew limit, which becomes a real bottleneck during rapid final-approach corrections.

Third, the reward gradient at distance: the exponential pad bonus is essentially flat beyond two meters, and the linear horizontal penalty at weight minus 0.30 may simply be too small relative to the vertical-speed and delta-v terms.

Fourth, the curriculum: the narrow initial spawn box helps the policy learn touchdown but may not provide enough training signal for large lateral corrections from the full plus-or-minus 2-meter range.

Standard quadrotor RL recipes don't directly transfer because they assume throttle-decoupled attitude authority — TVC doesn't have that.

---

## SLIDE 18 — Sim-to-Real Considerations  [~1:00]

Before we deploy on hardware, we need to think about the sim-to-real gap.

Four main sources.

Aerodynamic fidelity: our flat-plate model with empirical corrections may miss flow separation, fin-fin wake interactions, and ground-effect changes during final approach.

Servo nonlinearities: real MG996Rs have backlash, load-dependent speed, and temperature drift — none captured by the first-order lag.

Battery voltage sag: under high throttle the 6S pack droops, reducing available thrust — currently unmodeled.

And sensor noise: simulation gives us clean state observations, but the physical testbed will run on a noisy IMU and an external position estimate.

The primary mitigation across all of these is domain randomization — randomize aero coefficients, servo parameters, mass properties, and sensor noise during training so the policy is robust to the uncertainty.

---

## SLIDE 19 — Future Work  [~1:00]

Five directions, in priority order.

First, replace the feed-forward MLP with a Gated Transformer-XL — Parisotto's GTrXL — to give the policy a recurrent context across the descent for wind rejection and trajectory planning.

The 24-dimensional observation space is already compatible with GTrXL's sequence input.

Second, domain randomization, as just discussed.

Third, improving lateral guidance through potential-based reward shaping in the sense of Ng, Harada, and Russell, and possibly hierarchical policies that separate lateral from vertical control.

Fourth, hardware validation on the physical testbed, to characterize the reality gap directly.

And fifth, training and evaluating under stochastic wind disturbances using the simulator's wind model.

---

## SLIDE 20 — Conclusion  [~0:45]

To summarize.

We built a GPU-accelerated Isaac Sim environment for an EDF thrust-vectored drone, calibrated to a 3.1-kilogram physical testbed.

A residual-PID PPO hover policy achieves 0.085 meters mean position error in just 2,048 steps.

A pure-PPO landing policy achieves a 100% landing rate at 0.126 meters per second mean touchdown speed over 5 million steps, with lateral accuracy — 1.89 meters mean pad distance — as the primary open challenge.

Two takeaways I'd ask you to remember: initialization priors on the action distribution are not optional for TVC, and terminal reward magnitudes must clearly dominate integrated per-step costs.

Code, training scripts, and evaluation logs are all on GitHub at the link shown.

Thank you 
