"""
Add speaker notes to each slide of the generated presentation.
"""

from pptx import Presentation
import os

pptx_path = os.path.join(os.path.dirname(__file__), "TVC_RL_Presentation.pptx")
prs = Presentation(pptx_path)

speaker_notes = {
    0: (  # Slide 1 — Title
        "Good morning/afternoon everyone. My name is Jacob Tang from Minnesota State University, Mankato. "
        "Today I'll be presenting our work on using reinforcement learning — specifically Proximal Policy Optimization — "
        "to control a thrust-vectored electric ducted fan drone in simulation. "
        "This is a vehicle that steers not by spinning multiple rotors at different speeds like a typical quadcopter, "
        "but by deflecting small fins inside its own exhaust stream. "
        "It's a fundamentally different and much harder control problem, and we'll show how deep RL can tackle it.\n\n"
        "[~30 seconds]"
    ),
    1: (  # Slide 2 — Outline
        "Here's our roadmap for the next 20 minutes. We'll start with why this problem matters and what makes it hard. "
        "Then we'll walk through the vehicle itself, the simulation environment we built in NVIDIA Isaac Sim, "
        "the physics models under the hood, our PPO training setup and reward design, and then the results for both hover and landing tasks. "
        "We'll wrap up with the open challenges and where this work is headed. "
        "The key question we're answering: can deep RL learn to control a vehicle that steers by deflecting fins in its own exhaust stream? "
        "Spoiler — yes, but with some interesting caveats.\n\n"
        "[~30 seconds]"
    ),
    2: (  # Slide 3 — Motivation
        "So why are we doing this? Most drones you see today are multirotors — quadcopters, hexacopters. "
        "They control attitude by spinning each rotor at a different speed. It's elegant, well-understood, and there's a ton of RL work on it.\n\n"
        "But there's another class of vehicles that use thrust vector control — a single engine, and you steer by redirecting the exhaust. "
        "Think SpaceX Falcon 9 grid fins, or old-school rocket jet vanes. "
        "Our vehicle does exactly this: one electric ducted fan, four flat-plate fins sitting in the exhaust stream.\n\n"
        "The fundamental challenge is this coupling: fin force is proportional to throttle squared. "
        "So when you reduce throttle to descend, you simultaneously lose your ability to steer laterally. "
        "At 50% throttle, you only have 25% of your maximum control authority. "
        "This is a deeply nonlinear, tightly coupled control problem. "
        "Classical PID or LQR controllers work fine near hover, but they struggle across the full flight envelope. "
        "That's exactly where RL shines — learning the full nonlinear mapping end-to-end.\n\n"
        "[~1.5 minutes]"
    ),
    3: (  # Slide 4 — Related Work
        "Briefly on related work. RL for aerial vehicles has made great progress — "
        "Hwangbo showed PPO can learn agile quadrotor control, Koch compared multiple RL algorithms for attitude control, "
        "and GPU-parallel simulation from Isaac Gym has been a game-changer for training speed.\n\n"
        "On the thrust vector control side, there's a long history in rocketry, but applying RL to jet-vane TVC for VTOL vehicles is largely unexplored. "
        "Classical approaches linearize around hover and apply PID or LQR, which limits the operating envelope.\n\n"
        "For sim-to-real, domain randomization and system identification are the standard tools. "
        "The fidelity of the dynamics model is critical, which is why we put significant effort into our semi-empirical models.\n\n"
        "The research gap is clear: RL for thrust-vectored fin control of EDF vehicles, "
        "with the unique challenges of coupled throttle-authority, hobby-grade actuators, and gyroscopic effects.\n\n"
        "[~1 minute]"
    ),
    4: (  # Slide 5 — Vehicle Platform
        "Here's our testbed. It's a custom-built 3.1 kilogram EDF drone. "
        "Single FMS 90-millimeter 12-blade ducted fan, powered by a 6S LiPo battery. "
        "Four MG996R hobby servos actuate flat-plate jet vanes in the exhaust stream, arranged in a cruciform pattern.\n\n"
        "The control architecture gives us full 6-DOF authority through just 5 actuator commands: "
        "four fin angles for pitch, roll, and yaw, plus one throttle for altitude. "
        "The forward and aft fins handle pitch, left and right handle roll, and differential deflection gives us yaw. "
        "It's mechanically simpler than a quadrotor in some ways, but the control problem is significantly harder.\n\n"
        "[~1 minute]"
    ),
    5: (  # Slide 6 — Simulation Environment
        "We built our simulation on NVIDIA Isaac Sim using the Isaac Lab framework. "
        "The key enabler is GPU-accelerated parallel simulation — we run 128 copies of the drone simultaneously on a single GPU. "
        "Physics runs at 120 Hz, and the RL agent makes decisions at 30 Hz with a decimation factor of 4, "
        "meaning four physics substeps per RL step.\n\n"
        "The environment implements a standard Gymnasium interface, so it plugs directly into any RL training framework. "
        "All the aerodynamic forces, propulsion dynamics, and actuator models are updated at the full 120 Hz physics rate, "
        "not just at the RL decision rate. This matters for capturing the servo dynamics and spool-up behavior accurately.\n\n"
        "[~1 minute]"
    ),
    6: (  # Slide 7 — Propulsion Model
        "The EDF propulsion model has two key components. First, first-order spool dynamics — "
        "the motor doesn't instantly reach commanded speed, it has a time constant of 0.15 seconds. "
        "Second, quadratic thrust — thrust scales with the square of rotor speed.\n\n"
        "What makes this interesting are the three torque components. "
        "Static reaction torque opposes the rotor spin. "
        "Dynamic spool torque is the reaction to rotor acceleration — when you throttle up quickly, the body wants to spin the other way. "
        "And gyroscopic precession couples body rotation to rotor angular momentum. "
        "This last one is particularly important: if the vehicle pitches while the rotor is spinning at 4300 rad/s, you get a yaw torque. "
        "These are real physical effects that make the control problem non-trivial and that a good simulation must capture.\n\n"
        "[~1 minute]"
    ),
    7: (  # Slide 8 — Fin Aerodynamic Model
        "The fin aerodynamic model is semi-empirical. Each fin sees a dynamic pressure that depends on exhaust velocity and throttle squared — "
        "there's that coupling again. We use a lift-curve slope of 3.5 per radian from thin-airfoil theory with empirical correction, "
        "plus a stall saturation term to capture the nonlinear behavior at large deflections.\n\n"
        "The critical point I want to emphasize is at the bottom: this quadratic coupling between throttle and fin effectiveness "
        "is THE defining characteristic of jet-vane TVC. It's what makes this fundamentally different from a quadrotor, "
        "where each rotor provides independent, throttle-decoupled control authority. "
        "When our vehicle descends and reduces throttle, it simultaneously loses its ability to correct lateral errors. "
        "This is the core tension the RL agent must learn to navigate.\n\n"
        "[~1 minute]"
    ),
    8: (  # Slide 9 — PPO Algorithm
        "We use standard PPO with clipped surrogate objectives. For those less familiar, "
        "PPO is an on-policy actor-critic algorithm that constrains how much the policy can change in each update — "
        "preventing catastrophically large gradient steps while maintaining good sample efficiency. "
        "It's become the workhorse of continuous control RL.\n\n"
        "Our setup uses GAE for advantage estimation with gamma 0.99 and lambda 0.95, "
        "plus an adaptive KL early stopping criterion as an additional safeguard. "
        "The hyperparameters on the right are fairly standard for continuous control tasks. "
        "We chose PPO specifically because prior work by Koch and Hwangbo showed it provides robust training stability for aerial vehicle control.\n\n"
        "[~1 minute]"
    ),
    9: (  # Slide 10 — Observation & Action Space
        "The observation space is 24-dimensional. Position error to the target, attitude as a quaternion to avoid gimbal lock, "
        "linear and angular velocities in the body frame, height above ground, all four fin angles and their rates, "
        "normalized motor RPM, and contact state.\n\n"
        "Two design choices worth highlighting: we include fin angles and rates as proprioceptive feedback "
        "because the servo lag is significant — the policy needs to know where the fins actually are, not just where it commanded them. "
        "And we include normalized motor RPM so the policy knows how much fin authority it currently has, given that coupling.\n\n"
        "The action space is just 5 dimensions — four fin deflections and one throttle. "
        "That's actually smaller than a quadrotor's 4 rotor speeds, which is a nice property of this architecture.\n\n"
        "The network is straightforward: separate actor and critic, each with two hidden layers of 256 units and tanh activations.\n\n"
        "[~1 minute]"
    ),
