# Technical Review: Phase 1 Simulation Environment for EDF Jet-Vane Thrust-Vectoring Testbed

**Reviewer Role:** Simulation Expert / Technical Reviewer, JPL  
**Date:** March 22, 2026  
**Document Under Review:** Phase 1 Environment Design Proposal (Isaac Sim + Isaac Lab)  
**Associated Documents:** Research Proposal (Tang), Parts BOM, Jet-Vane TVC References (Chandra Murty & Chakraborty, 2015; Liu & Wang, 2013)

---

## 1. Executive Summary

The Phase 1 proposal describes the design and architecture of a simulation environment for a 6-DOF EDF drone with four jet-vane thrust-vectoring fins, built on Isaac Sim 5.1 + Isaac Lab 2.3 + PhysX. The environment is intended to support PID, PPO, and GTrXL-PPO controller evaluation for disturbance-resistant retro-propulsive landing.

**Overall assessment: The proposal is architecturally sound and well-reasoned, with several strong design decisions.** However, this review identifies **one critical versioning gap**, **several moderate issues** requiring resolution, and **a number of minor recommendations** that will strengthen the implementation.

---

## 2. Critical Finding: Version Selection Is Now Outdated

### 2.1 Isaac Lab Has Evolved Significantly Since the Proposal Was Written

The proposal recommends **Isaac Sim 5.1 + Isaac Lab 2.3 + PhysX**, which was the correct stable recommendation as of late 2025. However, as of March 2026, the Isaac Lab ecosystem has undergone major changes that directly affect this project:

**Isaac Lab 2.3.1 (stable, on Isaac Sim 5.1)** — This is the final release on the `main` branch. NVIDIA has stated this is the last release before shifting to the `develop` branch toward Isaac Lab 3.0. This release includes two features that are directly relevant:

- **WrenchComposer API** (PR #3287): Adds `permanent_wrench_composer()` and `instantaneous_wrench_composer()` for composing multiple forces and torques on the same body, with mixed local/global frame support. This is exactly the API the proposal recommends targeting. The old `set_external_force_and_torque()` is now deprecated in favor of `set_forces_and_torques()`.

- **Multirotor/Thruster Actuator** (PR #3760): Adds a `Multirotor` articulation class, `ThrusterCfg` actuators with asymmetric rise/fall dynamics, thrust allocation matrices, and `ThrustAction` MDP terms. This was built specifically for drone-class vehicles and represents a first-party pattern for the kind of force-based actuation this project needs.

**Isaac Lab 3.0 Beta (early access, on Isaac Sim 6.0)** — Announced at GTC 2026 just days ago. This is a ground-up architectural overhaul with multi-backend physics (PhysX and Newton 1.0), pluggable renderers, and Warp-native data pipelines. It is explicitly labeled as beta with breaking changes expected.

### 2.2 Recommendation

**Stay on Isaac Lab 2.3.1 + Isaac Sim 5.1 for Phase 1.** This is the correct decision for thesis-timeline stability. However, the implementation should:

1. **Use the WrenchComposer API from day one**, not the deprecated `set_external_force_and_torque()`. The proposal correctly identifies this migration but still shows code using the old pattern. The new API is available in 2.3.1 and supports `positions` arguments for applying forces at offset locations — exactly what fin-force-at-COP requires.

2. **Study the Multirotor/ThrusterCfg pattern** as architectural reference. While the EDF vehicle is not a standard multirotor (it has one thruster and four vanes, not four independent rotors), the allocation-matrix pattern, thrust rise/fall dynamics modeling, and the way the Multirotor class overrides `write_data_to_sim()` are directly instructive for how to structure the EDF actuation class.

3. **Do not adopt Isaac Lab 3.0 Beta** for Phase 1. The beta has stated instabilities, is Ubuntu-only, has incomplete Newton support, and involves breaking API changes. For a thesis with a March 2027 defense target, the risk of chasing a moving target is unacceptable.

---

## 3. Verification Against Official Documentation

### 3.1 Stack and Backend Claims — VERIFIED

The proposal's claims about Isaac Sim 5.1 conventions, PhysX as default backend, and Newton as experimental are confirmed by current official documentation. Specifically:

- Isaac Lab 2.3.0 is built on Isaac Sim 5.1 (confirmed in release notes and GitHub releases).
- PhysX remains the default and fully supported physics backend for rigid bodies, articulations, sensors, and contacts.
- Newton is described as experimental in Isaac Lab 3.0 Beta with "Not all environments have Newton presets yet" and "performance regressions may be observed."
- The `DirectRLEnv` workflow is confirmed as the correct pattern for custom tasks requiring direct control over actions, resets, rewards, and observations.

### 3.2 Frame Convention Claims — VERIFIED WITH CAVEAT

The proposal states Isaac Sim world uses **+X forward, +Z up** with **scalar-first quaternions (w,x,y,z)**. This is confirmed by the Isaac Sim 5.1 conventions documentation.

**Caveat:** The proposal's `frd_to_isaac_world()` helper assumes a specific body-to-world mapping (`y → -y, z → -z`). This is correct only if the USD asset's body frame aligns with the stated convention. The implementation must verify this by inspecting the actual USD prim orientation at rest. If the EDF asset was authored with a different local frame convention, the conversion will introduce systematic sign errors — exactly the class of bug this proposal is trying to prevent.

### 3.3 Articulation and Joint Claims — VERIFIED

The recommendation to model the vehicle as one articulated rigid body with four revolute fin links is consistent with Isaac Sim's articulation guidance. PhysX articulations are the documented preferred mechanism for jointed rigid-body systems. The USD scene hierarchy shown in the screenshot (`/Drone/Body/edf_drone` with `FwdFin`, `AftFin`, `LeftFin`, `RightFin` each containing `PhysicsRevolute` joints) is correctly structured.

### 3.4 Contact Sensor Claims — VERIFIED

Isaac Lab's ContactSensor provides persistent contact data, which is the correct choice for touchdown/landed/crash detection. The proposal correctly notes that old articulation link force sensors are deprecated.

**Important update from 2.3.1 release notes:** Friction force reporting has been added to ContactSensor (PR #3563). This may be useful for detecting sliding on the landing pad during post-touchdown dwell assessment.

### 3.5 Scene Cloning and Vectorization — VERIFIED

The `InteractiveSceneCfg(num_envs=128, replicate_physics=True)` pattern is confirmed in current documentation and tutorials. The proposal's 128-environment target is well within documented capabilities.

### 3.6 Visualization Markers — VERIFIED

Isaac Lab's `VisualizationMarkers` are confirmed as the recommended approach for debugging environment state. The proposal's gizmo plan (local axes, COM, thrust vector, per-fin forces, contact normals) aligns with documented capabilities.

---

## 4. Verification Against Jet-Vane TVC Research

### 4.1 Force Decomposition Model — WELL SUPPORTED

The proposal's core physics model — decomposing vane force into a **normal (control-producing) component** and a **tangential (thrust-loss) component** — is directly supported by both provided references:

- **Chandra Murty & Chakraborty (2015)** resolve vane forces into side force (control) and drag (thrust loss), confirming the normal/tangential decomposition. Their CFD results show approximately linear side force growth with vane angle up to about β/β_max ≈ 0.83, with stronger nonlinearity at larger angles.

- **Liu & Wang (2013)** explicitly decompose vane forces into F_n (normal) and F_t (tangential) in the body coordinate system, using the relations: F_n = F_TVCn·cos(δ) − F_TVCt·sin(δ) and F_t = F_TVCn·sin(δ) + F_TVCt·cos(δ). They also confirm that at zero vane deflection, normal force is zero while tangential drag persists due to vane thickness.

The proposal's semi-empirical model using C_L = C_Lα · δ · s(δ) with tanh saturation is a reasonable engineering approximation for Phase 1, provided it is later calibrated against thrust-stand data.

### 4.2 Torque from Geometry — CORRECT APPROACH

The proposal's key decision to derive body torque from **r × F** (force at position) rather than hand-coded roll/pitch/yaw torque channels is well-supported. Both references derive pitch/yaw control moments from side forces acting at specific radial offsets from the vehicle axis. The Liu & Wang paper explicitly models the pitching moment as F_n · (x_c − x_jc), where x_c and x_jc are the center-of-pressure and center-of-gravity positions — which is exactly the `r × F` formulation.

### 4.3 Engineering Correlation Structure — APPROPRIATE FOR PHASE 1

Chandra Murty's nonlinear regression model (F = P_c^a · δ_e^b / c) shows that engineering correlations can capture the dominant vane-angle-to-force mapping with errors under ~3.5% for moderate chamber pressures and vane angles. At lower pressures and extreme angles, errors increase to ~9%. For the EDF vehicle operating at a single "chamber pressure" (fixed fan RPM for a given throttle), the correlation degenerates to a simpler vane-angle-only function, which further simplifies Phase 1 modeling.

### 4.4 Concern: EDF vs. Rocket Exhaust Flow Regime

The provided references model jet vanes in **supersonic rocket exhaust** (Mach 3+, 3000K gas). The EDF drone operates in a fundamentally different flow regime — **subsonic, ambient-temperature airflow** (likely Mach 0.15–0.3 at the duct exit). This means:

- Oblique shock / expansion wave physics do not apply. The EDF vane operates in the subsonic aerodynamic regime.
- The linearized supersonic theory used by Liu & Wang (C_TVCy = 4δ / (57.3 · √(M² − 1))) is invalid for M < 1.
- The semi-empirical lift/drag model proposed (using C_Lα and C_D0 + kC_L²) is actually more appropriate for the subsonic EDF case than the supersonic theory in the references.

**Recommendation:** The proposal should explicitly acknowledge this regime mismatch and document that the jet-vane references are used for their force decomposition structure and geometric reasoning, not for their specific coefficient models. Phase 1 should use subsonic thin-airfoil approximations (C_Lα ≈ 2π for a flat plate, adjusted for aspect ratio and duct confinement), with calibration from thrust-stand testing.

---

## 5. Evaluation of Proposed Architecture

### 5.1 Module Structure — GOOD

The proposed code structure (common/, asset/, dynamics/, sim/, envs/, controllers/, tests/) provides clean separation of concerns. The shared physics core between single-env debug, unit tests, PID hover, and vectorized RL training is the correct architectural choice.

### 5.2 Frame Policy — EXCELLENT

The single-frame-contract approach (all control math in body FRD, single conversion to world frame at the application boundary) is the single most important architectural decision in the proposal. This will prevent the class of sign-error bugs that has plagued the existing branch.

### 5.3 Action Space Design — CORRECT

Exposing raw fin angles (4 continuous) plus optional throttle (1) as the environment action space, with PID mixing as an external adapter, is the right separation of concerns. This avoids baking PID assumptions into the RL environment.

### 5.4 Landed/Crashed State Machine — APPROPRIATE

The four-state model (AIRBORNE → GROUND_CONTACT_CANDIDATE → LANDED / CRASHED) with multi-frame dwell requirements is a sound design. The required conditions for LANDED (persistent contact + low velocities + bounded tilt + low angular rate) are physically reasonable.

### 5.5 Test Ladder — WELL ORDERED

The 13-step incremental test ladder correctly isolates failure modes from simplest to most complex. The ordering (asset → joint → single-fin → force sweep → superposition → motor effects → wind → contact → PID → hover → RL smoke) is the right way to de-risk the environment.

---

## 6. Issues Requiring Resolution

### 6.1 Moderate: Wrench Application Code Does Not Match Recommendation

The proposal recommends using the new WrenchComposer API but the provided Python code uses a manual `body_to_world(q, frd_to_isaac_world(F))` pattern and does not show how forces are actually dispatched to Isaac Lab. The implementation should use:

```python
# Per-fin force application via WrenchComposer
self.drone.instantaneous_wrench_composer.set_forces_and_torques(
    forces=fin_forces_world,
    torques=torch.zeros_like(fin_forces_world),
    positions=fin_cop_world,
    body_ids=fin_body_ids,
    env_ids=env_ids,
    is_global=True
)
```

This is the correct Isaac Lab 2.3.1 pattern and supports applying forces at arbitrary positions on specific body links.

### 6.2 Moderate: MG996R Servo Model Needs Quantitative Parameters

The parts list warns about the MG996R impact but the proposal does not provide specific parameter values for the simulation. The implementation needs:

| Parameter | Value | Source |
|-----------|-------|--------|
| Mass per servo | ~55g (0.055 kg) | MG996R datasheet |
| Stall torque @6V | ~11 kgf·cm (1.08 N·m) | MG996R datasheet |
| Transit time (60°) @6V | ~0.14 s | MG996R datasheet |
| Max angular velocity | ~430 °/s (~7.5 rad/s) | Derived from transit time |
| τ_servo (first-order lag) | ~0.04–0.07 s | Estimated from step response |
| Dead band | ~5 μs (~1–2°) | MG996R datasheet |

The dead band is particularly important for RL training — it creates a non-responsive zone that affects fine control authority and should be modeled.

### 6.3 Moderate: EDF Motor Parameters Not Specified

The proposal describes the motor dynamics equations but does not provide candidate values for:

- k_T (thrust coefficient): Should be derived from static thrust (~48N) and estimated max RPM
- k_Q (torque coefficient): Typically k_Q ≈ k_T / (2π · efficiency) for EDFs
- I_fan (rotor inertia): Needs to be measured or estimated from the FMS 90mm fan geometry
- τ_motor (spool lag): Typically 0.1–0.3s for 90mm class EDFs
- ω_max and dω_max: Derived from ESC + motor characteristics

Without these values, the simulation cannot produce physically meaningful behavior.

### 6.4 Minor: Missing Duct Confinement Effects

The vane aerodynamic model treats each fin independently in a uniform exhaust stream. In reality, the EDF duct constrains the flow, creating:

- Non-uniform velocity profile across the duct cross-section (higher velocity near the duct wall)
- Vane-to-vane interference when adjacent vanes deflect simultaneously
- Duct wall interaction at high deflection angles

For Phase 1, ignoring these effects is acceptable, but they should be documented as known modeling limitations and candidates for Phase 2 calibration.

### 6.5 Minor: Gyroscopic Precession Magnitude

The proposal includes gyroscopic precession torque (τ_gyro = ω_body × H), which is correct. However, for a 90mm EDF with relatively low rotor inertia compared to the vehicle's moment of inertia, the precession torque may be small relative to fin-generated torques. The implementation should include it but should also log its magnitude relative to other torque sources during validation so that it can be assessed whether it materially affects control behavior.

### 6.6 Minor: Wind Disturbance Model Unspecified

The proposal includes wind as an "optional toggle" but does not specify the wind model. For the research proposal's focus on disturbance resistance, this needs definition. A reasonable Phase 1 model would include:

- Steady-state wind vector (configurable direction and magnitude)
- Gust model (e.g., Dryden or simple step/ramp disturbances)
- Application as body force F_wind = 0.5 · ρ · C_D · A_ref · |V_wind − V_body|² · direction

---

## 7. Alignment with Research Proposal

The Phase 1 environment design is well-aligned with the research proposal's scope and timeline:

- The proposal targets "Construct Isaac Sim Environment" by March 2026 and "Construct Artifact" by July 2026. Phase 1 environment work fits within this window.
- The research questions (sim-to-hardware fidelity, disturbance robustness, controller comparison) all require the kind of simulation environment Phase 1 describes.
- The 128-env vectorization target supports the training scale needed for PPO and GTrXL-PPO.
- The shared physics core with task-configurable rewards (hover + landing) supports the incremental validation path from simulation to HIL.

**One concern:** The research proposal mentions MATLAB Simulink for HIL integration. The Phase 1 environment is built entirely in Python/PyTorch within Isaac Lab. The HIL bridge between Isaac Lab and Simulink (or between the Jetson Nano flight controller and the sim environment) is not addressed in Phase 1. This should be scoped as a separate integration task.

---

## 8. Summary of Findings

| Category | Finding | Severity |
|----------|---------|----------|
| Version selection | Isaac Lab 2.3.1 is correct for stability; use WrenchComposer API | Critical (action required) |
| Wrench API | Code examples use deprecated pattern | Moderate |
| Servo parameters | MG996R values not quantified in sim config | Moderate |
| Motor parameters | EDF thrust/torque coefficients unspecified | Moderate |
| Flow regime | Supersonic reference theory inapplicable; subsonic model needed | Moderate |
| Frame policy | Excellent — single FRD contract | Strength |
| Force-from-geometry | Correct — torque from r × F, not hand-coded mixer | Strength |
| Test ladder | Well-ordered incremental validation | Strength |
| Architecture | Clean module separation, shared physics core | Strength |
| Action space | Raw fin angles for RL, mixer only for PID | Strength |
| Wind model | Unspecified | Minor |
| Duct confinement | Ignored (acceptable for Phase 1) | Minor |
| HIL bridge | Not addressed | Minor (out of Phase 1 scope) |

---

## 9. Recommendation

**Approve Phase 1 with required modifications.** The proposal demonstrates strong engineering judgment in its core architectural decisions — the frame contract, force-from-geometry approach, articulation-based modeling, and incremental test ladder. The principal reviewer concerns are:

1. Migrate all wrench application code to the WrenchComposer API before implementation begins.
2. Populate all actuator and motor model parameters with quantitative values from datasheets and measurements.
3. Document the subsonic flow regime explicitly and use appropriate aerodynamic coefficients.
4. Study the Isaac Lab 2.3.1 Multirotor/ThrusterCfg pattern as architectural reference for the EDF actuation class.

With these modifications, Phase 1 is well-positioned to deliver a correct, debuggable, and vectorizable simulation environment suitable for the research proposal's PID, PPO, and GTrXL-PPO experiments.

---

*This review was conducted against Isaac Lab 2.3.1 release notes, Isaac Sim 5.1 documentation, Isaac Lab 3.0 Beta release notes, the WrenchComposer PR (#3287), and the Multirotor/Thruster PR (#3760), as well as the provided jet-vane TVC references.*
