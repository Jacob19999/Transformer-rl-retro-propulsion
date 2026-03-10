"""
edf_landing_task.py — IsaacLab DirectRLEnv task for EDF retro-propulsive landing.

Implements T012-T020:
  T012-T013: EdfLandingTaskCfg dataclass (scene, sim, landing pad)
  T014:      EdfLandingTask._setup_scene()
  T015:      EdfLandingTask._reset_idx()
  T016:      Physics lag state tensors (thrust, fin deflections)
  T017:      EdfLandingTask._pre_physics_step() — force injection
  T018:      EdfLandingTask._get_observations() — 20-dim obs vector
  T019:      EdfLandingTask._get_rewards()
  T020:      EdfLandingTask._get_dones()

Contract references:
  specs/001-isaac-sim-env/contracts/usd-asset-schema.md
  specs/001-isaac-sim-env/data-model.md
"""

from __future__ import annotations

import math
import sys
from dataclasses import MISSING, dataclass, field
from pathlib import Path
from typing import Any

import torch

# ---------------------------------------------------------------------------
# IsaacLab 2.3 imports
# ---------------------------------------------------------------------------
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils import configclass

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from simulation.config_loader import load_config        # noqa: E402
from simulation.training.reward import RewardConfig, RewardFunction  # noqa: E402

# ---------------------------------------------------------------------------
# Physical constants from vehicle YAML (loaded once at import)
# ---------------------------------------------------------------------------
_VEHICLE_YAML = REPO_ROOT / "simulation" / "configs" / "default_vehicle.yaml"
_REWARD_YAML  = REPO_ROOT / "simulation" / "configs" / "reward.yaml"

_T_MAX       = 45.0       # N, EDF max thrust
_DELTA_MAX   = 0.2618     # rad, ±15° fin control limit
_TAU_MOTOR   = 0.10       # s, EDF first-order thrust lag
_TAU_SERVO   = 0.04       # s, servo first-order position lag
_K_THRUST    = 4.55e-7    # N/(rad/s)²
_V_EXHAUST   = 70.0       # m/s nominal
_CL_ALPHA    = 6.283      # /rad, NACA0012 thin-airfoil
_FIN_AREA    = 0.003575   # m² per fin, chord × span
_GRAVITY     = 9.81       # m/s²

# Fin hinge positions in FRD body frame (m) — for fin force torque arms
_FIN_POS = torch.tensor(
    [[0.0,  0.055, 0.14],
     [0.0, -0.055, 0.14],
     [0.055, 0.0,  0.14],
     [-0.055, 0.0, 0.14]],
    dtype=torch.float32,
)  # (4, 3)

# Fin lift directions in body frame
_FIN_LIFT_DIR = torch.tensor(
    [[1.0, 0.0, 0.0],
     [1.0, 0.0, 0.0],
     [0.0, 1.0, 0.0],
     [0.0, 1.0, 0.0]],
    dtype=torch.float32,
)  # (4, 3)


# ---------------------------------------------------------------------------
# Scene configuration
# ---------------------------------------------------------------------------
@configclass
class EdfSceneCfg(InteractiveSceneCfg):
    """Declarative scene: flat ground plane + one drone articulation."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        physics_material=RigidBodyMaterialCfg(
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.1,
        ),
    )

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Drone",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(REPO_ROOT / "simulation" / "isaac" / "usd" / "drone.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 7.5, 0.0),  # Y-up: start 7.5 m above ground
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
    )

    # Number of envs is set by the parent DirectRLEnvCfg
    num_envs: int = MISSING
    env_spacing: float = 4.0


# ---------------------------------------------------------------------------
# Task configuration (T012-T013)
# ---------------------------------------------------------------------------
@configclass
class EdfLandingTaskCfg(DirectRLEnvCfg):
    """Configuration for EDF landing task."""

    # RL spaces
    observation_space: int = 20
    action_space: int = 5
    state_space: int = 0

    # Physics step: 1/120 s, 4 substeps (eff. dt ≈ 2.08 ms)
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=1,
        physx=sim_utils.PhysxCfg(
            num_position_iterations=8,
            num_velocity_iterations=1,
            bounce_threshold_velocity=0.2,
            max_depenetration_velocity=1.0,
        ),
    )

    # Policy step = 1 physics step
    decimation: int = 1

    # Episode parameters
    episode_length_s: float = 5.0   # 5 s → 600 steps at 1/120

    # Scene
    scene: EdfSceneCfg = EdfSceneCfg(num_envs=MISSING, env_spacing=4.0)

    # Spawn ranges
    spawn_altitude_min: float = 5.0
    spawn_altitude_max: float = 10.0
    spawn_vel_mag_min: float = 0.0
    spawn_vel_mag_max: float = 5.0

    # Landing target in world frame (Y-up: y=0 is ground level)
    target_pos_world: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Crash / landing thresholds
    crash_velocity_threshold: float = 3.0   # m/s, max |v| for successful landing
    landing_pad_radius: float = 0.5         # m

    # Vehicle / reward config paths
    vehicle_config_path: str = str(REPO_ROOT / "simulation" / "configs" / "default_vehicle.yaml")
    reward_config_path: str  = str(REPO_ROOT / "simulation" / "configs" / "reward.yaml")


# ---------------------------------------------------------------------------
# Task implementation (T014-T020)
# ---------------------------------------------------------------------------
class EdfLandingTask(DirectRLEnv):
    """EDF retro-propulsive landing task for IsaacLab 2.3."""

    cfg: EdfLandingTaskCfg

    def __init__(self, cfg: EdfLandingTaskCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Physics dt
        self._dt = cfg.sim.dt  # 1/120 s

        # Load reward config
        reward_cfg_raw = load_config(cfg.reward_config_path)
        self._reward_cfg = RewardConfig.from_config(reward_cfg_raw.get("reward", reward_cfg_raw))

        # Target position tensor (Y-up world frame)
        tp = cfg.target_pos_world
        self._target_pos_w = torch.tensor(
            [tp[0], tp[1], tp[2]], dtype=torch.float32, device=self.device
        ).unsqueeze(0).expand(self.num_envs, -1)  # (num_envs, 3)

        # T016: Physics lag state tensors
        self.thrust_actual = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.fin_deflections_actual = torch.zeros(
            (self.num_envs, 4), dtype=torch.float32, device=self.device
        )

        # Episode step counter
        self._episode_step = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # Max steps per episode
        self._max_steps = int(cfg.episode_length_s / cfg.sim.dt)

        # Move fin geometry tensors to device
        self._fin_pos = _FIN_POS.to(self.device)        # (4, 3)
        self._fin_lift = _FIN_LIFT_DIR.to(self.device)  # (4, 3)

        # Vehicle mass (for twr obs)
        vehicle_cfg = load_config(cfg.vehicle_config_path)
        vehicle_data = vehicle_cfg.get("vehicle", vehicle_cfg)
        from simulation.dynamics.mass_properties import compute_mass_properties
        mp = compute_mass_properties(vehicle_data["primitives"])
        self._mass = float(mp.total_mass)
        self._weight = self._mass * _GRAVITY

        # Wind EMA (zero — no wind model in Isaac Sim)
        self._wind_ema = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

        # Previous action (for smoothness reward)
        self._prev_action = torch.zeros(
            (self.num_envs, 5), dtype=torch.float32, device=self.device
        )

    # ------------------------------------------------------------------
    # T014: Scene setup
    # ------------------------------------------------------------------
    def _setup_scene(self) -> None:
        """Instantiate the interactive scene."""
        from isaaclab.scene import InteractiveScene
        self.scene = InteractiveScene(self.cfg.scene)
        self.robot: Articulation = self.scene["robot"]

        # Register scene entities for cloning / replication
        self._terrain = self.scene.terrain

        # Add articulation to sim
        self.sim.reset()

    # ------------------------------------------------------------------
    # T015: Reset per env indices
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        """Randomize spawn state for selected environments."""
        n = len(env_ids)
        if n == 0:
            return

        # Reset lag state
        self.thrust_actual[env_ids] = 0.0
        self.fin_deflections_actual[env_ids] = 0.0
        self._episode_step[env_ids] = 0
        self._prev_action[env_ids] = 0.0
        self._wind_ema[env_ids] = 0.0

        # Sample random altitude and velocity
        alt = torch.zeros(n, device=self.device).uniform_(
            self.cfg.spawn_altitude_min, self.cfg.spawn_altitude_max
        )
        vel_mag = torch.zeros(n, device=self.device).uniform_(
            self.cfg.spawn_vel_mag_min, self.cfg.spawn_vel_mag_max
        )

        # Random velocity direction
        theta = torch.zeros(n, device=self.device).uniform_(0.0, 2.0 * math.pi)
        phi   = torch.zeros(n, device=self.device).uniform_(0.0, math.pi)
        vx = vel_mag * torch.sin(phi) * torch.cos(theta)
        vy = vel_mag * torch.cos(phi)
        vz = vel_mag * torch.sin(phi) * torch.sin(theta)
        vel = torch.stack([vx, vy, vz], dim=-1)  # (n, 3)

        # Root pose: position at random altitude (Y-up: altitude = Y)
        # Place in env-local frame (Isaac handles per-env offsets automatically)
        pos = torch.zeros((n, 3), device=self.device)
        pos[:, 1] = alt  # Y = altitude in Y-up world

        # Identity quaternion [x, y, z, w] scalar-last
        quat = torch.zeros((n, 4), device=self.device)
        quat[:, 3] = 1.0  # w=1 → identity

        # Write root pose and velocity
        root_pose = torch.cat([pos, quat], dim=-1)  # (n, 7)
        root_vel  = torch.cat([vel, torch.zeros((n, 3), device=self.device)], dim=-1)  # (n, 6)

        self.robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(root_vel, env_ids=env_ids)

        # Reset joints to neutral
        joint_pos = torch.zeros((n, self.robot.num_joints), device=self.device)
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    # ------------------------------------------------------------------
    # T017: Pre-physics step — force injection
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Apply EDF thrust + fin aero forces before each PhysX step."""
        dt = self._dt

        # Clip actions to [-1, 1]
        actions = actions.clamp(-1.0, 1.0)
        self._prev_action = actions.clone()

        # Unpack: [thrust_cmd, d1, d2, d3, d4]
        thrust_cmd_norm = actions[:, 0]          # [-1, 1] → [0, T_MAX]
        fin_cmd_norm    = actions[:, 1:5]        # [-1, 1] → [-DELTA_MAX, DELTA_MAX]

        # Map thrust: normalize so [-1,0] → 0, [0,1] → T_MAX
        T_cmd = (thrust_cmd_norm.clamp(0.0, 1.0)) * _T_MAX
        delta_cmd = fin_cmd_norm * _DELTA_MAX

        # First-order lag update
        self.thrust_actual += (dt / _TAU_MOTOR) * (T_cmd - self.thrust_actual)
        self.fin_deflections_actual += (dt / _TAU_SERVO) * (
            delta_cmd - self.fin_deflections_actual
        )
        # Clamp fins to physical limit
        self.fin_deflections_actual.clamp_(-_DELTA_MAX, _DELTA_MAX)

        # ----------------------------------------------------------------
        # EDF thrust force in body frame (FRD: +Z is thrust direction)
        # Y-up mapping: body +Z → world +Y (after 90° X-rotation at drone root)
        # In Isaac body frame after rotation, thrust still acts in local +Y?
        # We apply forces in body-local frame; IsaacLab handles world transform.
        # Body FRD +Z is aligned → use local (0, 0, 1) in FRD
        # ----------------------------------------------------------------
        num_envs = self.num_envs
        # Force tensor shape: (num_envs, num_bodies, 3)
        forces  = torch.zeros((num_envs, 1, 3), device=self.device)
        torques = torch.zeros((num_envs, 1, 3), device=self.device)

        # Thrust along body +Z (FRD)
        forces[:, 0, 2] = self.thrust_actual

        # ----------------------------------------------------------------
        # Fin aerodynamic forces (NACA0012 in exhaust stream)
        # ----------------------------------------------------------------
        # Exhaust velocity scales with thrust
        omega_ratio = (self.thrust_actual / _T_MAX).clamp(0.0, 1.0).sqrt()
        V_ex = omega_ratio * _V_EXHAUST  # (num_envs,)

        # Lift per fin: L_i = 0.5 * rho_0 * V_ex² * A * Cl_alpha * delta_i
        # Use sea-level rho = 1.225 kg/m³
        rho = 1.225
        dyn_pressure = 0.5 * rho * V_ex.pow(2)  # (num_envs,)

        for i in range(4):
            delta_i = self.fin_deflections_actual[:, i]      # (num_envs,)
            L_i = dyn_pressure * _FIN_AREA * _CL_ALPHA * delta_i  # (num_envs,)
            lift_dir = self._fin_lift[i]                     # (3,)
            # Force contribution
            forces[:, 0, :] += L_i.unsqueeze(-1) * lift_dir.unsqueeze(0)
            # Torque = r × F
            r = self._fin_pos[i]  # (3,)
            F_i = L_i.unsqueeze(-1) * lift_dir.unsqueeze(0)  # (num_envs, 3)
            tau_i = torch.linalg.cross(r.unsqueeze(0).expand(num_envs, -1), F_i)
            torques[:, 0, :] += tau_i

        # Apply to simulation
        self.robot.set_external_force_and_torque(forces, torques, body_ids=[0])
        self.robot.write_data_to_sim()

        # Increment episode step
        self._episode_step += 1

    # ------------------------------------------------------------------
    # T018: Observations — 20-dim vector
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Compute 20-dim observation matching observation.py layout."""
        data = self.robot.data

        # World-frame position (Y-up)
        pos_w  = data.root_pos_w    # (num_envs, 3): [x_world, y_world, z_world]
        quat_w = data.root_quat_w   # (num_envs, 4): [qx, qy, qz, qw] scalar-last
        vel_b  = data.root_lin_vel_b  # (num_envs, 3): body-frame linear vel
        omega_b = data.root_ang_vel_b  # (num_envs, 3): body-frame angular vel

        # --- Altitude above ground (Y-up: Y = altitude) ---
        h_agl = pos_w[:, 1].clamp(min=0.0)  # (num_envs,)

        # --- Position error in world frame (to target) ---
        target_w = self._target_pos_w  # (num_envs, 3)
        err_w = target_w - pos_w      # (num_envs, 3)

        # --- Rotate error into body frame ---
        # q_world_body = root_quat_w → rotate world vector to body
        err_b = _rotate_world_to_body(err_w, quat_w)  # (num_envs, 3)

        # --- Gravity direction in body frame ---
        # World gravity is -Y in Y-up frame
        g_world = torch.zeros((self.num_envs, 3), device=self.device)
        g_world[:, 1] = -1.0  # unit vector pointing down in Y-up
        g_b = _rotate_world_to_body(g_world, quat_w)  # (num_envs, 3)

        # --- Thrust-to-weight ratio ---
        twr = (self.thrust_actual / self._weight).unsqueeze(-1)  # (num_envs, 1)

        # --- Scalars ---
        speed     = vel_b.norm(dim=-1, keepdim=True)     # (num_envs, 1)
        ang_speed = omega_b.norm(dim=-1, keepdim=True)   # (num_envs, 1)
        time_frac = (self._episode_step.float() / self._max_steps).clamp(0.0, 1.0).unsqueeze(-1)

        # --- Assemble 20-dim obs ---
        # [0:3]=e_p_body, [3:6]=v_body, [6:9]=g_body, [9:12]=omega,
        # [12]=twr, [13:16]=wind_ema(zeros), [16]=h_agl, [17]=speed,
        # [18]=ang_speed, [19]=time_frac
        obs = torch.cat([
            err_b,                   # 3
            vel_b,                   # 3
            g_b,                     # 3
            omega_b,                 # 3
            twr,                     # 1
            self._wind_ema,          # 3  (zeros)
            h_agl.unsqueeze(-1),     # 1
            speed,                   # 1
            ang_speed,               # 1
            time_frac,               # 1
        ], dim=-1)  # (num_envs, 20)

        return {"policy": obs}

    # ------------------------------------------------------------------
    # T019: Rewards — port from reward.py
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        """Compute shaped + terminal rewards. Returns (num_envs,) float32 tensor."""
        cfg = self._reward_cfg
        data = self.robot.data

        pos_w   = data.root_pos_w
        vel_b   = data.root_lin_vel_b
        omega_b = data.root_ang_vel_b
        quat_w  = data.root_quat_w

        h_agl = pos_w[:, 1].clamp(min=0.0)
        speed = vel_b.norm(dim=-1)

        # Distance to target
        dist = (self._target_pos_w - pos_w).norm(dim=-1)

        # Alive bonus
        r = torch.full((self.num_envs,), cfg.alive_bonus, device=self.device)

        # Shaping: distance + velocity
        r -= cfg.shaping_distance_coeff * dist
        r -= cfg.shaping_velocity_coeff * speed

        # Orientation penalty: deviation of gravity from body -Z (FRD down)
        g_world = torch.zeros((self.num_envs, 3), device=self.device)
        g_world[:, 1] = -1.0
        g_b = _rotate_world_to_body(g_world, quat_w)
        # ideal g_body = [0, 0, 1] in FRD (gravity points +Z down)
        cos_theta = g_b[:, 2].clamp(-1.0, 1.0)
        tilt_penalty = 1.0 - cos_theta  # 0 when upright
        r -= cfg.orientation_weight * tilt_penalty

        # Action smoothness penalty
        delta_action = (self._prev_action - torch.zeros_like(self._prev_action)).norm(dim=-1)
        r -= cfg.action_smooth_weight * delta_action

        # Terminal rewards are handled in _get_dones / caller
        return r

    # ------------------------------------------------------------------
    # T020: Dones
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (terminated, truncated) tensors."""
        data = self.robot.data
        pos_w = data.root_pos_w
        vel_b = data.root_lin_vel_b

        h_agl  = pos_w[:, 1]
        speed  = vel_b.norm(dim=-1)

        # Lateral distance from target
        target = self._target_pos_w
        lateral_dist = ((pos_w[:, 0] - target[:, 0]).pow(2)
                        + (pos_w[:, 2] - target[:, 2]).pow(2)).sqrt()

        # Crash: below ground with high speed
        crashed = (h_agl < 0.05) & (speed > self.cfg.crash_velocity_threshold)

        # Landed: near ground, low speed, within pad radius
        landed = (
            (h_agl < 0.05)
            & (speed <= self.cfg.crash_velocity_threshold)
            & (lateral_dist <= self.cfg.landing_pad_radius)
        )

        terminated = crashed | landed
        truncated  = self._episode_step >= self._max_steps

        return terminated, truncated

    # ------------------------------------------------------------------
    # DirectRLEnv override: apply actions before physics
    # ------------------------------------------------------------------
    def _apply_action(self) -> None:
        """Called by DirectRLEnv before sim.step(). Delegates to _pre_physics_step."""
        # actions are stored in self.actions by the parent class
        self._pre_physics_step(self.actions)


# ---------------------------------------------------------------------------
# Quaternion utility: rotate world vector to body frame
# ---------------------------------------------------------------------------
def _rotate_world_to_body(v_world: torch.Tensor, quat_w: torch.Tensor) -> torch.Tensor:
    """Rotate (num_envs, 3) world vectors into body frame using scalar-last quaternion.

    q = [qx, qy, qz, qw] (scalar-last)
    v_body = q_conj ⊗ v ⊗ q
    """
    qx = quat_w[:, 0]
    qy = quat_w[:, 1]
    qz = quat_w[:, 2]
    qw = quat_w[:, 3]

    # Conjugate rotate: q_conj = [-qx, -qy, -qz, qw]
    # Rotation: v' = v + 2*qw*(q_vec × v) + 2*(q_vec × (q_vec × v))
    q_vec = torch.stack([-qx, -qy, -qz], dim=-1)  # (n, 3) — conjugate imaginary
    # Actually use forward rotation with conjugate:
    # For q acting on world→body: apply q_conj to v
    # v' = v + 2w (q × v) + 2 (q × (q × v))  where q is the conjugate imaginary part
    qw_c = qw            # scalar part of conjugate = qw
    q_c  = torch.stack([-qx, -qy, -qz], dim=-1)  # imaginary part of conjugate

    t = 2.0 * torch.linalg.cross(q_c, v_world)        # (n, 3)
    v_body = v_world + qw_c.unsqueeze(-1) * t + torch.linalg.cross(q_c, t)
    return v_body
