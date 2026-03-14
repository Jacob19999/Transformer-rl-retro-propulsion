"""
edf_landing_task.py -- IsaacLab DirectRLEnv task for EDF retro-propulsive landing.

Implements T012-T020:
  T012-T013: EdfLandingTaskCfg dataclass (scene, sim, landing pad)
  T014:      EdfLandingTask._setup_scene()
  T015:      EdfLandingTask._reset_idx()
  T016:      Physics lag state tensors (thrust, fin deflections)
  T017:      EdfLandingTask._pre_physics_step() -- force injection
  T018:      EdfLandingTask._get_observations() -- 20-dim obs vector
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
# IsaacLab source-path bootstrap.
#
# The pip-installed `isaaclab` package ships its Python source at
#   <site-packages>/isaaclab/source/isaaclab/
# but its top-level __init__.py only exposes `isaaclab.app`.
# We locate the source tree via find_spec (no import side-effects) and
# prepend it to sys.path so that `import isaaclab.sim` etc. resolve correctly.
# ---------------------------------------------------------------------------
def _bootstrap_isaaclab_path() -> None:
    import importlib.util as _ilu
    spec = _ilu.find_spec("isaaclab")
    if spec is None:
        return
    pkg_root = Path(spec.origin).parent  # .../site-packages/isaaclab/
    # isaaclab source + isaaclab_contrib source both live under source/
    for subdir in ("isaaclab", "isaaclab_contrib"):
        src = pkg_root / "source" / subdir
        if src.exists() and str(src) not in sys.path:
            sys.path.insert(0, str(src))
    # Clear any partially-imported isaaclab submodules so the source version wins.
    stale = [k for k in sys.modules
             if k in ("isaaclab", "isaaclab_contrib")
             or k.startswith("isaaclab.") or k.startswith("isaaclab_contrib.")]
    for k in stale:
        del sys.modules[k]


_bootstrap_isaaclab_path()

# ---------------------------------------------------------------------------
# IsaacLab 2.3 imports
# ---------------------------------------------------------------------------
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
try:
    from isaaclab.actuators import ImplicitActuatorCfg
except Exception:  # pragma: no cover
    # IsaacLab occasionally relocates cfg classes across versions.
    from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from simulation.config_loader import load_config        # noqa: E402
from simulation.training.reward import RewardConfig, RewardFunction  # noqa: E402
from simulation.isaac.usd.parts_registry import BODY_PRIM, DRONE_ROOT, load_fin_specs, zup_to_frd  # noqa: E402
from simulation.isaac.wind.isaac_wind_model import IsaacWindModel  # noqa: E402
from simulation.isaac.quaternion_isaac import (  # noqa: E402
    rotate_body_to_world_wxyz,
    rotate_world_to_body_wxyz,
)

# ---------------------------------------------------------------------------
# Physical constants from vehicle YAML (loaded once at import)
# ---------------------------------------------------------------------------
_VEHICLE_YAML = REPO_ROOT / "simulation" / "configs" / "default_vehicle.yaml"
_REWARD_YAML  = REPO_ROOT / "simulation" / "configs" / "reward.yaml"

_T_MAX       = 45.0       # N, EDF max thrust
_DELTA_MAX   = 0.2618     # rad, +/-15 deg fin control limit
_TAU_MOTOR   = 0.10       # s, EDF first-order thrust lag
_TAU_SERVO   = 0.04       # s, servo first-order position lag
_K_THRUST    = 4.55e-7    # N/(rad/s)^2
_V_EXHAUST   = 70.0       # m/s nominal
_CL_ALPHA    = 6.283      # /rad, NACA0012 thin-airfoil
_FIN_AREA    = 0.003575   # m^2 per fin, chord x span
_GRAVITY     = 9.81       # m/s^2


def _zup_to_frd_tensor(vec: torch.Tensor) -> torch.Tensor:
    """Convert Z-up vectors to FRD vectors while preserving tensor shape."""
    out = vec.clone()
    out[..., 1] = -out[..., 1]
    out[..., 2] = -out[..., 2]
    return out


def _gravity_world_to_controller_frd(
    g_world: torch.Tensor, quat_w: torch.Tensor
) -> torch.Tensor:
    """Rotate world gravity into the controller's FRD observation convention.

    Isaac's body-rate channels already follow the FRD convention used by the
    shared PID controller, but the gravity-vector tilt projection coming from
    world->body rotation is mirrored in roll/pitch relative to those rates.
    Negating x/y keeps gravity-derived roll/pitch estimates aligned with
    omega_x/omega_y without changing the aerodynamic/body-rate conventions.
    """
    g_b = rotate_world_to_body_wxyz(g_world, quat_w)
    g_b[..., 0] = -g_b[..., 0]
    g_b[..., 1] = -g_b[..., 1]
    return g_b


# ---------------------------------------------------------------------------
# Scene configuration
# ---------------------------------------------------------------------------
@configclass
class EdfSceneCfg(InteractiveSceneCfg):
    """Declarative scene: drone articulation loaded from hand-authored USD.

    The drone USD (drone.usdc) is authored in Isaac Sim with rigid bodies,
    joints, materials, and collision already configured.  We do NOT pass
    rigid_props / articulation_props / collision_props so that the spawner
    preserves every API schema and material in the original asset.

    No terrain importer -- the USD's own GroundPlane (with CollisionPlane)
    is referenced into the live stage by _import_scene_prims_from_usd().
    """

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Drone",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(REPO_ROOT / "simulation" / "isaac" / "usd" / "drone_v2_physics.usdc"),
            # No rigid_props / articulation_props / collision_props -- preserve USD as-is,
            # including whatever lighting is authored in the source scene.
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 7.5),  # Z-up: start 7.5 m above ground
            joint_pos={},
            joint_vel={},
        ),
        actuators={
            # Drive the four fin revolute joints via implicit PD (targets set in task).
            # USD/postprocess uses FwdFin_Joint, AftFin_Joint, LeftFin_Joint, RightFin_Joint.
            "fins": ImplicitActuatorCfg(
                joint_names_expr=[r"(FwdFin|AftFin|LeftFin|RightFin)_Joint"],
                stiffness=20.0,
                damping=1.0,
                effort_limit_sim=2.0,
            ),
        },
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

    # Physics step: 1/120 s, 4 substeps (eff. dt ~ 2.08 ms)
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=1,
        physx=sim_utils.PhysxCfg(
            min_position_iteration_count=8,
            min_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
        ),
    )

    # Policy step = 1 physics step
    decimation: int = 1

    # Episode parameters
    episode_length_s: float = 5.0   # 5 s -> 600 steps at 1/120

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
    # The USD root frame sits about 0.31 m above the ground when the landing
    # gear/body is actually resting on the plane, so touchdown cannot be
    # detected with a near-zero altitude threshold.
    ground_contact_height: float = 0.35     # m, root height considered ground contact
    ground_contact_hysteresis: float = 0.05 # m, re-arm band to avoid ground-start resets

    # Vehicle / reward / environment config paths
    vehicle_config_path: str     = str(REPO_ROOT / "simulation" / "configs" / "default_vehicle.yaml")
    reward_config_path: str      = str(REPO_ROOT / "simulation" / "configs" / "reward.yaml")
    environment_config_path: str = str(REPO_ROOT / "simulation" / "configs" / "default_environment.yaml")


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

        # Load vehicle config (lift directions + rotor properties only)
        vehicle_cfg  = load_config(cfg.vehicle_config_path)
        vehicle_data = vehicle_cfg.get("vehicle", vehicle_cfg)

        # Lift directions remain in config for now; fin anchor positions come from the USD asset.
        fin_specs = load_fin_specs(vehicle_data)
        self._fin_lift = torch.tensor(
            [list(s.lift_direction) for s in fin_specs],
            dtype=torch.float32, device=self.device,
        )  # (4, 3)
        # FwdFin/AftFin (roll pair) are mounted with opposite positive rotation sign
        # relative to control/aero convention. Order matches fin_specs: Right, Left, Fwd, Aft.
        self._fin_joint_visual_sign = torch.tensor(
            [1.0, 1.0, -1.0, -1.0],
            dtype=torch.float32,
            device=self.device,
        )

        # Vehicle mass, CoM, and fin anchors come from the live Isaac asset.
        fin_names = [s.prim_name for s in fin_specs]
        self._body_id = self._resolve_single_body_id("Body")
        self._fin_body_ids = self._resolve_body_ids(fin_names)
        self._fin_anchor_pos_frd = self._read_fin_anchor_positions_from_usd(fin_names)
        self._mass = float(self.robot.data.default_mass[0].sum().item())
        self._weight = self._mass * _GRAVITY
        self._body_com_default_frd = _zup_to_frd_tensor(self.robot.data.body_com_pos_b[0, self._body_id, :])
        self._body_inertia_default = self.robot.data.default_inertia[0, self._body_id, :].reshape(3, 3).clone()
        print(
            "[EdfLandingTask] WARNING: ignoring YAML fin positions and explicit CoM/inertia. "
            f"Using Isaac Sim asset data instead: total_mass={self._mass:.3f} kg, "
            f"body_com_frd={tuple(float(x) for x in self._body_com_default_frd.tolist())}"
        )
        print(
            "[EdfLandingTask] Joint visual sign correction: "
            "RightFin/LeftFin=+1, FwdFin/AftFin=-1"
        )

        # T020: Wind model -- instantiate if isaac_wind.enabled: true in environment config
        env_cfg_raw  = load_config(cfg.environment_config_path)
        env_data     = env_cfg_raw.get("environment", env_cfg_raw)
        wind_cfg     = env_data.get("isaac_wind", {})
        if wind_cfg.get("enabled", False):
            self._wind_model: IsaacWindModel | None = IsaacWindModel(
                wind_cfg, self.num_envs, self.device
            )
        else:
            self._wind_model = None
        # Fallback zero tensor used when wind model is disabled
        self._wind_ema_zeros = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

        # Gyro precession -- load from vehicle YAML edf section
        edf_cfg = vehicle_data.get("edf", {})
        gyro_cfg = edf_cfg.get("gyro_precession", {})
        self._gyro_enabled: bool = bool(gyro_cfg.get("enabled", True))
        self._I_fan: float = float(edf_cfg.get("I_fan", 3.0e-5))  # kg·m², fan rotor MoI
        anti_torque_cfg = edf_cfg.get("anti_torque", {})
        self._anti_torque_enabled: bool = bool(anti_torque_cfg.get("enabled", True))
        self._k_torque: float = float(edf_cfg.get("k_torque", 0.0))
        self._tau_anti_b = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self._tau_ramp_b = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self._omega_fan = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._thrust_dot = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._prev_speed = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._was_airborne = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Previous action (for smoothness reward)
        self._prev_action = torch.zeros(
            (self.num_envs, 5), dtype=torch.float32, device=self.device
        )
        self._current_action = torch.zeros_like(self._prev_action)

        # Per-step terminal diagnostics cached for the wrapper / tuning scripts.
        self._last_landed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_crashed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_out_of_bounds = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_ground_hit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_h_agl = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._last_speed = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._last_impact_speed = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._last_lateral_dist = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # Runtime override flags (used by PID tuning/eval scripts).
        self._runtime_disable_wind: bool = False
        self._runtime_disable_gyro: bool = False
        self._runtime_disable_anti_torque: bool = False
        self._runtime_disable_gravity: bool = False
        self._runtime_reset_altitude_offset_m: float = 0.0
        self._runtime_reset_roll_offset_rad: float = 0.0
        self._runtime_reset_pitch_offset_rad: float = 0.0
        self._runtime_reset_ang_vel_frd = torch.zeros(
            3, dtype=torch.float32, device=self.device
        )

    # ------------------------------------------------------------------
    # Runtime toggles (PID tuning / evaluation)
    # ------------------------------------------------------------------
    def set_runtime_overrides(
        self,
        *,
        disable_wind: bool = False,
        disable_gyro: bool = False,
        disable_anti_torque: bool = False,
        disable_gravity: bool = False,
    ) -> None:
        """Apply per-run overrides for disturbances / gravity.

        These are soft runtime toggles layered on top of YAML config:
        they never re-enable an effect that the config has disabled.
        """
        self._runtime_disable_wind = bool(disable_wind)
        self._runtime_disable_gyro = bool(disable_gyro)
        self._runtime_disable_anti_torque = bool(disable_anti_torque)
        self._runtime_disable_gravity = bool(disable_gravity)

    def set_reset_perturbation(
        self,
        *,
        altitude_offset_m: float = 0.0,
        roll_offset_rad: float = 0.0,
        pitch_offset_rad: float = 0.0,
        ang_vel_frd: tuple[float, float, float] | list[float] | None = None,
    ) -> None:
        """Apply deterministic reset perturbations for hover tuning."""
        self._runtime_reset_altitude_offset_m = float(altitude_offset_m)
        self._runtime_reset_roll_offset_rad = float(roll_offset_rad)
        self._runtime_reset_pitch_offset_rad = float(pitch_offset_rad)
        if ang_vel_frd is None:
            self._runtime_reset_ang_vel_frd.zero_()
        else:
            vals = torch.tensor(
                list(ang_vel_frd),
                dtype=torch.float32,
                device=self.device,
            ).reshape(3)
            self._runtime_reset_ang_vel_frd.copy_(vals)

    def _reset_quat_from_frd_tilt(
        self,
        n: int,
        *,
        roll_rad: float,
        pitch_rad: float,
    ) -> torch.Tensor:
        """Build Isaac wxyz quaternions from FRD roll/pitch perturbations."""
        half_roll = 0.5 * float(roll_rad)
        half_pitch = -0.5 * float(pitch_rad)
        cr = math.cos(half_roll)
        sr = math.sin(half_roll)
        cp = math.cos(half_pitch)
        sp = math.sin(half_pitch)
        quat = torch.zeros((n, 4), device=self.device)
        quat[:, 0] = cr * cp
        quat[:, 1] = sr * cp
        quat[:, 2] = cr * sp
        quat[:, 3] = -sr * sp
        return quat

    def _resolve_single_body_id(self, body_name: str) -> int:
        """Resolve a single articulation body by name."""
        body_ids, body_names = self.robot.find_bodies([body_name], preserve_order=True)
        if len(body_ids) != 1:
            raise RuntimeError(
                f"Expected exactly one body named {body_name!r}, found {body_names!r}."
            )
        return int(body_ids[0])

    def _resolve_body_ids(self, body_names: list[str]) -> list[int]:
        """Resolve articulation body ids in the provided order."""
        body_ids, resolved_names = self.robot.find_bodies(body_names, preserve_order=True)
        if len(body_ids) != len(body_names):
            raise RuntimeError(
                f"Expected bodies {body_names!r}, found {resolved_names!r}."
            )
        return [int(idx) for idx in body_ids]

    def _read_fin_anchor_positions_from_usd(self, fin_names: list[str]) -> torch.Tensor:
        """Read fin-link origins from the source USD relative to `/Drone/Body`, in FRD."""
        from pxr import Gf, Usd, UsdGeom

        usd_path = Path(self.cfg.scene.robot.spawn.usd_path)
        stage = Usd.Stage.Open(str(usd_path))
        if stage is None:
            raise RuntimeError(f"Failed to open drone USD: {usd_path}")

        body_prim = stage.GetPrimAtPath(BODY_PRIM)
        if not body_prim.IsValid():
            raise RuntimeError(f"Missing body prim in USD: {BODY_PRIM}")

        body_xf = UsdGeom.Xformable(body_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        fin_positions_frd: list[list[float]] = []
        for fin_name in fin_names:
            fin_path = f"{DRONE_ROOT}/{fin_name}"
            fin_prim = stage.GetPrimAtPath(fin_path)
            if not fin_prim.IsValid():
                raise RuntimeError(f"Missing fin prim in USD: {fin_path}")
            fin_xf = UsdGeom.Xformable(fin_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            rel = body_xf.GetInverse() * fin_xf
            rel_xf = Gf.Transform()
            rel_xf.SetMatrix(rel)
            pos_zup = rel_xf.GetTranslation()
            fin_positions_frd.append(list(zup_to_frd(float(pos_zup[0]), float(pos_zup[1]), float(pos_zup[2]))))

        return torch.tensor(fin_positions_frd, dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------
    # T014: Scene setup
    # ------------------------------------------------------------------
    def _setup_scene(self) -> None:
        """Set up references to scene entities after the base class creates the scene.

        Also copies the GroundPlane and _materials prims from the source USD
        into the stage so that the authored visual appearance is preserved.
        """
        # self.scene is already created by DirectRLEnv.__init__ before this is called
        self.robot: Articulation = self.scene["robot"]
        # Cache fin joint ids lazily (robot.data buffers not ready during __init__).
        self._fin_joint_ids: list[int] = []

        # Import non-defaultPrim root prims (GroundPlane, _materials) from the
        # authored USD so visual appearance matches the hand-crafted scene.
        self._import_scene_prims_from_usd()

    def _import_scene_prims_from_usd(self) -> None:
        """Import non-defaultPrim root prims and restore material bindings.

        The UsdFileCfg spawner only brings in the defaultPrim (Drone).
        Root-level siblings like GroundPlane and _materials are skipped.
        We add them here so the visual scene matches the authored USD.

        Material bindings in the source USD use absolute paths (e.g.
        /_materials/MyMaterial) which break when the drone is cloned into
        /World/envs/env_0/Drone.  We re-bind each mesh to the global
        material scope after importing it.
        """
        from pxr import Usd, UsdGeom, UsdShade, Sdf

        usd_path = self.cfg.scene.robot.spawn.usd_path
        live_stage = self.sim.stage

        # Open the source USD as a read-only layer
        src_stage = Usd.Stage.Open(usd_path, Usd.Stage.LoadAll)
        if src_stage is None:
            return

        src_root = src_stage.GetPseudoRoot()
        default_prim_name = src_stage.GetDefaultPrim().GetName() if src_stage.HasDefaultPrim() else ""

        # 1. Import root-level siblings (GroundPlane, _materials, etc.)
        for child in src_root.GetChildren():
            name = child.GetName()
            if name == default_prim_name:
                continue
            dest_path = Sdf.Path(f"/{name}")
            if live_stage.GetPrimAtPath(dest_path).IsValid():
                continue
            over_prim = live_stage.DefinePrim(dest_path)
            over_prim.GetReferences().AddReference(usd_path, child.GetPath())

        # 2. Re-bind materials on cloned drone meshes.
        #    Walk the source defaultPrim, find every mesh with a material binding,
        #    and apply the same binding on the corresponding live-stage prim.
        src_default = src_stage.GetDefaultPrim()
        if not src_default.IsValid():
            return

        for src_prim in Usd.PrimRange(src_default):
            binding_api = UsdShade.MaterialBindingAPI(src_prim)
            bound_mat_path = binding_api.GetDirectBinding().GetMaterialPath()
            if not bound_mat_path or bound_mat_path == Sdf.Path():
                continue
            # The material path is absolute in the source (e.g. /_materials/Fin_Mat)
            mat_path_str = str(bound_mat_path)
            # Check material exists in live stage (imported above)
            live_mat = live_stage.GetPrimAtPath(mat_path_str)
            if not live_mat.IsValid():
                continue
            # Find the corresponding prim in each env clone
            rel_path = str(src_prim.GetPath()).lstrip("/")  # e.g. "Drone/FwdFin/Mesh"
            for env_id in range(self.num_envs):
                env_prim_path = f"/World/envs/env_{env_id}/{rel_path}"
                live_prim = live_stage.GetPrimAtPath(env_prim_path)
                if live_prim.IsValid():
                    live_binding = UsdShade.MaterialBindingAPI.Apply(live_prim)
                    mat = UsdShade.Material(live_mat)
                    live_binding.Bind(mat)

    # ------------------------------------------------------------------
    # T015: Reset per env indices
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        """Randomize spawn state for selected environments."""
        n = len(env_ids)
        if n == 0:
            return

        # Base class reset: scene.reset (actuators, wrenches) + episode_length_buf
        super()._reset_idx(env_ids)

        # Reset lag state
        self.thrust_actual[env_ids] = 0.0
        self.fin_deflections_actual[env_ids] = 0.0
        self._episode_step[env_ids] = 0
        self._prev_action[env_ids] = 0.0
        self._current_action[env_ids] = 0.0
        self._tau_anti_b[env_ids] = 0.0
        self._tau_ramp_b[env_ids] = 0.0
        self._omega_fan[env_ids] = 0.0
        self._thrust_dot[env_ids] = 0.0
        self._prev_speed[env_ids] = 0.0
        self._last_landed[env_ids] = False
        self._last_crashed[env_ids] = False
        self._last_out_of_bounds[env_ids] = False
        self._last_ground_hit[env_ids] = False
        self._last_h_agl[env_ids] = 0.0
        self._last_speed[env_ids] = 0.0
        self._last_impact_speed[env_ids] = 0.0
        self._last_lateral_dist[env_ids] = 0.0
        # T022: Reset wind model for selected environments
        if self._wind_model is not None:
            self._wind_model.reset(env_ids)

        # Sample random altitude and velocity
        alt = torch.zeros(n, device=self.device).uniform_(
            self.cfg.spawn_altitude_min, self.cfg.spawn_altitude_max
        )
        alt += float(self._runtime_reset_altitude_offset_m)
        alt.clamp_(min=0.0)
        vel_mag = torch.zeros(n, device=self.device).uniform_(
            self.cfg.spawn_vel_mag_min, self.cfg.spawn_vel_mag_max
        )
        self._was_airborne[env_ids] = (
            alt > (self.cfg.ground_contact_height + self.cfg.ground_contact_hysteresis)
        )

        # Random velocity direction
        theta = torch.zeros(n, device=self.device).uniform_(0.0, 2.0 * math.pi)
        phi   = torch.zeros(n, device=self.device).uniform_(0.0, math.pi)
        vx = vel_mag * torch.sin(phi) * torch.cos(theta)
        vy = vel_mag * torch.sin(phi) * torch.sin(theta)
        vz = vel_mag * torch.cos(phi)
        vel = torch.stack([vx, vy, vz], dim=-1)  # (n, 3)

        # Root pose: position at random altitude (Z-up: altitude = Z)
        pos = torch.zeros((n, 3), device=self.device)
        pos[:, 2] = alt  # Z = altitude in Z-up world

        # Identity quaternion in wxyz format (IsaacLab convention: w, x, y, z)
        quat = torch.zeros((n, 4), device=self.device)
        quat[:, 0] = 1.0  # w=1 -> identity (wxyz: [1, 0, 0, 0])
        if (
            abs(float(self._runtime_reset_roll_offset_rad)) > 1e-9
            or abs(float(self._runtime_reset_pitch_offset_rad)) > 1e-9
        ):
            quat = self._reset_quat_from_frd_tilt(
                n,
                roll_rad=float(self._runtime_reset_roll_offset_rad),
                pitch_rad=float(self._runtime_reset_pitch_offset_rad),
            )

        # Add env origins for multi-env support
        root_pose = torch.cat([pos, quat], dim=-1)  # (n, 7)
        root_pose[:, :3] += self.scene.env_origins[env_ids]

        ang_vel = torch.zeros((n, 3), device=self.device)
        if torch.linalg.norm(self._runtime_reset_ang_vel_frd).item() > 0.0:
            ang_vel[:, 0] = self._runtime_reset_ang_vel_frd[0]
            ang_vel[:, 1] = -self._runtime_reset_ang_vel_frd[1]
            ang_vel[:, 2] = -self._runtime_reset_ang_vel_frd[2]

        root_vel = torch.cat([vel, ang_vel], dim=-1)

        self.robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(root_vel, env_ids=env_ids)

        # Reset joints to neutral (guard for zero-joint articulation)
        if self.robot.num_joints > 0:
            joint_pos = torch.zeros((n, self.robot.num_joints), device=self.device)
            joint_vel = torch.zeros_like(joint_pos)
            self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    # ------------------------------------------------------------------
    # T017: Pre-physics step -- force injection
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Update lag state once per policy step (called by DirectRLEnv before decimation loop)."""
        dt = self._dt

        # Clip and store actions for use in _apply_action
        # Expand to (num_envs, 5) in case a (1, 5) action is broadcast from the wrapper
        self.actions = actions.expand(self.num_envs, -1).clamp(-1.0, 1.0)
        self._current_action.copy_(self.actions)

        # Unpack: [thrust_cmd, d1, d2, d3, d4]
        thrust_cmd_norm = self.actions[:, 0]
        fin_cmd_norm    = self.actions[:, 1:5]

        # Map thrust: [-1,0] -> 0, [0,1] -> T_MAX
        T_cmd     = thrust_cmd_norm.clamp(0.0, 1.0) * _T_MAX
        delta_cmd = fin_cmd_norm * _DELTA_MAX

        # First-order lag update (once per policy step)
        self.thrust_actual += (dt / _TAU_MOTOR) * (T_cmd - self.thrust_actual)
        self.fin_deflections_actual += (dt / _TAU_SERVO) * (
            delta_cmd - self.fin_deflections_actual
        )
        self.fin_deflections_actual.clamp_(-_DELTA_MAX, _DELTA_MAX)
        self._thrust_dot = (T_cmd - self.thrust_actual) / _TAU_MOTOR
        self._omega_fan = (self.thrust_actual / _K_THRUST).clamp(min=0.0).sqrt()

        # Increment episode step counter
        self._episode_step += 1

    def _apply_action(self) -> None:
        """Apply forces/torques each decimation substep using current lag state."""
        num_envs = self.num_envs
        forces  = torch.zeros((num_envs, 1, 3), device=self.device)
        torques = torch.zeros((num_envs, 1, 3), device=self.device)

        # ----------------------------------------------------------------
        # Fin joints -- set Drive/actuator position targets (degrees)
        # ----------------------------------------------------------------
        if not self._fin_joint_ids:
            # Debug: print all joints the articulation knows about
            print(f"[EdfLandingTask] robot.num_joints = {self.robot.num_joints}")
            print(f"[EdfLandingTask] robot.joint_names = {self.robot.joint_names}")

            # Manual scene: joints are /Drone/Fin_N/RevoluteJoint.
            # PhysX joint names appear as "RevoluteJoint" (possibly de-duplicated
            # with index suffixes).  Try exact names first, then regex fallback.
            # Find fin joints by name: FwdFin_Joint, AftFin_Joint, LeftFin_Joint, RightFin_Joint
            fin_specs = load_fin_specs(
                load_config(self.cfg.vehicle_config_path).get("vehicle", {})
            )
            joint_names_ordered = [f"{s.prim_name}_Joint" for s in fin_specs]
            try:
                joint_ids, joint_names = self.robot.find_joints(
                    joint_names_ordered,
                    preserve_order=True,
                )
                print(f"[EdfLandingTask] find_joints({joint_names_ordered}) -> ids={joint_ids}, names={joint_names}")
            except Exception as e:
                print(f"[EdfLandingTask] find_joints({joint_names_ordered}) FAILED: {e}")
                joint_ids, joint_names = [], []

            if len(joint_ids) == 4:
                self._fin_joint_ids = [int(i) for i in joint_ids]
            else:
                # Fallback: regex to catch any joint
                try:
                    joint_ids, joint_names = self.robot.find_joints(
                        ".*", preserve_order=False
                    )
                    print(f"[EdfLandingTask] find_joints('.*') -> ids={joint_ids}, names={joint_names}")
                    if len(joint_ids) == 4:
                        self._fin_joint_ids = [int(i) for i in joint_ids]
                except Exception as e:
                    print(f"[EdfLandingTask] find_joints('.*') FAILED: {e}")

            if not self._fin_joint_ids:
                print("[EdfLandingTask] WARNING: Could not find 4 fin joints! Fins will not move.")

        if self._fin_joint_ids:
            fin_target_deg = (
                self.fin_deflections_actual
                * self._fin_joint_visual_sign.unsqueeze(0)
                * (180.0 / math.pi)
            )
            self.robot.set_joint_position_target(fin_target_deg, joint_ids=self._fin_joint_ids)

        quat_w = self.robot.data.root_quat_w  # (N, 4) IsaacLab wxyz [qw, qx, qy, qz]

        # ----------------------------------------------------------------
        # EDF thrust -- body-frame +Z rotated into world.
        #
        # The previous world-up implementation let inverted vehicles hold
        # altitude. Here we make thrust follow the vehicle attitude by rotating
        # the body-axis thrust vector into world before applying it.
        # ----------------------------------------------------------------
        thrust_b = torch.zeros((num_envs, 3), dtype=torch.float32, device=self.device)
        thrust_b[:, 2] = self.thrust_actual
        forces[:, 0, :] += rotate_body_to_world_wxyz(quat_w, thrust_b)

        # ----------------------------------------------------------------
        # Fin aerodynamic forces (NACA0012 in exhaust stream)
        # ----------------------------------------------------------------
        omega_ratio  = (self.thrust_actual / _T_MAX).clamp(0.0, 1.0).sqrt()
        V_ex         = omega_ratio * _V_EXHAUST
        dyn_pressure = 0.5 * 1.225 * V_ex.pow(2)  # sea-level rho

        # Fin anchor points are authored in the USD asset. Convert them to live
        # lever arms about the Isaac-computed body CoM each step.
        body_com_frd = _zup_to_frd_tensor(self.robot.data.body_com_pos_b[:, self._body_id, :])

        for i in range(4):
            delta_i = self.fin_deflections_actual[:, i]                  # (N,)
            L_i     = dyn_pressure * _FIN_AREA * _CL_ALPHA * delta_i     # (N,)

            # Broadcast per-fin body-frame vectors to (N, 3)
            lift_dir_b = self._fin_lift[i].unsqueeze(0).expand(num_envs, -1)  # (N, 3)
            anchor_b   = self._fin_anchor_pos_frd[i].unsqueeze(0).expand(num_envs, -1)
            r_b        = anchor_b - body_com_frd

            # Rotate into world (Z-up) frame
            lift_dir_w = rotate_body_to_world_wxyz(quat_w, lift_dir_b)  # (N, 3)
            r_w        = rotate_body_to_world_wxyz(quat_w, r_b)         # (N, 3)

            F_i = L_i.unsqueeze(-1) * lift_dir_w  # (N, 3)
            # Use fin lift only to generate attitude torque. Omitting the net
            # translational force keeps lateral motion dominated by tilt+thrust
            # instead of direct side-force from the control surfaces.
            torques[:, 0, :] += torch.linalg.cross(r_w, F_i)

        # Gyro precession torque: τ_gyro = −ω_body × h_fan  (body frame → world frame)
        # PhysX handles ω×(I·ω) internally; we only inject the fan angular momentum term.
        # h_fan = [0, 0, I_fan·ω_fan] in body FRD frame (fan spins about body +Z = down).
        if self._gyro_enabled and not self._runtime_disable_gyro:
            omega_b   = self.robot.data.root_ang_vel_b                       # (N, 3) body frame
            h_fan_b   = torch.zeros((num_envs, 3), device=self.device)
            h_fan_b[:, 2] = self._I_fan * self._omega_fan                    # body +Z (FRD down)
            tau_gyro_b = -torch.linalg.cross(omega_b, h_fan_b)              # (N, 3)
            tau_gyro_w = rotate_body_to_world_wxyz(self.robot.data.root_quat_w, tau_gyro_b)
            torques[:, 0, :] += tau_gyro_w

        # Steady-state EDF anti-torque: τ_anti = -k_torque * ω_fan² about body +Z.
        self._tau_anti_b.zero_()
        self._tau_ramp_b.zero_()
        if self._anti_torque_enabled and not self._runtime_disable_anti_torque:
            self._tau_anti_b[:, 2] = -self._k_torque * self._omega_fan.square()
            tau_anti_w = rotate_body_to_world_wxyz(quat_w, self._tau_anti_b)
            torques[:, 0, :] += tau_anti_w

            omega_safe = self._omega_fan.clamp(min=1e-6)
            domega_dt = self._thrust_dot / (2.0 * _K_THRUST * omega_safe)
            self._tau_ramp_b[:, 2] = -self._I_fan * domega_dt
            tau_ramp_w = rotate_body_to_world_wxyz(quat_w, self._tau_ramp_b)
            torques[:, 0, :] += tau_ramp_w

        # T021: Wind drag force -- added if wind model is active
        if self._wind_model is not None and not self._runtime_disable_wind:
            wind_vec    = self._wind_model.step(self._dt)   # (num_envs, 3) world frame
            body_vel_w  = self.robot.data.root_lin_vel_w    # (num_envs, 3) world frame
            F_wind      = self._wind_model.compute_drag_force(wind_vec, body_vel_w)
            forces[:, 0, :] += F_wind

        self.robot.set_external_force_and_torque(forces, torques, body_ids=[self._body_id], is_global=True)
        self.robot.write_data_to_sim()

    # ------------------------------------------------------------------
    # T018: Observations -- 20-dim vector
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Compute 20-dim observation matching observation.py layout."""
        data = self.robot.data

        # World-frame position (Y-up)
        pos_w  = data.root_pos_w    # (num_envs, 3): [x_world, y_world, z_world]
        quat_w = data.root_quat_w   # (num_envs, 4): IsaacLab wxyz [qw, qx, qy, qz]
        vel_b  = data.root_lin_vel_b  # (num_envs, 3): body-frame linear vel
        omega_b = data.root_ang_vel_b  # (num_envs, 3): body-frame angular vel

        # --- Altitude above ground (Z-up: Z = altitude) ---
        h_agl = pos_w[:, 2].clamp(min=0.0)  # (num_envs,)

        # --- Position error in world frame (to target) ---
        target_w = self._target_pos_w  # (num_envs, 3)
        err_w = target_w - pos_w      # (num_envs, 3)

        # --- Rotate error into body frame ---
        # q_world_body = root_quat_w -> rotate world vector to body
        err_b = rotate_world_to_body_wxyz(err_w, quat_w)  # (num_envs, 3)

        # --- Gravity direction in body frame ---
        # World gravity is -Z in Z-up frame
        g_world = torch.zeros((self.num_envs, 3), device=self.device)
        g_world[:, 2] = -1.0  # unit vector pointing down in Z-up
        g_b = _gravity_world_to_controller_frd(g_world, quat_w)  # (num_envs, 3)

        # --- Thrust-to-weight ratio ---
        twr = (self.thrust_actual / self._weight).unsqueeze(-1)  # (num_envs, 1)

        # --- Scalars ---
        speed     = vel_b.norm(dim=-1, keepdim=True)     # (num_envs, 1)
        ang_speed = omega_b.norm(dim=-1, keepdim=True)   # (num_envs, 1)
        time_frac = (self._episode_step.float() / self._max_steps).clamp(0.0, 1.0).unsqueeze(-1)

        # --- Assemble 20-dim obs ---
        # T023: Wind EMA -- use model output when wind active, else zeros
        wind_ema = (
            self._wind_model.wind_ema
            if self._wind_model is not None
            else self._wind_ema_zeros
        )

        # [0:3]=e_p_body, [3:6]=v_body, [6:9]=g_body, [9:12]=omega,
        # [12]=twr, [13:16]=wind_ema, [16]=h_agl, [17]=speed,
        # [18]=ang_speed, [19]=time_frac
        obs = torch.cat([
            err_b,                   # 3
            vel_b,                   # 3
            g_b,                     # 3
            omega_b,                 # 3
            twr,                     # 1
            wind_ema,                # 3  (zeros if wind disabled)
            h_agl.unsqueeze(-1),     # 1
            speed,                   # 1
            ang_speed,               # 1
            time_frac,               # 1
        ], dim=-1)  # (num_envs, 20)

        return {"policy": obs}

    # ------------------------------------------------------------------
    # T019: Rewards -- port from reward.py
    # ------------------------------------------------------------------
    def _compute_terminal_metrics(self) -> dict[str, torch.Tensor]:
        """Compute touchdown / crash metrics shared by reward and done logic."""
        data = self.robot.data
        pos_w = data.root_pos_w
        vel_b = data.root_lin_vel_b

        h_agl = pos_w[:, 2]
        speed = vel_b.norm(dim=-1)

        target = self._target_pos_w
        lateral_dist = (
            (pos_w[:, 0] - target[:, 0]).pow(2)
            + (pos_w[:, 1] - target[:, 1]).pow(2)
        ).sqrt()

        contact_height = self.cfg.ground_contact_height
        airborne_height = contact_height + self.cfg.ground_contact_hysteresis
        was_airborne = self._was_airborne | (h_agl > airborne_height)
        near_ground = h_agl <= contact_height
        ground_hit = was_airborne & near_ground
        impact_speed = torch.maximum(speed, self._prev_speed)

        crashed = ground_hit & (impact_speed > self.cfg.crash_velocity_threshold)
        landed = (
            ground_hit
            & (impact_speed <= self.cfg.crash_velocity_threshold)
            & (lateral_dist <= self.cfg.landing_pad_radius)
        )
        out_of_bounds = torch.zeros_like(landed)

        return {
            "h_agl": h_agl,
            "speed": speed,
            "lateral_dist": lateral_dist,
            "was_airborne": was_airborne,
            "ground_hit": ground_hit,
            "impact_speed": impact_speed,
            "landed": landed,
            "crashed": crashed,
            "out_of_bounds": out_of_bounds,
        }

    def _get_rewards(self) -> torch.Tensor:
        """Compute shaped + terminal rewards. Returns (num_envs,) float32 tensor."""
        cfg = self._reward_cfg
        data = self.robot.data

        pos_w   = data.root_pos_w
        vel_b   = data.root_lin_vel_b
        omega_b = data.root_ang_vel_b
        quat_w  = data.root_quat_w

        term = self._compute_terminal_metrics()
        h_agl = term["h_agl"].clamp(min=0.0)
        speed = term["speed"]

        # Distance to target
        dist = (self._target_pos_w - pos_w).norm(dim=-1)

        # Alive bonus
        r = torch.full((self.num_envs,), cfg.alive_bonus, device=self.device)

        # Shaping: distance + velocity
        r -= cfg.shaping_distance_coeff * dist
        r -= cfg.shaping_velocity_coeff * speed

        # Orientation penalty: deviation of gravity from body -Z (FRD down)
        g_world = torch.zeros((self.num_envs, 3), device=self.device)
        g_world[:, 2] = -1.0
        g_b = rotate_world_to_body_wxyz(g_world, quat_w)
        # ideal g_body = [0, 0, 1] in FRD (gravity points +Z down)
        cos_theta = g_b[:, 2].clamp(-1.0, 1.0)
        tilt_penalty = 1.0 - cos_theta  # 0 when upright
        r -= cfg.orientation_weight * tilt_penalty

        # Action smoothness penalty
        delta_action = (self._current_action - self._prev_action).norm(dim=-1)
        r -= cfg.action_smooth_weight * delta_action

        sigma = max(float(cfg.precision_sigma), 1e-9)
        v_touch_max = max(float(self.cfg.crash_velocity_threshold), 1e-9)
        landed = term["landed"]
        crashed = term["crashed"]
        out_of_bounds = term["out_of_bounds"]

        precision_bonus = cfg.precision_bonus * torch.exp(
            -(term["lateral_dist"].square()) / (2.0 * sigma * sigma)
        )
        soft_touch_bonus = cfg.soft_touchdown * torch.clamp(
            1.0 - (term["impact_speed"] / v_touch_max),
            min=0.0,
        )
        landing_bonus = (
            cfg.landing_success
            + precision_bonus
            + soft_touch_bonus
        )
        r += landed.float() * landing_bonus
        r -= crashed.float() * cfg.crash_penalty
        r -= out_of_bounds.float() * cfg.oob_penalty

        self._prev_action.copy_(self._current_action)
        return r

    # ------------------------------------------------------------------
    # T020: Dones
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (terminated, truncated) tensors."""
        term = self._compute_terminal_metrics()
        landed = term["landed"]
        crashed = term["crashed"]
        out_of_bounds = term["out_of_bounds"]

        terminated = crashed | landed | out_of_bounds
        truncated  = self._episode_step >= self._max_steps
        self._prev_speed.copy_(term["speed"])
        self._was_airborne.copy_(term["was_airborne"])
        self._was_airborne[terminated] = False
        self._last_landed.copy_(landed)
        self._last_crashed.copy_(crashed)
        self._last_out_of_bounds.copy_(out_of_bounds)
        self._last_ground_hit.copy_(term["ground_hit"])
        self._last_h_agl.copy_(term["h_agl"])
        self._last_speed.copy_(term["speed"])
        self._last_impact_speed.copy_(term["impact_speed"])
        self._last_lateral_dist.copy_(term["lateral_dist"])

        return terminated, truncated


