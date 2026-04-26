"""
InteractiveScene setup and environment cloning for the TVC environment.

Creates InteractiveSceneCfg with configurable num_envs/env_spacing/replicate_physics,
spawns drone articulation and ground plane, clones environments, and filters
inter-environment collisions per research decision R3.

Requires Isaac Lab 2.3.2 runtime.
"""

from __future__ import annotations
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationContext

# Repo-relative root: simulation/isaac/ (two levels up from tvc_env/sim/)
_SIM_ROOT = Path(__file__).parents[2].resolve()
_DRONE_USD      = str(_SIM_ROOT / "assets/usd/drone_v2_physics.usd")
_LANDING_PAD_USD = str(_SIM_ROOT / "assets/usd/landing_pad.usd")
_METADATA_YAML  = str(_SIM_ROOT / "assets/metadata/edf_drone_v2.asset.yaml")


@dataclass
class SceneConfig:
    """Configuration for InteractiveScene setup."""
    num_envs: int = 1
    env_spacing: float = 4.0                # m, distance between environment origins
    replicate_physics: bool = True          # GPU-pipeline physics replication
    gizmos_enabled: bool = False
    dispatch_mode: str = "per_link_force"
    physics_dt: float = 1.0 / 120.0        # s
    decimation: int = 4                     # physics substeps per RL step
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    enable_scene_query_support: bool = False
    solver_type: int = 1                    # 1 = TGS
    num_position_iterations: int = 4
    num_velocity_iterations: int = 1
    enable_external_forces_every_iteration: bool = True
    gpu_temp_buffer_capacity: int | None = None
    gpu_max_rigid_contact_count: int | None = None
    gpu_max_rigid_patch_count: int | None = None
    gpu_found_lost_pairs_capacity: int | None = None

    # Asset paths — absolute, resolved from this file's location
    drone_usd_path: str = _DRONE_USD
    landing_pad_usd_path: str = _LANDING_PAD_USD
    metadata_yaml_path: str = _METADATA_YAML

    @classmethod
    def from_yaml(cls, env_config: dict[str, Any]) -> "SceneConfig":
        """Create SceneConfig from a parsed env YAML dict."""
        env = env_config.get("env", env_config)
        physics = env_config.get("physics", {})
        return cls(
            num_envs=env.get("num_envs", 1),
            env_spacing=env.get("env_spacing", 4.0),
            replicate_physics=env.get("replicate_physics", True),
            gizmos_enabled=env.get("gizmos_enabled", False),
            dispatch_mode=env.get("dispatch_mode", "per_link_force"),
            physics_dt=physics.get("dt", env.get("physics_dt", 1.0 / 120.0)),
            decimation=env.get("decimation", 4),
            gravity=tuple(physics.get("gravity", [0.0, 0.0, -9.81])),
            enable_scene_query_support=physics.get("enable_scene_query_support", False),
            solver_type=physics.get("solver_type", 1),
            num_position_iterations=physics.get("num_position_iterations", 4),
            num_velocity_iterations=physics.get("num_velocity_iterations", 1),
            enable_external_forces_every_iteration=physics.get(
                "enable_external_forces_every_iteration", True
            ),
            gpu_temp_buffer_capacity=physics.get("gpu_temp_buffer_capacity"),
            gpu_max_rigid_contact_count=physics.get("gpu_max_rigid_contact_count"),
            gpu_max_rigid_patch_count=physics.get("gpu_max_rigid_patch_count"),
            gpu_found_lost_pairs_capacity=physics.get("gpu_found_lost_pairs_capacity"),
        )


@dataclass
class TVCSimScene:
    """Simulation context plus :class:`InteractiveScene` with a single physics step helper."""

    sim: "SimulationContext"
    scene: "InteractiveScene"
    physics_dt: float

    def step(self, render: bool | None = None) -> None:
        """One physics substep: write commands, advance sim, refresh articulation buffers.

        render=None auto-detects ISAAC_VIZ_SLOW env var (set via --slow flag).
        When rendering, also pumps the Kit event loop so the viewport stays live.
        """
        if render is None:
            render = os.getenv("ISAAC_VIZ_SLOW", "0") == "1"
        self.scene.write_data_to_sim()
        self.sim.step(render=render)
        self.scene.update(self.physics_dt)
        if render:
            # Use sim.render() which safely guards against extra physics
            # steps, instead of raw app.update().
            self.sim.render()

    def __getitem__(self, key: str) -> Any:
        return self.scene[key]

    def wait(self, duration_s: float) -> None:
        """Keep the Kit UI responsive while holding the current scene state."""
        if duration_s <= 0.0:
            return

        try:
            import omni.kit.app
            app = omni.kit.app.get_app()
        except Exception:
            time.sleep(duration_s)
            return

        if app is None:
            time.sleep(duration_s)
            return

        deadline = time.perf_counter() + duration_s
        while True:
            if time.perf_counter() >= deadline:
                break

            try:
                app.update()
            except Exception:
                break

    def close(self) -> None:
        """Release the active Isaac Lab simulation context for the current process."""
        fast_close = os.getenv("TVC_ISAAC_FAST_CLOSE", "1") == "1"

        # IsaacLab 2.3.x can hang inside SimulationContext.stop() in headless
        # command-line runs on Windows. Keep it opt-in so tests and eval runners
        # can terminate reliably after logging their final state.
        if not fast_close and os.getenv("TVC_ISAAC_CALL_SIM_STOP_ON_CLOSE", "0") == "1":
            try:
                if not self.sim.has_gui():
                    self.sim.stop()
            except Exception:
                pass

        try:
            self.sim.clear_all_callbacks()
        except Exception:
            pass

        if not fast_close:
            # Detach the USD stage from the omni context before clearing the singleton,
            # so a subsequent build_scene() can attach a fresh stage without conflict.
            try:
                import omni.usd
                ctx = omni.usd.get_context()
                if ctx is not None:
                    ctx.close_stage()
            except Exception:
                pass

        try:
            type(self.sim).clear_instance()
        except Exception:
            pass


def build_scene(config: SceneConfig) -> TVCSimScene:
    """Create :class:`SimulationContext`, attach stage, build :class:`InteractiveScene`, and reset once.

    Isaac Lab requires a live :class:`~isaaclab.sim.SimulationContext` before
    :class:`~isaaclab.scene.InteractiveScene` can be constructed.

    Args:
        config: SceneConfig with environment parameters.

    Returns:
        TVCSimScene wrapping the simulation and scene handles.

    Raises:
        ImportError: If Isaac Lab is not available.
        RuntimeError: If a :class:`SimulationContext` already exists in this process.
    """
    try:
        from isaaclab.sim import PhysxCfg, SimulationContext, SimulationCfg
        from isaaclab.scene import InteractiveScene
        from isaaclab.sim.utils.stage import attach_stage_to_usd_context, use_stage
    except ImportError as e:
        raise ImportError(
            "Isaac Lab 2.3.2 required for scene building. "
            "Ensure Isaac Lab is installed and accessible."
        ) from e

    if SimulationContext.instance() is not None:
        raise RuntimeError(
            "A SimulationContext already exists; only one TVC scene stack per process is supported."
        )

    physx_kwargs = {
        "solver_type": config.solver_type,
        "min_position_iteration_count": config.num_position_iterations,
        "min_velocity_iteration_count": config.num_velocity_iterations,
        "enable_external_forces_every_iteration": config.enable_external_forces_every_iteration,
    }
    for field_name in (
        "gpu_temp_buffer_capacity",
        "gpu_max_rigid_contact_count",
        "gpu_max_rigid_patch_count",
        "gpu_found_lost_pairs_capacity",
    ):
        value = getattr(config, field_name)
        if value is not None:
            physx_kwargs[field_name] = value

    sim_cfg = SimulationCfg(
        dt=config.physics_dt,
        gravity=config.gravity,
        enable_scene_query_support=config.enable_scene_query_support,
        physx=PhysxCfg(**physx_kwargs),
    )
    sim = SimulationContext(sim_cfg)
    scene_cfg = _create_scene_cfg(config)
    with use_stage(sim.get_initial_stage()):
        scene = InteractiveScene(scene_cfg)
    attach_stage_to_usd_context()

    # Copy scene environment (ground, lights, materials) from the validated USD
    # so the test scene matches the authored scene exactly.
    _add_scene_environment(config.drone_usd_path)

    with use_stage(sim.get_initial_stage()):
        sim.reset()
    scene.reset()
    scene.update(config.physics_dt)

    # In slow/visual mode, warm up the viewport so the scene is visible
    # from the very first frame the tests see.
    if os.getenv("ISAAC_VIZ_SLOW", "0") == "1":
        _warmup_viewport(sim)

    return TVCSimScene(sim=sim, scene=scene, physics_dt=config.physics_dt)


def _warmup_viewport(sim: "SimulationContext") -> None:
    """Pump render frames so the Kit viewport initialises its render pipeline.

    Without this, the viewport stays white/frozen until the first physics step
    because no draw calls have been issued to the RTX renderer.
    """
    try:
        import omni.kit.app
        app = omni.kit.app.get_app()
    except ImportError:
        return

    # Phase 1: pump frames so the renderer picks up the stage contents.
    for _ in range(20):
        try:
            app.update()
        except Exception:
            break

    # Phase 2: position the camera to frame the drone (spawned at Z=5 m).
    try:
        sim.set_camera_view(eye=(4.0, 4.0, 9.0), target=(0.0, 0.0, 5.0))
    except Exception as exc:
        print(f"[scene_builder] WARN set_camera_view failed: {exc}")

    # Phase 3: render a few more frames with the camera in position so the
    # first test frame already shows the drone rather than a blank scene.
    for _ in range(10):
        try:
            app.update()
        except Exception:
            break


def _add_scene_environment(drone_usd_path: str) -> None:
    """Copy scene-level prims (ground, lights, materials) from the drone USD.

    The validated ``drone_v2_physics.usd`` already contains the full scene:
    FlatGrid, SphereLight, Environment, and ``/_materials`` as siblings of the
    Drone defaultPrim.  Isaac Lab's ``ArticulationCfg`` only spawns the
    defaultPrim, so we copy the remaining prims here so the test scene matches
    the validated scene exactly.
    """
    try:
        import omni.usd
        from pxr import Usd, Sdf
    except ImportError:
        return

    host_stage = omni.usd.get_context().get_stage()
    if host_stage is None:
        return

    drone_stage = Usd.Stage.Open(drone_usd_path)
    if drone_stage is None:
        return

    src_layer = drone_stage.GetRootLayer()
    dst_layer = host_stage.GetEditTarget().GetLayer()

    for prim_path_str in ["/_materials", "/FlatGrid", "/SphereLight", "/Environment"]:
        prim_path = Sdf.Path(prim_path_str)
        if drone_stage.GetPrimAtPath(prim_path).IsValid():
            if not host_stage.GetPrimAtPath(prim_path).IsValid():
                Sdf.CopySpec(src_layer, prim_path, dst_layer, prim_path)


def _create_scene_cfg(config: SceneConfig):
    """Create InteractiveSceneCfg from SceneConfig."""
    try:
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.assets import ArticulationCfg
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.utils import configclass
        import isaaclab.sim as sim_utils
    except ImportError as e:
        raise ImportError("Isaac Lab 2.3.2 required.") from e

    @configclass
    class TVCSceneCfg(InteractiveSceneCfg):
        # Scene environment (ground, lights, materials) is copied from
        # drone_v2_physics.usd by _add_scene_environment() — no spawners needed.

        # EDF drone articulation
        drone: ArticulationCfg = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/drone",
            # Let Isaac Lab auto-discover the ArticulationRootAPI prim (Body) by scanning
            # the subtree. Explicit "/Body" concatenates directly: prim_path + "/Body" which
            # is identical, but None is safer if the USD structure ever shifts.
            articulation_root_prim_path=None,
            spawn=sim_utils.UsdFileCfg(
                usd_path=config.drone_usd_path,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    retain_accelerations=False,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=100.0,
                    max_angular_velocity=100.0,
                    max_depenetration_velocity=1.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=config.num_position_iterations,
                    solver_velocity_iteration_count=config.num_velocity_iterations,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 5.0),  # Start 5m above ground in Isaac Z-up frame
                joint_pos={".*": 0.0},
            ),
            actuators={
                "fins": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=80.0,
                    damping=2.0,
                ),
            },
        )

    return TVCSceneCfg(
        num_envs=config.num_envs,
        env_spacing=config.env_spacing,
        replicate_physics=config.replicate_physics,
    )
