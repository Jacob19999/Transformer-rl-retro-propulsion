"""
InteractiveScene setup and environment cloning for the TVC environment.

Creates InteractiveSceneCfg with configurable num_envs/env_spacing/replicate_physics,
spawns drone articulation and ground plane, clones environments, and filters
inter-environment collisions per research decision R3.

Requires Isaac Lab 2.3.2 runtime.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


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

    # Asset paths
    drone_usd_path: str = "assets/usd/edf_drone_v2.usd"
    landing_pad_usd_path: str = "assets/usd/landing_pad.usd"
    metadata_yaml_path: str = "assets/metadata/edf_drone_v2.asset.yaml"

    @classmethod
    def from_yaml(cls, env_config: dict[str, Any]) -> "SceneConfig":
        """Create SceneConfig from a parsed env YAML dict."""
        env = env_config.get("env", env_config)
        return cls(
            num_envs=env.get("num_envs", 1),
            env_spacing=env.get("env_spacing", 4.0),
            replicate_physics=env.get("replicate_physics", True),
            gizmos_enabled=env.get("gizmos_enabled", False),
            dispatch_mode=env.get("dispatch_mode", "per_link_force"),
            physics_dt=env.get("physics_dt", 1.0 / 120.0),
            decimation=env.get("decimation", 4),
        )


def build_scene(config: SceneConfig) -> "InteractiveScene":
    """Build and return an Isaac Lab InteractiveScene for the TVC environment.

    Args:
        config: SceneConfig with environment parameters.

    Returns:
        Configured InteractiveScene object.

    Raises:
        ImportError: If Isaac Lab is not available.
    """
    try:
        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.assets import ArticulationCfg
        import isaaclab.sim.spawners as spawners
    except ImportError as e:
        raise ImportError(
            "Isaac Lab 2.3.2 required for scene building. "
            "Ensure Isaac Lab is installed and accessible."
        ) from e

    scene_cfg = _create_scene_cfg(config)
    scene = InteractiveScene(scene_cfg)
    return scene


def _create_scene_cfg(config: SceneConfig):
    """Create InteractiveSceneCfg from SceneConfig."""
    try:
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.assets import ArticulationCfg
        from isaaclab.utils import configclass
        import isaaclab.sim as sim_utils
    except ImportError as e:
        raise ImportError("Isaac Lab 2.3.2 required.") from e

    @configclass
    class TVCSceneCfg(InteractiveSceneCfg):
        # Ground plane
        ground = sim_utils.GroundPlaneCfg()

        # EDF drone articulation
        drone: ArticulationCfg = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/drone",
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
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=0,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 5.0, 0.0),  # Start 5m above ground in Isaac y-up frame
                joint_pos={".*": 0.0},
            ),
        )

    return TVCSceneCfg(
        num_envs=config.num_envs,
        env_spacing=config.env_spacing,
        replicate_physics=config.replicate_physics,
    )
