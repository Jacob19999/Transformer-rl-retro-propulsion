"""
Shared DirectRLEnv base class.

Provides: config loading, YAML deep-merge, common initialization
(scene, asset, actuators, contacts), shared step infrastructure,
and config validation (reject null to-be-calibrated values before training).
"""

from __future__ import annotations
import yaml
from pathlib import Path
from typing import Any

from tvc_env.envs.task_registry import deep_merge


class BaseEnvConfig:
    """Container for merged environment configuration."""

    def __init__(
        self,
        task_name: str,
        env_config_path: str | Path | None = None,
        disturbance_config_path: str | Path | None = None,
        overrides: dict | None = None,
        sim_root: str | Path | None = None,
    ):
        from tvc_env.envs.task_registry import load_merged_config
        import yaml

        if sim_root is None:
            sim_root = Path(__file__).parents[2]

        # Load env config if provided
        env_config = None
        if env_config_path is not None:
            with open(env_config_path, "r") as f:
                env_config = yaml.safe_load(f)

        # Load disturbance config if provided
        disturbance_config = None
        if disturbance_config_path is not None:
            with open(disturbance_config_path, "r") as f:
                disturbance_config = yaml.safe_load(f)

        self.config = load_merged_config(
            task_name=task_name,
            env_config=env_config,
            disturbance_config=disturbance_config,
            overrides=overrides,
            sim_root=sim_root,
        )

        # Extract common settings
        env = self.config.get("env", {})
        self.num_envs: int = env.get("num_envs", 1)
        self.env_spacing: float = env.get("env_spacing", 4.0)
        self.gizmos_enabled: bool = env.get("gizmos_enabled", False)
        self.dispatch_mode: str = env.get("dispatch_mode", "per_link_force")
        self.physics_dt: float = env.get("physics_dt", 1.0 / 120.0)
        self.decimation: int = env.get("decimation", 4)
        self.task_name: str = task_name

    def validate_for_training(self) -> None:
        """Validate config is safe for RL training (no null to-be-calibrated values).

        Raises:
            ValueError: If any null to-be-calibrated parameter is found.
        """
        # Check EDF params
        edf_config_path = Path(__file__).parents[2] / "configs/params/edf_90mm.yaml"
        if edf_config_path.exists():
            with open(edf_config_path, "r") as f:
                edf_cfg = yaml.safe_load(f).get("edf", {})
            null_fields = [k for k, v in edf_cfg.items() if v is None]
            if null_fields:
                raise ValueError(
                    f"Cannot start training: EDF config has null to-be-calibrated values: "
                    f"{null_fields}. Run bench calibration first and update edf_90mm.yaml."
                )


class TVCEnvBase:
    """Base class with common initialization logic for the TVC environment.

    Concrete environments should subclass this and the Isaac Lab DirectRLEnv.
    """

    def __init__(self, config: BaseEnvConfig):
        self._config = config
        self._step_count = None
        self._reset_manager = None
        self._contact_sm = None
        self._crash_detector = None

    def _initialize_physics_systems(
        self,
        scene,
        articulation,
        metadata: dict[str, Any],
        vehicle_config: dict[str, Any],
        edf_config: dict[str, Any],
        servo_config: dict[str, Any],
        device,
    ) -> None:
        """Initialize all physics subsystems after scene is built."""
        from tvc_env.asset.articulation_map import build_articulation_map
        from tvc_env.dynamics.fin_geometry import load_cop_positions, load_hinge_axes
        from tvc_env.dynamics.fin_aero import FinAeroModel
        from tvc_env.dynamics.fin_force_dispatch import FinForceDispatch
        from tvc_env.dynamics.actuator_servo import ServoModel
        from tvc_env.dynamics.propulsion_edf import EDFModel
        from tvc_env.dynamics.wind_model import WindModel
        from tvc_env.sim.body_interface import BodyInterface
        from tvc_env.sim.link_force_interface import LinkForceInterface
        from tvc_env.sim.wrench_dispatch import WrenchDispatch
        from tvc_env.sim.contacts import ContactStateMachine
        from tvc_env.sim.crash_logic import CrashDetector
        from tvc_env.sim.reset_logic import ResetManager

        num_envs = self._config.num_envs

        # Build articulation map
        art_map = build_articulation_map(metadata, articulation)

        # Geometry
        cops = load_cop_positions(metadata, device=device)
        hinge_axes = load_hinge_axes(metadata, device=device)

        # Physics models
        aero_model = FinAeroModel.from_config(vehicle_config, edf_config)
        servo_model = ServoModel(
            tau_servo=servo_config.get("servo", servo_config).get("tau_servo", 0.05),
            max_angular_velocity=servo_config.get("servo", servo_config).get("max_angular_velocity", 7.54),
            max_command_angle=servo_config.get("servo", servo_config).get("max_command_angle", 0.262),
            deadband=servo_config.get("servo", servo_config).get("deadband", 0.017),
        )
        edf_params = edf_config.get("edf", edf_config)
        edf_model = EDFModel(
            max_thrust=edf_params.get("max_thrust", 48.0),
            tau_motor=edf_params.get("tau_motor", 0.15),
            omega_max=edf_params.get("omega_max") or 3000.0,
            d_omega_max=edf_params.get("d_omega_max"),
            k_T=edf_params.get("k_T"),
            k_Q=edf_params.get("k_Q"),
            rotor_inertia=edf_params.get("rotor_inertia", 0.0005),
        )

        # Wind model (only if disturbance config enables it)
        dist_cfg = self._config.config.get("disturbances", {})
        if dist_cfg.get("enabled") and dist_cfg.get("wind", {}).get("enabled"):
            wind_model = WindModel.from_disturbance_config(
                self._config.config, device=device,
            )
        else:
            wind_model = None

        # Sim interfaces
        body_iface = BodyInterface(articulation, art_map)
        link_force_iface = LinkForceInterface(articulation, art_map, cops)
        wrench_dispatch = WrenchDispatch(
            mode=self._config.dispatch_mode,
            link_force_interface=link_force_iface,
            body_link_index=art_map.body_index,
        )

        # Contact state machine
        contact_sm = ContactStateMachine(num_envs=num_envs, device=device)
        crash_detector = CrashDetector.from_task_config(self._config.config)

        # Reset manager
        reset_mgr = ResetManager(body_iface, servo_model, edf_model, contact_sm, self._config.config)
        reset_mgr.initialize(num_envs, device)

        # Store as instance attributes
        self._art_map = art_map
        self._body_iface = body_iface
        self._link_force_iface = link_force_iface
        self._wrench_dispatch = wrench_dispatch
        self._wind_model = wind_model
        self._aero_model = aero_model
        self._fin_dispatch = FinForceDispatch(aero_model, cops, hinge_axes)
        self._servo_model = servo_model
        self._edf_model = edf_model
        self._contact_sm = contact_sm
        self._crash_detector = crash_detector
        self._reset_manager = reset_mgr
        self._cops = cops
        self._hinge_axes = hinge_axes
