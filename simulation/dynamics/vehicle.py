"""
6-DOF rigid-body vehicle dynamics assembly.

Implements vehicle.md §8 and tracker Stage 11:
- Top-level class owning state, integrator, and all sub-models
- State: 18 scalars [p(3), v_b(3), q(4), omega(3), T(1), delta_actual(4)]
- RK4 integration with quaternion normalization every N steps
- Force/torque from thrust, aero, fins, with servo dynamics

Wind and atmospheric conditions are provided by an external EnvironmentModel.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from .aero_model import AeroModel, AeroModelConfig
from .fin_model import FinModel
from .integrator import RK4Integrator
from .mass_properties import MassProperties, compute_mass_properties
from .quaternion_utils import quat_mult, quat_to_dcm
from .servo_model import ServoModel, ServoModelConfig
from .thrust_model import ThrustModel

if TYPE_CHECKING:
    from simulation.environment.environment_model import EnvironmentModel


STATE_DIM = 18
CONTROL_DIM = 5  # [T_cmd, delta_1, delta_2, delta_3, delta_4]


class VehicleDynamics:
    """6-DOF rigid-body plant model for EDF drone TVC landing sim.

    Wind and atmospheric conditions are provided by an external EnvironmentModel.
    """

    def __init__(
        self,
        config: Mapping[str, Any] | str | Path,
        env: "EnvironmentModel",
    ) -> None:
        """Initialize vehicle dynamics from config and environment.

        Parameters
        ----------
        config
            Vehicle config dict (may be nested under "vehicle" key), or path to YAML.
        env
            EnvironmentModel providing wind and atmosphere (sample_at_state).
        """
        cfg = self._load_config(config)
        self._cfg = cfg
        self.env = env

        # Mass properties (init-only)
        cad = cfg.get("cad_override") or {}
        if cad.get("use_cad_override", False):
            self.mass_props = MassProperties.from_cad(cad)
        else:
            primitives = cfg.get("primitives", [])
            if not primitives:
                raise ValueError("Config must have primitives or use_cad_override.")
            self.mass_props = compute_mass_properties(primitives)

        self.mass = float(self.mass_props.total_mass)
        self.I = np.asarray(self.mass_props.inertia_tensor, dtype=float)
        self.I_inv = np.asarray(self.mass_props.inertia_tensor_inv, dtype=float)
        self.com = np.asarray(self.mass_props.center_of_mass, dtype=float)

        # Sub-models
        edf = cfg.get("edf", {})
        if not edf:
            raise ValueError("Config must have edf section.")
        self.thrust_model = ThrustModel.from_edf_config(edf)

        aero = cfg.get("aero", {})
        self.aero_model = AeroModel(AeroModelConfig.from_config(aero), self.mass_props)

        fins = cfg.get("fins", {})
        self.fin_model = FinModel.from_config(fins, edf, self.com)
        self.servo_model = ServoModel(ServoModelConfig.from_config(fins))

        self.I_fan = float(edf["I_fan"])
        self.g = float(cfg.get("gravity", 9.81))
        self.dt = float(cfg.get("dt", 0.005))
        quat_interval = int(cfg.get("quat_normalize_interval", 10))

        self.integrator = RK4Integrator(
            quat_slice=slice(6, 10),
            quat_normalize_every_n=quat_interval,
        )

        # State: [p(3), v_b(3), q(4), omega(3), T(1), delta_actual(4)] = 18
        self.state = np.zeros(STATE_DIM, dtype=float)
        self.time = 0.0

    def _load_config(self, config: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
        if isinstance(config, (str, Path)):
            from simulation.config_loader import load_config
            loaded = load_config(config)
            return loaded.get("vehicle", loaded)
        if "vehicle" in config:
            return config["vehicle"]
        return config

    def _unpack(self, y: np.ndarray) -> tuple:
        """Extract state components from flat 18-dim array."""
        y = np.asarray(y, dtype=float)
        if y.size < STATE_DIM:
            raise ValueError(f"State must have at least {STATE_DIM} elements, got {y.size}.")
        p = y[0:3]
        v_b = y[3:6]
        q = y[6:10]
        omega = y[10:13]
        T = float(y[13])
        delta_actual = y[14:18]
        return p, v_b, q, omega, T, delta_actual

    def derivs(self, y: np.ndarray, u: object, t: float) -> np.ndarray:
        """Compute state derivatives. u = [T_cmd, delta_1..4].

        Queries EnvironmentModel once per call for consistent rho + wind.
        """
        u_arr = np.asarray(u, dtype=float)
        if u_arr.size < CONTROL_DIM:
            raise ValueError(f"Control u must have at least {CONTROL_DIM} elements.")
        T_cmd = float(u_arr[0])
        fin_deltas_cmd = u_arr[1:5]

        p, v_b, q, omega, T, delta_actual = self._unpack(y)
        R = quat_to_dcm(q)

        # Query environment once
        env_vars = self.env.sample_at_state(float(t), p)
        v_wind = np.asarray(env_vars["wind"], dtype=float).reshape(3)
        rho = float(env_vars["rho"])

        # Servo dynamics
        delta_dot = self.servo_model.compute_rate(fin_deltas_cmd, delta_actual)

        # Thrust (altitude h = -p[2] in NED)
        h = float(max(0.01, -p[2]))
        F_thrust_raw, _, T_dot = self.thrust_model.outputs(
            T=T, T_cmd=T_cmd, h=h, rho=rho
        )
        # Thrust opposes gravity: in FRD body frame z is down, so F_thrust = [0,0,-T_eff]
        F_thrust = -np.asarray(F_thrust_raw, dtype=float)
        omega_fan = self.thrust_model.omega_from_thrust(max(T, 0.0))
        # Torque about CoM: r_thrust in config is in body frame; use (r_thrust - com) x F
        r_thrust = self.thrust_model.config.r_thrust
        r_offset_thrust = r_thrust - self.com
        tau_thrust = np.cross(r_offset_thrust, F_thrust)
        tau_motor = self.thrust_model.motor_reaction_torque(T=T, T_dot=T_dot)
        tau_anti = (
            self.thrust_model.steady_state_anti_torque(T=T)
            if self.thrust_model.config.anti_torque_enabled
            else np.zeros(3, dtype=float)
        )

        # Aero
        F_aero, tau_aero = self.aero_model.compute(v_b, R, v_wind, rho=rho)

        # Fins (use actual servo positions)
        F_fins, tau_fins = self.fin_model.compute(delta_actual, omega_fan, rho=rho)

        F_total = F_thrust + F_aero + F_fins
        tau_total = tau_thrust + tau_aero + tau_fins + tau_motor + tau_anti

        # Position kinematics: p_dot = R @ v_b
        p_dot = R @ v_b

        # Translational dynamics: v_dot = F/m + g_b - omega x v_b
        g_inertial = np.array([0.0, 0.0, self.g], dtype=float)
        g_b = R.T @ g_inertial
        v_dot = F_total / self.mass + g_b - np.cross(omega, v_b)

        # Quaternion kinematics: q_dot = 0.5 * q ⊗ [0, omega]
        omega_quat = np.array([0.0, omega[0], omega[1], omega[2]], dtype=float)
        q_dot = 0.5 * quat_mult(q, omega_quat)

        # Rotational dynamics: I*omega_dot + omega x (I*omega) + omega x h_fan = tau
        h_fan = np.array([0.0, 0.0, self.I_fan * omega_fan], dtype=float)
        gyro_body = np.cross(omega, self.I @ omega)
        gyro_fan = np.cross(omega, h_fan)
        omega_dot = self.I_inv @ (tau_total - gyro_body - gyro_fan)

        return np.concatenate([
            p_dot, v_dot, q_dot, omega_dot,
            np.array([T_dot], dtype=float),
            delta_dot,
        ]).astype(float)

    def step(self, u: np.ndarray) -> np.ndarray:
        """Advance state by one RK4 integration step."""
        u_arr = np.asarray(u, dtype=float)
        if u_arr.size < CONTROL_DIM:
            u_arr = np.resize(u_arr, CONTROL_DIM)
            u_arr[1:5] = 0.0
        u_5 = u_arr[:CONTROL_DIM].copy()

        self.state = self.integrator.step(
            self.derivs, self.state, u_5, self.time, self.dt
        )
        self.time += self.dt
        return self.state.copy()

    def reset(
        self,
        initial_state: np.ndarray,
        seed: int | None = None,
    ) -> np.ndarray:
        """Reset to initial conditions for new episode."""
        init = np.asarray(initial_state, dtype=float)
        self.state = np.zeros(STATE_DIM, dtype=float)
        n_copy = min(init.size, 14)  # p,v_b,q,omega,T
        self.state[:n_copy] = init[:n_copy]
        self.state[14:18] = 0.0  # servo positions start neutral
        self.time = 0.0
        self.integrator.reset()

        T0 = float(self.state[13]) if self.state.size > 13 else 0.0
        self.thrust_model.reset(T0=T0)
        self.servo_model.reset(seed=seed)
        return self.state.copy()


__all__ = ["VehicleDynamics", "STATE_DIM", "CONTROL_DIM"]
