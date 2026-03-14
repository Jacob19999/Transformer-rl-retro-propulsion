"""Single-source Isaac runtime parameters loaded from vehicle YAML."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from simulation.config_loader import load_config


@dataclass(frozen=True)
class IsaacVehicleParams:
    """Vehicle parameters Isaac runtime code consumes directly."""

    gravity: float
    t_max: float
    delta_max: float
    tau_motor: float
    tau_servo: float
    k_thrust: float
    k_torque: float
    i_fan: float
    v_exhaust_nominal: float
    cl_alpha: float
    fin_area: float


def _resolve_vehicle_section(config_path: str | Path) -> tuple[Path, dict]:
    path = Path(config_path).resolve()
    raw = load_config(path)
    return path, raw.get("vehicle", raw)


@lru_cache(maxsize=None)
def load_isaac_vehicle_params(config_path: str | Path) -> IsaacVehicleParams:
    """Load the Isaac-used subset of the vehicle config once per path."""
    _, vehicle = _resolve_vehicle_section(config_path)
    edf_cfg = vehicle.get("edf", {})
    fins_cfg = vehicle.get("fins", {})
    servo_cfg = fins_cfg.get("servo", {})
    return IsaacVehicleParams(
        gravity=float(vehicle.get("gravity", 9.81)),
        t_max=float(edf_cfg.get("max_static_thrust", 45.0)),
        delta_max=float(fins_cfg.get("max_deflection", 0.349)),
        tau_motor=float(edf_cfg.get("tau_motor", 0.10)),
        tau_servo=float(servo_cfg.get("tau_servo", 0.04)),
        k_thrust=float(edf_cfg.get("k_thrust", 4.55e-7)),
        k_torque=float(edf_cfg.get("k_torque", 1.0e-10)),
        i_fan=float(edf_cfg.get("I_fan", 3.0e-7)),
        v_exhaust_nominal=float(fins_cfg.get("V_exhaust_nominal", 70.0)),
        cl_alpha=float(fins_cfg.get("CL_alpha", 6.283)),
        fin_area=float(fins_cfg.get("planform_area", 0.003575)),
    )

