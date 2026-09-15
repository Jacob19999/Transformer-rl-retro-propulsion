"""Vectorized LiPo 1-RC equivalent circuit coupled to motor energy and voltage.

SOC uses coulomb counting; OCV(SOC), series resistance and an RC polarization
voltage set the loaded bus. No regenerative charging is assumed. Motor speed
is voltage-limited and cannot receive more kinetic energy than delivered shaft
power permits. This optional model does not expand the calibrated EDF RPM cap.
The structure follows a Thevenin equivalent circuit; values in battery_6s.yaml
are explicitly estimated and require pack/motor bench identification.
"""
from __future__ import annotations
import math
import torch


class LiPoBattery:
    def __init__(self, config, num_envs, device='cpu'):
        self.config = dict(config)
        c = self.config
        for name in ('cells', 'capacity_ah', 'cell_resistance_ohm', 'polarization_time_s',
                     'max_current_a', 'reference_voltage_v', 'thermal_capacity_j_k'):
            if not math.isfinite(float(c[name])) or float(c[name]) <= 0:
                raise ValueError(f'Battery {name} must be finite and positive')
        if not 0 < c['motor_efficiency'] <= 1 or not 0 <= c['initial_soc'] <= 1:
            raise ValueError('Invalid battery efficiency or initial SOC')
        for name in ('polarization_resistance_ohm', 'shaft_power_at_max_w',
                     'auxiliary_power_w', 'cooling_w_k'):
            if not math.isfinite(float(c[name])) or float(c[name]) < 0:
                raise ValueError(f'Battery {name} must be finite and nonnegative')
        knots, volts = c['soc_knots'], c['cell_ocv_knots']
        if (len(knots) != len(volts) or len(knots) < 2 or knots[0] != 0 or knots[-1] != 1
                or any(a >= b for a, b in zip(knots, knots[1:]))
                or any(a > b for a, b in zip(volts, volts[1:]))):
            raise ValueError('OCV knots must increase over SOC [0,1]')
        self.soc_knots = torch.tensor(knots, device=device)
        self.ocv_knots = torch.tensor(volts, device=device)
        self.soc = torch.full((num_envs,), float(c['initial_soc']), device=device)
        self.polarization_v = torch.zeros_like(self.soc)
        self.temperature_c = torch.full_like(self.soc, float(c['ambient_c']))
        self.energy_wh = torch.zeros_like(self.soc)
        self.current_a = torch.zeros_like(self.soc)
        self.voltage_v = self.ocv()
        self.power_w = torch.zeros_like(self.soc)
        self.cutoff = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.current_limited = torch.zeros_like(self.cutoff)

    def ocv(self):
        x = self.soc.clamp(0, 1)
        i = torch.searchsorted(self.soc_knots, x).clamp(1, len(self.soc_knots) - 1)
        f = (x - self.soc_knots[i - 1]) / (self.soc_knots[i] - self.soc_knots[i - 1])
        return torch.lerp(self.ocv_knots[i - 1], self.ocv_knots[i], f) * self.config['cells']

    def reset(self, env_ids):
        self.soc[env_ids] = self.config['initial_soc']
        self.polarization_v[env_ids] = 0
        self.temperature_c[env_ids] = self.config['ambient_c']
        self.energy_wh[env_ids] = 0
        self.current_a[env_ids] = self.power_w[env_ids] = 0
        self.cutoff[env_ids] = self.current_limited[env_ids] = False
        self.voltage_v[env_ids] = self.ocv()[env_ids]

    def solve_load(self, requested_w):
        """High-voltage root of P=I(E-I*R), with current/power limits."""
        c = self.config
        resistance = c['cells'] * c['cell_resistance_ohm']
        emf = (self.ocv() - self.polarization_v).clamp(min=0)
        max_i = torch.minimum(torch.full_like(emf, c['max_current_a']), emf / (2 * resistance))
        max_power = max_i * (emf - max_i * resistance)
        requested = requested_w.clamp(min=0)
        power = torch.minimum(requested, max_power)
        discriminant = (emf.square() - 4 * resistance * power).clamp(min=0)
        current = 2 * power / (emf + discriminant.sqrt()).clamp(min=1e-8)
        voltage = (emf - current * resistance).clamp(min=0)
        cutoff = self.cutoff | (self.soc <= 0) | (voltage < c['cutoff_cell_v'] * c['cells'])
        voltage = torch.where(cutoff, 0., voltage)
        current = torch.where(cutoff, 0., current)
        return voltage, current, cutoff, requested > max_power + 1e-4

    def integrate(self, voltage, current, cutoff, limited, dt):
        c = self.config
        self.voltage_v = voltage
        self.current_a = current
        self.power_w = voltage * current
        self.cutoff = cutoff
        self.current_limited = limited
        self.soc = (self.soc - current * dt / (3600 * c['capacity_ah'])).clamp(0, 1)
        self.energy_wh += self.power_w * dt / 3600
        decay = math.exp(-dt / c['polarization_time_s'])
        heat = current.square() * c['cells'] * c['cell_resistance_ohm']
        if c['polarization_resistance_ohm'] > 0:
            heat += self.polarization_v.square() / c['polarization_resistance_ohm']
        self.polarization_v = decay * self.polarization_v + (1 - decay) * current * c['polarization_resistance_ohm']
        self.temperature_c += dt * (heat - c['cooling_w_k'] * (self.temperature_c - c['ambient_c'])) / c['thermal_capacity_j_k']

    def update_motor(self, edf, omega, throttle, dt):
        """Advance one motor/pack substep, resolving load/voltage together.

        Five damped fixed-point iterations resolve the voltage-dependent demand.
        If the pack reaches a limit, an energy bound reduces rotor speed rather
        than continuing to grant the ideal-voltage acceleration/thrust.
        """
        if dt <= 0:
            raise ValueError('Battery timestep must be positive')
        c = self.config
        voltage = self.voltage_v.clone()
        aero = c['shaft_power_at_max_w'] * (omega / edf.omega_max).clamp(0, 1).pow(3)

        def candidate(bus):
            effective = (throttle * bus / c['reference_voltage_v']).clamp(0, 1)
            next_omega = edf.update(omega, effective, dt)
            kinetic = .5 * edf.rotor_inertia * (next_omega.square() - omega.square()) / dt
            power = (aero + kinetic).clamp(min=0) / c['motor_efficiency'] + c['auxiliary_power_w']
            return next_omega, power

        for _ in range(5):
            _, power = candidate(voltage)
            loaded, _, _, _ = self.solve_load(power)
            voltage = .5 * voltage + .5 * loaded
        target, requested = candidate(voltage)
        loaded, current, cutoff, limited = self.solve_load(requested)
        # Recompute the voltage bound with the solved load, then apply both
        # voltage and energy constraints. Missing electrical energy cannot turn
        # into rotor kinetic energy. Braking energy is dissipated, not recharged.
        voltage_target, _ = candidate(loaded)
        target = torch.minimum(target, voltage_target)
        shaft = (loaded * current - c['auxiliary_power_w']).clamp(min=0) * c['motor_efficiency']
        energy_cap = (omega.square() + 2 * dt * (shaft - aero) / edf.rotor_inertia).clamp(min=0).sqrt()
        next_omega = torch.minimum(target, energy_cap).clamp(0, edf.omega_max)
        self.integrate(loaded, current, cutoff, limited, dt)
        return next_omega

    def telemetry(self):
        return dict(soc=self.soc, voltage_v=self.voltage_v, ocv_v=self.ocv(),
                    current_a=self.current_a, power_w=self.power_w,
                    energy_wh=self.energy_wh, temperature_c=self.temperature_c,
                    cutoff=self.cutoff, current_limited=self.current_limited)

    def observation(self):
        """Dimensionless electrical state appended after the original 24 inputs.

        SOC and polarization expose the hidden states controlling future sag;
        voltage and current describe the present operating point. No commands
        or simulated future state are included.
        """
        c = self.config
        return torch.stack([self.soc, self.voltage_v / c['reference_voltage_v'],
                            self.current_a / c['max_current_a'],
                            self.polarization_v / c['reference_voltage_v']], dim=-1)
