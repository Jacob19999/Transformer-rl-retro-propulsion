"""fin_telemetry.py — GPU tensor ring buffer for per-fin debug telemetry.

Records per-fin, per-step diagnostic data during simulation for offline analysis.
All tensors remain on GPU during recording; CPU transfer only on flush.

Usage in EdfLandingTask (when debug.fin_telemetry: true in config):
    telemetry = FinTelemetryBuffer(num_envs=N, max_steps=600, device="cuda")
    # ... each step:
    telemetry.record(step, cmd_angle, meas_angle, ...)
    # ... on episode end:
    if cfg.debug.fin_telemetry_save:
        telemetry.flush("runs/telemetry/ep0.pt")
    telemetry.reset()

Data schema per env per fin per step (from data-model.md):
  cmd_angle    (N, 4)     rad    commanded fin deflection
  meas_angle   (N, 4)     rad    measured joint position (from PhysX)
  link_pos     (N, 4, 3)  m      fin link world position
  link_quat    (N, 4, 4)  -      fin link world orientation (wxyz)
  exhaust_vel  (N, 4)     m/s    local exhaust stream velocity at fin
  aoa          (N, 4)     rad    angle of attack
  aero_force   (N, 4, 3)  N      applied aerodynamic force (world frame)
  joint_wrench (N, 4, 6)  N,N·m  incoming joint wrench (force[3] + torque[3])

Memory: ~80 floats per env per step → 384 KB per env for a 1200-step episode.
"""

from __future__ import annotations

from pathlib import Path

import torch


class FinTelemetryBuffer:
    """GPU tensor ring buffer for per-fin debug telemetry.

    All write operations are on-GPU; no CPU transfers during recording.
    Flush transfers the buffer to CPU and saves as a .pt file.
    """

    NUM_FINS = 4

    def __init__(
        self,
        num_envs: int,
        max_steps: int,
        device: str | torch.device,
    ) -> None:
        """Allocate GPU tensor buffers for the full episode.

        Args:
            num_envs: Number of parallel environments.
            max_steps: Maximum steps per episode (ring buffer depth).
            device: PyTorch device string (e.g. "cuda:0") or device object.
        """
        self._num_envs = num_envs
        self._max_steps = max_steps
        self._device = device
        self._step_count = 0

        N, F, S = num_envs, self.NUM_FINS, max_steps
        self._cmd_angle    = torch.zeros((S, N, F),    dtype=torch.float32, device=device)
        self._meas_angle   = torch.zeros((S, N, F),    dtype=torch.float32, device=device)
        self._link_pos     = torch.zeros((S, N, F, 3), dtype=torch.float32, device=device)
        self._link_quat    = torch.zeros((S, N, F, 4), dtype=torch.float32, device=device)
        self._exhaust_vel  = torch.zeros((S, N, F),    dtype=torch.float32, device=device)
        self._aoa          = torch.zeros((S, N, F),    dtype=torch.float32, device=device)
        self._aero_force   = torch.zeros((S, N, F, 3), dtype=torch.float32, device=device)
        self._joint_wrench = torch.zeros((S, N, F, 6), dtype=torch.float32, device=device)

    def record(
        self,
        step: int,
        cmd_angle: torch.Tensor,
        meas_angle: torch.Tensor,
        link_pos: torch.Tensor,
        link_quat: torch.Tensor,
        exhaust_vel: torch.Tensor,
        aoa: torch.Tensor,
        aero_force: torch.Tensor,
        joint_wrench: torch.Tensor,
    ) -> None:
        """Write one step of telemetry into the ring buffer.

        All tensors must already be on the correct device.  No CPU transfer.

        Args:
            step: Step index within the episode (0-based).
            cmd_angle:    (N, 4) rad
            meas_angle:   (N, 4) rad
            link_pos:     (N, 4, 3) m
            link_quat:    (N, 4, 4) wxyz
            exhaust_vel:  (N, 4) m/s
            aoa:          (N, 4) rad
            aero_force:   (N, 4, 3) N
            joint_wrench: (N, 4, 6) N and N·m
        """
        idx = step % self._max_steps
        self._cmd_angle[idx].copy_(cmd_angle)
        self._meas_angle[idx].copy_(meas_angle)
        self._link_pos[idx].copy_(link_pos)
        self._link_quat[idx].copy_(link_quat)
        self._exhaust_vel[idx].copy_(exhaust_vel)
        self._aoa[idx].copy_(aoa)
        self._aero_force[idx].copy_(aero_force)
        self._joint_wrench[idx].copy_(joint_wrench)
        self._step_count = min(step + 1, self._max_steps)

    def flush(self, path: str | Path) -> None:
        """Transfer buffer to CPU and save as a .pt file.

        The saved dict has keys matching the data schema field names.
        Tensors have shape (steps, num_envs, ...) where steps <= max_steps.

        Args:
            path: Destination file path (parent directories are created).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        steps = self._step_count
        data = {
            "cmd_angle":    self._cmd_angle[:steps].cpu(),
            "meas_angle":   self._meas_angle[:steps].cpu(),
            "link_pos":     self._link_pos[:steps].cpu(),
            "link_quat":    self._link_quat[:steps].cpu(),
            "exhaust_vel":  self._exhaust_vel[:steps].cpu(),
            "aoa":          self._aoa[:steps].cpu(),
            "aero_force":   self._aero_force[:steps].cpu(),
            "joint_wrench": self._joint_wrench[:steps].cpu(),
            "num_envs":     self._num_envs,
            "num_fins":     self.NUM_FINS,
            "steps":        steps,
        }
        torch.save(data, str(path))

    def reset(self) -> None:
        """Clear the buffer for a new episode (in-place zero fill on GPU)."""
        self._step_count = 0
        self._cmd_angle.zero_()
        self._meas_angle.zero_()
        self._link_pos.zero_()
        self._link_quat.zero_()
        self._exhaust_vel.zero_()
        self._aoa.zero_()
        self._aero_force.zero_()
        self._joint_wrench.zero_()
