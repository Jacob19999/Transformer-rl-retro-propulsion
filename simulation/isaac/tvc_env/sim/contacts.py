"""
4-state contact state machine for landing/crash detection.

States per data-model ContactStateMachine:
  AIRBORNE (0) → ground contact starts → GROUND_CONTACT_CANDIDATE (1)
  CANDIDATE (1) → dwell criteria met → LANDED (2)
  CANDIDATE (1) → bounce → AIRBORNE (0)
  CANDIDATE/AIRBORNE → crash trigger → CRASHED (3)

State is a vectorized integer tensor of shape (num_envs,).
All dwell thresholds are configurable from task YAML.
"""

from __future__ import annotations
import torch
from torch import Tensor
from tvc_env.common.constants import ContactState


class ContactStateMachine:
    """Vectorized 4-state contact state machine for all environments."""

    def __init__(
        self,
        num_envs: int,
        dwell_frames: int = 10,         # Number of consecutive contact frames required for LANDED
        min_contact_force: float = 1.0, # N, minimum contact force to count as contact
        device: torch.device = None,
    ):
        self.num_envs = num_envs
        self.dwell_frames = dwell_frames
        self.min_contact_force = min_contact_force
        self.device = device

        # State tensor
        self._state = torch.zeros(num_envs, dtype=torch.int32, device=device)
        # Dwell counter: number of consecutive candidate frames
        self._dwell_count = torch.zeros(num_envs, dtype=torch.int32, device=device)

    @classmethod
    def from_task_config(cls, task_config: dict, num_envs: int, device=None) -> "ContactStateMachine":
        """Create state machine from task YAML config."""
        task = task_config.get("task", task_config)
        contact_cfg = task.get("contact", {})
        return cls(
            num_envs=num_envs,
            dwell_frames=contact_cfg.get("dwell_frames", 10),
            min_contact_force=contact_cfg.get("min_contact_force", 1.0),
            device=device,
        )

    @property
    def state(self) -> Tensor:
        """Current contact state for all environments, shape (num_envs,)."""
        return self._state.clone()

    def update(
        self,
        in_contact: Tensor,       # (num_envs,) bool — any landing contact
        is_crashed: Tensor,       # (num_envs,) bool — crash detected by crash_logic
        contact_force: Tensor,    # (num_envs,) float — contact force magnitude (N)
    ) -> Tensor:
        """Update state machine for one step.

        State transitions:
          AIRBORNE + contact → CANDIDATE
          CANDIDATE + no contact → AIRBORNE
          CANDIDATE + dwell_frames consecutive → LANDED
          Any non-LANDED + crash → CRASHED
          CRASHED stays CRASHED

        Args:
            in_contact: Whether landing contact regions are touching ground.
            is_crashed: Whether crash criteria are met.
            contact_force: Contact force magnitude per environment.

        Returns:
            Updated state tensor (num_envs,).
        """
        prev_state = self._state.clone()
        force_contact = contact_force.to(device=self.device).abs() >= self.min_contact_force
        in_contact_bool = in_contact.bool() & force_contact
        is_crashed_bool = is_crashed.bool()

        # Reset dwell counter when leaving contact
        left_contact = (prev_state == ContactState.GROUND_CONTACT_CANDIDATE) & ~in_contact_bool
        self._dwell_count[left_contact] = 0

        # Transition: AIRBORNE → CANDIDATE on contact
        airborne_to_candidate = (prev_state == ContactState.AIRBORNE) & in_contact_bool
        self._state[airborne_to_candidate] = ContactState.GROUND_CONTACT_CANDIDATE

        # Increment dwell counter for CANDIDATE environments still in contact
        still_candidate = (self._state == ContactState.GROUND_CONTACT_CANDIDATE) & in_contact_bool
        self._dwell_count[still_candidate] += 1

        # Transition: CANDIDATE → AIRBORNE on contact loss
        candidate_no_contact = (self._state == ContactState.GROUND_CONTACT_CANDIDATE) & ~in_contact_bool
        self._state[candidate_no_contact] = ContactState.AIRBORNE
        self._dwell_count[candidate_no_contact] = 0

        # Transition: CANDIDATE → LANDED when dwell criteria met
        dwell_met = (
            (self._state == ContactState.GROUND_CONTACT_CANDIDATE) &
            (self._dwell_count >= self.dwell_frames)
        )
        self._state[dwell_met] = ContactState.LANDED

        # Transition: ANY non-LANDED → CRASHED on crash detection
        not_landed = self._state != ContactState.LANDED
        crashed_now = not_landed & is_crashed_bool
        self._state[crashed_now] = ContactState.CRASHED

        return self._state.clone()

    def reset(self, env_ids: Tensor | None = None) -> None:
        """Reset state machine to AIRBORNE for specified environments.

        Args:
            env_ids: Optional tensor of environment indices to reset. None = reset all.
        """
        if env_ids is None:
            self._state.fill_(ContactState.AIRBORNE)
            self._dwell_count.fill_(0)
        else:
            self._state[env_ids] = ContactState.AIRBORNE
            self._dwell_count[env_ids] = 0

    def is_landed(self) -> Tensor:
        """Return bool tensor where True = LANDED state."""
        return self._state == ContactState.LANDED

    def is_crashed(self) -> Tensor:
        """Return bool tensor where True = CRASHED state."""
        return self._state == ContactState.CRASHED

    def is_terminal(self) -> Tensor:
        """Return bool tensor where True = LANDED or CRASHED (episode should end)."""
        return (self._state == ContactState.LANDED) | (self._state == ContactState.CRASHED)
