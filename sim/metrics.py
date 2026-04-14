from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from abstraction.robot_agent import ActionType, CapabilityProfile
from sim.tasks import Target


@dataclass
class RoleMetrics:
    """Episode-level role-differentiation observables."""

    handoff_rate: float = 0.0
    spot_interact_given_valid: float = 0.0
    ghost_scout_given_unrevealed: float = 0.0
    # Backwards-compat aliases populated with semantically closest new metric
    scout_rate_ghost: float = 0.0    # = ghost_scout_given_unrevealed
    interact_rate_spot: float = 0.0  # = handoff_rate


class MetricsTracker:
    """Accumulates per-episode coordination-quality metrics."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._ghost_toward_unrevealed: int = 0
        self._unrevealed_steps: int = 0
        self._spot_interact_picked: int = 0
        self._spot_mask_on: int = 0

    def update(
        self,
        action_types: dict[str, ActionType],
        targets: list[Target],
        capabilities: dict[str, CapabilityProfile],
        positions: dict[str, tuple[float, float]],
        velocities: dict[str, tuple[float, float]],
        pre_step_masks: dict[str, np.ndarray],
    ) -> RoleMetrics:
        # ------------------------------------------------------------------ #
        # Metric 1: handoff_rate                                               #
        # Computed directly from target state — no accumulator needed.         #
        # ------------------------------------------------------------------ #
        n_interacted = sum(1 for t in targets if t.interacted)
        n_handoff = sum(
            1 for t in targets if t.interacted and t.revealed_by == "ghost"
        )
        handoff = float(n_handoff) / max(1, n_interacted) if n_interacted > 0 else 0.0

        # ------------------------------------------------------------------ #
        # Metric 2: spot_interact_given_valid                                  #
        # Did Spot pick INTERACT every time its mask allowed it?               #
        # ------------------------------------------------------------------ #
        spot_mask = pre_step_masks.get("spot", np.zeros(5, dtype=np.float32))
        if spot_mask[2] > 0.5:  # INTERACT was available when action was chosen
            self._spot_mask_on += 1
            if action_types.get("spot") == ActionType.INTERACT:
                self._spot_interact_picked += 1

        # ------------------------------------------------------------------ #
        # Metric 3: ghost_scout_given_unrevealed                              #
        # Did Ghost move within 45° of the nearest unrevealed target?         #
        # ------------------------------------------------------------------ #
        unrevealed = [t for t in targets if not t.revealed]
        if unrevealed:
            self._unrevealed_steps += 1
            ghost_action = action_types.get("ghost")
            if ghost_action in (ActionType.SCOUT, ActionType.MOVE_TO):
                gx, gy = positions.get("ghost", (0.0, 0.0))
                gvx, gvy = velocities.get("ghost", (0.0, 0.0))
                speed = float(np.hypot(gvx, gvy))
                if speed > 1e-6:
                    nearest = min(unrevealed, key=lambda t: np.hypot(gx - t.x, gy - t.y))
                    bearing = np.array([nearest.x - gx, nearest.y - gy], dtype=np.float64)
                    bearing_len = float(np.linalg.norm(bearing))
                    if bearing_len > 1e-9:
                        bearing_dir = bearing / bearing_len
                        vel_dir = np.array([gvx, gvy], dtype=np.float64) / speed
                        if float(np.dot(vel_dir, bearing_dir)) > 0.707:
                            self._ghost_toward_unrevealed += 1

        ghost_scout = (
            self._ghost_toward_unrevealed / self._unrevealed_steps
            if self._unrevealed_steps > 0 else 0.0
        )
        spot_interact = (
            self._spot_interact_picked / self._spot_mask_on
            if self._spot_mask_on > 0 else 0.0
        )

        return RoleMetrics(
            handoff_rate=handoff,
            spot_interact_given_valid=spot_interact,
            ghost_scout_given_unrevealed=ghost_scout,
            scout_rate_ghost=ghost_scout,
            interact_rate_spot=handoff,
        )
