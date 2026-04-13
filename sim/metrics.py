from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from abstraction.robot_agent import ActionType, CapabilityProfile
from sim.tasks import Target


@dataclass
class RoleMetrics:
    """Episode-level role-differentiation observables."""

    scout_rate_ghost: float = 0.0
    interact_rate_spot: float = 0.0


class MetricsTracker:
    """Accumulates per-episode role metrics for logging."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._ghost_scout_hits = 0
        self._ghost_scout_total = 0
        self._spot_interact_hits = 0
        self._spot_interact_total = 0

    def update(
        self,
        action_types: dict[str, ActionType],
        commanded: dict[str, tuple[float, float]],
        targets: list[Target],
        capabilities: dict[str, CapabilityProfile],
    ) -> RoleMetrics:
        """Update counters and return current metrics snapshot."""
        ghost_action = action_types.get("ghost")
        if ghost_action in (ActionType.MOVE_TO, ActionType.SCOUT):
            self._ghost_scout_total += 1
            gx, gy = commanded["ghost"]
            if any(not t.revealed and np.hypot(gx - t.x, gy - t.y) <= 2.0 for t in targets):
                self._ghost_scout_hits += 1

        spot_action = action_types.get("spot")
        if spot_action == ActionType.MOVE_TO:
            self._spot_interact_total += 1
            sx, sy = commanded["spot"]
            if any(
                t.revealed and not t.interacted and np.hypot(sx - t.x, sy - t.y) <= 0.5
                for t in targets
            ):
                self._spot_interact_hits += 1

        return RoleMetrics(
            scout_rate_ghost=self._ghost_scout_hits / max(1, self._ghost_scout_total),
            interact_rate_spot=self._spot_interact_hits / max(1, self._spot_interact_total),
        )
