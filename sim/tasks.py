from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from abstraction.robot_agent import ActionType, CapabilityProfile, RobotState
from sim.world import World


@dataclass
class Target:
    """Target metadata for reveal/interaction progression."""

    x: float
    y: float
    revealed: bool = False
    interacted: bool = False
    revealed_by: str | None = None  # agent id that first revealed this target


class TaskManager:
    """Random target spawning and target state transitions."""

    def __init__(self, num_targets: int) -> None:
        self.num_targets = int(num_targets)
        self.targets: list[Target] = []

    def reset(self, world: World, rng: np.random.Generator) -> list[Target]:
        self.targets = []
        attempts = 0
        while len(self.targets) < self.num_targets:
            attempts += 1
            if attempts > 10000:
                raise RuntimeError("Failed to place all targets in free space.")
            x = float(rng.uniform(0.0, world.width))
            y = float(rng.uniform(0.0, world.height))
            if not world.is_free(x, y, clearance=1.0):
                continue
            if any(np.hypot(t.x - x, t.y - y) < 2.0 for t in self.targets):
                continue
            self.targets.append(Target(x=x, y=y))
        return self.targets

    def update_reveals(self, positions: dict[str, tuple[float, float]]) -> dict[str, int]:
        """Return per-agent reveal counts {agent_id: n_revealed_this_step}."""
        counts: dict[str, int] = {}
        for target in self.targets:
            if target.revealed:
                continue
            for agent, (x, y) in positions.items():
                if np.hypot(x - target.x, y - target.y) <= 3.6:
                    target.revealed = True
                    target.revealed_by = agent
                    counts[agent] = counts.get(agent, 0) + 1
                    break
        return counts

    def try_interactions(
        self,
        action_types: dict[str, ActionType],
        states: dict[str, RobotState],
        capabilities: dict[str, CapabilityProfile],
        interaction_range: float = 3.6,
    ) -> tuple[int, int]:
        """Return (total_new_interactions, handoff_interactions).

        A handoff interaction is one where Ghost revealed the target and Spot
        subsequently interacted it — the intended coordination pattern.
        """
        new_interactions = 0
        handoff_interactions = 0
        for agent, action_type in action_types.items():
            capability = capabilities[agent]
            if action_type != ActionType.INTERACT or not capability.has_arm:
                continue
            sx, sy = states[agent].x, states[agent].y
            for target in self.targets:
                if target.interacted or not target.revealed:
                    continue
                if np.hypot(sx - target.x, sy - target.y) <= interaction_range:
                    target.interacted = True
                    new_interactions += 1
                    if target.revealed_by == "ghost":
                        handoff_interactions += 1
                    break
        return new_interactions, handoff_interactions

    def all_interacted(self) -> bool:
        return all(target.interacted for target in self.targets)

    def flattened_target_obs(self, agent_x: float = 0.0, agent_y: float = 0.0) -> np.ndarray:
        """Return per-target (dx, dy, state) relative to the observing agent."""
        flat: list[float] = []
        for target in self.targets:
            if target.interacted:
                state = 1.0
            elif target.revealed:
                state = 0.5
            else:
                state = 0.0
            flat.extend([target.x - agent_x, target.y - agent_y, state])
        return np.asarray(flat, dtype=np.float32)
