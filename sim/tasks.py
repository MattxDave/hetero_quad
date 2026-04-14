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
    revealed_by: str | None = None  # agent id that first revealed this target (measurement only)


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

    def update_reveals(self, positions: dict[str, tuple[float, float]]) -> int:
        new_reveals = 0
        for target in self.targets:
            if target.revealed:
                continue
            for agent, (x, y) in positions.items():
                if np.hypot(x - target.x, y - target.y) <= 2.0:
                    target.revealed = True
                    target.revealed_by = agent
                    new_reveals += 1
                    break
        return new_reveals

    def try_interactions(
        self,
        action_types: dict[str, ActionType],
        states: dict[str, RobotState],
        capabilities: dict[str, CapabilityProfile],
    ) -> int:
        new_interactions = 0
        for agent, action_type in action_types.items():
            capability = capabilities[agent]
            if action_type != ActionType.INTERACT or not capability.has_arm:
                continue
            sx, sy = states[agent].x, states[agent].y
            for target in self.targets:
                if target.interacted or not target.revealed:
                    continue
                if np.hypot(sx - target.x, sy - target.y) <= 0.5:
                    target.interacted = True
                    new_interactions += 1
                    break
        return new_interactions

    def all_interacted(self) -> bool:
        return all(target.interacted for target in self.targets)

    def flattened_target_obs(self) -> np.ndarray:
        flat: list[float] = []
        for target in self.targets:
            if target.revealed:
                flat.extend([target.x, target.y, 1.0 if target.interacted else 0.0])
            else:
                flat.extend([0.0, 0.0, 0.0])
        return np.asarray(flat, dtype=np.float32)
