from __future__ import annotations

from typing import Iterable

from abstraction.robot_agent import RobotState
from sim.tasks import Target
from sim.world import Rect, World


class MatplotlibRenderer:
    """Simple matplotlib renderer for the multi-agent 2D world."""

    def __init__(self) -> None:
        import matplotlib.pyplot as plt

        self._plt = plt
        self._fig, self._ax = plt.subplots(figsize=(6, 6))

    def render(
        self,
        world: World,
        agent_states: dict[str, RobotState],
        targets: Iterable[Target],
        charge_point: tuple[float, float] = (1.0, 1.0),
    ) -> None:
        ax = self._ax
        ax.clear()
        ax.set_xlim(0, world.width)
        ax.set_ylim(0, world.height)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("hetero_quad sim")

        for rect in world.obstacles:
            self._draw_rect(rect, color="black", alpha=1.0)

        for rect in world.rough_zones:
            self._draw_rect(rect, color="saddlebrown", alpha=0.25)

        for target in targets:
            if target.interacted:
                color = "green"
            elif target.revealed:
                color = "yellow"
            else:
                color = "red"
            ax.scatter([target.x], [target.y], marker="*", s=140, c=color, edgecolors="k")

        for agent, state in agent_states.items():
            color = "tab:orange" if agent == "spot" else "tab:purple"
            label = "S" if agent == "spot" else "G"
            circle = self._plt.Circle((state.x, state.y), 0.3, color=color, fill=True, alpha=0.8)
            ax.add_patch(circle)
            ax.text(state.x, state.y, label, ha="center", va="center", color="white", fontsize=10, fontweight="bold")

        ax.scatter([charge_point[0]], [charge_point[1]], marker="s", s=120, c="blue")
        self._fig.canvas.draw_idle()
        self._plt.pause(0.001)

    def close(self) -> None:
        self._plt.close(self._fig)

    def _draw_rect(self, rect: Rect, color: str, alpha: float) -> None:
        patch = self._plt.Rectangle(
            (rect.x_min, rect.y_min),
            rect.x_max - rect.x_min,
            rect.y_max - rect.y_min,
            color=color,
            alpha=alpha,
        )
        self._ax.add_patch(patch)
