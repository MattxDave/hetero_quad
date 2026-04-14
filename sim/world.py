from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

from abstraction.robot_agent import CapabilityProfile


@dataclass(frozen=True)
class Rect:
    """Axis-aligned rectangle: [x_min, y_min, x_max, y_max]."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def contains(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max


@dataclass
class MotionResult:
    """Output of a single world kinematic advance."""

    x: float
    y: float
    vx: float
    vy: float
    arrived: bool
    moved: bool


def load_map_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


class World:
    """2D world with obstacles, rough zones, and movement/collision logic."""

    def __init__(
        self,
        width: float = 20.0,
        height: float = 20.0,
        obstacles: Iterable[Rect] | None = None,
        rough_zones: Iterable[Rect] | None = None,
        dt: float = 0.1,
        agent_radius: float = 0.3,
        charge_point: tuple[float, float] = (1.0, 1.0),
    ) -> None:
        self.width = float(width)
        self.height = float(height)
        self.dt = float(dt)
        self.agent_radius = float(agent_radius)
        self.obstacles = list(obstacles or [])
        self.rough_zones = list(rough_zones or [])
        self.charge_point = charge_point

    @classmethod
    def from_config(cls, cfg: dict, use_rough_zones: bool = True, use_walls: bool = True, dt: float = 0.1) -> "World":
        bounds = cfg.get("bounds", [20.0, 20.0])
        width, height = float(bounds[0]), float(bounds[1])
        walls = cfg.get("walls", []) if use_walls else []
        obstacles = [Rect(*map(float, rect)) for rect in walls]
        rough = cfg.get("rough_zones", []) if use_rough_zones else []
        rough_zones = [Rect(*map(float, rect)) for rect in rough]
        cp = cfg.get("charge_point", [1.0, 1.0])
        return cls(width=width, height=height, obstacles=obstacles, rough_zones=rough_zones, dt=dt, charge_point=(float(cp[0]), float(cp[1])))

    def point_in_rough(self, x: float, y: float) -> bool:
        return any(zone.contains(x, y) for zone in self.rough_zones)

    def _outside_bounds(self, x: float, y: float, clearance: float) -> bool:
        return (
            x < clearance
            or y < clearance
            or x > (self.width - clearance)
            or y > (self.height - clearance)
        )

    def _rect_distance(self, x: float, y: float, rect: Rect) -> float:
        dx = max(rect.x_min - x, 0.0, x - rect.x_max)
        dy = max(rect.y_min - y, 0.0, y - rect.y_max)
        return float(np.hypot(dx, dy))

    def is_free(self, x: float, y: float, clearance: float | None = None) -> bool:
        c = self.agent_radius if clearance is None else float(clearance)
        if self._outside_bounds(x, y, c):
            return False
        for rect in self.obstacles:
            if self._rect_distance(x, y, rect) < c:
                return False
        return True

    def effective_speed(self, capability: CapabilityProfile, x: float, y: float) -> float:
        speed = float(capability.max_speed)
        if capability.agent_id == "spot" and self.point_in_rough(x, y):
            return 0.3 * speed
        return speed

    def clip_target(self, target_xy: np.ndarray) -> np.ndarray:
        x = float(np.clip(target_xy[0], 0.0, self.width))
        y = float(np.clip(target_xy[1], 0.0, self.height))
        return np.array([x, y], dtype=np.float32)

    def advance_towards(
        self,
        position_xy: np.ndarray,
        target_xy: np.ndarray,
        capability: CapabilityProfile,
        enabled: bool = True,
    ) -> MotionResult:
        if not enabled:
            return MotionResult(
                x=float(position_xy[0]),
                y=float(position_xy[1]),
                vx=0.0,
                vy=0.0,
                arrived=True,
                moved=False,
            )

        start = np.asarray(position_xy, dtype=np.float64)
        target = self.clip_target(np.asarray(target_xy, dtype=np.float64))
        delta = target - start
        dist = float(np.linalg.norm(delta))
        if dist <= 1e-9:
            return MotionResult(x=float(start[0]), y=float(start[1]), vx=0.0, vy=0.0, arrived=True, moved=False)

        speed = self.effective_speed(capability, float(start[0]), float(start[1]))
        max_step = speed * self.dt
        if dist > max_step:
            desired = start + (delta / dist) * max_step
        else:
            desired = target

        if self.is_free(float(desired[0]), float(desired[1])):
            end = desired
        else:
            lo, hi = 0.0, 1.0
            for _ in range(28):
                mid = 0.5 * (lo + hi)
                probe = start + (desired - start) * mid
                if self.is_free(float(probe[0]), float(probe[1])):
                    lo = mid
                else:
                    hi = mid
            end = start + (desired - start) * lo

        move = end - start
        moved = float(np.linalg.norm(move)) > 1e-7
        vx, vy = move / self.dt
        arrived = float(np.linalg.norm(target - end)) < 0.2
        return MotionResult(
            x=float(end[0]),
            y=float(end[1]),
            vx=float(vx),
            vy=float(vy),
            arrived=arrived,
            moved=moved,
        )

    def occupancy_grid(self, grid_size: int = 10) -> np.ndarray:
        xs = np.linspace(self.width / (2 * grid_size), self.width - self.width / (2 * grid_size), grid_size)
        ys = np.linspace(self.height / (2 * grid_size), self.height - self.height / (2 * grid_size), grid_size)
        grid = np.zeros((grid_size, grid_size), dtype=np.float32)
        for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                grid[i, j] = 0.0 if self.is_free(float(x), float(y), clearance=self.agent_radius) else 1.0
        return grid.reshape(-1)
