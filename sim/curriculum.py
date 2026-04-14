from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pettingzoo import ParallelEnv

from sim.env import HeteroQuadEnv


@dataclass(frozen=True)
class StageConfig:
    """Curriculum stage parameters for environment construction."""

    num_targets: int
    use_rough_zones: bool
    use_walls: bool
    battery_drain: bool
    full_dynamics: bool
    max_steps: int
    dt: float = 0.1  # physics timestep (larger = faster movement per step)


def stage_config(stage: int) -> StageConfig:
    if stage == 1:
        return StageConfig(
            num_targets=2,
            use_rough_zones=False,
            use_walls=False,
            battery_drain=False,
            full_dynamics=True,
            max_steps=200,
            dt=0.5,
        )
    if stage == 2:
        return StageConfig(
            num_targets=3,
            use_rough_zones=True,
            use_walls=True,
            battery_drain=True,
            full_dynamics=True,
            max_steps=500,
            dt=0.2,
        )
    if stage == 3:
        return StageConfig(
            num_targets=5,
            use_rough_zones=True,
            use_walls=True,
            battery_drain=True,
            full_dynamics=True,
            max_steps=1000,
            dt=0.1,
        )
    raise ValueError(f"Unsupported curriculum stage: {stage}")


def make_env(stage: int, render_mode: str | None = None) -> ParallelEnv:
    cfg = stage_config(stage)
    map_path = Path(__file__).resolve().parents[1] / "configs" / "map_default.yaml"
    return HeteroQuadEnv(
        num_targets=cfg.num_targets,
        max_steps=cfg.max_steps,
        map_path=map_path,
        use_rough_zones=cfg.use_rough_zones,
        use_walls=cfg.use_walls,
        battery_drain=cfg.battery_drain,
        full_dynamics=cfg.full_dynamics,
        dt=cfg.dt,
        render_mode=render_mode,
    )
