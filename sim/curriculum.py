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
    battery_drain: bool
    full_dynamics: bool
    max_steps: int


def stage_config(stage: int) -> StageConfig:
    if stage == 1:
        return StageConfig(
            num_targets=2,
            use_rough_zones=False,
            battery_drain=False,
            full_dynamics=True,
            max_steps=200,
        )
    if stage == 2:
        return StageConfig(
            num_targets=3,
            use_rough_zones=True,
            battery_drain=True,
            full_dynamics=True,
            max_steps=500,
        )
    if stage == 3:
        return StageConfig(
            num_targets=5,
            use_rough_zones=True,
            battery_drain=True,
            full_dynamics=True,
            max_steps=1000,
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
        battery_drain=cfg.battery_drain,
        full_dynamics=cfg.full_dynamics,
        render_mode=render_mode,
    )
