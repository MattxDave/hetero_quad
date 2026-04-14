from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from abstraction.robot_agent import ActionType


@dataclass(frozen=True)
class RewardContext:
    """Input frame data needed for deterministic reward computation."""

    agents: tuple[str, ...]
    action_types: dict[str, ActionType]
    arrived: dict[str, bool]
    commanded_waypoints: dict[str, tuple[float, float]]
    positions: dict[str, tuple[float, float]]
    battery: dict[str, float]
    battery_death_event: dict[str, bool]
    new_interactions: int
    new_reveals: int
    completed: bool


def compute_reward_terms(ctx: RewardContext) -> dict[str, dict[str, float]]:
    terms = {
        agent: {
            "interaction": 0.0,
            "reveal": 0.0,
            "time": -0.01,
            "coverage_gap": 0.0,
            "sync_coverage": 0.0,
            "low_battery": 0.0,
            "battery_death": 0.0,
            "completion": 0.0,
        }
        for agent in ctx.agents
    }

    team_interaction = 10.0 * float(ctx.new_interactions)
    team_reveal = 5.0 * float(ctx.new_reveals)
    for agent in ctx.agents:
        terms[agent]["interaction"] = team_interaction
        terms[agent]["reveal"] = team_reveal

    if len(ctx.agents) == 2:
        a0, a1 = ctx.agents
        p0 = np.asarray(ctx.positions[a0], dtype=np.float64)
        p1 = np.asarray(ctx.positions[a1], dtype=np.float64)
        centroid_dist = float(np.linalg.norm(p0 - p1))
        if centroid_dist < 3.0:
            for agent in ctx.agents:
                terms[agent]["coverage_gap"] = -0.05

        both_arrived = all(ctx.arrived[a] for a in ctx.agents)
        wp0 = np.asarray(ctx.commanded_waypoints[a0], dtype=np.float64)
        wp1 = np.asarray(ctx.commanded_waypoints[a1], dtype=np.float64)
        waypoint_sep = float(np.linalg.norm(wp0 - wp1))
        if both_arrived and waypoint_sep > 2.0:
            for agent in ctx.agents:
                terms[agent]["sync_coverage"] = 1.0

    for agent in ctx.agents:
        if ctx.battery[agent] < 0.2 and ctx.action_types[agent] != ActionType.RECHARGE:
            terms[agent]["low_battery"] = -1.0
        if ctx.battery_death_event[agent]:
            terms[agent]["battery_death"] = -20.0

    if ctx.completed:
        for agent in ctx.agents:
            terms[agent]["completion"] = 100.0

    return terms


def compute_rewards(ctx: RewardContext) -> tuple[dict[str, float], dict[str, float]]:
    """Return per-agent reward totals and a flat per-term breakdown for logging."""
    terms = compute_reward_terms(ctx)
    rewards = {agent: float(sum(parts.values())) for agent, parts in terms.items()}
    breakdown: dict[str, float] = {}
    for agent, parts in terms.items():
        for term, val in parts.items():
            breakdown[f"{term}/{agent}"] = val
    return rewards, breakdown
