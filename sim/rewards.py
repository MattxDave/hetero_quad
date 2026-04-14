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
    new_handoffs: int
    new_reveals_by_agent: dict[str, int]
    completed: bool
    # Approach shaping (optional — distances to nearest relevant targets)
    prev_nearest_target_dist: dict[str, float] | None = None
    curr_nearest_target_dist: dict[str, float] | None = None


def compute_reward_terms(ctx: RewardContext) -> dict[str, dict[str, float]]:
    terms = {
        agent: {
            "interaction": 0.0,
            "handoff": 0.0,
            "reveal": 0.0,
            "time": -0.05,
            "coverage_gap": 0.0,
            "sync_coverage": 0.0,
            "low_battery": 0.0,
            "battery_death": 0.0,
            "completion": 0.0,
            "approach": 0.0,
        }
        for agent in ctx.agents
    }

    # Interaction reward: shared for team completion
    team_interaction = 50.0 * float(ctx.new_interactions)
    for agent in ctx.agents:
        terms[agent]["interaction"] = team_interaction

    # Handoff bonus: Ghost reveals → Spot interacts (the intended coordination pattern)
    # Both agents rewarded to incentivize the full Ghost-scout → Spot-interact chain.
    team_handoff = 25.0 * float(ctx.new_handoffs)
    for agent in ctx.agents:
        terms[agent]["handoff"] = team_handoff

    # Role-differentiated reveal reward: only Ghost earns reveal credit.
    # With common_reward, team reward = 10.0 when Ghost reveals vs 0.0 when Spot reveals,
    # creating a gradient that pushes Ghost to be the scout.
    ghost_reveals = float(ctx.new_reveals_by_agent.get("ghost", 0))
    terms["ghost"]["reveal"] = 10.0 * ghost_reveals
    # Spot gets nothing for revealing — don't incentivize it to scout.

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
                terms[agent]["sync_coverage"] = 0.0  # disabled: per-step reward was exploited

    for agent in ctx.agents:
        if ctx.battery[agent] < 0.2 and ctx.action_types[agent] != ActionType.RECHARGE:
            terms[agent]["low_battery"] = -1.0
        if ctx.battery_death_event[agent]:
            terms[agent]["battery_death"] = -20.0

    if ctx.completed:
        for agent in ctx.agents:
            terms[agent]["completion"] = 200.0

    # Potential-based approach shaping: reward for reducing distance to nearest relevant target
    if ctx.prev_nearest_target_dist is not None and ctx.curr_nearest_target_dist is not None:
        for agent in ctx.agents:
            prev_d = ctx.prev_nearest_target_dist.get(agent, 0.0)
            curr_d = ctx.curr_nearest_target_dist.get(agent, 0.0)
            # Positive when getting closer, negative when moving away (capped)
            delta = prev_d - curr_d
            terms[agent]["approach"] = float(np.clip(delta * 0.5, -0.5, 0.5))

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
