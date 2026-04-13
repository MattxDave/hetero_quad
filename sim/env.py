from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from abstraction.robot_agent import ActionType, GHOST_CAPABILITY, RobotState, SPOT_CAPABILITY
from sim.metrics import MetricsTracker
from sim.rewards import RewardContext, compute_rewards
from sim.tasks import TaskManager
from sim.world import World, load_map_config

_ACTION_ORDER = [
    ActionType.SCOUT,
    ActionType.MOVE_TO,
    ActionType.INTERACT,
    ActionType.RECHARGE,
    ActionType.HOLD,
]


class HeteroQuadEnv(ParallelEnv):
    """Parallel multi-agent RL environment for Spot + Ghost coordination."""

    metadata = {"render_modes": ["human"], "name": "hetero_quad_v0"}

    def __init__(
        self,
        num_targets: int,
        max_steps: int,
        map_path: str | Path,
        use_rough_zones: bool = True,
        battery_drain: bool = True,
        full_dynamics: bool = True,
        render_mode: str | None = None,
    ) -> None:
        self.possible_agents = ["spot", "ghost"]
        self.agents = self.possible_agents[:]
        self.capabilities = {"spot": SPOT_CAPABILITY, "ghost": GHOST_CAPABILITY}
        self.num_targets = int(num_targets)
        self.max_steps = int(max_steps)
        self.use_rough_zones = bool(use_rough_zones)
        self.battery_drain = bool(battery_drain)
        self.full_dynamics = bool(full_dynamics)
        map_cfg = load_map_config(map_path)
        self.world = World.from_config(map_cfg, use_rough_zones=use_rough_zones)
        self.charge_point = np.array(self.world.charge_point, dtype=np.float32)
        self.tasks = TaskManager(self.num_targets)
        self._rng = np.random.default_rng(0)
        self._step_count = 0

        self._states: dict[str, RobotState] = {}
        self._death_latched = {agent: False for agent in self.possible_agents}
        self._metrics = MetricsTracker()

        obs_dim = 10 + 4 + 3 * self.num_targets + 100
        hi = np.full((obs_dim,), np.float32(np.finfo(np.float32).max), dtype=np.float32)
        self._observation_spaces = {
            agent: spaces.Box(low=-hi, high=hi, dtype=np.float32) for agent in self.possible_agents
        }

        target_high = np.array([self.world.width, self.world.height], dtype=np.float32)
        self._action_spaces = {
            agent: spaces.Dict(
                {
                    "type": spaces.Discrete(5),
                    "target": spaces.Box(low=np.zeros(2, dtype=np.float32), high=target_high, dtype=np.float32),
                }
            )
            for agent in self.possible_agents
        }

        self.render_mode = render_mode
        self._renderer = None

    def observation_space(self, agent: str):
        return self._observation_spaces[agent]

    def action_space(self, agent: str):
        return self._action_spaces[agent]

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        del options
        self._rng = np.random.default_rng(seed)
        self.agents = self.possible_agents[:]
        self._step_count = 0
        self._death_latched = {agent: False for agent in self.possible_agents}

        self._states = {
            "spot": RobotState(agent_id="spot", x=1.0, y=1.0, battery=1.0, online=True),
            "ghost": RobotState(agent_id="ghost", x=2.0, y=1.0, battery=1.0, online=True),
        }
        self.tasks.reset(self.world, self._rng)
        self.tasks.update_reveals({a: (s.x, s.y) for a, s in self._states.items()})
        self._metrics.reset()

        obs = {agent: self._build_obs(agent) for agent in self.possible_agents}
        infos = {
            agent: {
                "reward_terms": {},
                "action_mask": self._compute_action_mask(agent),
                "role_metrics": {"scout_rate_ghost": 0.0, "interact_rate_spot": 0.0},
            }
            for agent in self.possible_agents
        }
        return obs, infos

    def step(self, actions: dict[str, dict[str, Any]]):
        if not self.agents:
            return {}, {}, {}, {}, {}

        commanded = {}
        action_types = {}
        arrived = {}

        for agent in self.possible_agents:
            action = actions.get(agent, {"type": 4, "target": np.array([self._states[agent].x, self._states[agent].y])})
            type_idx = int(action.get("type", 4))
            type_idx = int(np.clip(type_idx, 0, 4))
            action_type = _ACTION_ORDER[type_idx]
            target = np.asarray(action.get("target", [self._states[agent].x, self._states[agent].y]), dtype=np.float32)
            commanded[agent] = tuple(self.world.clip_target(target).tolist())
            action_types[agent] = action_type

        for agent in self.possible_agents:
            state = self._states[agent]
            capability = self.capabilities[agent]
            action_type = action_types[agent]

            if action_type in (ActionType.SCOUT, ActionType.MOVE_TO):
                result = self.world.advance_towards(
                    np.array([state.x, state.y], dtype=np.float32),
                    np.array(commanded[agent], dtype=np.float32),
                    capability,
                    enabled=self.full_dynamics,
                )
            else:
                result = self.world.advance_towards(
                    np.array([state.x, state.y], dtype=np.float32),
                    np.array([state.x, state.y], dtype=np.float32),
                    capability,
                    enabled=False,
                )

            state.x = result.x
            state.y = result.y
            state.vx = result.vx
            state.vy = result.vy
            state.yaw = float(np.arctan2(result.vy, result.vx)) if result.moved else state.yaw
            arrived[agent] = result.arrived

            moving = action_type in (ActionType.SCOUT, ActionType.MOVE_TO) and result.moved
            if self.battery_drain:
                if agent == "ghost":
                    state.battery -= 1e-4 if moving else 2e-5
                else:
                    state.battery -= 3e-4 if moving else 6e-5

            if action_type == ActionType.RECHARGE:
                dist_charge = float(np.hypot(state.x - self.charge_point[0], state.y - self.charge_point[1]))
                if dist_charge <= 1.0:
                    state.battery += 0.05

            state.battery = float(np.clip(state.battery, 0.0, 1.0))

        new_reveals = self.tasks.update_reveals({a: (s.x, s.y) for a, s in self._states.items()})
        new_interactions = self.tasks.try_interactions(action_types, self._states, self.capabilities)

        completion = self.tasks.all_interacted()
        battery_failure = any(self._states[a].battery <= 0.0 for a in self.possible_agents)

        death_event = {}
        for agent in self.possible_agents:
            dead_now = self._states[agent].battery <= 0.0 and not self._death_latched[agent]
            death_event[agent] = dead_now
            if dead_now:
                self._death_latched[agent] = True

        ctx = RewardContext(
            agents=tuple(self.possible_agents),
            action_types=action_types,
            arrived=arrived,
            commanded_waypoints=commanded,
            positions={a: (self._states[a].x, self._states[a].y) for a in self.possible_agents},
            battery={a: self._states[a].battery for a in self.possible_agents},
            battery_death_event=death_event,
            new_interactions=new_interactions,
            new_reveals=new_reveals,
            completed=completion,
        )
        rewards, breakdown = compute_rewards(ctx)
        role_metrics = self._metrics.update(action_types, commanded, self.tasks.targets, self.capabilities)

        self._step_count += 1
        terminated = completion or battery_failure
        truncated = self._step_count >= self.max_steps and not terminated

        terminations = {agent: terminated for agent in self.possible_agents}
        truncations = {agent: truncated for agent in self.possible_agents}
        infos = {
            agent: {
                "reward_terms": breakdown,
                "action_mask": self._compute_action_mask(agent),
                "role_metrics": {
                    "scout_rate_ghost": role_metrics.scout_rate_ghost,
                    "interact_rate_spot": role_metrics.interact_rate_spot,
                },
            }
            for agent in self.possible_agents
        }

        obs = {agent: self._build_obs(agent) for agent in self.possible_agents}
        if terminated or truncated:
            self.agents = []

        if self.render_mode == "human":
            self.render()

        return obs, rewards, terminations, truncations, infos

    def _build_obs(self, agent: str) -> np.ndarray:
        own = self._states[agent]
        teammate_id = "ghost" if agent == "spot" else "spot"
        mate = self._states[teammate_id]
        cap = self.capabilities[agent]

        own_vec = np.array(
            [
                own.x,
                own.y,
                own.yaw,
                own.vx,
                own.vy,
                own.battery,
                cap.max_speed,
                cap.battery_hours,
                float(cap.has_arm),
                float(cap.has_lidar),
            ],
            dtype=np.float32,
        )
        teammate_vec = np.array([mate.x, mate.y, mate.yaw, mate.battery], dtype=np.float32)
        target_vec = self.tasks.flattened_target_obs()
        occ = self.world.occupancy_grid(10)
        return np.concatenate([own_vec, teammate_vec, target_vec, occ], dtype=np.float32)

    def _compute_action_mask(self, agent: str) -> np.ndarray:
        """Return length-5 mask: 1.0=valid, 0.0=invalid per action type."""
        mask = np.ones(5, dtype=np.float32)
        cap = self.capabilities[agent]
        state = self._states[agent]
        if not cap.has_arm:
            mask[2] = 0.0
        else:
            can_interact = any(
                t.revealed and not t.interacted and np.hypot(state.x - t.x, state.y - t.y) <= 0.5
                for t in self.tasks.targets
            )
            if not can_interact:
                mask[2] = 0.0
        dist_cp = float(np.hypot(state.x - self.charge_point[0], state.y - self.charge_point[1]))
        if dist_cp > 1.0:
            mask[3] = 0.0
        return mask

    def render(self):
        if self.render_mode != "human":
            return None
        if self._renderer is None:
            from sim.render import MatplotlibRenderer

            self._renderer = MatplotlibRenderer()
        self._renderer.render(self.world, self._states, self.tasks.targets, tuple(self.charge_point.tolist()))
        return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
