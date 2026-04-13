"""EPyMARL MultiAgentEnv adapter for HeteroQuadEnv.

Converts the PettingZoo ParallelEnv with Dict(type=Discrete(5), target=Box(2))
action space into a single Discrete action space via grid discretization,
as required by EPyMARL's learners and action selectors.

Action encoding:
    action_id = type_idx * grid_cells + grid_idx
    where grid_idx = row * grid_size + col
    grid_cells = grid_size^2

Total actions = 5 * grid_size^2  (default grid_size=8 → 320 actions)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Ensure project root is importable
_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from sim.curriculum import make_env
from sim.env import HeteroQuadEnv


class HeteroQuadMAEnv:
    """EPyMARL-compatible MultiAgentEnv wrapping HeteroQuadEnv."""

    def __init__(
        self,
        stage: int = 1,
        grid_size: int = 8,
        common_reward: bool = True,
        seed: int = 0,
    ) -> None:
        self._env: HeteroQuadEnv = make_env(stage)  # type: ignore[assignment]
        self._grid_size = grid_size
        self._grid_cells = grid_size * grid_size
        self._common_reward = common_reward
        self._seed = seed

        self.n_agents = len(self._env.possible_agents)
        self.episode_limit = self._env.max_steps
        self._agent_ids = list(self._env.possible_agents)

        # Pre-compute grid cell centres
        w, h = self._env.world.width, self._env.world.height
        xs = np.linspace(w / (2 * grid_size), w - w / (2 * grid_size), grid_size)
        ys = np.linspace(h / (2 * grid_size), h - h / (2 * grid_size), grid_size)
        self._grid_centres = np.array(
            [(x, y) for y in ys for x in xs], dtype=np.float32
        )  # shape (grid_cells, 2)

        self.n_actions = 5 * self._grid_cells
        self._obs: list[np.ndarray] | None = None
        self._info: dict[str, Any] = {}

    # ----- action conversion -----

    def _decode_action(self, action_id: int) -> dict[str, Any]:
        type_idx = action_id // self._grid_cells
        grid_idx = action_id % self._grid_cells
        target = self._grid_centres[grid_idx]
        return {"type": int(type_idx), "target": target}

    # ----- MultiAgentEnv interface -----

    def reset(self, seed: int | None = None, options: dict | None = None):
        s = seed if seed is not None else self._seed
        obs_dict, info_dict = self._env.reset(seed=s)
        self._obs = [obs_dict[a] for a in self._agent_ids]
        self._info = info_dict
        return self._obs, info_dict

    def step(self, actions):
        act_dict = {}
        for i, agent in enumerate(self._agent_ids):
            act_dict[agent] = self._decode_action(int(actions[i]))

        obs_dict, rew_dict, term_dict, trunc_dict, info_dict = self._env.step(act_dict)

        if obs_dict:
            self._obs = [obs_dict[a] for a in self._agent_ids]
        rewards = [rew_dict.get(a, 0.0) for a in self._agent_ids]

        if self._common_reward:
            reward = float(sum(rewards))
        else:
            reward = rewards

        terminated = all(term_dict.get(a, False) for a in self._agent_ids)
        truncated = all(trunc_dict.get(a, False) for a in self._agent_ids)
        self._info = info_dict
        return self._obs, reward, terminated, truncated, info_dict

    def get_obs(self) -> list[np.ndarray]:
        return list(self._obs) if self._obs is not None else []

    def get_obs_agent(self, agent_id: int) -> np.ndarray:
        return self._obs[agent_id] if self._obs is not None else np.array([])

    def get_obs_size(self) -> int:
        if self._obs is not None:
            return int(self._obs[0].shape[0])
        obs_dict, _ = self._env.reset(seed=0)
        return int(obs_dict[self._agent_ids[0]].shape[0])

    def get_state(self) -> np.ndarray:
        if self._obs is None:
            return np.zeros(self.get_state_size(), dtype=np.float32)
        return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_state_size(self) -> int:
        return self.n_agents * self.get_obs_size()

    def get_avail_actions(self) -> list[list[int]]:
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id: int) -> list[int]:
        agent = self._agent_ids[agent_id]
        if self._info and agent in self._info:
            mask_5 = self._info[agent].get("action_mask", np.ones(5, dtype=np.float32))
        else:
            mask_5 = np.ones(5, dtype=np.float32)
        # Expand 5-element type mask across all grid cells
        avail = []
        for t in range(5):
            avail.extend([int(mask_5[t])] * self._grid_cells)
        return avail

    def get_total_actions(self) -> int:
        return self.n_actions

    def get_env_info(self) -> dict[str, Any]:
        return {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
        }

    def render(self) -> None:
        self._env.render()

    def close(self) -> None:
        self._env.close()

    def seed(self, seed: int) -> None:
        self._seed = seed

    def save_replay(self) -> None:
        pass

    def get_stats(self) -> dict:
        return {}
