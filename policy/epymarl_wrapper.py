"""EPyMARL MultiAgentEnv adapter for HeteroQuadEnv.

Converts the PettingZoo ParallelEnv with Dict(type=Discrete(5), target=Box(2))
action space into a single Discrete action space via grid discretization,
as required by EPyMARL's learners and action selectors.

Action encoding (compact — only movement actions carry a grid target):
    SCOUT  (type 0): action_ids 0 .. grid_cells-1        (grid target)
    MOVE_TO(type 1): action_ids grid_cells .. 2*grid_cells-1 (grid target)
    INTERACT(type 2): action_id  2*grid_cells              (point action)
    RECHARGE(type 3): action_id  2*grid_cells + 1          (point action)
    HOLD    (type 4): action_id  2*grid_cells + 2          (point action)

Total actions = 2 * grid_size^2 + 3  (default grid_size=8 → 131 actions)
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

        # Compact action space: 2*grid_cells (SCOUT + MOVE_TO) + 3 (INTERACT, RECHARGE, HOLD)
        self.n_actions = 2 * self._grid_cells + 3
        self._obs: list[np.ndarray] | None = None
        self._info: dict[str, Any] = {}

    # ----- action conversion -----

    def _decode_action(self, action_id: int) -> dict[str, Any]:
        gc = self._grid_cells
        if action_id < gc:
            # SCOUT with grid target
            return {"type": 0, "target": self._grid_centres[action_id]}
        elif action_id < 2 * gc:
            # MOVE_TO with grid target
            return {"type": 1, "target": self._grid_centres[action_id - gc]}
        else:
            # Point actions: INTERACT=2, RECHARGE=3, HOLD=4
            point_idx = action_id - 2 * gc  # 0=INTERACT, 1=RECHARGE, 2=HOLD
            return {"type": 2 + point_idx, "target": np.zeros(2, dtype=np.float32)}

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

        # Force INTERACT when available: mask out everything else.
        # In Stage 1 (no battery drain) there is never a reason to delay
        # interaction, and this removes the exploration bottleneck where
        # a single INTERACT action competes with 64 MOVE_TO logits.
        if mask_5[2]:
            mask_5 = np.array([0, 0, 1, 0, 0], dtype=np.float32)

        gc = self._grid_cells
        avail: list[int] = []
        # SCOUT actions (grid_cells entries)
        avail.extend([int(mask_5[0])] * gc)
        # MOVE_TO actions (grid_cells entries)
        avail.extend([int(mask_5[1])] * gc)
        # Point actions: INTERACT, RECHARGE, HOLD (1 entry each)
        avail.append(int(mask_5[2]))  # INTERACT
        avail.append(int(mask_5[3]))  # RECHARGE
        avail.append(int(mask_5[4]))  # HOLD
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
