#!/usr/bin/env python3
"""Smoke test: verifies EPyMARL adapter wraps HeteroQuadEnv correctly.

Runs 100 random steps through the MultiAgentEnv interface, checking shapes,
action masks, and env_info consistency.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from policy.epymarl_wrapper import HeteroQuadMAEnv


def main() -> int:
    for stage in (1, 2, 3):
        env = HeteroQuadMAEnv(stage=stage, grid_size=8, seed=42)
        info = env.get_env_info()
        print(f"[stage {stage}] env_info: {info}")

        obs, _ = env.reset(seed=42)
        assert len(obs) == info["n_agents"]
        assert obs[0].shape[0] == info["obs_shape"]

        state = env.get_state()
        assert state.shape[0] == info["state_shape"]

        avail = env.get_avail_actions()
        assert len(avail) == info["n_agents"]
        assert len(avail[0]) == info["n_actions"]

        env.close()
        print(f"  [stage {stage}] build + info check OK")

    # 100 random steps on stage 2
    env = HeteroQuadMAEnv(stage=2, grid_size=8, seed=7)
    info = env.get_env_info()
    obs, _ = env.reset(seed=7)
    rng = np.random.default_rng(7)

    for step in range(100):
        avail = env.get_avail_actions()
        actions = []
        for i in range(env.n_agents):
            valid = [a for a, v in enumerate(avail[i]) if v == 1]
            actions.append(rng.choice(valid))
        obs, reward, terminated, truncated, infos = env.step(actions)

        for i in range(env.n_agents):
            assert np.all(np.isfinite(obs[i])), f"NaN in obs at step {step}"
        if isinstance(reward, list):
            assert all(np.isfinite(r) for r in reward)
        else:
            assert np.isfinite(reward)

        if terminated or truncated:
            obs, _ = env.reset(seed=7)

    env.close()
    print(f"[smoke] 100 random steps OK — no crashes, no NaN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
