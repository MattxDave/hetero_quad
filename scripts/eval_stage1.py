#!/usr/bin/env python3
"""Detailed evaluation of Stage 1 checkpoint.

Runs N episodes with the trained policy and reports:
- episode return, length, completion
- per-agent action-type distributions
- role metrics (scout_rate_ghost, interact_rate_spot)
- actual reveals/interactions per episode
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from policy.epymarl_wrapper import HeteroQuadMAEnv
from policy.train_mappo import ActorNet, CriticNet

CKPT = "checkpoints/mappo_stage1_step451200.pt"
N_EPISODES = 20
SEED = 99999
STAGE = 1
GRID_SIZE = 4
HIDDEN = 256
TYPE_NAMES = ["SCOUT", "MOVE_TO", "INTERACT", "RECHARGE", "HOLD"]


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = HeteroQuadMAEnv(stage=STAGE, grid_size=GRID_SIZE, seed=0, common_reward=True)
    info = env.get_env_info()
    n_agents = info["n_agents"]
    obs_dim = info["obs_shape"]
    state_dim = info["state_shape"]
    n_actions = info["n_actions"]
    grid_cells = GRID_SIZE * GRID_SIZE
    agent_ids = env._agent_ids

    # Load checkpoint
    ckpt = torch.load(CKPT, map_location=device, weights_only=True)
    actors = [ActorNet(obs_dim, n_actions, HIDDEN).to(device) for _ in range(n_agents)]
    for i, sd in enumerate(ckpt["actors"]):
        actors[i].load_state_dict(sd)
        actors[i].eval()

    print(f"Loaded checkpoint: {CKPT} (step {ckpt['step']})")
    print(f"Env: {n_agents} agents, obs={obs_dim}, actions={n_actions}")
    print(f"Running {N_EPISODES} eval episodes (greedy)...\n")

    all_returns = []
    all_lengths = []
    all_completions = []
    all_action_dists = {name: np.zeros(5, dtype=np.int64) for name in agent_ids}
    all_role_metrics = []

    for ep in range(N_EPISODES):
        obs, _ = env.reset(seed=SEED + ep * 137)
        done = False
        total_r = 0.0
        ep_len = 0
        ep_action_counts = {name: np.zeros(5, dtype=np.int64) for name in agent_ids}
        last_role = {"scout_rate_ghost": 0.0, "interact_rate_spot": 0.0}
        completed = False

        while not done:
            avail = env.get_avail_actions()
            actions = []
            for i in range(n_agents):
                obs_t = torch.tensor(obs[i], dtype=torch.float32, device=device).unsqueeze(0)
                avail_t = torch.tensor(avail[i], dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    dist = actors[i](obs_t, avail_t)
                a = int(dist.probs.argmax().item())
                actions.append(a)
                if a < grid_cells:
                    type_idx = 0  # SCOUT
                elif a < 2 * grid_cells:
                    type_idx = 1  # MOVE_TO
                else:
                    type_idx = 2 + (a - 2 * grid_cells)  # INTERACT/RECHARGE/HOLD
                ep_action_counts[agent_ids[i]][type_idx] += 1

            obs, reward, terminated, truncated, info_dict = env.step(actions)
            r = float(reward) if not isinstance(reward, list) else float(sum(reward))
            total_r += r
            ep_len += 1
            done = terminated or truncated

            if isinstance(info_dict, dict):
                for ak in ("spot", "ghost"):
                    if ak in info_dict and "role_metrics" in info_dict[ak]:
                        last_role = info_dict[ak]["role_metrics"]
                        break

            # Check actual completion via env internals
            if terminated and not truncated:
                completed = env._env.tasks.all_interacted()

        all_returns.append(total_r)
        all_lengths.append(ep_len)
        all_completions.append(completed)
        all_role_metrics.append(last_role.copy())
        for name in agent_ids:
            all_action_dists[name] += ep_action_counts[name]

    # ---- Report ----
    print("=" * 70)
    print("STAGE 1 EVALUATION RESULTS")
    print("=" * 70)
    print(f"Episodes:      {N_EPISODES}")
    print(f"Success rate:  {sum(all_completions)}/{N_EPISODES} = {sum(all_completions)/N_EPISODES:.1%}")
    print(f"Return:        {np.mean(all_returns):.1f} ± {np.std(all_returns):.1f}")
    print(f"Ep length:     {np.mean(all_lengths):.1f} ± {np.std(all_lengths):.1f}")
    print()

    print("--- Coordination-quality metrics (end-of-episode) ---")
    hr   = [m["handoff_rate"]               for m in all_role_metrics]
    sig  = [m["spot_interact_given_valid"]   for m in all_role_metrics]
    gsu  = [m["ghost_scout_given_unrevealed"] for m in all_role_metrics]
    print(f"handoff_rate:                {np.mean(hr):.3f} ± {np.std(hr):.3f}")
    print(f"spot_interact_given_valid:   {np.mean(sig):.3f} ± {np.std(sig):.3f}")
    print(f"ghost_scout_given_unrevealed:{np.mean(gsu):.3f} ± {np.std(gsu):.3f}")
    print()

    print("--- Per-agent action-type distribution ---")
    for name in agent_ids:
        total = all_action_dists[name].sum()
        pcts = all_action_dists[name] / max(1, total) * 100
        parts = "  ".join(f"{TYPE_NAMES[j]}={pcts[j]:.1f}%" for j in range(5))
        print(f"  {name:>5s}: {parts}")
    print()

    # Per-episode table
    print("--- Per-episode detail ---")
    print(f"{'Ep':>3s}  {'Ret':>7s}  {'Len':>4s}  {'Done':>4s}  {'handoff':>8s}  {'sig':>6s}  {'gsu':>6s}")
    for ep in range(N_EPISODES):
        print(
            f"{ep:3d}  {all_returns[ep]:7.1f}  {all_lengths[ep]:4d}  "
            f"{'✓' if all_completions[ep] else '✗':>4s}  "
            f"{all_role_metrics[ep]['handoff_rate']:8.3f}  "
            f"{all_role_metrics[ep]['spot_interact_given_valid']:6.3f}  "
            f"{all_role_metrics[ep]['ghost_scout_given_unrevealed']:6.3f}"
        )


if __name__ == "__main__":
    main()
