"""Render Stage 1 rollouts from a trained checkpoint to diagnose coordination.

Prints a step-by-step log showing:
- Who reveals each target (and the positions of both agents at that moment)
- What both agents do every 5 steps
- Per-episode handoff / scout metrics for cross-referencing

Usage:
    python scripts/watch_rollout.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from sim import make_env
from sim.env import HeteroQuadEnv
from policy.epymarl_wrapper import HeteroQuadMAEnv
from policy.train_mappo import ActorNet

# ---- config ----------------------------------------------------------------
CHECKPOINT = "checkpoints/mappo_stage1_step451200.pt"
N_EPISODES = 3
FRAME_DELAY = 0.15   # seconds per step; increase to slow down
STAGE = 1
GRID_SIZE = 4
HIDDEN = 256
TYPE_NAMES = ["SCOUT", "MOVE_TO", "INTERACT", "RECHARGE", "HOLD"]
# ----------------------------------------------------------------------------


def _expand_mask(mask_5: np.ndarray, gc: int) -> list[int]:
    """Replicate the wrapper's mask expansion (forced-INTERACT included)."""
    m = mask_5.copy()
    if m[2]:
        m = np.array([0, 0, 1, 0, 0], dtype=np.float32)
    avail = [int(m[0])] * gc + [int(m[1])] * gc
    avail += [int(m[2]), int(m[3]), int(m[4])]
    return avail


def _decode(action_id: int, gc: int, centres: np.ndarray) -> dict:
    if action_id < gc:
        return {"type": 0, "target": centres[action_id]}
    if action_id < 2 * gc:
        return {"type": 1, "target": centres[action_id - gc]}
    return {"type": 2 + (action_id - 2 * gc), "target": np.zeros(2, dtype=np.float32)}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Raw env for rendering (render_mode="human" opens matplotlib window)
    env = make_env(stage=STAGE, render_mode="human")
    assert isinstance(env, HeteroQuadEnv)

    # Wrapper used only for grid metadata and obs/action shapes
    wrapper = HeteroQuadMAEnv(stage=STAGE, grid_size=GRID_SIZE, seed=0)
    info = wrapper.get_env_info()
    obs_dim   = info["obs_shape"]
    n_actions = info["n_actions"]
    gc        = wrapper._grid_cells
    centres   = wrapper._grid_centres
    agent_ids = wrapper._agent_ids   # ["spot", "ghost"]

    # Load checkpoint
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    actors = [ActorNet(obs_dim, n_actions, HIDDEN).to(device) for _ in range(2)]
    for i, sd in enumerate(ckpt["actors"]):
        actors[i].load_state_dict(sd)
        actors[i].eval()

    print(f"Loaded: {CHECKPOINT}")
    print(f"obs_dim={obs_dim}  n_actions={n_actions}  device={device}\n")

    for ep in range(N_EPISODES):
        obs_dict, info_dict = env.reset(seed=ep * 137)
        done = False
        step = 0
        ep_reveals = {"spot": 0, "ghost": 0}
        prev_revealed   = [t.revealed   for t in env.tasks.targets]
        prev_interacted = [t.interacted for t in env.tasks.targets]

        print(f"{'='*64}")
        print(f"Episode {ep}")
        for j, t in enumerate(env.tasks.targets):
            print(f"  target {j}: ({t.x:.1f}, {t.y:.1f})")
        s, g = env._states["spot"], env._states["ghost"]
        print(f"  spot  start: ({s.x:.1f}, {s.y:.1f})")
        print(f"  ghost start: ({g.x:.1f}, {g.y:.1f})")

        while not done:
            # Choose greedy actions
            actions: dict[str, dict] = {}
            for i, agent in enumerate(agent_ids):
                m5   = info_dict[agent]["action_mask"]
                avail = _expand_mask(m5, gc)
                obs_t   = torch.tensor(obs_dict[agent],  dtype=torch.float32, device=device).unsqueeze(0)
                avail_t = torch.tensor(avail,            dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    dist = actors[i](obs_t, avail_t)
                a = int(dist.probs.argmax().item())
                actions[agent] = _decode(a, gc, centres)

            obs_dict, rewards, terms, truncs, info_dict = env.step(actions)

            # Detect new reveals
            for j, t in enumerate(env.tasks.targets):
                if t.revealed and not prev_revealed[j]:
                    ep_reveals[t.revealed_by] += 1
                    sp = env._states["spot"]
                    gh = env._states["ghost"]
                    print(
                        f"  step {step:3d}  *** TARGET {j} REVEALED by {t.revealed_by.upper()}"
                        f" @ ({t.x:.1f},{t.y:.1f})"
                        f"  |  spot=({sp.x:.1f},{sp.y:.1f})"
                        f"  ghost=({gh.x:.1f},{gh.y:.1f})"
                    )

            # Detect new interactions
            for j, t in enumerate(env.tasks.targets):
                if t.interacted and not prev_interacted[j]:
                    sp = env._states["spot"]
                    print(f"  step {step:3d}  *** TARGET {j} INTERACTED"
                          f"  |  spot=({sp.x:.1f},{sp.y:.1f})")

            prev_revealed   = [t.revealed   for t in env.tasks.targets]
            prev_interacted = [t.interacted for t in env.tasks.targets]

            # Periodic position log
            if step % 5 == 0:
                for agent in agent_ids:
                    a  = actions[agent]
                    tg = a["target"]
                    st = env._states[agent]
                    print(
                        f"  step {step:3d}  {agent:6s} -> {TYPE_NAMES[a['type']]:8s}"
                        f"  tgt=({tg[0]:.1f},{tg[1]:.1f})"
                        f"  pos=({st.x:.1f},{st.y:.1f})"
                    )

            step += 1
            done = all(terms.values()) or all(truncs.values())
            time.sleep(FRAME_DELAY)

        # Episode summary
        rm = info_dict["spot"]["role_metrics"]
        success = all(terms.values()) and not all(truncs.values())
        print(f"\n  --- Episode {ep} done in {step} steps  success={success} ---")
        print(f"  Reveals  → spot: {ep_reveals['spot']}  ghost: {ep_reveals['ghost']}")
        print(f"  handoff_rate:                {rm['handoff_rate']:.3f}")
        print(f"  spot_interact_given_valid:   {rm['spot_interact_given_valid']:.3f}")
        print(f"  ghost_scout_given_unrevealed:{rm['ghost_scout_given_unrevealed']:.3f}")
        print()

    env.close()


if __name__ == "__main__":
    main()
