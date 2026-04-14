#!/usr/bin/env python3
"""Minimal MAPPO trainer for HeteroQuadEnv.

Uses a centralized-critic / decentralized-actor architecture with the
EPyMARL-compatible wrapper. Reads config from policy/configs/*.yaml.

Usage:
    python policy/train_mappo.py --config policy/configs/mappo_stage1.yaml
    python policy/train_mappo.py --config policy/configs/mappo_stage1.yaml --t_max 10000  # quick test
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from policy.epymarl_wrapper import HeteroQuadMAEnv


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class ActorNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor, avail: torch.Tensor) -> torch.distributions.Categorical:
        logits = self.net(obs)
        logits[avail == 0] = -1e10
        return torch.distributions.Categorical(logits=logits)


class CriticNet(nn.Module):
    def __init__(self, state_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    def __init__(self) -> None:
        self.obs: list[list[np.ndarray]] = []
        self.states: list[np.ndarray] = []
        self.actions: list[list[int]] = []
        self.avails: list[list[list[int]]] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.log_probs: list[list[float]] = []
        self.values: list[float] = []
        self.last_value: float = 0.0  # critic bootstrap for buffer boundary

    def clear(self) -> None:
        self.__init__()  # type: ignore[misc]

    def __len__(self) -> int:
        return len(self.rewards)


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lam: float,
    last_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_adv = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            # Bootstrap from critic if the buffer ended mid-episode; 0 if terminal.
            next_val = 0.0 if dones[t] else last_value
        else:
            next_val = 0.0 if dones[t] else values[t + 1]
        delta = rewards[t] + gamma * next_val - values[t]
        advantages[t] = last_adv = delta + gamma * lam * (1.0 - float(dones[t])) * last_adv
    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class MAPPOTrainer:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.device = torch.device(
            "cuda" if cfg.get("use_cuda", True) and torch.cuda.is_available() else "cpu"
        )
        self.stage = cfg.get("stage", 1)

        # Build env for shapes
        self.env = HeteroQuadMAEnv(
            stage=self.stage,
            grid_size=cfg.get("grid_size", 8),
            seed=cfg.get("seed", 0),
            common_reward=cfg.get("common_reward", True),
        )
        info = self.env.get_env_info()
        self.n_agents = info["n_agents"]
        self.obs_dim = info["obs_shape"]
        self.state_dim = info["state_shape"]
        self.n_actions = info["n_actions"]

        hidden = cfg.get("hidden_dim", 256)
        self.actors = [
            ActorNet(self.obs_dim, self.n_actions, hidden).to(self.device)
            for _ in range(self.n_agents)
        ]
        self.critic = CriticNet(self.state_dim, hidden).to(self.device)

        all_params = list(self.critic.parameters())
        for actor in self.actors:
            all_params += list(actor.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=cfg.get("lr", 3e-4))

        # LR schedule: linear decay to 10% of initial LR
        self._lr_init = cfg.get("lr", 3e-4)
        self._lr_end = self._lr_init * 0.1

        # Optional warm-start from checkpoint (only loads network weights; resets step counter)
        resume_from = cfg.get("resume_from", None)
        if resume_from:
            ckpt = torch.load(resume_from, map_location=self.device, weights_only=True)
            for i, sd in enumerate(ckpt["actors"]):
                self.actors[i].load_state_dict(sd)
            self.critic.load_state_dict(ckpt["critic"])
            print(f"Warm-start: loaded weights from {resume_from} (step {ckpt.get('step', '?')})")

        # TensorBoard
        self.writer = None
        if cfg.get("use_tensorboard", False):
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = Path(cfg.get("tb_log_dir", "tb_logs")) / f"mappo_stage{self.stage}"
            tb_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(tb_dir))

    def collect_rollout(self, batch_size: int, seed: int) -> tuple[RolloutBuffer, dict[str, float]]:
        buf = RolloutBuffer()
        obs, _ = self.env.reset(seed=seed)
        total = 0
        ep_reward = 0.0
        ep_returns: list[float] = []
        last_role_metrics: dict[str, float] = {
            "handoff_rate": 0.0, "spot_interact_given_valid": 0.0, "ghost_scout_given_unrevealed": 0.0,
        }
        all_role_metrics: list[dict[str, float]] = []
        # Action-type distribution tracking per agent (5 types)
        action_counts: list[np.ndarray] = [np.zeros(5, dtype=np.int64) for _ in range(self.n_agents)]
        while total < batch_size:
            state = self.env.get_state()
            avail = self.env.get_avail_actions()

            # Actor forward
            actions, log_probs = [], []
            for i in range(self.n_agents):
                obs_t = torch.tensor(obs[i], dtype=torch.float32, device=self.device).unsqueeze(0)
                avail_t = torch.tensor(avail[i], dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    dist = self.actors[i](obs_t, avail_t)
                a = dist.sample()
                actions.append(int(a.item()))
                log_probs.append(float(dist.log_prob(a).item()))

            # Critic forward
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                value = float(self.critic(state_t).item())

            obs_next, reward, terminated, truncated, info = self.env.step(actions)
            done = terminated or truncated
            r = float(reward) if not isinstance(reward, list) else float(sum(reward))

            # Extract role metrics from env info
            if isinstance(info, dict):
                for agent_key in ("spot", "ghost"):
                    if agent_key in info and "role_metrics" in info[agent_key]:
                        last_role_metrics = info[agent_key]["role_metrics"]
                        break

            # Track action type distributions
            grid_cells = self.env._grid_cells
            for i, a in enumerate(actions):
                if a < grid_cells:
                    type_idx = 0  # SCOUT
                elif a < 2 * grid_cells:
                    type_idx = 1  # MOVE_TO
                else:
                    type_idx = 2 + (a - 2 * grid_cells)  # INTERACT/RECHARGE/HOLD
                action_counts[i][type_idx] += 1

            buf.obs.append([o.copy() for o in obs])
            buf.states.append(state.copy())
            buf.actions.append(actions)
            buf.avails.append(avail)
            buf.rewards.append(r)
            buf.dones.append(done)
            buf.log_probs.append(log_probs)
            buf.values.append(value)

            ep_reward += r
            total += 1

            if done:
                ep_returns.append(ep_reward)
                all_role_metrics.append(last_role_metrics.copy())
                obs, _ = self.env.reset(seed=seed + total)
                ep_reward = 0.0
                last_role_metrics = {
                    "handoff_rate": 0.0, "spot_interact_given_valid": 0.0, "ghost_scout_given_unrevealed": 0.0,
                }
            else:
                obs = obs_next

        # Aggregate episode stats from this rollout
        stats: dict[str, float] = {}
        if ep_returns:
            stats["train_ep_return_mean"] = float(np.mean(ep_returns))
            stats["train_ep_return_std"] = float(np.std(ep_returns))
            stats["train_episodes"] = float(len(ep_returns))
        if all_role_metrics:
            for key in ("handoff_rate", "spot_interact_given_valid", "ghost_scout_given_unrevealed"):
                stats[key] = float(np.mean([m.get(key, 0.0) for m in all_role_metrics]))
        # Action type distributions: agent 0=spot, 1=ghost
        type_names = ["scout", "move_to", "interact", "recharge", "hold"]
        agent_names = self.env._agent_ids  # ["spot", "ghost"]
        for i, aname in enumerate(agent_names):
            total_acts = action_counts[i].sum()
            if total_acts > 0:
                for j, tname in enumerate(type_names):
                    stats[f"act_{aname}_{tname}"] = float(action_counts[i][j]) / float(total_acts)

        # Bootstrap critic value for the final state if the buffer ended mid-episode.
        if not buf.dones[-1]:
            last_state = self.env.get_state()
            last_state_t = torch.tensor(last_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                buf.last_value = float(self.critic(last_state_t).item())

        return buf, stats

    def update(self, buf: RolloutBuffer) -> dict[str, float]:
        cfg = self.cfg
        gamma = cfg.get("gamma", 0.99)
        lam = cfg.get("gae_lambda", 0.95)
        eps = cfg.get("eps_clip", 0.2)
        epochs = cfg.get("epochs_per_update", 10)
        ent_coef = cfg.get("entropy_coef", 0.01)
        vf_coef = cfg.get("value_loss_coef", 0.5)
        max_gn = cfg.get("max_grad_norm", 0.5)

        rewards = np.array(buf.rewards, dtype=np.float32)
        # Scale rewards to reduce value function loss magnitude
        reward_scale = cfg.get("reward_scale", 0.1)
        rewards = rewards * reward_scale
        last_value_scaled = buf.last_value  # critic is already converging to scaled space
        values = np.array(buf.values, dtype=np.float32)
        dones = np.array(buf.dones, dtype=np.float32)
        advantages, returns = compute_gae(rewards, values, dones, gamma, lam, last_value=last_value_scaled)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        T = len(buf.rewards)
        mini_batch_size = cfg.get("mini_batch_size", T)  # default: full batch (no splitting)
        obs_arr = [np.array([buf.obs[t][i] for t in range(T)]) for i in range(self.n_agents)]
        avail_arr = [np.array([buf.avails[t][i] for t in range(T)]) for i in range(self.n_agents)]
        act_arr = [np.array([buf.actions[t][i] for t in range(T)]) for i in range(self.n_agents)]
        old_lp_arr = [np.array([buf.log_probs[t][i] for t in range(T)]) for i in range(self.n_agents)]
        states_arr = np.array(buf.states)
        adv_arr = advantages
        ret_arr = returns

        rng = np.random.default_rng(seed=int(np.sum(rewards[:10]) * 1e6) % (2**31))
        total_pg, total_vf, total_ent, n_updates = 0.0, 0.0, 0.0, 0
        all_params = sum([list(a.parameters()) for a in self.actors], []) + list(self.critic.parameters())
        for _ in range(epochs):
            indices = rng.permutation(T)
            for start in range(0, T, mini_batch_size):
                mb = indices[start:start + mini_batch_size]
                obs_t = [torch.tensor(obs_arr[i][mb], dtype=torch.float32, device=self.device) for i in range(self.n_agents)]
                avail_t = [torch.tensor(avail_arr[i][mb], dtype=torch.float32, device=self.device) for i in range(self.n_agents)]
                act_t = [torch.tensor(act_arr[i][mb], dtype=torch.long, device=self.device) for i in range(self.n_agents)]
                old_lp_t = [torch.tensor(old_lp_arr[i][mb], dtype=torch.float32, device=self.device) for i in range(self.n_agents)]
                states_t = torch.tensor(states_arr[mb], dtype=torch.float32, device=self.device)
                adv_t = torch.tensor(adv_arr[mb], dtype=torch.float32, device=self.device)
                ret_t = torch.tensor(ret_arr[mb], dtype=torch.float32, device=self.device)

                # Critic
                v = self.critic(states_t)
                vf_loss = F.mse_loss(v, ret_t)

                # Actors
                pg_loss = torch.tensor(0.0, device=self.device)
                ent_loss = torch.tensor(0.0, device=self.device)
                for i in range(self.n_agents):
                    dist = self.actors[i](obs_t[i], avail_t[i])
                    new_lp = dist.log_prob(act_t[i])
                    ratio = torch.exp(new_lp - old_lp_t[i])
                    surr1 = ratio * adv_t
                    surr2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * adv_t
                    pg_loss = pg_loss - torch.min(surr1, surr2).mean()
                    ent_loss = ent_loss - dist.entropy().mean()

                loss = pg_loss / self.n_agents + vf_coef * vf_loss + ent_coef * ent_loss / self.n_agents
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(all_params, max_gn)
                self.optimizer.step()

                total_pg += pg_loss.item()
                total_vf += vf_loss.item()
                total_ent += ent_loss.item()
                n_updates += 1

        return {
            "pg_loss": total_pg / max(n_updates, 1),
            "vf_loss": total_vf / max(n_updates, 1),
            "entropy": -total_ent / max(n_updates, 1) / self.n_agents,
        }

    def evaluate(self, n_episodes: int, seed: int) -> dict[str, float]:
        wins, ep_returns = 0, []
        all_role_metrics: list[dict[str, float]] = []
        _zero_metrics: dict[str, float] = {
            "handoff_rate": 0.0, "spot_interact_given_valid": 0.0, "ghost_scout_given_unrevealed": 0.0,
        }
        for ep in range(n_episodes):
            obs, _ = self.env.reset(seed=seed + ep * 1000)
            done, total_r = False, 0.0
            last_role_metrics: dict[str, float] = dict(_zero_metrics)
            while not done:
                avail = self.env.get_avail_actions()
                actions = []
                for i in range(self.n_agents):
                    obs_t = torch.tensor(obs[i], dtype=torch.float32, device=self.device).unsqueeze(0)
                    avail_t = torch.tensor(avail[i], dtype=torch.float32, device=self.device).unsqueeze(0)
                    with torch.no_grad():
                        dist = self.actors[i](obs_t, avail_t)
                    actions.append(int(dist.probs.argmax().item()))
                obs, reward, terminated, truncated, info = self.env.step(actions)
                r = float(reward) if not isinstance(reward, list) else float(sum(reward))
                total_r += r
                done = terminated or truncated
                # Extract role metrics
                if isinstance(info, dict):
                    for agent_key in ("spot", "ghost"):
                        if agent_key in info and "role_metrics" in info[agent_key]:
                            last_role_metrics = info[agent_key]["role_metrics"]
                            break
            ep_returns.append(total_r)
            all_role_metrics.append(last_role_metrics)
            if terminated and not truncated:
                wins += 1
        result = {
            "eval_return_mean": float(np.mean(ep_returns)),
            "eval_return_std": float(np.std(ep_returns)),
            "eval_win_rate": wins / max(1, n_episodes),
        }
        if all_role_metrics:
            for key in ("handoff_rate", "spot_interact_given_valid", "ghost_scout_given_unrevealed"):
                result[f"eval_{key}"] = float(np.mean([m.get(key, 0.0) for m in all_role_metrics]))
        return result

    def save(self, path: str, step: int) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(
            {
                "step": step,
                "actors": [a.state_dict() for a in self.actors],
                "critic": self.critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            os.path.join(path, f"mappo_stage{self.stage}_step{step}.pt"),
        )

    def train(self) -> None:
        cfg = self.cfg
        t_max = cfg.get("t_max", 300_000)
        batch_size = cfg.get("batch_size", 3200)
        eval_interval = cfg.get("eval_interval", 10_000)
        eval_episodes = cfg.get("eval_episodes", 20)
        log_interval = cfg.get("log_interval", 2000)
        save_interval = cfg.get("save_interval", 50_000)
        ckpt_dir = cfg.get("checkpoint_dir", "checkpoints")
        seed = cfg.get("seed", 42)

        total_steps = 0
        update_count = 0
        best_win_rate = -1.0
        t0 = time.time()
        print(f"MAPPO training — stage {self.stage}, t_max={t_max}, n_actions={self.n_actions}, device={self.device}")

        while total_steps < t_max:
            buf, ep_stats = self.collect_rollout(batch_size, seed=seed + total_steps)
            total_steps += len(buf)
            update_count += 1

            losses = self.update(buf)

            # Linear LR decay
            frac = min(1.0, total_steps / t_max)
            new_lr = self._lr_init + (self._lr_end - self._lr_init) * frac
            for pg in self.optimizer.param_groups:
                pg["lr"] = new_lr

            if total_steps % log_interval < batch_size:
                elapsed = time.time() - t0
                sps = total_steps / max(elapsed, 1e-6)
                role_str = ""
                if "handoff_rate" in ep_stats:
                    role_str = (
                        f"  handoff={ep_stats['handoff_rate']:.3f}"
                        f"  sig={ep_stats['spot_interact_given_valid']:.3f}"
                        f"  gsu={ep_stats['ghost_scout_given_unrevealed']:.3f}"
                    )
                ep_ret_str = ""
                if "train_ep_return_mean" in ep_stats:
                    ep_ret_str = f"  ep_ret={ep_stats['train_ep_return_mean']:.1f}"
                print(
                    f"[step {total_steps:>7d}] pg={losses['pg_loss']:.4f}  "
                    f"vf={losses['vf_loss']:.4f}  ent={losses['entropy']:.4f}  "
                    f"sps={sps:.0f}{ep_ret_str}{role_str}"
                )
                if self.writer:
                    self.writer.add_scalar("loss/pg", losses["pg_loss"], total_steps)
                    self.writer.add_scalar("loss/vf", losses["vf_loss"], total_steps)
                    self.writer.add_scalar("loss/entropy", losses["entropy"], total_steps)
                    if "train_ep_return_mean" in ep_stats:
                        self.writer.add_scalar("train/ep_return_mean", ep_stats["train_ep_return_mean"], total_steps)
                    if "handoff_rate" in ep_stats:
                        self.writer.add_scalar("role/handoff_rate", ep_stats["handoff_rate"], total_steps)
                        self.writer.add_scalar("role/spot_interact_given_valid", ep_stats["spot_interact_given_valid"], total_steps)
                        self.writer.add_scalar("role/ghost_scout_given_unrevealed", ep_stats["ghost_scout_given_unrevealed"], total_steps)
                    # Action type distributions
                    for key in ep_stats:
                        if key.startswith("act_"):
                            self.writer.add_scalar(f"actions/{key}", ep_stats[key], total_steps)

            if total_steps % eval_interval < batch_size:
                eval_seed = seed + 100_000 + total_steps * 7  # vary eval seeds each cycle
                ev = self.evaluate(eval_episodes, seed=eval_seed)
                role_eval_str = ""
                if "eval_handoff_rate" in ev:
                    role_eval_str = (
                        f"  handoff={ev['eval_handoff_rate']:.3f}"
                        f"  sig={ev['eval_spot_interact_given_valid']:.3f}"
                        f"  gsu={ev['eval_ghost_scout_given_unrevealed']:.3f}"
                    )
                print(
                    f"  [eval] return={ev['eval_return_mean']:.1f}±{ev['eval_return_std']:.1f}  "
                    f"win_rate={ev['eval_win_rate']:.2f}{role_eval_str}"
                )
                if self.writer:
                    self.writer.add_scalar("eval/return_mean", ev["eval_return_mean"], total_steps)
                    self.writer.add_scalar("eval/win_rate", ev["eval_win_rate"], total_steps)
                    if "eval_handoff_rate" in ev:
                        self.writer.add_scalar("eval/handoff_rate", ev["eval_handoff_rate"], total_steps)
                        self.writer.add_scalar("eval/spot_interact_given_valid", ev["eval_spot_interact_given_valid"], total_steps)
                        self.writer.add_scalar("eval/ghost_scout_given_unrevealed", ev["eval_ghost_scout_given_unrevealed"], total_steps)
                # Save best checkpoint
                if cfg.get("save_model", False) and ev["eval_win_rate"] > best_win_rate:
                    best_win_rate = ev["eval_win_rate"]
                    self.save(ckpt_dir, total_steps)
                    print(f"  [best] win_rate={best_win_rate:.2f} → {ckpt_dir}/mappo_stage{self.stage}_step{total_steps}.pt")

            if cfg.get("save_model", False) and total_steps % save_interval < batch_size:
                self.save(ckpt_dir, total_steps)
                print(f"  [saved] {ckpt_dir}/mappo_stage{self.stage}_step{total_steps}.pt")

        # Final save
        if cfg.get("save_model", False):
            self.save(ckpt_dir, total_steps)
        if self.writer:
            self.writer.close()
        print(f"Training complete: {total_steps} steps in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--t_max", type=int, default=None, help="Override t_max")
    parser.add_argument("--stage", type=int, default=None, help="Override curriculum stage")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to warm-start from")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.t_max is not None:
        cfg["t_max"] = args.t_max
    if args.stage is not None:
        cfg["stage"] = args.stage
    if args.resume is not None:
        cfg["resume_from"] = args.resume

    trainer = MAPPOTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
