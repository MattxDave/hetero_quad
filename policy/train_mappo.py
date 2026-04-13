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
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
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
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
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
) -> tuple[np.ndarray, np.ndarray]:
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_adv = 0.0
    for t in reversed(range(T)):
        next_val = 0.0 if t == T - 1 else values[t + 1]
        non_terminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_val * non_terminal - values[t]
        advantages[t] = last_adv = delta + gamma * lam * non_terminal * last_adv
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

        # TensorBoard
        self.writer = None
        if cfg.get("use_tensorboard", False):
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = Path(cfg.get("tb_log_dir", "tb_logs")) / f"mappo_stage{self.stage}"
            tb_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(tb_dir))

    def collect_rollout(self, batch_size: int, seed: int) -> RolloutBuffer:
        buf = RolloutBuffer()
        obs, _ = self.env.reset(seed=seed)
        total = 0
        ep_reward = 0.0
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
                obs, _ = self.env.reset(seed=seed + total)
                ep_reward = 0.0
            else:
                obs = obs_next

        return buf

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
        values = np.array(buf.values, dtype=np.float32)
        dones = np.array(buf.dones, dtype=np.float32)
        advantages, returns = compute_gae(rewards, values, dones, gamma, lam)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        T = len(buf.rewards)
        obs_t = [
            torch.tensor(np.array([buf.obs[t][i] for t in range(T)]), dtype=torch.float32, device=self.device)
            for i in range(self.n_agents)
        ]
        avail_t = [
            torch.tensor(np.array([buf.avails[t][i] for t in range(T)]), dtype=torch.float32, device=self.device)
            for i in range(self.n_agents)
        ]
        act_t = [
            torch.tensor([buf.actions[t][i] for t in range(T)], dtype=torch.long, device=self.device)
            for i in range(self.n_agents)
        ]
        old_lp_t = [
            torch.tensor([buf.log_probs[t][i] for t in range(T)], dtype=torch.float32, device=self.device)
            for i in range(self.n_agents)
        ]
        states_t = torch.tensor(np.array(buf.states), dtype=torch.float32, device=self.device)
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

        total_pg, total_vf, total_ent = 0.0, 0.0, 0.0
        for _ in range(epochs):
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
            nn.utils.clip_grad_norm_(
                sum([list(a.parameters()) for a in self.actors], []) + list(self.critic.parameters()),
                max_gn,
            )
            self.optimizer.step()

            total_pg += pg_loss.item()
            total_vf += vf_loss.item()
            total_ent += ent_loss.item()

        return {
            "pg_loss": total_pg / epochs,
            "vf_loss": total_vf / epochs,
            "entropy": -total_ent / epochs / self.n_agents,
        }

    def evaluate(self, n_episodes: int, seed: int) -> dict[str, float]:
        wins, ep_returns = 0, []
        for ep in range(n_episodes):
            obs, _ = self.env.reset(seed=seed + ep * 1000)
            done, total_r = False, 0.0
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
            ep_returns.append(total_r)
            # Check if completion reward was hit
            if total_r > 50:
                wins += 1
        return {
            "eval_return_mean": float(np.mean(ep_returns)),
            "eval_return_std": float(np.std(ep_returns)),
            "eval_win_rate": wins / max(1, n_episodes),
        }

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
        t0 = time.time()
        print(f"MAPPO training — stage {self.stage}, t_max={t_max}, device={self.device}")

        while total_steps < t_max:
            buf = self.collect_rollout(batch_size, seed=seed + total_steps)
            total_steps += len(buf)
            update_count += 1

            losses = self.update(buf)

            if total_steps % log_interval < batch_size:
                elapsed = time.time() - t0
                sps = total_steps / max(elapsed, 1e-6)
                print(
                    f"[step {total_steps:>7d}] pg={losses['pg_loss']:.4f}  "
                    f"vf={losses['vf_loss']:.4f}  ent={losses['entropy']:.4f}  "
                    f"sps={sps:.0f}"
                )
                if self.writer:
                    self.writer.add_scalar("loss/pg", losses["pg_loss"], total_steps)
                    self.writer.add_scalar("loss/vf", losses["vf_loss"], total_steps)
                    self.writer.add_scalar("loss/entropy", losses["entropy"], total_steps)

            if total_steps % eval_interval < batch_size:
                ev = self.evaluate(eval_episodes, seed=seed + 100_000)
                print(
                    f"  [eval] return={ev['eval_return_mean']:.1f}±{ev['eval_return_std']:.1f}  "
                    f"win_rate={ev['eval_win_rate']:.2f}"
                )
                if self.writer:
                    self.writer.add_scalar("eval/return_mean", ev["eval_return_mean"], total_steps)
                    self.writer.add_scalar("eval/win_rate", ev["eval_win_rate"], total_steps)

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
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.t_max is not None:
        cfg["t_max"] = args.t_max
    if args.stage is not None:
        cfg["stage"] = args.stage

    trainer = MAPPOTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
