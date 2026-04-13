#!/usr/bin/env python3
"""Render a 100-step random rollout to PNG frames (Agg backend, no display needed).

Saves:
  renders/frame_000.png … frame_099.png  — every step
  renders/rollout_summary.png            — first, middle, last side-by-side
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import numpy as np

from sim import make_env
from sim.world import Rect


OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "renders")
SEED = 42
NUM_STEPS = 100
STAGE = 2  # rough zones + battery = most visual elements


def draw_frame(ax, env, step: int) -> None:
    ax.clear()
    world = env.world
    ax.set_xlim(0, world.width)
    ax.set_ylim(0, world.height)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"hetero_quad sim — step {step}")

    # Walls
    for rect in world.obstacles:
        ax.add_patch(Rectangle(
            (rect.x_min, rect.y_min),
            rect.x_max - rect.x_min,
            rect.y_max - rect.y_min,
            color="black", alpha=1.0,
        ))

    # Rough zones
    for rect in world.rough_zones:
        ax.add_patch(Rectangle(
            (rect.x_min, rect.y_min),
            rect.x_max - rect.x_min,
            rect.y_max - rect.y_min,
            color="saddlebrown", alpha=0.25,
        ))

    # Charge point
    cp = env.charge_point
    ax.scatter([cp[0]], [cp[1]], marker="s", s=120, c="blue", zorder=5, label="charge")

    # Targets
    for t in env.tasks.targets:
        if t.interacted:
            c = "green"
        elif t.revealed:
            c = "yellow"
        else:
            c = "red"
        ax.scatter([t.x], [t.y], marker="*", s=180, c=c, edgecolors="k", zorder=4)

    # Agents
    for agent, state in env._states.items():
        color = "tab:orange" if agent == "spot" else "tab:purple"
        label = "S" if agent == "spot" else "G"
        ax.add_patch(Circle((state.x, state.y), 0.3, color=color, alpha=0.8, zorder=6))
        ax.text(state.x, state.y, label, ha="center", va="center",
                color="white", fontsize=10, fontweight="bold", zorder=7)

    ax.legend(loc="upper right", fontsize=7)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    env = make_env(STAGE)
    obs, _ = env.reset(seed=SEED)
    rng = np.random.default_rng(SEED)

    fig, ax = plt.subplots(figsize=(7, 7))
    key_frames = {}  # step -> filepath

    for step in range(NUM_STEPS):
        draw_frame(ax, env, step)
        path = os.path.join(OUT_DIR, f"frame_{step:03d}.png")
        fig.savefig(path, dpi=100, bbox_inches="tight")
        if step in (0, NUM_STEPS // 2, NUM_STEPS - 1):
            key_frames[step] = path

        actions = {}
        for agent in ["spot", "ghost"]:
            actions[agent] = {
                "type": int(rng.integers(0, 5)),
                "target": np.array(
                    [rng.uniform(0.0, env.world.width), rng.uniform(0.0, env.world.height)],
                    dtype=np.float32,
                ),
            }
        obs, rewards, terms, truncs, infos = env.step(actions)
        if all(terms.values()) or all(truncs.values()):
            obs, _ = env.reset(seed=SEED)

    plt.close(fig)

    # Summary: first / middle / last side-by-side
    fig_s, axes = plt.subplots(1, 3, figsize=(21, 7))
    import matplotlib.image as mpimg
    for ax_s, (s, p) in zip(axes, sorted(key_frames.items())):
        img = mpimg.imread(p)
        ax_s.imshow(img)
        ax_s.set_title(f"Step {s}")
        ax_s.axis("off")
    summary_path = os.path.join(OUT_DIR, "rollout_summary.png")
    fig_s.savefig(summary_path, dpi=100, bbox_inches="tight")
    plt.close(fig_s)

    env.close()
    print(f"Saved {NUM_STEPS} frames to {OUT_DIR}/")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
