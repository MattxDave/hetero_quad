#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from abstraction.robot_agent import Action, ActionType, GHOST_CAPABILITY, GhostAgent


def main() -> int:
    base_url = os.getenv("GHOST_BASE_URL", "http://192.168.168.100:5002")

    agent = GhostAgent(
        agent_id=GHOST_CAPABILITY.agent_id,
        base_url=base_url,
        capability=GHOST_CAPABILITY,
    )

    if not agent.connect():
        print("Error: failed to connect to GhostAgent /status endpoint.")
        return 1

    state_before = agent.get_state()
    print(
        f"Initial state: x={state_before.x:.3f}, y={state_before.y:.3f}, "
        f"yaw={state_before.yaw:.3f}, battery={state_before.battery:.3f}, online={state_before.online}"
    )

    if state_before.battery == 0.0:
        print("Warning: battery parsed as 0.0; parser may need updating for your real API format.")

    confirm = input("Send motion command to Ghost (+0.5m x)? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Aborted by user.")
        return 1

    target_x = state_before.x + 0.5
    action_ok = agent.send_action(Action(type=ActionType.MOVE_TO, x=target_x, y=state_before.y, yaw=state_before.yaw))
    if not action_ok:
        print("Error: MOVE_TO action failed.")
        return 1

    time.sleep(0.5)
    state_after = agent.get_state()

    dx = state_after.x - state_before.x
    dy = state_after.y - state_before.y
    dyaw = state_after.yaw - state_before.yaw
    moved = abs(dx) > 1e-3 or abs(dy) > 1e-3 or abs(dyaw) > 1e-3

    print(
        f"After action: x={state_after.x:.3f}, y={state_after.y:.3f}, "
        f"yaw={state_after.yaw:.3f}, battery={state_after.battery:.3f}, online={state_after.online}"
    )
    print(f"Delta: dx={dx:.3f}, dy={dy:.3f}, dyaw={dyaw:.3f}, moved={moved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
