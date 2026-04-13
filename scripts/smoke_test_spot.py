#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from abstraction.robot_agent import Action, ActionType, SPOT_CAPABILITY, SpotAgent


def main() -> int:
    base_url = os.getenv("SPOT_BASE_URL", "http://localhost:5000/api")
    username = os.getenv("SPOT_USERNAME", "admin")
    password = os.getenv("SPOT_PASSWORD")

    if not password:
        print("Error: SPOT_PASSWORD is missing. Set it in your environment or .env file.")
        return 1

    agent = SpotAgent(
        agent_id=SPOT_CAPABILITY.agent_id,
        base_url=base_url,
        capability=SPOT_CAPABILITY,
        username=username,
        password=password,
    )

    if not agent.connect():
        print("Error: failed to connect/authenticate SpotAgent.")
        return 1

    state_before = agent.get_state()
    print(
        f"Initial state: x={state_before.x:.3f}, y={state_before.y:.3f}, "
        f"yaw={state_before.yaw:.3f}, battery={state_before.battery:.3f}, online={state_before.online}"
    )

    action_ok = agent.send_action(Action(type=ActionType.MOVE_TO, x=2.0, y=1.0, yaw=0.5))
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
