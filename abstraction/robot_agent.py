from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import Enum
from math import atan2
import time
from typing import Any

import requests


class ActionType(str, Enum):
    SCOUT = "SCOUT"
    MOVE_TO = "MOVE_TO"
    INTERACT = "INTERACT"
    RECHARGE = "RECHARGE"
    HOLD = "HOLD"


@dataclass
class Action:
    type: ActionType
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0


@dataclass
class RobotState:
    agent_id: str
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    battery: float = 0.0
    online: bool = False
    powered_on: bool = False
    standing: bool = False
    stamp: float = 0.0

    def as_obs(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CapabilityProfile:
    agent_id: str
    max_speed: float
    battery_hours: float
    has_arm: bool
    has_lidar: bool
    index: int


class RobotAgent(ABC):
    def __init__(self, agent_id: str, base_url: str, capability: CapabilityProfile) -> None:
        self.agent_id = agent_id
        self.base_url = base_url.rstrip("/")
        self.capability = capability
        self._session = requests.Session()
        self._timeout = 1.5

    @staticmethod
    def _quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
        numerator = 2.0 * (qw * qz + qx * qy)
        denominator = 1.0 - 2.0 * (qy * qy + qz * qz)
        return atan2(numerator, denominator)

    @abstractmethod
    def connect(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> RobotState:
        raise NotImplementedError

    @abstractmethod
    def send_action(self, action: Action) -> bool:
        raise NotImplementedError

    @abstractmethod
    def estop(self) -> bool:
        raise NotImplementedError


class GhostAgent(RobotAgent):
    def connect(self) -> bool:
        try:
            response = self._session.get(f"{self.base_url}/status", timeout=self._timeout)
            return response.status_code == 200
        except requests.RequestException:
            return False

    @staticmethod
    def _parse_battery(raw: Any) -> float:
        try:
            if isinstance(raw, (int, float)):
                value = float(raw)
            elif isinstance(raw, str):
                value = float(raw.strip().replace("%", ""))
            else:
                return 0.0
            if value > 1.0:
                value = value / 100.0
            return max(0.0, min(value, 1.0))
        except (TypeError, ValueError):
            return 0.0

    def get_state(self) -> RobotState:
        state = RobotState(agent_id=self.agent_id, online=False, stamp=time.time())
        try:
            status_resp = self._session.get(f"{self.base_url}/status", timeout=self._timeout)
            odom_resp = self._session.get(f"{self.base_url}/odom", timeout=self._timeout)
            status_resp.raise_for_status()
            odom_resp.raise_for_status()
        except requests.RequestException:
            return state

        status_data = status_resp.json() if status_resp.content else {}
        odom_data = odom_resp.json() if odom_resp.content else {}

        pose = odom_data.get("pose", {})
        position = pose.get("position", {})
        orientation = pose.get("orientation", {})
        twist = odom_data.get("twist", {})
        linear = twist.get("linear", {})
        angular = twist.get("angular", {})

        qx = float(orientation.get("x", 0.0))
        qy = float(orientation.get("y", 0.0))
        qz = float(orientation.get("z", 0.0))
        qw = float(orientation.get("w", 1.0))

        state.x = float(position.get("x", 0.0))
        state.y = float(position.get("y", 0.0))
        state.z = float(position.get("z", 0.0))
        state.yaw = self._quat_to_yaw(qx, qy, qz, qw)
        state.vx = float(linear.get("x", 0.0))
        state.vy = float(linear.get("y", 0.0))
        state.wz = float(angular.get("z", 0.0))
        state.battery = self._parse_battery(status_data.get("battery", 0.0))
        state.online = bool(status_data.get("online", True))
        state.powered_on = bool(status_data.get("powered_on", state.online))
        state.standing = bool(status_data.get("standing", state.powered_on))
        state.stamp = time.time()
        return state

    def send_action(self, action: Action) -> bool:
        try:
            if action.type in (ActionType.MOVE_TO, ActionType.SCOUT):
                payload = {"x": action.x, "y": action.y, "yaw": action.yaw, "frame_id": "map"}
                response = self._session.post(
                    f"{self.base_url}/command/send_local_goal",
                    json=payload,
                    timeout=self._timeout,
                )
                return response.ok
            if action.type == ActionType.HOLD:
                response = self._session.post(
                    f"{self.base_url}/command",
                    json={"topic": "/command/stop"},
                    timeout=self._timeout,
                )
                return response.ok
            if action.type == ActionType.RECHARGE:
                return self.send_action(Action(type=ActionType.HOLD))
            if action.type == ActionType.INTERACT:
                return False
            return False
        except requests.RequestException:
            return False

    def estop(self) -> bool:
        try:
            response = self._session.post(
                f"{self.base_url}/command",
                json={"topic": "/command/setEStop"},
                timeout=self._timeout,
            )
            return response.ok
        except requests.RequestException:
            return False


class SpotAgent(RobotAgent):
    def __init__(
        self,
        agent_id: str,
        base_url: str,
        capability: CapabilityProfile,
        username: str,
        password: str,
    ) -> None:
        super().__init__(agent_id=agent_id, base_url=base_url, capability=capability)
        self.username = username
        self.password = password
        self._token: str | None = None

    def _auth_headers(self) -> dict[str, str]:
        if not self._token:
            return {}
        return {"Authorization": f"Bearer {self._token}"}

    def connect(self) -> bool:
        try:
            response = self._session.post(
                f"{self.base_url}/auth/login",
                json={"username": self.username, "password": self.password},
                timeout=self._timeout,
            )
            response.raise_for_status()
            data = response.json() if response.content else {}
            token = data.get("token") or data.get("access_token")
            if not token:
                return False
            self._token = str(token)
            return True
        except requests.RequestException:
            return False

    def get_state(self) -> RobotState:
        state = RobotState(agent_id=self.agent_id, online=False, stamp=time.time())
        headers = self._auth_headers()
        try:
            robot_state_resp = self._session.get(
                f"{self.base_url}/robot/state",
                headers=headers,
                timeout=self._timeout,
            )
            pose_resp = self._session.get(
                f"{self.base_url}/robot/pose",
                headers=headers,
                timeout=self._timeout,
            )
            robot_state_resp.raise_for_status()
            pose_resp.raise_for_status()
        except requests.RequestException:
            return state

        robot_state_data = robot_state_resp.json() if robot_state_resp.content else {}
        pose_data = pose_resp.json() if pose_resp.content else {}

        qx = float(pose_data.get("qx", 0.0))
        qy = float(pose_data.get("qy", 0.0))
        qz = float(pose_data.get("qz", 0.0))
        qw = float(pose_data.get("qw", 1.0))

        state.x = float(pose_data.get("x", 0.0))
        state.y = float(pose_data.get("y", 0.0))
        state.z = float(pose_data.get("z", 0.0))
        state.yaw = self._quat_to_yaw(qx, qy, qz, qw)
        state.vx = float(pose_data.get("vx", 0.0))
        state.vy = float(pose_data.get("vy", 0.0))
        state.wz = float(pose_data.get("wz", 0.0))
        state.battery = GhostAgent._parse_battery(robot_state_data.get("battery", 0.0))
        state.online = True
        state.powered_on = bool(robot_state_data.get("powered_on", True))
        state.standing = bool(robot_state_data.get("standing", state.powered_on))
        state.stamp = time.time()
        return state

    def send_action(self, action: Action) -> bool:
        headers = self._auth_headers()
        try:
            if action.type in (ActionType.MOVE_TO, ActionType.SCOUT):
                payload = {"x": action.x, "y": action.y, "yaw": action.yaw, "frame": "odom"}
                response = self._session.post(
                    f"{self.base_url}/robot/goto",
                    json=payload,
                    headers=headers,
                    timeout=self._timeout,
                )
                return response.ok
            if action.type == ActionType.HOLD:
                response = self._session.post(
                    f"{self.base_url}/robot/stop",
                    headers=headers,
                    timeout=self._timeout,
                )
                return response.ok
            if action.type == ActionType.INTERACT:
                response = self._session.post(
                    f"{self.base_url}/robot/arm/unstow",
                    headers=headers,
                    timeout=self._timeout,
                )
                return response.ok
            if action.type == ActionType.RECHARGE:
                return self.send_action(Action(type=ActionType.HOLD))
            return False
        except requests.RequestException:
            return False

    def estop(self) -> bool:
        try:
            response = self._session.post(
                f"{self.base_url}/robot/stop",
                headers=self._auth_headers(),
                timeout=self._timeout,
            )
            return response.ok
        except requests.RequestException:
            return False


SPOT_CAPABILITY = CapabilityProfile(
    agent_id="spot",
    max_speed=1.6,
    battery_hours=1.5,
    has_arm=True,
    has_lidar=False,
    index=0,
)

GHOST_CAPABILITY = CapabilityProfile(
    agent_id="ghost",
    max_speed=3.0,
    battery_hours=8.0,
    has_arm=False,
    has_lidar=True,
    index=1,
)
