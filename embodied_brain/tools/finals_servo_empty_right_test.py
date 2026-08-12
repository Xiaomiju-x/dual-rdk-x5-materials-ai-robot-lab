#!/usr/bin/env python3
"""Isolated PB8/TIM4 servo-right test with a firmware-estop boundary."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import rclpy
from my_robot_msgs.msg import LiftStatus
from my_robot_msgs.srv import SetLiftHeight
from rclpy.node import Node
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger


SERVO_RIGHT_COMMAND = -3.0
TEST_WINDOW_S = 5.0


class ServoTest(Node):
    def __init__(self) -> None:
        super().__init__("finals_servo_empty_right_test")
        self.clear_estop = self.create_client(Trigger, "/clear_estop")
        self.estop = self.create_client(Trigger, "/estop")
        self.set_lift = self.create_client(SetLiftHeight, "/set_lift_height")
        self.identity_valid: bool | None = None
        self.estop_latched: bool | None = None
        self.firmware_info: str | None = None
        self.latest_lift: LiftStatus | None = None
        self.create_subscription(
            Bool, "/f407/firmware_identity_valid", self._on_identity, 10
        )
        self.create_subscription(Bool, "/f407/estop_latched", self._on_estop, 10)
        self.create_subscription(String, "/f407/firmware_info", self._on_firmware, 10)
        self.create_subscription(LiftStatus, "/lift_status", self._on_lift, 10)

    def _on_identity(self, message: Bool) -> None:
        self.identity_valid = bool(message.data)

    def _on_estop(self, message: Bool) -> None:
        self.estop_latched = bool(message.data)

    def _on_firmware(self, message: String) -> None:
        self.firmware_info = str(message.data)

    def _on_lift(self, message: LiftStatus) -> None:
        self.latest_lift = message

    def spin_until(self, predicate, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if predicate():
                return True
            rclpy.spin_once(self, timeout_sec=0.05)
        return bool(predicate())

    def wait_future(self, future, timeout_s: float) -> Any:
        if not self.spin_until(future.done, timeout_s):
            raise RuntimeError("ROS service response timeout")
        response = future.result()
        if response is None:
            raise RuntimeError("ROS service returned no response")
        return response

    def call_trigger(self, client, name: str) -> tuple[bool, str]:
        if not client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(f"{name} service unavailable")
        response = self.wait_future(client.call_async(Trigger.Request()), 5.0)
        return bool(response.success), str(response.message)

    def send_servo_right_command(self) -> tuple[bool, str]:
        if not self.set_lift.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("set_lift_height service unavailable")
        request = SetLiftHeight.Request()
        request.target_height_m = SERVO_RIGHT_COMMAND
        request.timeout_s = 0.0
        request.wait_for_arrival = False
        response = self.wait_future(self.set_lift.call_async(request), 5.0)
        return bool(response.success), str(response.message)


def lift_snapshot(message: LiftStatus | None) -> dict[str, Any] | None:
    if message is None:
        return None
    return {
        "height_m": float(message.height_m),
        "target_height_m": float(message.target_height_m),
        "velocity_mps": float(message.velocity_mps),
        "moving": bool(message.moving),
        "electromagnet_on": bool(message.electromagnet_on),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument(
        "--report",
        default=str(Path.home() / "finals_servo_test" / "latest.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.confirm:
        raise SystemExit("Refusing hardware command without --confirm")

    rclpy.init()
    node = ServoTest()
    report: dict[str, Any] = {
        "test": "empty_fixture_direct_right_servo",
        "test_window_s": TEST_WINDOW_S,
        "servo_right_command": SERVO_RIGHT_COMMAND,
        "started_unix_s": time.time(),
        "steps": [],
        "passed_software_boundary": False,
    }
    estop_asserted = False

    try:
        if not node.spin_until(
            lambda: node.identity_valid is not None
            and node.estop_latched is not None,
            8.0,
        ):
            raise RuntimeError("F407 identity/estop topics unavailable")
        report["firmware_info"] = node.firmware_info
        report["identity_valid"] = node.identity_valid
        report["initial_estop_latched"] = node.estop_latched
        report["initial_lift"] = lift_snapshot(node.latest_lift)
        if node.identity_valid is not True:
            raise RuntimeError("F407 firmware identity is invalid")
        if node.estop_latched is not True:
            raise RuntimeError("Initial F407 estop is not latched")
        if node.latest_lift is not None:
            if bool(node.latest_lift.moving):
                raise RuntimeError("Fixture reports moving before servo test")
            if bool(node.latest_lift.electromagnet_on):
                raise RuntimeError("Electromagnet reports ON before servo test")

        ok, message = node.call_trigger(node.clear_estop, "clear_estop")
        report["steps"].append({"step": "clear_estop", "ok": ok, "message": message})
        if not ok or not node.spin_until(lambda: node.estop_latched is False, 3.0):
            raise RuntimeError("F407 estop did not clear")

        command_started = time.monotonic()
        ok, message = node.send_servo_right_command()
        report["steps"].append(
            {"step": "send_servo_right_command", "ok": ok, "message": message}
        )
        if not ok:
            raise RuntimeError(f"Servo-right command rejected: {message}")

        print(
            "[WATCH] Servo should turn directly RIGHT immediately; "
            "no lift/actuator/magnet command is sent; automatic estop at t=5.0s.",
            flush=True,
        )
        while time.monotonic() - command_started < TEST_WINDOW_S:
            rclpy.spin_once(node, timeout_sec=0.03)

        ok, message = node.call_trigger(node.estop, "estop")
        estop_asserted = True
        report["steps"].append({"step": "estop", "ok": ok, "message": message})
        report["actual_window_s"] = time.monotonic() - command_started
        if not ok or not node.spin_until(lambda: node.estop_latched is True, 3.0):
            raise RuntimeError("F407 estop did not relatch")

        node.spin_until(lambda: node.latest_lift is not None, 0.5)
        report["final_lift"] = lift_snapshot(node.latest_lift)
        report["final_estop_latched"] = node.estop_latched
        report["passed_software_boundary"] = True
        print("[SAFE] Test window ended; F407 estop is latched.", flush=True)
        print("[RESULT] Report whether the servo physically turned RIGHT.", flush=True)
        return 0
    except Exception as exc:
        report["error"] = str(exc)
        print(f"[ERROR] {exc}", flush=True)
        return 1
    finally:
        if not estop_asserted:
            try:
                ok, message = node.call_trigger(node.estop, "estop_finally")
                report["steps"].append(
                    {"step": "estop_finally", "ok": ok, "message": message}
                )
                node.spin_until(lambda: node.estop_latched is True, 2.0)
            except Exception as exc:
                report["estop_finally_error"] = str(exc)
        report["finished_unix_s"] = time.time()
        report_path = Path(args.report).expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[REPORT] {report_path}", flush=True)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
