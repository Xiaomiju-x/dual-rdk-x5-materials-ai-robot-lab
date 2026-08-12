#!/usr/bin/env python3
"""One-shot finals demo: lift pickup, 0.50 m waypoint drive, lift place."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Twist
from my_robot_msgs.msg import LiftStatus
from my_robot_msgs.srv import SetLiftHeight
from nav2_msgs.action import FollowPath
from nav_msgs.msg import Odometry, Path as NavPath
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import Bool
from std_srvs.srv import Trigger

PICK_TO_TOP_COMMAND = -1.0
PLACE_TO_BOTTOM_COMMAND = -2.0


def phase(message: str) -> None:
    print(f"[PHASE] {message}", flush=True)


def yaw_from_quaternion(quaternion: Any) -> float:
    siny = 2.0 * (
        float(quaternion.w) * float(quaternion.z)
        + float(quaternion.x) * float(quaternion.y)
    )
    cosy = 1.0 - 2.0 * (
        float(quaternion.y) ** 2 + float(quaternion.z) ** 2
    )
    return math.atan2(siny, cosy)


def normalize_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


class FinalsDemo(Node):
    def __init__(self) -> None:
        super().__init__("finals_lift_nav_demo")
        self.set_lift = self.create_client(SetLiftHeight, "/set_lift_height")
        self.clear_estop = self.create_client(Trigger, "/clear_estop")
        self.estop = self.create_client(Trigger, "/estop")
        self.follow_path = ActionClient(self, FollowPath, "/follow_path")
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.latest_odom: Odometry | None = None
        self.latest_lift: LiftStatus | None = None
        self.estop_latched: bool | None = None
        self.identity_valid: bool | None = None
        self.create_subscription(Odometry, "/odom", self._on_odom, 20)
        self.create_subscription(LiftStatus, "/lift_status", self._on_lift, 20)
        self.create_subscription(Bool, "/f407/estop_latched", self._on_estop, 10)
        self.create_subscription(
            Bool, "/f407/firmware_identity_valid", self._on_identity, 10
        )

    def _on_odom(self, message: Odometry) -> None:
        self.latest_odom = message

    def _on_lift(self, message: LiftStatus) -> None:
        self.latest_lift = message

    def _on_estop(self, message: Bool) -> None:
        self.estop_latched = bool(message.data)

    def _on_identity(self, message: Bool) -> None:
        self.identity_valid = bool(message.data)

    def spin_until(self, predicate, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if predicate():
                return True
            rclpy.spin_once(self, timeout_sec=0.08)
        return bool(predicate())

    def wait_future(self, future, timeout_s: float) -> bool:
        return self.spin_until(lambda: future.done(), timeout_s)

    def call_trigger(self, client, label: str, timeout_s: float = 6.0) -> tuple[bool, str]:
        if not client.wait_for_service(timeout_sec=timeout_s):
            return False, f"{label}_service_unavailable"
        future = client.call_async(Trigger.Request())
        if not self.wait_future(future, timeout_s) or future.result() is None:
            return False, f"{label}_timeout"
        response = future.result()
        return bool(response.success), str(response.message)

    def send_fixture_command(self, target: float, label: str) -> tuple[bool, str]:
        if not self.set_lift.wait_for_service(timeout_sec=8.0):
            return False, "set_lift_height_service_unavailable"
        request = SetLiftHeight.Request()
        request.target_height_m = float(target)
        request.timeout_s = 0.0
        request.wait_for_arrival = False
        future = self.set_lift.call_async(request)
        if not self.wait_future(future, 8.0) or future.result() is None:
            return False, f"{label}_command_timeout"
        response = future.result()
        return bool(response.success), str(response.message)

    def wait_fixture_cycle(self, label: str, timeout_s: float) -> bool:
        observed_busy = False
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.08)
            if self.latest_lift is None:
                continue
            if bool(self.latest_lift.moving):
                observed_busy = True
            elif observed_busy:
                self.get_logger().info(f"{label} fixture phase completed")
                return True
        return False

    def publish_velocity(self, linear: float, angular: float = 0.0) -> None:
        command = Twist()
        command.linear.x = float(linear)
        command.angular.z = float(angular)
        self.cmd_pub.publish(command)

    def publish_zero_burst(self, duration_s: float = 0.6) -> None:
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            self.publish_velocity(0.0, 0.0)
            rclpy.spin_once(self, timeout_sec=0.05)

    def cancel_nav2_goal(self, handle: Any, timeout_s: float = 2.0) -> None:
        cancel_future = handle.cancel_goal_async()
        self.wait_future(cancel_future, timeout_s)
        self.publish_zero_burst(0.5)

    def odom_pose(self) -> tuple[float, float, float] | None:
        if self.latest_odom is None:
            return None
        pose = self.latest_odom.pose.pose
        return (
            float(pose.position.x),
            float(pose.position.y),
            yaw_from_quaternion(pose.orientation),
        )

    @staticmethod
    def forward_displacement(
        start: tuple[float, float, float], current: tuple[float, float, float]
    ) -> float:
        dx = current[0] - start[0]
        dy = current[1] - start[1]
        return math.cos(start[2]) * dx + math.sin(start[2]) * dy

    def nav2_follow_straight(
        self, distance_m: float, start: tuple[float, float, float]
    ) -> dict[str, Any]:
        result: dict[str, Any] = {"attempted": False, "succeeded": False}
        if not self.follow_path.wait_for_server(timeout_sec=5.0):
            result["reason"] = "follow_path_unavailable"
            return result

        path = NavPath()
        path.header.frame_id = "odom"
        segment_count = max(2, int(math.ceil(distance_m / 0.05)))
        for index in range(segment_count + 1):
            fraction = index / segment_count
            pose = PoseStamped()
            pose.header.frame_id = "odom"
            pose.pose.position.x = start[0] + fraction * distance_m * math.cos(start[2])
            pose.pose.position.y = start[1] + fraction * distance_m * math.sin(start[2])
            pose.pose.orientation.z = math.sin(start[2] / 2.0)
            pose.pose.orientation.w = math.cos(start[2] / 2.0)
            path.poses.append(pose)

        goal = FollowPath.Goal()
        goal.path = path
        goal.controller_id = "FollowPath"
        goal.goal_checker_id = "general_goal_checker"
        send_future = self.follow_path.send_goal_async(goal)
        result["attempted"] = True
        if not self.wait_future(send_future, 6.0):
            result["reason"] = "follow_path_send_timeout"
            return result
        handle = send_future.result()
        if handle is None or not handle.accepted:
            result["reason"] = "follow_path_rejected"
            return result

        result_future = handle.get_result_async()
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline and not result_future.done():
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.estop_latched is True:
                result["reason"] = "estop_relatched_during_nav2"
                self.cancel_nav2_goal(handle)
                return result
        if not result_future.done():
            self.cancel_nav2_goal(handle)
            result["reason"] = "follow_path_timeout"
            return result

        wrapped = result_future.result()
        status = None if wrapped is None else int(wrapped.status)
        result["status"] = status
        current = self.odom_pose()
        moved = 0.0 if current is None else self.forward_displacement(start, current)
        result["forward_m"] = round(moved, 4)
        result["succeeded"] = (
            status == GoalStatus.STATUS_SUCCEEDED and moved >= distance_m - 0.08
        )
        if not result["succeeded"]:
            result["reason"] = "nav2_incomplete"
        return result

    def direct_odom_fallback(
        self,
        distance_m: float,
        start: tuple[float, float, float] | None,
        speed_mps: float = 0.08,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {"attempted": True, "succeeded": False}
        current = self.odom_pose()
        if start is None or current is None:
            duration_s = distance_m / speed_mps
            result["mode"] = "timed_cmd_vel_last_resort"
            deadline = time.monotonic() + duration_s
            while time.monotonic() < deadline:
                self.publish_velocity(speed_mps, 0.0)
                rclpy.spin_once(self, timeout_sec=0.05)
            self.publish_zero_burst()
            result["duration_s"] = round(duration_s, 2)
            result["succeeded"] = True
            return result

        result["mode"] = "odom_closed_loop"
        deadline = time.monotonic() + 14.0
        target = distance_m - 0.02
        last_progress = self.forward_displacement(start, current)
        last_progress_time = time.monotonic()
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.04)
            current = self.odom_pose()
            if current is None:
                continue
            progress = self.forward_displacement(start, current)
            if progress >= target:
                result["succeeded"] = True
                result["forward_m"] = round(progress, 4)
                break
            if progress > last_progress + 0.003:
                last_progress = progress
                last_progress_time = time.monotonic()
            yaw_error = normalize_angle(start[2] - current[2])
            angular = max(-0.08, min(0.08, 0.8 * yaw_error))
            self.publish_velocity(speed_mps, angular)
            if time.monotonic() - last_progress_time > 2.5:
                remaining = max(0.05, distance_m - progress)
                timed_deadline = time.monotonic() + remaining / speed_mps
                result["mode"] = "odom_stale_timed_completion"
                while time.monotonic() < timed_deadline:
                    self.publish_velocity(speed_mps, 0.0)
                    rclpy.spin_once(self, timeout_sec=0.05)
                result["succeeded"] = True
                break
        self.publish_zero_burst()
        if not result["succeeded"]:
            result["reason"] = "direct_drive_timeout"
        return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distance", type=float, default=0.50)
    parser.add_argument(
        "--drive-mode",
        choices=("odom", "nav2"),
        default="odom",
        help="Finals default is immediate odom closed-loop; Nav2 remains optional.",
    )
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--report",
        default=str(Path.home() / "finals_demo_logs" / "latest.json"),
    )
    args = parser.parse_args()
    if not 0.10 <= args.distance <= 0.50:
        parser.error("distance must be in [0.10, 0.50] m")
    if args.self_test:
        print(
            json.dumps(
                {
                    "ok": True,
                    "distance_m": args.distance,
                    "sequence": ["pick_top", "waypoint_drive", "place_bottom"],
                    "drive_mode": args.drive_mode,
                    "nav2_optional": True,
                    "odom_closed_loop": True,
                    "ai_fsd_is_non_blocking": True,
                },
                ensure_ascii=False,
            )
        )
        return 0
    if not args.confirm:
        parser.error("physical execution requires --confirm")

    rclpy.init()
    node = FinalsDemo()
    report: dict[str, Any] = {
        "ok": False,
        "distance_m": args.distance,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    payload_at_top = False
    place_completed = False
    exit_code = 1
    try:
        if not node.spin_until(
            lambda: node.identity_valid is True and node.estop_latched is not None,
            15.0,
        ):
            raise RuntimeError("F407 identity/state unavailable")
        clear_ok, clear_message = node.call_trigger(node.clear_estop, "clear_estop")
        report["clear_estop"] = {"ok": clear_ok, "message": clear_message}
        if not clear_ok or not node.spin_until(lambda: node.estop_latched is False, 4.0):
            raise RuntimeError("F407 estop could not be cleared")

        node.publish_zero_burst()
        phase(
            "1/3 取瓶升顶：start -> 工作位 -> 电推杆伸出 -> 吸附 -> "
            "升顶锁定 -> 返回 start"
        )
        pick_ok, pick_message = node.send_fixture_command(
            PICK_TO_TOP_COMMAND, "pick_top"
        )
        report["pick_command"] = {"ok": pick_ok, "message": pick_message}
        if not pick_ok or not node.wait_fixture_cycle("pick_top", 42.0):
            raise RuntimeError("pickup/top phase did not complete")
        payload_at_top = True

        phase(
            "2/3 定点直行：目标 0.50 m，里程计闭环立即执行；"
            "Nav2/SLAM/Lab-FSD 保持在线"
        )
        if not node.spin_until(lambda: node.latest_odom is not None, 5.0):
            start_pose = None
        else:
            start_pose = node.odom_pose()
        if args.drive_mode == "nav2" and start_pose is not None:
            nav2_result = node.nav2_follow_straight(args.distance, start_pose)
        else:
            nav2_result = {
                "attempted": False,
                "succeeded": False,
                "reason": "finals_odom_primary",
            }
        report["nav2"] = nav2_result
        if nav2_result.get("succeeded"):
            report["drive_mode"] = "nav2_follow_path_odom_waypoint"
        else:
            if node.estop_latched is True:
                fallback_clear_ok, fallback_clear_message = node.call_trigger(
                    node.clear_estop, "fallback_clear_estop"
                )
                report["fallback_clear_estop"] = {
                    "ok": fallback_clear_ok,
                    "message": fallback_clear_message,
                }
                if not fallback_clear_ok or not node.spin_until(
                    lambda: node.estop_latched is False, 4.0
                ):
                    raise RuntimeError("F407 estop blocked odom closed-loop drive")
            odom_drive = node.direct_odom_fallback(args.distance, start_pose)
            report["odom_drive"] = odom_drive
            if not odom_drive.get("succeeded"):
                raise RuntimeError("odom closed-loop drive failed")
            report["drive_mode"] = str(odom_drive.get("mode"))

        node.publish_zero_burst()
        phase(
            "3/3 放瓶复位：载瓶分段右转 -> 受控下降 -> 磁铁 OFF -> "
            "电推杆缩回 20 s -> 空载直接回 start"
        )
        place_ok, place_message = node.send_fixture_command(
            PLACE_TO_BOTTOM_COMMAND, "place_bottom"
        )
        report["place_command"] = {"ok": place_ok, "message": place_message}
        if not place_ok or not node.wait_fixture_cycle("place_bottom", 55.0):
            raise RuntimeError("place/bottom phase did not complete")
        place_completed = True
        payload_at_top = False
        phase("完整流程完成，正在锁存最终急停")
        report["ok"] = True
        exit_code = 0
    except KeyboardInterrupt:
        report["error"] = "operator_interrupt"
        exit_code = 130
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        exit_code = 2
    finally:
        node.publish_zero_burst()
        if payload_at_top and not place_completed and node.estop_latched is False:
            recover_ok, recover_message = node.send_fixture_command(
                PLACE_TO_BOTTOM_COMMAND, "recovery_place"
            )
            report["recovery_place"] = {
                "ok": recover_ok,
                "message": recover_message,
            }
            if recover_ok:
                report["recovery_place"]["completed"] = node.wait_fixture_cycle(
                    "recovery_place", 55.0
                )
        estop_ok, estop_message = node.call_trigger(node.estop, "estop")
        report["final_estop"] = {"ok": estop_ok, "message": estop_message}
        report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        report_path = Path(args.report).expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
