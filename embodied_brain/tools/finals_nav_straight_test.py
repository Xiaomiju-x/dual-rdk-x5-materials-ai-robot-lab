#!/usr/bin/env python3
"""Plan first, then optionally execute one supervised straight Nav2 path."""
from __future__ import annotations

import argparse
import json
import math
import time
from typing import Any

import rclpy
from action_msgs.msg import GoalStatus
from finals_nav_path_guard import validate_straight_path
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import ComputePathToPose, FollowPath
from nav_msgs.msg import Path
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener

EXECUTION_CONFIRMATION = "SAFE_STRAIGHT_ONLY"


class RunAbort(RuntimeError):
    def __init__(self, reason: str, exit_code: int) -> None:
        super().__init__(reason)
        self.reason = reason
        self.exit_code = exit_code


class StraightTest(Node):
    def __init__(self, command_topic: str = "/cmd_vel_safe") -> None:
        super().__init__("finals_nav_straight_test")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.planner = ActionClient(self, ComputePathToPose, "/compute_path_to_pose")
        self.follower = ActionClient(self, FollowPath, "/follow_path")
        self.clear_client = self.create_client(Trigger, "/clear_estop")
        self.estop_client = self.create_client(Trigger, "/estop")
        self.estop_latched: bool | None = None
        self.monitor_commands = False
        self.max_angular = 0.0
        self.max_lateral_command = 0.0
        self.min_linear_x = 0.0
        self.safe_command_count = 0
        self.tripwire_reason: str | None = None
        self.angular_limit = 0.10
        self.lateral_command_limit = 0.03
        self.reverse_command_limit = 0.015
        self.create_subscription(Bool, "/f407/estop_latched", self._on_estop, 10)
        self.command_topic = command_topic
        self.create_subscription(Twist, command_topic, self._on_safe_cmd, 20)

    def _on_estop(self, message: Bool) -> None:
        self.estop_latched = bool(message.data)

    def _on_safe_cmd(self, message: Twist) -> None:
        if not self.monitor_commands:
            return
        angular = float(message.angular.z)
        lateral = float(message.linear.y)
        linear_x = float(message.linear.x)
        self.safe_command_count += 1
        self.max_angular = max(self.max_angular, abs(angular))
        self.max_lateral_command = max(self.max_lateral_command, abs(lateral))
        self.min_linear_x = min(self.min_linear_x, linear_x)
        if abs(angular) > self.angular_limit:
            self.tripwire_reason = "ANGULAR_COMMAND_TRIPWIRE"
        elif abs(lateral) > self.lateral_command_limit:
            self.tripwire_reason = "LATERAL_COMMAND_TRIPWIRE"
        elif linear_x < -self.reverse_command_limit:
            self.tripwire_reason = "REVERSE_COMMAND_TRIPWIRE"

    def reset_command_guard(
        self,
        *,
        angular_limit: float,
        lateral_limit: float,
        reverse_limit: float,
    ) -> None:
        self.angular_limit = angular_limit
        self.lateral_command_limit = lateral_limit
        self.reverse_command_limit = reverse_limit
        self.max_angular = 0.0
        self.max_lateral_command = 0.0
        self.min_linear_x = 0.0
        self.safe_command_count = 0
        self.tripwire_reason = None
        self.monitor_commands = True

    def wait_until(self, predicate, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if predicate():
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return bool(predicate())

    def trigger(self, client, name: str, timeout_s: float = 5.0) -> tuple[bool, str]:
        if not client.wait_for_service(timeout_sec=timeout_s):
            return False, f"{name}_service_unavailable"
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_s)
        if not future.done() or future.result() is None:
            return False, f"{name}_timeout"
        response = future.result()
        return bool(response.success), str(response.message)

    def latch_estop(self) -> tuple[bool, str]:
        ok, message = self.trigger(self.estop_client, "estop")
        confirmed = self.wait_until(lambda: self.estop_latched is True, 3.0)
        return ok and confirmed, message

    def lookup_robot_pose(self, target_frame: str = "map", timeout_s: float = 8.0):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                if self.tf_buffer.can_transform(
                    target_frame,
                    "base_footprint",
                    Time(),
                    timeout=Duration(seconds=0.0),
                ):
                    return self.tf_buffer.lookup_transform(target_frame, "base_footprint", Time())
            except Exception:
                pass
            rclpy.spin_once(self, timeout_sec=0.1)
        raise RunAbort("MAP_TRANSFORM_UNAVAILABLE", 5)


def yaw_from_quaternion(quaternion: Any) -> float:
    siny = 2.0 * (
        float(quaternion.w) * float(quaternion.z)
        + float(quaternion.x) * float(quaternion.y)
    )
    cosy = 1.0 - 2.0 * (
        float(quaternion.y) ** 2 + float(quaternion.z) ** 2
    )
    return math.atan2(siny, cosy)


def pose_from_transform(node: Node, transform: Any, frame_id: str = "map") -> PoseStamped:
    pose = PoseStamped()
    pose.header.frame_id = frame_id
    pose.header.stamp = node.get_clock().now().to_msg()
    pose.pose.position.x = float(transform.transform.translation.x)
    pose.pose.position.y = float(transform.transform.translation.y)
    pose.pose.position.z = float(transform.transform.translation.z)
    pose.pose.orientation = transform.transform.rotation
    return pose


def spin_for_future(node: Node, future, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and not future.done():
        rclpy.spin_once(node, timeout_sec=0.05)
    return bool(future.done())


def send_action_goal(node: Node, client, goal: Any, *, timeout_s: float, label: str):
    send_future = client.send_goal_async(goal)
    if not spin_for_future(node, send_future, timeout_s):
        raise RunAbort(f"{label}_SEND_TIMEOUT", 6)
    handle = send_future.result()
    if handle is None or not handle.accepted:
        raise RunAbort(f"{label}_REJECTED", 7)
    return handle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distance", type=float, default=0.35)
    parser.add_argument("--plan-timeout", type=float, default=10.0)
    parser.add_argument("--motion-timeout", type=float, default=22.0)
    parser.add_argument("--max-path-lateral", type=float, default=0.07)
    parser.add_argument("--max-path-heading", type=float, default=0.15)
    parser.add_argument("--max-path-backtrack", type=float, default=0.02)
    parser.add_argument("--endpoint-tolerance", type=float, default=0.10)
    parser.add_argument("--max-angular-command", type=float, default=0.10)
    parser.add_argument("--max-lateral-command", type=float, default=0.03)
    parser.add_argument("--max-reverse-command", type=float, default=0.015)
    parser.add_argument("--planner-id", default="GridBased")
    parser.add_argument("--controller-id", default="")
    parser.add_argument("--goal-checker-id", default="")
    parser.add_argument(
        "--direct-path",
        action="store_true",
        help="Execute a geometrically straight Path through Nav2 FollowPath without global replanning.",
    )
    parser.add_argument(
        "--path-frame",
        choices=("map", "odom"),
        default="map",
        help="Frame for a direct straight path; odom avoids SLAM map corrections.",
    )
    parser.add_argument(
        "--command-topic",
        choices=("/cmd_vel", "/cmd_vel_safe"),
        default="/cmd_vel_safe",
        help="Actuator command topic to monitor for straight-motion tripwires.",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation", default="")
    arguments = parser.parse_args()
    if not 0.10 <= arguments.distance <= 0.50:
        parser.error("distance must be in [0.10, 0.50] m")
    if arguments.path_frame != "map" and not arguments.direct_path:
        parser.error("--path-frame odom requires --direct-path")
    if arguments.execute and arguments.confirmation != EXECUTION_CONFIRMATION:
        parser.error(
            f"--execute requires --confirmation {EXECUTION_CONFIRMATION} after fresh physical safety confirmation"
        )

    rclpy.init()
    node = StraightTest(arguments.command_topic)
    result: dict[str, Any] = {
        "ok": False,
        "mode": "execute" if arguments.execute else "plan_only",
        "distance_m": arguments.distance,
        "estop_relatched": False,
        "executed": False,
        "command_topic": arguments.command_topic,
        "path_frame": arguments.path_frame,
    }
    exit_code = 1
    plan_handle = None
    follow_handle = None
    follow_result_future = None
    try:
        if not node.wait_until(lambda: node.estop_latched is not None, 5.0):
            raise RunAbort("ESTOP_STATE_UNAVAILABLE", 2)
        if node.estop_latched is not True:
            raise RunAbort("PRECONDITION_ESTOP_NOT_LATCHED", 3)
        if not arguments.direct_path and not node.planner.wait_for_server(timeout_sec=6.0):
            raise RunAbort("COMPUTE_PATH_ACTION_UNAVAILABLE", 4)

        transform = node.lookup_robot_pose(arguments.path_frame)
        start = pose_from_transform(node, transform, arguments.path_frame)
        start_yaw = yaw_from_quaternion(start.pose.orientation)
        goal = PoseStamped()
        goal.header.frame_id = arguments.path_frame
        goal.header.stamp = node.get_clock().now().to_msg()
        goal.pose.position.x = start.pose.position.x + arguments.distance * math.cos(start_yaw)
        goal.pose.position.y = start.pose.position.y + arguments.distance * math.sin(start_yaw)
        goal.pose.orientation = start.pose.orientation
        result["start"] = [
            round(start.pose.position.x, 5),
            round(start.pose.position.y, 5),
            round(start_yaw, 5),
        ]
        result["goal"] = [
            round(goal.pose.position.x, 5),
            round(goal.pose.position.y, 5),
            round(start_yaw, 5),
        ]

        if arguments.direct_path:
            path = Path()
            path.header = goal.header
            path.header.stamp.sec = 0
            path.header.stamp.nanosec = 0
            segment_count = max(2, int(math.ceil(arguments.distance / 0.05)))
            for index in range(segment_count + 1):
                fraction = index / segment_count
                pose = PoseStamped()
                pose.header.frame_id = arguments.path_frame
                pose.header.stamp.sec = 0
                pose.header.stamp.nanosec = 0
                pose.pose.position.x = start.pose.position.x + fraction * arguments.distance * math.cos(start_yaw)
                pose.pose.position.y = start.pose.position.y + fraction * arguments.distance * math.sin(start_yaw)
                pose.pose.orientation = start.pose.orientation
                path.poses.append(pose)
            result["plan_action_status"] = "DIRECT_STRAIGHT_PATH"
        else:
            plan_goal = ComputePathToPose.Goal()
            plan_goal.start = start
            plan_goal.goal = goal
            plan_goal.planner_id = arguments.planner_id
            plan_goal.use_start = True
            plan_handle = send_action_goal(
                node,
                node.planner,
                plan_goal,
                timeout_s=6.0,
                label="COMPUTE_PATH",
            )
            plan_result_future = plan_handle.get_result_async()
            if not spin_for_future(node, plan_result_future, arguments.plan_timeout):
                plan_handle.cancel_goal_async()
                raise RunAbort("COMPUTE_PATH_RESULT_TIMEOUT", 8)
            wrapped_plan = plan_result_future.result()
            if wrapped_plan is None or int(wrapped_plan.status) != GoalStatus.STATUS_SUCCEEDED:
                status = None if wrapped_plan is None else int(wrapped_plan.status)
                result["plan_action_status"] = status
                raise RunAbort("COMPUTE_PATH_FAILED", 9)
            path = wrapped_plan.result.path
            result["plan_action_status"] = int(wrapped_plan.status)
        if path.header.frame_id != arguments.path_frame or any(
            pose.header.frame_id not in ("", arguments.path_frame) for pose in path.poses
        ):
            raise RunAbort("PATH_FRAME_MISMATCH", 10)
        points = [(pose.pose.position.x, pose.pose.position.y) for pose in path.poses]
        guard = validate_straight_path(
            points,
            start_x=start.pose.position.x,
            start_y=start.pose.position.y,
            start_yaw=start_yaw,
            requested_distance=arguments.distance,
            max_lateral_m=arguments.max_path_lateral,
            max_heading_rad=arguments.max_path_heading,
            max_backtrack_m=arguments.max_path_backtrack,
            endpoint_tolerance_m=arguments.endpoint_tolerance,
        )
        result["path_guard"] = guard
        if not guard["ok"]:
            raise RunAbort("PATH_GUARD_REJECTED", 11)
        result["ready_to_execute"] = True

        if not arguments.execute:
            result["ok"] = True
            result["reason"] = "PLAN_VALID_ESTOP_REMAINS_LATCHED"
            exit_code = 0
        else:
            if not node.follower.wait_for_server(timeout_sec=6.0):
                raise RunAbort("FOLLOW_PATH_ACTION_UNAVAILABLE", 12)
            follow_goal = FollowPath.Goal()
            follow_goal.path = path
            follow_goal.controller_id = arguments.controller_id
            follow_goal.goal_checker_id = arguments.goal_checker_id
            node.reset_command_guard(
                angular_limit=arguments.max_angular_command,
                lateral_limit=arguments.max_lateral_command,
                reverse_limit=arguments.max_reverse_command,
            )
            follow_handle = send_action_goal(
                node,
                node.follower,
                follow_goal,
                timeout_s=6.0,
                label="FOLLOW_PATH",
            )
            follow_result_future = follow_handle.get_result_async()

            clear_ok, clear_message = node.trigger(node.clear_client, "clear_estop")
            result["clear_estop"] = {"ok": clear_ok, "message": clear_message}
            if not clear_ok or not node.wait_until(
                lambda: node.estop_latched is False,
                3.0,
            ):
                raise RunAbort("CLEAR_ESTOP_NOT_CONFIRMED", 13)
            result["executed"] = True

            deadline = time.monotonic() + arguments.motion_timeout
            while time.monotonic() < deadline and not follow_result_future.done():
                rclpy.spin_once(node, timeout_sec=0.03)
                if node.tripwire_reason is not None:
                    node.latch_estop()
                    follow_handle.cancel_goal_async()
                    raise RunAbort(node.tripwire_reason, 14)
                if node.estop_latched is True:
                    follow_handle.cancel_goal_async()
                    raise RunAbort("ESTOP_RELATCHED_DURING_FOLLOW", 15)
            if not follow_result_future.done():
                node.latch_estop()
                follow_handle.cancel_goal_async()
                raise RunAbort("FOLLOW_PATH_TIMEOUT", 16)
            wrapped_follow = follow_result_future.result()
            status = None if wrapped_follow is None else int(wrapped_follow.status)
            result["follow_action_status"] = status
            if status != GoalStatus.STATUS_SUCCEEDED:
                raise RunAbort("FOLLOW_PATH_FAILED", 17)

            final_transform = node.lookup_robot_pose(arguments.path_frame, timeout_s=3.0)
            dx = float(final_transform.transform.translation.x) - start.pose.position.x
            dy = float(final_transform.transform.translation.y) - start.pose.position.y
            actual_forward = math.cos(start_yaw) * dx + math.sin(start_yaw) * dy
            actual_lateral = -math.sin(start_yaw) * dx + math.cos(start_yaw) * dy
            result["actual_motion"] = {
                "forward_m": round(actual_forward, 5),
                "lateral_m": round(actual_lateral, 5),
            }
            if actual_forward < 0.60 * arguments.distance:
                raise RunAbort("INSUFFICIENT_FORWARD_DISPLACEMENT", 18)
            if abs(actual_lateral) > arguments.max_path_lateral:
                raise RunAbort("ACTUAL_LATERAL_DEVIATION", 19)
            result["ok"] = True
            result["reason"] = "FOLLOW_PATH_STRAIGHT_TEST_SUCCEEDED"
            exit_code = 0
    except RunAbort as exc:
        result["reason"] = exc.reason
        exit_code = exc.exit_code
    except KeyboardInterrupt:
        result["reason"] = "INTERRUPTED"
        exit_code = 130
    except Exception as exc:
        result["reason"] = f"UNEXPECTED_{type(exc).__name__}:{exc}"
        exit_code = 99
    finally:
        node.monitor_commands = False
        if follow_handle is not None and (
            follow_result_future is None or not follow_result_future.done()
        ):
            try:
                cancel_future = follow_handle.cancel_goal_async()
                spin_for_future(node, cancel_future, 1.0)
            except Exception:
                pass
        estop_ok, estop_message = node.latch_estop()
        result["estop_relatched"] = estop_ok
        result["estop_message"] = estop_message
        result["command_guard"] = {
            "safe_command_count": node.safe_command_count,
            "max_angular_seen": round(node.max_angular, 5),
            "max_lateral_seen": round(node.max_lateral_command, 5),
            "min_linear_x_seen": round(node.min_linear_x, 5),
            "tripwire_reason": node.tripwire_reason,
        }
        if not estop_ok:
            result["ok"] = False
            result["reason"] = "FINAL_ESTOP_RELATCH_FAILED"
            exit_code = 20
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
