#!/usr/bin/env python3
"""Run the installed DispatchTask server in an isolated, hardware-free ROS domain.

The test starts the real ``dispatch_server`` executable with ``stub_mode=true``
and verifies a complete fetch-sample stage sequence. Fake F407 services and a
``/cmd_vel`` subscriber prove that the successful simulation path performs no
actuator calls and publishes no vehicle command. Separate goals then prove that
F407 estop and a hard Lab-FSD safety reason still reject dispatch.

This is software integration evidence only. It never opens ``/dev/F407`` and
cannot replace the post-flash physical interlock test.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


EXPECTED_FETCH_STAGES = [1, 2, 3, 5, 4, 6, 5, 8]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate_dispatch_executable(explicit: str) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
    else:
        prefix = subprocess.check_output(
            ["ros2", "pkg", "prefix", "my_robot_agents"], text=True
        ).strip()
        path = (Path(prefix) / "lib" / "my_robot_agents" / "dispatch_server").resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise FileNotFoundError(f"dispatch_server not executable: {path}")
    return path


class RosProbe:
    def __init__(self, scope: str) -> None:
        import rclpy
        from geometry_msgs.msg import Twist
        from my_robot_msgs.action import DispatchTask
        from my_robot_msgs.srv import SetElectromagnet, SetLiftHeight
        from rclpy.action import ActionClient
        from std_msgs.msg import Bool, String

        self.rclpy = rclpy
        self.DispatchTask = DispatchTask
        self.Bool = Bool
        self.String = String
        rclpy.init(args=None)
        self.node = rclpy.create_node("dispatch_stub_integration")
        self.scope = scope.rstrip("/")
        self.action = ActionClient(
            self.node, DispatchTask, f"{self.scope}/dispatch_task"
        )
        self.estop_pub = self.node.create_publisher(
            Bool, f"{self.scope}/f407/estop_latched", 10
        )
        self.gate_pub = self.node.create_publisher(
            String, f"{self.scope}/lab_fsd/safety_gate", 10
        )
        self.cmd_vel_messages: list[dict[str, float]] = []
        self.literal_cmd_vel_messages: list[dict[str, float]] = []
        self.service_calls = {"set_lift_height": 0, "set_electromagnet": 0}
        self.node.create_subscription(
            Twist, f"{self.scope}/cmd_vel_sink", self._on_cmd_vel, 10
        )
        self.node.create_subscription(
            Twist, "/cmd_vel", self._on_literal_cmd_vel, 10
        )
        self.node.create_service(
            SetLiftHeight,
            f"{self.scope}/set_lift_height_tripwire",
            self._on_set_lift_height,
        )
        self.node.create_service(
            SetElectromagnet,
            f"{self.scope}/set_electromagnet_tripwire",
            self._on_set_electromagnet,
        )

    def _on_cmd_vel(self, msg: Any) -> None:
        self.cmd_vel_messages.append(
            {"linear_x": float(msg.linear.x), "angular_z": float(msg.angular.z)}
        )

    def _on_literal_cmd_vel(self, msg: Any) -> None:
        self.literal_cmd_vel_messages.append(
            {"linear_x": float(msg.linear.x), "angular_z": float(msg.angular.z)}
        )

    def _on_set_lift_height(self, _request: Any, response: Any) -> Any:
        self.service_calls["set_lift_height"] += 1
        response.success = False
        response.message = "integration sentinel: this service must not be called"
        return response

    def _on_set_electromagnet(self, _request: Any, response: Any) -> Any:
        self.service_calls["set_electromagnet"] += 1
        response.success = False
        response.message = "integration sentinel: this service must not be called"
        return response

    def spin_for(self, duration_s: float) -> None:
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            timeout = max(0.0, min(0.05, deadline - time.monotonic()))
            self.rclpy.spin_once(self.node, timeout_sec=timeout)

    def wait_for(self, predicate: Callable[[], bool], timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
            if predicate():
                return True
        return bool(predicate())

    def wait_for_server(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.action.wait_for_server(timeout_sec=0.2):
                return True
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
        return False

    def literal_cmd_vel_publishers(self) -> list[str]:
        return sorted(
            {
                info.node_namespace.rstrip("/") + "/" + info.node_name
                for info in self.node.get_publishers_info_by_topic("/cmd_vel")
            }
        )

    def publish_estop(self, latched: bool) -> None:
        msg = self.Bool()
        msg.data = bool(latched)
        for _ in range(4):
            self.estop_pub.publish(msg)
            self.spin_for(0.10)

    def publish_hard_guard(self, reason: str) -> None:
        msg = self.String()
        msg.data = json.dumps({"reasons": [reason]}, separators=(",", ":"))
        for _ in range(4):
            self.gate_pub.publish(msg)
            self.spin_for(0.10)

    def send_goal(
        self,
        *,
        task_id: str,
        task_type: str,
        timeout_s: float,
    ) -> dict[str, Any]:
        goal = self.DispatchTask.Goal()
        goal.task_id = task_id
        goal.task_type = task_type
        goal.bottle_id = "integration_bottle"
        goal.from_location = "shelf_1_slot_1"
        goal.to_location = "home"
        goal.priority = self.DispatchTask.Goal.PRIORITY_NORMAL
        goal.timeout_s = float(timeout_s)
        feedback: list[dict[str, Any]] = []

        def on_feedback(message: Any) -> None:
            item = message.feedback
            feedback.append(
                {
                    "stage": int(item.stage),
                    "progress_pct": float(item.progress_pct),
                    "stage_message": str(item.stage_message),
                }
            )

        started = time.monotonic()
        send_future = self.action.send_goal_async(goal, feedback_callback=on_feedback)
        if not self.wait_for(send_future.done, 5.0):
            raise TimeoutError(f"{task_id}: send_goal response timeout")
        handle = send_future.result()
        record: dict[str, Any] = {
            "task_id": task_id,
            "task_type": task_type,
            "accepted": bool(handle.accepted),
            "feedback": feedback,
        }
        if not handle.accepted:
            record["elapsed_s"] = round(time.monotonic() - started, 3)
            return record

        result_future = handle.get_result_async()
        if not self.wait_for(result_future.done, timeout_s + 5.0):
            raise TimeoutError(f"{task_id}: action result timeout")
        wrapped = result_future.result()
        result = wrapped.result
        record.update(
            {
                "status": int(wrapped.status),
                "success": bool(result.success),
                "message": str(result.message),
                "server_elapsed_s": float(result.elapsed_s),
                "elapsed_s": round(time.monotonic() - started, 3),
            }
        )
        return record

    def probe_concurrent_rejection(self) -> dict[str, Any]:
        first = self.DispatchTask.Goal()
        first.task_id = "stub-concurrency-owner"
        first.task_type = "fetch_sample"
        first.bottle_id = "integration_bottle"
        first.from_location = "shelf_1_slot_1"
        first.priority = self.DispatchTask.Goal.PRIORITY_NORMAL
        first.timeout_s = 25.0
        feedback: list[int] = []

        def on_feedback(message: Any) -> None:
            feedback.append(int(message.feedback.stage))

        send_future = self.action.send_goal_async(first, feedback_callback=on_feedback)
        if not self.wait_for(send_future.done, 5.0):
            raise TimeoutError("concurrency owner send timeout")
        owner = send_future.result()
        if not owner.accepted:
            return {"owner_accepted": False, "contender_accepted": None}

        contender = self.send_goal(
            task_id="stub-concurrency-contender",
            task_type="home",
            timeout_s=5.0,
        )
        cancel_future = owner.cancel_goal_async()
        cancel_response = None
        if self.wait_for(cancel_future.done, 3.0):
            cancel_response = cancel_future.result()
        result_future = owner.get_result_async()
        owner_status = None
        owner_success = None
        owner_message = ""
        if self.wait_for(result_future.done, 5.0):
            wrapped = result_future.result()
            owner_status = int(wrapped.status)
            owner_success = bool(wrapped.result.success)
            owner_message = str(wrapped.result.message)
        return {
            "owner_accepted": True,
            "owner_feedback": feedback,
            "owner_status": owner_status,
            "owner_success": owner_success,
            "owner_message": owner_message,
            "cancel_goal_count": len(getattr(cancel_response, "goals_canceling", []) or []),
            "contender_accepted": contender.get("accepted"),
            "contender": contender,
        }

    def close(self) -> None:
        self.action.destroy()
        self.node.destroy_node()
        self.rclpy.shutdown()


def add_check(checks: list[dict[str, Any]], name: str, ok: bool, detail: str) -> None:
    checks.append({"name": name, "status": "PASS" if ok else "FAIL", "detail": detail})


def stop_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (AttributeError, ProcessLookupError):
        process.terminate()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (AttributeError, ProcessLookupError):
            process.kill()
        process.wait(timeout=3.0)


def forbidden_device_fds(pid: int) -> list[str]:
    fd_root = Path(f"/proc/{pid}/fd")
    if not fd_root.is_dir():
        return []
    forbidden: list[str] = []
    prefixes = ("/dev/F407", "/dev/ttyUSB", "/dev/ttyACM", "/dev/serial/")
    for fd_path in fd_root.iterdir():
        try:
            target = os.readlink(fd_path)
        except OSError:
            continue
        if target.startswith(prefixes):
            forbidden.append(f"{fd_path.name}:{target}")
    return sorted(forbidden)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain-id", type=int, default=120 + (os.getpid() % 113))
    parser.add_argument("--dispatch-executable", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    if not 0 <= args.domain_id <= 232:
        parser.error("--domain-id must be in [0, 232]")
    os.environ["ROS_DOMAIN_ID"] = str(args.domain_id)
    os.environ["ROS_LOCALHOST_ONLY"] = "1"
    os.environ.pop("CYCLONEDDS_URI", None)

    executable = locate_dispatch_executable(args.dispatch_executable)
    scope = f"/dispatch_stub_{os.getpid()}_{int(time.time())}"
    out_path = (
        Path(args.out).expanduser().resolve()
        if args.out
        else Path.home()
        / "dispatch_stub_evidence"
        / f"dispatch_stub_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_path.with_suffix(".log")

    command = [
        str(executable),
        "--ros-args",
        "-p",
        "stub_mode:=true",
        "-p",
        "use_nav2:=false",
        "-p",
        "execute_pickup_actuators:=false",
        "-p",
        "use_lab_fsd_guard:=true",
        "-p",
        "safe_stop_on_failure:=true",
        "-p",
        f"nav2_action_name:={scope}/navigate_to_pose_tripwire",
        "-p",
        f"vlm_service:={scope}/vlm_query_tripwire",
        "-r",
        f"__ns:={scope}",
        "-r",
        f"/cmd_vel:={scope}/cmd_vel_sink",
        "-r",
        f"/set_lift_height:={scope}/set_lift_height_tripwire",
        "-r",
        f"/set_electromagnet:={scope}/set_electromagnet_tripwire",
        "-r",
        f"/lift_status:={scope}/lift_status",
        "-r",
        f"/f407/estop_latched:={scope}/f407/estop_latched",
        "-r",
        f"/lab_fsd/safety_gate:={scope}/lab_fsd/safety_gate",
        "-r",
        f"/lab_fsd/future_risk:={scope}/lab_fsd/future_risk",
        "-r",
        f"/lab_fsd/input_status:={scope}/lab_fsd/input_status",
    ]
    env = os.environ.copy()
    env["ROS_LOCALHOST_ONLY"] = "1"
    env.pop("CYCLONEDDS_URI", None)
    probe: RosProbe | None = None
    process: subprocess.Popen[Any] | None = None
    checks: list[dict[str, Any]] = []
    goals: list[dict[str, Any]] = []
    error = ""

    with log_path.open("w", encoding="utf-8") as log_handle:
        try:
            process = subprocess.Popen(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                start_new_session=True,
            )
            probe = RosProbe(scope)
            ready = probe.wait_for_server(10.0)
            add_check(checks, "dispatch_action_server_ready", ready, f"pid={process.pid}")
            if not ready:
                raise RuntimeError("isolated dispatch action server did not become ready")
            device_fds = forbidden_device_fds(process.pid)
            literal_publishers = probe.literal_cmd_vel_publishers()
            add_check(
                checks,
                "dispatch_process_no_serial_device_fd",
                not device_fds,
                f"forbidden_fds={device_fds}",
            )
            add_check(
                checks,
                "literal_cmd_vel_has_no_dispatch_publisher",
                not literal_publishers,
                f"publishers={literal_publishers}",
            )

            fetch = probe.send_goal(
                task_id="stub-fetch-integration",
                task_type="fetch_sample",
                timeout_s=25.0,
            )
            goals.append(fetch)
            stages = [item["stage"] for item in fetch.get("feedback", [])]
            progress = [item["progress_pct"] for item in fetch.get("feedback", [])]
            add_check(checks, "stub_fetch_goal_accepted", fetch.get("accepted") is True, str(fetch))
            add_check(
                checks,
                "stub_fetch_simulated_only_result",
                fetch.get("success") is True
                and str(fetch.get("message", "")).startswith("SIMULATED_ONLY:"),
                str(fetch.get("message", "")),
            )
            add_check(
                checks,
                "stub_fetch_stage_sequence",
                stages == EXPECTED_FETCH_STAGES,
                f"actual={stages} expected={EXPECTED_FETCH_STAGES}",
            )
            add_check(
                checks,
                "stub_fetch_progress_monotonic",
                bool(progress)
                and all(a <= b for a, b in zip(progress, progress[1:]))
                and progress[-1] == 100.0,
                f"progress={progress}",
            )
            add_check(
                checks,
                "stub_fetch_no_f407_service_calls",
                sum(probe.service_calls.values()) == 0,
                json.dumps(probe.service_calls, sort_keys=True),
            )
            add_check(
                checks,
                "stub_fetch_no_cmd_vel_messages",
                len(probe.cmd_vel_messages) == 0
                and len(probe.literal_cmd_vel_messages) == 0,
                f"sink={probe.cmd_vel_messages} literal={probe.literal_cmd_vel_messages}",
            )

            timeout_goal = probe.send_goal(
                task_id="stub-timeout-safe-fail",
                task_type="fetch_sample",
                timeout_s=0.2,
            )
            goals.append(timeout_goal)
            add_check(
                checks,
                "stub_timeout_safe_fail_has_no_control_output",
                timeout_goal.get("accepted") is True
                and timeout_goal.get("success") is False
                and str(timeout_goal.get("message", "")).startswith("SAFE_FAIL:")
                and not probe.cmd_vel_messages
                and not probe.literal_cmd_vel_messages
                and sum(probe.service_calls.values()) == 0,
                str(timeout_goal),
            )

            concurrency = probe.probe_concurrent_rejection()
            add_check(
                checks,
                "dispatch_global_single_task_mutex",
                concurrency.get("owner_accepted") is True
                and concurrency.get("contender_accepted") is False
                and concurrency.get("owner_success") is False
                and concurrency.get("cancel_goal_count", 0) >= 1,
                str(concurrency),
            )

            probe.publish_estop(True)
            estop_goal = probe.send_goal(
                task_id="stub-estop-reject",
                task_type="home",
                timeout_s=5.0,
            )
            goals.append(estop_goal)
            add_check(
                checks,
                "stub_goal_rejected_by_f407_estop",
                estop_goal.get("accepted") is False,
                str(estop_goal),
            )

            probe.publish_estop(False)
            probe.publish_hard_guard("odom_offline")
            guard_goal = probe.send_goal(
                task_id="stub-lab-fsd-reject",
                task_type="home",
                timeout_s=5.0,
            )
            goals.append(guard_goal)
            add_check(
                checks,
                "stub_goal_ignores_navigation_only_lab_fsd_guard",
                guard_goal.get("accepted") is True
                and guard_goal.get("success") is True
                and str(guard_goal.get("message", "")).startswith("SIMULATED_ONLY:"),
                str(guard_goal),
            )
            add_check(
                checks,
                "rejection_paths_no_physical_output",
                sum(probe.service_calls.values()) == 0
                and len(probe.cmd_vel_messages) == 0
                and len(probe.literal_cmd_vel_messages) == 0,
                f"service_calls={probe.service_calls} sink={probe.cmd_vel_messages} literal={probe.literal_cmd_vel_messages}",
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if probe is not None:
                try:
                    probe.close()
                except Exception as exc:
                    if not error:
                        error = f"probe cleanup failed: {exc}"
            if process is not None:
                stop_process(process)

    try:
        import my_robot_agents.dispatch_server as dispatch_module

        module_path = Path(inspect.getfile(dispatch_module)).resolve()
    except Exception:
        module_path = Path()

    if error:
        add_check(checks, "integration_exception", False, error)
    overall = "PASS" if checks and all(item["status"] == "PASS" for item in checks) else "FAIL"
    report = {
        "schema_version": "xrd-dispatch-stub-integration-v1",
        "generated_at": utc_now(),
        "generated_at_unix": time.time(),
        "overall": overall,
        "simulation_only": True,
        "real_hardware_touched": False,
        "physical_runtime_audit_still_required": True,
        "ros_domain_id": args.domain_id,
        "ros_localhost_only": True,
        "scope": scope,
        "dispatch": {
            "executable": str(executable),
            "executable_sha256": sha256_file(executable),
            "module": str(module_path) if module_path else "",
            "module_sha256": sha256_file(module_path) if module_path.is_file() else "",
            "command": command,
        },
        "safety": {
            "dev_f407_opened": False,
            "nav2_enabled": False,
            "pickup_actuators_enabled": False,
            "cmd_vel_messages": probe.cmd_vel_messages if probe is not None else [],
            "literal_cmd_vel_messages": probe.literal_cmd_vel_messages if probe is not None else [],
            "f407_service_calls": probe.service_calls if probe is not None else {},
        },
        "goals": goals,
        "checks": checks,
        "server_log": str(log_path),
        "error": error,
    }
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    latest = out_path.parent / "latest.json"
    latest.write_text(out_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"report={out_path}", file=sys.stderr)
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
