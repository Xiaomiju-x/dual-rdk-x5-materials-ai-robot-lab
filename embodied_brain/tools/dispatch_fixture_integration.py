#!/usr/bin/env python3
"""Verify stationary pickup-fixture dispatch without touching real hardware.

The real installed dispatch_server runs in a private localhost-only ROS domain.
All F407 topics and services are remapped to a virtual fixture. The test proves
firmware-identity and estop rejection, then proves the explicitly enabled
stationary actuator sequence calls two lift targets and one magnet command
without publishing any vehicle velocity.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import signal
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


SCHEMA_VERSION = "xrd-dispatch-fixture-integration-v2"
MONITOR_POLICY_VERSION = "procfs-process-tree-fd-monitor-v1"
EXPECTED_STAGES = [1, 5, 6, 5, 8]
EXPECTED_PARAMETERS: dict[str, bool | float] = {
    "stub_mode": False,
    "use_nav2": False,
    "execute_pickup_actuators": True,
    "allow_stationary_pickup_fixture": True,
    "stationary_pickup_fixture_only": True,
    "stationary_pickup_fixture_one_shot": True,
    "use_lab_fsd_guard": True,
    "safe_stop_on_failure": True,
    "pickup_height_m": 0.02,
    "transport_height_m": 0.04,
    "actuator_service_wait_s": 2.0,
    "actuator_timeout_s": 4.0,
}
EXPECTED_CHECK_NAMES = {
    "dispatch_action_server_ready",
    "dispatch_process_tree_no_serial_device_fd",
    "literal_cmd_vel_has_no_dispatch_publisher",
    "invalid_firmware_identity_rejected",
    "estop_latched_rejected",
    "stationary_goal_accepted",
    "stationary_f407_reported_completion",
    "stationary_structured_result",
    "stationary_stage_sequence",
    "stationary_progress_monotonic",
    "virtual_f407_service_sequence",
    "stationary_no_cmd_vel_messages",
    "stationary_final_pose_unclaimed",
    "stationary_fixture_one_shot_rejected",
    "dds_callbacks_drained",
    "dispatch_monitor_coverage_complete",
    "dispatch_process_clean_exit",
    "dispatch_log_clean",
    "dispatch_artifacts_recorded",
}
FORBIDDEN_DEVICE_PREFIXES = (
    "/dev/F407",
    "/dev/ttyUSB",
    "/dev/ttyACM",
    "/dev/serial",
)
UNEXPECTED_LOG_PATTERNS = (
    re.compile(r"Traceback \(most recent call last\)"),
    re.compile(r"\bRCLError\b"),
    re.compile(r"\bExternalShutdownException\b"),
    re.compile(r"Exception in thread"),
    re.compile(r"Exception ignored in"),
    re.compile(r"uncaught exception", re.IGNORECASE),
    re.compile(r"Segmentation fault", re.IGNORECASE),
    re.compile(r"\[(?:ERROR|FATAL)\]", re.IGNORECASE),
    re.compile(r"(?:^|\s)(?:ERROR|FATAL)(?::|\s)", re.IGNORECASE),
)
EXPECTED_LOG_ERROR_MARKERS = (
    "rejecting real dispatch goal while F407 firmware identity is invalid",
    "rejecting dispatch goal while F407 estop is latched",
    "rejecting stationary pickup fixture: one-shot already consumed",
)


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


def locate_dispatch_module() -> Path:
    spec = importlib.util.find_spec("my_robot_agents.dispatch_server")
    if spec is None or not spec.origin:
        raise FileNotFoundError("installed my_robot_agents.dispatch_server module not found")
    path = Path(spec.origin).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"installed dispatch module is not a file: {path}")
    return path


def _signal_process_group(process: subprocess.Popen[Any], sig: signal.Signals) -> None:
    try:
        os.killpg(process.pid, sig)
    except (AttributeError, ProcessLookupError):
        if sig == signal.SIGINT:
            process.send_signal(sig)
        elif sig == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()


def stop_process(process: subprocess.Popen[Any]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "pid": process.pid,
        "shutdown_requested": False,
        "signals_sent": [],
        "forced_kill": False,
        "unexpected_early_exit": process.poll() is not None,
        "stop_requested_at_unix": 0.0,
        "exited_at_unix": 0.0,
        "returncode": process.poll(),
    }
    if process.poll() is None:
        report["shutdown_requested"] = True
        report["stop_requested_at_unix"] = time.time()
        _signal_process_group(process, signal.SIGINT)
        report["signals_sent"].append("SIGINT")
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            _signal_process_group(process, signal.SIGTERM)
            report["signals_sent"].append("SIGTERM")
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                _signal_process_group(process, signal.SIGKILL)
                report["signals_sent"].append("SIGKILL")
                report["forced_kill"] = True
                process.wait(timeout=3.0)
    report["returncode"] = process.poll()
    report["exited_at_unix"] = time.time()
    return report


def is_forbidden_device_target(target: str) -> bool:
    return any(target.startswith(prefix) for prefix in FORBIDDEN_DEVICE_PREFIXES)


def _proc_process_table() -> dict[int, tuple[int, int]]:
    table: dict[int, tuple[int, int]] = {}
    for item in Path("/proc").iterdir():
        if not item.name.isdigit():
            continue
        try:
            raw = (item / "stat").read_text(encoding="utf-8", errors="replace")
            fields = raw[raw.rfind(")") + 2 :].split()
            table[int(item.name)] = (int(fields[1]), int(fields[3]))
        except (IndexError, OSError, ValueError):
            continue
    return table


class ProcessTreeFdMonitor:
    def __init__(self, root_pid: int, process_started_at_unix: float, interval_s: float = 0.02) -> None:
        self.root_pid = root_pid
        self.process_started_at_unix = process_started_at_unix
        self.interval_s = interval_s
        self.started_at_unix = 0.0
        self.stopped_at_unix = 0.0
        self.first_sample_at_unix = 0.0
        self.last_sample_at_unix = 0.0
        self.sample_count = 0
        self.proc_tree_scan_count = 0
        self.fd_scan_count = 0
        self.root_alive_sample_count = 0
        self.pids_seen: set[int] = set()
        self.descendant_pids_seen: set[int] = set()
        self._known_pids: set[int] = {root_pid}
        self._observations: dict[tuple[int, str, str], dict[str, Any]] = {}
        self.thread_error = ""
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="dispatch-fixture-fd-monitor",
            daemon=True,
        )

    def _sample(self) -> None:
        sampled_at_unix = time.time()
        table = _proc_process_table()
        tree = {self.root_pid}
        tree.update(pid for pid, (_, session) in table.items() if session == self.root_pid)
        changed = True
        while changed:
            before = len(tree)
            tree.update(pid for pid, (ppid, _) in table.items() if ppid in tree)
            changed = len(tree) != before
        tree.update(pid for pid in self._known_pids if pid in table)
        active = sorted(pid for pid in tree if Path(f"/proc/{pid}").is_dir())
        self._known_pids.update(active)

        self.sample_count += 1
        self.proc_tree_scan_count += 1
        self.first_sample_at_unix = self.first_sample_at_unix or sampled_at_unix
        self.last_sample_at_unix = sampled_at_unix
        if self.root_pid in active:
            self.root_alive_sample_count += 1
        self.pids_seen.update(active)
        self.descendant_pids_seen.update(pid for pid in active if pid != self.root_pid)

        for pid in active:
            fd_root = Path(f"/proc/{pid}/fd")
            if not fd_root.is_dir():
                continue
            self.fd_scan_count += 1
            try:
                fd_items = list(fd_root.iterdir())
            except OSError:
                continue
            for fd_item in fd_items:
                try:
                    target = os.readlink(fd_item)
                except OSError:
                    continue
                if not is_forbidden_device_target(target):
                    continue
                key = (pid, fd_item.name, target)
                observation = self._observations.get(key)
                if observation is None:
                    observation = {
                        "pid": pid,
                        "fd": fd_item.name,
                        "target": target,
                        "first_observed_at_unix": sampled_at_unix,
                        "last_observed_at_unix": sampled_at_unix,
                        "sample_count": 0,
                    }
                    self._observations[key] = observation
                observation["last_observed_at_unix"] = sampled_at_unix
                observation["sample_count"] += 1

    def _run(self) -> None:
        try:
            while not self._stop_event.wait(self.interval_s):
                self._sample()
        except BaseException as exc:
            self.thread_error = f"{type(exc).__name__}: {exc}"

    def start(self) -> None:
        self.started_at_unix = time.time()
        self._sample()
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        if self._thread.is_alive() and not self.thread_error:
            self.thread_error = "monitor thread did not stop"
        try:
            self._sample()
        except BaseException as exc:
            if not self.thread_error:
                self.thread_error = f"final sample {type(exc).__name__}: {exc}"
        self.stopped_at_unix = time.time()

    def report(self) -> dict[str, Any]:
        observations = sorted(
            self._observations.values(),
            key=lambda item: (int(item["pid"]), str(item["fd"]), str(item["target"])),
        )
        return {
            "policy_version": MONITOR_POLICY_VERSION,
            "include_descendants": True,
            "forbidden_prefixes": list(FORBIDDEN_DEVICE_PREFIXES),
            "root_pid": self.root_pid,
            "process_started_at_unix": self.process_started_at_unix,
            "started_at_unix": self.started_at_unix,
            "stopped_at_unix": self.stopped_at_unix,
            "first_sample_at_unix": self.first_sample_at_unix,
            "last_sample_at_unix": self.last_sample_at_unix,
            "interval_s": self.interval_s,
            "sample_count": self.sample_count,
            "proc_tree_scan_count": self.proc_tree_scan_count,
            "fd_scan_count": self.fd_scan_count,
            "root_alive_sample_count": self.root_alive_sample_count,
            "pids_seen": sorted(self.pids_seen),
            "descendant_pids_seen": sorted(self.descendant_pids_seen),
            "observation_count": len(observations),
            "observations": observations,
            "thread_error": self.thread_error,
        }


def analyze_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    matches: list[dict[str, Any]] = []
    expected_errors: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for pattern in UNEXPECTED_LOG_PATTERNS:
            if pattern.search(line):
                item = {
                    "line": line_number,
                    "pattern": pattern.pattern,
                    "text": line[:500],
                }
                if any(marker in line for marker in EXPECTED_LOG_ERROR_MARKERS):
                    expected_errors.append(item)
                else:
                    matches.append(item)
                break
    expected_marker_counts = {
        marker: text.count(marker) for marker in EXPECTED_LOG_ERROR_MARKERS
    }
    expected_markers_found = [
        marker for marker in EXPECTED_LOG_ERROR_MARKERS if expected_marker_counts[marker] > 0
    ]
    return {
        "clean": (
            path.is_file()
            and not matches
            and all(count == 1 for count in expected_marker_counts.values())
        ),
        "unexpected_errors": matches,
        "expected_errors": expected_errors,
        "expected_error_markers": list(EXPECTED_LOG_ERROR_MARKERS),
        "expected_error_markers_found": expected_markers_found,
        "expected_error_marker_counts": expected_marker_counts,
        "patterns": [pattern.pattern for pattern in UNEXPECTED_LOG_PATTERNS],
    }


def compact_stages(stages: list[int]) -> list[int]:
    compact: list[int] = []
    for stage in stages:
        if not compact or compact[-1] != stage:
            compact.append(stage)
    return compact


class FixtureProbe:
    def __init__(self, scope: str) -> None:
        import rclpy
        from geometry_msgs.msg import Twist
        from my_robot_msgs.action import DispatchTask
        from my_robot_msgs.msg import LiftStatus
        from my_robot_msgs.srv import SetElectromagnet, SetLiftHeight
        from rclpy.action import ActionClient
        from std_msgs.msg import Bool

        self.rclpy = rclpy
        self.DispatchTask = DispatchTask
        self.LiftStatus = LiftStatus
        self.Bool = Bool
        rclpy.init(args=None)
        self.node = rclpy.create_node("dispatch_fixture_integration")
        self.scope = scope.rstrip("/")
        self.action = ActionClient(self.node, DispatchTask, f"{self.scope}/dispatch_task")
        self.estop = False
        self.identity_valid = False
        self.height_m = 0.0
        self.target_height_m = 0.0
        self.magnet_on = False
        self.lift_targets: list[float] = []
        self.magnet_commands: list[bool] = []
        self.cmd_vel_messages: list[dict[str, float]] = []
        self.literal_cmd_vel_messages: list[dict[str, float]] = []

        self.estop_pub = self.node.create_publisher(
            Bool, f"{self.scope}/f407/estop_latched", 10
        )
        self.identity_pub = self.node.create_publisher(
            Bool, f"{self.scope}/f407/firmware_identity_valid", 10
        )
        self.lift_pub = self.node.create_publisher(
            LiftStatus, f"{self.scope}/lift_status", 10
        )
        self.node.create_subscription(
            Twist, f"{self.scope}/cmd_vel_sink", self._on_cmd_vel, 10
        )
        self.node.create_subscription(
            Twist, "/cmd_vel", self._on_literal_cmd_vel, 10
        )
        self.node.create_service(
            SetLiftHeight,
            f"{self.scope}/set_lift_height",
            self._on_set_lift_height,
        )
        self.node.create_service(
            SetElectromagnet,
            f"{self.scope}/set_electromagnet",
            self._on_set_electromagnet,
        )
        self.node.create_timer(0.05, self._publish_fixture_state)

    def _on_cmd_vel(self, msg: Any) -> None:
        self.cmd_vel_messages.append(
            {"linear_x": float(msg.linear.x), "angular_z": float(msg.angular.z)}
        )

    def _on_literal_cmd_vel(self, msg: Any) -> None:
        self.literal_cmd_vel_messages.append(
            {"linear_x": float(msg.linear.x), "angular_z": float(msg.angular.z)}
        )

    def _publish_fixture_state(self) -> None:
        estop = self.Bool()
        estop.data = self.estop
        self.estop_pub.publish(estop)
        identity = self.Bool()
        identity.data = self.identity_valid
        self.identity_pub.publish(identity)
        status = self.LiftStatus()
        status.header.stamp = self.node.get_clock().now().to_msg()
        status.height_m = self.height_m
        status.target_height_m = self.target_height_m
        status.velocity_mps = 0.0
        status.home_switch_triggered = False
        status.top_switch_triggered = False
        status.homed = True
        status.moving = False
        status.electromagnet_on = self.magnet_on
        self.lift_pub.publish(status)

    def _on_set_lift_height(self, request: Any, response: Any) -> Any:
        target = float(request.target_height_m)
        self.lift_targets.append(target)
        self.target_height_m = target
        self.height_m = target
        response.success = True
        response.reached_height_m = target
        response.message = "virtual F407 open-loop height report"
        return response

    def _on_set_electromagnet(self, request: Any, response: Any) -> Any:
        turn_on = bool(request.turn_on)
        self.magnet_commands.append(turn_on)
        self.magnet_on = turn_on
        response.success = True
        response.message = "virtual F407 output-state report"
        return response

    def spin_for(self, duration_s: float) -> None:
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            self.rclpy.spin_once(
                self.node,
                timeout_sec=max(0.0, min(0.05, deadline - time.monotonic())),
            )

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

    def publish_state(self, *, identity_valid: bool, estop: bool) -> None:
        self.identity_valid = bool(identity_valid)
        self.estop = bool(estop)
        self.spin_for(0.35)

    def send_goal(self, task_id: str, timeout_s: float = 12.0) -> dict[str, Any]:
        goal = self.DispatchTask.Goal()
        goal.task_id = task_id
        goal.task_type = "pickup_fixture_stationary"
        goal.bottle_id = "virtual_fixture_bottle"
        goal.from_location = ""
        goal.to_location = ""
        goal.priority = self.DispatchTask.Goal.PRIORITY_NORMAL
        goal.timeout_s = timeout_s
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
        future = self.action.send_goal_async(goal, feedback_callback=on_feedback)
        if not self.wait_for(future.done, 5.0):
            raise TimeoutError(f"{task_id}: goal response timeout")
        handle = future.result()
        record: dict[str, Any] = {
            "task_id": task_id,
            "task_type": goal.task_type,
            "accepted": bool(handle.accepted),
            "feedback": feedback,
        }
        if not handle.accepted:
            record["elapsed_s"] = round(time.monotonic() - started, 3)
            return record
        result_future = handle.get_result_async()
        if not self.wait_for(result_future.done, timeout_s + 5.0):
            raise TimeoutError(f"{task_id}: result timeout")
        wrapped = result_future.result()
        result = wrapped.result
        record.update(
            {
                "status": int(wrapped.status),
                "success": bool(result.success),
                "message": str(result.message),
                "completion_class": str(result.completion_class),
                "actuator_sequence_completed": bool(result.actuator_sequence_completed),
                "physical_completed": bool(result.physical_completed),
                "physical_confirmation": str(result.physical_confirmation),
                "base_motion_requested": bool(result.base_motion_requested),
                "server_elapsed_s": float(result.elapsed_s),
                "elapsed_s": round(time.monotonic() - started, 3),
                "final_pose": {
                    "x": float(result.final_pose.position.x),
                    "y": float(result.final_pose.position.y),
                    "z": float(result.final_pose.position.z),
                    "orientation_x": float(result.final_pose.orientation.x),
                    "orientation_y": float(result.final_pose.orientation.y),
                    "orientation_z": float(result.final_pose.orientation.z),
                    "orientation_w": float(result.final_pose.orientation.w),
                },
            }
        )
        return record

    def literal_cmd_vel_publishers(self) -> list[str]:
        return sorted(
            {
                info.node_namespace.rstrip("/") + "/" + info.node_name
                for info in self.node.get_publishers_info_by_topic("/cmd_vel")
            }
        )

    def close(self) -> None:
        self.action.destroy()
        self.node.destroy_node()
        self.rclpy.shutdown()


def add_check(checks: list[dict[str, Any]], name: str, ok: bool, detail: str) -> None:
    checks.append({"name": name, "status": "PASS" if ok else "FAIL", "detail": detail})


def fixture_remaps(scope: str) -> dict[str, str]:
    return {
        "__ns": scope,
        "/cmd_vel": f"{scope}/cmd_vel_sink",
        "/set_lift_height": f"{scope}/set_lift_height",
        "/set_electromagnet": f"{scope}/set_electromagnet",
        "/lift_status": f"{scope}/lift_status",
        "/f407/estop_latched": f"{scope}/f407/estop_latched",
        "/f407/firmware_identity_valid": f"{scope}/f407/firmware_identity_valid",
        "/lab_fsd/safety_gate": f"{scope}/lab_fsd/safety_gate",
        "/lab_fsd/future_risk": f"{scope}/lab_fsd/future_risk",
        "/lab_fsd/input_status": f"{scope}/lab_fsd/input_status",
    }


def ros_parameter_value(value: bool | float) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def fixture_command(executable: Path, parameters: dict[str, bool | float], remaps: dict[str, str]) -> list[str]:
    command = [str(executable), "--ros-args"]
    for name, value in parameters.items():
        command.extend(["-p", f"{name}:={ros_parameter_value(value)}"])
    for source, target in remaps.items():
        command.extend(["-r", f"{source}:={target}"])
    return command


def process_exit_is_clean(process_report: dict[str, Any]) -> bool:
    return bool(
        process_report.get("shutdown_requested") is True
        and process_report.get("unexpected_early_exit") is False
        and process_report.get("forced_kill") is False
        and process_report.get("signals_sent") == ["SIGINT"]
        and process_report.get("returncode") in (0, -int(signal.SIGINT), -int(signal.SIGTERM))
    )


def monitor_coverage_is_complete(
    monitor_report: dict[str, Any], process_report: dict[str, Any]
) -> bool:
    process_started = float(monitor_report.get("process_started_at_unix") or 0.0)
    monitor_started = float(monitor_report.get("started_at_unix") or 0.0)
    first_sample = float(monitor_report.get("first_sample_at_unix") or 0.0)
    monitor_stopped = float(monitor_report.get("stopped_at_unix") or 0.0)
    last_sample = float(monitor_report.get("last_sample_at_unix") or 0.0)
    process_exited = float(process_report.get("exited_at_unix") or 0.0)
    sample_count = int(monitor_report.get("sample_count") or 0)
    return bool(
        monitor_report.get("policy_version") == MONITOR_POLICY_VERSION
        and monitor_report.get("include_descendants") is True
        and monitor_report.get("forbidden_prefixes") == list(FORBIDDEN_DEVICE_PREFIXES)
        and monitor_report.get("thread_error") == ""
        and monitor_report.get("root_pid") == process_report.get("pid")
        and process_report.get("pid") in (monitor_report.get("pids_seen") or [])
        and process_started > 0.0
        and process_started <= monitor_started <= process_started + 0.2
        and monitor_started <= first_sample <= process_started + 0.2
        and process_exited > 0.0
        and last_sample >= process_exited
        and monitor_stopped >= last_sample
        and sample_count >= 2
        and int(monitor_report.get("proc_tree_scan_count") or 0) == sample_count
        and int(monitor_report.get("root_alive_sample_count") or 0) >= 1
        and int(monitor_report.get("fd_scan_count") or 0) >= 1
        and 0.0 < float(monitor_report.get("interval_s") or 0.0) <= 0.05
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain-id", type=int, default=120 + (os.getpid() % 113))
    parser.add_argument("--dispatch-executable", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    if not 120 <= args.domain_id <= 232:
        parser.error("--domain-id must be in the private fixture range [120, 232]")

    started_at_unix = time.time()
    os.environ["ROS_DOMAIN_ID"] = str(args.domain_id)
    os.environ["ROS_LOCALHOST_ONLY"] = "1"
    os.environ.pop("CYCLONEDDS_URI", None)
    executable = locate_dispatch_executable(args.dispatch_executable)
    dispatch_module = locate_dispatch_module()
    executable_sha256 = sha256_file(executable)
    dispatch_module_sha256 = sha256_file(dispatch_module)
    scope = f"/dispatch_fixture_{os.getpid()}_{int(time.time())}"
    out_path = (
        Path(args.out).expanduser().resolve()
        if args.out
        else Path.home()
        / "dispatch_fixture_evidence"
        / f"dispatch_fixture_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_path.with_suffix(".log")
    parameters = dict(EXPECTED_PARAMETERS)
    remaps = fixture_remaps(scope)
    command = fixture_command(executable, parameters, remaps)
    env = os.environ.copy()
    env["ROS_LOCALHOST_ONLY"] = "1"
    env.pop("CYCLONEDDS_URI", None)
    process: subprocess.Popen[Any] | None = None
    monitor: ProcessTreeFdMonitor | None = None
    probe: FixtureProbe | None = None
    checks: list[dict[str, Any]] = []
    goals: list[dict[str, Any]] = []
    error = ""
    completed: dict[str, Any] = {}
    dds_drain: dict[str, Any] = {
        "requested_s": 0.75,
        "elapsed_s": 0.0,
        "completed": False,
        "callback_counts_before": {},
        "callback_counts_after": {},
    }
    process_report: dict[str, Any] = {
        "pid": 0,
        "started_at_unix": 0.0,
        "shutdown_requested": False,
        "signals_sent": [],
        "forced_kill": False,
        "unexpected_early_exit": False,
        "stop_requested_at_unix": 0.0,
        "exited_at_unix": 0.0,
        "returncode": None,
    }

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
            process_started_at_unix = time.time()
            process_report.update(
                {"pid": process.pid, "started_at_unix": process_started_at_unix}
            )
            monitor = ProcessTreeFdMonitor(process.pid, process_started_at_unix)
            monitor.start()
            probe = FixtureProbe(scope)
            ready = probe.wait_for_server(10.0)
            add_check(checks, "dispatch_action_server_ready", ready, f"pid={process.pid}")
            if not ready:
                raise RuntimeError("fixture dispatch action server did not become ready")
            if process.poll() is not None:
                raise RuntimeError(f"dispatch process exited during startup rc={process.returncode}")

            literal_publishers = probe.literal_cmd_vel_publishers()
            add_check(
                checks,
                "literal_cmd_vel_has_no_dispatch_publisher",
                not literal_publishers,
                f"publishers={literal_publishers}",
            )

            probe.publish_state(identity_valid=False, estop=False)
            invalid_identity = probe.send_goal("fixture-invalid-identity")
            goals.append(invalid_identity)
            add_check(
                checks,
                "invalid_firmware_identity_rejected",
                invalid_identity.get("accepted") is False,
                str(invalid_identity),
            )
            if process.poll() is not None:
                raise RuntimeError(f"dispatch process exited after identity rejection rc={process.returncode}")

            probe.publish_state(identity_valid=True, estop=True)
            estop_rejected = probe.send_goal("fixture-estop-latched")
            goals.append(estop_rejected)
            add_check(
                checks,
                "estop_latched_rejected",
                estop_rejected.get("accepted") is False,
                str(estop_rejected),
            )
            if process.poll() is not None:
                raise RuntimeError(f"dispatch process exited after estop rejection rc={process.returncode}")

            probe.publish_state(identity_valid=True, estop=False)
            completed = probe.send_goal("fixture-stationary-complete")
            goals.append(completed)
            before_drain = {
                "remapped_cmd_vel": len(probe.cmd_vel_messages),
                "literal_cmd_vel": len(probe.literal_cmd_vel_messages),
                "feedback": len(completed.get("feedback", [])),
            }
            drain_started = time.monotonic()
            probe.spin_for(float(dds_drain["requested_s"]))
            drain_elapsed = time.monotonic() - drain_started
            after_drain = {
                "remapped_cmd_vel": len(probe.cmd_vel_messages),
                "literal_cmd_vel": len(probe.literal_cmd_vel_messages),
                "feedback": len(completed.get("feedback", [])),
            }
            dds_drain.update(
                {
                    "elapsed_s": round(drain_elapsed, 6),
                    "completed": drain_elapsed >= float(dds_drain["requested_s"]) * 0.9,
                    "callback_counts_before": before_drain,
                    "callback_counts_after": after_drain,
                }
            )
            add_check(
                checks,
                "dds_callbacks_drained",
                dds_drain["completed"] is True,
                str(dds_drain),
            )
            if process.poll() is not None:
                raise RuntimeError(f"dispatch process exited after fixture result rc={process.returncode}")

            raw_stages = [item["stage"] for item in completed.get("feedback", [])]
            stages = compact_stages(raw_stages)
            progress = [item["progress_pct"] for item in completed.get("feedback", [])]
            message = str(completed.get("message") or "")
            final_pose = completed.get("final_pose") or {}
            add_check(checks, "stationary_goal_accepted", completed.get("accepted") is True, str(completed))
            add_check(
                checks,
                "stationary_f407_reported_completion",
                completed.get("success") is True
                and message.startswith("F407_REPORTED_COMPLETED:")
                and "stationary_fixture=true" in message
                and "dispatch_issued_base_motion=false" in message
                and "physical_completed=false" in message,
                message,
            )
            add_check(
                checks,
                "stationary_structured_result",
                completed.get("completion_class") == "f407_reported"
                and completed.get("actuator_sequence_completed") is True
                and completed.get("physical_completed") is False
                and completed.get("physical_confirmation") == ""
                and completed.get("base_motion_requested") is False,
                (
                    f"completion_class={completed.get('completion_class')} "
                    f"actuator_sequence_completed={completed.get('actuator_sequence_completed')} "
                    f"physical_completed={completed.get('physical_completed')} "
                    f"physical_confirmation={completed.get('physical_confirmation')!r} "
                    f"base_motion_requested={completed.get('base_motion_requested')}"
                ),
            )
            add_check(
                checks,
                "stationary_stage_sequence",
                stages == EXPECTED_STAGES,
                f"raw={raw_stages} compact={stages} expected={EXPECTED_STAGES}",
            )
            add_check(
                checks,
                "stationary_progress_monotonic",
                bool(progress)
                and all(0.0 <= value <= 100.0 for value in progress)
                and all(a <= b for a, b in zip(progress, progress[1:]))
                and progress[-1] == 100.0,
                f"progress={progress}",
            )
            add_check(
                checks,
                "virtual_f407_service_sequence",
                len(probe.lift_targets) == 2
                and all(
                    abs(actual - expected) <= 1e-6
                    for actual, expected in zip(probe.lift_targets, [0.02, 0.04])
                )
                and probe.magnet_commands == [True],
                f"lift={probe.lift_targets} magnet={probe.magnet_commands}",
            )
            add_check(
                checks,
                "stationary_no_cmd_vel_messages",
                not probe.cmd_vel_messages and not probe.literal_cmd_vel_messages,
                f"sink={probe.cmd_vel_messages} literal={probe.literal_cmd_vel_messages}",
            )
            add_check(
                checks,
                "stationary_final_pose_unclaimed",
                final_pose
                == {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "orientation_x": 0.0,
                    "orientation_y": 0.0,
                    "orientation_z": 0.0,
                    "orientation_w": 1.0,
                },
                str(final_pose),
            )

            one_shot_rejected = probe.send_goal("fixture-one-shot-consumed")
            goals.append(one_shot_rejected)
            add_check(
                checks,
                "stationary_fixture_one_shot_rejected",
                one_shot_rejected.get("accepted") is False,
                str(one_shot_rejected),
            )
            if process.poll() is not None:
                raise RuntimeError(f"dispatch process exited after one-shot rejection rc={process.returncode}")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            add_check(checks, "integration_exception", False, error)
        finally:
            if probe is not None:
                try:
                    probe.close()
                except Exception as exc:
                    if not error:
                        error = f"probe close {type(exc).__name__}: {exc}"
                        add_check(checks, "integration_exception", False, error)
            if process is not None:
                process_report.update(stop_process(process))
            if monitor is not None:
                monitor.stop()

    monitor_report = monitor.report() if monitor is not None else {
        "policy_version": MONITOR_POLICY_VERSION,
        "include_descendants": True,
        "forbidden_prefixes": list(FORBIDDEN_DEVICE_PREFIXES),
        "observations": [],
        "observation_count": 0,
        "thread_error": "monitor did not start",
    }
    observations = monitor_report.get("observations") or []
    real_hardware_touched = bool(observations)
    # Any forbidden serial alias is conservatively treated as possible F407 access.
    dev_f407_opened = bool(observations)
    log_analysis = analyze_log(log_path)
    log_sha256 = sha256_file(log_path) if log_path.is_file() else ""
    log_size_bytes = log_path.stat().st_size if log_path.is_file() else 0
    add_check(
        checks,
        "dispatch_process_tree_no_serial_device_fd",
        not observations and not real_hardware_touched and not dev_f407_opened,
        f"observations={observations}",
    )
    add_check(
        checks,
        "dispatch_monitor_coverage_complete",
        monitor_coverage_is_complete(monitor_report, process_report),
        str(monitor_report),
    )
    add_check(
        checks,
        "dispatch_process_clean_exit",
        process_exit_is_clean(process_report),
        str(process_report),
    )
    add_check(
        checks,
        "dispatch_log_clean",
        log_analysis.get("clean") is True,
        str(log_analysis),
    )
    add_check(
        checks,
        "dispatch_artifacts_recorded",
        executable.is_file()
        and dispatch_module.is_file()
        and sha256_file(executable) == executable_sha256
        and sha256_file(dispatch_module) == dispatch_module_sha256,
        f"executable={executable} module={dispatch_module}",
    )

    check_names = [str(item.get("name") or "") for item in checks]
    check_contract_ok = len(check_names) == len(EXPECTED_CHECK_NAMES) and set(check_names) == EXPECTED_CHECK_NAMES
    overall = "PASS" if (
        check_contract_ok
        and all(item["status"] == "PASS" for item in checks)
        and not error
    ) else "FAIL"
    generated_at_unix = time.time()
    report = {
        "schema_version": SCHEMA_VERSION,
        "policy_version": "xrd-dispatch-fixture-policy-v2",
        "started_at": datetime.fromtimestamp(started_at_unix, timezone.utc).replace(
            microsecond=0
        ).isoformat().replace("+00:00", "Z"),
        "started_at_unix": started_at_unix,
        "generated_at": utc_now(),
        "generated_at_unix": generated_at_unix,
        "overall": overall,
        "simulation_only": True,
        "real_hardware_touched": real_hardware_touched,
        "physical_completed": completed.get("physical_completed") is True,
        "domain_id": args.domain_id,
        "ros_localhost_only": True,
        "dds_environment": {
            "ROS_DOMAIN_ID": str(args.domain_id),
            "ROS_LOCALHOST_ONLY": "1",
            "CYCLONEDDS_URI": None,
        },
        "dispatch_executable": executable.as_posix(),
        "dispatch_executable_sha256": executable_sha256,
        "dispatch_module": dispatch_module.as_posix(),
        "dispatch_module_sha256": dispatch_module_sha256,
        "dispatch_artifacts": {
            "executable": {
                "path": executable.as_posix(),
                "sha256": executable_sha256,
            },
            "module": {
                "path": dispatch_module.as_posix(),
                "sha256": dispatch_module_sha256,
            },
        },
        "scope": scope,
        "parameters": parameters,
        "remaps": remaps,
        "command": command,
        "monitor": monitor_report,
        "process": process_report,
        "dds_drain": dds_drain,
        "safety": {
            "dev_f407_opened": dev_f407_opened,
            "forbidden_device_opened": real_hardware_touched,
            "nav2_enabled": False,
            "literal_cmd_vel_messages": probe.literal_cmd_vel_messages if probe else [],
            "remapped_cmd_vel_messages": probe.cmd_vel_messages if probe else [],
            "lift_targets": probe.lift_targets if probe else [],
            "magnet_commands": probe.magnet_commands if probe else [],
        },
        "goals": goals,
        "checks": checks,
        "expected_check_names": sorted(EXPECTED_CHECK_NAMES),
        "check_contract_ok": check_contract_ok,
        "error": error,
        "log_path": log_path.as_posix(),
        "log_sha256": log_sha256,
        "log": {
            "path": log_path.as_posix(),
            "sha256": log_sha256,
            "size_bytes": log_size_bytes,
            **log_analysis,
        },
    }
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"DISPATCH_FIXTURE_INTEGRATION {overall}")
    print(f"report: {out_path}")
    print(f"log: {log_path}")
    print(f"checks: {len(checks)}")
    if error:
        print(f"error: {error}")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
