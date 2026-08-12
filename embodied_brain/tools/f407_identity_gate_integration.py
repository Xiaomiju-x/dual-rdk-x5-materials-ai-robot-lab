#!/usr/bin/env python3
"""Test the installed F407 ROS driver with an isolated PTY firmware.

This X5-only test never opens /dev/F407. It starts the real
serial_f407_node binary on a pseudo-terminal and a private ROS_DOMAIN_ID,
then proves the firmware identity gate for missing, valid, mismatched, and
stale identities.

The JSON output is simulation evidence. It cannot replace the post-flash
physical interlock report produced by f407_link_test.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pty
import select
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable


HEADER = b"\xAA\x55"
DN_CMD_VEL = 0x01
DN_SET_LIFT_HEIGHT = 0x02
DN_SET_MAGNET = 0x03
DN_CLEAR_ESTOP = 0x11
UP_EXT_TELEMETRY = 0x02
UP_SAFETY_STATE = 0x03
UP_FIRMWARE_INFO = 0x04

EXPECTED_PROTOCOL_VERSION = 2
EXPECTED_CAPABILITIES = 0x003F
EXPECTED_BUILD_ID = 2026071907
EXPECTED_TEST_MODE = 0
EXPECTED_HW_VARIANT = 1


def build_frame(frame_type: int, payload: bytes = b"") -> bytes:
    body = HEADER + bytes((frame_type & 0xFF, len(payload) & 0xFF)) + payload
    return body + bytes((sum(body) & 0xFF,))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate_serial_node(explicit: str) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
    else:
        prefix = subprocess.check_output(
            ["ros2", "pkg", "prefix", "my_robot_drivers"], text=True
        ).strip()
        path = (Path(prefix) / "lib" / "my_robot_drivers" / "serial_f407_node").resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise FileNotFoundError(f"serial_f407_node not executable: {path}")
    return path


class PtyFirmware:
    def __init__(self, master_fd: int) -> None:
        self.master_fd = master_fd
        self.stop_event = threading.Event()
        self.mode_lock = threading.Lock()
        self.tx_lock = threading.Lock()
        self.mode = "none"
        self.frames_lock = threading.Lock()
        self.frames: list[dict[str, Any]] = []
        self.thread = threading.Thread(target=self._run, name="pty-f407", daemon=True)

    def start(self) -> None:
        os.set_blocking(self.master_fd, False)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)

    def set_mode(self, mode: str) -> None:
        if mode not in {"none", "valid", "mismatch"}:
            raise ValueError(mode)
        with self.mode_lock:
            self.mode = mode

    def snapshot(self) -> int:
        with self.frames_lock:
            return len(self.frames)

    def frames_since(self, index: int) -> list[dict[str, Any]]:
        with self.frames_lock:
            return [dict(item) for item in self.frames[index:]]

    def all_frames(self) -> list[dict[str, Any]]:
        return self.frames_since(0)

    def _send_identity(self, mismatch: bool) -> None:
        build_id = EXPECTED_BUILD_ID + (1 if mismatch else 0)
        identity = struct.pack(
            "<HHIBBH",
            EXPECTED_PROTOCOL_VERSION,
            EXPECTED_CAPABILITIES,
            build_id,
            EXPECTED_TEST_MODE,
            EXPECTED_HW_VARIANT,
            0,
        )
        safety = struct.pack("<BBH", 0, 0, 0)
        with self.tx_lock:
            os.write(self.master_fd, build_frame(UP_FIRMWARE_INFO, identity))
            os.write(self.master_fd, build_frame(UP_SAFETY_STATE, safety))

    def send_lift_telemetry(
        self, height_m: float, velocity_mps: float = 0.0, accel_z: float = 9.81
    ) -> None:
        telemetry = struct.pack(
            "<ffBBBBffffffff",
            float(height_m),
            float(velocity_mps),
            0,
            0,
            0,
            1,
            0.0,
            0.0,
            float(accel_z),
            0.0,
            0.0,
            0.0,
            35.0,
            12.0,
        )
        with self.tx_lock:
            os.write(self.master_fd, build_frame(UP_EXT_TELEMETRY, telemetry))

    def _consume_frames(self, buffer: bytearray) -> None:
        while True:
            header_at = buffer.find(HEADER)
            if header_at < 0:
                if len(buffer) > 1:
                    del buffer[:-1]
                return
            if header_at:
                del buffer[:header_at]
            if len(buffer) < 5:
                return
            payload_len = buffer[3]
            frame_len = 5 + payload_len
            if len(buffer) < frame_len:
                return
            raw = bytes(buffer[:frame_len])
            del buffer[:frame_len]
            if (sum(raw[:-1]) & 0xFF) != raw[-1]:
                continue
            payload = raw[4:-1]
            item: dict[str, Any] = {
                "t_monotonic": time.monotonic(),
                "type": raw[2],
                "payload_hex": payload.hex(),
            }
            if raw[2] == DN_CMD_VEL and len(payload) == 8:
                linear, angular = struct.unpack("<ff", payload)
                item["linear"] = float(linear)
                item["angular"] = float(angular)
                item["nonzero"] = abs(linear) > 1e-6 or abs(angular) > 1e-6
            with self.frames_lock:
                self.frames.append(item)

    def _run(self) -> None:
        buffer = bytearray()
        next_identity = 0.0
        while not self.stop_event.is_set():
            now = time.monotonic()
            with self.mode_lock:
                mode = self.mode
            if mode != "none" and now >= next_identity:
                try:
                    self._send_identity(mode == "mismatch")
                except OSError:
                    return
                next_identity = now + 0.20
            ready, _, _ = select.select([self.master_fd], [], [], 0.02)
            if not ready:
                continue
            try:
                chunk = os.read(self.master_fd, 4096)
            except BlockingIOError:
                continue
            except OSError:
                if self.stop_event.is_set():
                    return
                time.sleep(0.02)
                continue
            if chunk:
                buffer.extend(chunk)
                self._consume_frames(buffer)


class RosProbe:
    def __init__(self) -> None:
        import rclpy
        from geometry_msgs.msg import Twist
        from my_robot_msgs.msg import LiftStatus
        from my_robot_msgs.srv import SetElectromagnet, SetLiftHeight
        from sensor_msgs.msg import Imu
        from std_msgs.msg import Bool, String
        from std_srvs.srv import Trigger

        self.rclpy = rclpy
        self.Twist = Twist
        self.SetElectromagnet = SetElectromagnet
        self.SetLiftHeight = SetLiftHeight
        self.Trigger = Trigger
        rclpy.init(args=None)
        self.node = rclpy.create_node("f407_identity_gate_integration")
        self.latest_identity: bool | None = None
        self.latest_info: dict[str, Any] = {}
        self.latest_lift_height: float | None = None
        self.latest_imu_valid: bool | None = None
        self.raw_imu_count = 0
        self.filtered_imu_count = 0
        self.identity_history: list[dict[str, Any]] = []
        self.cmd_pub = self.node.create_publisher(Twist, "/cmd_vel", 10)
        self.node.create_subscription(
            Bool, "/f407/firmware_identity_valid", self._on_identity, 10
        )
        self.node.create_subscription(String, "/f407/firmware_info", self._on_info, 10)
        self.node.create_subscription(LiftStatus, "/lift_status", self._on_lift_status, 10)
        self.node.create_subscription(Bool, "/f407/imu_valid", self._on_imu_valid, 10)
        self.node.create_subscription(Imu, "/imu/raw", self._on_raw_imu, 10)
        self.node.create_subscription(Imu, "/imu", self._on_filtered_imu, 10)
        self.magnet_client = self.node.create_client(
            SetElectromagnet, "/set_electromagnet"
        )
        self.lift_client = self.node.create_client(SetLiftHeight, "/set_lift_height")
        self.clear_client = self.node.create_client(Trigger, "/clear_estop")

    def _on_identity(self, msg: Any) -> None:
        self.latest_identity = bool(msg.data)
        self.identity_history.append(
            {"t_monotonic": time.monotonic(), "valid": bool(msg.data)}
        )

    def _on_info(self, msg: Any) -> None:
        try:
            self.latest_info = json.loads(msg.data)
        except (TypeError, json.JSONDecodeError):
            self.latest_info = {"raw": str(msg.data)}

    def _on_lift_status(self, msg: Any) -> None:
        self.latest_lift_height = float(msg.height_m)

    def _on_imu_valid(self, msg: Any) -> None:
        self.latest_imu_valid = bool(msg.data)

    def _on_raw_imu(self, _msg: Any) -> None:
        self.raw_imu_count += 1

    def _on_filtered_imu(self, _msg: Any) -> None:
        self.filtered_imu_count += 1

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

    def publish_cmd(self, linear: float, angular: float, repeats: int = 3) -> None:
        msg = self.Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        for _ in range(repeats):
            self.cmd_pub.publish(msg)
            self.spin_for(0.10)

    def call_magnet(self, turn_on: bool, timeout_s: float = 2.0) -> Any:
        if not self.magnet_client.wait_for_service(timeout_sec=timeout_s):
            raise TimeoutError("/set_electromagnet unavailable")
        request = self.SetElectromagnet.Request()
        request.turn_on = bool(turn_on)
        future = self.magnet_client.call_async(request)
        if not self.wait_for(future.done, timeout_s):
            raise TimeoutError("/set_electromagnet response timeout")
        return future.result()

    def call_lift(
        self,
        target_height_m: float,
        arrival_timeout_s: float,
        call_timeout_s: float = 2.0,
    ) -> Any:
        if not self.lift_client.wait_for_service(timeout_sec=call_timeout_s):
            raise TimeoutError("/set_lift_height unavailable")
        request = self.SetLiftHeight.Request()
        request.target_height_m = float(target_height_m)
        request.timeout_s = float(arrival_timeout_s)
        request.wait_for_arrival = True
        future = self.lift_client.call_async(request)
        if not self.wait_for(future.done, call_timeout_s):
            raise TimeoutError("/set_lift_height response timeout")
        return future.result()

    def call_clear(self, timeout_s: float = 2.0) -> Any:
        if not self.clear_client.wait_for_service(timeout_sec=timeout_s):
            raise TimeoutError("/clear_estop unavailable")
        future = self.clear_client.call_async(self.Trigger.Request())
        if not self.wait_for(future.done, timeout_s):
            raise TimeoutError("/clear_estop response timeout")
        return future.result()

    def close(self) -> None:
        self.node.destroy_node()
        self.rclpy.shutdown()

    def info_matches_expected(self, *, valid: bool, build_id: int = EXPECTED_BUILD_ID) -> bool:
        return (
            self.latest_info.get("protocol_version") == EXPECTED_PROTOCOL_VERSION
            and self.latest_info.get("capabilities") == EXPECTED_CAPABILITIES
            and self.latest_info.get("build_id") == build_id
            and self.latest_info.get("test_mode") == EXPECTED_TEST_MODE
            and self.latest_info.get("hw_variant") == EXPECTED_HW_VARIANT
            and self.latest_info.get("identity_valid") is valid
            and self.latest_info.get("required") is True
            and self.latest_info.get("identity_enforcement_enabled") is True
            and self.latest_info.get("cmd_vel_authority_when_invalid") is False
        )


def cmd_vel_frames(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in frames if item.get("type") == DN_CMD_VEL]


def has_nonzero_cmd(frames: list[dict[str, Any]]) -> bool:
    return any(bool(item.get("nonzero")) for item in cmd_vel_frames(frames))


def wait_frames(
    probe: RosProbe,
    simulator: PtyFirmware,
    start_index: int,
    predicate: Callable[[list[dict[str, Any]]], bool],
    timeout_s: float = 1.5,
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        probe.spin_for(0.05)
        frames = simulator.frames_since(start_index)
        if predicate(frames):
            return frames
    return simulator.frames_since(start_index)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-id", type=int, default=93)
    parser.add_argument("--identity-stale-s", type=float, default=1.2)
    parser.add_argument("--serial-node", default="")
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    if not 1 <= args.domain_id <= 232:
        parser.error("--domain-id must be in [1, 232]")
    stale_s = max(1.0, float(args.identity_stale_s))
    os.environ["ROS_DOMAIN_ID"] = str(args.domain_id)

    serial_node = locate_serial_node(args.serial_node)
    master_fd, slave_fd = pty.openpty()
    slave_path = os.ttyname(slave_fd)
    simulator = PtyFirmware(master_fd)
    simulator.start()

    log_file = tempfile.NamedTemporaryFile(
        prefix="f407_identity_gate_", suffix=".log", delete=False
    )
    log_path = Path(log_file.name)
    command = [
        str(serial_node),
        "--ros-args",
        "-r", "__node:=serial_f407_identity_sim",
        "-p", f"port_name:={slave_path}",
        "-p", "publish_tf:=false",
        "-p", "require_firmware_identity:=true",
        "-p", f"firmware_identity_stale_s:={stale_s}",
        "-p", "heartbeat_hz:=5.0",
        "-p", "cmd_vel_timeout_s:=0.6",
    ]
    process = subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=os.environ.copy(),
    )
    time.sleep(0.25)
    os.close(slave_fd)

    probe: RosProbe | None = None
    results: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        result = {"name": name, "status": "PASS" if ok else "FAIL", "detail": detail}
        results.append(result)
        print(f"[{result['status']}] {name}: {detail}")

    try:
        probe = RosProbe()
        check("serial_node_started", process.poll() is None, f"pid={process.pid} pty={slave_path}")
        check(
            "initial_identity_false",
            probe.wait_for(lambda: probe.latest_identity is False, 4.0),
            f"identity={probe.latest_identity}",
        )
        check(
            "identity_enforcement_authority_disclosed",
            probe.wait_for(
                lambda: probe.latest_info.get("required") is True
                and probe.latest_info.get("identity_enforcement_enabled") is True
                and probe.latest_info.get("cmd_vel_authority_when_invalid") is False,
                3.0,
            ),
            f"info={probe.latest_info}",
        )

        marker = simulator.snapshot()
        probe.publish_cmd(0.12, 0.20)
        frames = wait_frames(probe, simulator, marker, lambda value: bool(cmd_vel_frames(value)))
        check(
            "missing_identity_blocks_nonzero_cmd_vel",
            bool(cmd_vel_frames(frames)) and not has_nonzero_cmd(frames),
            f"cmd_frames={cmd_vel_frames(frames)}",
        )

        marker = simulator.snapshot()
        response = probe.call_magnet(True)
        frames = simulator.frames_since(marker)
        magnet_on_sent = any(
            item.get("type") == DN_SET_MAGNET and item.get("payload_hex") == "01"
            for item in frames
        )
        check(
            "missing_identity_blocks_magnet_on",
            response is not None and not response.success and not magnet_on_sent,
            f"success={getattr(response, 'success', None)} message={getattr(response, 'message', '')}",
        )

        simulator.set_mode("valid")
        accepted = probe.wait_for(
            lambda: probe.latest_identity is True
            and probe.info_matches_expected(valid=True),
            4.0,
        )
        check(
            "valid_identity_is_accepted",
            accepted,
            f"identity={probe.latest_identity} info={probe.latest_info}",
        )

        for _ in range(6):
            simulator.send_lift_telemetry(0.0, accel_z=0.0)
            probe.spin_for(0.03)
        check(
            "zero_imu_is_gated",
            probe.raw_imu_count >= 5
            and probe.filtered_imu_count == 0
            and probe.latest_imu_valid is False,
            f"raw={probe.raw_imu_count} filtered={probe.filtered_imu_count} "
            f"valid={probe.latest_imu_valid}",
        )
        for _ in range(6):
            simulator.send_lift_telemetry(0.0, accel_z=9.81)
            probe.spin_for(0.03)
        check(
            "plausible_imu_passes_gate",
            probe.filtered_imu_count >= 1 and probe.latest_imu_valid is True,
            f"raw={probe.raw_imu_count} filtered={probe.filtered_imu_count} "
            f"valid={probe.latest_imu_valid}",
        )

        stale_target = 0.12
        simulator.send_lift_telemetry(stale_target)
        check(
            "pre_command_lift_telemetry_cached",
            probe.wait_for(
                lambda: probe.latest_lift_height is not None
                and abs(probe.latest_lift_height - stale_target) < 1e-4,
                2.0,
            ),
            f"height={probe.latest_lift_height}",
        )
        marker = simulator.snapshot()
        response = probe.call_lift(stale_target, arrival_timeout_s=0.25)
        frames = simulator.frames_since(marker)
        lift_target_sent = any(
            item.get("type") == DN_SET_LIFT_HEIGHT for item in frames
        )
        check(
            "cached_lift_telemetry_cannot_satisfy_new_command",
            response is not None
            and not response.success
            and lift_target_sent
            and "no fresh post-command F407 lift telemetry" in response.message,
            f"success={getattr(response, 'success', None)} "
            f"message={getattr(response, 'message', '')} lift_frame={lift_target_sent}",
        )

        fresh_target = 0.16
        marker = simulator.snapshot()
        fresh_telemetry_sent = threading.Event()

        def send_telemetry_after_lift_command() -> None:
            deadline = time.monotonic() + 1.5
            while time.monotonic() < deadline:
                if any(
                    item.get("type") == DN_SET_LIFT_HEIGHT
                    for item in simulator.frames_since(marker)
                ):
                    simulator.send_lift_telemetry(fresh_target)
                    fresh_telemetry_sent.set()
                    return
                time.sleep(0.01)

        responder = threading.Thread(
            target=send_telemetry_after_lift_command,
            name="fresh-lift-telemetry",
            daemon=True,
        )
        responder.start()
        response = probe.call_lift(
            fresh_target, arrival_timeout_s=0.8, call_timeout_s=2.0
        )
        responder.join(timeout=2.0)
        check(
            "fresh_post_command_lift_telemetry_allows_arrival",
            response is not None
            and response.success
            and fresh_telemetry_sent.is_set()
            and "telemetry_seq=" in response.message,
            f"success={getattr(response, 'success', None)} "
            f"message={getattr(response, 'message', '')} "
            f"fresh_sent={fresh_telemetry_sent.is_set()}",
        )

        marker = simulator.snapshot()
        probe.publish_cmd(0.12, 0.20)
        frames = wait_frames(probe, simulator, marker, has_nonzero_cmd)
        check(
            "valid_identity_allows_bounded_cmd_vel",
            has_nonzero_cmd(frames),
            f"cmd_frames={cmd_vel_frames(frames)}",
        )

        marker = simulator.snapshot()
        response = probe.call_magnet(True)
        frames = wait_frames(
            probe,
            simulator,
            marker,
            lambda value: any(item.get("type") == DN_SET_MAGNET for item in value),
        )
        magnet_on_sent = any(
            item.get("type") == DN_SET_MAGNET and item.get("payload_hex") == "01"
            for item in frames
        )
        check(
            "valid_identity_allows_magnet_service",
            response is not None and response.success and magnet_on_sent,
            f"success={getattr(response, 'success', None)} magnet_frame={magnet_on_sent}",
        )

        simulator.set_mode("mismatch")
        check(
            "mismatched_build_revokes_identity",
            probe.wait_for(
                lambda: probe.latest_identity is False
                and probe.info_matches_expected(
                    valid=False, build_id=EXPECTED_BUILD_ID + 1
                ),
                3.0,
            ),
            f"identity={probe.latest_identity} info={probe.latest_info}",
        )
        marker = simulator.snapshot()
        probe.publish_cmd(0.12, 0.20)
        frames = wait_frames(probe, simulator, marker, lambda value: bool(cmd_vel_frames(value)))
        check(
            "mismatched_build_blocks_nonzero_cmd_vel",
            bool(cmd_vel_frames(frames)) and not has_nonzero_cmd(frames),
            f"cmd_frames={cmd_vel_frames(frames)}",
        )

        simulator.set_mode("valid")
        check(
            "identity_recovers_after_valid_frame",
            probe.wait_for(
                lambda: probe.latest_identity is True
                and probe.info_matches_expected(valid=True),
                3.0,
            ),
            f"identity={probe.latest_identity}",
        )
        simulator.set_mode("none")
        check(
            "stale_identity_is_revoked",
            probe.wait_for(
                lambda: probe.latest_identity is False
                and probe.info_matches_expected(valid=False)
                and isinstance(probe.latest_info.get("age_s"), (int, float))
                and float(probe.latest_info["age_s"]) > stale_s,
                stale_s + 3.0,
            ),
            f"stale_s={stale_s} identity={probe.latest_identity} info={probe.latest_info}",
        )
        marker = simulator.snapshot()
        probe.publish_cmd(0.12, 0.20)
        frames = wait_frames(probe, simulator, marker, lambda value: bool(cmd_vel_frames(value)))
        check(
            "stale_identity_blocks_nonzero_cmd_vel",
            bool(cmd_vel_frames(frames)) and not has_nonzero_cmd(frames),
            f"cmd_frames={cmd_vel_frames(frames)}",
        )

        marker = simulator.snapshot()
        response = probe.call_clear()
        frames = simulator.frames_since(marker)
        clear_sent = any(item.get("type") == DN_CLEAR_ESTOP for item in frames)
        check(
            "stale_identity_blocks_clear_estop",
            response is not None and not response.success and not clear_sent,
            f"success={getattr(response, 'success', None)} clear_frame={clear_sent}",
        )
    except Exception as exc:
        check("integration_exception", False, f"{type(exc).__name__}: {exc}")
    finally:
        if probe is not None:
            probe.close()
        simulator.stop()
        try:
            os.killpg(process.pid, signal.SIGINT)
            process.wait(timeout=4.0)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=2.0)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                process.kill()
        log_file.close()
        os.close(master_fd)

    try:
        log_tail = "\n".join(
            log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
        )
    finally:
        try:
            log_path.unlink()
        except OSError:
            pass

    all_frames = simulator.all_frames()
    failures = [item for item in results if item["status"] != "PASS"]
    report = {
        "schema": "xrd-f407-identity-gate-pty-integration-v1",
        "created_unix": time.time(),
        "overall": "PASS" if not failures else "FAIL",
        "evidence_class": "simulation_only",
        "real_hardware_touched": False,
        "real_serial_path_opened": False,
        "ros_domain_id": args.domain_id,
        "pty_slave": slave_path,
        "serial_node": str(serial_node),
        "serial_node_sha256": sha256_file(serial_node),
        "identity_stale_s": stale_s,
        "results": results,
        "frame_counts": dict(Counter(f"0x{item['type']:02X}" for item in all_frames)),
        "identity_history": probe.identity_history if probe is not None else [],
        "node_log_tail": log_tail,
        "physical_runtime_audit_still_required": True,
    }
    if args.report:
        report_path = Path(args.report).expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"REPORT {report_path}")
    print(f"F407_IDENTITY_GATE_PTY {report['overall']} checks={len(results)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
