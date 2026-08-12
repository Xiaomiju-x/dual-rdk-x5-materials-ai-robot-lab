#!/usr/bin/env python3
"""Dependency-free contract check for the F407/C++ 0xAA55 protocol."""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
C_HEADER = ROOT / "stm32_f407" / "App" / "proto.h"
C_MAIN_HEADER = ROOT / "stm32_f407" / "App" / "main.h"
C_MAIN_SOURCE = ROOT / "stm32_f407" / "App" / "main.c"
C_SOURCE = ROOT / "stm32_f407" / "App" / "proto.c"
C_LIFT_HEADER = ROOT / "stm32_f407" / "App" / "bsp_lift.h"
C_LIFT_SOURCE = ROOT / "stm32_f407" / "App" / "bsp_lift.c"
C_UART_HEADER = ROOT / "stm32_f407" / "App" / "bsp_uart.h"
C_UART_SOURCE = ROOT / "stm32_f407" / "App" / "bsp_uart.c"
C_IMU_SOURCE = ROOT / "stm32_f407" / "App" / "bsp_imu.c"
CPP_HEADER = (
    ROOT
    / "ros2_ws"
    / "src"
    / "my_robot_drivers"
    / "include"
    / "my_robot_drivers"
    / "serial_protocol.hpp"
)


EXPECTED_UP = {
    "BASIC_ODOM": 0x01,
    "EXT_TELEMETRY": 0x02,
    "SAFETY_STATE": 0x03,
    "FIRMWARE_INFO": 0x04,
    "ACK": 0x10,
    "ERROR": 0x1F,
}
EXPECTED_DOWN = {
    "CMD_VEL": 0x01,
    "SET_LIFT_HEIGHT": 0x02,
    "SET_ELECTROMAGNET": 0x03,
    "LIFT_HOME": 0x04,
    "EMERGENCY_STOP": 0x10,
    "CLEAR_ESTOP": 0x11,
    "HEARTBEAT": 0xFF,
}


def parse_c_define(text: str, name: str) -> int | None:
    match = re.search(
        rf"^\s*#define\s+{re.escape(name)}\s+(0x[0-9A-Fa-f]+|\d+)[uUlL]*\s*(?:/\*.*\*/)?\s*$",
        text,
        re.MULTILINE,
    )
    return int(match.group(1), 0) if match else None


def enum_block(text: str, enum_name: str) -> str:
    match = re.search(rf"enum\s+class\s+{re.escape(enum_name)}\s*:\s*uint8_t\s*\{{(.*?)\}};", text, re.DOTALL)
    return match.group(1) if match else ""


def parse_cpp_enum(text: str, enum_name: str, name: str) -> int | None:
    block = enum_block(text, enum_name)
    match = re.search(rf"\b{re.escape(name)}\s*=\s*(0x[0-9A-Fa-f]+|\d+)", block)
    return int(match.group(1), 0) if match else None


def parse_cpp_const(text: str, name: str) -> int | None:
    match = re.search(
        rf"constexpr\s+uint(?:8|16|32)_t\s+{re.escape(name)}\s*=\s*(0x[0-9A-Fa-f]+|\d+)[uUlL]*\s*;",
        text,
    )
    return int(match.group(1), 0) if match else None


def c_function_body(text: str, name: str) -> str:
    """Return a C function body with balanced braces, or an empty string."""
    match = re.search(rf"\b{re.escape(name)}\s*\([^;]*?\)\s*\{{", text, re.DOTALL)
    if not match:
        return ""
    opening = text.find("{", match.start())
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[opening + 1 : index]
    return ""


def run_contract() -> dict[str, Any]:
    c_header = C_HEADER.read_text(encoding="utf-8", errors="replace")
    c_main_header = C_MAIN_HEADER.read_text(encoding="utf-8", errors="replace")
    c_main_source = C_MAIN_SOURCE.read_text(encoding="utf-8", errors="replace")
    c_source = C_SOURCE.read_text(encoding="utf-8", errors="replace")
    c_lift_header = C_LIFT_HEADER.read_text(encoding="utf-8", errors="replace")
    c_lift_source = C_LIFT_SOURCE.read_text(encoding="utf-8", errors="replace")
    c_uart_header = C_UART_HEADER.read_text(encoding="utf-8", errors="replace")
    c_uart_source = C_UART_SOURCE.read_text(encoding="utf-8", errors="replace")
    c_imu_source = C_IMU_SOURCE.read_text(encoding="utf-8", errors="replace")
    cpp_header = CPP_HEADER.read_text(encoding="utf-8", errors="replace")
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    for name, expected in EXPECTED_UP.items():
        c_value = parse_c_define(c_header, f"UP_{name}")
        cpp_value = parse_cpp_enum(cpp_header, "UpType", name)
        add(f"up:{name}", c_value == cpp_value == expected, f"c={c_value} cpp={cpp_value} expected={expected}")
    for name, expected in EXPECTED_DOWN.items():
        c_value = parse_c_define(c_header, f"DN_{name}")
        cpp_value = parse_cpp_enum(cpp_header, "DownType", name)
        add(f"down:{name}", c_value == cpp_value == expected, f"c={c_value} cpp={cpp_value} expected={expected}")

    add("safety_payload_size", struct.calcsize("<BBH") == 4, f"size={struct.calcsize('<BBH')}")
    add("firmware_info_payload_size", struct.calcsize("<HHIBBH") == 12, f"size={struct.calcsize('<HHIBBH')}")
    identity_specs = [
        ("PROTO_PROTOCOL_VERSION", "TARGET_FIRMWARE_PROTOCOL_VERSION", 2),
        ("PROTO_CAPABILITIES", "TARGET_FIRMWARE_CAPABILITIES", 0x003F),
        ("PROTO_FIRMWARE_BUILD_ID", "TARGET_FIRMWARE_BUILD_ID", 2026071907),
        ("PROTO_REQUIRED_TEST_MODE", "TARGET_FIRMWARE_TEST_MODE", 0),
        ("PROTO_HW_VARIANT", "TARGET_FIRMWARE_HW_VARIANT", 1),
    ]
    for c_name, cpp_name, expected in identity_specs:
        c_value = parse_c_define(c_header, c_name)
        cpp_value = parse_cpp_const(cpp_header, cpp_name)
        add(
            f"firmware_identity:{c_name}",
            c_value == cpp_value == expected,
            f"c={c_value} cpp={cpp_value} expected={expected}",
        )
    protocol_timeout = parse_c_define(c_header, "PROTO_COMMAND_LINK_TIMEOUT_MS")
    main_timeout = parse_c_define(c_main_header, "HEARTBEAT_TIMEOUT_MS")
    add(
        "heartbeat_timeout_contract",
        protocol_timeout == main_timeout == 1000,
        f"protocol={protocol_timeout} main={main_timeout}",
    )
    add(
        "imu_autobaud:rates",
        parse_c_define(c_uart_header, "IMU_BAUDRATE_FAST") == 115200
        and parse_c_define(c_uart_header, "IMU_BAUDRATE_FALLBACK") == 9600,
        "USART3 probes 115200 and 9600 baud",
    )
    for token in [
        "imu_uart_set_baud",
        "usart3_imu_dma_start",
        "DMA1_Stream1->CR &= ~DMA_SxCR_EN",
        "disable_guard > 0u",
        "USART3->CR3 = 0",
        "s_imu_rx_rd = 0",
    ]:
        add(f"imu_autobaud:uart:{token}", token in c_uart_source, token)
    for token in [
        "IMU_BAUD_PROBE_INTERVAL_MS",
        "IMU_LINK_STALE_MS",
        "s_baud_locked = 1",
        "imu_uart_baud() == IMU_BAUDRATE_FAST",
        "imu_uart_set_baud(next_baud)",
    ]:
        add(f"imu_autobaud:state:{token}", token in c_imu_source, token)
    for token in [
        "case DN_CLEAR_ESTOP:",
        "PROTO_ACK_ESTOP_LATCHED",
        "PROTO_ACK_LINK_STALE",
        "motion_interlock_status",
        "s_state.estop_latched = 1",
        "s_state.estop_latched = 0",
        "proto_send_safety_state",
        "proto_send_firmware_info",
    ]:
        add(f"firmware:{token}", token in c_source, token)
    for command in ["DN_SET_LIFT_HEIGHT", "DN_SET_ELECTROMAGNET", "DN_LIFT_HOME"]:
        match = re.search(
            rf"case\s+{command}:(.*?)(?=\n\s*case\s+|\n\s*default:)",
            c_source,
            re.DOTALL,
        )
        block = match.group(1) if match else ""
        ok = (
            "motion_interlock_status(now_ms)" in block
            and f"proto_send_ack({command}, interlock)" in block
        )
        add(f"firmware_interlock:{command}", ok, command)

    add(
        "main:periodic_firmware_identity",
        "proto_send_firmware_info(PROTO_PROTOCOL_VERSION, PROTO_CAPABILITIES" in c_main_source,
        "main.c periodically publishes the exact firmware identity",
    )

    move_steps_body = c_function_body(c_lift_source, "bsp_lift_move_steps")
    bitbang_body = c_function_body(c_lift_source, "bsp_lift_bitbang_pulses")
    lift_service_body = c_function_body(c_lift_source, "bsp_lift_service")
    lift_stop_body = c_function_body(c_lift_source, "bsp_lift_stop")
    actuator_stop_body = c_function_body(c_lift_source, "bsp_lift_actuator_stop")
    servo_hold_body = c_function_body(c_lift_source, "bsp_lift_servo_hold")
    proto_init_body = c_function_body(c_source, "proto_init")
    proto_service_body = c_function_body(c_source, "proto_service")
    servo_ramp_start_body = c_function_body(c_source, "video_servo_ramp_start")
    servo_ramp_service_body = c_function_body(c_source, "video_servo_ramp_service")
    set_lift_target_body = c_function_body(c_source, "set_lift_target_m")
    aux_stop_body = c_function_body(c_main_source, "apply_auxiliary_safety_stop")

    add(
        "firmware:boot_estop_latched",
        "s_state.estop_latched = 1u" in proto_init_body
        and "s_state.emergency_stop_request = 1u" in proto_init_body,
        "F407 reset starts fail-closed with the firmware estop latched",
    )
    add(
        "servo:nonblocking_segmented_ramp",
        all(
            token in servo_ramp_start_body + servo_ramp_service_body
            for token in ["2100u", "1900u", "1700u", "1550u", "LIFT_SERVO_RIGHT_US"]
        )
        and "delay_ms(" not in servo_ramp_start_body + servo_ramp_service_body,
        "servo uses a cooperative 2300-to-1400 us ramp without blocking heartbeat",
    )
    add(
        "servo:isolated_diagnostic_commands",
        all(
            token in c_source
            for token in ["VIDEO_SERVO_RIGHT_COMMAND", "VIDEO_SERVO_LEFT_COMMAND"]
        )
        and "video_servo_ramp_start(1u, now_ms)" in set_lift_target_body
        and "video_servo_ramp_start(0u, now_ms)" in set_lift_target_body,
        "-3/-4 commands route only to the servo ramp",
    )
    add(
        "servo:estop_preserves_hold_pwm",
        "bsp_lift_servo_pwm_valid" in servo_hold_body
        and "bsp_lift_servo_hold()" in aux_stop_body,
        "motion estop preserves the current servo hold PWM",
    )
    add(
        "lift:service_api",
        "void    bsp_lift_service(uint32_t now_ms);" in c_lift_header and bool(lift_service_body),
        "finite GPIO motion exposes a cooperative service hook",
    )
    add(
        "lift:finite_move_non_blocking",
        bool(move_steps_body)
        and "LIFT_MOTION_FINITE_GPIO" in move_steps_body
        and "delay_ms(" not in move_steps_body
        and not re.search(r"\b(?:for|while)\s*\(", move_steps_body),
        "bsp_lift_move_steps schedules work without delay/loop pulse generation",
    )
    add(
        "lift:bitpulse_non_blocking",
        bool(bitbang_body)
        and "LIFT_MOTION_BITBANG_GPIO" in bitbang_body
        and "delay_ms(" not in bitbang_body
        and not re.search(r"\b(?:for|while)\s*\(", bitbang_body),
        "BITPULSE is STOP-responsive instead of monopolizing the ASCII loop",
    )
    add(
        "lift:segmented_gpio_service",
        all(
            token in lift_service_body
            for token in [
                "gpio_write(GPIOC, 9, 1)",
                "gpio_write(GPIOC, 9, 0)",
                "LIFT_SEGMENT_STEPS",
                "LIFT_SEGMENT_DWELL_MS",
            ]
        ),
        "service retains the validated GPIO pulse and segmented dwell behavior",
    )
    add(
        "lift:stop_forces_pulse_low",
        "lift_pul_gpio_mode(0)" in lift_stop_body and "delay_ms(" not in lift_stop_body,
        "lift stop cancels scheduling and forces PUL low without waiting",
    )
    add(
        "lift:pushrod_stop_immediate",
        all(token in actuator_stop_body for token in ["gpio_write(GPIOC, 0, 0)", "gpio_write(GPIOC, 13, 0)"])
        and "delay_ms(" not in actuator_stop_body,
        "pushrod stop de-energizes both relay inputs without a guard delay",
    )
    add(
        "protocol:fixture_non_blocking",
        bool(proto_service_body) and "delay_ms(" not in c_source,
        "pick/place/home fixture timing is cooperatively serviced",
    )
    add(
        "protocol:fixture_link_preemption",
        "motion_interlock_status(now_ms)" in proto_service_body
        and "video_fixture_abort(1u)" in proto_service_body,
        "estop or stale heartbeat aborts an in-flight fixture sequence",
    )
    add(
        "main:normal_service_hooks",
        "proto_service(now);" in c_main_source and "bsp_lift_service(now);" in c_main_source,
        "normal mode services protocol and finite lift state machines",
    )
    add(
        "main:test_mode3_service_hook",
        c_main_source.count("bsp_lift_service(now);") >= 2
        and "while (bsp_lift_busy())" not in c_main_source,
        "TEST_MODE=3 remains responsive during finite moves and SAFEZERO",
    )
    add(
        "main:auxiliary_safety_outputs",
        all(
            token in aux_stop_body
            for token in [
                "bsp_lift_stop()",
                "bsp_lift_actuator_stop()",
                "bsp_lift_magnet_set(0)",
            ]
        ),
        "auxiliary safety stop covers lift, pushrod, and magnet",
    )
    heartbeat_guard = re.search(
        r"if\s*\(ps->estop_latched\s*\|\|\s*ps->last_heartbeat_ms\s*==\s*0\s*\|\|"
        r".*?HEARTBEAT_TIMEOUT_MS\)\s*\{(.*?)\}",
        c_main_source,
        re.DOTALL,
    )
    add(
        "main:heartbeat_auxiliary_preemption",
        bool(heartbeat_guard) and "apply_auxiliary_safety_stop();" in heartbeat_guard.group(1),
        "heartbeat timeout invokes the auxiliary fail-safe immediately",
    )

    failed = [item["name"] for item in checks if item["status"] == "FAIL"]
    return {
        "schema_version": "xrd-f407-protocol-contract-v3",
        "overall": "PASS" if not failed else "FAIL",
        "failed": failed,
        "checks": checks,
    }


def main() -> int:
    report = run_contract()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["overall"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
