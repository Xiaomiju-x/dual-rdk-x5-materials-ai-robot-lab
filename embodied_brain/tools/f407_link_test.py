#!/usr/bin/env python3
"""f407_link_test.py — STM32F407 0xAA55 链路裸测 (不依赖 ROS).

跑在车载 X5 上, 烧完 TEST_MODE 0 固件后第一时间验链路:
    python3 f407_link_test.py                 # 听 5s + 心跳 + 微速 cmd_vel + 急停
    python3 f407_link_test.py --listen-only   # 只听帧流 (车架空时最安全)
    python3 f407_link_test.py --port /dev/F407 --baud 115200

协议 (proto.h, 双向): AA 55 | type | len | payload | sum8(全帧前 N-1 字节)
  上行: 0x01 BASIC_ODOM(20B: x y vx wz yaw_deg) / 0x02 EXT_TELEMETRY(44B 含 IMU)
        0x03 SAFETY_STATE(4B: estop/emergency/blocked_count)
        0x04 FIRMWARE_INFO(12B: protocol/capabilities/build/test-mode/hw)
        0x10 ACK(3B) / 0x1F ERROR(1+msg)
  下行: 0x01 CMD_VEL / 0x02 LIFT / 0x03 MAGNET / 0x04 HOME /
        0x10 EMERGENCY_STOP / 0x11 CLEAR_ESTOP / 0xFF HEARTBEAT
固件看门狗: 1s 无心跳 → 急停; cmd_vel 0.5s 不刷新 → 速度归零.
"""
import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path

import serial

UP_NAMES = {
    0x01: 'ODOM', 0x02: 'EXT_TELEM', 0x03: 'SAFETY', 0x04: 'FIRMWARE_INFO',
    0x10: 'ACK', 0x1F: 'ERROR',
}
EXPECTED_PROTOCOL_VERSION = 2
EXPECTED_CAPABILITIES = 0x003F
EXPECTED_BUILD_ID = 2026071907
EXPECTED_TEST_MODE = 0
EXPECTED_HW_VARIANT = 1


def frame(ftype: int, payload: bytes = b'') -> bytes:
    f = bytes([0xAA, 0x55, ftype, len(payload)]) + payload
    return f + bytes([sum(f) & 0xFF])


def parse_stream(buf: bytearray, on_frame):
    """从 buf 中切出完整帧, 残余留在 buf."""
    while True:
        i = buf.find(b'\xaa\x55')
        if i < 0:
            del buf[:-1]
            return
        if len(buf) < i + 4:
            del buf[:i]
            return
        ln = buf[i + 3]
        end = i + 4 + ln + 1
        if len(buf) < end:
            del buf[:i]
            return
        body = buf[i:end - 1]
        cks = buf[end - 1]
        if (sum(body) & 0xFF) == cks:
            on_frame(buf[i + 2], bytes(buf[i + 4:end - 1]))
            del buf[:end]
        else:
            del buf[:i + 2]   # 坏帧, 跳过这个头


def describe(ftype: int, pl: bytes) -> str:
    if ftype == 0x01 and len(pl) == 20:
        x, y, vx, wz, yaw = struct.unpack('<5f', pl)
        return f'ODOM x={x:+.3f} y={y:+.3f} vx={vx:+.3f} wz={wz:+.3f} yaw={yaw:+.1f}°'
    if ftype == 0x02 and len(pl) == 44:
        ax, ay, az, gx, gy, gz = struct.unpack('<6f', pl[12:36])
        return (f'EXT_TELEM acc=({ax:+.2f},{ay:+.2f},{az:+.2f}) '
                f'gyro=({gx:+.1f},{gy:+.1f},{gz:+.1f})')
    if ftype == 0x03 and len(pl) == 4:
        estop, emergency, blocked = struct.unpack('<BBH', pl)
        return f'SAFETY estop_latched={bool(estop)} emergency_active={bool(emergency)} blocked={blocked}'
    if ftype == 0x04 and len(pl) == 12:
        protocol, capabilities, build_id, test_mode, hw_variant, _reserved = struct.unpack('<HHIBBH', pl)
        return (f'FIRMWARE_INFO protocol={protocol} capabilities=0x{capabilities:04X} '
                f'build={build_id} test_mode={test_mode} hw={hw_variant}')
    if ftype == 0x10 and len(pl) >= 2:
        return f'ACK for=0x{pl[0]:02X} status={pl[1]} ({"OK" if pl[1] == 0 else "ERR"})'
    if ftype == 0x1F:
        return f'ERROR code=0x{pl[0]:02X} msg={pl[1:].decode("ascii", "replace")}'
    return f'{UP_NAMES.get(ftype, f"0x{ftype:02X}")} len={len(pl)} {pl.hex()}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', default='/dev/F407')
    ap.add_argument('--baud', type=int, default=115200)
    ap.add_argument('--listen-only', action='store_true')
    ap.add_argument('--require-firmware-identity', action='store_true',
                    help='listen-only 模式也要求目标固件身份精确匹配')
    ap.add_argument('--v', type=float, default=0.02, help='测试线速度 m/s (默认 0.02 蹭着走)')
    ap.add_argument('--move-sec', type=float, default=2.0, help='动多久')
    ap.add_argument('--ack-timeout', type=float, default=0.5, help='等待指定命令 ACK 的超时秒数')
    ap.add_argument('--require-ack', action='store_true', help='急停 ACK 超时/错误时返回失败')
    ap.add_argument('--verify-estop-interlock', action='store_true',
                    help='验证急停后升降/吸附 ON/home 均被固件 ACK=3 拒绝; 不发送运动 cmd_vel')
    ap.add_argument('--clear-estop', action='store_true',
                    help='测试结束显式发送 CLEAR_ESTOP; 默认保留急停锁存')
    ap.add_argument('--report', default='',
                    help='写 JSON 验收报告, 例如 ~/f407_interlock_evidence/latest.json')
    args = ap.parse_args()

    serial_kwargs = {"port": args.port, "baudrate": args.baud, "timeout": 0.05}
    if os.name == "posix":
        serial_kwargs["exclusive"] = True
    try:
        ser = serial.Serial(**serial_kwargs)
    except Exception as exc:
        print(f'  ✗ 无法独占打开 {args.port}: {exc}')
        sys.exit(2)
    buf = bytearray()
    stats = {}
    acks_by_type = {}
    ack_failures = []
    ack_checks = []
    last_safety = {}
    last_firmware = {}
    commands_started = False

    def on_frame(ftype, pl):
        nonlocal last_safety, last_firmware
        stats[ftype] = stats.get(ftype, 0) + 1
        if ftype == 0x10 and len(pl) >= 2:
            acks_by_type[pl[0]] = (pl[1], time.time())
        if ftype == 0x03 and len(pl) == 4:
            estop, emergency, blocked = struct.unpack('<BBH', pl)
            last_safety = {
                'estop_latched': bool(estop),
                'emergency_active': bool(emergency),
                'blocked_command_count': int(blocked),
            }
        if ftype == 0x04 and len(pl) == 12:
            protocol, capabilities, build_id, test_mode, hw_variant, reserved = struct.unpack('<HHIBBH', pl)
            last_firmware = {
                'protocol_version': int(protocol),
                'capabilities': int(capabilities),
                'build_id': int(build_id),
                'test_mode': int(test_mode),
                'hw_variant': int(hw_variant),
                'reserved': int(reserved),
                'received_at_unix': time.time(),
            }
        # ODOM 50Hz 太刷屏, 每 25 帧打一条; 其他帧全打
        if ftype != 0x01 or stats[ftype] % 25 == 1:
            print(f'  [{time.strftime("%H:%M:%S")}] {describe(ftype, pl)}')

    def pump(sec):
        t0 = time.time()
        while time.time() - t0 < sec:
            data = ser.read(256)
            if data:
                buf.extend(data)
                parse_stream(buf, on_frame)

    def send_and_wait_ack(ftype: int, payload: bytes = b'', label=None, expected_status=0) -> bool:
        label = label or f'0x{ftype:02X}'
        prev = acks_by_type.get(ftype)
        if ftype != 0xFF:
            ser.write(frame(0xFF))
            pump(0.02)
        ser.write(frame(ftype, payload))
        deadline = time.time() + args.ack_timeout
        while time.time() < deadline:
            pump(0.05)
            cur = acks_by_type.get(ftype)
            if cur and cur != prev:
                status, _stamp = cur
                if status == expected_status:
                    print(f'  ACK OK: {label} status={status} (expected)')
                    ack_checks.append({'label': label, 'status': status,
                                       'expected_status': expected_status, 'ok': True})
                    return True
                print(f'  ACK ERR: {label} status={status} expected={expected_status}')
                ack_failures.append((label, f'status={status} expected={expected_status}'))
                ack_checks.append({'label': label, 'status': status,
                                   'expected_status': expected_status, 'ok': False})
                return False
        print(f'  ACK TIMEOUT: {label} after {args.ack_timeout:.2f}s')
        ack_failures.append((label, 'timeout'))
        ack_checks.append({'label': label, 'status': None,
                           'expected_status': expected_status, 'ok': False})
        return False

    def validate_firmware_identity():
        reasons = []
        if not last_firmware:
            reasons.append('FIRMWARE_INFO missing')
        else:
            if last_firmware.get('protocol_version') != EXPECTED_PROTOCOL_VERSION:
                reasons.append('protocol_version mismatch')
            if (int(last_firmware.get('capabilities') or 0) & EXPECTED_CAPABILITIES) != EXPECTED_CAPABILITIES:
                reasons.append('required capabilities missing')
            if last_firmware.get('build_id') != EXPECTED_BUILD_ID:
                reasons.append('build_id mismatch')
            if last_firmware.get('test_mode') != EXPECTED_TEST_MODE:
                reasons.append('TEST_MODE mismatch')
            if last_firmware.get('hw_variant') != EXPECTED_HW_VARIANT:
                reasons.append('hardware variant mismatch')
        return not reasons, reasons

    def write_report(overall: str, failure_reason: str = ''):
        identity_ok, identity_reasons = validate_firmware_identity()
        firmware = dict(last_firmware)
        firmware.update({
            'valid': bool(identity_ok),
            'reasons': identity_reasons,
            'expected': {
                'protocol_version': EXPECTED_PROTOCOL_VERSION,
                'required_capabilities': EXPECTED_CAPABILITIES,
                'build_id': EXPECTED_BUILD_ID,
                'test_mode': EXPECTED_TEST_MODE,
                'hw_variant': EXPECTED_HW_VARIANT,
            },
        })
        report = {
            'schema_version': 'xrd-f407-interlock-evidence-v2',
            'generated_at_unix': time.time(),
            'port': args.port,
            'baud': args.baud,
            'verify_estop_interlock': bool(args.verify_estop_interlock),
            'clear_estop_requested': bool(args.clear_estop),
            'overall': overall,
            'failure_reason': failure_reason,
            'firmware_identity': firmware,
            'stats': {UP_NAMES.get(key, f'0x{key:02X}'): value for key, value in sorted(stats.items())},
            'ack_checks': ack_checks,
            'ack_failures': ack_failures,
            'last_safety': last_safety,
            'safety': {
                'motion_cmd_sent': bool(commands_started and not args.verify_estop_interlock),
                'interlock_test_sends_no_nonzero_cmd_vel': bool(args.verify_estop_interlock),
                'default_leaves_estop_latched': not bool(args.clear_estop),
                'serial_exclusive_open': bool(os.name == 'posix'),
                'identity_verified_before_commands': bool(identity_ok and commands_started),
                'commands_started': bool(commands_started),
            },
        }
        if args.report:
            report_path = Path(args.report).expanduser()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
                encoding='utf-8',
            )
            print(f'  report: {report_path}')
        return report

    print(f'== 1. 静听 5s ({args.port} @ {args.baud}) — 期待 50Hz ODOM + 10Hz EXT_TELEM ==')
    pump(5)
    if not stats:
        print('  ✗ 没有任何帧 — 检查: ①固件烧的是 TEST_MODE 0? ②CH340 接 PD5/PD6? ③TX/RX 有没有反')
        write_report('FAIL', 'no F407 frames received')
        sys.exit(1)
    print(f'  帧统计: ' + ', '.join(f'{UP_NAMES.get(k, hex(k))}×{v}' for k, v in sorted(stats.items())))

    if args.listen_only:
        identity_ok, identity_reasons = validate_firmware_identity()
        if args.require_firmware_identity and not identity_ok:
            print(f'  ✗ 固件身份校验失败，未发送任何命令: {identity_reasons}')
            write_report('FAIL', '; '.join(identity_reasons))
            sys.exit(3)
        write_report('PASS' if (identity_ok or not args.require_firmware_identity) else 'FAIL')
        sys.exit(0)

    identity_ok, identity_reasons = validate_firmware_identity()
    if not identity_ok:
        print(f'  ✗ 固件身份校验失败，拒绝发送心跳、速度、急停或执行器命令: {identity_reasons}')
        write_report('FAIL', '; '.join(identity_reasons))
        sys.exit(3)
    commands_started = True
    print(f'  ✓ 固件身份已验证: protocol={EXPECTED_PROTOCOL_VERSION} build={EXPECTED_BUILD_ID} '
          f'capabilities=0x{int(last_firmware["capabilities"]):04X}')

    if args.verify_estop_interlock:
        print('== 2. 互锁测试预备: 只发零速和心跳, 不请求底盘运动 ==')
        for _ in range(5):
            ser.write(frame(0xFF))
            ser.write(frame(0x01, struct.pack('<2f', 0.0, 0.0)))
            pump(0.2)
    else:
        print(f'== 2. 心跳 + CMD_VEL v={args.v} m/s, {args.move_sec}s (轮子会动!) ==')
        t0 = time.time()
        while time.time() - t0 < args.move_sec:
            ser.write(frame(0xFF))                                   # heartbeat
            ser.write(frame(0x01, struct.pack('<2f', args.v, 0.0)))  # cmd_vel
            pump(0.2)

    print('== 3. 停车 (cmd_vel=0 ×5 + 急停) ==')
    for _ in range(5):
        ser.write(frame(0xFF))
        ser.write(frame(0x01, struct.pack('<2f', 0.0, 0.0)))
        pump(0.2)
    send_and_wait_ack(0x10, label='EMERGENCY_STOP')              # emergency stop
    pump(0.5)

    if args.verify_estop_interlock:
        print('== 4. 固件级急停互锁 (命令应被拒绝, 不执行运动) ==')
        send_and_wait_ack(0x02, struct.pack('<f', 0.05), 'SET_LIFT_HEIGHT blocked', expected_status=3)
        send_and_wait_ack(0x03, b'\x01', 'SET_ELECTROMAGNET ON blocked', expected_status=3)
        send_and_wait_ack(0x03, b'\x00', 'SET_ELECTROMAGNET OFF allowed', expected_status=0)
        send_and_wait_ack(0x04, b'', 'LIFT_HOME blocked', expected_status=3)
        pump(0.5)

    if args.clear_estop:
        print('== 5. 显式清除 F407 急停锁存 ==')
        send_and_wait_ack(0x11, label='CLEAR_ESTOP')
        pump(0.5)
    else:
        print('  急停保持锁存; 仅 --clear-estop 会发送 CLEAR_ESTOP')

    odom = stats.get(0x01, 0)
    telem = stats.get(0x02, 0)
    ack = stats.get(0x10, 0)
    safety = stats.get(0x03, 0)
    print(f'== 结果: ODOM×{odom} EXT_TELEM×{telem} SAFETY×{safety} ACK×{ack} ACK_FAIL×{len(ack_failures)} ==')
    ok = odom > 100 and ack >= 1 and (not args.require_ack or not ack_failures)
    if args.verify_estop_interlock:
        ok = ok and safety > 0 and not ack_failures
    write_report('PASS' if ok else 'FAIL', '' if ok else 'frame count or ACK validation failed')
    print('  ✓ 链路 OK' if ok else '  ⚠ 帧数偏少或 ACK 异常, 检查接线/波特率/固件 ACK')
    if not ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
