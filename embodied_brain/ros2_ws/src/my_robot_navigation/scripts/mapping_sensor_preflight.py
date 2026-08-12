#!/usr/bin/env python3
"""Read-only sensor gate for production-quality SLAM mapping."""

import argparse
import json
import math
import statistics
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Bool


class MappingSensorGate(Node):
    def __init__(self, odom_topic: str) -> None:
        super().__init__('mapping_sensor_preflight')
        self.raw_imu = []
        self.filtered_imu_count = 0
        self.scan_count = 0
        self.odom_count = 0
        self.imu_valid_seen = False
        self.create_subscription(Imu, '/imu/raw', self._raw_imu, qos_profile_sensor_data)
        self.create_subscription(Imu, '/imu', self._filtered_imu, qos_profile_sensor_data)
        self.create_subscription(Bool, '/f407/imu_valid', self._imu_valid, 10)
        self.create_subscription(LaserScan, '/scan', self._scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, odom_topic, self._odom, 20)

    def _raw_imu(self, msg: Imu) -> None:
        accel = msg.linear_acceleration
        gyro = msg.angular_velocity
        values = (accel.x, accel.y, accel.z, gyro.x, gyro.y, gyro.z)
        if all(math.isfinite(v) for v in values):
            norm = math.sqrt(accel.x ** 2 + accel.y ** 2 + accel.z ** 2)
            zero = all(abs(v) < 1e-9 for v in values)
            self.raw_imu.append((norm, zero))

    def _filtered_imu(self, _msg: Imu) -> None:
        self.filtered_imu_count += 1

    def _imu_valid(self, msg: Bool) -> None:
        self.imu_valid_seen = self.imu_valid_seen or bool(msg.data)

    def _scan(self, _msg: LaserScan) -> None:
        self.scan_count += 1

    def _odom(self, _msg: Odometry) -> None:
        self.odom_count += 1


def check(name: str, ok: bool, detail: str) -> dict:
    return {'name': name, 'status': 'PASS' if ok else 'FAIL', 'detail': detail}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=float, default=8.0)
    parser.add_argument('--odom-topic', default='/wheel_odom')
    parser.add_argument('--out')
    args = parser.parse_args()
    duration = max(3.0, args.duration)

    rclpy.init()
    node = MappingSensorGate(args.odom_topic)
    started = time.monotonic()
    try:
        while rclpy.ok() and time.monotonic() - started < duration:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    elapsed = max(0.001, time.monotonic() - started)
    norms = [sample[0] for sample in node.raw_imu]
    zero_count = sum(1 for sample in node.raw_imu if sample[1])
    median_norm = statistics.median(norms) if norms else None
    zero_ratio = zero_count / len(node.raw_imu) if node.raw_imu else 1.0
    scan_hz = node.scan_count / elapsed
    odom_hz = node.odom_count / elapsed
    raw_imu_hz = len(node.raw_imu) / elapsed

    checks = [
        check('scan_rate', scan_hz >= 5.0, f'{scan_hz:.2f} Hz'),
        check('wheel_odom_rate', odom_hz >= 10.0, f'{odom_hz:.2f} Hz on {args.odom_topic}'),
        check('raw_imu_rate', raw_imu_hz >= 5.0, f'{raw_imu_hz:.2f} Hz'),
        check('imu_accel_norm', median_norm is not None and 5.0 <= median_norm <= 15.0,
              'n/a' if median_norm is None else f'median={median_norm:.4f} m/s^2'),
        check('imu_not_all_zero', zero_ratio <= 0.05, f'zero_ratio={zero_ratio:.4f}'),
        check('imu_valid_gate', node.imu_valid_seen, f'valid_seen={node.imu_valid_seen}'),
        check('filtered_imu_stream', node.filtered_imu_count >= 5,
              f'count={node.filtered_imu_count}'),
    ]
    result = {
        'schema': 'xrd-mapping-sensor-preflight-v1',
        'read_only': True,
        'duration_s': round(elapsed, 3),
        'metrics': {
            'scan_hz': round(scan_hz, 3),
            'wheel_odom_hz': round(odom_hz, 3),
            'raw_imu_hz': round(raw_imu_hz, 3),
            'imu_accel_norm_median_mps2': median_norm,
            'imu_zero_ratio': zero_ratio,
        },
        'checks': checks,
        'passed': all(item['status'] == 'PASS' for item in checks),
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    print(payload)
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as handle:
            handle.write(payload + '\n')
    return 0 if result['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
