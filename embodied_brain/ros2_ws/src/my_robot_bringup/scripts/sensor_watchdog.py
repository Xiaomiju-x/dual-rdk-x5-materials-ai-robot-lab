#!/usr/bin/env python3
"""sensor_watchdog.py — 传感器流断供自愈看门狗 (2026-06-10).

背景: 车载 USB hub 供电不稳, 偶发整树掉电重枚举. 两个驱动的表现:
  - ldlidar: 已打补丁 (连续超时 5s 自杀 + launch respawn) — 通常自愈, 这里兜底.
  - astra_camera: 热插拔 listener 只报 onDeviceConnected 但不重新拉流 →
    节点活着但僵死, respawn 不触发. 必须外部杀掉让 respawn 重开.

逻辑: 订阅 /scan 与 /depth_camera/depth/image_raw, 流停 >STALE_SEC 且
对应 USB 设备还在总线上 (说明不是物理拔线) → pkill 驱动进程 →
launch 层 respawn 3s 内重启重新 open 设备.

跑法: 独立 systemd 服务 eb_sensor_watchdog.service (不进 full.launch —
看门狗不能和被看的进程同生共死).
"""
import subprocess
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, LaserScan

STALE_SEC = 20.0     # 流停多久算僵死
KILL_GRACE = 40.0    # 杀完后的宽限期 — Astra 冷启动到首帧实测要 15-25s, 给足余量防误杀循环
CHECK_PERIOD = 5.0

# (名字, USB vid:pid 在线检查, 进程 pkill 匹配串)
LIDAR_USB = '10c4:ea60'   # D300 的 CP2102 转接板
ASTRA_USB = '2bc5:0403'   # Astra Pro 深度分支 (OpenNI)


def usb_present(vidpid: str) -> bool:
    try:
        out = subprocess.run(['lsusb'], capture_output=True, text=True, timeout=5).stdout
        return vidpid in out
    except Exception:
        return False


class SensorWatchdog(Node):
    def __init__(self):
        super().__init__('sensor_watchdog')
        now = time.monotonic()
        # 启动宽限: 看门狗自己可能在驱动冷启动中途起来, 给跟 kill 后一样的余量
        self.last_scan = now + KILL_GRACE - STALE_SEC
        self.last_depth = now + KILL_GRACE - STALE_SEC
        self.create_subscription(LaserScan, '/scan',
                                 self._on_scan, qos_profile_sensor_data)
        self.create_subscription(Image, '/depth_camera/depth/image_raw',
                                 self._on_depth, qos_profile_sensor_data)
        self.create_timer(CHECK_PERIOD, self._check)
        self.get_logger().info('sensor watchdog up: /scan + /depth_camera/depth/image_raw, '
                               f'stale={STALE_SEC}s')

    def _on_scan(self, _msg):
        self.last_scan = time.monotonic()

    def _on_depth(self, _msg):
        self.last_depth = time.monotonic()

    def _kill(self, pattern: str, why: str):
        self.get_logger().warn(f'{why} -> pkill -f {pattern} (respawn 接管重开)')
        subprocess.run(['pkill', '-f', pattern], timeout=5)

    def _check(self):
        now = time.monotonic()

        if now - self.last_depth > STALE_SEC:
            if usb_present(ASTRA_USB):
                self._kill('astra_camera_node',
                           f'depth 流停 {now - self.last_depth:.0f}s 但 Astra 在总线上 (僵死)')
                self.last_depth = now + KILL_GRACE - STALE_SEC
            else:
                self.get_logger().warn('depth 流停且 Astra 不在 USB 总线上 — 等设备回来/检查线缆')
                self.last_depth = now  # 设备离线不反复刷屏

        if now - self.last_scan > STALE_SEC:
            if usb_present(LIDAR_USB):
                # ldlidar 自杀补丁通常先触发; 这里兜底 (e.g. 卡在非超时状态)
                self._kill('ldlidar_stl_ros2_node',
                           f'/scan 停 {now - self.last_scan:.0f}s 但 CP2102 在总线上')
                self.last_scan = now + KILL_GRACE - STALE_SEC
            else:
                self.get_logger().warn('/scan 停且雷达 CP2102 不在 USB 总线上 — 需要物理拔插')
                self.last_scan = now


def main():
    rclpy.init()
    node = SensorWatchdog()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == '__main__':
    main()
