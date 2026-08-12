"""telemetry_publisher — 1Hz 周期发 /system_telemetry, 让 AI 脑 dashboard 显示具身脑健康度.

订阅:
    /odom        (拿当前位姿)
    /scan        (探测 SLAM 是否在跑)
    /map         (探测 SLAM 是否激活)

发布:
    /system_telemetry (my_robot_msgs/SystemTelemetry, 1Hz)

资源探测:
    - CPU: /proc/stat (前后两次采样取差)
    - RAM: /proc/meminfo
    - 温度: /sys/class/thermal/thermal_zone*/temp
    - BPU: /sys/devices/system/bpu/bpu_load (RDK X5) — 不一定有, fallback 0
    - CMA: /sys/kernel/debug/cma/cma-reserved/size
    - ai_brain ping: 简单 socket connect 测试 192.0.2.103:8888

参数 (ros params):
    rate_hz: 1.0
    ai_brain_url: http://192.0.2.103:8888 (从 env EB_AI_BRAIN_URL 覆盖)
"""
from __future__ import annotations

import os
import socket
import time
from urllib.parse import urlparse

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid

from my_robot_msgs.msg import SystemTelemetry


class TelemetryPublisher(Node):
    def __init__(self):
        super().__init__('telemetry_publisher')

        self.declare_parameter('rate_hz', 1.0)
        self.declare_parameter('ai_brain_url',
                               os.environ.get('EB_AI_BRAIN_URL', 'http://192.0.2.103:8888'))

        rate_hz = float(self.get_parameter('rate_hz').value)
        self.ai_brain_url = self.get_parameter('ai_brain_url').value

        # 解析 ai_brain host:port (用于 ping)
        u = urlparse(self.ai_brain_url)
        self.ai_brain_host = u.hostname or '192.0.2.103'
        self.ai_brain_port = u.port or 8888

        # 状态缓存
        self._last_odom: Odometry | None = None
        self._last_scan_t: float = 0.0
        self._last_map_t: float = 0.0
        self._distance_m: float = 0.0
        self._last_xy: tuple[float, float] | None = None

        # CPU 占用历史 (用差分算 %)
        self._prev_cpu = self._read_proc_stat()

        # I/O
        self.create_subscription(Odometry, '/odom', self._on_odom, 10)
        self.create_subscription(LaserScan, '/scan',
                                 lambda m: self._touch('scan'), 10)
        self.create_subscription(OccupancyGrid, '/map',
                                 lambda m: self._touch('map'), 10)

        self.pub = self.create_publisher(SystemTelemetry, '/system_telemetry', 5)
        self.create_timer(1.0 / rate_hz, self._tick)

        self.get_logger().info(
            f'telemetry_publisher started, rate={rate_hz}Hz, '
            f'ai_brain={self.ai_brain_host}:{self.ai_brain_port}'
        )

    def _on_odom(self, msg: Odometry):
        self._last_odom = msg
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        if self._last_xy is not None:
            dx = x - self._last_xy[0]
            dy = y - self._last_xy[1]
            d = (dx * dx + dy * dy) ** 0.5
            if d < 5.0:  # 异常跳变忽略
                self._distance_m += d
        self._last_xy = (x, y)

    def _touch(self, key: str):
        now = time.time()
        if key == 'scan':
            self._last_scan_t = now
        elif key == 'map':
            self._last_map_t = now

    # ====================== 主 tick ======================

    def _tick(self):
        msg = SystemTelemetry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_footprint'

        msg.cpu_pct = self._cpu_pct()
        ram_used, ram_total = self._ram_gb()
        msg.ram_used_gb = ram_used
        msg.ram_total_gb = ram_total
        msg.bpu_pct = self._bpu_pct()
        msg.cma_used_mb = self._cma_used_mb()
        msg.cpu_temp_c = self._cpu_temp_c()
        msg.battery_pct = -1.0  # 没接电池, 用 -1 标记无效

        msg.ai_brain_reachable, msg.ai_brain_latency_ms = self._ping_ai_brain()
        msg.current_wifi_ssid = self._current_ssid()

        now = time.time()
        msg.slam_active = (now - self._last_map_t) < 5.0
        msg.nav2_active = False  # Phase 4d 完整后从 lifecycle 读取
        msg.nav2_state = 'IDLE'

        if self._last_odom is not None:
            msg.current_pose = self._last_odom.pose.pose
        msg.distance_traveled_m = self._distance_m

        self.pub.publish(msg)

    # ====================== /proc 探测 ======================

    def _read_proc_stat(self):
        """读 /proc/stat 的 cpu 总行: (idle, total)."""
        try:
            with open('/proc/stat', 'r') as f:
                line = f.readline()
            parts = line.split()
            assert parts[0] == 'cpu'
            nums = [int(x) for x in parts[1:8]]
            idle = nums[3] + nums[4]  # idle + iowait
            total = sum(nums)
            return idle, total
        except Exception:
            return 0, 1

    def _cpu_pct(self) -> float:
        cur = self._read_proc_stat()
        prev = self._prev_cpu
        self._prev_cpu = cur
        d_idle = cur[0] - prev[0]
        d_total = cur[1] - prev[1]
        if d_total <= 0:
            return 0.0
        return float(100.0 * (1.0 - d_idle / d_total))

    def _ram_gb(self) -> tuple[float, float]:
        try:
            d = {}
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    k, v = line.split(':')
                    d[k.strip()] = v.strip().split()[0]  # KB
            total_kb = float(d.get('MemTotal', '0'))
            avail_kb = float(d.get('MemAvailable', '0'))
            used_kb = max(0.0, total_kb - avail_kb)
            return used_kb / 1024 / 1024, total_kb / 1024 / 1024
        except Exception:
            return 0.0, 0.0

    def _bpu_pct(self) -> float:
        # RDK X5 Bayes-e: /sys/devices/virtual/misc/bpu_status (估测路径, 找不到就 0)
        for path in ('/sys/kernel/debug/sysctl/bpu/bpu0_load',
                     '/sys/devices/virtual/misc/bpu_status'):
            try:
                with open(path, 'r') as f:
                    s = f.read().strip()
                # 简单数字 0~100
                v = float(s.split()[0]) if s else 0.0
                return min(100.0, max(0.0, v))
            except Exception:
                continue
        return 0.0

    def _cma_used_mb(self) -> float:
        # /proc/meminfo CmaTotal 和 CmaFree
        try:
            with open('/proc/meminfo', 'r') as f:
                cma_total = cma_free = 0
                for line in f:
                    if line.startswith('CmaTotal:'):
                        cma_total = float(line.split()[1])
                    elif line.startswith('CmaFree:'):
                        cma_free = float(line.split()[1])
            return max(0.0, (cma_total - cma_free) / 1024)  # MB
        except Exception:
            return 0.0

    def _cpu_temp_c(self) -> float:
        # /sys/class/thermal/thermal_zone0/temp 单位 1/1000 °C
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                return float(f.read().strip()) / 1000.0
        except Exception:
            return 0.0

    def _ping_ai_brain(self) -> tuple[bool, float]:
        try:
            t0 = time.time()
            with socket.create_connection((self.ai_brain_host, self.ai_brain_port), timeout=1.5):
                pass
            return True, (time.time() - t0) * 1000.0
        except Exception:
            return False, -1.0

    def _current_ssid(self) -> str:
        try:
            import subprocess
            out = subprocess.run(['iwgetid', '-r'], capture_output=True, text=True, timeout=1)
            return (out.stdout or '').strip()
        except Exception:
            return ''


def main():
    rclpy.init()
    node = TelemetryPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
