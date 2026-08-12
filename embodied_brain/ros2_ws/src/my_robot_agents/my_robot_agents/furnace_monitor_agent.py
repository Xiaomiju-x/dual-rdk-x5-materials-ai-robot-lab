"""furnace_monitor_agent — 4 类异常检测 (I1+I2+I4+I6) + 发 /alarm.

订阅:
    /furnace_reading (my_robot_msgs/FurnaceReading)

发布:
    /alarm (my_robot_msgs/Alarm)

报警规则 (按 ADR-EB-5):
    I1 SOURCE_TEMPERATURE_OUT_OF_RANGE (CRITICAL)
        PV < min_temp 或 PV > max_temp (默认 [-10, 1600] °C)
    I2 SOURCE_PV_SV_DEVIATION (WARNING / CRITICAL)
        |PV - SV| > deviation_threshold 持续 sustain_seconds 秒 (默认 100°C / 30s)
        持续 > 60s 升级 CRITICAL
    I4 SOURCE_POWER_LIGHT_OFF (CRITICAL)
        power_indicator_on = false 持续 > 5s (短暂闪烁不算)
    I6 SOURCE_FIRE_OR_SMOKE (CRITICAL)
        fire_detected 或 smoke_detected = true (任一即报)

防止报警轰炸:
    每个 source 同一报警在 alarm_cooldown 秒内只发一次

参数 (ros params):
    min_temp_c (float, 默认 -10.0)
    max_temp_c (float, 默认 1600.0)
    deviation_threshold_c (float, 默认 100.0)
    deviation_sustain_s (float, 默认 30.0)
    power_off_sustain_s (float, 默认 5.0)
    alarm_cooldown_s (float, 默认 60.0)
"""
from __future__ import annotations

import math
from typing import Dict

import rclpy
from rclpy.node import Node

from my_robot_msgs.msg import Alarm, FurnaceReading


# Alarm.msg 里定义的常量, 这里硬编码以便不依赖运行时 enum
SOURCE_TEMPERATURE_OUT_OF_RANGE = 1
SOURCE_PV_SV_DEVIATION          = 2
SOURCE_POWER_LIGHT_OFF          = 3
SOURCE_FIRE_OR_SMOKE            = 4

LEVEL_INFO     = 1
LEVEL_WARNING  = 2
LEVEL_CRITICAL = 3

# 通道位掩码 (跟 alert_dispatcher 约定)
CHAN_TTS    = 1
CHAN_EMAIL  = 2
CHAN_WECHAT = 4
CHAN_LOG    = 8


class FurnaceMonitorAgent(Node):
    def __init__(self):
        super().__init__('furnace_monitor_agent')

        # 参数
        self.declare_parameter('min_temp_c', -10.0)
        self.declare_parameter('max_temp_c', 1600.0)
        self.declare_parameter('deviation_threshold_c', 100.0)
        self.declare_parameter('deviation_sustain_s', 30.0)
        self.declare_parameter('deviation_critical_s', 60.0)
        self.declare_parameter('power_off_sustain_s', 5.0)
        self.declare_parameter('alarm_cooldown_s', 60.0)

        self.min_temp = float(self.get_parameter('min_temp_c').value)
        self.max_temp = float(self.get_parameter('max_temp_c').value)
        self.dev_thresh = float(self.get_parameter('deviation_threshold_c').value)
        self.dev_sustain = float(self.get_parameter('deviation_sustain_s').value)
        self.dev_critical = float(self.get_parameter('deviation_critical_s').value)
        self.power_sustain = float(self.get_parameter('power_off_sustain_s').value)
        self.cooldown = float(self.get_parameter('alarm_cooldown_s').value)

        # 状态机
        self._dev_start_t: float | None = None
        self._power_off_start_t: float | None = None
        self._last_alarm_time: Dict[int, float] = {}  # source → time

        # I/O
        self.sub = self.create_subscription(
            FurnaceReading, '/furnace_reading', self._on_reading, 10)
        self.pub = self.create_publisher(Alarm, '/alarm', 10)

        self.get_logger().info(
            f'furnace_monitor_agent started, range=[{self.min_temp}, {self.max_temp}]°C '
            f'dev>{self.dev_thresh}@{self.dev_sustain}s power>{self.power_sustain}s '
            f'cooldown={self.cooldown}s'
        )

    # ============ 主回调 ============

    def _on_reading(self, msg: FurnaceReading):
        # 当前时间 (秒)
        t = self._now_s(msg)

        # 屏幕不亮 / OCR 没结果 时跳过 PV/SV/MV 类报警, 但 I4 (Power Light) 仍要看
        screen_ok = msg.screen_visible

        # I1: 温度超阈值
        if screen_ok and not _isnan(msg.pv):
            if msg.pv < self.min_temp or msg.pv > self.max_temp:
                self._maybe_publish(
                    t, SOURCE_TEMPERATURE_OUT_OF_RANGE, LEVEL_CRITICAL,
                    title='烧结炉温度超出合理范围',
                    description=(
                        f'实测 PV = {msg.pv:.0f}°C 超出 '
                        f'[{self.min_temp}, {self.max_temp}]°C 安全区间'
                    ),
                    snapshot=msg.snapshot_b64,
                )

        # I2: PV-SV 偏差大持续
        if screen_ok and not _isnan(msg.pv) and not _isnan(msg.sv):
            dev = abs(msg.pv - msg.sv)
            if dev > self.dev_thresh:
                if self._dev_start_t is None:
                    self._dev_start_t = t
                else:
                    duration = t - self._dev_start_t
                    if duration > self.dev_sustain:
                        level = LEVEL_CRITICAL if duration > self.dev_critical else LEVEL_WARNING
                        self._maybe_publish(
                            t, SOURCE_PV_SV_DEVIATION, level,
                            title='烧结炉温控偏差过大',
                            description=(
                                f'实测 PV = {msg.pv:.0f}°C, 设定 SV = {msg.sv:.0f}°C, '
                                f'偏差 {dev:.0f}°C 持续 {duration:.0f} 秒, '
                                f'温控环路可能失效'
                            ),
                            snapshot=msg.snapshot_b64,
                        )
            else:
                self._dev_start_t = None  # 偏差恢复正常, 重置

        # I4: Power Indicator 灭
        if not msg.power_indicator_on:
            if self._power_off_start_t is None:
                self._power_off_start_t = t
            else:
                duration = t - self._power_off_start_t
                if duration > self.power_sustain:
                    self._maybe_publish(
                        t, SOURCE_POWER_LIGHT_OFF, LEVEL_CRITICAL,
                        title='烧结炉 Power 指示灯熄灭',
                        description=(
                            f'红色 Power Indicator 熄灭已 {duration:.0f} 秒, '
                            f'设备可能掉电'
                        ),
                        snapshot=msg.snapshot_b64,
                    )
        else:
            self._power_off_start_t = None  # 灯回来, 重置

        # I6: 火焰 / 烟雾 (突发, 不需 sustain)
        if msg.fire_detected:
            self._maybe_publish(
                t, SOURCE_FIRE_OR_SMOKE, LEVEL_CRITICAL,
                title='烧结炉区域检测到火焰',
                description=f'红橙色像素聚集 confidence={msg.fire_confidence:.2f}',
                snapshot=msg.snapshot_b64,
            )
        if msg.smoke_detected:
            self._maybe_publish(
                t, SOURCE_FIRE_OR_SMOKE, LEVEL_CRITICAL,
                title='烧结炉区域检测到烟雾',
                description=f'灰色像素 + 运动差分 confidence={msg.smoke_confidence:.2f}',
                snapshot=msg.snapshot_b64,
            )

    # ============ 工具 ============

    def _now_s(self, msg: FurnaceReading) -> float:
        # 首选 msg 时间戳, 防节点时钟偏差
        try:
            return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        except Exception:
            return self.get_clock().now().nanoseconds * 1e-9

    def _maybe_publish(self, t: float, source: int, level: int,
                       title: str, description: str, snapshot: str = ''):
        # cooldown 抑制
        last = self._last_alarm_time.get(source, -1e18)
        if t - last < self.cooldown:
            return

        a = Alarm()
        a.header.stamp = self.get_clock().now().to_msg()
        a.source = source
        a.level = level
        # 通道分级
        if level == LEVEL_INFO:
            a.channels = CHAN_LOG
        elif level == LEVEL_WARNING:
            a.channels = CHAN_TTS | CHAN_LOG
        else:  # CRITICAL
            a.channels = CHAN_TTS | CHAN_EMAIL | CHAN_WECHAT | CHAN_LOG

        a.title = title
        a.description = description
        a.snapshot_b64 = snapshot
        a.ros2_topic = '/furnace_reading'

        self.pub.publish(a)
        self._last_alarm_time[source] = t

        self.get_logger().warn(f'[ALARM] source={source} level={level} {title} :: {description}')


def _isnan(v: float) -> bool:
    return v != v


def main():
    rclpy.init()
    node = FurnaceMonitorAgent()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
