"""furnace_ocr_node — ROS2 节点, 把 furnace_ocr.py 包成 image → /furnace_reading.

订阅:
    输入图像 topic (默认 /pt_camera/image_raw, Phase 7 小米云台或 K3 备选云台发)
    可改成 /lift_camera/image_raw 用 200W USB 测 (开发期)

发布:
    /furnace_reading (my_robot_msgs/FurnaceReading) @ 1 Hz (炉温变化慢)

参数:
    image_topic (str): 输入图像 topic, 默认 /pt_camera/image_raw
    rate_hz (float):   处理频率, 默认 1.0
    config_yaml (str): OCR ROI 配置 yaml 路径, 空则用 default
    test_image_path (str): 离线测试: 不订阅 topic, 反复读这张图. 默认空.
"""
from __future__ import annotations

import base64
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from my_robot_msgs.msg import FurnaceReading

from .furnace_ocr import FurnaceOcrProcessor, OcrConfig, OcrResult


def _load_config_from_yaml(path: str) -> OcrConfig:
    """从 yaml 加载 OcrConfig. 文件不存在则返回 default. 实现极简, Phase 6 再丰富."""
    import os
    import yaml
    if not path or not os.path.exists(path):
        return OcrConfig.default()
    with open(path, 'r', encoding='utf-8') as f:
        d = yaml.safe_load(f) or {}

    # 极简加载, 只支持顶层字段; 全部缺失字段用 default 填
    base = OcrConfig.default()
    # 这里就先只支持几个最常改的:
    for k in ('panel_x', 'panel_y', 'panel_w', 'panel_h',
              'power_led_x', 'power_led_y', 'power_led_w', 'power_led_h',
              'seg_on_threshold', 'confidence_threshold'):
        if k in d:
            setattr(base, k, d[k])
    return base


class FurnaceOcrNode(Node):
    def __init__(self):
        super().__init__('furnace_ocr_node')

        self.declare_parameter('image_topic', '/pt_camera/image_raw')
        self.declare_parameter('rate_hz', 1.0)
        self.declare_parameter('config_yaml', '')
        self.declare_parameter('test_image_path', '')
        self.declare_parameter('publish_snapshot_threshold', 0.7)

        self.image_topic = self.get_parameter('image_topic').value
        self.rate_hz = float(self.get_parameter('rate_hz').value)
        config_yaml = self.get_parameter('config_yaml').value
        self.test_image_path = self.get_parameter('test_image_path').value
        self.snapshot_thresh = float(self.get_parameter('publish_snapshot_threshold').value)

        # OCR processor
        self.cfg = _load_config_from_yaml(config_yaml)
        self.processor = FurnaceOcrProcessor(self.cfg)
        self.bridge = CvBridge()

        # 缓存最新图
        self._latest_bgr: Optional[np.ndarray] = None

        # I/O
        sensor_qos = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.pub = self.create_publisher(FurnaceReading, '/furnace_reading', 10)

        if self.test_image_path:
            # 离线测试: 加载静态图, 反复处理
            self._latest_bgr = cv2.imread(self.test_image_path)
            if self._latest_bgr is None:
                self.get_logger().error(f'cannot read test image: {self.test_image_path}')
            else:
                self.get_logger().info(
                    f'TEST MODE: using static image {self.test_image_path} '
                    f'shape={self._latest_bgr.shape}'
                )
        else:
            self.create_subscription(Image, self.image_topic, self._on_image, sensor_qos)
            self.get_logger().info(f'subscribing image: {self.image_topic}')

        # 处理定时器
        self.create_timer(1.0 / self.rate_hz, self._tick)

        self.get_logger().info(
            f'furnace_ocr_node started, rate={self.rate_hz}Hz, '
            f'snapshot_thresh={self.snapshot_thresh}'
        )

    def _on_image(self, msg: Image):
        try:
            self._latest_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge failed: {e}')

    def _tick(self):
        if self._latest_bgr is None:
            return

        result = self.processor.process_frame(self._latest_bgr)

        # 转 ROS2 msg
        msg = FurnaceReading()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'pt_camera_optical_frame'
        msg.pv = float(result.pv) if not _isnan(result.pv) else float('nan')
        msg.sv = float(result.sv) if not _isnan(result.sv) else float('nan')
        msg.mv = float(result.mv) if not _isnan(result.mv) else float('nan')
        msg.pv_confidence = result.pv_confidence
        msg.sv_confidence = result.sv_confidence
        msg.mv_confidence = result.mv_confidence
        msg.power_indicator_on = result.power_indicator_on
        msg.screen_visible = result.screen_visible
        msg.fire_detected = result.fire_detected
        msg.fire_confidence = result.fire_confidence
        msg.smoke_detected = result.smoke_detected
        msg.smoke_confidence = result.smoke_confidence
        msg.needs_vl_recheck = result.needs_vl_recheck

        # 截图: 异常 OR 置信度低时附 base64 (供 dispatcher 上传 Qwen-VL 或邮件附件)
        if result.needs_vl_recheck or result.fire_detected or result.smoke_detected:
            ok, buf = cv2.imencode('.jpg', self._latest_bgr,
                                   [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                msg.snapshot_b64 = base64.b64encode(buf.tobytes()).decode('ascii')

        self.pub.publish(msg)


def _isnan(v: float) -> bool:
    try:
        return v != v  # NaN != NaN
    except Exception:
        return True


def main():
    rclpy.init()
    node = FurnaceOcrNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
