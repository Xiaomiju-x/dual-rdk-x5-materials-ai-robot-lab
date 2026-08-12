"""furnace_ocr_bpu_node — ROS2 wrapper, image → BPU YOLOv8n LCD OCR → /furnace_reading.

跟 furnace_ocr_node 接口一致 (drop-in replacement). 用 launch arg use_bpu_ocr 切换.

参数:
    image_topic   (str):   输入图像 topic, 默认 /pt_camera/image_raw
    rate_hz       (float): 处理频率, 默认 1.0 (BPU 推理 ~7ms, 1Hz 远低于上限)
    bin_path      (str):   BPU bin 路径, 默认 ~/bpu_models/lcd_yolov8n.bin
    config_yaml   (str):   ROI yaml (panel_x/y/w/h + power_led_*)
    conf_thresh   (float): YOLO score 阈值, 默认 0.3
    iou_thresh    (float): NMS IoU 阈值, 默认 0.45
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

from .furnace_ocr import OcrConfig
from .furnace_ocr_node import _load_config_from_yaml, _isnan
from .furnace_ocr_bpu import FurnaceOcrBpuProcessor, DEFAULT_BIN


class FurnaceOcrBpuNode(Node):
    def __init__(self):
        super().__init__('furnace_ocr_bpu_node')

        self.declare_parameter('image_topic', '/pt_camera/image_raw')
        self.declare_parameter('rate_hz', 1.0)
        self.declare_parameter('bin_path', DEFAULT_BIN)
        self.declare_parameter('config_yaml', '')
        self.declare_parameter('conf_thresh', 0.3)
        self.declare_parameter('iou_thresh', 0.45)
        self.declare_parameter('test_image_path', '')
        self.declare_parameter('publish_snapshot_threshold', 0.7)

        self.image_topic = self.get_parameter('image_topic').value
        self.rate_hz = float(self.get_parameter('rate_hz').value)
        bin_path = self.get_parameter('bin_path').value
        config_yaml = self.get_parameter('config_yaml').value
        conf_thresh = float(self.get_parameter('conf_thresh').value)
        iou_thresh = float(self.get_parameter('iou_thresh').value)
        self.test_image_path = self.get_parameter('test_image_path').value
        self.snapshot_thresh = float(self.get_parameter('publish_snapshot_threshold').value)

        cfg = _load_config_from_yaml(config_yaml)
        self.processor = FurnaceOcrBpuProcessor(
            cfg, bin_path=bin_path,
            conf_thresh=conf_thresh, iou_thresh=iou_thresh,
        )
        self.bridge = CvBridge()

        self._latest_bgr: Optional[np.ndarray] = None

        sensor_qos = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.pub = self.create_publisher(FurnaceReading, '/furnace_reading', 10)

        if self.test_image_path:
            self._latest_bgr = cv2.imread(self.test_image_path)
            if self._latest_bgr is None:
                self.get_logger().error(f'cannot read test image: {self.test_image_path}')
            else:
                self.get_logger().info(
                    f'TEST MODE: {self.test_image_path} shape={self._latest_bgr.shape}'
                )
        else:
            self.create_subscription(Image, self.image_topic, self._on_image, sensor_qos)

        self.create_timer(1.0 / self.rate_hz, self._tick)
        self.get_logger().info(
            f'furnace_ocr_bpu_node started, bin={bin_path}, '
            f'image_topic={self.image_topic}, rate={self.rate_hz}Hz, '
            f'conf={conf_thresh}, iou={iou_thresh}'
        )

    def _on_image(self, msg: Image):
        try:
            self._latest_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge failed: {e}')

    def _tick(self):
        if self._latest_bgr is None:
            return

        try:
            result = self.processor.process_frame(self._latest_bgr)
        except Exception as e:
            self.get_logger().error(f'BPU forward failed: {e}')
            return

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

        if result.needs_vl_recheck or result.fire_detected or result.smoke_detected:
            ok, buf = cv2.imencode('.jpg', self._latest_bgr,
                                   [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                msg.snapshot_b64 = base64.b64encode(buf.tobytes()).decode('ascii')

        self.pub.publish(msg)


def main():
    rclpy.init()
    node = FurnaceOcrBpuNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
