"""pt_camera_node.py — 通用图像源 → ROS2 image.

支持两种 source:
    1. RTSP 流 (Phase 7 J 路径, 比如小米云台):
       source='rtsp://user:pass@ip:554/path'
    2. USB 摄像头 (Phase 8 K3 备选, 推荐):
       source='/dev/PT_CAM'  或  '/dev/video2'  或  '0' (设备索引)

cv_bridge 转 sensor_msgs/Image, 发 /pt_camera/image_raw (默认),
给 furnace_ocr_node 当输入源.

参数:
    source:      RTSP URL 或 USB 设备路径 (向后兼容旧的 rtsp_url 参数)
    image_topic: /pt_camera/image_raw
    target_fps:  10  (默认; 烧结炉 OCR 不需要高帧率)
    width/height: 0 表示用驱动默认 (USB 设备建议 1280x720 或 1920x1080)
    fourcc:      'MJPG'/'YUYV'/''  (USB 摄像头编码, RTSP 忽略)
    reconnect_sec: 5  (断流自动重连间隔)

2026-04-26 米家 LAN 完全锁死 (rotate/视频流都不响应),
小米云台弃用, K3 USB cam 顶替.
"""
import time
from typing import Optional

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image


class PtCameraNode(Node):
    def __init__(self) -> None:
        super().__init__('pt_camera_node')

        self.declare_parameter('source', '')
        self.declare_parameter('rtsp_url', '')
        self.declare_parameter('image_topic', '/pt_camera/image_raw')
        self.declare_parameter('target_fps', 10.0)
        self.declare_parameter('width', 0)
        self.declare_parameter('height', 0)
        self.declare_parameter('fourcc', '')
        self.declare_parameter('reconnect_sec', 5.0)
        self.declare_parameter('frame_id', 'pt_camera_link')

        source: str = self.get_parameter('source').value
        rtsp_url_legacy: str = self.get_parameter('rtsp_url').value
        self.source: str = source or rtsp_url_legacy
        self.image_topic: str = self.get_parameter('image_topic').value
        self.target_fps: float = float(self.get_parameter('target_fps').value)
        self.width: int = int(self.get_parameter('width').value)
        self.height: int = int(self.get_parameter('height').value)
        self.fourcc: str = self.get_parameter('fourcc').value
        self.reconnect_sec: float = float(self.get_parameter('reconnect_sec').value)
        self.frame_id: str = self.get_parameter('frame_id').value

        if not self.source:
            self.get_logger().error(
                "source is empty. Examples:\n"
                "  USB:  -p source:='/dev/PT_CAM'  or  -p source:='/dev/video2'\n"
                "  RTSP: -p source:='rtsp://user:pass@ip:554/live'"
            )
            raise SystemExit(2)

        self.is_usb: bool = self.source.startswith('/dev/') or self.source.isdigit()

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.pub = self.create_publisher(Image, self.image_topic, qos)
        self.bridge = CvBridge()

        self.cap: Optional[cv2.VideoCapture] = None
        self._open_capture()

        period = 1.0 / max(self.target_fps, 1.0)
        self.timer = self.create_timer(period, self._tick)

        self._frame_count = 0
        self._last_log_t = time.time()

        self.get_logger().info(
            f'pt_camera_node ready. source={self._safe_source()} '
            f'mode={"USB" if self.is_usb else "RTSP"} '
            f'topic={self.image_topic} fps={self.target_fps}'
        )

    def _safe_source(self) -> str:
        # RTSP 密码遮一下; USB 路径直接显示
        if self.is_usb:
            return self.source
        if '@' in self.source and '://' in self.source:
            scheme, rest = self.source.split('://', 1)
            if '@' in rest:
                _, host = rest.split('@', 1)
                return f'{scheme}://***:***@{host}'
        return self.source

    def _open_capture(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        if self.is_usb:
            cap_arg = int(self.source) if self.source.isdigit() else self.source
            self.cap = cv2.VideoCapture(cap_arg, cv2.CAP_V4L2)
            if self.fourcc:
                fcc = cv2.VideoWriter_fourcc(*self.fourcc.upper())
                self.cap.set(cv2.CAP_PROP_FOURCC, fcc)
        else:
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        if self.width > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            self.get_logger().warn(f'Failed to open {self._safe_source()}, will retry in {self.reconnect_sec}s')

    def _tick(self) -> None:
        if self.cap is None or not self.cap.isOpened():
            self.get_logger().warn('Capture closed, reopening...')
            self._open_capture()
            time.sleep(self.reconnect_sec)
            return

        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.get_logger().warn('cap.read() failed, reopening...')
            self._open_capture()
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        self.pub.publish(msg)

        self._frame_count += 1
        now = time.time()
        if now - self._last_log_t >= 5.0:
            fps = self._frame_count / (now - self._last_log_t)
            self.get_logger().info(
                f'streaming {fps:.1f} fps {frame.shape[1]}x{frame.shape[0]} → {self.image_topic}'
            )
            self._frame_count = 0
            self._last_log_t = now


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[PtCameraNode] = None
    try:
        node = PtCameraNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            if node.cap is not None:
                try:
                    node.cap.release()
                except Exception:
                    pass
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
