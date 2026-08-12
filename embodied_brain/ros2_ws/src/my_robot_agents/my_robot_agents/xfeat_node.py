"""xfeat_node — XFeat 视觉特征 BPU 节点 (Round 4 Day 19 D1).

订 /image (sensor_msgs/Image, RGB8) → BPU XFeat → 发:
    /xfeat/keypoints    (visualization_msgs/Marker, sphere list, 关键点坐标)
    /xfeat/heatmap      (sensor_msgs/Image, 480x640 mono8, 可视化)
    /xfeat/stats        (std_msgs/String JSON, FPS / keypoints count / 时延)

性能 (X5 Bayes-e BPU 实测):
    BPU forward 17.4 ms (57 FPS), CPU postproc (NMS+desc) 50 ms (~12 FPS 总)
    bin 大小 985 KB INT8

参数:
    image_topic       /image (default)
    bin_path          /home/rdk/bpu_models/xfeat.bin
    top_k             1024
    publish_heatmap   false (默认关; 开了 ~+5ms cv_bridge encode)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String, Header, ColorRGBA
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from cv_bridge import CvBridge


sys.path.insert(0, '/tmp')   # xfeat_x5_infer.py
try:
    from xfeat_x5_infer import XFeatX5
except Exception as e:
    XFeatX5 = None
    _IMPORT_ERR = str(e)


class XFeatNode(Node):
    def __init__(self):
        super().__init__('xfeat_node')
        self.declare_parameter('image_topic', '/image')
        self.declare_parameter('bin_path', '/home/rdk/bpu_models/xfeat.bin')
        self.declare_parameter('top_k', 1024)
        self.declare_parameter('publish_heatmap', False)

        self.image_topic = self.get_parameter('image_topic').value
        bin_path = self.get_parameter('bin_path').value
        self.top_k = int(self.get_parameter('top_k').value)
        self.publish_heatmap = bool(self.get_parameter('publish_heatmap').value)

        self.bridge = CvBridge()
        if XFeatX5 is None:
            self.get_logger().error(f'cannot import XFeatX5: {_IMPORT_ERR}')
            raise SystemExit(1)
        self.engine = XFeatX5(bin_path=bin_path)

        # publishers
        self.pub_kp = self.create_publisher(Marker, '/xfeat/keypoints', 10)
        self.pub_stats = self.create_publisher(String, '/xfeat/stats', 10)
        self.pub_heat = self.create_publisher(Image, '/xfeat/heatmap', 10) if self.publish_heatmap else None

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(Image, self.image_topic, self._on_image, qos)
        self._frame_count = 0
        self._t_start = time.time()
        self.get_logger().info(
            f'xfeat_node up | sub={self.image_topic} | bin={bin_path} | top_k={self.top_k} | heatmap={self.publish_heatmap}')

    def _on_image(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge fail: {e}')
            return
        try:
            r = self.engine.detect_and_compute(bgr, top_k=self.top_k)
        except Exception as e:
            self.get_logger().error(f'XFeat infer fail: {e}')
            return

        # publish keypoints as marker (sphere list, 在 image frame 下显示)
        marker = Marker()
        marker.header.frame_id = msg.header.frame_id or 'camera'
        marker.header.stamp = msg.header.stamp
        marker.ns = 'xfeat_keypoints'
        marker.id = 0
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 4.0
        marker.scale.y = 4.0
        marker.scale.z = 4.0
        for (x, y) in r['keypoints'][:self.top_k]:
            p = Point()
            p.x, p.y, p.z = float(x), float(y), 0.0
            marker.points.append(p)
            marker.colors.append(ColorRGBA(r=1.0, g=0.5, b=0.0, a=0.9))
        self.pub_kp.publish(marker)

        # stats
        self._frame_count += 1
        avg_fps = self._frame_count / max(time.time() - self._t_start, 0.01)
        stats = {
            'frame': self._frame_count,
            'keypoints': int(len(r['keypoints'])),
            'bpu_ms': round(r['bpu_ms'], 1),
            'post_ms': round(r['post_ms'], 1),
            'avg_fps_total': round(avg_fps, 1),
        }
        s = String(); s.data = json.dumps(stats)
        self.pub_stats.publish(s)


def main():
    rclpy.init()
    try:
        node = XFeatNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
