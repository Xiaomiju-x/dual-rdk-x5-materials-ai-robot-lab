"""location_visualizer_node — 读 lab_locations.yaml + 发 MarkerArray.

把实验室固定点位 (home, furnace_1/2/3, shelf_1/2/3 等) 以 Marker 形式发到
/goal_markers (visualization_msgs/MarkerArray), rviz 里能直接看到所有目标点.

每个 location 出 2 个 marker:
    1. 圆柱 (Cylinder, 30cm 高 20cm 半径) 标位置, 颜色按类别
    2. 文本 (Text View Facing) 标 location_id

颜色:
    home/工位      绿色  (0, 1, 0)
    furnace 烧结炉  红色  (1, 0.3, 0)
    shelf 试剂柜    蓝色  (0.2, 0.4, 1)
    其他           白色

发布频率: 1 Hz (latch 用 Transient Local QoS).

Phase 6 后期可扩展: 加 approach_distance 半径圆 + slot 微位置的 marker.
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, Tuple

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


COLOR_HOME = (0.0, 1.0, 0.2, 0.85)
COLOR_FURNACE = (1.0, 0.3, 0.0, 0.85)
COLOR_SHELF = (0.2, 0.4, 1.0, 0.85)
COLOR_DEFAULT = (0.9, 0.9, 0.9, 0.85)


def _category_color(loc_id: str) -> Tuple[float, float, float, float]:
    if loc_id.startswith('home'):
        return COLOR_HOME
    if loc_id.startswith('furnace'):
        return COLOR_FURNACE
    if loc_id.startswith('shelf'):
        return COLOR_SHELF
    return COLOR_DEFAULT


class LocationVisualizer(Node):
    def __init__(self) -> None:
        super().__init__('location_visualizer')
        self.declare_parameter('locations_yaml', '')
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('publish_rate_hz', 1.0)

        self.frame_id: str = self.get_parameter('frame_id').value
        rate: float = float(self.get_parameter('publish_rate_hz').value)

        self.locations: Dict[str, Dict[str, Any]] = self._load_yaml()

        # Latch: rviz 后开也能拿到上次发的 marker
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.pub = self.create_publisher(MarkerArray, '/goal_markers', qos)
        self.timer = self.create_timer(1.0 / max(rate, 0.1), self._tick)

        self.get_logger().info(
            f'location_visualizer ready, {len(self.locations)} locations → /goal_markers'
        )

    def _load_yaml(self) -> Dict[str, Dict[str, Any]]:
        path = self.get_parameter('locations_yaml').value
        if not path:
            try:
                share = get_package_share_directory('my_robot_navigation')
                path = os.path.join(share, 'config', 'lab_locations.yaml')
            except Exception as e:
                self.get_logger().error(f'cannot find my_robot_navigation share: {e}')
                return {}
        if not os.path.exists(path):
            self.get_logger().error(f'lab_locations.yaml not found at {path}')
            return {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
            return data.get('locations', {}) or {}
        except Exception as e:
            self.get_logger().error(f'failed to parse {path}: {e}')
            return {}

    def _tick(self) -> None:
        msg = MarkerArray()
        now = self.get_clock().now().to_msg()

        for i, (loc_id, loc) in enumerate(self.locations.items()):
            pose = loc.get('pose', {}) or {}
            x = float(pose.get('x', 0.0))
            y = float(pose.get('y', 0.0))
            theta = float(pose.get('theta', 0.0))
            r, g, b, a = _category_color(loc_id)

            # Marker 1: 圆柱标位置
            cyl = Marker()
            cyl.header.stamp = now
            cyl.header.frame_id = self.frame_id
            cyl.ns = 'location_cylinder'
            cyl.id = i
            cyl.type = Marker.CYLINDER
            cyl.action = Marker.ADD
            cyl.pose.position.x = x
            cyl.pose.position.y = y
            cyl.pose.position.z = 0.15
            cyl.pose.orientation.z = math.sin(theta / 2.0)
            cyl.pose.orientation.w = math.cos(theta / 2.0)
            cyl.scale.x = 0.40   # 直径 40cm
            cyl.scale.y = 0.40
            cyl.scale.z = 0.30   # 高 30cm
            cyl.color = ColorRGBA(r=r, g=g, b=b, a=a)
            cyl.lifetime.sec = 2  # 没新的 2s 后过期 (节点死了 rviz 不挂)
            msg.markers.append(cyl)

            # Marker 2: 朝向箭头 (theta)
            arrow = Marker()
            arrow.header = cyl.header
            arrow.ns = 'location_arrow'
            arrow.id = i
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.pose.position.x = x
            arrow.pose.position.y = y
            arrow.pose.position.z = 0.30
            arrow.pose.orientation.z = math.sin(theta / 2.0)
            arrow.pose.orientation.w = math.cos(theta / 2.0)
            arrow.scale.x = 0.50  # 长 50cm
            arrow.scale.y = 0.08
            arrow.scale.z = 0.08
            arrow.color = ColorRGBA(r=r, g=g, b=b, a=1.0)
            arrow.lifetime.sec = 2
            msg.markers.append(arrow)

            # Marker 3: 文本 location_id + description
            txt = Marker()
            txt.header = cyl.header
            txt.ns = 'location_text'
            txt.id = i
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.pose.position.x = x
            txt.pose.position.y = y
            txt.pose.position.z = 0.55
            txt.pose.orientation.w = 1.0
            txt.scale.z = 0.18  # 字高 18cm
            txt.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            description = loc.get('description', '')
            txt.text = f'{loc_id}\n{description[:30]}'
            txt.lifetime.sec = 2
            msg.markers.append(txt)

        self.pub.publish(msg)


def main():
    rclpy.init()
    node = LocationVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
