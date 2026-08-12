"""fake_odom_node — 临时假 odom 发送器, 仅用于 STM32F407 没烧固件前测试 SLAM/Nav2.

行为:
    1. 订阅 /cmd_vel, 用差分驱动模型积分得到 (x, y, yaw)
    2. 50Hz 发 /odom (nav_msgs/Odometry) + odom→base_footprint TF
    3. 没收到 /cmd_vel 时车不动 (vx=0 wz=0)

警告:
    这个节点假装"小车每收到 cmd_vel 都精确按速度运动 (无打滑/无延迟/无误差)".
    跟真实 STM32F407 + 步进电机 + 编码器闭环不一样, 真车上必须换成 serial_f407_node.

参数:
    publish_tf (bool, 默认 True): 是否发 TF
    odom_frame (str, 默认 odom)
    base_frame (str, 默认 base_footprint)
    rate_hz (float, 默认 50.0)
    cmd_vel_timeout_s (float, 默认 0.60): 超时后自动把模拟速度归零
"""
import math
from threading import Lock

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster


class FakeOdomNode(Node):
    def __init__(self):
        super().__init__('fake_odom')

        self.declare_parameter('publish_tf', True)
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('rate_hz', 50.0)
        self.declare_parameter('cmd_vel_timeout_s', 0.60)

        self.publish_tf = self.get_parameter('publish_tf').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        rate_hz = float(self.get_parameter('rate_hz').value)
        self.cmd_vel_timeout_s = max(
            0.05, float(self.get_parameter('cmd_vel_timeout_s').value)
        )

        # 状态
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.vx = 0.0
        self.wz = 0.0
        self.last_t = self.get_clock().now()
        self.last_cmd_t = None
        self.lock = Lock()

        # I/O
        cmd_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(Twist, 'cmd_vel', self.on_cmd_vel, cmd_qos)
        self.odom_pub = self.create_publisher(Odometry, 'odom', 10)
        self.tf_bc = TransformBroadcaster(self) if self.publish_tf else None

        # 主循环
        self.timer = self.create_timer(1.0 / rate_hz, self.tick)

        self.get_logger().info(
            f'fake_odom started: rate={rate_hz}Hz tf={self.publish_tf} '
            f'{self.odom_frame}->{self.base_frame} '
            f'cmd_timeout={self.cmd_vel_timeout_s:.2f}s'
        )

    def on_cmd_vel(self, msg: Twist):
        with self.lock:
            self.vx = msg.linear.x
            self.wz = msg.angular.z
            self.last_cmd_t = self.get_clock().now()

    def tick(self):
        now = self.get_clock().now()
        with self.lock:
            dt = (now - self.last_t).nanoseconds * 1e-9
            self.last_t = now
            cmd_age = (
                float('inf') if self.last_cmd_t is None
                else (now - self.last_cmd_t).nanoseconds * 1e-9
            )
            if cmd_age > self.cmd_vel_timeout_s:
                self.vx = 0.0
                self.wz = 0.0
            if dt <= 0 or dt > 0.5:
                # 第一帧或时间跳变, 跳过积分
                vx = wz = 0.0
            else:
                vx, wz = self.vx, self.wz

            # 简单单环差分积分 (中点法可减误差, 这里直接欧拉, 反正是假 odom)
            self.x += vx * math.cos(self.yaw) * dt
            self.y += vx * math.sin(self.yaw) * dt
            self.yaw += wz * dt
            # 归一化 yaw 到 [-π, π]
            self.yaw = math.atan2(math.sin(self.yaw), math.cos(self.yaw))

            x, y, yaw = self.x, self.y, self.yaw
            cur_vx, cur_wz = vx, wz

        # 四元数 (yaw only)
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)

        # /odom
        odom = Odometry()
        odom.header.stamp = now.to_msg()
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = cur_vx
        odom.twist.twist.angular.z = cur_wz
        # 协方差: 假 odom 标记为不可信 (大方差)
        cov = [0.0] * 36
        cov[0] = cov[7] = 0.5         # x, y
        cov[14] = 1e6                  # z
        cov[21] = cov[28] = 1e6        # roll, pitch
        cov[35] = 0.5                  # yaw
        odom.pose.covariance = cov
        odom.twist.covariance = cov
        self.odom_pub.publish(odom)

        # TF
        if self.tf_bc is not None:
            tf = TransformStamped()
            tf.header.stamp = now.to_msg()
            tf.header.frame_id = self.odom_frame
            tf.child_frame_id = self.base_frame
            tf.transform.translation.x = x
            tf.transform.translation.y = y
            tf.transform.rotation.z = qz
            tf.transform.rotation.w = qw
            self.tf_bc.sendTransform(tf)


def main():
    rclpy.init()
    node = FakeOdomNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
