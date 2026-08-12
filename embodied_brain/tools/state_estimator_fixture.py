#!/usr/bin/env python3
"""Private-domain zero-motion fixture for the EKF; never publishes commands."""

import argparse
import os
import time

import rclpy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=float, default=8.0)
    parser.add_argument('--rate', type=float, default=20.0)
    args = parser.parse_args()
    domain = int(os.environ.get('ROS_DOMAIN_ID', '0'))
    if domain < 200 or os.environ.get('ROS_LOCALHOST_ONLY') != '1':
        parser.error('requires ROS_DOMAIN_ID>=200 and ROS_LOCALHOST_ONLY=1')

    rclpy.init()
    node = rclpy.create_node('state_estimator_zero_fixture')
    wheel_pub = node.create_publisher(Odometry, '/wheel_odom', 20)
    imu_pub = node.create_publisher(Imu, '/imu', 20)
    period = 1.0 / max(1.0, args.rate)
    deadline = time.monotonic() + max(1.0, args.duration)
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            stamp = node.get_clock().now().to_msg()
            wheel = Odometry()
            wheel.header.stamp = stamp
            wheel.header.frame_id = 'odom'
            wheel.child_frame_id = 'base_footprint'
            wheel.pose.pose.orientation.w = 1.0
            wheel.twist.covariance[0] = 0.05
            wheel.twist.covariance[35] = 0.10
            wheel_pub.publish(wheel)

            imu = Imu()
            imu.header.stamp = stamp
            imu.header.frame_id = 'imu_link'
            imu.orientation_covariance[0] = -1.0
            imu.angular_velocity_covariance[0] = 0.001
            imu.angular_velocity_covariance[4] = 0.001
            imu.angular_velocity_covariance[8] = 0.001
            imu.linear_acceleration.z = 9.81
            imu_pub.publish(imu)
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(period)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    print('STATE_ESTIMATOR_FIXTURE_COMPLETE no_cmd_topics_published=true')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
