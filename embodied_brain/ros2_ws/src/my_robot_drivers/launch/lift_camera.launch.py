"""lift_camera.launch.py — 启动 200W USB 升降台广角摄像头.

包装 ROS2 现成的 usb_cam 包.

输出 topic:
    /lift_camera/image_raw      (sensor_msgs/Image, 1280×720 @ 30 fps)
    /lift_camera/camera_info    (sensor_msgs/CameraInfo, 未标定时填默认)

参数:
    video_device (str): /dev/lift_camera (或 /dev/video2 fallback)
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare('my_robot_drivers')
    default_config = PathJoinSubstitution([pkg_share, 'config', 'lift_camera.yaml'])

    declare_config = DeclareLaunchArgument(
        'config_file', default_value=default_config,
        description='usb_cam 参数 yaml'
    )
    declare_namespace = DeclareLaunchArgument(
        'namespace', default_value='lift_camera',
        description='topic 前缀'
    )

    usb_cam_node = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='lift_camera',
        namespace=LaunchConfiguration('namespace'),
        output='screen',
        parameters=[LaunchConfiguration('config_file')],
    )

    return LaunchDescription([
        declare_config,
        declare_namespace,
        usb_cam_node,
    ])
