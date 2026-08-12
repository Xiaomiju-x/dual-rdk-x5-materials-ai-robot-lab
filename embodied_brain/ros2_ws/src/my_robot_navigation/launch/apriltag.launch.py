"""apriltag.launch.py — 拉起 AprilTag 检测节点 (CPU, 不上 BPU).

输入:
    /pt_camera/image_raw          (K3 USB cam) — 主输入
    或 /lift_camera/image_raw     (升降台 cam) — Phase 6 取料用

输出:
    /detections (apriltag_msgs/AprilTagDetectionArray)
    /tf  (每个 tag 一个 child frame 'shelf_X_slot_Y' or 'tagX')

性能 (2026-04-26 调研): 720p decimate=2 threads=4 → 12-20ms/frame, 稳 30Hz.

用法:
    ros2 launch my_robot_navigation apriltag.launch.py
        (默认订阅 /pt_camera/image_raw + /pt_camera/camera_info)

    ros2 launch my_robot_navigation apriltag.launch.py image_topic:=/lift_camera/image_raw
        (取料时切到升降台 cam)
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare('my_robot_navigation')
    config = PathJoinSubstitution([pkg_share, 'config', 'apriltag.yaml'])

    declare_image = DeclareLaunchArgument(
        'image_topic',
        default_value='/pt_camera/image_raw',
        description='输入图像 topic. 取料时切 /lift_camera/image_raw',
    )
    declare_info = DeclareLaunchArgument(
        'camera_info_topic',
        default_value='/pt_camera/camera_info',
    )

    return LaunchDescription([
        declare_image, declare_info,
        Node(
            package='apriltag_ros',
            executable='apriltag_node',
            name='apriltag',
            parameters=[config],
            remappings=[
                ('image_rect', LaunchConfiguration('image_topic')),
                ('camera_info', LaunchConfiguration('camera_info_topic')),
            ],
            output='screen',
        ),
    ])
