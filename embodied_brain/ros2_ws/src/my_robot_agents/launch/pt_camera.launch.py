"""pt_camera.launch.py — 通用图像源 → /pt_camera/image_raw.

支持两种 source:
    1. USB 摄像头 (Phase 8 K3 备选, 推荐 — 米家锁死后的现实方案):
       ros2 launch my_robot_agents pt_camera.launch.py source:='/dev/PT_CAM'
    2. RTSP 流 (旧 J 路径, 现已弃用):
       ros2 launch my_robot_agents pt_camera.launch.py source:='rtsp://...'

环境变量备选: EB_PT_CAMERA_SOURCE
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    declare_source = DeclareLaunchArgument(
        'source',
        default_value=os.environ.get('EB_PT_CAMERA_SOURCE', '/dev/PT_CAM'),
        description='USB device (/dev/PT_CAM, /dev/video2, 0) or RTSP URL',
    )
    declare_topic = DeclareLaunchArgument('image_topic', default_value='/pt_camera/image_raw')
    declare_fps = DeclareLaunchArgument('target_fps', default_value='10.0')
    declare_width = DeclareLaunchArgument('width', default_value='1280')
    declare_height = DeclareLaunchArgument('height', default_value='720')
    declare_fourcc = DeclareLaunchArgument(
        'fourcc', default_value='MJPG',
        description='USB camera FOURCC (MJPG for higher fps; YUYV fallback). RTSP ignored.',
    )

    return LaunchDescription([
        declare_source, declare_topic, declare_fps,
        declare_width, declare_height, declare_fourcc,
        Node(
            package='my_robot_agents',
            executable='pt_camera',
            name='pt_camera_node',
            output='screen',
            parameters=[{
                'source': LaunchConfiguration('source'),
                'image_topic': LaunchConfiguration('image_topic'),
                'target_fps': LaunchConfiguration('target_fps'),
                'width': LaunchConfiguration('width'),
                'height': LaunchConfiguration('height'),
                'fourcc': LaunchConfiguration('fourcc'),
            }],
        ),
    ])
