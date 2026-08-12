"""depth_to_laserscan.launch.py — Astra 深度图 → /scan_depth (假激光).

把深度相机做成"水平一行像素"的伪激光雷达, 喂 Nav2 局部 costmap 的 ObstacleLayer
让小车避开 LD14 看不到的桌脚 / 横梁等 (LD14 是单线 2D, 看不到不在它高度的障碍).

输出:
    /scan_depth (sensor_msgs/LaserScan, frame_id='depth_camera_optical_frame', ~30Hz)

参数:
    output_frame: TF frame, 默认 'depth_camera_optical_frame' (跟 URDF 一致)
    range_min/max: 截断 (m), 默认 0.6 / 8.0 (Astra Pro spec)
    scan_height: 取深度图水平条带高度 (像素), 默认 10
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    declare_input_topic = DeclareLaunchArgument(
        'input_image_topic', default_value='/depth_camera/depth/image_raw'
    )
    declare_input_info = DeclareLaunchArgument(
        'input_info_topic', default_value='/depth_camera/depth/camera_info'
    )
    declare_output_topic = DeclareLaunchArgument(
        'output_topic', default_value='/scan_depth'
    )

    d2l_node = Node(
        package='depthimage_to_laserscan',
        executable='depthimage_to_laserscan_node',
        name='depth_to_laserscan',
        output='screen',
        remappings=[
            ('depth', LaunchConfiguration('input_image_topic')),
            ('depth_camera_info', LaunchConfiguration('input_info_topic')),
            ('scan', LaunchConfiguration('output_topic')),
        ],
        parameters=[{
            'output_frame': 'depth_camera_optical_frame',
            'range_min': 0.6,        # Astra Pro 最近测距
            'range_max': 8.0,        # Astra Pro 最远 8m
            'scan_height': 10,       # 取深度图中央 10 行 (大概水平线)
            'scan_time': 0.033,      # 30Hz
        }]
    )

    return LaunchDescription([
        declare_input_topic,
        declare_input_info,
        declare_output_topic,
        d2l_node,
    ])
