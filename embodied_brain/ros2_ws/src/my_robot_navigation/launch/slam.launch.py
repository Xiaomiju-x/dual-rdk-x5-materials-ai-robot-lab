"""slam.launch.py — 启动 slam_toolbox 在线异步建图.

依赖 (跑前必须有):
    /scan          (LD14 雷达, 来自 my_robot_drivers/lidar.launch.py)
    /odom + TF     (来自 serial_f407_node 真机 OR fake_odom 假数据)
    URDF TF        (来自 my_robot_description display.launch.py 或 robot_state_publisher)

输出:
    /map (nav_msgs/OccupancyGrid)
    map → odom TF
    动态服务: /slam_toolbox/save_map, /slam_toolbox/serialize_map 等

参数:
    slam_config_file (str): slam_toolbox yaml 路径
    use_sim_time (bool): rosbag 回放时改 true
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare('my_robot_navigation')
    default_config = PathJoinSubstitution([
        pkg_share, 'config', 'slam_toolbox_online_async.yaml'
    ])

    declare_config = DeclareLaunchArgument(
        'slam_config_file', default_value=default_config,
        description='slam_toolbox 参数 yaml'
    )
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='rosbag 回放时改 true'
    )

    slam_node = Node(
        package='slam_toolbox',
        executable='sync_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[
            LaunchConfiguration('slam_config_file'),
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
        ]
    )

    return LaunchDescription([
        declare_config,
        declare_use_sim_time,
        slam_node,
    ])
