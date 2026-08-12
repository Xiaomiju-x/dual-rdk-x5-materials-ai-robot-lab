"""full_nav.launch.py — 一键拉起 SLAM + Nav2 + 深度伪激光 (但不含传感器驱动).

调用顺序:
    [前置] my_robot_drivers/sensors.launch.py 必须先跑 (出 /scan + /odom + 深度数据)
    [前置] my_robot_description display.launch.py 必须先跑 (出 robot_description + URDF TF)

本 launch 包含:
    1. depth_to_laserscan.launch.py     /depth_camera → /scan_depth
    2. slam.launch.py                    slam_toolbox 建图 + 发 map→odom TF
    3. nav2.launch.py                    Nav2 完整栈

参数:
    use_sim_time: rosbag 回放时改 true
    use_slam:     默认 true (mapping 模式); false 时加载 map 并启动 AMCL
    map:          use_slam=false 时必填的地图 YAML 绝对路径
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_nav = FindPackageShare('my_robot_navigation')

    declare_use_sim_time = DeclareLaunchArgument('use_sim_time', default_value='false')
    declare_use_slam = DeclareLaunchArgument('use_slam', default_value='true')
    declare_use_depth_scan = DeclareLaunchArgument('use_depth_scan', default_value='true')
    declare_use_lab_fsd_shadow = DeclareLaunchArgument('use_lab_fsd_shadow', default_value='false')
    declare_use_collision_monitor = DeclareLaunchArgument('use_collision_monitor', default_value='false')
    declare_map = DeclareLaunchArgument(
        'map', default_value='',
        description='Saved map YAML; required when use_slam=false')

    depth_scan = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_nav, 'launch', 'depth_to_laserscan.launch.py'])
        ),
        condition=IfCondition(LaunchConfiguration('use_depth_scan')),
    )

    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_nav, 'launch', 'slam.launch.py'])
        ),
        launch_arguments={
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }.items(),
        condition=IfCondition(LaunchConfiguration('use_slam')),
    )

    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_nav, 'launch', 'localization.launch.py'])
        ),
        launch_arguments={
            'map': LaunchConfiguration('map'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }.items(),
        condition=UnlessCondition(LaunchConfiguration('use_slam')),
    )

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_nav, 'launch', 'nav2.launch.py'])
        ),
        launch_arguments={
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'use_collision_monitor': LaunchConfiguration('use_collision_monitor'),
        }.items(),
    )

    lab_fsd_shadow = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_nav, 'launch', 'lab_fsd_shadow.launch.py'])
        ),
        condition=IfCondition(LaunchConfiguration('use_lab_fsd_shadow')),
    )

    return LaunchDescription([
        declare_use_sim_time,
        declare_use_slam,
        declare_use_depth_scan,
        declare_use_lab_fsd_shadow,
        declare_use_collision_monitor,
        declare_map,
        depth_scan,
        slam,
        localization,
        nav2,
        lab_fsd_shadow,
    ])
