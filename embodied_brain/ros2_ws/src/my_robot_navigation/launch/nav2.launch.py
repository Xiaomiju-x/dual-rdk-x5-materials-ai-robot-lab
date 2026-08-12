"""nav2.launch.py — 启动 Nav2 完整导航栈.

依赖:
    /scan, /scan_depth (障碍源)
    /odom + TF (定位源)
    /map (来自 slam_toolbox 或 map_server)
    URDF TF (robot_state_publisher)

输出 topic + service + action:
    /cmd_vel (Nav2 velocity_smoother 输出)
    /cmd_vel_safe (启用 Collision Monitor 时的唯一执行器输入)
    /local_costmap/costmap, /global_costmap/costmap
    Nav2 actions: /navigate_to_pose, /navigate_through_poses, /follow_path 等

使用方式:
    在 RViz 上 "2D Goal Pose" 戳一个目标, 看小车自己规划过去.
    或者 ROS2 action client 调用:
        ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose ...

注意: 第一次启 Nav2 后要 'Initial Pose' 标定, 否则 amcl 不知道在哪.
mapping 模式 (slam_toolbox 在跑) 时 amcl 不必要, 但本 launch 还是带上以备 localization 模式切换.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_nav = FindPackageShare('my_robot_navigation')
    pkg_nav2_bringup = FindPackageShare('nav2_bringup')

    default_params = PathJoinSubstitution([pkg_nav, 'config', 'nav2_params.yaml'])
    default_collision_params = PathJoinSubstitution([pkg_nav, 'config', 'collision_monitor.yaml'])

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='false'
    )
    declare_params = DeclareLaunchArgument(
        'params_file', default_value=default_params,
        description='Nav2 配置 yaml'
    )
    declare_autostart = DeclareLaunchArgument(
        'autostart', default_value='true',
        description='自动激活 lifecycle 节点'
    )
    declare_use_composition = DeclareLaunchArgument(
        'use_composition', default_value='False',
        description='X5 ARM CPU 弱, 不用 composition (内存共享但 debug 难)'
    )
    declare_use_collision_monitor = DeclareLaunchArgument(
        'use_collision_monitor', default_value='false',
        description='启用 Nav2 Collision Monitor: Nav2 /cmd_vel -> monitor -> /cmd_vel_safe'
    )
    declare_collision_params = DeclareLaunchArgument(
        'collision_monitor_params_file', default_value=default_collision_params,
        description='Collision Monitor 配置 yaml'
    )

    # Nav2 navigation_launch.py (官方提供, 拉起所有 nav2 节点)
    # Keep the official Humble chain unchanged:
    # controller_server -> /cmd_vel_nav -> velocity_smoother -> /cmd_vel.
    # Collision Monitor consumes /cmd_vel and emits /cmd_vel_safe.  The
    # bringup launch binds the F407 (or fake_odom) to that final topic whenever
    # the monitor is enabled.
    nav2_stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_nav2_bringup, 'launch', 'navigation_launch.py'])
        ),
        launch_arguments={
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'params_file': LaunchConfiguration('params_file'),
            'autostart': LaunchConfiguration('autostart'),
            'use_composition': LaunchConfiguration('use_composition'),
        }.items(),
    )

    collision_monitor = Node(
        package='nav2_collision_monitor',
        executable='collision_monitor',
        name='collision_monitor',
        output='screen',
        parameters=[
            LaunchConfiguration('collision_monitor_params_file'),
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
        ],
        condition=IfCondition(LaunchConfiguration('use_collision_monitor')),
    )

    collision_lifecycle = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_collision_monitor',
        output='screen',
        parameters=[
            LaunchConfiguration('collision_monitor_params_file'),
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
        ],
        condition=IfCondition(LaunchConfiguration('use_collision_monitor')),
    )

    return LaunchDescription([
        declare_use_sim_time,
        declare_params,
        declare_autostart,
        declare_use_composition,
        declare_use_collision_monitor,
        declare_collision_params,
        nav2_stack,
        collision_monitor,
        collision_lifecycle,
    ])
