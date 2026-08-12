"""Wheel/IMU state estimation for the real F407 chassis."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution([
        FindPackageShare('my_robot_navigation'), 'config', 'ekf_odom.yaml'
    ])
    declare_config = DeclareLaunchArgument(
        'ekf_config_file', default_value=default_config,
        description='robot_localization EKF configuration')

    ekf = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[LaunchConfiguration('ekf_config_file')],
        remappings=[('odometry/filtered', '/odom')],
    )
    return LaunchDescription([declare_config, ekf])
