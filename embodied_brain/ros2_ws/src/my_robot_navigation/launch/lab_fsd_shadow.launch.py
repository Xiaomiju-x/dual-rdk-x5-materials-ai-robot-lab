"""Launch Lab-FSD BEV Occupancy Shadow Planner."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_nav = FindPackageShare("my_robot_navigation")
    config = PathJoinSubstitution([pkg_nav, "config", "lab_fsd_shadow.yaml"])

    declare_params = DeclareLaunchArgument("lab_fsd_params_file", default_value=config)
    declare_ai_brain = DeclareLaunchArgument(
        "ai_brain_url", default_value="http://192.0.2.103:8888"
    )

    shadow = Node(
        package="my_robot_navigation",
        executable="bev_shadow_planner.py",
        name="lab_fsd_bev_shadow_planner",
        output="screen",
        parameters=[LaunchConfiguration("lab_fsd_params_file")],
    )

    vision_bridge = Node(
        package="my_robot_navigation",
        executable="vision_bev_bridge.py",
        name="lab_fsd_vision_bev_bridge",
        output="screen",
        parameters=[
            LaunchConfiguration("lab_fsd_params_file"),
            {"ai_brain_url": LaunchConfiguration("ai_brain_url")},
        ],
    )

    return LaunchDescription([declare_params, declare_ai_brain, shadow, vision_bridge])
