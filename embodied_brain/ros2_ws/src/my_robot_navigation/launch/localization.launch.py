"""Load a validated saved occupancy map and start Nav2 AMCL localization."""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def _validate_map(context):
    raw_map = LaunchConfiguration("map").perform(context).strip()
    if not raw_map:
        raise RuntimeError("map is required when localization mode is selected")
    map_path = Path(raw_map).expanduser()
    if not map_path.is_absolute():
        raise RuntimeError(f"map must be an absolute YAML path: {raw_map}")
    if not map_path.is_file():
        raise RuntimeError(f"map YAML does not exist: {map_path}")
    image_value = ""
    for line in map_path.read_text(encoding="utf-8", errors="replace").splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() == "image":
            image_value = value.strip().strip("'\"")
            break
    if not image_value:
        raise RuntimeError(f"map YAML has no image field: {map_path}")
    image_path = Path(image_value).expanduser()
    if not image_path.is_absolute():
        image_path = map_path.parent / image_path
    if not image_path.is_file():
        raise RuntimeError(f"map image does not exist: {image_path}")
    return []


def generate_launch_description() -> LaunchDescription:
    pkg_nav = FindPackageShare("my_robot_navigation")
    pkg_nav2 = FindPackageShare("nav2_bringup")
    default_params = PathJoinSubstitution([pkg_nav, "config", "nav2_params.yaml"])

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "map",
                default_value="",
                description="Absolute path to a map_server YAML file; required for localization",
            ),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("autostart", default_value="true"),
            DeclareLaunchArgument("use_composition", default_value="False"),
            DeclareLaunchArgument("params_file", default_value=default_params),
            OpaqueFunction(function=_validate_map),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [pkg_nav2, "launch", "localization_launch.py"]
                    )
                ),
                launch_arguments={
                    "map": LaunchConfiguration("map"),
                    "use_sim_time": LaunchConfiguration("use_sim_time"),
                    "autostart": LaunchConfiguration("autostart"),
                    "use_composition": LaunchConfiguration("use_composition"),
                    "params_file": LaunchConfiguration("params_file"),
                }.items(),
            ),
        ]
    )
