"""Start the finals LDROBOT D300 chain with mandatory self filtering.

The hardware driver is frozen to ``/scan_raw``.  The only publisher of the
consumer-facing ``/scan`` topic is ``scan_self_filter``, which loads the
commissioned body-contour v2 artifact and fails closed when it cannot verify or
transform a frame.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

RAW_SCAN_TOPIC = "/scan_raw"
FILTERED_SCAN_TOPIC = "/scan"
BODY_CONTOUR_PATH = "/home/rdk/rb_voe/lidar_body_contour.v1.json"


def generate_launch_description():
    declare_port = DeclareLaunchArgument(
        "port_name",
        default_value="/dev/LD14",
        description="D300 serial device symlink",
    )
    declare_range_max = DeclareLaunchArgument(
        "range_max",
        default_value="12.0",
        description="D300 maximum valid range in metres",
    )

    lidar_node = Node(
        package="ldlidar_stl_ros2",
        executable="ldlidar_stl_ros2_node",
        name="ld14_lidar",
        output="screen",
        respawn=True,
        respawn_delay=3.0,
        parameters=[
            {
                "product_name": "LDLiDAR_LD19",
                "topic_name": RAW_SCAN_TOPIC,
                "frame_id": "laser_link",
                "port_name": LaunchConfiguration("port_name"),
                "port_baudrate": 230400,
                "laser_scan_dir": True,
                "enable_angle_crop_func": False,
                "angle_crop_min": 135.0,
                "angle_crop_max": 225.0,
                "range_min": 0.10,
                "range_max": LaunchConfiguration("range_max"),
            }
        ],
        # Keep the chain correct even if a vendor build ignores topic_name.
        remappings=[("scan", RAW_SCAN_TOPIC), ("/scan", RAW_SCAN_TOPIC)],
    )

    scan_self_filter = Node(
        package="my_robot_drivers",
        executable="scan_self_filter",
        name="scan_self_filter",
        output="screen",
        respawn=True,
        respawn_delay=3.0,
        parameters=[
            {
                "contour_path": BODY_CONTOUR_PATH,
                "input_topic": RAW_SCAN_TOPIC,
                "output_topic": FILTERED_SCAN_TOPIC,
                "target_frame": "base_footprint",
                "transform_timeout_s": 0.05,
            }
        ],
    )

    return LaunchDescription(
        [
            declare_port,
            declare_range_max,
            lidar_node,
            scan_self_filter,
        ]
    )
