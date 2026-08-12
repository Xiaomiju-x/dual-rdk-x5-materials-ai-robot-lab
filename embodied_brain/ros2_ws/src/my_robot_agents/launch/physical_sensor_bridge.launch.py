"""Launch one calibrated physical-sensor evidence bridge, disabled by default."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    enabled = LaunchConfiguration("enabled")
    return LaunchDescription(
        [
            DeclareLaunchArgument("enabled", default_value="false"),
            DeclareLaunchArgument(
                "node_name", default_value="physical_sensor_evidence_bridge"
            ),
            DeclareLaunchArgument(
                "sample_topic", default_value="/pickup/hardware_sensor_sample"
            ),
            DeclareLaunchArgument(
                "request_topic",
                default_value="/pickup/physical_evidence_request",
            ),
            DeclareLaunchArgument(
                "evidence_topic", default_value="/pickup/physical_evidence"
            ),
            DeclareLaunchArgument(
                "status_topic",
                default_value="/pickup/physical_evidence_bridge_status",
            ),
            DeclareLaunchArgument("calibration_manifest", default_value=""),
            DeclareLaunchArgument("expected_calibration_sha256", default_value=""),
            DeclareLaunchArgument("allow_unapproved_calibration", default_value="false"),
            DeclareLaunchArgument("expected_driver_instance_id", default_value=""),
            DeclareLaunchArgument("expected_publisher_node", default_value=""),
            DeclareLaunchArgument("expected_publisher_namespace", default_value="/"),
            DeclareLaunchArgument("require_unique_publisher", default_value="true"),
            DeclareLaunchArgument("max_sample_age_s", default_value="0.75"),
            DeclareLaunchArgument("max_future_skew_s", default_value="0.25"),
            DeclareLaunchArgument("minimum_quality", default_value="0.80"),
            Node(
                package="my_robot_agents",
                executable="physical_sensor_evidence_bridge",
                name=LaunchConfiguration("node_name"),
                output="screen",
                condition=IfCondition(enabled),
                parameters=[
                    {
                        "enabled": ParameterValue(enabled, value_type=bool),
                        "sample_topic": LaunchConfiguration("sample_topic"),
                        "request_topic": LaunchConfiguration("request_topic"),
                        "evidence_topic": LaunchConfiguration("evidence_topic"),
                        "status_topic": LaunchConfiguration("status_topic"),
                        "calibration_manifest": LaunchConfiguration(
                            "calibration_manifest"
                        ),
                        "expected_calibration_sha256": LaunchConfiguration(
                            "expected_calibration_sha256"
                        ),
                        "allow_unapproved_calibration": ParameterValue(
                            LaunchConfiguration("allow_unapproved_calibration"),
                            value_type=bool,
                        ),
                        "expected_driver_instance_id": LaunchConfiguration(
                            "expected_driver_instance_id"
                        ),
                        "expected_publisher_node": LaunchConfiguration(
                            "expected_publisher_node"
                        ),
                        "expected_publisher_namespace": LaunchConfiguration(
                            "expected_publisher_namespace"
                        ),
                        "require_unique_publisher": ParameterValue(
                            LaunchConfiguration("require_unique_publisher"),
                            value_type=bool,
                        ),
                        "max_sample_age_s": ParameterValue(
                            LaunchConfiguration("max_sample_age_s"), value_type=float
                        ),
                        "max_future_skew_s": ParameterValue(
                            LaunchConfiguration("max_future_skew_s"), value_type=float
                        ),
                        "minimum_quality": ParameterValue(
                            LaunchConfiguration("minimum_quality"), value_type=float
                        ),
                    }
                ],
            ),
        ]
    )
