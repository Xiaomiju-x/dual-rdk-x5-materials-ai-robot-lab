"""serial_f407.launch.py — 启动 ROS2 ↔ STM32F407 串口桥节点.

输入:
    cmd_vel_topic (geometry_msgs/Twist; 默认 /cmd_vel)
    /lift/target_height (std_msgs/Float32)
    服务: /set_lift_height (my_robot_msgs/SetLiftHeight)
    服务: /set_electromagnet (my_robot_msgs/SetElectromagnet)
    服务: /lift_home (std_srvs/Trigger)
    服务: /estop, /clear_estop (std_srvs/Trigger)
    /estop (std_msgs/Bool, true=assert only; false is ignored for safety)

输出:
    /odom (nav_msgs/Odometry)
    /imu/raw (sensor_msgs/Imu, unconditionally published diagnostics stream)
    /imu  (sensor_msgs/Imu, validity-gated state-estimation stream)
    /f407/imu_valid (std_msgs/Bool)
    /lift_status (my_robot_msgs/LiftStatus)
    /f407/estop_latched, /f407/cmd_vel_expired (std_msgs/Bool)
    /f407/firmware_identity_valid (std_msgs/Bool)
    /f407/firmware_info (std_msgs/String JSON)
    /diagnostics (diagnostic_msgs/DiagnosticArray)
    TF: odom → base_footprint (默认开)
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ⚠ 参数名必须独占 ('f407_port' 而非 'port_name'): lidar.launch.py 也声明
    # 'port_name' (默认 /dev/LD14), 两者经 sensors.launch.py 同一 GroupAction 包含时
    # 共享 launch configuration 作用域, 先声明者胜 → serial 节点曾被喂成雷达口
    # (2026-06-11 实测: serial_f407 打开 /dev/LD14, /odom 不出 + 雷达流被抢).
    declare_port = DeclareLaunchArgument(
        'f407_port', default_value=os.environ.get('EB_SERIAL_PORT', '/dev/F407'),
        description='STM32 USB-TTL 串口设备路径 (env EB_SERIAL_PORT 可覆盖)'
    )
    declare_baud = DeclareLaunchArgument(
        'baud_rate', default_value='115200',
        description='波特率'
    )
    declare_publish_tf = DeclareLaunchArgument(
        'publish_tf', default_value='true',
        description='是否发布 odom→base_footprint TF; 用 robot_localization EKF 融合时改 false'
    )
    declare_odom_topic = DeclareLaunchArgument(
        'odom_topic', default_value='/odom',
        description='F407 odometry output; use /wheel_odom when EKF is enabled'
    )
    declare_cmd_vel_timeout = DeclareLaunchArgument(
        'cmd_vel_timeout_s', default_value='0.60',
        description='ROS-side cmd_vel stale timeout; sends zero cmd_vel when expired'
    )
    declare_cmd_vel_topic = DeclareLaunchArgument(
        'cmd_vel_topic', default_value='/cmd_vel',
        description='执行器速度输入；Collision Monitor 开启时由顶层绑定为 /cmd_vel_safe'
    )
    declare_ack_timeout = DeclareLaunchArgument(
        'ack_timeout_ms', default_value='300',
        description='ACK wait timeout for service commands'
    )
    declare_write_timeout = DeclareLaunchArgument(
        'write_timeout_ms', default_value='50',
        description='Serial write retry timeout for one 0xAA55 frame'
    )
    declare_require_ack = DeclareLaunchArgument(
        'require_ack_for_services', default_value='true',
        description='Wait for F407 ACK before reporting service success'
    )
    declare_diagnostics_hz = DeclareLaunchArgument(
        'diagnostics_hz', default_value='1.0',
        description='diagnostic_msgs/DiagnosticArray publish rate'
    )
    declare_rx_stale_timeout = DeclareLaunchArgument(
        'rx_stale_timeout_s', default_value='1.0',
        description='Warn when no F407 frames arrive within this timeout'
    )
    declare_max_linear = DeclareLaunchArgument(
        'max_linear_mps', default_value='0.25',
        description='Hard ROS-side linear velocity limit before F407'
    )
    declare_max_angular = DeclareLaunchArgument(
        'max_angular_rps', default_value='1.20',
        description='Hard ROS-side angular velocity limit before F407'
    )
    declare_min_lift = DeclareLaunchArgument(
        'min_lift_height_m', default_value='0.0',
        description='Reject lower lift targets before F407'
    )
    declare_max_lift = DeclareLaunchArgument(
        'max_lift_height_m', default_value='0.20',
        description='Reject targets above current F407 5000-step/0.20m soft cap'
    )
    declare_lift_arrival_tol = DeclareLaunchArgument(
        'lift_arrival_tolerance_m', default_value='0.015',
        description='Tolerance for /set_lift_height wait_for_arrival'
    )
    declare_lift_arrival_timeout = DeclareLaunchArgument(
        'lift_arrival_default_timeout_s', default_value='30.0',
        description='Default timeout when /set_lift_height wait_for_arrival=true and request timeout_s=0'
    )
    declare_require_firmware_identity = DeclareLaunchArgument(
        'require_firmware_identity',
        default_value=os.environ.get('EB_REQUIRE_F407_IDENTITY', 'true'),
        description='Fail closed for nonzero motion/actuation unless exact F407 firmware identity is fresh'
    )
    declare_firmware_identity_stale = DeclareLaunchArgument(
        'firmware_identity_stale_s', default_value='3.0',
        description='Maximum age of the periodic F407 FIRMWARE_INFO frame'
    )
    declare_gate_invalid_imu = DeclareLaunchArgument(
        'gate_invalid_imu', default_value='true',
        description='Publish /imu only after physically plausible consecutive samples'
    )

    serial_node = Node(
        package='my_robot_drivers',
        executable='serial_f407_node',
        name='serial_f407',
        output='screen',
        parameters=[{
            'port_name':  LaunchConfiguration('f407_port'),
            'baud_rate':  LaunchConfiguration('baud_rate'),
            'odom_frame': 'odom',
            'base_frame': 'base_footprint',
            'imu_frame':  'imu_link',
            'publish_tf': LaunchConfiguration('publish_tf'),
            'cmd_vel_topic': LaunchConfiguration('cmd_vel_topic'),
            'heartbeat_hz': 5.0,
            'cmd_vel_timeout_s': LaunchConfiguration('cmd_vel_timeout_s'),
            'ack_timeout_ms': LaunchConfiguration('ack_timeout_ms'),
            'write_timeout_ms': LaunchConfiguration('write_timeout_ms'),
            'require_ack_for_services': LaunchConfiguration('require_ack_for_services'),
            'diagnostics_hz': LaunchConfiguration('diagnostics_hz'),
            'rx_stale_timeout_s': LaunchConfiguration('rx_stale_timeout_s'),
            'max_linear_mps': LaunchConfiguration('max_linear_mps'),
            'max_angular_rps': LaunchConfiguration('max_angular_rps'),
            'min_lift_height_m': LaunchConfiguration('min_lift_height_m'),
            'max_lift_height_m': LaunchConfiguration('max_lift_height_m'),
            'lift_arrival_tolerance_m': LaunchConfiguration('lift_arrival_tolerance_m'),
            'lift_arrival_default_timeout_s': LaunchConfiguration('lift_arrival_default_timeout_s'),
            'require_firmware_identity': LaunchConfiguration('require_firmware_identity'),
            'firmware_identity_stale_s': LaunchConfiguration('firmware_identity_stale_s'),
            'gate_invalid_imu': LaunchConfiguration('gate_invalid_imu'),
            'imu_accel_norm_min_mps2': 5.0,
            'imu_accel_norm_max_mps2': 15.0,
            'imu_min_valid_samples': 5,
        }],
        remappings=[('odom', LaunchConfiguration('odom_topic'))],
    )

    return LaunchDescription([
        declare_port,
        declare_baud,
        declare_publish_tf,
        declare_odom_topic,
        declare_cmd_vel_topic,
        declare_cmd_vel_timeout,
        declare_ack_timeout,
        declare_write_timeout,
        declare_require_ack,
        declare_diagnostics_hz,
        declare_rx_stale_timeout,
        declare_max_linear,
        declare_max_angular,
        declare_min_lift,
        declare_max_lift,
        declare_lift_arrival_tol,
        declare_lift_arrival_timeout,
        declare_require_firmware_identity,
        declare_firmware_identity_stale,
        declare_gate_invalid_imu,
        serial_node,
    ])
