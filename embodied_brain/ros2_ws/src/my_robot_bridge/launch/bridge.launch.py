"""bridge.launch.py — 启动具身脑的"对外通信"全套.

启动 3 个节点:
    1. dispatch_server          DispatchTask action server (在 my_robot_agents 包)
    2. ai_brain_bridge          HTTP 拉 AI 脑 task → action client → dispatch_server
    3. telemetry_publisher      1Hz 周期 SystemTelemetry

参数:
    ai_brain_url: AI 脑 dashboard URL (默认 http://192.0.2.103:8888)
    poll_interval_s: 拉 AI 脑 dispatch_queue 周期
    stub_mode: dispatch_server 是否走假执行 (Phase 4 默认 true, Phase 6 改 false)
    use_nav2: stub_mode=false 时是否把导航阶段交给 /navigate_to_pose
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    declare_url = DeclareLaunchArgument(
        'ai_brain_url',
        default_value=os.environ.get('EB_AI_BRAIN_URL', 'http://192.0.2.103:8888'),
    )
    declare_poll = DeclareLaunchArgument('poll_interval_s', default_value='2.0')
    declare_report = DeclareLaunchArgument('report_interval_s', default_value='5.0')
    declare_stub = DeclareLaunchArgument(
        'stub_mode',
        default_value=os.environ.get('EB_DISPATCH_STUB_MODE', 'true'),
    )
    declare_use_nav2 = DeclareLaunchArgument(
        'use_nav2',
        default_value=os.environ.get('EB_USE_NAV2', 'false'),
    )
    declare_locations = DeclareLaunchArgument('locations_yaml', default_value='')
    declare_nav_action = DeclareLaunchArgument('nav2_action_name', default_value='/navigate_to_pose')
    declare_nav_wait = DeclareLaunchArgument('nav2_server_wait_s', default_value='5.0')
    declare_nav_timeout = DeclareLaunchArgument('nav2_goal_timeout_s', default_value='120.0')
    declare_allow_origin = DeclareLaunchArgument(
        'allow_origin_placeholder_targets',
        default_value='false',
    )
    declare_execute_actuators = DeclareLaunchArgument(
        'execute_pickup_actuators',
        default_value=os.environ.get('EB_EXECUTE_PICKUP_ACTUATORS', 'false'),
    )
    declare_pickup_height = DeclareLaunchArgument(
        'pickup_height_m', default_value=os.environ.get('EB_PICKUP_HEIGHT_M', '0.05'))
    declare_transport_height = DeclareLaunchArgument(
        'transport_height_m', default_value=os.environ.get('EB_TRANSPORT_HEIGHT_M', '0.08'))
    declare_place_height = DeclareLaunchArgument(
        'place_height_m', default_value=os.environ.get('EB_PLACE_HEIGHT_M', '0.05'))
    declare_use_physical_evidence_gate = DeclareLaunchArgument(
        'use_physical_evidence_gate',
        default_value=os.environ.get('EB_USE_PHYSICAL_EVIDENCE_GATE', 'false'),
    )
    declare_physical_evidence_mode = DeclareLaunchArgument(
        'physical_evidence_mode',
        default_value=os.environ.get('EB_PHYSICAL_EVIDENCE_MODE', 'disabled'),
    )
    declare_physical_evidence_service = DeclareLaunchArgument(
        'physical_evidence_service', default_value='/verify_physical_evidence')

    dispatch_server = Node(
        package='my_robot_agents',
        executable='dispatch_server',
        name='dispatch_server',
        output='screen',
        parameters=[{
            'stub_mode': ParameterValue(LaunchConfiguration('stub_mode'), value_type=bool),
            'use_nav2': ParameterValue(LaunchConfiguration('use_nav2'), value_type=bool),
            'locations_yaml': LaunchConfiguration('locations_yaml'),
            'nav2_action_name': LaunchConfiguration('nav2_action_name'),
            'nav2_server_wait_s': ParameterValue(LaunchConfiguration('nav2_server_wait_s'), value_type=float),
            'nav2_goal_timeout_s': ParameterValue(LaunchConfiguration('nav2_goal_timeout_s'), value_type=float),
            'allow_origin_placeholder_targets': ParameterValue(
                LaunchConfiguration('allow_origin_placeholder_targets'),
                value_type=bool,
            ),
            'execute_pickup_actuators': ParameterValue(
                LaunchConfiguration('execute_pickup_actuators'), value_type=bool),
            'pickup_height_m': ParameterValue(LaunchConfiguration('pickup_height_m'), value_type=float),
            'transport_height_m': ParameterValue(LaunchConfiguration('transport_height_m'), value_type=float),
            'place_height_m': ParameterValue(LaunchConfiguration('place_height_m'), value_type=float),
            'physical_evidence_mode': LaunchConfiguration('physical_evidence_mode'),
            'physical_evidence_service': LaunchConfiguration('physical_evidence_service'),
        }],
    )

    physical_evidence_gate = Node(
        package='my_robot_agents',
        executable='physical_evidence_gate',
        name='physical_evidence_gate',
        output='screen',
        parameters=[{
            'service_name': LaunchConfiguration('physical_evidence_service'),
        }],
        condition=IfCondition(LaunchConfiguration('use_physical_evidence_gate')),
    )

    bridge = Node(
        package='my_robot_bridge',
        executable='ai_brain_bridge',
        name='ai_brain_bridge',
        output='screen',
        parameters=[{
            'ai_brain_url': LaunchConfiguration('ai_brain_url'),
            'poll_interval_s': LaunchConfiguration('poll_interval_s'),
            'report_interval_s': LaunchConfiguration('report_interval_s'),
        }],
    )

    telemetry = Node(
        package='my_robot_agents',
        executable='telemetry_publisher',
        name='telemetry_publisher',
        output='screen',
        parameters=[{
            'rate_hz': 1.0,
            'ai_brain_url': LaunchConfiguration('ai_brain_url'),
        }],
    )

    return LaunchDescription([
        declare_url, declare_poll, declare_report,
        declare_stub, declare_use_nav2, declare_locations,
        declare_nav_action, declare_nav_wait, declare_nav_timeout, declare_allow_origin,
        declare_execute_actuators,
        declare_pickup_height, declare_transport_height, declare_place_height,
        declare_use_physical_evidence_gate, declare_physical_evidence_mode,
        declare_physical_evidence_service,
        physical_evidence_gate, dispatch_server, bridge, telemetry,
    ])
