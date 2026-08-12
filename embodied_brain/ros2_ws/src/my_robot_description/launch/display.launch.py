"""display.launch.py — 启动 RSP + JSP-gui + rviz2, 用于可视化 URDF 模型.

仅做 URDF 检查/调试用. 实际部署在车载脑时, RSP 单独由 bringup launch 启.

用法:
    ros2 launch my_robot_description display.launch.py
    ros2 launch my_robot_description display.launch.py use_xiaomi_pt:=false
    ros2 launch my_robot_description display.launch.py use_jsp_gui:=false   # 不需要拖动滑条
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare('my_robot_description')

    # 参数: URDF 路径, 是否启 JSP-gui, 云台是否小米
    default_urdf = PathJoinSubstitution([pkg_share, 'urdf', 'my_robot.urdf.xacro'])
    default_rviz = PathJoinSubstitution([pkg_share, 'rviz', 'display.rviz'])

    declare_urdf = DeclareLaunchArgument(
        'urdf', default_value=default_urdf,
        description='URDF xacro 文件路径'
    )
    declare_use_jsp_gui = DeclareLaunchArgument(
        'use_jsp_gui', default_value='true',
        description='是否启动 joint_state_publisher_gui 拖动滑条'
    )
    declare_use_xiaomi_pt = DeclareLaunchArgument(
        'use_xiaomi_pt', default_value='true',
        description='云台用小米 (true) 还是备选 S20 (false)'
    )
    declare_rviz_config = DeclareLaunchArgument(
        'rviz_config', default_value=default_rviz,
        description='rviz2 配置文件路径'
    )

    # xacro 处理 → robot_description 字符串
    robot_description_content = ParameterValue(
        Command([
            'xacro ',
            LaunchConfiguration('urdf'),
            ' use_xiaomi_pt:=',
            LaunchConfiguration('use_xiaomi_pt'),
        ]),
        value_type=str,
    )

    rsp_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description_content,
            'publish_frequency': 30.0,
        }]
    )

    # JSP-gui (拖滑条) — 调试 URDF 时打开
    jsp_gui_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_jsp_gui'))
    )

    # JSP (无 gui, 静态发布) — 实车上跑这个
    jsp_node = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        output='screen',
        condition=UnlessCondition(LaunchConfiguration('use_jsp_gui'))
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        arguments=['-d', LaunchConfiguration('rviz_config')]
    )

    return LaunchDescription([
        declare_urdf,
        declare_use_jsp_gui,
        declare_use_xiaomi_pt,
        declare_rviz_config,
        rsp_node,
        jsp_gui_node,
        jsp_node,
        rviz_node,
    ])
