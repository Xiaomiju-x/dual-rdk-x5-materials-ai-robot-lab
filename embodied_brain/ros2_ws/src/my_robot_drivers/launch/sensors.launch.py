"""sensors.launch.py — 一键启动具身脑全部硬件驱动.

包含:
    1. serial_f407_node     底盘 + 升降台 + 电磁铁 (USB-TTL → STM32F407)
    2. lidar.launch.py      LD14 /scan_raw -> scan_self_filter -> /scan
    3. depth_camera.launch.py  Astra Pro 深度相机 → /depth_camera/* /points
    4. lift_camera.launch.py   200W USB 升降台相机 → /lift_camera/image_raw
    5. pt_camera.launch.py     K3 USB 前向相机 (替代米家云台) → /pt_camera/image_raw

参数 (传给各子 launch):
    use_lidar           (bool): 默认 true
    use_depth_camera    (bool): 默认 true
    use_lift_camera     (bool): 默认 true
    use_pt_camera       (bool): 默认 false  (Phase 8 K3 USB cam 接好后改 true)
    use_serial_f407     (bool): 默认 true (硬件没接的时候关掉, 否则 fatal)
    cmd_vel_topic       (str): F407 执行器速度输入；安全栈可绑定 /cmd_vel_safe
    pt_camera_source    (str):  默认 /dev/PT_CAM  (K3 USB cam udev symlink)

如果某个硬件没接好, 关掉对应的开关启 launch, 其他还能用.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare("my_robot_drivers")
    pkg_agents = FindPackageShare("my_robot_agents")

    declare_use_lidar = DeclareLaunchArgument("use_lidar", default_value="true", description="LD14 雷达")
    declare_use_depth = DeclareLaunchArgument(
        "use_depth_camera", default_value="true", description="Astra Pro 深度相机"
    )
    declare_use_lift_cam = DeclareLaunchArgument(
        "use_lift_camera", default_value="true", description="200W 升降台相机"
    )
    declare_use_pt_cam = DeclareLaunchArgument(
        "use_pt_camera", default_value="false", description="K3 USB cam (替代小米云台). Phase 8 接好后改 true"
    )
    declare_pt_cam_source = DeclareLaunchArgument(
        "pt_camera_source", default_value="/dev/PT_CAM", description="K3 USB cam 设备路径 (udev symlink)"
    )
    declare_use_serial = DeclareLaunchArgument(
        "use_serial_f407", default_value="true", description="STM32F407 串口"
    )
    declare_cmd_vel_topic = DeclareLaunchArgument(
        "cmd_vel_topic", default_value="/cmd_vel", description="F407 执行器速度输入 topic"
    )
    declare_f407_odom_topic = DeclareLaunchArgument(
        "f407_odom_topic", default_value="/odom", description="F407 odometry output topic"
    )
    declare_f407_publish_tf = DeclareLaunchArgument(
        "f407_publish_tf",
        default_value="true",
        description="F407 publishes odom->base TF; disable when EKF owns TF",
    )

    # 各子 launch 路径
    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([pkg_share, "launch", "lidar.launch.py"])),
        condition=IfCondition(LaunchConfiguration("use_lidar")),
    )

    depth_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([pkg_share, "launch", "depth_camera.launch.py"])),
        condition=IfCondition(LaunchConfiguration("use_depth_camera")),
    )

    lift_cam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([pkg_share, "launch", "lift_camera.launch.py"])),
        condition=IfCondition(LaunchConfiguration("use_lift_camera")),
    )

    serial_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([pkg_share, "launch", "serial_f407.launch.py"])),
        launch_arguments={
            "cmd_vel_topic": LaunchConfiguration("cmd_vel_topic"),
            "odom_topic": LaunchConfiguration("f407_odom_topic"),
            "publish_tf": LaunchConfiguration("f407_publish_tf"),
        }.items(),
        condition=IfCondition(LaunchConfiguration("use_serial_f407")),
    )

    pt_cam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([pkg_agents, "launch", "pt_camera.launch.py"])),
        launch_arguments={
            "source": LaunchConfiguration("pt_camera_source"),
        }.items(),
        condition=IfCondition(LaunchConfiguration("use_pt_camera")),
    )

    return LaunchDescription(
        [
            declare_use_lidar,
            declare_use_depth,
            declare_use_lift_cam,
            declare_use_pt_cam,
            declare_pt_cam_source,
            declare_use_serial,
            declare_cmd_vel_topic,
            declare_f407_odom_topic,
            declare_f407_publish_tf,
            GroupAction(
                [
                    lidar_launch,
                    depth_launch,
                    lift_cam_launch,
                    pt_cam_launch,
                    serial_launch,
                ]
            ),
        ]
    )
