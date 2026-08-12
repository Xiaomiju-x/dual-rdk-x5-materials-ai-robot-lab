"""depth_camera.launch.py — 启动 Orbbec Astra Pro 深度相机.

策略: 不重新发明轮子, 直接 include Orbbec 官方 astra_pro.launch.xml.
理由: Astra Pro 是双分支硬件 (深度走 OpenNI2 vid=2bc5:0403, RGB 走 UVC vid=2bc5:0501),
官方 launch 把 use_uvc_camera + uvc_vendor_id + 一堆 sensor 模式参数都预设好,
我们只覆盖 camera_name 让 topic 进 /depth_camera/* 命名空间.

2026-06-10: RGB (UVC) 分支默认关闭 — X5 上 libuvc 抢 uvcvideo 内核驱动的接口失败
("attempt to claim already-claimed interface 1") 后直接段错误 (exit -11), 整个驱动
进程带着深度流一起死. 建图/避障只需要深度; RGB 感知归 K3 USB cam (/pt_camera).
要 RGB 时显式传 enable_color:=true use_uvc_camera:=true 再调.

输出 topic:
    /depth_camera/depth/image_raw       (sensor_msgs/Image, mono16, 0.6-8m)
    /depth_camera/depth/camera_info
    /depth_camera/depth/points          (sensor_msgs/PointCloud2, 给 Nav2 voxel layer)
    /depth_camera/ir/image_raw          (mono16 红外)
    /depth_camera/ir/camera_info
    /depth_camera/color/*               (默认不出 — enable_color=false, 见上)

注意: 官方驱动会发 TF (camera_link → depth_camera_optical / color_optical / ir_optical).
我们的 URDF 里 depth_camera_optical_frame 已经存在, 名字冲突时以官方驱动为准.
我们的 RSP 发的那条只在没插相机时占位用; 插上相机后驱动会覆盖, 不冲突.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    declare_camera_name = DeclareLaunchArgument(
        'camera_name', default_value='depth_camera',
        description='相机命名空间 (topic 前缀)'
    )
    declare_color_w = DeclareLaunchArgument('color_width',  default_value='640')
    declare_color_h = DeclareLaunchArgument('color_height', default_value='480')
    declare_color_fps = DeclareLaunchArgument('color_fps',  default_value='30')
    declare_depth_w = DeclareLaunchArgument('depth_width',  default_value='640')
    declare_depth_h = DeclareLaunchArgument('depth_height', default_value='480')
    declare_depth_fps = DeclareLaunchArgument('depth_fps',  default_value='30')
    declare_enable_pc = DeclareLaunchArgument(
        'enable_point_cloud',
        # 2026-06-11 第 0 期 CPU 优化: 30Hz 点云生成是 astra 节点 ~92% CPU 的大头,
        # 而当前没有任何节点消费 /depth_camera/depth/points (SLAM 用雷达,
        # 避障副 /scan_depth 走 depth image 不走点云). Nav2 voxel layer 上线时
        # 环境变量 EB_DEPTH_POINTCLOUD=true 打开, 不用改代码.
        default_value=os.environ.get('EB_DEPTH_POINTCLOUD', 'false'),
        description='点云 (给 Nav2 voxel layer, 默认关省 CPU)'
    )
    declare_enable_color = DeclareLaunchArgument(
        'enable_color', default_value='false',
        description='RGB 流. X5 上 libuvc 段错误连带拖死深度流, 默认关 (见文件头)'
    )
    declare_use_uvc = DeclareLaunchArgument(
        'use_uvc_camera', default_value='false',
        description='Astra Pro 的 RGB 走独立 UVC 设备 (2bc5:0501), 跟 enable_color 一起开关'
    )

    astra_official = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('astra_camera'),
                'launch',
                'astra_pro.launch.xml',
            ])
        ),
        launch_arguments={
            'camera_name':         LaunchConfiguration('camera_name'),
            'color_width':         LaunchConfiguration('color_width'),
            'color_height':        LaunchConfiguration('color_height'),
            'color_fps':           LaunchConfiguration('color_fps'),
            'depth_width':         LaunchConfiguration('depth_width'),
            'depth_height':        LaunchConfiguration('depth_height'),
            'depth_fps':           LaunchConfiguration('depth_fps'),
            'enable_point_cloud':  LaunchConfiguration('enable_point_cloud'),
            'enable_color':        LaunchConfiguration('enable_color'),
            'use_uvc_camera':      LaunchConfiguration('use_uvc_camera'),
        }.items(),
    )

    return LaunchDescription([
        declare_camera_name,
        declare_color_w, declare_color_h, declare_color_fps,
        declare_depth_w, declare_depth_h, declare_depth_fps,
        declare_enable_pc,
        declare_enable_color, declare_use_uvc,
        astra_official,
    ])
