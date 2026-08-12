"""furnace_monitor.launch.py — 一键拉起 烧结炉监控全套.

启动 3 个节点:
    1. furnace_ocr_node       订 image, 发 /furnace_reading
    2. furnace_monitor_agent  订 /furnace_reading, 发 /alarm
    3. alert_dispatcher       订 /alarm, 派发 TTS/Email/微信/Log

参数:
    image_topic (str): OCR 输入图像 topic, 默认 /pt_camera/image_raw
                       开发期可改成 /lift_camera/image_raw 用 200W USB 测
    test_image_path (str): 离线模式: 不订 topic, 反复读这张图
    enable_tts/enable_email/enable_wechat (bool): 各通道总开关
    config_yaml (str): OCR ROI 标定 yaml 路径

用法:
    # 真实部署 (云台拉流后)
    ros2 launch my_robot_agents furnace_monitor.launch.py

    # 开发测试 (用静态图)
    ros2 launch my_robot_agents furnace_monitor.launch.py \
      test_image_path:=/tmp/furnace_test.jpg \
      enable_email:=false enable_wechat:=false

    # 用 200W USB 临时拍炉子调 ROI
    ros2 launch my_robot_agents furnace_monitor.launch.py \
      image_topic:=/lift_camera/image_raw
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    declare_use_bpu_ocr = DeclareLaunchArgument(
        'use_bpu_ocr', default_value=os.environ.get('EB_USE_BPU_OCR', 'false'),
        description='Round 4 Day 5-7: 用 BPU YOLOv8n LCD OCR 替代 OpenCV 七段法. EB_USE_BPU_OCR=true 开'
    )
    declare_bpu_bin = DeclareLaunchArgument(
        'bpu_bin_path', default_value='~/bpu_models/lcd_yolov8n.bin',
        description='BPU lcd_yolov8n.bin 路径'
    )
    declare_image_topic = DeclareLaunchArgument(
        'image_topic', default_value='/pt_camera/image_raw',
        description='OCR 输入图像 topic'
    )
    declare_test_image = DeclareLaunchArgument(
        'test_image_path', default_value='',
        description='离线测试: 用此静态图替代 topic 订阅; 留空则订 topic'
    )
    declare_config = DeclareLaunchArgument(
        'config_yaml', default_value='',
        description='OCR ROI 标定 yaml 路径; 空则用 default'
    )
    declare_rate = DeclareLaunchArgument('rate_hz', default_value='1.0')
    declare_tts = DeclareLaunchArgument('enable_tts', default_value='true')
    declare_email = DeclareLaunchArgument('enable_email', default_value='true')
    declare_wechat = DeclareLaunchArgument('enable_wechat', default_value='true')
    declare_log = DeclareLaunchArgument('enable_log', default_value='true')

    ocr_node_cv = Node(
        package='my_robot_agents',
        executable='furnace_ocr',
        name='furnace_ocr_node',
        output='screen',
        condition=UnlessCondition(LaunchConfiguration('use_bpu_ocr')),
        parameters=[{
            'image_topic': LaunchConfiguration('image_topic'),
            'rate_hz': LaunchConfiguration('rate_hz'),
            'config_yaml': LaunchConfiguration('config_yaml'),
            'test_image_path': LaunchConfiguration('test_image_path'),
        }]
    )

    ocr_node_bpu = Node(
        package='my_robot_agents',
        executable='furnace_ocr_bpu',
        name='furnace_ocr_bpu_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_bpu_ocr')),
        parameters=[{
            'image_topic': LaunchConfiguration('image_topic'),
            'rate_hz': LaunchConfiguration('rate_hz'),
            'config_yaml': LaunchConfiguration('config_yaml'),
            'test_image_path': LaunchConfiguration('test_image_path'),
            'bin_path': LaunchConfiguration('bpu_bin_path'),
        }]
    )

    monitor_node = Node(
        package='my_robot_agents',
        executable='furnace_monitor',
        name='furnace_monitor_agent',
        output='screen',
        parameters=[{
            'min_temp_c': -10.0,
            'max_temp_c': 1600.0,
            'deviation_threshold_c': 100.0,
            'deviation_sustain_s': 30.0,
            'power_off_sustain_s': 5.0,
            'alarm_cooldown_s': 60.0,
        }]
    )

    dispatcher_node = Node(
        package='my_robot_agents',
        executable='alert_dispatcher',
        name='alert_dispatcher',
        output='screen',
        parameters=[{
            'enable_tts': LaunchConfiguration('enable_tts'),
            'enable_email': LaunchConfiguration('enable_email'),
            'enable_wechat': LaunchConfiguration('enable_wechat'),
            'enable_log': LaunchConfiguration('enable_log'),
        }]
    )

    return LaunchDescription([
        declare_use_bpu_ocr, declare_bpu_bin,
        declare_image_topic, declare_test_image, declare_config, declare_rate,
        declare_tts, declare_email, declare_wechat, declare_log,
        ocr_node_cv, ocr_node_bpu, monitor_node, dispatcher_node,
    ])
