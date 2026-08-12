"""edgesam.launch.py — 起 BPU 跑 EdgeSAM 像素级分割.

订阅:
    /pt_camera/image_raw                  (sensor_msgs/Image, BGR8)
    /perception/yolo_world                (ai_msgs/PerceptionTargets, Day 1 yolo_world 出框)

发布:
    /perception/segmentation/edgesam      (ai_msgs/PerceptionTargets, 含 mask 字段)

工作模式 (is_regular_box):
    0 = Dynamic   订阅 yolo_world 输出的 box, 对每个框出 mask (主用途)
    1 = Fixed     图中心固定一个 box, 出 mask (调试用)

模型:
    encoder/decoder = 512×512 (默认, 速度快) 或 1024×1024 (精度高, 速度慢一档)

依赖: mono_edgesam 包 (D-Robotics 官方, 2026-03), TROS Humble.
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_prefix


def generate_launch_description():
    cam_topic_arg = DeclareLaunchArgument(
        'cam_topic', default_value='/pt_camera/image_raw',
        description='Image topic (sensor_msgs/Image, BGR8)'
    )
    box_sub_topic_arg = DeclareLaunchArgument(
        'box_sub_topic', default_value='/hobot_yolo_world',
        description='Detection box topic (ai_msgs/PerceptionTargets, Day 1 yolo_world default pub)'
    )
    mask_pub_topic_arg = DeclareLaunchArgument(
        'mask_pub_topic', default_value='/perception/segmentation/edgesam',
        description='Output segmentation topic'
    )
    is_regular_box_arg = DeclareLaunchArgument(
        'is_regular_box', default_value='0',
        description='0 = Dynamic (subscribe yolo_world box) | 1 = Fixed center box'
    )
    encoder_arg = DeclareLaunchArgument(
        'encoder_model', default_value='edgesam_encoder_512.bin',
        description='edgesam_encoder_512.bin (fast) | edgesam_encoder_1024.bin (higher quality)'
    )
    decoder_arg = DeclareLaunchArgument(
        'decoder_model', default_value='edgesam_decoder_512.bin',
        description='Match encoder resolution'
    )
    dump_arg = DeclareLaunchArgument(
        'dump_render_img', default_value='0',
        description='1 = save rendered jpg to /tmp for debug'
    )

    es_prefix = get_package_prefix('mono_edgesam')
    es_cwd = os.path.join(es_prefix, 'lib', 'mono_edgesam')
    es_exe = os.path.join(es_cwd, 'mono_edgesam')

    edgesam_proc = ExecuteProcess(
        cmd=[
            es_exe,
            '--ros-args',
            '-p', 'feed_type:=1',
            '-p', 'is_shared_mem_sub:=0',
            '-p', ['ros_img_sub_topic_name:=', LaunchConfiguration('cam_topic')],
            '-p', ['ai_msg_sub_topic_name:=', LaunchConfiguration('box_sub_topic')],
            '-p', ['ai_msg_pub_topic_name:=', LaunchConfiguration('mask_pub_topic')],
            '-p', ['is_regular_box:=', LaunchConfiguration('is_regular_box')],
            '-p', ['encoder_model_file_name:=config/', LaunchConfiguration('encoder_model')],
            '-p', ['decoder_model_file_name:=config/', LaunchConfiguration('decoder_model')],
            '-p', ['dump_render_img:=', LaunchConfiguration('dump_render_img')],
        ],
        cwd=es_cwd,
        output='screen'
    )

    return LaunchDescription([
        cam_topic_arg, box_sub_topic_arg, mask_pub_topic_arg,
        is_regular_box_arg, encoder_arg, decoder_arg, dump_arg,
        edgesam_proc,
    ])
