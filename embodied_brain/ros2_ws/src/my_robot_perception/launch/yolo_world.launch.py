"""yolo_world.launch.py — 起 BPU 跑 YOLO-World 开放词检测.

订阅 /pt_camera/image_raw (默认), 发 /perception/yolo_world (ai_msgs/PerceptionTargets).
词表通过 -p texts 参数传, 运行时也可通过 /target_words std_msgs/String 实时换.

依赖:
    - hobot_yolo_world 包 (TROS, 含 BPU bin 模型 + 离线词表 embedding)
    - pt_camera_node 在发 /pt_camera/image_raw (Phase 0-5 已部署)

参数:
    cam_topic          订阅的图片 topic, 默认 /pt_camera/image_raw
    pub_topic          检测结果 topic, 默认 /perception/yolo_world
    texts              检测词表 (逗号分隔, ≤32 类, 跟 BPU bin input 1 的 1×32×512 对齐)
    score_threshold    置信度阈值, 默认 0.30
    dump_render_img    1=保存渲染 jpg 到 /tmp 调试

注意: yolo_world_node.cpp 把 model_file_name 写死成 "config/yolo_world.bin" (相对 cwd),
所以 ExecuteProcess 必须 cwd=hobot_yolo_world install/lib 目录.
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_prefix


# 默认 lab 检测词 (≤32 类, 不能超过 BPU bin input 1 的 1×32×512 上限)
DEFAULT_TEXTS = (
    "muffle furnace,vacuum pump,crucible,glass beaker,test tube rack,"
    "lab coat,pipette,plastic bottle,metal spatula,centrifuge tube,"
    "NIR phosphor sample bottle,Cr doped powder,rare earth oxide bottle,"
    "yttrium oxide jar,ceramic crucible,quartz tube,analytical balance,"
    "fume hood,magnetic stirrer,sintered sample,AprilTag marker"
)


def generate_launch_description():
    cam_topic_arg = DeclareLaunchArgument(
        'cam_topic', default_value='/pt_camera/image_raw',
        description='ROS image topic to subscribe (sensor_msgs/Image, BGR8)'
    )
    pub_topic_arg = DeclareLaunchArgument(
        'pub_topic', default_value='/hobot_yolo_world',
        description='Publish ai_msgs/PerceptionTargets here (default = hobot_yolo_world topic, EdgeSAM 订这个)'
    )
    texts_arg = DeclareLaunchArgument(
        'texts', default_value=DEFAULT_TEXTS,
        description='Detection vocabulary (comma-separated, <=32 classes)'
    )
    score_arg = DeclareLaunchArgument(
        'score_threshold', default_value='0.30',
        description='Detection score threshold'
    )
    dump_arg = DeclareLaunchArgument(
        'dump_render_img', default_value='0',
        description='1 = save rendered jpg to /tmp for debug'
    )

    yw_prefix = get_package_prefix('hobot_yolo_world')
    yw_cwd = os.path.join(yw_prefix, 'lib', 'hobot_yolo_world')
    yw_exe = os.path.join(yw_cwd, 'hobot_yolo_world')

    yolo_world_proc = ExecuteProcess(
        cmd=[
            yw_exe,
            '--ros-args',
            '-p', 'feed_type:=1',
            '-p', 'is_shared_mem_sub:=0',
            '-p', 'image_type:=0',
            '-p', ['ros_img_sub_topic_name:=', LaunchConfiguration('cam_topic')],
            '-p', ['msg_pub_topic_name:=', LaunchConfiguration('pub_topic')],
            '-p', ['texts:=', LaunchConfiguration('texts')],
            '-p', ['score_threshold:=', LaunchConfiguration('score_threshold')],
            '-p', ['dump_render_img:=', LaunchConfiguration('dump_render_img')],
        ],
        cwd=yw_cwd,
        output='screen'
    )

    return LaunchDescription([
        cam_topic_arg, pub_topic_arg, texts_arg, score_arg, dump_arg,
        yolo_world_proc,
    ])
