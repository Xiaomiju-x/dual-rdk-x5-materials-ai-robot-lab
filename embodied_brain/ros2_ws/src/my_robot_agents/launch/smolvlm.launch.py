"""smolvlm.launch.py — SmolVLM-256M 视觉语言模型服务节点 (Round 4 Day 8-15 C1).

提供 /vlm_query 服务. 接 200W USB cam (升降台) 或 Astra Pro 默认.

后端切换:
    backend = 'hybrid'   (default, 33s/query, 语义稳, 答辩主线)
    backend = 'full_bpu' (14s/query, 但 INT8 PTQ 在 30L Llama 上语义有损)

依赖:
    - HF transformers + torch CPU (~/smolvlm_256m/)
    - hobot_dnn (full_bpu / hybrid 视觉部分)
    - vlm 调用脚本 在 /tmp/smolvlm_x5_hybrid.py + /tmp/smolvlm_x5_full_bpu.py
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    declare_backend = DeclareLaunchArgument(
        'backend', default_value='hybrid',
        description='hybrid (CPU decoder, 语义稳) | full_bpu (全 BPU, 但 INT8 PTQ 精度需 QAT)'
    )
    declare_model_dir = DeclareLaunchArgument(
        'model_dir', default_value='/home/rdk/smolvlm_256m',
        description='HF SmolVLM-256M weights dir'
    )
    declare_image_topic = DeclareLaunchArgument(
        'image_topic_default', default_value='/lift_camera/image_raw',
        description='默认抓帧 topic, 200W USB cam 升降台'
    )

    return LaunchDescription([
        declare_backend, declare_model_dir, declare_image_topic,
        Node(
            package='my_robot_agents',
            executable='smolvlm',
            name='smolvlm_node',
            output='screen',
            parameters=[{
                'backend': LaunchConfiguration('backend'),
                'model_dir': LaunchConfiguration('model_dir'),
                'image_topic_default': LaunchConfiguration('image_topic_default'),
            }],
        ),
    ])
