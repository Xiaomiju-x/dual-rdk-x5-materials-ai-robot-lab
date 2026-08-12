"""smolvlm_full.launch.py — SmolVLM C1 全栈一键启动 (Round 4 Day 15).

包含 4 个节点:
    1. smolvlm_node      — /vlm_query 服务 (lazy 加载, 首次 query 装 ~600MB HF)
    2. dispatch_server   — DispatchTask action server (含 'observe' task type 调 VLM)
    3. vlm_voice_relay   — ASR 触发词 → /vlm_query → TTS 答案
    4. (可选) voice_input + voice_output — 真语音麦克风/扬声器

参数:
    enable_voice    (bool, default False) — 起 voice_input + voice_output (硬件就位再开)
    image_topic     (string, default /lift_camera/image_raw) — VLM 抓帧 topic

示例:
    # 仅文本/HTTP 模式 (没麦克风也能演示):
    ros2 launch my_robot_agents smolvlm_full.launch.py

    # 带语音 (M260C 麦克风 + 扬声器):
    ros2 launch my_robot_agents smolvlm_full.launch.py enable_voice:=true
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    declare_voice = DeclareLaunchArgument(
        'enable_voice', default_value='false',
        description='起 voice_input + voice_output (麦克风/扬声器硬件就位才开)'
    )
    declare_image_topic = DeclareLaunchArgument(
        'image_topic', default_value='/lift_camera/image_raw',
        description='VLM 抓帧 topic (默认 200W USB 升降台 cam)'
    )
    declare_backend = DeclareLaunchArgument(
        'backend', default_value='hybrid',
        description='hybrid (vision CPU + decoder CPU, 33s/query 语义稳) | full_bpu (INT8 PTQ 精度需 QAT)'
    )

    return LaunchDescription([
        declare_voice, declare_image_topic, declare_backend,

        # 1. SmolVLM /vlm_query 服务
        Node(
            package='my_robot_agents',
            executable='smolvlm',
            name='smolvlm_node',
            output='screen',
            parameters=[{
                'backend': LaunchConfiguration('backend'),
                'model_dir': '/home/rdk/smolvlm_256m',
                'image_topic_default': LaunchConfiguration('image_topic'),
            }],
        ),

        # 2. DispatchTask action server (含 observe 模式调 VLM)
        Node(
            package='my_robot_agents',
            executable='dispatch_server',
            name='dispatch_server',
            output='screen',
            parameters=[{
                'stub_mode': True,
                'vlm_service': '/vlm_query',
                'vlm_image_topic': LaunchConfiguration('image_topic'),
                'vlm_max_tokens': 25,
                'vlm_timeout_s': 90.0,
            }],
        ),

        # 3. ASR ↔ VLM ↔ TTS relay
        Node(
            package='my_robot_agents',
            executable='vlm_voice_relay',
            name='vlm_voice_relay',
            output='screen',
            parameters=[{
                'asr_topic': '/asr/text',
                'tts_topic': '/tts/say',
                'vlm_service': '/vlm_query',
                'image_topic': LaunchConfiguration('image_topic'),
                'max_new_tokens': 30,
            }],
        ),

        # 4. (可选) voice_input + voice_output
        GroupAction(
            condition=IfCondition(LaunchConfiguration('enable_voice')),
            actions=[
                Node(
                    package='my_robot_agents',
                    executable='voice_input',
                    name='voice_input_node',
                    output='screen',
                ),
                Node(
                    package='my_robot_agents',
                    executable='voice_output',
                    name='voice_output_node',
                    output='screen',
                ),
            ],
        ),
    ])
