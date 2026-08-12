"""voice_output.launch.py — 起本地中文 Piper TTS 节点 (Round 4 Day 4 B2).

订阅 /tts/say (std_msgs/String), 走 piper VITS 合成 → aplay plughw:0,0 (M260C 扬声器).
依赖: piper-tts pip + zh_CN-huayan-medium 模型在 ~/tts_models/.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    declare_alsa = DeclareLaunchArgument(
        'alsa_device', default_value='plughw:0,0',
        description='ALSA playback device (M260C speaker = card 0)'
    )
    declare_topic = DeclareLaunchArgument(
        'topic', default_value='/tts/say',
        description='Input topic for text-to-speak'
    )
    declare_norm = DeclareLaunchArgument(
        'normalize_chem', default_value='true',
        description='Normalize chemical formulas (Y3Al5O12 → 钇铝石榴石)'
    )

    return LaunchDescription([
        declare_alsa, declare_topic, declare_norm,
        Node(
            package='my_robot_agents',
            executable='voice_output',
            name='voice_output_node',
            output='screen',
            parameters=[{
                'alsa_device': LaunchConfiguration('alsa_device'),
                'topic': LaunchConfiguration('topic'),
                'normalize_chem': LaunchConfiguration('normalize_chem'),
            }],
        ),
    ])
