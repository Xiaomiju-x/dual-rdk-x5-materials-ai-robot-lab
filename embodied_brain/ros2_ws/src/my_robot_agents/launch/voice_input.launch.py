"""voice_input.launch.py — 起本地中文 ASR 节点 (Round 4 Day 4 B1).

订阅 M260C 麦克风阵列 (hw:2,0), 发 /asr/text (std_msgs/String).
依赖: sherpa-onnx + pyalsaaudio + 模型在 ~/asr_models/.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    declare_alsa = DeclareLaunchArgument(
        'alsa_device', default_value='hw:2,0',
        description='ALSA capture device (M260C mic array = hw:2,0)'
    )
    declare_topic = DeclareLaunchArgument(
        'topic', default_value='/asr/text',
        description='Output topic for recognized text'
    )
    declare_threads = DeclareLaunchArgument(
        'num_threads', default_value='4',
        description='SenseVoice ONNX num_threads'
    )

    return LaunchDescription([
        declare_alsa, declare_topic, declare_threads,
        Node(
            package='my_robot_agents',
            executable='voice_input',
            name='voice_input_node',
            output='screen',
            parameters=[{
                'alsa_device': LaunchConfiguration('alsa_device'),
                'topic': LaunchConfiguration('topic'),
                'num_threads': LaunchConfiguration('num_threads'),
            }],
        ),
    ])
