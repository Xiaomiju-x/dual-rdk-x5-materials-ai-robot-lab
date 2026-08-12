"""audio_pipeline.launch.py — E1/E2/E3 麦阵全栈启动.

启动顺序:
    1. rnnoise_node   — E3: 实时降噪 (M260C 48kHz 4ch → 16kHz denoised)
    2. odas_node      — E1/E2: 波束成形 + DOA (M260C 4-mic)
    3. voice_input_node (可选, 需 --launch-arguments with_asr:=true)

Usage:
    ros2 launch my_robot_agents audio_pipeline.launch.py
    ros2 launch my_robot_agents audio_pipeline.launch.py alsa_device:=hw:3,0
"""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    alsa_arg = DeclareLaunchArgument('alsa_device', default_value='hw:2,0')
    alsa = LaunchConfiguration('alsa_device')

    rnnoise = Node(
        package='my_robot_agents',
        executable='rnnoise',
        name='rnnoise_node',
        parameters=[{
            'alsa_device': alsa,
            'n_channels': 4,
            'alsa_sr': 48000,
        }],
        output='screen',
    )

    odas = Node(
        package='my_robot_agents',
        executable='odas',
        name='odas_node',
        parameters=[{
            'alsa_device': alsa,
            'n_channels': 4,
            'alsa_sr': 16000,
            'use_odas_binary': False,
        }],
        output='screen',
    )

    return LaunchDescription([alsa_arg, rnnoise, odas])
