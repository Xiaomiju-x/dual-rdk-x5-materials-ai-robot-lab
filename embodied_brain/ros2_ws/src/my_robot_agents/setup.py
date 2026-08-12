"""my_robot_agents — Python task agents for embodied_brain."""
import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'my_robot_agents'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zhouLingxuan',
    maintainer_email='zhouLingxuan@todo.todo',
    description='Python task agents (furnace_monitor / pickup / dispatcher) for embodied_brain.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'fake_odom = my_robot_agents.fake_odom_node:main',
            'furnace_ocr = my_robot_agents.furnace_ocr_node:main',
            'furnace_ocr_bpu = my_robot_agents.furnace_ocr_bpu_node:main',
            'furnace_monitor = my_robot_agents.furnace_monitor_agent:main',
            'alert_dispatcher = my_robot_agents.alert_dispatcher:main',
            'telemetry_publisher = my_robot_agents.telemetry_publisher:main',
            'dispatch_server = my_robot_agents.dispatch_server:main',
            'command_interpreter = my_robot_agents.command_interpreter_node:main',
            'pt_camera = my_robot_agents.pt_camera_node:main',
            'location_visualizer = my_robot_agents.location_visualizer_node:main',
            'voice_input = my_robot_agents.voice_input_node:main',
            'voice_output = my_robot_agents.voice_output_node:main',
            'smolvlm = my_robot_agents.smolvlm_node:main',
            'vlm_voice_relay = my_robot_agents.vlm_voice_relay:main',
            'xfeat = my_robot_agents.xfeat_node:main',
            'mppi = my_robot_agents.mppi_node:main',
            'bottle_ocr_bpu = my_robot_agents.bottle_ocr_bpu_node:main',
            'rnnoise = my_robot_agents.rnnoise_node:main',
            'odas = my_robot_agents.odas_node:main',
            'cockpit_bridge = my_robot_agents.cockpit_bridge:main',
            'physical_evidence_gate = my_robot_agents.physical_evidence_gate:main',
            'physical_sensor_evidence_bridge = my_robot_agents.physical_sensor_evidence_bridge:main',
            # Phase 6+ 填:
            # 'pickup_agent = my_robot_agents.pickup_agent:main',
        ],
    },
)
