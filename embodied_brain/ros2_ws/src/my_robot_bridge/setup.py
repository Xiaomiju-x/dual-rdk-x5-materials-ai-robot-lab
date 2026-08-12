"""my_robot_bridge — AI brain ↔ embodied brain HTTP/DDS bridge."""
import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'my_robot_bridge'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.xml') + glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools', 'requests'],
    zip_safe=True,
    maintainer='zhouLingxuan',
    maintainer_email='zhouLingxuan@todo.todo',
    description='Cross-network bridge: HTTP poll AI brain dashboard:8888 + cyclonedds peers config.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ai_brain_bridge = my_robot_bridge.ai_brain_bridge:main',
        ],
    },
)
