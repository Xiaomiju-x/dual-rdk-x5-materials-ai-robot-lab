from setuptools import find_packages, setup

package_name = 'my_robot_dashboard'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test', 'frontend', 'frontend.*']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/dashboard.launch.py']),
    ],
    install_requires=[
        'setuptools',
        'fastapi>=0.115',
        'uvicorn[standard]>=0.32',
        'websockets>=13',
        'pydantic>=2.9',
    ],
    zip_safe=True,
    maintainer='zhouLingxuan',
    maintainer_email='zhouLingxuan@todo.todo',
    description='NavCockpit dashboard (Phase 0 scaffold).',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dashboard_node = my_robot_dashboard.dashboard_node:main',
        ],
    },
)
