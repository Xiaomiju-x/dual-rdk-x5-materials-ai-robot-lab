#!/bin/bash
# install_third_party.sh — 在车载脑 X5 上一次性装好所有第三方 ROS2 包.
#
# 装的内容:
#   1. LD14 雷达驱动 (LD Robot 官方, GitHub clone)
#   2. ros2_astra_camera (Orbbec 官方, GitHub clone)
#   3. usb_cam (apt 装)
#   4. slam_toolbox + nav2 (apt 装, Phase 2 用)
#
# 用法:
#     ssh rdk@198.51.100.85
#     cd ~/ros2_ws/src/my_robot_drivers/scripts
#     bash install_third_party.sh
#
# 跑一次就够, 跑完后这些第三方包跟我们自己的包一起 colcon build.

set -e

WS_SRC="${ROS2_WS:-$HOME/ros2_ws}/src"

echo "==================================="
echo " Installing third-party deps to ${WS_SRC}"
echo "==================================="

# 0. apt 依赖
echo ""
echo "[0/4] apt 装 ROS2 现成包..."
sudo apt update
sudo apt install -y \
    python3-colcon-common-extensions python3-rosdep python3-vcstool \
    ros-humble-xacro liburdfdom-tools \
    ros-humble-robot-state-publisher \
    ros-humble-joint-state-publisher ros-humble-joint-state-publisher-gui \
    ros-humble-usb-cam \
    ros-humble-image-transport ros-humble-image-transport-plugins \
    ros-humble-image-publisher \
    ros-humble-camera-info-manager \
    ros-humble-cv-bridge \
    ros-humble-tf2-tools \
    ros-humble-rqt-tf-tree \
    ros-humble-slam-toolbox \
    ros-humble-nav2-bringup \
    ros-humble-nav2-smac-planner \
    ros-humble-depthimage-to-laserscan \
    libudev-dev libusb-1.0-0-dev libopenni2-dev libuvc-dev libuvc0

# setuptools 80.x 跟 colcon --symlink-install 不兼容 (Python 包 build 报 --editable not recognized).
# 降到 65.7 兼容版.
echo ""
echo "[0b/4] 降级 setuptools 到 65.7 (跟 colcon symlink 兼容)..."
sudo pip3 install "setuptools==65.7.0" --quiet 2>&1 | tail -2 || true

# 1. 雷达驱动 (LD14P / D300 同一驱动, LDROBOT 官方)
echo ""
echo "[1/4] LDROBOT lidar driver..."
if [ ! -d "${WS_SRC}/ldlidar_stl_ros2" ]; then
    cd "${WS_SRC}"
    git clone https://github.com/ldrobotSensorTeam/ldlidar_stl_ros2.git
else
    echo "  already cloned, skip"
fi
# ★ 必须打的补丁 (2026-06-10): src/demo.cpp 超时退出 — USB hub 掉电重枚举后
# 串口 fd 永久失效, 上游 SDK 只会无限报 DATA_TIME_OUT 不退出, respawn 救不回.
# 补丁版全文在仓库 embodied_brain/tools/ldlidar_demo.cpp, 重新 clone 后覆盖:
#     cp <repo>/embodied_brain/tools/ldlidar_demo.cpp ${WS_SRC}/ldlidar_stl_ros2/src/demo.cpp
if ! grep -q timeout_streak "${WS_SRC}/ldlidar_stl_ros2/src/demo.cpp" 2>/dev/null; then
    echo "  ⚠ demo.cpp 缺超时退出补丁! 用 embodied_brain/tools/ldlidar_demo.cpp 覆盖后重编"
fi

# 2. ros2_astra_camera (Orbbec 官方)
echo ""
echo "[2/4] ros2_astra_camera (Orbbec 官方)..."
if [ ! -d "${WS_SRC}/ros2_astra_camera" ]; then
    cd "${WS_SRC}"
    git clone https://github.com/orbbec/ros2_astra_camera.git
else
    echo "  already cloned, skip"
fi
# ★ 必须打的补丁 (2026-06-10): astra_pro.launch.xml 的 <node> 加
# respawn="true" respawn_delay="3.0" — USB 抖动后官方热插拔只报
# onDeviceConnected 不重新拉流 (僵死), 靠 sensor_watchdog pkill + respawn 自愈.
ASTRA_XML="${WS_SRC}/ros2_astra_camera/astra_camera/launch/astra_pro.launch.xml"
if [ -f "${ASTRA_XML}" ] && ! grep -q 'respawn="true"' "${ASTRA_XML}"; then
    sed -i 's|<node name="camera" pkg="astra_camera" exec="astra_camera_node" output="screen">|<node name="camera" pkg="astra_camera" exec="astra_camera_node" output="screen" respawn="true" respawn_delay="3.0">|' "${ASTRA_XML}"
    echo "  已打 astra respawn 补丁 (重编后生效)"
fi

# 装 Astra Pro 的 udev 规则 (Orbbec 官方版, 跟我们自己的 99-cameras.rules 互补)
if [ -f "${WS_SRC}/ros2_astra_camera/astra_camera/scripts/install.sh" ]; then
    cd "${WS_SRC}/ros2_astra_camera/astra_camera/scripts"
    sudo bash install.sh
fi

# 3. 装我们自己的 udev 规则
echo ""
echo "[3/4] 装 embodied_brain udev rules..."
sudo bash "${WS_SRC}/my_robot_drivers/udev/install_udev.sh"

# 4. rosdep
echo ""
echo "[4/4] rosdep install..."
cd "${WS_SRC}/.."
source /opt/ros/humble/setup.bash
rosdep update --rosdistro=humble || true
rosdep install --from-paths src --ignore-src -r -y || true

echo ""
echo "==================================="
echo " Done. 现在去 ws 根目录 colcon build:"
echo "   cd ${WS_SRC}/.. && colcon build --symlink-install --merge-install"
echo "==================================="
