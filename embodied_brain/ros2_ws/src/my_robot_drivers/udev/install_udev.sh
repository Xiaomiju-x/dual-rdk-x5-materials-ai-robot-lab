#!/bin/bash
# install_udev.sh — 把所有 udev 规则装到系统, 让 /dev/LD14 /dev/F407 /dev/astra_rgb /dev/lift_camera 生效
#
# 用法: 在 X5 上跑
#     cd ~/ros2_ws/src/my_robot_drivers/udev
#     sudo bash install_udev.sh
#
# 或者通过 deploy_to_car.sh 远程跑.

set -e

if [ "$EUID" -ne 0 ]; then
    echo "ERROR: must run as root (sudo bash install_udev.sh)"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET=/etc/udev/rules.d/

echo "Installing udev rules from ${SCRIPT_DIR} to ${TARGET}"

cp -v "${SCRIPT_DIR}/99-ld14-lidar.rules"  "${TARGET}/"
cp -v "${SCRIPT_DIR}/99-stm32-f407.rules"  "${TARGET}/"
cp -v "${SCRIPT_DIR}/99-cameras.rules"     "${TARGET}/"

# 加用户到 dialout (可读写 ttyACM/ttyUSB), 加 video (可读写 /dev/video*)
SUNRISE_USER="${SUDO_USER:-sunrise}"
usermod -a -G dialout "${SUNRISE_USER}" || true
usermod -a -G video   "${SUNRISE_USER}" || true

# 重新加载规则
udevadm control --reload-rules
udevadm trigger

echo ""
echo "Done. 拔插一次设备后, 检查:"
echo "  ls -l /dev/LD14 /dev/F407 /dev/astra_rgb /dev/lift_camera"
echo ""
echo "如果 /dev/lift_camera 没出来, lsusb 看 200W 相机的 vid:pid, 改 99-cameras.rules"
