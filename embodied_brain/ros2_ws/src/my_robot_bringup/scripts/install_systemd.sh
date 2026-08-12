#!/bin/bash
# install_systemd.sh — 把 embodied_brain.service 装到 X5 systemd, 开机自启
#
# 在车载脑 X5 上跑:
#     cd ~/ros2_ws/src/my_robot_bringup/scripts
#     sudo bash install_systemd.sh
#
# 装完管理命令:
#     sudo systemctl status embodied_brain      # 看状态
#     sudo systemctl start embodied_brain       # 手动启
#     sudo systemctl stop embodied_brain        # 手动停
#     sudo systemctl restart embodied_brain     # 重启
#     journalctl -u embodied_brain -f           # 看实时日志
#     sudo systemctl disable embodied_brain     # 关开机自启

set -e

if [ "$EUID" -ne 0 ]; then
    echo "ERROR: must run as root (sudo bash install_systemd.sh)"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SVC_FILE="${SCRIPT_DIR}/../config/embodied_brain.service"

if [ ! -f "${SVC_FILE}" ]; then
    echo "ERROR: ${SVC_FILE} not found"
    exit 2
fi

echo "Installing ${SVC_FILE} → /etc/systemd/system/embodied_brain.service"
cp -v "${SVC_FILE}" /etc/systemd/system/embodied_brain.service
chmod 644 /etc/systemd/system/embodied_brain.service

echo "Reloading systemd..."
systemctl daemon-reload

# 默认启用开机自启 + 立即启动
read -p "立即启动 embodied_brain.service? (y/n, 默认 y): " yn
yn=${yn:-y}
if [ "$yn" = "y" ] || [ "$yn" = "Y" ]; then
    systemctl enable embodied_brain
    systemctl start embodied_brain
    sleep 3
    systemctl status embodied_brain --no-pager | head -20
    echo ""
    echo "实时日志 (Ctrl-C 退):"
    echo "    journalctl -u embodied_brain -f"
else
    systemctl enable embodied_brain
    echo "已设开机自启, 但当前未启动. 手动启:"
    echo "    sudo systemctl start embodied_brain"
fi
