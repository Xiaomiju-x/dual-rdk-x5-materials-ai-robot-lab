#!/bin/bash
# /etc/NetworkManager/dispatcher.d/93-xrd-ula  (车载脑 x5-car, 2026-06-11)
# ULA 固定 v6 门牌: [fd00:31::85]:8890 = NavCockpit 永久入口.
# 前缀 fd00:31::/64 由 AI 脑 radvd 通告 (本机不跑 radvd, 只挂固定地址).
# 配套: xrd-v6-8890.service = socat TCP6-LISTEN:8890,ipv6only=1 -> 127.0.0.1:8890
#       /etc/cron.d/xrd-ula 每分钟兜底
[ "$1" = "wlan0" ] || exit 0
case "$2" in up|dhcp4-change|dhcp6-change|connectivity-change) ;; *) exit 0;; esac
ip -6 addr replace fd00:31::85/64 dev wlan0
exit 0
