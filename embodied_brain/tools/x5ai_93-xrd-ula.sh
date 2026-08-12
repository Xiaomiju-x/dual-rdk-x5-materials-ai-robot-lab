#!/bin/bash
# /etc/NetworkManager/dispatcher.d/93-xrd-ula  (AI 脑 x5-ai, 2026-06-11)
# ULA 平台固定 v6 门牌: [fd00:31::103]:8888 = 平板/任意客户端的永久平台入口.
# 背景: MIUI 平板静态 IPv4 配置会让 netd 卡死 (v4 单播全静默, 连 ARP 都不发,
#       开关 WLAN/忘记网络/重启全救不回), 而 v6 邻居发现走组播, BE6500 中继组播全通.
#       平板保持 DHCP 零配置, 经 SLAAC 自动获得 fd00:31::/64 地址直达本机.
# 配套 (均已部署在 x5-ai):
#   /etc/sysctl.d/91-xrd-radvd.conf   accept_ra=2 必须先于 forwarding=1 (否则丢自身 v6 默认路由)
#   /etc/radvd.conf                   通告 fd00:31::/64, AdvDefaultLifetime 0 = 不当默认网关
#   xrd-v6-8888.service               socat TCP6-LISTEN:8888,ipv6only=1 -> 127.0.0.1:8888
#                                     (Flask 绑 0.0.0.0 是 v4-only, 必须桥)
#   /etc/cron.d/xrd-ula               每分钟兜底重跑本脚本
[ "$1" = "wlan0" ] || exit 0
case "$2" in up|dhcp4-change|dhcp6-change|connectivity-change) ;; *) exit 0;; esac
sysctl -qw net.ipv6.conf.wlan0.accept_ra=2 net.ipv6.conf.all.forwarding=1
ip -6 addr replace fd00:31::103/64 dev wlan0
systemctl is-active radvd >/dev/null || systemctl restart radvd
exit 0
