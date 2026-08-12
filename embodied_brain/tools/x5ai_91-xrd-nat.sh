#!/bin/bash
# xrd NAT for tablet (2026-06-10): AI 脑给 overlay 网段 (192.0.2.0/24) 的
# 无路由设备 (小米平板 .50, 网关指向本机 .103) 做 MASQUERADE 借道上外网.
# 部署位: x5-ai /etc/NetworkManager/dispatcher.d/91-xrd-nat
# 守护: /etc/cron.d/xrd-overlay 每分钟以 "wlan0 up" 参数重跑 (幂等).
[ "$2" = "up" ] || exit 0
[ "$1" = "wlan0" ] || exit 0

sysctl -qw net.ipv4.ip_forward=1

# 出 wlan0 的 31 网段流量伪装成本机 DHCP 地址 (跟着 K70 网段漂, 免维护)
iptables -t nat -C POSTROUTING -s 192.0.2.0/24 ! -d 192.0.2.0/24 -o wlan0 -j MASQUERADE 2>/dev/null \
  || iptables -t nat -A POSTROUTING -s 192.0.2.0/24 ! -d 192.0.2.0/24 -o wlan0 -j MASQUERADE

exit 0
