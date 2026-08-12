#!/bin/bash
# 92-xrd-dhcp-guard — DHCP 租约丢失自愈 (2026-06-10).
#
# 背景: K70 手机热点偶发不应答 DHCP 续租, NM "no lease" 后把动态地址+默认路由
# 一起撤掉, 只剩 overlay 静态 IP (局域网互访活着, 外网/跨段回包全断), 且 NM
# 不会自己恢复 (AI 脑 21:55 掉租约, 1 小时没自愈, 实测).
#
# 逻辑: wlan0 已关联且有 inet 但没有 dynamic 地址 → 距上次自愈 >10min 则
# systemd-run 脱会话重新激活连接 (con up 会重走 DHCP + 触发 dispatcher 重钉邻居).
# 限速 10min 防 K70 真宕机时每分钟 bounce 把 overlay 也搞断.
#
# 部署位: x5-ai / x5-car 的 /etc/NetworkManager/dispatcher.d/92-xrd-dhcp-guard
# 由 /etc/cron.d/xrd-overlay 每分钟以 "wlan0 up" 参数调用.
[ "$2" = "up" ] || exit 0
[ "$1" = "wlan0" ] || exit 0

# 有 dynamic 地址 = 租约健康, 不用管
ip -4 addr show wlan0 2>/dev/null | grep -q ' dynamic ' && exit 0
# 连 inet 都没有 = WiFi 没关联, NM 自己会重连, 不掺和
ip -4 addr show wlan0 2>/dev/null | grep -q 'inet ' || exit 0

STAMP=/run/xrd-dhcp-guard.stamp
now=$(date +%s)
last=$(cat "$STAMP" 2>/dev/null || echo 0)
[ $((now - last)) -lt 600 ] && exit 0
echo "$now" > "$STAMP"

PROFILE=$(nmcli -t -f NAME,DEVICE con show --active | awk -F: '$2=="wlan0"{print $1; exit}')
logger -t xrd-dhcp-guard "wlan0 丢 DHCP 租约, 重新激活连接 ${PROFILE}"
[ -n "$PROFILE" ] && systemd-run --collect nmcli con up "$PROFILE" >/dev/null 2>&1
exit 0
