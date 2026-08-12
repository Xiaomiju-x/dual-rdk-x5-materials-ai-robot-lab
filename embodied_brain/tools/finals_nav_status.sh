#!/usr/bin/env bash
set -eo pipefail
set +u

WATCH=0
INTERVAL_S=8
if [ "${1:-}" = "--watch" ]; then
  WATCH=1
  [[ "${2:-}" =~ ^[0-9]+([.][0-9]+)?$ ]] && INTERVAL_S="$2"
elif [ "$#" -gt 0 ]; then
  echo "Usage: finals_nav_status.sh [--watch [SECONDS]]"
  exit 0
fi

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export LD_LIBRARY_PATH="/opt/tros/humble/lib:${LD_LIBRARY_PATH:-}"
[ ! -f /opt/ros/humble/setup.bash ] || source /opt/ros/humble/setup.bash
[ ! -f /opt/tros/humble/setup.bash ] || source /opt/tros/humble/setup.bash
[ ! -f "$HOME/ros2_ws/install/setup.bash" ] || source "$HOME/ros2_ws/install/setup.bash"

STATUS_PY="${FINALS_NAV_STATUS_PY:-$HOME/tools/finals_nav_status.py}"
[ -f "$STATUS_PY" ] || { echo "ERR missing $STATUS_PY" >&2; exit 2; }

if [ "$WATCH" -eq 1 ]; then
  trap 'exit 0' INT TERM
  while true; do
    printf '\033[2J\033[H'
    python3 "$STATUS_PY" --timeout 5 || true
    printf '\nRead-only refresh every %ss. Ctrl-C exits.\n' "$INTERVAL_S"
    sleep "$INTERVAL_S"
  done
else
  exec python3 "$STATUS_PY" --timeout 5
fi
