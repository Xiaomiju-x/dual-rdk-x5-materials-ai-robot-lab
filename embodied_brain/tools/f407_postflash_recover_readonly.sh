#!/usr/bin/env bash
# Source ROS safely, then run the read-only post-flash recovery tool.

set -Eeuo pipefail

TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$TOOLS_DIR/f407_postflash_recover_readonly.py" ] || {
  echo "ERR missing $TOOLS_DIR/f407_postflash_recover_readonly.py" >&2
  exit 2
}

# ROS Humble/TROS generated setup files are not nounset-safe.
set +u
source /opt/ros/humble/setup.bash 2>/dev/null || true
source /opt/tros/humble/setup.bash 2>/dev/null || true
source "$HOME/ros2_ws/install/setup.bash" 2>/dev/null || true
set -u

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ROS2CLI_NO_DAEMON=1
export LD_LIBRARY_PATH="/opt/tros/humble/lib:${LD_LIBRARY_PATH:-}"
exec python3 "$TOOLS_DIR/f407_postflash_recover_readonly.py" "$@"
