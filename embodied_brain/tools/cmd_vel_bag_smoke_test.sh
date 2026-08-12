#!/usr/bin/env bash
# Isolated rosbag2 reader smoke test. Publishes zero Twist in a private domain only.

set -eo pipefail
set +u

TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_BASE="${CMD_VEL_SMOKE_BASE:-$HOME/embodied_v3_smoke}"
DOMAIN_ID="${CMD_VEL_SMOKE_DOMAIN_ID:-$((180 + $$ % 40))}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$OUT_BASE/cmd_vel_bag_$STAMP"
BAG_DIR="$OUT_DIR/bag"
REPORT="$OUT_DIR/cmd_vel_evidence.json"
REC_PID=""
PUB_PID=""

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  if [ -n "$PUB_PID" ] && kill -0 "$PUB_PID" 2>/dev/null; then
    kill -INT "$PUB_PID" 2>/dev/null || true
    wait "$PUB_PID" 2>/dev/null || true
  fi
  if [ -n "$REC_PID" ] && kill -0 "$REC_PID" 2>/dev/null; then
    kill -INT "$REC_PID" 2>/dev/null || true
    wait "$REC_PID" 2>/dev/null || true
  fi
  exit "$rc"
}
trap cleanup EXIT INT TERM

export ROS_DOMAIN_ID="$DOMAIN_ID"
export ROS_LOCALHOST_ONLY=1
export ROS2CLI_NO_DAEMON=1
export LD_LIBRARY_PATH="/opt/tros/humble/lib:${LD_LIBRARY_PATH:-}"
source /opt/ros/humble/setup.bash
if [ -f /opt/tros/humble/setup.bash ]; then
  source /opt/tros/humble/setup.bash
fi

command -v ros2 >/dev/null 2>&1
test -f "$TOOLS_DIR/verify_cmd_vel_bag.py"
mkdir -p "$OUT_DIR"

ros2 topic pub --rate 10 /cmd_vel geometry_msgs/msg/Twist '{}' > "$OUT_DIR/publisher.log" 2>&1 &
PUB_PID="$!"
topic_ready=0
for _ in $(seq 1 40); do
  kill -0 "$PUB_PID"
  if ros2 topic list 2>/dev/null | grep -qxF /cmd_vel; then
    topic_ready=1
    break
  fi
  sleep 0.5
done
if [ "$topic_ready" != "1" ]; then
  echo "ERR isolated zero-Twist publisher did not become discoverable" >&2
  exit 3
fi

ros2 bag record -s sqlite3 -o "$BAG_DIR" /cmd_vel > "$OUT_DIR/record.log" 2>&1 &
REC_PID="$!"
sleep 4
kill -0 "$REC_PID"
sleep 3
kill -INT "$PUB_PID" 2>/dev/null || true
wait "$PUB_PID" 2>/dev/null || true
PUB_PID=""

kill -INT "$REC_PID" 2>/dev/null || true
for _ in $(seq 1 20); do
  if ! kill -0 "$REC_PID" 2>/dev/null; then
    break
  fi
  sleep 0.25
done
wait "$REC_PID" 2>/dev/null || true
REC_PID=""

python3 "$TOOLS_DIR/verify_cmd_vel_bag.py" \
  --bag-dir "$BAG_DIR" \
  --expect zero \
  --out "$REPORT"

python3 - "$REPORT" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
counts = report.get("counts") or {}
assert report.get("status") == "PASS", report
assert report.get("expectation") == "zero", report
assert int(counts.get("message_count") or 0) > 0, report
assert int(counts.get("nonzero_count") or 0) == 0, report
print(
    "CMD_VEL_BAG_SMOKE PASS "
    f"storage={report.get('storage_id')} messages={counts.get('message_count')} "
    f"nonzero={counts.get('nonzero_count')}"
)
PY

echo "domain_id: $DOMAIN_ID"
echo "report: $REPORT"
echo "safety: ROS_LOCALHOST_ONLY=1, private domain, zero Twist only, no bag playback"
