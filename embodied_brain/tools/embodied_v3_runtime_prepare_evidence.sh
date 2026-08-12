#!/usr/bin/env bash
# Prepare fresh runtime evidence for embodied_v3_acceptance_check.sh.
#
# Safety contract:
# - stops serial_f407_node before direct serial interlock verification
# - never publishes a non-zero /cmd_vel
# - leaves the F407 estop latched after the interlock test
# - starts sensors/SLAM/Lab-FSD plus BPU MPPI in proposed-only mode
# - records and finalizes one fresh data-loop run

set -eo pipefail
set +u

CAPTURE_S="${EMBODIED_V3_CAPTURE_S:-25}"
SETTLE_S="${EMBODIED_V3_SETTLE_S:-35}"
STORAGE="${EMBODIED_V3_STORAGE:-auto}"
STATE_DIR="${EMBODIED_V3_RUNTIME_STATE_DIR:-$HOME/embodied_v3_runtime}"
INTERLOCK_REPORT="${F407_INTERLOCK_REPORT:-$HOME/f407_interlock_evidence/latest.json}"
POSTFLASH_MANIFEST="${F407_POSTFLASH_MANIFEST:-}"
TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LATEST_RUN_FILE="$STATE_DIR/latest_data_run.txt"
LATEST_PREP_JSON="$STATE_DIR/latest_prepare.json"
ZERO_LOG="/tmp/embodied_v3_zero_cmd_vel.log"
MPPI_STATS_RAW="$STATE_DIR/latest_mppi_stats.txt"
MPPI_PROPOSED_RAW="$STATE_DIR/latest_mppi_cmd_vel_proposed.txt"
MPPI_CMD_VEL_INFO="$STATE_DIR/latest_cmd_vel_publishers.txt"
LATEST_POSTFLASH_MANIFEST="$STATE_DIR/latest_postflash_manifest.json"
LATEST_POSTFLASH_BUNDLE_INDEX="$STATE_DIR/latest_postflash_bundle_index.json"
SERVICE_SCOPE="none"
INTERLOCK_SOURCE_MODE="direct_interlock"
POSTFLASH_MANIFEST_COPY=""
POSTFLASH_MANIFEST_SHA256=""
POSTFLASH_BUNDLE_INDEX=""
POSTFLASH_BUNDLE_INDEX_SHA256=""

usage() {
  cat <<'USAGE'
Usage: embodied_v3_runtime_prepare_evidence.sh [options]

Options:
  --capture-s SEC     Zero-velocity rosbag capture duration. Default: 25.
  --settle-s SEC      Runtime stack settle time. Default: 35.
  --storage MODE      auto|mcap|sqlite3. Default: auto.
  -h, --help          Show this help.

This script does not move the robot. It publishes only zero Twist messages.
The F407 remains estop-latched after the direct firmware interlock test.
When F407_POSTFLASH_MANIFEST is supplied, its strictly validated interlock is
reused and no second physical interlock probe is sent.
USAGE
}

die() {
  echo "ERR $*" >&2
  exit 2
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --capture-s)
      shift || die "--capture-s needs a value"
      CAPTURE_S="${1:-}"
      ;;
    --settle-s)
      shift || die "--settle-s needs a value"
      SETTLE_S="${1:-}"
      ;;
    --storage)
      shift || die "--storage needs a value"
      STORAGE="${1:-}"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option '$1'"
      ;;
  esac
  shift
done

[[ "$CAPTURE_S" =~ ^[0-9]+$ ]] || die "capture duration must be an integer"
[[ "$SETTLE_S" =~ ^[0-9]+$ ]] || die "settle duration must be an integer"
[ "$CAPTURE_S" -ge 10 ] && [ "$CAPTURE_S" -le 300 ] || die "capture duration must be 10..300 seconds"
[ "$SETTLE_S" -ge 10 ] && [ "$SETTLE_S" -le 180 ] || die "settle duration must be 10..180 seconds"
case "$STORAGE" in auto|mcap|sqlite3) ;; *) die "storage must be auto, mcap, or sqlite3" ;; esac

for tool in \
  start_embodied_v3_stack.sh \
  f407_link_test.py \
  f407_postflash_bundle.py \
  f407_postflash_report.py \
  data_loop_start.sh \
  data_loop_stop.sh
do
  [ -f "$TOOLS_DIR/$tool" ] || die "missing $TOOLS_DIR/$tool"
done
[ -e /dev/F407 ] || die "/dev/F407 missing"
if [ -z "$POSTFLASH_MANIFEST" ]; then
  python3 -c 'import serial' >/dev/null 2>&1 || die "python3 pyserial is required for F407 interlock verification"
fi

mkdir -p "$STATE_DIR"
rm -f "$LATEST_RUN_FILE" "$LATEST_PREP_JSON" \
  "$LATEST_POSTFLASH_MANIFEST" "$LATEST_POSTFLASH_BUNDLE_INDEX"
if [ -n "$POSTFLASH_MANIFEST" ]; then
  POSTFLASH_STAGE_DIR="$STATE_DIR/postflash_input_$(date -u +%Y%m%d_%H%M%S)_$$"
  mkdir "$POSTFLASH_STAGE_DIR"
  python3 "$TOOLS_DIR/f407_postflash_bundle.py" \
    --manifest "$POSTFLASH_MANIFEST" \
    --out-dir "$POSTFLASH_STAGE_DIR" \
    --index "$POSTFLASH_STAGE_DIR/f407_postflash_bundle_index.json" \
    > "$POSTFLASH_STAGE_DIR/bundle_stdout.txt"
  INTERLOCK_REPORT="$POSTFLASH_STAGE_DIR/f407_interlock_report.json"
  cp "$POSTFLASH_STAGE_DIR/f407_postflash_manifest.json" "$LATEST_POSTFLASH_MANIFEST"
  cp "$POSTFLASH_STAGE_DIR/f407_postflash_bundle_index.json" "$LATEST_POSTFLASH_BUNDLE_INDEX"
  POSTFLASH_MANIFEST_COPY="$LATEST_POSTFLASH_MANIFEST"
  POSTFLASH_BUNDLE_INDEX="$LATEST_POSTFLASH_BUNDLE_INDEX"
  python3 "$TOOLS_DIR/f407_postflash_report.py" \
    --report "$INTERLOCK_REPORT" \
    --out "$POSTFLASH_STAGE_DIR/runtime_interlock_revalidation.json" \
    > "$POSTFLASH_STAGE_DIR/revalidation_stdout.txt"
  POSTFLASH_MANIFEST_SHA256="$(sha256sum "$POSTFLASH_MANIFEST_COPY" | awk '{print $1}')"
  POSTFLASH_BUNDLE_INDEX_SHA256="$(sha256sum "$POSTFLASH_BUNDLE_INDEX" | awk '{print $1}')"
  INTERLOCK_SOURCE_MODE="postflash_manifest_reuse"
else
  mkdir -p "$(dirname "$INTERLOCK_REPORT")"
  rm -f "$INTERLOCK_REPORT"
fi

ZERO_PID=""
RUN_DIR=""
DATA_STARTED=0
DATA_STOPPED=0

restore_managed_service_after_failure() {
  local failed=0
  bash "$TOOLS_DIR/start_embodied_v3_stack.sh" stop \
    > /tmp/embodied_v3_runtime_failure_stack_stop.log 2>&1 || true
  case ",$SERVICE_SCOPE," in
    *,system,*)
      if ! sudo -n systemctl start embodied_brain.service; then
        failed=1
      fi
      ;;
  esac
  case ",$SERVICE_SCOPE," in
    *,user,*)
      if ! systemctl --user start embodied_brain.service; then
        failed=1
      fi
      ;;
  esac
  if [ "$failed" = "0" ]; then
    echo "RUNTIME_PREP_FAILURE_OWNER_RESTORED scope=$SERVICE_SCOPE" >&2
    return 0
  fi
  echo "ERR runtime preparation failed and original service owner could not be restored: $SERVICE_SCOPE" >&2
  return 1
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  if [ -n "$ZERO_PID" ] && kill -0 "$ZERO_PID" 2>/dev/null; then
    kill "$ZERO_PID" 2>/dev/null || true
    wait "$ZERO_PID" 2>/dev/null || true
  fi
  if [ "$DATA_STARTED" = "1" ] && [ "$DATA_STOPPED" != "1" ] && [ -n "$RUN_DIR" ]; then
    bash "$TOOLS_DIR/data_loop_stop.sh" "$RUN_DIR" --no-video2-copy \
      > /tmp/embodied_v3_data_loop_cleanup.log 2>&1 || true
  fi
  if [ "$rc" -ne 0 ] && [ "$SERVICE_SCOPE" != "none" ]; then
    if ! restore_managed_service_after_failure; then
      rc=9
    fi
  fi
  exit "$rc"
}
trap cleanup EXIT INT TERM

stop_managed_service() {
  local stopped=()
  if command -v systemctl >/dev/null 2>&1 && systemctl --user is-active --quiet embodied_brain.service 2>/dev/null; then
    systemctl --user stop embodied_brain.service || die "failed to stop user embodied_brain.service"
    stopped+=(user)
  fi
  if command -v systemctl >/dev/null 2>&1 && systemctl is-active --quiet embodied_brain.service 2>/dev/null; then
    command -v sudo >/dev/null 2>&1 || die "system embodied_brain.service is active but sudo is unavailable"
    sudo -n systemctl stop embodied_brain.service || die "cannot stop system embodied_brain.service without interactive sudo"
    stopped+=(system)
  fi
  if [ "${#stopped[@]}" -gt 0 ]; then
    SERVICE_SCOPE="$(IFS=,; echo "${stopped[*]}")"
  fi
}

assert_f407_unowned() {
  local owners=""
  if command -v fuser >/dev/null 2>&1; then
    owners="$(fuser /dev/F407 2>/dev/null || true)"
  elif command -v lsof >/dev/null 2>&1; then
    owners="$(lsof -t /dev/F407 2>/dev/null || true)"
  fi
  [ -z "$owners" ] || die "/dev/F407 is still owned by PID(s): $owners"
}

echo "[1/5] Stop ROS owners of /dev/F407"
stop_managed_service
bash "$TOOLS_DIR/start_embodied_v3_stack.sh" stop
sleep 2
assert_f407_unowned

if [ "$INTERLOCK_SOURCE_MODE" = "postflash_manifest_reuse" ]; then
  echo "[2/5] Reuse strictly validated post-flash interlock; no serial probe commands sent"
else
  echo "[2/5] Verify post-flash F407 estop interlock without non-zero cmd_vel"
  python3 "$TOOLS_DIR/f407_link_test.py" \
    --verify-estop-interlock \
    --require-ack \
    --report "$INTERLOCK_REPORT"
fi
[ -s "$INTERLOCK_REPORT" ] || die "F407 interlock report was not written"

echo "[3/5] Start sensor/SLAM/Lab-FSD + BPU MPPI proposed-only runtime"
EMBODIED_V3_SETTLE_S="$SETTLE_S" bash "$TOOLS_DIR/start_embodied_v3_stack.sh" start

source /opt/ros/humble/setup.bash 2>/dev/null || true
source /opt/tros/humble/setup.bash 2>/dev/null || true
source "$HOME/ros2_ws/install/setup.bash" 2>/dev/null || true
export LD_LIBRARY_PATH="/opt/tros/humble/lib:${LD_LIBRARY_PATH:-}"
command -v ros2 >/dev/null 2>&1 || die "ros2 not available after runtime start"

timeout 15 ros2 topic echo --once --full-length /mppi/stats > "$MPPI_STATS_RAW" 2>&1 \
  || die "MPPI stats unavailable after runtime start"
timeout 15 ros2 topic echo --once /mppi/cmd_vel_proposed > "$MPPI_PROPOSED_RAW" 2>&1 \
  || die "MPPI proposed command unavailable after runtime start"
timeout 10 ros2 topic info /cmd_vel -v > "$MPPI_CMD_VEL_INFO" 2>&1 \
  || die "cannot inspect /cmd_vel publishers"
grep -q '"proposed_only": true' "$MPPI_STATS_RAW" \
  || die "MPPI stats do not confirm proposed_only=true"
grep -q '"use_bpu": true' "$MPPI_STATS_RAW" \
  || die "MPPI stats do not confirm BPU execution"
grep -q '"estop_latched": true' "$MPPI_STATS_RAW" \
  || die "MPPI stats do not observe F407 estop latch"
if grep -qi 'Node name: mppi_node' "$MPPI_CMD_VEL_INFO"; then
  die "MPPI unexpectedly appears as a /cmd_vel publisher"
fi

echo "[4/5] Record a fresh data loop with zero Twist only"
: > "$ZERO_LOG"
ros2 topic pub --rate 5 /cmd_vel geometry_msgs/msg/Twist '{}' \
  > "$ZERO_LOG" 2>&1 &
ZERO_PID="$!"
sleep 2
kill -0 "$ZERO_PID" 2>/dev/null || die "zero /cmd_vel publisher exited early; see $ZERO_LOG"

DATA_LOOP_VIDEO2=skip bash "$TOOLS_DIR/data_loop_start.sh" \
  --name embodied_v3_runtime_accept \
  --storage "$STORAGE" \
  --cmd-vel-expect zero \
  --no-video2
RUN_DIR="$(cat "$HOME/data_loop_runs/current_run.txt" 2>/dev/null || true)"
[ -n "$RUN_DIR" ] && [ -d "$RUN_DIR" ] || die "data-loop run directory was not created"
DATA_STARTED=1
sleep "$CAPTURE_S"

kill "$ZERO_PID" 2>/dev/null || true
wait "$ZERO_PID" 2>/dev/null || true
ZERO_PID=""
bash "$TOOLS_DIR/data_loop_stop.sh" "$RUN_DIR" --no-video2-copy
DATA_STOPPED=1

[ -s "$RUN_DIR/manifest.json" ] || die "final manifest missing: $RUN_DIR/manifest.json"
[ -s "$RUN_DIR/manifest.sha256" ] || die "final manifest hash missing"
[ -s "$RUN_DIR/logs/data_loop_audit.json" ] || die "data-loop audit missing"
printf '%s\n' "$RUN_DIR" > "$LATEST_RUN_FILE"

DATA_MANIFEST_SHA256="$(sha256sum "$RUN_DIR/manifest.json" | awk '{print $1}')"
INTERLOCK_SHA256="$(sha256sum "$INTERLOCK_REPORT" | awk '{print $1}')"
MPPI_STATS_SHA256="$(sha256sum "$MPPI_STATS_RAW" | awk '{print $1}')"
MPPI_PROPOSED_SHA256="$(sha256sum "$MPPI_PROPOSED_RAW" | awk '{print $1}')"
MPPI_CMD_VEL_INFO_SHA256="$(sha256sum "$MPPI_CMD_VEL_INFO" | awk '{print $1}')"

python3 - "$LATEST_PREP_JSON" "$RUN_DIR" "$INTERLOCK_REPORT" "$CAPTURE_S" "$STORAGE" \
  "$DATA_MANIFEST_SHA256" "$INTERLOCK_SHA256" "$SERVICE_SCOPE" \
  "$MPPI_STATS_RAW" "$MPPI_STATS_SHA256" "$MPPI_PROPOSED_RAW" "$MPPI_PROPOSED_SHA256" \
  "$MPPI_CMD_VEL_INFO" "$MPPI_CMD_VEL_INFO_SHA256" "$INTERLOCK_SOURCE_MODE" \
  "$POSTFLASH_MANIFEST_COPY" "$POSTFLASH_MANIFEST_SHA256" \
  "$POSTFLASH_BUNDLE_INDEX" "$POSTFLASH_BUNDLE_INDEX_SHA256" <<'PY'
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

(
    out, run_dir, interlock, capture_s, storage, manifest_sha, interlock_sha,
    service_scope, mppi_stats, mppi_stats_sha, mppi_proposed, mppi_proposed_sha,
    cmd_vel_info, cmd_vel_info_sha, interlock_source_mode, postflash_manifest,
    postflash_manifest_sha, postflash_bundle_index, postflash_bundle_index_sha,
) = sys.argv[1:]
payload = {
    "schema_version": "xrd-embodied-v3-runtime-prepare-v4",
    "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    "generated_at_unix": time.time(),
    "data_run": run_dir,
    "data_manifest_sha256": manifest_sha,
    "interlock_report": interlock,
    "interlock_report_sha256": interlock_sha,
    "interlock_source_mode": interlock_source_mode,
    "postflash_manifest": postflash_manifest,
    "postflash_manifest_sha256": postflash_manifest_sha,
    "postflash_bundle_index": postflash_bundle_index,
    "postflash_bundle_index_sha256": postflash_bundle_index_sha,
    "capture_s": int(capture_s),
    "storage_requested": storage,
    "runtime_mode": "shadow_plus_mppi_proposed_only",
    "cmd_vel_capture": "zero_twist_only",
    "cmd_vel_topic": "/cmd_vel",
    "cmd_vel_rate_hz": 5,
    "cmd_vel_message": {},
    "nonzero_cmd_vel_published": False,
    "f407_estop_left_latched": True,
    "mppi": {
        "enabled": True,
        "use_bpu_required": True,
        "proposed_only": True,
        "proposed_topic": "/mppi/cmd_vel_proposed",
        "direct_cmd_vel": False,
        "stats_path": mppi_stats,
        "stats_sha256": mppi_stats_sha,
        "proposed_path": mppi_proposed,
        "proposed_sha256": mppi_proposed_sha,
        "cmd_vel_publishers_path": cmd_vel_info,
        "cmd_vel_publishers_sha256": cmd_vel_info_sha,
        "f407_estop_required": True,
    },
    "managed_service_stopped": service_scope,
}
Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

echo "[5/5] Runtime evidence prepared"
echo "EMBODIED_V3_RUNTIME_PREPARED"
echo "data_run: $RUN_DIR"
echo "data_run_file: $LATEST_RUN_FILE"
echo "interlock_report: $INTERLOCK_REPORT"
echo "prepare_json: $LATEST_PREP_JSON"
echo "safety: zero /cmd_vel only; MPPI proposed-only; F407 estop remains latched"
