#!/usr/bin/env bash
# Run the one physical F407 test allowed before clearing estop after a reflash.
#
# This script never clears estop and never sends non-zero velocity. The wrapped
# interlock test does send ELECTROMAGNET OFF, so any suspended load can fall.

set -Eeuo pipefail
umask 077

TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPECTED_HOSTNAME="${EXPECTED_CAR_HOSTNAME:-embodied-x5}"
EXPECTED_CONFIRMATION="NO_LOAD_PATH_CLEAR_BASE_FIXED_HANDS_CLEAR_OPERATOR_PRESENT"
CONFIRMATION=""
MAGNET_OFF_ACK=0
EVIDENCE_ROOT="${F407_POSTFLASH_EVIDENCE_ROOT:-$HOME/f407_postflash_acceptance}"
OUT_DIR=""
PORT="/dev/F407"

PHASE="argument_validation"
FAILURE_REASON=""
FINALIZED=0
RESTORE_NEEDED=0
RESTORE_ATTEMPTED=0
RESTORE_SUCCESS=0
SERVICES_QUIESCED=0
SERIAL_OWNERS_PRE=""
SERIAL_OWNERS_AFTER_STOP=""

SYSTEM_EMBODIED_PRE="unknown"
USER_EMBODIED_PRE="unknown"
SYSTEM_COCKPIT_PRE="unknown"
USER_COCKPIT_PRE="unknown"
SYSTEM_EMBODIED_POST="unknown"
USER_EMBODIED_POST="unknown"
SYSTEM_COCKPIT_POST="unknown"
USER_COCKPIT_POST="unknown"

STARTED_AT_UNIX="$(date +%s)"
STARTED_AT_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
HOSTNAME_ACTUAL=""
CONFIRMATION_SHA256=""
SCRIPT_SHA256=""
LINK_TEST_SHA256=""
VALIDATOR_SHA256=""
INTERLOCK_REPORT=""
INTERLOCK_LOG=""
VALIDATION_REPORT=""
POST_FIRMWARE_TOPIC=""
POST_IDENTITY_TOPIC=""
POST_ESTOP_TOPIC=""
MANIFEST=""

usage() {
  cat <<'USAGE'
Usage:
  f407_postflash_interlock_acceptance.sh \
    --confirm-safe-field-state NO_LOAD_PATH_CLEAR_BASE_FIXED_HANDS_CLEAR_OPERATOR_PRESENT \
    --acknowledge-magnet-off-can-drop-load

Optional:
  --out-dir PATH   Unique evidence directory under $HOME.

Required field state:
  - no bottle, suspended load, or loose fixture is attached
  - lift path is clear and hands are outside the mechanism
  - mobile base is fixed and an operator is present
  - operator understands that ELECTROMAGNET OFF is sent and can drop a load

Safety contract:
  The F407 remains estop-latched. Only zero velocity, emergency stop, blocked
  actuator probes, and electromagnet OFF are sent. No physical completion is
  claimed by this test.
USAGE
}

die() {
  FAILURE_REASON="phase=${PHASE}: $*"
  echo "ERR $*" >&2
  exit 2
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --confirm-safe-field-state)
      shift || die "--confirm-safe-field-state needs the exact token"
      CONFIRMATION="${1:-}"
      ;;
    --acknowledge-magnet-off-can-drop-load)
      MAGNET_OFF_ACK=1
      ;;
    --out-dir)
      shift || die "--out-dir needs a path"
      OUT_DIR="${1:-}"
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

[ "$CONFIRMATION" = "$EXPECTED_CONFIRMATION" ] || die "safe-field confirmation token missing or incorrect"
[ "$MAGNET_OFF_ACK" = "1" ] || die "magnet-off drop hazard was not acknowledged"
CONFIRMATION_SHA256="$(printf '%s' "$CONFIRMATION" | sha256sum | awk '{print $1}')"

STAMP="$(date -u +%Y%m%d_%H%M%S)"
if [ -z "$OUT_DIR" ]; then
  OUT_DIR="$EVIDENCE_ROOT/postflash_${STAMP}"
fi
case "$OUT_DIR" in
  "$HOME"/*) ;;
  *) die "evidence path must be an absolute path below $HOME" ;;
esac
[ ! -e "$OUT_DIR" ] || die "evidence directory already exists: $OUT_DIR"
mkdir -p "$(dirname "$OUT_DIR")"
mkdir "$OUT_DIR"

INTERLOCK_REPORT="$OUT_DIR/f407_interlock_report.json"
INTERLOCK_LOG="$OUT_DIR/f407_interlock.log"
VALIDATION_REPORT="$OUT_DIR/f407_interlock_validation.json"
POST_FIRMWARE_TOPIC="$OUT_DIR/post_f407_firmware_info.txt"
POST_IDENTITY_TOPIC="$OUT_DIR/post_f407_firmware_identity_valid.txt"
POST_ESTOP_TOPIC="$OUT_DIR/post_f407_estop_latched.txt"
MANIFEST="$OUT_DIR/postflash_interlock_manifest.json"

script_hash() {
  sha256sum "$1" | awk '{print $1}'
}

service_state() {
  local scope="$1"
  local service="$2"
  if [ "$scope" = "system" ]; then
    systemctl is-active "$service" 2>/dev/null || true
  else
    systemctl --user is-active "$service" 2>/dev/null || true
  fi
}

serial_owners() {
  if command -v fuser >/dev/null 2>&1; then
    fuser "$PORT" 2>/dev/null || true
  elif command -v lsof >/dev/null 2>&1; then
    lsof -t "$PORT" 2>/dev/null || true
  else
    echo "owner-check-tool-missing"
  fi
}

wait_service_state() {
  local scope="$1"
  local service="$2"
  local expected="$3"
  local i
  for i in $(seq 1 20); do
    if [ "$(service_state "$scope" "$service")" = "$expected" ]; then
      return 0
    fi
    sleep 1
  done
  return 1
}

start_original_service() {
  local scope="$1"
  local service="$2"
  if [ "$scope" = "system" ]; then
    sudo -n systemctl start "$service"
  else
    systemctl --user start "$service"
  fi
  wait_service_state "$scope" "$service" active
}

restore_original_services() {
  RESTORE_ATTEMPTED=1
  local failed=0

  if [ "$SYSTEM_EMBODIED_PRE" = "active" ]; then
    start_original_service system embodied_brain.service || failed=1
  fi
  if [ "$USER_EMBODIED_PRE" = "active" ]; then
    start_original_service user embodied_brain.service || failed=1
  fi
  if [ "$SYSTEM_COCKPIT_PRE" = "active" ]; then
    start_original_service system cockpit_bridge.service || failed=1
  fi
  if [ "$USER_COCKPIT_PRE" = "active" ]; then
    start_original_service user cockpit_bridge.service || failed=1
  fi

  SYSTEM_EMBODIED_POST="$(service_state system embodied_brain.service)"
  USER_EMBODIED_POST="$(service_state user embodied_brain.service)"
  SYSTEM_COCKPIT_POST="$(service_state system cockpit_bridge.service)"
  USER_COCKPIT_POST="$(service_state user cockpit_bridge.service)"

  if [ "$failed" = "0" ]; then
    RESTORE_SUCCESS=1
    RESTORE_NEEDED=0
    return 0
  fi
  RESTORE_SUCCESS=0
  return 1
}

write_manifest() {
  local overall="$1"
  local finished_at_unix finished_at_utc
  finished_at_unix="$(date +%s)"
  finished_at_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  PF_OVERALL="$overall" \
  PF_FAILURE_REASON="$FAILURE_REASON" \
  PF_STARTED_AT_UNIX="$STARTED_AT_UNIX" \
  PF_STARTED_AT_UTC="$STARTED_AT_UTC" \
  PF_FINISHED_AT_UNIX="$finished_at_unix" \
  PF_FINISHED_AT_UTC="$finished_at_utc" \
  PF_HOSTNAME="$HOSTNAME_ACTUAL" \
  PF_EXPECTED_HOSTNAME="$EXPECTED_HOSTNAME" \
  PF_CONFIRMATION_SHA256="$CONFIRMATION_SHA256" \
  PF_MAGNET_OFF_ACK="$MAGNET_OFF_ACK" \
  PF_SCRIPT="$TOOLS_DIR/f407_postflash_interlock_acceptance.sh" \
  PF_SCRIPT_SHA256="$SCRIPT_SHA256" \
  PF_LINK_TEST="$TOOLS_DIR/f407_link_test.py" \
  PF_LINK_TEST_SHA256="$LINK_TEST_SHA256" \
  PF_VALIDATOR="$TOOLS_DIR/f407_postflash_report.py" \
  PF_VALIDATOR_SHA256="$VALIDATOR_SHA256" \
  PF_INTERLOCK_REPORT="$INTERLOCK_REPORT" \
  PF_INTERLOCK_LOG="$INTERLOCK_LOG" \
  PF_VALIDATION_REPORT="$VALIDATION_REPORT" \
  PF_POST_FIRMWARE_TOPIC="$POST_FIRMWARE_TOPIC" \
  PF_POST_IDENTITY_TOPIC="$POST_IDENTITY_TOPIC" \
  PF_POST_ESTOP_TOPIC="$POST_ESTOP_TOPIC" \
  PF_SYSTEM_EMBODIED_PRE="$SYSTEM_EMBODIED_PRE" \
  PF_USER_EMBODIED_PRE="$USER_EMBODIED_PRE" \
  PF_SYSTEM_COCKPIT_PRE="$SYSTEM_COCKPIT_PRE" \
  PF_USER_COCKPIT_PRE="$USER_COCKPIT_PRE" \
  PF_SYSTEM_EMBODIED_POST="$SYSTEM_EMBODIED_POST" \
  PF_USER_EMBODIED_POST="$USER_EMBODIED_POST" \
  PF_SYSTEM_COCKPIT_POST="$SYSTEM_COCKPIT_POST" \
  PF_USER_COCKPIT_POST="$USER_COCKPIT_POST" \
  PF_SERIAL_OWNERS_PRE="$SERIAL_OWNERS_PRE" \
  PF_SERIAL_OWNERS_AFTER_STOP="$SERIAL_OWNERS_AFTER_STOP" \
  PF_SERVICES_QUIESCED="$SERVICES_QUIESCED" \
  PF_RESTORE_ATTEMPTED="$RESTORE_ATTEMPTED" \
  PF_RESTORE_SUCCESS="$RESTORE_SUCCESS" \
  python3 - "$MANIFEST" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path


def flag(name: str) -> bool:
    return os.environ.get(name) == "1"


def artifact(path_text: str) -> dict:
    path = Path(path_text)
    item = {"path": str(path), "exists": path.is_file()}
    if path.is_file():
        item["size_bytes"] = path.stat().st_size
        item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return item


manifest = {
    "schema_version": "xrd-f407-postflash-interlock-orchestration-v1",
    "overall": os.environ["PF_OVERALL"],
    "failure_reason": os.environ.get("PF_FAILURE_REASON", ""),
    "started_at_unix": int(os.environ["PF_STARTED_AT_UNIX"]),
    "started_at_utc": os.environ["PF_STARTED_AT_UTC"],
    "finished_at_unix": int(os.environ["PF_FINISHED_AT_UNIX"]),
    "finished_at_utc": os.environ["PF_FINISHED_AT_UTC"],
    "hostname": {
        "expected": os.environ["PF_EXPECTED_HOSTNAME"],
        "actual": os.environ.get("PF_HOSTNAME", ""),
        "matched": os.environ.get("PF_HOSTNAME") == os.environ["PF_EXPECTED_HOSTNAME"],
    },
    "operator_confirmation": {
        "safe_field_state_confirmed": True,
        "confirmation_token_sha256": os.environ["PF_CONFIRMATION_SHA256"],
        "magnet_off_drop_hazard_acknowledged": flag("PF_MAGNET_OFF_ACK"),
        "raw_confirmation_token_stored": False,
    },
    "command_contract": {
        "argv": [
            "python3",
            os.environ["PF_LINK_TEST"],
            "--port",
            "/dev/F407",
            "--verify-estop-interlock",
            "--require-ack",
            "--report",
            os.environ["PF_INTERLOCK_REPORT"],
        ],
        "clear_estop_requested": False,
        "nonzero_cmd_vel_requested": False,
        "electromagnet_off_is_sent": True,
        "physical_completion_claimed": False,
    },
    "serial_exclusivity": {
        "device": "/dev/F407",
        "owners_before": os.environ.get("PF_SERIAL_OWNERS_PRE", "").split(),
        "owners_after_stop": os.environ.get("PF_SERIAL_OWNERS_AFTER_STOP", "").split(),
        "unowned_before_test": not bool(os.environ.get("PF_SERIAL_OWNERS_AFTER_STOP", "").strip()),
    },
    "service_restore": {
        "services_quiesced": flag("PF_SERVICES_QUIESCED"),
        "attempted": flag("PF_RESTORE_ATTEMPTED"),
        "success": flag("PF_RESTORE_SUCCESS"),
        "pre": {
            "system": {
                "embodied_brain.service": os.environ["PF_SYSTEM_EMBODIED_PRE"],
                "cockpit_bridge.service": os.environ["PF_SYSTEM_COCKPIT_PRE"],
            },
            "user": {
                "embodied_brain.service": os.environ["PF_USER_EMBODIED_PRE"],
                "cockpit_bridge.service": os.environ["PF_USER_COCKPIT_PRE"],
            },
        },
        "post": {
            "system": {
                "embodied_brain.service": os.environ["PF_SYSTEM_EMBODIED_POST"],
                "cockpit_bridge.service": os.environ["PF_SYSTEM_COCKPIT_POST"],
            },
            "user": {
                "embodied_brain.service": os.environ["PF_USER_EMBODIED_POST"],
                "cockpit_bridge.service": os.environ["PF_USER_COCKPIT_POST"],
            },
        },
    },
    "tooling": {
        "orchestrator": {"path": os.environ["PF_SCRIPT"], "sha256": os.environ["PF_SCRIPT_SHA256"]},
        "link_test": {"path": os.environ["PF_LINK_TEST"], "sha256": os.environ["PF_LINK_TEST_SHA256"]},
        "validator": {"path": os.environ["PF_VALIDATOR"], "sha256": os.environ["PF_VALIDATOR_SHA256"]},
    },
    "artifacts": {
        "interlock_report": artifact(os.environ["PF_INTERLOCK_REPORT"]),
        "interlock_log": artifact(os.environ["PF_INTERLOCK_LOG"]),
        "validation_report": artifact(os.environ["PF_VALIDATION_REPORT"]),
        "post_firmware_topic": artifact(os.environ["PF_POST_FIRMWARE_TOPIC"]),
        "post_identity_topic": artifact(os.environ["PF_POST_IDENTITY_TOPIC"]),
        "post_estop_topic": artifact(os.environ["PF_POST_ESTOP_TOPIC"]),
    },
}
Path(sys.argv[1]).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  if [ "$RESTORE_NEEDED" = "1" ]; then
    PHASE="cleanup_restore"
    if ! restore_original_services; then
      if [ -z "$FAILURE_REASON" ]; then
        FAILURE_REASON="phase=cleanup_restore: failed to restore original services"
      else
        FAILURE_REASON="$FAILURE_REASON; cleanup_restore_failed"
      fi
      rc=9
    fi
  fi
  if [ "$FINALIZED" != "1" ] && [ -n "$MANIFEST" ] && [ -d "$OUT_DIR" ]; then
    if [ -z "$FAILURE_REASON" ]; then
      FAILURE_REASON="phase=${PHASE}: unexpected exit rc=${rc}"
    fi
    write_manifest FAIL || true
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

PHASE="host_and_tool_preflight"
HOSTNAME_ACTUAL="$(hostname)"
[ "$HOSTNAME_ACTUAL" = "$EXPECTED_HOSTNAME" ] || die "hostname mismatch expected=$EXPECTED_HOSTNAME actual=$HOSTNAME_ACTUAL"
command -v systemctl >/dev/null 2>&1 || die "systemctl is required"
command -v python3 >/dev/null 2>&1 || die "python3 is required"
command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required"
command -v timeout >/dev/null 2>&1 || die "timeout is required"
[ -e "$PORT" ] || die "$PORT is missing"
[ -f "$TOOLS_DIR/f407_link_test.py" ] || die "f407_link_test.py is missing"
[ -f "$TOOLS_DIR/f407_postflash_report.py" ] || die "f407_postflash_report.py is missing"
python3 -c 'import serial' >/dev/null 2>&1 || die "python3 pyserial is required"

SCRIPT_SHA256="$(script_hash "$TOOLS_DIR/f407_postflash_interlock_acceptance.sh")"
LINK_TEST_SHA256="$(script_hash "$TOOLS_DIR/f407_link_test.py")"
VALIDATOR_SHA256="$(script_hash "$TOOLS_DIR/f407_postflash_report.py")"

SYSTEM_EMBODIED_PRE="$(service_state system embodied_brain.service)"
USER_EMBODIED_PRE="$(service_state user embodied_brain.service)"
SYSTEM_COCKPIT_PRE="$(service_state system cockpit_bridge.service)"
USER_COCKPIT_PRE="$(service_state user cockpit_bridge.service)"
SERIAL_OWNERS_PRE="$(serial_owners)"

if [ "$SYSTEM_EMBODIED_PRE" = "active" ] && [ "$USER_EMBODIED_PRE" = "active" ]; then
  die "embodied_brain.service is active in both system and user scopes"
fi
if [ "$SYSTEM_EMBODIED_PRE" != "active" ] && [ "$USER_EMBODIED_PRE" != "active" ]; then
  die "embodied_brain.service is not active in either scope; refusing an unowned maintenance state"
fi
if [ "$SYSTEM_COCKPIT_PRE" = "active" ] && [ "$USER_COCKPIT_PRE" = "active" ]; then
  die "cockpit_bridge.service is active in both system and user scopes"
fi
if [ "$SYSTEM_EMBODIED_PRE" = "active" ] || [ "$SYSTEM_COCKPIT_PRE" = "active" ]; then
  command -v sudo >/dev/null 2>&1 || die "sudo is required to stop and restore system services"
  sudo -n true || die "passwordless sudo is required to stop and restore system services"
fi

echo "WARNING: the test sends ELECTROMAGNET OFF. No load may be attached."
echo "Safety confirmation accepted; F407 estop will remain latched."

PHASE="quiesce_services"
RESTORE_NEEDED=1
if [ "$SYSTEM_COCKPIT_PRE" = "active" ]; then sudo -n systemctl stop cockpit_bridge.service; fi
if [ "$USER_COCKPIT_PRE" = "active" ]; then systemctl --user stop cockpit_bridge.service; fi
if [ "$SYSTEM_EMBODIED_PRE" = "active" ]; then sudo -n systemctl stop embodied_brain.service; fi
if [ "$USER_EMBODIED_PRE" = "active" ]; then systemctl --user stop embodied_brain.service; fi
bash "$TOOLS_DIR/start_embodied_v3_stack.sh" stop
sleep 2
SERVICES_QUIESCED=1
SERIAL_OWNERS_AFTER_STOP="$(serial_owners)"
[ "$SERIAL_OWNERS_AFTER_STOP" != "owner-check-tool-missing" ] || die "fuser or lsof is required"
[ -z "$SERIAL_OWNERS_AFTER_STOP" ] || die "$PORT is still owned by PID(s): $SERIAL_OWNERS_AFTER_STOP"

PHASE="physical_interlock_test"
INTERLOCK_CMD=(
  python3 "$TOOLS_DIR/f407_link_test.py"
  --port "$PORT"
  --verify-estop-interlock
  --require-ack
  --report "$INTERLOCK_REPORT"
)
timeout 60 "${INTERLOCK_CMD[@]}" 2>&1 | tee "$INTERLOCK_LOG"
[ -s "$INTERLOCK_REPORT" ] || die "interlock report was not written"

PHASE="strict_report_validation"
python3 "$TOOLS_DIR/f407_postflash_report.py" \
  --report "$INTERLOCK_REPORT" \
  --out "$VALIDATION_REPORT"

PHASE="restore_services"
restore_original_services || die "failed to restore original service state"

PHASE="post_restore_readonly_topics"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ROS2CLI_NO_DAEMON=1
export LD_LIBRARY_PATH="/opt/tros/humble/lib:${LD_LIBRARY_PATH:-}"
# ROS Humble/TROS generated setup files are not nounset-safe.
set +u
source /opt/ros/humble/setup.bash 2>/dev/null || true
source /opt/tros/humble/setup.bash 2>/dev/null || true
source "$HOME/ros2_ws/install/setup.bash" 2>/dev/null || true
set -u
command -v ros2 >/dev/null 2>&1 || die "ros2 unavailable after service restore"

capture_topic() {
  local topic="$1"
  local output="$2"
  local i
  for i in $(seq 1 4); do
    if timeout 10 ros2 topic echo --once --full-length "$topic" > "$output" 2>&1; then
      return 0
    fi
    sleep 2
  done
  return 1
}

capture_topic /f407/firmware_info "$POST_FIRMWARE_TOPIC" || die "post-restore firmware_info topic unavailable"
capture_topic /f407/firmware_identity_valid "$POST_IDENTITY_TOPIC" || die "post-restore firmware identity topic unavailable"
capture_topic /f407/estop_latched "$POST_ESTOP_TOPIC" || die "post-restore estop topic unavailable"
grep -q '"build_id":2026071907' "$POST_FIRMWARE_TOPIC" || die "post-restore firmware build mismatch"
grep -q '"identity_valid":true' "$POST_FIRMWARE_TOPIC" || die "post-restore firmware identity invalid"
grep -Eq '^data: true$' "$POST_IDENTITY_TOPIC" || die "post-restore identity_valid is not true"
grep -Eq '^data: true$' "$POST_ESTOP_TOPIC" || die "post-restore estop is not latched"

PHASE="finalize"
FAILURE_REASON=""
write_manifest PASS
FINALIZED=1

echo "F407_POSTFLASH_INTERLOCK_ACCEPTANCE_PASS"
echo "evidence_dir: $OUT_DIR"
echo "manifest: $MANIFEST"
echo "safety: estop remains latched; no non-zero cmd_vel; no physical completion claim"
