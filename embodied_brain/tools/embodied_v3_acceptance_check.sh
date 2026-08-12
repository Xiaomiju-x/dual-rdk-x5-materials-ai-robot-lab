#!/usr/bin/env bash
# Collect an evidence bundle for the embodied_brain v3 stack on the car X5.
# This script is read-only: it does not start navigation and never publishes cmd_vel.

set -uo pipefail

OUT_DIR="${1:-$HOME/embodied_v3_acceptance/accept_$(date +%Y%m%d_%H%M%S)_$$}"
MODEL_BIN="${LAB_FSD_TINY_OCC_BIN:-$HOME/models/lab_fsd/lab_fsd_tiny_occ_risk.bin}"
ANOMALY_BIN="${LAB_FSD_ANOMALY_BIN:-$HOME/models/lab_fsd/lab_anomaly_autoencoder.bin}"
MPPI_BIN="${MPPI_COST_BIN:-$HOME/bpu_models/cost_mlp.bin}"
TOPIC_ECHO_ATTEMPTS="${EMBODIED_V3_TOPIC_ECHO_ATTEMPTS:-3}"
TOPIC_ECHO_TIMEOUT_S="${EMBODIED_V3_TOPIC_ECHO_TIMEOUT_S:-10}"
TOPIC_ECHO_INTERVAL_S="${EMBODIED_V3_TOPIC_ECHO_INTERVAL_S:-2}"
REQUIRE_DATA_RUN="${EMBODIED_V3_REQUIRE_DATA_RUN:-1}"
DATA_RUN_OVERRIDE="${EMBODIED_V3_DATA_RUN:-}"
DATA_RUN_FILE_OVERRIDE="${EMBODIED_V3_DATA_RUN_FILE:-}"
REQUIRE_RUNTIME_PREP="${EMBODIED_V3_REQUIRE_RUNTIME_PREP:-0}"
if [ "${EMBODIED_V3_RUNTIME_PREP+x}" = "x" ]; then
  RUNTIME_PREP_EXPLICIT=1
else
  RUNTIME_PREP_EXPLICIT=0
fi
RUNTIME_PREP_PATH="${EMBODIED_V3_RUNTIME_PREP:-$HOME/embodied_v3_runtime/latest_prepare.json}"
REQUIRE_POSTFLASH_INTERLOCK="${EMBODIED_V3_REQUIRE_POSTFLASH_INTERLOCK:-0}"
POSTFLASH_MANIFEST="${F407_POSTFLASH_MANIFEST:-}"
POSTFLASH_BUNDLE_TOOL="${F407_POSTFLASH_BUNDLE_TOOL:-$HOME/tools/f407_postflash_bundle.py}"
POSTFLASH_SKIP_STALE_RUNTIME="${F407_POSTFLASH_SKIP_STALE_RUNTIME:-1}"
DISPATCH_STUB_TOOL="${EMBODIED_V3_DISPATCH_STUB_TOOL:-$HOME/tools/dispatch_stub_integration.py}"
DISPATCH_STUB_REPORT="${EMBODIED_V3_DISPATCH_STUB_REPORT:-$HOME/dispatch_stub_evidence/latest.json}"
DISPATCH_STUB_DOMAIN_ID="${EMBODIED_V3_DISPATCH_STUB_DOMAIN_ID:-$((120 + ($$ % 113)))}"
DISPATCH_FIXTURE_TOOL="${EMBODIED_V3_DISPATCH_FIXTURE_TOOL:-$HOME/tools/dispatch_fixture_integration.py}"
DISPATCH_FIXTURE_DOMAIN_ID="${EMBODIED_V3_DISPATCH_FIXTURE_DOMAIN_ID:-$((120 + (($$ + 1) % 113)))}"
EXPECTED_OCC_SHA256="3b1a96483351f72746fdcacfb179b69f4527076046e5dd73d5bcae7688d99c90"
EXPECTED_ANOMALY_SHA256="1045be38ff947ad3c97c365416170970f59735504a1f38663bd8cce8d112ad7f"
EXPECTED_MPPI_SHA256="fe54f08d12285cf66c37ee7168b51a6762bb086b30a681a12f18374d8eea853d"

case "$REQUIRE_POSTFLASH_INTERLOCK" in
  0|1) ;;
  *) echo "ERR EMBODIED_V3_REQUIRE_POSTFLASH_INTERLOCK must be 0 or 1" >&2; exit 64 ;;
esac
case "$POSTFLASH_SKIP_STALE_RUNTIME" in
  0|1) ;;
  *) echo "ERR F407_POSTFLASH_SKIP_STALE_RUNTIME must be 0 or 1" >&2; exit 64 ;;
esac
if [ "$REQUIRE_POSTFLASH_INTERLOCK" = "1" ] && [ -z "$POSTFLASH_MANIFEST" ]; then
  echo "ERR F407_POSTFLASH_MANIFEST is required" >&2
  exit 64
fi

mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/summary.txt"
CHECKS="$OUT_DIR/checks.tsv"
ACCEPTANCE_START_FILE="$OUT_DIR/acceptance_start.json"
DISPATCH_FIXTURE_REPORT="$OUT_DIR/dispatch_fixture_integration.json"
DISPATCH_FIXTURE_LOG="$OUT_DIR/dispatch_fixture_integration.log"
: > "$SUMMARY"
: > "$CHECKS"

log() {
  echo "$*" | tee -a "$SUMMARY"
}

record() {
  local name="$1"
  local status="$2"
  local detail="${3:-}"
  printf "%s\t%s\t%s\n" "$name" "$status" "$detail" >> "$CHECKS"
}

capture() {
  local name="$1"
  local timeout_s="$2"
  shift 2
  local file="$OUT_DIR/${name}.txt"
  {
    echo "### $name"
    echo "\$ $*"
    echo
  } > "$file"
  timeout "$timeout_s" "$@" >> "$file" 2>&1
  local rc=$?
  if [ "$rc" -eq 0 ]; then
    record "$name" "OK" "$file"
  elif [ "$rc" -eq 124 ]; then
    record "$name" "TIMEOUT" "$file"
  else
    record "$name" "WARN" "rc=$rc $file"
  fi
  return 0
}

capture_sh() {
  local name="$1"
  local timeout_s="$2"
  local cmd="$3"
  capture "$name" "$timeout_s" bash -o pipefail -lc "$cmd"
}

capture_topic_once() {
  local name="$1"
  local topic="$2"
  local echo_args="--once"
  case "$topic" in
    /lab_fsd/fsd_v3_status|/lab_fsd/input_status|/lab_fsd/vision_objects|/lab_fsd/safety_gate|/lab_fsd/trajectory_scores|/lab_fsd/policy_tokens|/f407/firmware_info|/mppi/stats)
      echo_args="--once --full-length"
      ;;
  esac
  local total_timeout_s=$((TOPIC_ECHO_ATTEMPTS * TOPIC_ECHO_TIMEOUT_S + (TOPIC_ECHO_ATTEMPTS - 1) * TOPIC_ECHO_INTERVAL_S + 5))
  capture_sh "$name" "$total_timeout_s" \
    "for i in \$(seq 1 ${TOPIC_ECHO_ATTEMPTS}); do echo \"attempt=\$i topic=${topic}\"; timeout ${TOPIC_ECHO_TIMEOUT_S} ros2 topic echo ${echo_args} '${topic}' && exit 0; sleep ${TOPIC_ECHO_INTERVAL_S}; done; exit 124"
}

source_ros() {
  # ROS generated setup scripts are not nounset-safe on all Humble/TROS images.
  set +u
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash 2>/dev/null || true
  # shellcheck disable=SC1091
  source /opt/tros/humble/setup.bash 2>/dev/null || true
  # shellcheck disable=SC1091
  source "$HOME/ros2_ws/install/setup.bash" 2>/dev/null || true
  export LD_LIBRARY_PATH="/opt/tros/humble/lib:${LD_LIBRARY_PATH:-}"
  set -u
}

log "EMBODIED_V3_ACCEPTANCE"
log "out_dir: $OUT_DIR"
ACCEPTANCE_START_UNIX="$(date +%s)"
ACCEPTANCE_START_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
acceptance_start_rc=0
python3 - "$ACCEPTANCE_START_FILE" "$OUT_DIR" "$ACCEPTANCE_START_UNIX" "$ACCEPTANCE_START_ISO" "$$" <<'PY' || acceptance_start_rc=$?
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]).expanduser().resolve()
out_dir = Path(sys.argv[2]).expanduser().resolve()
payload = {
    "schema_version": "xrd-embodied-v3-acceptance-start-v1",
    "started_at": sys.argv[4],
    "started_at_unix": float(sys.argv[3]),
    "out_dir": out_dir.as_posix(),
    "pid": int(sys.argv[5]),
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
if [ "$acceptance_start_rc" -eq 0 ] && [ -s "$ACCEPTANCE_START_FILE" ]; then
  record "acceptance_start" "OK" "$ACCEPTANCE_START_FILE"
else
  record "acceptance_start" "FAIL" "rc=$acceptance_start_rc $ACCEPTANCE_START_FILE"
fi
log "time: $ACCEPTANCE_START_ISO"
log "acceptance_start_unix: $ACCEPTANCE_START_UNIX"
log "host: $(hostname 2>/dev/null || echo unknown)"
log "model_bin: $MODEL_BIN"
log "anomaly_bin: $ANOMALY_BIN"
log "mppi_bin: $MPPI_BIN"
log ""

source_ros

if [ -f "$DISPATCH_STUB_TOOL" ]; then
  mkdir -p "$(dirname "$DISPATCH_STUB_REPORT")"
  dispatch_stub_rc=0
  python3 "$DISPATCH_STUB_TOOL" \
    --domain-id "$DISPATCH_STUB_DOMAIN_ID" \
    --out "$DISPATCH_STUB_REPORT" \
    > "$OUT_DIR/dispatch_stub_integration_stdout.txt" 2>&1 || dispatch_stub_rc=$?
  if [ -s "$DISPATCH_STUB_REPORT" ]; then
    cp "$DISPATCH_STUB_REPORT" "$OUT_DIR/dispatch_stub_integration.json"
  fi
  if [ "$dispatch_stub_rc" -eq 0 ] && [ -s "$OUT_DIR/dispatch_stub_integration.json" ]; then
    record "dispatch_stub_integration" "OK" "$OUT_DIR/dispatch_stub_integration.json"
  else
    record "dispatch_stub_integration" "FAIL" "rc=$dispatch_stub_rc report=$DISPATCH_STUB_REPORT"
  fi
else
  record "dispatch_stub_integration" "FAIL" "$DISPATCH_STUB_TOOL missing"
fi

if [ -f "$DISPATCH_FIXTURE_TOOL" ]; then
  rm -f -- "$DISPATCH_FIXTURE_REPORT" "$DISPATCH_FIXTURE_LOG"
  dispatch_fixture_rc=0
  python3 "$DISPATCH_FIXTURE_TOOL" \
    --domain-id "$DISPATCH_FIXTURE_DOMAIN_ID" \
    --out "$DISPATCH_FIXTURE_REPORT" \
    > "$OUT_DIR/dispatch_fixture_integration_stdout.txt" 2>&1 || dispatch_fixture_rc=$?
  dispatch_fixture_fresh_rc=0
  python3 - "$DISPATCH_FIXTURE_REPORT" "$DISPATCH_FIXTURE_LOG" "$OUT_DIR" "$ACCEPTANCE_START_UNIX" \
    > "$OUT_DIR/dispatch_fixture_freshness.txt" 2>&1 <<'PY' || dispatch_fixture_fresh_rc=$?
import json
import sys
import time
from pathlib import Path

report_path = Path(sys.argv[1]).expanduser().resolve()
log_path = Path(sys.argv[2]).expanduser().resolve()
out_dir = Path(sys.argv[3]).expanduser().resolve()
acceptance_start = float(sys.argv[4])
if report_path.parent != out_dir or log_path.parent != out_dir:
    raise SystemExit("fixture artifacts are not direct children of acceptance OUT_DIR")
if not report_path.is_file() or not log_path.is_file():
    raise SystemExit("fixture report or log missing")
report = json.loads(report_path.read_text(encoding="utf-8"))
report_started = float(report.get("started_at_unix") or 0.0)
report_generated = float(report.get("generated_at_unix") or 0.0)
now = time.time()
if report_started < acceptance_start - 1.0:
    raise SystemExit("fixture report predates acceptance start")
if not report_started <= report_generated <= now + 5.0:
    raise SystemExit("fixture report timestamps are inconsistent")
if log_path.stat().st_mtime < acceptance_start - 1.0:
    raise SystemExit("fixture log predates acceptance start")
reported_log = Path(str((report.get("log") or {}).get("path") or "")).expanduser().resolve()
if reported_log != log_path:
    raise SystemExit("fixture report is not bound to its acceptance log")
print(f"fresh report={report_path} log={log_path} age_s={now - report_generated:.3f}")
PY
  if [ "$dispatch_fixture_rc" -eq 0 ] \
    && [ "$dispatch_fixture_fresh_rc" -eq 0 ] \
    && [ -s "$DISPATCH_FIXTURE_REPORT" ] \
    && [ -f "$DISPATCH_FIXTURE_LOG" ]; then
    record "dispatch_fixture_integration" "OK" "$DISPATCH_FIXTURE_REPORT log=$DISPATCH_FIXTURE_LOG"
  else
    record "dispatch_fixture_integration" "FAIL" "rc=$dispatch_fixture_rc freshness_rc=$dispatch_fixture_fresh_rc report=$DISPATCH_FIXTURE_REPORT log=$DISPATCH_FIXTURE_LOG"
  fi
else
  record "dispatch_fixture_integration" "FAIL" "$DISPATCH_FIXTURE_TOOL missing"
fi

capture_sh "system_identity" 5 \
  'date -Is; hostname; uname -a; ip -brief addr; echo ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-unset}; echo LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-unset}'
capture_sh "system_resources" 5 \
  'free -h; echo; df -h "$HOME" /tmp 2>/dev/null; echo; ps -eo pid,ppid,%cpu,%mem,comm,args --sort=-%mem | head -35'
capture_sh "bpu_status" 8 \
  'hrut_somstatus 2>/dev/null || sudo -n hrut_somstatus 2>/dev/null || echo "hrut_somstatus unavailable"'

check_model_bin() {
  local check_name="$1"
  local path="$2"
  local expected_sha="$3"
  if [ -f "$path" ]; then
    local actual_sha
    local size_bytes
    actual_sha="$(sha256sum "$path" | awk '{print $1}')"
    size_bytes="$(wc -c < "$path" | tr -d ' ')"
    log "${check_name}: present size=${size_bytes} sha256=${actual_sha}"
    if [ "$actual_sha" = "$expected_sha" ]; then
      record "$check_name" "OK" "$path"
    else
      record "$check_name" "WARN" "sha256=$actual_sha expected=$expected_sha"
    fi
  else
    log "${check_name}: missing"
    record "$check_name" "WARN" "$path missing"
  fi
}

check_model_bin "tiny_occ_risk_bin" "$MODEL_BIN" "$EXPECTED_OCC_SHA256"
check_model_bin "lab_anomaly_autoencoder_bin" "$ANOMALY_BIN" "$EXPECTED_ANOMALY_SHA256"
check_model_bin "mppi_cost_bin" "$MPPI_BIN" "$EXPECTED_MPPI_SHA256"

capture_sh "hobot_dnn_import" 8 \
  'python3 - <<'"'"'PY'"'"'
try:
    import hobot_dnn.pyeasy_dnn as dnn
    print("hobot_dnn import OK")
    print("module:", dnn)
except Exception as exc:
    print("hobot_dnn import FAIL:", repr(exc))
    raise SystemExit(1)
PY'

capture_sh "ros_nodes" 8 'ros2 node list | sort'
capture_sh "ros_topics" 8 'ros2 topic list | sort'
capture_sh "ros_services" 8 'ros2 service list | sort'
capture_sh "ros_actions" 8 'ros2 action list | sort'
capture_sh "ros_tf_frames_hint" 8 'ros2 topic echo --once /tf_static 2>/dev/null | head -120 || true'
capture_sh "physical_evidence_config" 20 \
  'echo "### mode"; ros2 param get /dispatch_server physical_evidence_mode 2>&1 || true; echo "### evidence_nodes"; ros2 node list 2>/dev/null | grep -E "^/(physical_evidence_gate|physical_sensor_evidence_bridge([_.-][A-Za-z0-9_.-]+)?)$" || true; echo "### service"; ros2 service list -t 2>&1 | grep -F /verify_physical_evidence || true; echo "### topics"; ros2 topic list -t 2>&1 | grep -E "^/pickup/(hardware_sensor_sample|physical_evidence(_request)?|physical_evidence_bridge_status)([[:space:]]|$)" || true'
capture_sh "device_links" 5 \
  'id; groups; for d in /dev/F407 /dev/LD14 /dev/lift_camera /dev/PT_CAM; do echo "### $d"; ls -l "$d" 2>&1 || true; readlink -f "$d" 2>&1 || true; done'

if [ -n "$POSTFLASH_MANIFEST" ]; then
  postflash_bundle_rc=0
  if [ -f "$POSTFLASH_BUNDLE_TOOL" ]; then
    python3 "$POSTFLASH_BUNDLE_TOOL" \
      --manifest "$POSTFLASH_MANIFEST" \
      --out-dir "$OUT_DIR" \
      --index "$OUT_DIR/f407_postflash_bundle_index.json" \
      > "$OUT_DIR/f407_postflash_bundle_stdout.txt" 2>&1 || postflash_bundle_rc=$?
  else
    postflash_bundle_rc=127
    printf '%s\n' "$POSTFLASH_BUNDLE_TOOL missing" > "$OUT_DIR/f407_postflash_bundle_stdout.txt"
  fi
  if [ "$postflash_bundle_rc" -eq 0 ] \
    && [ -s "$OUT_DIR/f407_postflash_manifest.json" ] \
    && [ -s "$OUT_DIR/f407_postflash_bundle_index.json" ] \
    && [ -s "$OUT_DIR/f407_interlock_report.json" ]; then
    record "f407_postflash_manifest" "OK" "$OUT_DIR/f407_postflash_manifest.json"
    record "f407_interlock_report" "OK" "$OUT_DIR/f407_interlock_report.json"
  else
    record "f407_postflash_manifest" "FAIL" "rc=$postflash_bundle_rc manifest=$POSTFLASH_MANIFEST"
    record "f407_interlock_report" "FAIL" "post-flash bundle staging failed"
  fi
elif [ -s "$HOME/f407_interlock_evidence/latest.json" ]; then
  cp "$HOME/f407_interlock_evidence/latest.json" "$OUT_DIR/f407_interlock_report.json"
  record "f407_interlock_report" "OK" "$OUT_DIR/f407_interlock_report.json"
  if [ "$REQUIRE_POSTFLASH_INTERLOCK" = "1" ]; then
    record "f407_postflash_manifest" "FAIL" "F407_POSTFLASH_MANIFEST is required"
  else
    record "f407_postflash_manifest" "WARN" "exact post-flash orchestration manifest not supplied"
  fi
else
  record "f407_interlock_report" "FAIL" "$HOME/f407_interlock_evidence/latest.json missing"
  if [ "$REQUIRE_POSTFLASH_INTERLOCK" = "1" ]; then
    record "f407_postflash_manifest" "FAIL" "F407_POSTFLASH_MANIFEST is required"
  else
    record "f407_postflash_manifest" "WARN" "exact post-flash orchestration manifest not supplied"
  fi
fi

if [ -n "$POSTFLASH_MANIFEST" ] \
  && [ "$REQUIRE_RUNTIME_PREP" != "1" ] \
  && [ "$POSTFLASH_SKIP_STALE_RUNTIME" = "1" ]; then
  record "runtime_prepare_report" "WARN" "skipped optional stale runtime preparation; post-flash manifest supplied"
elif { [ "$REQUIRE_RUNTIME_PREP" = "1" ] || [ "$RUNTIME_PREP_EXPLICIT" = "1" ]; } \
  && [ -s "$RUNTIME_PREP_PATH" ]; then
  cp "$RUNTIME_PREP_PATH" "$OUT_DIR/runtime_prepare_report.json"
  record "runtime_prepare_report" "OK" "$OUT_DIR/runtime_prepare_report.json"
  runtime_state_dir="$(dirname "$RUNTIME_PREP_PATH")"
  runtime_mppi_ok=1
  for spec in \
    "latest_mppi_stats.txt:runtime_mppi_stats.txt" \
    "latest_mppi_cmd_vel_proposed.txt:runtime_mppi_cmd_vel_proposed.txt" \
    "latest_cmd_vel_publishers.txt:runtime_mppi_cmd_vel_publishers.txt"
  do
    IFS=: read -r src_name dst_name <<< "$spec"
    if [ -s "$runtime_state_dir/$src_name" ]; then
      cp "$runtime_state_dir/$src_name" "$OUT_DIR/$dst_name"
    else
      runtime_mppi_ok=0
    fi
  done
  if [ "$runtime_mppi_ok" = "1" ]; then
    record "runtime_mppi_raw_evidence" "OK" "$OUT_DIR/runtime_mppi_stats.txt"
  elif [ "$REQUIRE_RUNTIME_PREP" = "1" ]; then
    record "runtime_mppi_raw_evidence" "FAIL" "runtime MPPI raw evidence incomplete"
  else
    record "runtime_mppi_raw_evidence" "WARN" "runtime MPPI raw evidence incomplete"
  fi
  if [ -s "$runtime_state_dir/latest_postflash_manifest.json" ]; then
    cp "$runtime_state_dir/latest_postflash_manifest.json" "$OUT_DIR/runtime_postflash_manifest.json"
  fi
  if [ -s "$runtime_state_dir/latest_postflash_bundle_index.json" ]; then
    cp "$runtime_state_dir/latest_postflash_bundle_index.json" "$OUT_DIR/runtime_postflash_bundle_index.json"
  fi
elif [ "$REQUIRE_RUNTIME_PREP" = "1" ]; then
  record "runtime_prepare_report" "FAIL" "$RUNTIME_PREP_PATH missing"
else
  record "runtime_prepare_report" "WARN" "runtime preparation not requested; stale report ignored for this read-only check"
fi
capture_sh "kernel_device_tail" 5 \
  'dmesg 2>/dev/null | tail -120 || echo "dmesg unavailable without sudo"'
capture_sh "ros_topic_info_verbose" 12 \
  'for t in /cmd_vel /scan /scan_depth /odom /map /lab_fsd/fsd_v3_status /lab_fsd/future_risk /lab_fsd/input_status /lab_fsd/vision_bev /lab_fsd/vision_risk /lab_fsd/vision_objects /lab_fsd/safety_gate /lab_fsd/shadow_path /lab_fsd/trajectory_scores /lab_fsd/bev /lab_fsd/future_bev /lab_fsd/policy_tokens /lab_fsd/anomaly_score /mppi/cmd_vel_proposed /mppi/stats /diagnostics /lift_status /f407/estop_latched /f407/cmd_vel_expired /f407/firmware_identity_valid /f407/firmware_info; do echo "### $t"; ros2 topic info "$t" -v 2>&1 || true; done'
capture_sh "ros_node_info_verbose" 15 \
  'for n in $(ros2 node list 2>/dev/null | sort); do echo "### $n"; timeout 3 ros2 node info "$n" 2>&1 || true; done'

for topic in \
  /scan \
  /scan_depth \
  /odom \
  /map \
  /lab_fsd/fsd_v3_status \
  /lab_fsd/future_risk \
  /lab_fsd/input_status \
  /lab_fsd/vision_bev \
  /lab_fsd/vision_risk \
  /lab_fsd/vision_objects \
  /lab_fsd/safety_gate \
  /lab_fsd/shadow_path \
  /lab_fsd/trajectory_scores \
  /lab_fsd/bev \
  /lab_fsd/future_bev \
  /diagnostics \
  /lift_status \
  /f407/estop_latched \
  /f407/cmd_vel_expired \
  /f407/firmware_identity_valid \
  /f407/firmware_info
do
  safe_name="$(echo "$topic" | sed 's#^/##; s#[/ ]#_#g')"
  capture_topic_once "topic_${safe_name}" "$topic"
done

MPPI_NODE_PROBE="$OUT_DIR/mppi_node_probe.txt"
if timeout 8 ros2 node list > "$MPPI_NODE_PROBE" 2>&1 \
  && grep -Eq '(^|/)mppi_node$' "$MPPI_NODE_PROBE"; then
  capture_topic_once "topic_mppi_cmd_vel_proposed" "/mppi/cmd_vel_proposed"
  capture_topic_once "topic_mppi_stats" "/mppi/stats"
else
  printf '%s\n' "MPPI node absent; live proposed Twist sampling is optional in normal verify." \
    > "$OUT_DIR/topic_mppi_cmd_vel_proposed.txt"
  printf '%s\n' "MPPI node absent; live stats sampling is optional in normal verify." \
    > "$OUT_DIR/topic_mppi_stats.txt"
  record "topic_mppi_cmd_vel_proposed" "WARN" "MPPI node absent; runtime/raw evidence is audited separately"
  record "topic_mppi_stats" "WARN" "MPPI node absent; runtime/raw evidence is audited separately"
fi
capture_topic_once "topic_lab_fsd_policy_tokens" "/lab_fsd/policy_tokens"

capture_sh "cmd_vel_publishers" 8 'ros2 topic info /cmd_vel -v'
if timeout 8 bash -lc "ros2 topic info /cmd_vel -v" > "$OUT_DIR/cmd_vel_publishers_check.txt" 2>&1; then
  if grep -i "lab_fsd" "$OUT_DIR/cmd_vel_publishers_check.txt" >/dev/null; then
    record "lab_fsd_not_cmd_vel_publisher" "FAIL" "lab_fsd appears in /cmd_vel publishers"
    log "lab_fsd_cmd_vel_check: FAIL"
  else
    record "lab_fsd_not_cmd_vel_publisher" "OK" "no lab_fsd publisher on /cmd_vel"
    log "lab_fsd_cmd_vel_check: OK"
  fi
  if grep -i "mppi" "$OUT_DIR/cmd_vel_publishers_check.txt" >/dev/null; then
    record "mppi_not_cmd_vel_publisher" "FAIL" "mppi appears in /cmd_vel publishers"
    log "mppi_cmd_vel_check: FAIL"
  else
    record "mppi_not_cmd_vel_publisher" "OK" "no mppi publisher on /cmd_vel"
    log "mppi_cmd_vel_check: OK"
  fi
else
  record "lab_fsd_not_cmd_vel_publisher" "WARN" "could not inspect /cmd_vel"
  record "mppi_not_cmd_vel_publisher" "WARN" "could not inspect /cmd_vel"
  log "lab_fsd_cmd_vel_check: WARN"
  log "mppi_cmd_vel_check: WARN"
fi

if [ -x "$HOME/tools/data_loop_status.sh" ]; then
  capture_sh "data_loop_status" 8 'bash "$HOME/tools/data_loop_status.sh"'
else
  record "data_loop_status" "WARN" "$HOME/tools/data_loop_status.sh missing"
fi

capture_sh "cockpit_blackbox_recent" 5 \
  'latest="$(ls -1t "$HOME"/blackbox/bb-*.jsonl 2>/dev/null | head -1)"; [ -n "$latest" ] && [ -f "$latest" ] || exit 1; echo "### $latest"; tail -400 "$latest"'

capture_sh "recent_logs" 5 \
  'for f in /tmp/embodied_v3_stack.log /tmp/embodied_v3_stack_status.txt /tmp/lab_fsd_shadow.log /tmp/video2_capture.log /tmp/slam_mapping.log /tmp/serial_f407.log; do [ -f "$f" ] && { echo "### $f"; tail -100 "$f"; }; done'

ok_count="$(awk -F '\t' '$2=="OK"{n++} END{print n+0}' "$CHECKS")"
warn_count="$(awk -F '\t' '$2=="WARN"{n++} END{print n+0}' "$CHECKS")"
timeout_count="$(awk -F '\t' '$2=="TIMEOUT"{n++} END{print n+0}' "$CHECKS")"
fail_count="$(awk -F '\t' '$2=="FAIL"{n++} END{print n+0}' "$CHECKS")"

log ""
log "checks: OK=${ok_count} WARN=${warn_count} TIMEOUT=${timeout_count} FAIL=${fail_count}"
log "checks_tsv: $CHECKS"
log "done: $OUT_DIR"

AUDIT_TOOL="$HOME/tools/embodied_v3_acceptance_audit.py"
if [ -f "$AUDIT_TOOL" ]; then
  record "acceptance_audit_tool" "OK" "$AUDIT_TOOL"
  audit_rc=0
  data_run="$DATA_RUN_OVERRIDE"
  if [ -n "$DATA_RUN_FILE_OVERRIDE" ]; then
    if [ -f "$DATA_RUN_FILE_OVERRIDE" ]; then
      data_run="$(cat "$DATA_RUN_FILE_OVERRIDE" 2>/dev/null || true)"
    else
      data_run=""
      log "data_run_binding: FAIL missing $DATA_RUN_FILE_OVERRIDE"
    fi
  elif [ -z "$data_run" ] && [ -f "$HOME/data_loop_runs/current_run.txt" ]; then
    data_run="$(cat "$HOME/data_loop_runs/current_run.txt" 2>/dev/null || true)"
  fi
  if [ -n "$data_run" ] && [ -d "$data_run" ]; then
    printf '%s\n' "$data_run" > "$OUT_DIR/data_run_reference.txt"
    python3 "$AUDIT_TOOL" --accept-dir "$OUT_DIR" --data-run "$data_run" --require-data-run \
      > "$OUT_DIR/audit_stdout.txt" 2>&1 || audit_rc=$?
  else
    audit_require_args=()
    if [ "$REQUIRE_DATA_RUN" = "1" ]; then
      audit_require_args=(--require-data-run)
    fi
    python3 "$AUDIT_TOOL" --accept-dir "$OUT_DIR" "${audit_require_args[@]}" \
      > "$OUT_DIR/audit_stdout.txt" 2>&1 || audit_rc=$?
  fi
  log "audit: $OUT_DIR/audit_report.json"
  if [ "$audit_rc" -ne 0 ]; then
    log "audit_result: FAIL rc=$audit_rc"
    exit "$audit_rc"
  fi
else
  record "acceptance_audit_tool" "FAIL" "$AUDIT_TOOL missing"
  log "audit_result: FAIL audit tool missing: $AUDIT_TOOL"
  exit 11
fi

if [ "$fail_count" -gt 0 ]; then
  exit 2
fi
exit 0
