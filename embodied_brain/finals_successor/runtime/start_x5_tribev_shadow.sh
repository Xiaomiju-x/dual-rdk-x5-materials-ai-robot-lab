#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${X5_TRIFLOW_ROOT:-$HOME/x5_tribev_flow_successor}"
RUN_DIR="${X5_TRIFLOW_RUN_DIR:-$HOME/.local/state/x5_triflow_shadow}"
STATE_FILE="$RUN_DIR/monitor.state"
LOG_FILE="$RUN_DIR/monitor.log"
PYTHON_BIN="${X5_TRIFLOW_PYTHON:-python3}"

TINY_BIN="$ROOT/bpu/artifacts/tiny_occ_flow/90e01859991c2eab/tiny_occ_flow.bin"
CAM_BIN="$ROOT/bpu/artifacts/cam_sem_lite/cb582808a90ae93c/cam_sem_lite.bin"
PARAMS="$ROOT/config/x5_tribev_shadow.ros.yaml"
NODE="$ROOT/runtime/x5_tribev_shadow_node.py"
IDENTITY_HELPER="$ROOT/runtime/process_identity.sh"

for required in "$TINY_BIN" "$CAM_BIN" "$PARAMS" "$NODE" "$IDENTITY_HELPER"; do
  if [[ ! -f "$required" ]]; then
    printf 'missing candidate file: %s\n' "$required" >&2
    exit 2
  fi
done

ROOT="$(cd -- "$ROOT" && pwd -P)"
NODE="$(realpath "$NODE")"
source "$IDENTITY_HELPER"
mkdir -p "$RUN_DIR"
if [[ -f "$STATE_FILE" ]]; then
  if x5_triflow_process_matches "$STATE_FILE" "$NODE"; then
    old_pid="$(x5_triflow_state_value "$STATE_FILE" pid)"
    printf 'X5-TriBEV-Flow shadow monitor already running: pid=%s\n' "$old_pid"
    exit 0
  fi
  rm -f -- "$STATE_FILE"
fi

source /opt/ros/humble/setup.bash
if [[ -f "$HOME/ros2_ws/install/setup.bash" ]]; then
  source "$HOME/ros2_ws/install/setup.bash"
fi
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

nohup nice -n 10 "$PYTHON_BIN" "$NODE" --ros-args \
  --params-file "$PARAMS" \
  -p "tiny_model_bin:=$TINY_BIN" \
  -p "cam_model_bin:=$CAM_BIN" >"$LOG_FILE" 2>&1 &
pid=$!
sleep 0.1
if ! kill -0 "$pid" 2>/dev/null; then
  printf 'candidate exited during startup; log follows:\n' >&2
  tail -80 "$LOG_FILE" >&2 || true
  exit 3
fi
x5_triflow_write_state "$STATE_FILE" "$pid" "$NODE" "$ROOT"
sleep 0.9
if ! kill -0 "$pid" 2>/dev/null; then
  printf 'candidate exited during startup; log follows:\n' >&2
  tail -80 "$LOG_FILE" >&2 || true
  rm -f -- "$STATE_FILE"
  exit 3
fi

printf 'X5-TriBEV-Flow shadow monitor started\n'
printf 'pid=%s\nlog=%s\n' "$pid" "$LOG_FILE"
printf 'authority=none namespace=/x5_triflow_shadow validated_demo_unchanged=true\n'
