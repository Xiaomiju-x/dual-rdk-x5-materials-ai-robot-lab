#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${X5_TRIFLOW_ROOT:-$HOME/x5_tribev_flow_successor}"
RUN_DIR="${X5_TRIFLOW_COLLECTOR_RUN_DIR:-$HOME/.local/state/x5_triflow_collector}"
STATE_FILE="$RUN_DIR/collector.state"
LOG_FILE="$RUN_DIR/collector.log"
PYTHON_BIN="${X5_TRIFLOW_PYTHON:-python3}"
PARAMS="$ROOT/config/x5_tribev_collector.ros.yaml"
NODE="$ROOT/runtime/x5_tribev_readonly_collector.py"
IDENTITY_HELPER="$ROOT/runtime/process_identity.sh"

for required in "$PARAMS" "$NODE" "$IDENTITY_HELPER"; do
  if [[ ! -f "$required" ]]; then
    printf 'missing collector file: %s\n' "$required" >&2
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
    printf 'X5 TriBEV read-only collector already running: pid=%s\n' "$old_pid"
    exit 0
  fi
  rm -f -- "$STATE_FILE"
fi

source /opt/ros/humble/setup.bash
if [[ -f "$HOME/ros2_ws/install/setup.bash" ]]; then
  source "$HOME/ros2_ws/install/setup.bash"
fi
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

nohup nice -n 12 "$PYTHON_BIN" "$NODE" --ros-args \
  --params-file "$PARAMS" >"$LOG_FILE" 2>&1 &
pid=$!
sleep 0.1
if ! kill -0 "$pid" 2>/dev/null; then
  printf 'collector exited during startup; log follows:\n' >&2
  tail -80 "$LOG_FILE" >&2 || true
  exit 3
fi
x5_triflow_write_state "$STATE_FILE" "$pid" "$NODE" "$ROOT"
sleep 0.9
if ! x5_triflow_process_matches "$STATE_FILE" "$NODE"; then
  printf 'collector exited during startup or identity changed; log follows:\n' >&2
  tail -80 "$LOG_FILE" >&2 || true
  rm -f -- "$STATE_FILE"
  exit 3
fi

printf 'X5 TriBEV read-only collector started\n'
printf 'pid=%s\nlog=%s\n' "$pid" "$LOG_FILE"
printf 'ros_outputs=0 control_authority=none validated_demo_unchanged=true\n'
