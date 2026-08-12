#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${X5_TRIFLOW_ROOT:-$HOME/x5_tribev_flow_successor}"
RUN_DIR="${X5_TRIFLOW_COLLECTOR_RUN_DIR:-$HOME/.local/state/x5_triflow_collector}"
STATE_FILE="$RUN_DIR/collector.state"
LOG_FILE="$RUN_DIR/collector.log"
NODE="$ROOT/runtime/x5_tribev_readonly_collector.py"
IDENTITY_HELPER="$ROOT/runtime/process_identity.sh"

printf 'candidate=x5-tribev-readonly-collector-v1\n'
printf 'ros_outputs=0 control_authority=none\n'
printf 'validated_entry=bash ~/tools/finals_lift_nav_demo.sh\n'
if [[ -f "$STATE_FILE" && -f "$IDENTITY_HELPER" && -f "$NODE" ]]; then
  NODE="$(realpath "$NODE")"
  source "$IDENTITY_HELPER"
  if x5_triflow_process_matches "$STATE_FILE" "$NODE"; then
    pid="$(x5_triflow_state_value "$STATE_FILE" pid)"
    printf 'state=RUNNING pid=%s\n' "$pid"
  else
    printf 'state=STALE_IDENTITY_NO_SIGNAL\n'
  fi
else
  printf 'state=STOPPED\n'
fi
if [[ -f "$LOG_FILE" ]]; then
  printf 'log=%s\n' "$LOG_FILE"
  tail -30 "$LOG_FILE"
fi
