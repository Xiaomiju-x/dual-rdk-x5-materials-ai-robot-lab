#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${X5_TRIFLOW_ROOT:-$HOME/x5_tribev_flow_successor}"
RUN_DIR="${X5_TRIFLOW_RUN_DIR:-$HOME/.local/state/x5_triflow_shadow}"
STATE_FILE="$RUN_DIR/monitor.state"
NODE="$ROOT/runtime/x5_tribev_shadow_node.py"
IDENTITY_HELPER="$ROOT/runtime/process_identity.sh"
if [[ ! -f "$STATE_FILE" ]]; then
  printf 'X5-TriBEV-Flow shadow monitor is not running\n'
  exit 0
fi

if [[ ! -f "$IDENTITY_HELPER" || ! -f "$NODE" ]]; then
  printf 'candidate identity files are missing; refusing to signal any PID\n' >&2
  exit 2
fi
NODE="$(realpath "$NODE")"
source "$IDENTITY_HELPER"
if ! x5_triflow_process_matches "$STATE_FILE" "$NODE"; then
  rm -f -- "$STATE_FILE"
  printf 'removed stale candidate state; no process was signalled\n'
  exit 0
fi

pid="$(x5_triflow_state_value "$STATE_FILE" pid)"
kill -TERM "$pid"
for _ in $(seq 1 50); do
  x5_triflow_process_matches "$STATE_FILE" "$NODE" || break
  sleep 0.1
done
if x5_triflow_process_matches "$STATE_FILE" "$NODE"; then
  printf 'STOP_FAILED: candidate pid=%s ignored SIGTERM; state retained\n' "$pid" >&2
  exit 3
fi
rm -f -- "$STATE_FILE"
printf 'X5-TriBEV-Flow shadow monitor stopped; validated demo was not changed\n'
