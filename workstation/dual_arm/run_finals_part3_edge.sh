#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/rdk/xrd_finals_part3
ORCHESTRATOR="$ROOT/workstation/dual_arm/finals_part3_orchestrator.py"
MODE=--execute
PAUSE=false

for argument in "$@"; do
  case "$argument" in
    --execute) MODE=--execute ;;
    --validate-only) MODE=--validate-only ;;
    --plan-only) MODE=--plan-only ;;
    --desktop) PAUSE=true ;;
    *) echo "Unknown argument: $argument" >&2; exit 2 ;;
  esac
done

test "$(id -un)" = sunrise
test "$(hostname)" = embodied-x5
test -f "$ORCHESTRATOR"

exec 9>/tmp/xrd_finals_part3_demo.lock
if ! flock -n 9; then
  echo "Finals Part 3 is already running." >&2
  exit 3
fi

mkdir -p "$HOME/finals_demo_logs"
timestamp=$(date +%Y%m%d_%H%M%S)
log="$HOME/finals_demo_logs/finals_part3_${timestamp}.log"

set +e
cd "$ROOT"
python3 -u "$ORCHESTRATOR" "$MODE" 2>&1 | tee "$log"
status=${PIPESTATUS[0]}
set -e

echo
if test "$status" -eq 0; then
  case "$MODE" in
    --execute) echo "CLOSED_LOOP_DONE / physical demo completed successfully" ;;
    --validate-only) echo "VALIDATE_ONLY_PASS / no robot motion command sent" ;;
    --plan-only) echo "PLAN_ONLY_PASS / no network or robot motion command sent" ;;
  esac
else
  echo "FAILED / exit code $status"
fi
echo "Log: $log"

if test "$PAUSE" = true; then
  read -r -p "Press Enter to close this window... " _
fi
exit "$status"
