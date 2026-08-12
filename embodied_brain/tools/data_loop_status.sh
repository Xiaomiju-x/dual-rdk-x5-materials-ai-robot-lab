#!/usr/bin/env bash
set -eo pipefail
set +u

base="${DATA_LOOP_BASE:-$HOME/data_loop_runs}"
run_dir="${1:-}"
if [ -z "$run_dir" ] && [ -f "$base/current_run.txt" ]; then
  run_dir="$(cat "$base/current_run.txt")"
fi

if [ -z "$run_dir" ]; then
  echo "DATA_LOOP_STATUS none"
  exit 0
fi

state_file="$run_dir/state.env"
if [ ! -f "$state_file" ]; then
  echo "DATA_LOOP_STATUS missing_state"
  echo "$run_dir"
  exit 1
fi

# shellcheck disable=SC1090
source "$state_file"

echo "DATA_LOOP_STATUS"
echo "run_id: ${RUN_ID:-unknown}"
echo "run_dir: $run_dir"
echo "storage: ${STORAGE:-unknown}"
echo "bag_dir: ${BAG_DIR:-unknown}"

if [ -n "${ROSBAG_PID:-}" ] && kill -0 "$ROSBAG_PID" 2>/dev/null; then
  echo "rosbag: running pid=$ROSBAG_PID"
else
  echo "rosbag: stopped pid=${ROSBAG_PID:-unset}"
fi

if [ -n "${VIDEO2_SESSION:-}" ]; then
  echo "video2_session: $VIDEO2_SESSION"
fi

if [ -f "$run_dir/manifest.json" ]; then
  echo "manifest: $run_dir/manifest.json"
else
  echo "manifest: pending"
fi

if [ -f "$run_dir/topics_recorded.txt" ]; then
  echo "topics_recorded:"
  sed 's/^/  /' "$run_dir/topics_recorded.txt" | head -80
fi
