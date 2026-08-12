#!/usr/bin/env bash
set -eo pipefail
set +u

usage() {
  cat <<'USAGE'
Usage:
  data_loop_stop.sh [RUN_DIR] [options]

Stop the current data-loop run, export video2 if attached, copy video2 outputs,
and generate manifest.json, hashes.sha256, and LeRobot/RoboMimic index files.

Options:
  --base DIR             Base directory. Default: $DATA_LOOP_BASE or ~/data_loop_runs.
  --video2-session DIR   Override or add the video2 session to copy.
  --no-video2-copy       Do not copy video2 output into this run.
  -h, --help             Show this help.
USAGE
}

die() {
  echo "ERR $*" >&2
  exit 2
}

warn() {
  echo "WARN $*" >&2
}

script_dir() {
  local src="${BASH_SOURCE[0]}"
  while [ -h "$src" ]; do
    local dir
    dir="$(cd -P "$(dirname "$src")" >/dev/null 2>&1 && pwd)"
    src="$(readlink "$src")"
    [[ "$src" != /* ]] && src="$dir/$src"
  done
  cd -P "$(dirname "$src")" >/dev/null 2>&1 && pwd
}

setup_ros_if_available() {
  export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
  export LD_LIBRARY_PATH="/opt/tros/humble/lib:${LD_LIBRARY_PATH:-}"
  if [ -f /opt/ros/humble/setup.bash ]; then
    source /opt/ros/humble/setup.bash
  fi
  if [ -f /opt/tros/humble/setup.bash ]; then
    source /opt/tros/humble/setup.bash
  fi
  if [ -f "$HOME/ros2_ws/install/setup.bash" ]; then
    source "$HOME/ros2_ws/install/setup.bash"
  fi
}

copy_tree() {
  local src="$1"
  local dst="$2"
  mkdir -p "$dst"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete "$src"/ "$dst"/
  else
    cp -a "$src"/. "$dst"/
  fi
}

base="${DATA_LOOP_BASE:-$HOME/data_loop_runs}"
run_dir=""
copy_video2=1
video2_session_override=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --base)
      shift || die "--base needs a value"
      base="${1:-}"
      ;;
    --video2-session)
      shift || die "--video2-session needs a value"
      video2_session_override="${1:-}"
      ;;
    --no-video2-copy)
      copy_video2=0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      die "unknown option '$1'"
      ;;
    *)
      if [ -n "$run_dir" ]; then
        die "multiple RUN_DIR arguments"
      fi
      run_dir="$1"
      ;;
  esac
  shift
done

if [ -z "$run_dir" ]; then
  if [ -f "$base/current_run.txt" ]; then
    run_dir="$(cat "$base/current_run.txt")"
  else
    die "no current run; pass RUN_DIR explicitly"
  fi
fi

state_file="$run_dir/state.env"
[ -f "$state_file" ] || die "state file not found: $state_file"
# shellcheck disable=SC1090
source "$state_file"

tools_dir="${TOOLS_DIR:-$(script_dir)}"
python_bin="${PYTHON:-python3}"
mkdir -p "$run_dir/logs"

if [ -n "${ROSBAG_PID:-}" ] && kill -0 "$ROSBAG_PID" 2>/dev/null; then
  echo "Stopping rosbag pid $ROSBAG_PID"
  kill -INT "$ROSBAG_PID" 2>/dev/null || true
  for _ in $(seq 1 35); do
    if ! kill -0 "$ROSBAG_PID" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if kill -0 "$ROSBAG_PID" 2>/dev/null; then
    warn "rosbag pid $ROSBAG_PID still alive after SIGINT; sending SIGTERM"
    kill "$ROSBAG_PID" 2>/dev/null || true
    sleep 2
  fi
else
  warn "rosbag pid is not alive: ${ROSBAG_PID:-unset}"
fi

setup_ros_if_available
if command -v ros2 >/dev/null 2>&1; then
  ros2 topic list > "$run_dir/topics_at_stop.txt" 2>"$run_dir/logs/topics_at_stop.err" || true
  ros2 topic list -t > "$run_dir/topics_with_types_at_stop.txt" 2>"$run_dir/logs/topics_with_types_at_stop.err" || true
  ros2 node list > "$run_dir/nodes_at_stop.txt" 2>"$run_dir/logs/nodes_at_stop.err" || true
  if [ -n "${BAG_DIR:-}" ] && [ -d "$BAG_DIR" ]; then
    ros2 bag info "$BAG_DIR" > "$run_dir/logs/rosbag_info.txt" 2>&1 || \
      warn "ros2 bag info failed; see logs/rosbag_info.txt"
  fi
else
  warn "ros2 not available during stop; skipping end topic snapshot"
fi

cmd_vel_verify_rc=0
cmd_vel_evidence="$run_dir/logs/cmd_vel_evidence.json"
cmd_vel_tool="$tools_dir/verify_cmd_vel_bag.py"
if [ -f "$cmd_vel_tool" ] && [ -n "${BAG_DIR:-}" ] && [ -d "$BAG_DIR" ]; then
  "$python_bin" "$cmd_vel_tool" \
    --bag-dir "$BAG_DIR" \
    --expect "${CMD_VEL_EXPECTATION:-any}" \
    --out "$cmd_vel_evidence" \
    > "$run_dir/logs/cmd_vel_evidence.stdout" 2>&1 || cmd_vel_verify_rc=$?
  if [ "$cmd_vel_verify_rc" -ne 0 ]; then
    warn "offline /cmd_vel verification failed; see $cmd_vel_evidence"
  fi
else
  warn "offline /cmd_vel verifier or bag directory missing"
  cmd_vel_verify_rc=12
fi

video2_session="${video2_session_override:-${VIDEO2_SESSION:-}}"
if [ "${VIDEO2_STARTED_BY_DATA_LOOP:-0}" = "1" ] && [ -n "$video2_session" ]; then
  video2_stop="$tools_dir/video2_stop_capture.sh"
  if [ -f "$video2_stop" ]; then
    bash "$video2_stop" "$video2_session" > "$run_dir/logs/video2_stop.log" 2>&1 || warn "video2 stop failed; see logs/video2_stop.log"
  else
    warn "video2_stop_capture.sh not found in $tools_dir; copying raw video2 session if present"
  fi
fi

if [ "$copy_video2" = "1" ] && [ -n "$video2_session" ]; then
  if [ -d "$video2_session" ]; then
    copy_tree "$video2_session" "$run_dir/video2"
  else
    warn "video2 session missing, not copied: $video2_session"
  fi
fi

{
  printf 'STOPPED_AT=%q\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'STOP_EPOCH=%q\n' "$(date +%s)"
  printf 'VIDEO2_COPIED=%q\n' "$copy_video2"
} >> "$state_file"

"$python_bin" "$tools_dir/data_loop_finalize.py" --run-dir "$run_dir" --status stopped
"$python_bin" "$tools_dir/data_loop_to_lerobot.py" --run-dir "$run_dir" \
  > "$run_dir/logs/data_loop_to_lerobot.log" 2>&1 || \
  warn "training skeleton export failed; see logs/data_loop_to_lerobot.log"

audit_rc=0
audit_tool="$tools_dir/embodied_v3_acceptance_audit.py"
audit_json="$run_dir/logs/data_loop_audit.json"
audit_txt="$run_dir/logs/data_loop_audit.txt"
if [ -f "$audit_tool" ]; then
  "$python_bin" "$audit_tool" --data-run "$run_dir" --out "$audit_json" --text-out "$audit_txt" \
    > "$run_dir/logs/data_loop_audit.stdout" 2>&1 || audit_rc=$?
else
  warn "embodied_v3_acceptance_audit.py missing; data-loop audit cannot run"
  audit_rc=11
fi

if [ "$cmd_vel_verify_rc" -ne 0 ] && [ "$audit_rc" -eq 0 ]; then
  audit_rc="$cmd_vel_verify_rc"
fi
if [ "$audit_rc" -ne 0 ]; then
  echo "DATA_LOOP_STOPPED_WITH_AUDIT_FAIL"
else
  echo "DATA_LOOP_STOPPED"
fi
echo "run_dir: $run_dir"
echo "manifest: $run_dir/manifest.json"
echo "hashes: $run_dir/hashes.sha256"
echo "lerobot_index: $run_dir/exports/lerobot/episode_index.jsonl"
echo "robomimic_index: $run_dir/exports/robomimic/demo_index.jsonl"
echo "training_skeleton: $run_dir/exports/training_skeleton"
echo "cmd_vel_evidence: $cmd_vel_evidence"
echo "audit_json: $audit_json"
echo "audit_txt: $audit_txt"

if [ "$audit_rc" -ne 0 ]; then
  exit "$audit_rc"
fi
