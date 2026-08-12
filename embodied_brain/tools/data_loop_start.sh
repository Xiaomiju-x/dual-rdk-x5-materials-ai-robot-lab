#!/usr/bin/env bash
set -eo pipefail
set +u

usage() {
  cat <<'USAGE'
Usage:
  data_loop_start.sh [options]

Start one data-closed-loop run under $HOME/data_loop_runs.
The script records rosbag2, prefers MCAP when available, falls back to sqlite3,
and can start the existing video2 overlay recorder.

Options:
  --name NAME             Append a readable label to the run id.
  --out DIR              Base directory for runs. Default: $DATA_LOOP_BASE or ~/data_loop_runs.
  --storage MODE         auto|mcap|sqlite3. Default: auto.
  --all                  Record all ROS topics with ros2 bag record -a.
  --topic TOPIC          Add one exact topic to the default filtered topic set.
  --topic-regex REGEX    Regex used to select currently visible topics.
  --required-topic TOPIC  Add one required topic gate before recording.
  --no-required-topic-gate
                         Disable the default required-topic gate.
  --cmd-vel-expect MODE   any|zero|nonzero semantic gate for recorded /cmd_vel.
                         Default: DATA_LOOP_CMD_VEL_EXPECTATION or any.
  --video2               Start video2 capture with video2_start_capture.sh.
  --no-video2            Do not start or attach video2.
  --attach-video2 DIR    Attach an existing video2 session and copy it on stop.
  -h, --help             Show this help.

Environment:
  DATA_LOOP_BASE         Run base directory.
  DATA_LOOP_STORAGE      auto|mcap|sqlite3.
  DATA_LOOP_VIDEO2       auto|start|skip.
  DATA_LOOP_TOPIC_REGEX  Overrides the default topic selection regex.
  DATA_LOOP_REQUIRED_TOPIC_GATE
                        1 enables the required-topic gate, 0 disables it.
  DATA_LOOP_REQUIRED_TOPICS
                        Optional comma/space separated required-topic list.
  DATA_LOOP_CMD_VEL_EXPECTATION
                        any|zero|nonzero; checked by offline bag deserialization.
  ROS_DOMAIN_ID          Passed through to ROS 2. Default: 0.
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

setup_ros() {
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

  command -v ros2 >/dev/null 2>&1 || die "ros2 not found; source ROS 2 before starting data loop"
}

sanitize_name() {
  printf '%s' "$1" | tr -cs 'A-Za-z0-9._-' '_' | sed -E 's/^_+//; s/_+$//; s/_{2,}/_/g'
}

has_mcap_storage() {
  ros2 pkg prefix rosbag2_storage_mcap >/dev/null 2>&1
}

select_storage() {
  local requested="$1"
  case "$requested" in
    auto)
      if has_mcap_storage; then
        DATA_LOOP_STORAGE_SELECTED="mcap"
        DATA_LOOP_STORAGE_REASON="rosbag2_storage_mcap package found"
      else
        DATA_LOOP_STORAGE_SELECTED="sqlite3"
        DATA_LOOP_STORAGE_REASON="rosbag2_storage_mcap package not found; graceful fallback"
      fi
      ;;
    mcap)
      if has_mcap_storage; then
        DATA_LOOP_STORAGE_SELECTED="mcap"
        DATA_LOOP_STORAGE_REASON="mcap requested and package found"
      else
        die "mcap requested but rosbag2_storage_mcap package was not found; use --storage auto or sqlite3 to allow fallback"
      fi
      ;;
    sqlite3)
      DATA_LOOP_STORAGE_SELECTED="sqlite3"
      DATA_LOOP_STORAGE_REASON="sqlite3 requested"
      ;;
    *)
      die "bad --storage '$requested'; expected auto, mcap, or sqlite3"
      ;;
  esac
}

collect_topics() {
  local run_dir="$1"
  local regex="$2"
  shift 2
  local exact_topics=("$@")
  local topics_file="$run_dir/topics_recorded.txt"
  local tmp_file="$run_dir/topics_recorded.unsorted.txt"

  : > "$tmp_file"
  if ros2 topic list > "$run_dir/topics_at_start.txt" 2>"$run_dir/logs/topics_at_start.err"; then
    grep -E "$regex" "$run_dir/topics_at_start.txt" >> "$tmp_file" || true
  else
    warn "ros2 topic list failed; see logs/topics_at_start.err"
    : > "$run_dir/topics_at_start.txt"
  fi

  ros2 topic list -t > "$run_dir/topics_with_types_at_start.txt" 2>"$run_dir/logs/topics_with_types_at_start.err" || true
  ros2 node list > "$run_dir/nodes_at_start.txt" 2>"$run_dir/logs/nodes_at_start.err" || true

  local topic
  for topic in "${exact_topics[@]}"; do
    [ -n "$topic" ] && printf '%s\n' "$topic" >> "$tmp_file"
  done

  sort -u "$tmp_file" > "$topics_file"
  rm -f "$tmp_file"
}

check_required_topics() {
  local run_dir="$1"
  local topics_file="$2"
  shift 2
  local required_topics=("$@")
  local missing_file="$run_dir/missing_required_topics.txt"
  : > "$missing_file"

  local topic
  for topic in "${required_topics[@]}"; do
    [ -n "$topic" ] || continue
    if ! grep -qxF "$topic" "$topics_file" 2>/dev/null; then
      printf '%s\n' "$topic" >> "$missing_file"
    fi
  done

  if [ -s "$missing_file" ]; then
    echo "ERR missing required topics before data-loop recording:" >&2
    sed 's/^/  - /' "$missing_file" >&2
    echo "See: $missing_file" >&2
    return 1
  fi

  rm -f "$missing_file"
  return 0
}

write_state() {
  local state_file="$1"
  shift
  : > "$state_file"
  while [ "$#" -gt 0 ]; do
    local key="$1"
    local value="$2"
    shift 2
    printf '%s=%q\n' "$key" "$value" >> "$state_file"
  done
}

write_command_file() {
  local out="$1"
  shift
  : > "$out"
  printf '%q ' "$@" >> "$out"
  printf '\n' >> "$out"
}

start_rosbag() {
  local run_dir="$1"
  local bag_dir="$2"
  local storage="$3"
  local record_all="$4"
  local topics_file="$5"

  local cmd=()
  if [ "$record_all" = "1" ]; then
    cmd=(ros2 bag record -a -s "$storage" -o "$bag_dir")
  else
    mapfile -t topics < "$topics_file"
    if [ "${#topics[@]}" -eq 0 ]; then
      die "no topics selected; use --all, --topic TOPIC, or start the ROS graph first"
    fi
    cmd=(ros2 bag record -s "$storage" -o "$bag_dir" "${topics[@]}")
  fi

  write_command_file "$run_dir/logs/rosbag_command.txt" "${cmd[@]}"
  nohup "${cmd[@]}" > "$run_dir/logs/rosbag_record.log" 2>&1 < /dev/null &
  echo "$!"
}

storage_requested="${DATA_LOOP_STORAGE:-auto}"
base="${DATA_LOOP_BASE:-$HOME/data_loop_runs}"
name=""
record_all=0
video_mode="${DATA_LOOP_VIDEO2:-auto}"
video2_session=""
topic_regex="${DATA_LOOP_TOPIC_REGEX:-^/(tf|tf_static|clock|cmd_vel|cmd_vel_safe|odom|scan|scan_depth|map|map_metadata|goal_pose|initialpose|diagnostics)$|^/f407/(estop_latched|cmd_vel_expired|firmware_identity_valid|firmware_info)$|^/lab_fsd(/|$)|^/mppi(/|$)|^/pickup/(hardware_sensor_sample|physical_evidence|physical_evidence_request|physical_evidence_bridge_status)$|^/hobot_yolo_world$|^/(system_telemetry|lift_status|furnace_reading|alarms)$}"
declare -a exact_topics=()
required_topic_gate="${DATA_LOOP_REQUIRED_TOPIC_GATE:-1}"
cmd_vel_expectation="${DATA_LOOP_CMD_VEL_EXPECTATION:-any}"
declare -a required_topics=(
  /cmd_vel
  /odom
  /scan
  /scan_depth
  /map
  /lab_fsd/fsd_v3_status
  /lab_fsd/future_risk
  /lab_fsd/input_status
  /lab_fsd/vision_bev
  /lab_fsd/vision_risk
  /lab_fsd/vision_objects
  /lab_fsd/safety_gate
  /lab_fsd/shadow_path
  /lab_fsd/trajectory_scores
  /lab_fsd/bev
  /lab_fsd/future_bev
  /lab_fsd/policy_tokens
  /diagnostics
  /lift_status
  /f407/estop_latched
  /f407/cmd_vel_expired
  /f407/firmware_identity_valid
  /f407/firmware_info
)
if [ -n "${DATA_LOOP_REQUIRED_TOPICS:-}" ]; then
  mapfile -t required_topics < <(printf '%s\n' "$DATA_LOOP_REQUIRED_TOPICS" | tr ', ' '\n' | sed '/^$/d')
fi

while [ "$#" -gt 0 ]; do
  case "$1" in
    --name)
      shift || die "--name needs a value"
      name="${1:-}"
      ;;
    --out|--base)
      shift || die "--out needs a value"
      base="${1:-}"
      ;;
    --storage)
      shift || die "--storage needs a value"
      storage_requested="${1:-}"
      ;;
    --all)
      record_all=1
      ;;
    --topic)
      shift || die "--topic needs a value"
      exact_topics+=("${1:-}")
      ;;
    --required-topic)
      shift || die "--required-topic needs a value"
      required_topics+=("${1:-}")
      ;;
    --no-required-topic-gate)
      required_topic_gate=0
      ;;
    --cmd-vel-expect)
      shift || die "--cmd-vel-expect needs a value"
      cmd_vel_expectation="${1:-}"
      ;;
    --topic-regex)
      shift || die "--topic-regex needs a value"
      topic_regex="${1:-}"
      ;;
    --video2)
      video_mode="start"
      ;;
    --no-video2)
      video_mode="skip"
      ;;
    --attach-video2)
      shift || die "--attach-video2 needs a value"
      video_mode="attach"
      video2_session="${1:-}"
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

case "$cmd_vel_expectation" in
  any|zero|nonzero) ;;
  *) die "bad --cmd-vel-expect '$cmd_vel_expectation'; expected any, zero, or nonzero" ;;
esac

tools_dir="$(script_dir)"
setup_ros

stamp="$(date +%Y%m%d_%H%M%S)"
if [ -n "$name" ]; then
  safe_name="$(sanitize_name "$name")"
  [ -n "$safe_name" ] || safe_name="run"
  run_id="dl_${stamp}_${safe_name}"
else
  run_id="dl_${stamp}"
fi

run_dir="$base/$run_id"
mkdir -p "$run_dir/logs" "$run_dir/exports/lerobot" "$run_dir/exports/robomimic"
echo "$run_dir" > "$base/current_run.txt"

select_storage "$storage_requested"
storage="$DATA_LOOP_STORAGE_SELECTED"
storage_reason="$DATA_LOOP_STORAGE_REASON"
bag_dir="$run_dir/rosbag_${storage}"

if [ "$record_all" = "1" ]; then
  ros2 topic list > "$run_dir/topics_at_start.txt" 2>"$run_dir/logs/topics_at_start.err" || true
  ros2 topic list -t > "$run_dir/topics_with_types_at_start.txt" 2>"$run_dir/logs/topics_with_types_at_start.err" || true
  ros2 node list > "$run_dir/nodes_at_start.txt" 2>"$run_dir/logs/nodes_at_start.err" || true
  : > "$run_dir/topics_recorded.txt"
  required_gate_topics_file="$run_dir/topics_at_start.txt"
else
  collect_topics "$run_dir" "$topic_regex" "${exact_topics[@]}"
  required_gate_topics_file="$run_dir/topics_recorded.txt"
fi

if [ "$required_topic_gate" = "1" ]; then
  check_required_topics "$run_dir" "$required_gate_topics_file" "${required_topics[@]}" || exit 2
fi

rosbag_pid="$(start_rosbag "$run_dir" "$bag_dir" "$storage" "$record_all" "$run_dir/topics_recorded.txt")"
sleep 3
if ! kill -0 "$rosbag_pid" 2>/dev/null; then
  if [ "$storage" = "mcap" ]; then
    if [ "$storage_requested" = "mcap" ]; then
      tail -80 "$run_dir/logs/rosbag_record.log" >&2 || true
      die "rosbag mcap recorder exited early and --storage mcap forbids fallback"
    fi
    warn "rosbag mcap recorder exited early; falling back to sqlite3"
    storage="sqlite3"
    storage_reason="mcap recorder exited early; graceful fallback to sqlite3"
    bag_dir="$run_dir/rosbag_sqlite3"
    rosbag_pid="$(start_rosbag "$run_dir" "$bag_dir" "$storage" "$record_all" "$run_dir/topics_recorded.txt")"
    sleep 3
  fi
fi

if ! kill -0 "$rosbag_pid" 2>/dev/null; then
  tail -80 "$run_dir/logs/rosbag_record.log" >&2 || true
  die "rosbag recorder did not stay alive"
fi

video2_started=0
case "$video_mode" in
  auto|start)
    video2_start="$tools_dir/video2_start_capture.sh"
    if [ -x "$video2_start" ] || [ -f "$video2_start" ]; then
      if bash "$video2_start" > "$run_dir/logs/video2_start.log" 2>&1; then
        video2_session="$(grep -E '^/' "$run_dir/logs/video2_start.log" | tail -1 || true)"
        if [ -n "$video2_session" ]; then
          video2_started=1
        else
          warn "video2 started but session path was not parsed; see logs/video2_start.log"
        fi
      else
        warn "video2 start failed; see logs/video2_start.log"
      fi
    elif [ "$video_mode" = "start" ]; then
      warn "video2_start_capture.sh not found in $tools_dir"
    fi
    ;;
  attach)
    [ -d "$video2_session" ] || warn "attached video2 session does not exist yet: $video2_session"
    ;;
  skip)
    ;;
  *)
    warn "unknown DATA_LOOP_VIDEO2='$video_mode'; video2 skipped"
    video_mode="skip"
    ;;
esac

write_state "$run_dir/state.env" \
  RUN_ID "$run_id" \
  RUN_DIR "$run_dir" \
  STARTED_AT "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  START_EPOCH "$(date +%s)" \
  ROS_DOMAIN_ID "${ROS_DOMAIN_ID:-0}" \
  STORAGE_REQUESTED "$storage_requested" \
  STORAGE "$storage" \
  STORAGE_REASON "$storage_reason" \
  RECORD_ALL "$record_all" \
	  TOPIC_REGEX "$topic_regex" \
  REQUIRED_TOPIC_GATE "$required_topic_gate" \
  REQUIRED_TOPIC_GATE_FILE "$required_gate_topics_file" \
  REQUIRED_TOPICS "$(printf '%s,' "${required_topics[@]}" | sed 's/,$//')" \
	CMD_VEL_EXPECTATION "$cmd_vel_expectation" \
	  TOPICS_FILE "$run_dir/topics_recorded.txt" \
  BAG_DIR "$bag_dir" \
  ROSBAG_PID "$rosbag_pid" \
  VIDEO2_MODE "$video_mode" \
  VIDEO2_SESSION "$video2_session" \
  VIDEO2_STARTED_BY_DATA_LOOP "$video2_started" \
  TOOLS_DIR "$tools_dir"

echo "DATA_LOOP_STARTED"
echo "run_id: $run_id"
echo "run_dir: $run_dir"
echo "storage: $storage ($storage_reason)"
echo "rosbag_pid: $rosbag_pid"
echo "cmd_vel_expectation: $cmd_vel_expectation"
if [ "$record_all" = "1" ]; then
  echo "topics: all"
else
  echo "topics_file: $run_dir/topics_recorded.txt"
fi
if [ -n "$video2_session" ]; then
  echo "video2_session: $video2_session"
fi
echo "Stop with: $tools_dir/data_loop_stop.sh"
