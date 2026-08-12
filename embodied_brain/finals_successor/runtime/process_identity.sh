#!/usr/bin/env bash

# Shared PID identity checks for the candidate-only shadow process.
# A bare PID is never sufficient because Linux can reuse it after a crash.

x5_triflow_state_value() {
  local state_file="$1"
  local key="$2"
  awk -F= -v wanted="$key" '
    $1 == wanted {
      print substr($0, length($1) + 2)
      exit
    }
  ' "$state_file"
}

x5_triflow_process_matches() {
  local state_file="$1"
  local expected_node="$2"
  [[ -f "$state_file" ]] || return 1

  local pid start_ticks current_start
  pid="$(x5_triflow_state_value "$state_file" pid)"
  start_ticks="$(x5_triflow_state_value "$state_file" start_ticks)"
  [[ "$pid" =~ ^[0-9]+$ && "$start_ticks" =~ ^[0-9]+$ ]] || return 1
  [[ -r "/proc/$pid/stat" && -r "/proc/$pid/cmdline" ]] || return 1

  current_start="$(awk '{print $22}' "/proc/$pid/stat" 2>/dev/null || true)"
  [[ "$current_start" == "$start_ticks" ]] || return 1
  tr '\0' '\n' <"/proc/$pid/cmdline" | grep -Fqx -- "$expected_node"
}

x5_triflow_write_state() {
  local state_file="$1"
  local pid="$2"
  local expected_node="$3"
  local expected_root="$4"
  local start_ticks
  start_ticks="$(awk '{print $22}' "/proc/$pid/stat")"
  local temporary="${state_file}.tmp.$$"
  umask 077
  {
    printf 'pid=%s\n' "$pid"
    printf 'start_ticks=%s\n' "$start_ticks"
    printf 'node=%s\n' "$expected_node"
    printf 'root=%s\n' "$expected_root"
  } >"$temporary"
  mv -f -- "$temporary" "$state_file"
}
