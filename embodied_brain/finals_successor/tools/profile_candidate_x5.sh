#!/usr/bin/env bash
# Read-only X5 evidence collection for the independent successor candidate.
# It never starts, stops, restarts, enables, disables, or kills frozen services.
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SUCCESSOR_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
ARTIFACT_ROOT="${SUCCESSOR_ROOT}/bpu/artifacts"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${HOME}/finals_successor_evidence/x5_profile_${TIMESTAMP}"
FRAME_COUNT=200
ACK_IDLE=0
SNAPSHOT_ONLY=0
MODELS=()

usage() {
  cat <<'EOF'
Usage:
  profile_candidate_x5.sh --snapshot-only [--output DIR]

  profile_candidate_x5.sh --model PATH [--model PATH ...] --ack-idle \
    [--frame-count N] [--output DIR]

The snapshot path is read-only except for writing its evidence directory.
Model perf loads and executes only an explicitly supplied successor .bin.
It does not change service state. Run model perf only while the vehicle and
lift are idle and no demonstration is in progress.
EOF
}

while (($#)); do
  case "$1" in
    --snapshot-only)
      SNAPSHOT_ONLY=1
      shift
      ;;
    --model)
      MODELS+=("${2:?missing value for --model}")
      shift 2
      ;;
    --ack-idle)
      ACK_IDLE=1
      shift
      ;;
    --frame-count)
      FRAME_COUNT="${2:?missing value for --frame-count}"
      shift 2
      ;;
    --output)
      OUT_DIR="${2:?missing value for --output}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! [[ "${FRAME_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--frame-count must be a positive integer" >&2
  exit 2
fi
if ((SNAPSHOT_ONLY == 1 && ${#MODELS[@]} > 0)); then
  echo "--snapshot-only cannot be combined with --model" >&2
  exit 2
fi
if ((${#MODELS[@]} > 0 && ACK_IDLE != 1)); then
  echo "Model perf requires --ack-idle" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"
OUT_DIR="$(cd -- "${OUT_DIR}" && pwd -P)"
STATUS_FILE="${OUT_DIR}/collection_status.tsv"
printf 'section\tstatus\trc\n' >"${STATUS_FILE}"

capture_shell() {
  local name="$1"
  local command="$2"
  local output="${OUT_DIR}/${name}.txt"
  {
    printf '# captured_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '# command=%s\n\n' "${command}"
    bash -o pipefail -c "${command}"
  } >"${output}" 2>&1
  local rc=$?
  local state="OK"
  if ((rc != 0)); then
    state="UNAVAILABLE"
  fi
  printf '%s\t%s\t%s\n' "${name}" "${state}" "${rc}" >>"${STATUS_FILE}"
  return 0
}

capture_memory_snapshot() {
  local name="$1"
  capture_shell "${name}" \
    "grep -E '^(MemTotal|MemAvailable|SwapTotal|SwapFree|CmaTotal|CmaFree):' /proc/meminfo; printf '\\n'; free -h"
}

cat >"${OUT_DIR}/MEASUREMENT_BOUNDARY.txt" <<'EOF'
All values in this directory are observations from this invocation.
They are not design budgets and are not proof of sustained 10 TOPS use.
CmaFree/ION observations are not dedicated BPU VRAM measurements.
Pure hrt_model_exec task time is not end-to-end application latency.
No frozen service was started, stopped, restarted, enabled, or disabled.
EOF

capture_shell "identity_and_rdkos" \
  'date -u +%Y-%m-%dT%H:%M:%SZ; hostname; uname -a; printf "\n### rdkos_info\n"; if command -v rdkos_info >/dev/null 2>&1; then rdkos_info; else echo "rdkos_info unavailable"; fi'

capture_shell "runtime_inventory" \
  'printf "### hrt_model_exec\n"; command -v hrt_model_exec || true; hrt_model_exec --version 2>&1 || true; printf "\n### packages\n"; if command -v dpkg-query >/dev/null 2>&1; then dpkg-query -W -f="\${Package}\t\${Version}\n" 2>/dev/null | grep -Ei "(hbm|hobot|dnn|tros|horizon|bpu)" || true; fi; printf "\n### python distributions\n"; python3 - <<'"'"'PY'"'"' 2>/dev/null || true
import importlib.metadata
for name in ("hbm-runtime", "hobot-dnn-rdkx5", "hobot_dnn"):
    try:
        print(f"{name}\t{importlib.metadata.version(name)}")
    except importlib.metadata.PackageNotFoundError:
        pass
PY'

capture_shell "frozen_service_state_read_only" \
  'for unit in embodied_brain.service xrd-embodied.service; do printf "%s\t" "$unit"; systemctl is-active "$unit" 2>&1 || true; done; printf "\n### matching processes\n"; ps -eo pid,ppid,user,%cpu,%mem,comm,args --sort=-%cpu | grep -E "(finals_lift_nav|serial_f407|slam_toolbox|nav2|lab_fsd|triflow|shadow)" | grep -v grep || true'

capture_memory_snapshot "memory_before"

capture_shell "ion_and_contiguous_memory" \
  'printf "### meminfo\n"; grep -E "^(CmaTotal|CmaFree):" /proc/meminfo; printf "\n### iomem filtered\n"; grep -Ei "(ion|cma|carveout|bpu|reserved)" /proc/iomem 2>/dev/null || true; printf "\n### debugfs/proc readable nodes\n"; for path in /sys/kernel/debug/ion/heaps/* /sys/kernel/debug/dma_buf/bufinfo /proc/ion/*; do if [ -f "$path" ] && [ -r "$path" ]; then echo "----- $path"; cat "$path"; fi; done; printf "\n### boot messages filtered\n"; dmesg 2>/dev/null | grep -Ei "(ion|cma|carveout|bpu)" | tail -200 || true'

capture_shell "process_pss" \
  'printf "pss_kib\tpid\tuser\tcomm\tcmdline\n"; for proc in /proc/[0-9]*; do pid=${proc##*/}; rollup="$proc/smaps_rollup"; [ -r "$rollup" ] || continue; pss=$(awk "/^Pss:/{print \$2; exit}" "$rollup" 2>/dev/null); [ -n "$pss" ] || continue; user=$(stat -c "%U" "$proc" 2>/dev/null || echo "?"); comm=$(cat "$proc/comm" 2>/dev/null || echo "?"); cmd=$(tr "\0" " " <"$proc/cmdline" 2>/dev/null | cut -c1-240); printf "%s\t%s\t%s\t%s\t%s\n" "$pss" "$pid" "$user" "$comm" "$cmd"; done | sort -nr | head -100'

capture_shell "thermal_zones" \
  'for zone in /sys/class/thermal/thermal_zone*; do [ -d "$zone" ] || continue; type=$(cat "$zone/type" 2>/dev/null || echo unknown); temp=$(cat "$zone/temp" 2>/dev/null || echo unavailable); printf "%s\t%s\t%s\n" "${zone##*/}" "$type" "$temp"; done'

capture_shell "bpu_status" \
  'if command -v hrut_somstatus >/dev/null 2>&1; then timeout 10s hrut_somstatus; else echo "hrut_somstatus unavailable"; fi'

if ((${#MODELS[@]} > 0)); then
  if ! command -v hrt_model_exec >/dev/null 2>&1; then
    echo "hrt_model_exec is unavailable; refusing requested model perf" >&2
    exit 3
  fi
  ARTIFACT_ROOT_REAL="$(realpath "${ARTIFACT_ROOT}")"
  index=0
  for supplied_model in "${MODELS[@]}"; do
    index=$((index + 1))
    if [[ ! -f "${supplied_model}" ]]; then
      echo "Model not found: ${supplied_model}" >&2
      exit 3
    fi
    model="$(realpath "${supplied_model}")"
    case "${model}" in
      "${ARTIFACT_ROOT_REAL}"/*) ;;
      *)
        echo "Refusing to profile a model outside successor artifacts: ${model}" >&2
        exit 3
        ;;
    esac
    model_tag="$(basename -- "${model}" .bin)"
    model_dir="${OUT_DIR}/model_${index}_${model_tag}"
    mkdir -p "${model_dir}/hrt_profile"
    sha256sum "${model}" >"${model_dir}/model.sha256"
    capture_shell "model_${index}_memory_before" \
      "grep -E '^(MemAvailable|CmaTotal|CmaFree):' /proc/meminfo"
    capture_shell "model_${index}_info" \
      "hrt_model_exec model_info --model_file $(printf '%q' "${model}")"
    {
      printf '# captured_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      printf '# thread_num=1 frame_count=%s\n\n' "${FRAME_COUNT}"
      hrt_model_exec perf \
        --model_file "${model}" \
        --thread_num 1 \
        --frame_count "${FRAME_COUNT}" \
        --profile_path "${model_dir}/hrt_profile"
    } >"${model_dir}/perf_thread1.txt" 2>&1
    perf_rc=$?
    printf 'model_%s_perf\t%s\t%s\n' \
      "${index}" "$([[ ${perf_rc} -eq 0 ]] && echo OK || echo FAIL)" "${perf_rc}" \
      >>"${STATUS_FILE}"
    capture_shell "model_${index}_memory_after" \
      "grep -E '^(MemAvailable|CmaTotal|CmaFree):' /proc/meminfo"
    capture_shell "model_${index}_thermal_after" \
      'for zone in /sys/class/thermal/thermal_zone*; do [ -d "$zone" ] || continue; printf "%s\t" "$(cat "$zone/type" 2>/dev/null || echo unknown)"; cat "$zone/temp" 2>/dev/null || true; done'
    capture_shell "model_${index}_bpu_after" \
      'if command -v hrut_somstatus >/dev/null 2>&1; then timeout 10s hrut_somstatus; else echo "hrut_somstatus unavailable"; fi'
    if ((perf_rc != 0)); then
      echo "Model perf failed for ${model}; evidence preserved in ${model_dir}" >&2
      exit "${perf_rc}"
    fi
  done
fi

capture_memory_snapshot "memory_after"

(
  cd -- "${OUT_DIR}"
  find . -type f ! -name SHA256SUMS -print0 \
    | sort -z \
    | xargs -0 sha256sum >SHA256SUMS
)

echo "Profile evidence: ${OUT_DIR}"
echo "No frozen service state was changed."
