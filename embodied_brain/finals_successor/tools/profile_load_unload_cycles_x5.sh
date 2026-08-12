#!/usr/bin/env bash
# Repeatedly load and execute isolated successor BPU models on an idle RDK X5.
set -uo pipefail

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${HOME}/finals_successor_evidence/x5_cycles_${TIMESTAMP}"
CYCLES=30
ACK_IDLE=0
MODELS=()

while (($#)); do
  case "$1" in
    --model)
      MODELS+=("${2:?missing value for --model}")
      shift 2
      ;;
    --cycles)
      CYCLES="${2:?missing value for --cycles}"
      shift 2
      ;;
    --output)
      OUT_DIR="${2:?missing value for --output}"
      shift 2
      ;;
    --ack-idle)
      ACK_IDLE=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if ((ACK_IDLE != 1)); then
  echo "Repeated model execution requires --ack-idle" >&2
  exit 2
fi
if ! [[ "${CYCLES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--cycles must be a positive integer" >&2
  exit 2
fi
if ((${#MODELS[@]} == 0)); then
  echo "At least one --model is required" >&2
  exit 2
fi
if ! command -v hrt_model_exec >/dev/null 2>&1; then
  echo "hrt_model_exec is unavailable" >&2
  exit 3
fi

mkdir -p "${OUT_DIR}/logs" "${OUT_DIR}/profiles"
OUT_DIR="$(cd -- "${OUT_DIR}" && pwd -P)"
SUMMARY="${OUT_DIR}/cycles.tsv"
printf 'cycle\tmodel\tmodel_sha256\trc\tcma_before_kib\tcma_after_kib\tmem_available_before_kib\tmem_available_after_kib\n' >"${SUMMARY}"

declare -a RESOLVED_MODELS=()
for supplied in "${MODELS[@]}"; do
  model="$(realpath "${supplied}")"
  if [[ ! -f "${model}" ]]; then
    echo "Model not found: ${supplied}" >&2
    exit 3
  fi
  case "${model}" in
    "${HOME}/xrd_candidates/"*) ;;
    *)
      echo "Refusing model outside ${HOME}/xrd_candidates: ${model}" >&2
      exit 3
      ;;
  esac
  RESOLVED_MODELS+=("${model}")
done

read_kib() {
  local key="$1"
  awk -v key="${key}:" '$1 == key {print $2; exit}' /proc/meminfo
}

failures=0
for ((cycle = 1; cycle <= CYCLES; cycle++)); do
  model="${RESOLVED_MODELS[$(((cycle - 1) % ${#RESOLVED_MODELS[@]}))]}"
  tag="$(basename -- "${model}" .bin)"
  model_sha="$(sha256sum "${model}")"
  model_sha="${model_sha%% *}"
  cma_before="$(read_kib CmaFree)"
  mem_before="$(read_kib MemAvailable)"
  profile_dir="${OUT_DIR}/profiles/$(printf '%03d' "${cycle}")_${tag}"
  mkdir -p "${profile_dir}"
  log="${OUT_DIR}/logs/$(printf '%03d' "${cycle}")_${tag}.log"

  hrt_model_exec perf \
    --model_file "${model}" \
    --thread_num 1 \
    --frame_count 1 \
    --profile_path "${profile_dir}" >"${log}" 2>&1
  rc=$?
  cma_after="$(read_kib CmaFree)"
  mem_after="$(read_kib MemAvailable)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${cycle}" "${tag}" "${model_sha}" "${rc}" \
    "${cma_before}" "${cma_after}" "${mem_before}" "${mem_after}" \
    >>"${SUMMARY}"
  if ((rc != 0)); then
    failures=$((failures + 1))
  fi
done

{
  echo "captured_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "cycles=${CYCLES}"
  echo "failures=${failures}"
  echo "frozen_services_changed=false"
  echo "motion_authority=false"
  echo "measurement_boundary=one random-input BPU frame per fresh hrt_model_exec process"
} >"${OUT_DIR}/RESULT.txt"

(
  cd -- "${OUT_DIR}"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS
)

echo "Cycle evidence: ${OUT_DIR}"
exit "${failures}"
