#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
VPY="${XRD_CLOUD5090_VENV:-${ROOT}/.venv}/bin/python"
CONFIG="${ROOT}/configs/fixture_replay.yaml"
FIXTURE_JSONL=""
FIXTURE_MANIFEST=""
OUTPUT_ROOT="${XRD_OUTPUT_ROOT:-${ROOT}/outputs}"
ORIGINAL_ARGV="$(printf '%q ' "$0" "$@")"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"

usage() {
  cat <<'EOF'
Usage: run_fixture_replay.sh --fixture-jsonl FILE --fixture-manifest FILE [options]

Trains Tiny-ACT and a temporal world model on command-derived digital-twin
replays of the two frozen demonstrations. The result is fixture-only, has no
motion authority, and is not a real-robot policy.
EOF
}

while (($#)); do
  case "$1" in
    --fixture-jsonl) FIXTURE_JSONL="${2:?missing file}"; shift 2 ;;
    --fixture-manifest) FIXTURE_MANIFEST="${2:?missing file}"; shift 2 ;;
    --output-root) OUTPUT_ROOT="${2:?missing directory}"; shift 2 ;;
    --config) CONFIG="${2:?missing config}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -x "${VPY}" ]] || { echo "Missing isolated Python: ${VPY}" >&2; exit 2; }
[[ -f "${FIXTURE_JSONL}" ]] || { echo "Missing fixture JSONL" >&2; exit 2; }
[[ -f "${FIXTURE_MANIFEST}" ]] || { echo "Missing fixture manifest" >&2; exit 2; }

mkdir -p "${OUTPUT_ROOT}"
RUN_DIR="$(cd -- "${OUTPUT_ROOT}" && pwd -P)/fixture_${RUN_ID}_$$"
mkdir -p "${RUN_DIR}"
printf '%s\n' "${ORIGINAL_ARGV}" > "${RUN_DIR}/invocation.txt"

finish() {
  local rc=$?
  set +e
  "${VPY}" "${ROOT}/collect_receipt.py" \
    --run-dir "${RUN_DIR}" --exit-code "${rc}" --argv "${ORIGINAL_ARGV}"
  exit "${rc}"
}
trap finish EXIT

"${VPY}" "${ROOT}/fixture_preflight.py" \
  --config "${CONFIG}" \
  --train-jsonl "${FIXTURE_JSONL}" \
  --manifest "${FIXTURE_MANIFEST}" \
  --out "${RUN_DIR}/fixture_preflight.json"

"${VPY}" "${ROOT}/reference_stage.py" \
  --config "${ROOT}/configs/smolvla.yaml" \
  --out "${RUN_DIR}/reference_models.json"

mapfile -t SEEDS < <("${VPY}" - "${CONFIG}" <<'PY'
import sys, yaml
value = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
for seed in value["training"]["seeds"]:
    print(int(seed))
PY
)

for model in tiny_act world_model; do
  for seed in "${SEEDS[@]}"; do
    "${VPY}" "${ROOT}/fixture_train_models.py" \
      --model "${model}" \
      --config "${CONFIG}" \
      --train-jsonl "${FIXTURE_JSONL}" \
      --preflight "${RUN_DIR}/fixture_preflight.json" \
      --seed "${seed}" \
      --out-dir "${RUN_DIR}/${model}/seed_${seed}"
  done
done

"${VPY}" "${ROOT}/aggregate_fixture_results.py" \
  --run-dir "${RUN_DIR}" \
  --config "${CONFIG}" \
  --out "${RUN_DIR}/fixture_aggregate.json"

"${VPY}" "${ROOT}/package_results.py" \
  --run-dir "${RUN_DIR}" \
  --out-dir "${OUTPUT_ROOT}/packages"

echo "FIXTURE_REPLAY_DONE_NOT_REAL_POLICY: ${RUN_DIR}"
