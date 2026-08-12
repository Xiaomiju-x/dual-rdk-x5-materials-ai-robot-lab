#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
VPY="${XRD_CLOUD5090_VENV:-${ROOT}/.venv}/bin/python"
CONFIG="${ROOT}/configs/base.yaml"
SMOLVLA_CONFIG="${ROOT}/configs/smolvla.yaml"
TRAIN_JSONL="${XRD_TRAIN_JSONL:-}"
READINESS_REPORT="${XRD_READINESS_REPORT:-}"
OUTPUT_ROOT="${XRD_OUTPUT_ROOT:-${ROOT}/outputs}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR=""
DRY_RUN=0
ALLOW_NO_GPU=0
ENABLE_SMOLVLA=0
SYNTHETIC_SMOKE=0
SMOLVLA_MODEL_PATH=""
LEROBOT_DATASET_ROOT=""

usage() {
  cat <<'EOF'
Usage: run_all.sh --train-jsonl FILE --readiness-report FILE [options]

Options:
  --dry-run                    Audit and write plans; do not train.
  --allow-no-gpu               Only valid with --dry-run.
  --synthetic-smoke            Train/export the student on generated data only.
  --enable-smolvla             Explicitly enable gated SmolVLA training.
  --smolvla-model-path DIR     Existing local checkpoint (no auto-download).
  --lerobot-dataset-root DIR   Existing local LeRobot dataset.
  --output-root DIR            Result parent directory.
  --config FILE                Base training configuration.

Default order: data audit -> Tiny-ACT/world-model multi-seed -> SmolVLA dry-run.
XR-0, OpenVLA-OFT, and XR-U0 are always reference-only and never downloaded.
EOF
}

while (($#)); do
  case "$1" in
    --train-jsonl) TRAIN_JSONL="${2:?missing file}"; shift 2 ;;
    --readiness-report) READINESS_REPORT="${2:?missing file}"; shift 2 ;;
    --output-root) OUTPUT_ROOT="${2:?missing directory}"; shift 2 ;;
    --config) CONFIG="${2:?missing config}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --allow-no-gpu) ALLOW_NO_GPU=1; shift ;;
    --synthetic-smoke) SYNTHETIC_SMOKE=1; shift ;;
    --enable-smolvla) ENABLE_SMOLVLA=1; shift ;;
    --smolvla-model-path) SMOLVLA_MODEL_PATH="${2:?missing directory}"; shift 2 ;;
    --lerobot-dataset-root) LEROBOT_DATASET_ROOT="${2:?missing directory}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -x "${VPY}" ]] || { echo "Run ${ROOT}/bootstrap_ubuntu.sh first" >&2; exit 2; }
if ((ALLOW_NO_GPU && !DRY_RUN)); then
  echo "--allow-no-gpu is only valid with --dry-run" >&2
  exit 2
fi
if ((SYNTHETIC_SMOKE && DRY_RUN)); then
  echo "--synthetic-smoke and --dry-run are separate modes" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}"
RUN_DIR="$(cd -- "${OUTPUT_ROOT}" && pwd -P)/run_${RUN_ID}_$$"
mkdir -p "${RUN_DIR}"
printf '%q ' "$0" "$@" > "${RUN_DIR}/invocation.txt"

finish() {
  local rc=$?
  set +e
  "${VPY}" "${ROOT}/collect_receipt.py" \
    --run-dir "${RUN_DIR}" --exit-code "${rc}" --argv "$(cat "${RUN_DIR}/invocation.txt")"
  exit "${rc}"
}
trap finish EXIT

PREFLIGHT_ARGS=(
  "${ROOT}/preflight.py"
  --config "${CONFIG}"
  --train-jsonl "${TRAIN_JSONL}"
  --readiness-report "${READINESS_REPORT}"
  --out "${RUN_DIR}/preflight.json"
)
if ((ALLOW_NO_GPU)); then PREFLIGHT_ARGS+=(--allow-no-gpu); fi
if ((!DRY_RUN && !SYNTHETIC_SMOKE)); then PREFLIGHT_ARGS+=(--require-real-gate); fi
"${VPY}" "${PREFLIGHT_ARGS[@]}"

"${VPY}" "${ROOT}/reference_stage.py" \
  --config "${SMOLVLA_CONFIG}" --out "${RUN_DIR}/reference_models.json"

if ((SYNTHETIC_SMOKE)); then
  "${VPY}" "${ROOT}/synthetic_smoke.py" \
    --out-dir "${RUN_DIR}/synthetic_smoke" \
    --seed 20260730 \
    --steps 12
  "${VPY}" "${ROOT}/package_results.py" --run-dir "${RUN_DIR}" --out-dir "${OUTPUT_ROOT}/packages"
  echo "SYNTHETIC_SMOKE_DONE_NOT_REAL_POLICY: ${RUN_DIR}"
  exit 0
fi

SMOL_ARGS=(
  "${ROOT}/smolvla_stage.py"
  --config "${SMOLVLA_CONFIG}"
  --preflight "${RUN_DIR}/preflight.json"
  --out-dir "${RUN_DIR}/smolvla"
)

if ((DRY_RUN)); then
  "${VPY}" "${SMOL_ARGS[@]}"
  "${VPY}" "${ROOT}/package_results.py" --run-dir "${RUN_DIR}" --out-dir "${OUTPUT_ROOT}/packages"
  echo "DRY_RUN_DONE: ${RUN_DIR}"
  exit 0
fi

mapfile -t SEEDS < <("${VPY}" - "${CONFIG}" <<'PY'
import sys, yaml
value = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
for seed in value["training"]["seeds"]:
    print(int(seed))
PY
)

for model in tiny_act world_model; do
  for seed in "${SEEDS[@]}"; do
    "${VPY}" "${ROOT}/train_models.py" \
      --model "${model}" \
      --config "${CONFIG}" \
      --train-jsonl "${TRAIN_JSONL}" \
      --preflight "${RUN_DIR}/preflight.json" \
      --seed "${seed}" \
      --out-dir "${RUN_DIR}/${model}/seed_${seed}"
  done
done

if ((ENABLE_SMOLVLA)); then
  SMOL_ARGS+=(
    --enable-smolvla
    --model-path "${SMOLVLA_MODEL_PATH}"
    --dataset-root "${LEROBOT_DATASET_ROOT}"
  )
fi
"${VPY}" "${SMOL_ARGS[@]}"
"${VPY}" "${ROOT}/package_results.py" --run-dir "${RUN_DIR}" --out-dir "${OUTPUT_ROOT}/packages"
echo "TRAINING_DONE: ${RUN_DIR}"
