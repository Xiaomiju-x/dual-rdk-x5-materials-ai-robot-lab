#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
VENV="${XRD_CLOUD5090_VENV:-${ROOT}/.venv}"
PYTHON="${XRD_PYTHON:-python3}"
TORCH_INDEX_URL="${XRD_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
WITH_SMOLVLA=0
OFFLINE_WHEELHOUSE=""

usage() {
  cat <<'EOF'
Usage: bootstrap_ubuntu.sh [--with-smolvla] [--offline-wheelhouse DIR]

Creates an isolated venv inside this bundle. It never invokes apt, sudo,
nvidia-driver, nvcc, or modifies the system CUDA installation.
EOF
}

while (($#)); do
  case "$1" in
    --with-smolvla) WITH_SMOLVLA=1; shift ;;
    --offline-wheelhouse) OFFLINE_WHEELHOUSE="${2:?missing wheelhouse}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

command -v "${PYTHON}" >/dev/null || { echo "${PYTHON} not found" >&2; exit 2; }
"${PYTHON}" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10+ is required")
PY

if [[ ! -x "${VENV}/bin/python" ]]; then
  "${PYTHON}" -m venv "${VENV}"
fi
VPY="${VENV}/bin/python"
"${VPY}" -m pip install --upgrade "pip==25.1.1" "setuptools==80.9.0" "wheel==0.45.1"

if [[ -n "${OFFLINE_WHEELHOUSE}" ]]; then
  "${VPY}" -m pip install --no-index --find-links "${OFFLINE_WHEELHOUSE}" \
    "torch==2.7.1" "torchvision==0.22.1"
  "${VPY}" -m pip install --no-index --find-links "${OFFLINE_WHEELHOUSE}" \
    -r "${ROOT}/requirements-lock.txt"
else
  "${VPY}" -m pip install --index-url "${TORCH_INDEX_URL}" \
    "torch==2.7.1" "torchvision==0.22.1"
  "${VPY}" -m pip install -r "${ROOT}/requirements-lock.txt"
fi

if ((WITH_SMOLVLA)); then
  if [[ -n "${OFFLINE_WHEELHOUSE}" ]]; then
    "${VPY}" -m pip install --no-index --find-links "${OFFLINE_WHEELHOUSE}" \
      -r "${ROOT}/requirements-smolvla-lock.txt"
  else
    "${VPY}" -m pip install -r "${ROOT}/requirements-smolvla-lock.txt"
  fi
fi

"${VPY}" -m pip freeze > "${VENV}/installed-freeze.txt"
echo "READY: ${VENV}"
echo "This bootstrap did not modify system CUDA or NVIDIA drivers."
