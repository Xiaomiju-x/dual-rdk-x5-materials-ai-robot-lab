#!/usr/bin/env bash
# Restore a validated release snapshot and verify its exact release identity.
set -euo pipefail

CD="${XRD_CMD_ROOT:-/home/rdk/cmdcenter}"
RELEASES="$CD/_releases"
PY="$CD/.venv/bin/python"
LEDGER_PY=/usr/bin/python3
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
STATE_BRIDGE="$SCRIPT_DIR/site32_state_bridge.py"
STAMP="$(date +%Y%m%d-%H%M%S)"
REVIEW_CURL=(curl -fsS --max-time 8 -H 'X-User: rollback-audit' -H 'X-Role: judge')
GATE_MAX_AGE_S=$((26 * 3600))

# R0_GATE_EVIDENCE_VALIDATOR_BEGIN
GATE_EVIDENCE_VALIDATOR="$(cat <<'PY'
import datetime
import json
import math
import sys
import time


def require(condition, message):
    if not condition:
        raise SystemExit(message)


data = json.load(sys.stdin)
expected_release, expected_digest = sys.argv[1:3]
max_age_s = int(sys.argv[3])
require(data.get("valid") is True, "rollback gate evidence is not valid")
require(data.get("gate") == "pass", "rollback gate evidence did not pass")
require(data.get("phase") == "deployed", "rollback gate evidence is not deployed")
require(data.get("release") == expected_release, "rollback gate evidence release mismatch")
manifest = data.get("asset_manifest") or {}
require(manifest.get("valid") is True, "rollback gate evidence manifest is not valid")
require(
    manifest.get("manifest_digest") == expected_digest,
    "rollback gate evidence manifest digest mismatch",
)
generated_at = data.get("generated_at")
if isinstance(generated_at, (int, float)) and not isinstance(generated_at, bool):
    generated_s = float(generated_at)
elif isinstance(generated_at, str) and generated_at.strip():
    normalized = generated_at.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    parsed = datetime.datetime.fromisoformat(normalized)
    require(parsed.tzinfo is not None, "rollback gate evidence timestamp has no timezone")
    generated_s = parsed.timestamp()
else:
    raise SystemExit("rollback gate evidence timestamp is missing")
require(math.isfinite(generated_s), "rollback gate evidence timestamp is invalid")
age_s = max(0.0, time.time() - generated_s)
require(age_s <= max_age_s, f"rollback gate evidence expired: age_s={age_s:.1f}")
PY
)"
# R0_GATE_EVIDENCE_VALIDATOR_END

# R0_SCORECARD_VALIDATOR_BEGIN
SCORECARD_VALIDATOR="$(cat <<'PY'
import json
import sys


def require(condition, message):
    if not condition:
        raise SystemExit(message)


data = json.load(sys.stdin)
expected_release = sys.argv[1]
require(data.get("release") == expected_release, "rollback scorecard release mismatch")
require(data.get("gate") == "pass", "rollback scorecard gate did not pass")
gate = data.get("gate_evidence") or {}
require(gate.get("valid") is True, "rollback scorecard gate evidence is not valid")
require(gate.get("phase") == "deployed", "rollback scorecard gate evidence is not deployed")
require(gate.get("gate") == "pass", "rollback scorecard evidence gate did not pass")
PY
)"
# R0_SCORECARD_VALIDATOR_END

is_r4_release() {
  case "$1" in
    site31-global-commercial-r4-*|site31-global-commercial-r4.*-*|site32-global-commercial-v*-*) return 0 ;;
    *) return 1 ;;
  esac
}

is_site32_release() {
  case "$1" in
    site32-global-commercial-v*-*) return 0 ;;
    *) return 1 ;;
  esac
}

manifest_requires_asset() {
  "$PY" - "$1" "$2" <<'PY'
import json
import pathlib
import sys

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
raise SystemExit(0 if sys.argv[2] in manifest.get("required_critical_assets", []) else 1)
PY
}

install_target_unit() {
  local root="$1"
  local unit=""
  if is_site32_release "$TARGET_RELEASE"; then
    unit="$root/systemd/xrd-cmdcenter.service"
  elif [ -s "$root/xrd-cmdcenter.service.active" ]; then
    unit="$root/xrd-cmdcenter.service.active"
  elif [ -f "$root/systemd/xrd-cmdcenter.service" ]; then
    unit="$root/systemd/xrd-cmdcenter.service"
  fi
  [ -n "$unit" ] && [ -f "$unit" ] || {
    echo "rollback snapshot has no compatible service unit" >&2
    return 1
  }
  sudo -n install -m 0644 "$unit" /etc/systemd/system/xrd-cmdcenter.service
  sudo -n systemctl daemon-reload
}

read_release() {
  "$PY" - "$1" <<'PY'
import ast
import pathlib
import sys

app_path = pathlib.Path(sys.argv[1])
for path in (app_path, app_path.parent / "cmdcenter" / "config.py"):
    if not path.is_file():
        continue
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        if any(isinstance(item, ast.Name) and item.id == "ASSET_VER" for item in targets):
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                print(value.value)
                raise SystemExit(0)
raise SystemExit(f"{app_path} has no literal ASSET_VER release source")
PY
}

prepare_target_state_db() {
  local current_release="$1"
  local target_release="$2"
  local source_db target_db owner group
  if is_site32_release "$current_release"; then
    source_db=/var/lib/xrd-cmdcenter/data.db
  else
    source_db="$CD/data.db"
  fi
  if is_site32_release "$target_release"; then
    target_db=/var/lib/xrd-cmdcenter/data.db
    owner=xrd-cmdcenter
    group=xrd-cmdcenter
  else
    target_db="$CD/data.db"
    owner="$(stat -c %u -- "$CD")"
    group="$(stat -c %g -- "$CD")"
  fi
  sudo -n "$LEDGER_PY" "$STATE_BRIDGE" "$source_db" "$target_db" \
    --owner "$owner" --group "$group" --mode 0600
}

require_r4_snapshot() {
  local root="$1"
  local asset
  test -f "$root/asset-manifest.json"
  test -f "$root/tools/site31_asset_manifest.py"
  for asset in r4.css r4.js r4-performance.js r4-accessibility.js; do
    test -f "$root/static/$asset"
  done
  case "$TARGET_RELEASE" in
    site32-global-commercial-v*-*)
      test -d "$root/cmdcenter"
      test -f "$root/cmdcenter/config.py"
      test -f "$root/cmdcenter/public_dto.py"
      test -f "$root/cmdcenter/route_contract.py"
      test -f "$root/cmdcenter/runtime.py"
      test -f "$root/cmdcenter/storage.py"
      test -f "$root/public_evidence/rb_voe_r1_public.json"
      test -f "$root/requirements-production.txt"
      test -f "$root/systemd/xrd-cmdcenter.service"
      test -f "$root/static/site32.css"
      test -f "$root/static/site32.js"
      test -f "$root/static/src/site32/release.js"
      test -f "$root/static/src/site32/runtime.js"
      test -d "$root/static/src/site32"
      find "$root/static/src/site32" -type f | grep -q .
      test -f "$root/tools/site32_style_audit.py"
      if manifest_requires_asset "$root/asset-manifest.json" tools/site32_state_bridge.py; then
        test -f "$root/tools/site32_state_bridge.py"
      fi
      ;;
    site31-global-commercial-r4-*|site31-global-commercial-r4.*-*)
      test -s "$root/xrd-cmdcenter.service.active" || test -f "$root/systemd/xrd-cmdcenter.service"
      ;;
  esac
}

sync_exact_tree() {
  local source_dir="$1"
  local target_dir="$2"
  rsync -a --chmod=D755,F644 --delete --delete-excluded --delay-updates \
    --exclude='__pycache__/' --exclude='.pytest_cache/' \
    --exclude='*.pyc' --exclude='*.pyo' --exclude='*.swp' --exclude='*.tmp' \
    "$source_dir/" "$target_dir/"
}

command -v realpath >/dev/null
command -v rsync >/dev/null
test -x "$PY"
test -f "$STATE_BRIDGE"
RELEASES_REAL="$(realpath -e -- "$RELEASES")"
CURRENT_RELEASE="$(read_release "$CD/app.py")"

if [ -n "${1:-}" ]; then
  case "$1" in
    *[!A-Za-z0-9._-]*|'.'|'..') echo "invalid rollback snapshot name" >&2; exit 2 ;;
  esac
  CANDIDATE="$RELEASES/$1"
else
  IFS= read -r CANDIDATE < "$RELEASES/.prev" || true
fi

if [ -z "${CANDIDATE:-}" ]; then
  echo "no rollback snapshot selected" >&2
  exit 1
fi
PREV="$(realpath -e -- "$CANDIDATE" 2>/dev/null || true)"
case "$PREV/" in
  "$RELEASES_REAL"/*) ;;
  *) echo "rollback snapshot must stay under $RELEASES_REAL" >&2; exit 2 ;;
esac
if [ ! -d "$PREV" ] || [ ! -f "$PREV/app.py" ] || [ ! -d "$PREV/static" ]; then
  echo "rollback snapshot is incomplete: $PREV" >&2
  exit 1
fi

TARGET_RELEASE="$(read_release "$PREV/app.py")"
if is_r4_release "$TARGET_RELEASE"; then
  require_r4_snapshot "$PREV"
fi
case "$TARGET_RELEASE" in
  site32-global-commercial-v*-*)
    XRD_CMD_TEST_MODE=1 "$PY" "$PREV/tools/site32_style_audit.py" --root "$PREV" >/dev/null
    ;;
esac

if [ ! -f "$PREV/asset-manifest.json" ] || [ ! -f "$PREV/tools/site31_asset_manifest.py" ]; then
  echo "rollback snapshot lacks a verifiable asset manifest" >&2
  exit 1
fi
TARGET_MANIFEST_JSON="$(XRD_CMD_TEST_MODE=1 "$PY" "$PREV/tools/site31_asset_manifest.py" "$PREV" --verify)"
TARGET_MANIFEST_RELEASE="$(printf '%s' "$TARGET_MANIFEST_JSON" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["release"])')"
TARGET_DIGEST="$(printf '%s' "$TARGET_MANIFEST_JSON" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["manifest_digest"])')"
if [ "$TARGET_MANIFEST_RELEASE" != "$TARGET_RELEASE" ]; then
  echo "rollback snapshot manifest/app release mismatch" >&2
  exit 1
fi

# Keep a complete rollback-forward snapshot before touching the live tree.
BACKOUT="$RELEASES/rollback-pre-$STAMP"
if [ -e "$BACKOUT" ]; then
  echo "rollback-forward snapshot already exists: $BACKOUT" >&2
  exit 1
fi
mkdir "$BACKOUT"
cp -a "$CD/app.py" "$BACKOUT/app.py"
if [ -f "$CD/requirements-production.txt" ]; then
  cp -a "$CD/requirements-production.txt" "$BACKOUT/requirements-production.txt"
else
  : > "$BACKOUT/.requirements-production.absent"
fi
if [ -d "$CD/cmdcenter" ]; then
  cp -a "$CD/cmdcenter" "$BACKOUT/cmdcenter"
else
  : > "$BACKOUT/.cmdcenter.absent"
fi
if [ -d "$CD/systemd" ]; then
  cp -a "$CD/systemd" "$BACKOUT/systemd"
else
  : > "$BACKOUT/.systemd.absent"
fi
if [ -d "$CD/public_evidence" ]; then
  cp -a "$CD/public_evidence" "$BACKOUT/public_evidence"
else
  : > "$BACKOUT/.public-evidence.absent"
fi
if sudo -n test -f /etc/systemd/system/xrd-cmdcenter.service; then
  sudo -n cat /etc/systemd/system/xrd-cmdcenter.service > "$BACKOUT/xrd-cmdcenter.service.active"
else
  : > "$BACKOUT/.active-unit.absent"
fi
if [ -f "$CD/assets.json" ]; then
  cp -a "$CD/assets.json" "$BACKOUT/assets.json"
else
  : > "$BACKOUT/.assets.absent"
fi
if [ -f "$CD/asset-manifest.json" ]; then
  cp -a "$CD/asset-manifest.json" "$BACKOUT/asset-manifest.json"
else
  : > "$BACKOUT/.asset-manifest.absent"
fi
cp -a "$CD/static" "$BACKOUT/static"
cp -a "$CD/tools" "$BACKOUT/tools"

restore_backout() {
  status=$?
  trap - ERR
  set +e
  cp -a "$BACKOUT/app.py" "$CD/app.py"
  if [ -f "$BACKOUT/requirements-production.txt" ]; then
    cp -a "$BACKOUT/requirements-production.txt" "$CD/requirements-production.txt"
  else
    rm -f -- "$CD/requirements-production.txt"
  fi
  if [ -d "$BACKOUT/cmdcenter" ]; then
    sync_exact_tree "$BACKOUT/cmdcenter" "$CD/cmdcenter"
  else
    rm -rf -- "$CD/cmdcenter"
  fi
  if [ -d "$BACKOUT/systemd" ]; then
    sync_exact_tree "$BACKOUT/systemd" "$CD/systemd"
  else
    rm -rf -- "$CD/systemd"
  fi
  if [ -d "$BACKOUT/public_evidence" ]; then
    sync_exact_tree "$BACKOUT/public_evidence" "$CD/public_evidence"
  else
    rm -rf -- "$CD/public_evidence"
  fi
  if [ -f "$BACKOUT/xrd-cmdcenter.service.active" ]; then
    sudo -n install -m 0644 "$BACKOUT/xrd-cmdcenter.service.active" /etc/systemd/system/xrd-cmdcenter.service
    sudo -n systemctl daemon-reload
  fi
  if [ -f "$BACKOUT/assets.json" ]; then
    cp -a "$BACKOUT/assets.json" "$CD/assets.json"
  else
    rm -f -- "$CD/assets.json"
  fi
  if [ -f "$BACKOUT/asset-manifest.json" ]; then
    cp -a "$BACKOUT/asset-manifest.json" "$CD/asset-manifest.json"
  else
    rm -f -- "$CD/asset-manifest.json"
  fi
  sync_exact_tree "$BACKOUT/static" "$CD/static"
  sync_exact_tree "$BACKOUT/tools" "$CD/tools"
  sudo -n systemctl restart xrd-cmdcenter
  echo "rollback failed; restored $BACKOUT" >&2
  exit "$status"
}
trap restore_backout ERR

echo "rolling back to $PREV"
sudo -n systemctl stop xrd-cmdcenter
prepare_target_state_db "$CURRENT_RELEASE" "$TARGET_RELEASE"
cp -a "$PREV/app.py" "$CD/app.py"
if [ -f "$PREV/requirements-production.txt" ]; then
  cp -a "$PREV/requirements-production.txt" "$CD/requirements-production.txt"
else
  rm -f -- "$CD/requirements-production.txt"
fi
if [ -d "$PREV/cmdcenter" ]; then
  sync_exact_tree "$PREV/cmdcenter" "$CD/cmdcenter"
else
  rm -rf -- "$CD/cmdcenter"
fi
if [ -d "$PREV/public_evidence" ]; then
  sync_exact_tree "$PREV/public_evidence" "$CD/public_evidence"
else
  rm -rf -- "$CD/public_evidence"
fi
if [ -d "$PREV/systemd" ]; then
  sync_exact_tree "$PREV/systemd" "$CD/systemd"
fi
install_target_unit "$PREV"
if [ -f "$PREV/assets.json" ]; then
  cp -a "$PREV/assets.json" "$CD/assets.json"
else
  rm -f -- "$CD/assets.json"
fi
if [ -f "$PREV/asset-manifest.json" ]; then
  cp -a "$PREV/asset-manifest.json" "$CD/asset-manifest.json"
else
  rm -f -- "$CD/asset-manifest.json"
fi
sync_exact_tree "$PREV/static" "$CD/static"
if [ -d "$PREV/tools" ]; then
  sync_exact_tree "$PREV/tools" "$CD/tools"
fi

XRD_CMD_TEST_MODE=1 "$PY" "$CD/tools/site31_asset_manifest.py" "$CD" --verify >/dev/null

sudo -n systemctl restart xrd-cmdcenter
for _ in 1 2 3 4 5 6 7 8; do
  if curl -fsS --max-time 8 http://127.0.0.1:29100/api/public_status >/dev/null; then break; fi
  sleep 1
done
STATUS_JSON="$(curl -fsS --max-time 8 http://127.0.0.1:29100/api/public_status)"
printf '%s' "$STATUS_JSON" | "$PY" -c \
  'import json,sys; data=json.load(sys.stdin); actual=data.get("release") or (data.get("summary") or {}).get("release"); assert actual==sys.argv[1], data' "$TARGET_RELEASE"
LIVE_MANIFEST_JSON="$(XRD_CMD_TEST_MODE=1 "$PY" "$CD/tools/site31_asset_manifest.py" "$CD" --verify)"
printf '%s' "$LIVE_MANIFEST_JSON" | "$PY" -c \
  'import json,sys; data=json.load(sys.stdin); assert data.get("release")==sys.argv[1], data; assert data.get("manifest_digest")==sys.argv[2], data' \
  "$TARGET_RELEASE" "$TARGET_DIGEST"

GATE_JSON="$("${REVIEW_CURL[@]}" http://127.0.0.1:29100/api/site31_gate_evidence)"
printf '%s' "$GATE_JSON" | "$PY" -c "$GATE_EVIDENCE_VALIDATOR" \
  "$TARGET_RELEASE" "$TARGET_DIGEST" "$GATE_MAX_AGE_S"
SCORECARD_JSON="$("${REVIEW_CURL[@]}" http://127.0.0.1:29100/api/site31_scorecard)"
printf '%s' "$SCORECARD_JSON" | "$PY" -c "$SCORECARD_VALIDATOR" "$TARGET_RELEASE"

if grep -q '^Environment=XRD_CMD_DB_PATH=/var/lib/xrd-cmdcenter/data.db$' \
    "$CD/systemd/xrd-cmdcenter.service" 2>/dev/null; then
  RELEASE_DB=/var/lib/xrd-cmdcenter/data.db
  LEDGER_RUN=(sudo -n -u xrd-cmdcenter)
else
  RELEASE_DB="$CD/data.db"
  LEDGER_RUN=()
fi
"${LEDGER_RUN[@]}" "$LEDGER_PY" - "$TARGET_RELEASE" "$(basename "$PREV")" \
  "$TARGET_DIGEST" "$RELEASE_DB" <<'PY'
import sqlite3
import sys
import time

release, snapshot, digest, database = sys.argv[1:5]
con = sqlite3.connect(database, timeout=10)
con.execute("CREATE TABLE IF NOT EXISTS releases(id INTEGER PRIMARY KEY AUTOINCREMENT, ver TEXT, ts INTEGER, files TEXT, sha TEXT, notes TEXT, by TEXT)")
con.execute("INSERT INTO releases(ver,ts,files,sha,notes,by) VALUES(?,?,?,?,?,?)",
            ("rollback->" + release, int(time.time()), "", digest[:16], "snapshot=" + snapshot, "operator"))
con.commit()
con.close()
PY

printf '%s\n' "$BACKOUT" > "$RELEASES/.prev"
trap - ERR
echo "rolled back to $TARGET_RELEASE ($TARGET_DIGEST); rollback-forward snapshot: $BACKOUT"
