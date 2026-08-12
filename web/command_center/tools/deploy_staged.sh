#!/usr/bin/env bash
# Validated staged release: candidate-owned tools -> exact promote -> runtime evidence.
set -euo pipefail

CD="${XRD_CMD_ROOT:-/home/rdk/cmdcenter}"
VER="${1:?release version required}"
NOTES="${2:-staged release}"
BY="${3:-operator}"
STAGE="${4:-$CD/_staging/$VER}"
STAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP="$CD/_releases/${VER}-predeploy-$STAMP"
PY="$CD/.venv/bin/python"
REVIEW_CURL=(curl -fsS --max-time 8 -H 'X-User: deploy-audit' -H 'X-Role: judge')

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

is_site32_v15_or_newer_release() {
  case "$1" in
    site32-global-commercial-v1.5-*|site32-global-commercial-v1.[6-9]-*|site32-global-commercial-v[2-9]*-*) return 0 ;;
    *) return 1 ;;
  esac
}

verify_site32_runtime_namespace() {
  local candidate_root="${1:?candidate root required}"
  local unit_name="site32-deploy-prereq-$$"
  sudo -n systemd-run --quiet --wait --collect \
    --unit="$unit_name" \
    --property=Type=oneshot \
    --property=User=xrd-cmdcenter \
    --property=Group=xrd-cmdcenter \
    --property=SupplementaryGroups=xrd-auth-readers \
    --property=NoNewPrivileges=yes \
    --property=PrivateTmp=yes \
    --property=ProtectSystem=strict \
    --property=ProtectHome=tmpfs \
    --property="WorkingDirectory=$CD" \
    --property="BindReadOnlyPaths=$candidate_root:$CD $CD/.venv:$CD/.venv /var/log/xrd-auth/logins.jsonl:/var/lib/xrd-auth/logins.jsonl" \
    --property="BindPaths=/var/lib/xrd-cmdcenter/reports:$CD/reports" \
    --property="ReadOnlyPaths=/var/lib/xrd-auth /var/log/xrd-auth /etc/xrd-cmdcenter" \
    --property="ReadWritePaths=/var/lib/xrd-cmdcenter $CD/reports" \
    /bin/sh -eu -c \
      'test -x /home/rdk/cmdcenter/.venv/bin/python
       test -r /home/rdk/cmdcenter/app.py
       test -r /var/lib/xrd-auth/users.json
       test -r /var/lib/xrd-auth/logins.jsonl
       test -w /var/lib/xrd-cmdcenter
       test -w /home/rdk/cmdcenter/reports
       /home/rdk/cmdcenter/.venv/bin/python -c "from cmdcenter import RuntimeController, register_site32"'
}

check_js_syntax() {
  local path="$1"
  if grep -Eq '^[[:space:]]*(import|export)([[:space:]]|[({*])' "$path"; then
    node --input-type=module --check < "$path"
  else
    node --check "$path"
  fi
}

require_release_assets() {
  local root="$1"
  local release="$2"
  local asset
  for asset in index.html app.js style.css i18n.js twin.js sw.js; do
    test -f "$root/static/$asset"
  done
  if is_r4_release "$release"; then
    for asset in r4.css r4.js r4-performance.js r4-accessibility.js; do
      test -f "$root/static/$asset"
    done
    test -f "$root/systemd/xrd-cmdcenter.service"
    test -f "$root/auth/app.py"
    test -f "$root/auth/security_smoke.py"
  fi
  if is_site32_release "$release"; then
    test -f "$root/requirements-production.txt"
    for asset in site32.css site32.js; do
      test -f "$root/static/$asset"
    done
    test -f "$root/static/src/site32/release.js"
    test -f "$root/static/src/site32/runtime.js"
    test -f "$root/cmdcenter/__init__.py"
    test -f "$root/cmdcenter/access.py"
    test -f "$root/cmdcenter/config.py"
    test -f "$root/cmdcenter/public_dto.py"
    test -f "$root/cmdcenter/release.py"
    test -f "$root/cmdcenter/route_contract.py"
    test -f "$root/cmdcenter/site32_blueprint.py"
    test -f "$root/cmdcenter/runtime.py"
    test -f "$root/cmdcenter/storage.py"
    test -f "$root/public_evidence/rb_voe_r1_public.json"
    test -f "$root/tools/site32_state_bridge.py"
    test -f "$root/tools/site32_style_audit.py"
    test -d "$root/static/src/site32"
    find "$root/static/src/site32" -type f | grep -q .
    if is_site32_v15_or_newer_release "$release"; then
      test -f "$root/cmdcenter/research_search.py"
      test -f "$root/tools/test_site32_research_search.py"
      test -f "$root/tools/site32_environment_matrix.py"
      test -f "$root/static/quality/site32_production_snapshot.json"
      test -f "$root/static/quality/site32_r0_baseline.json"
    fi
  fi
}

normalize_release_payload_modes() {
  local root="$1"
  local path
  chmod 0755 "$root"
  for path in static cmdcenter public_evidence systemd tools auth; do
    test -d "$root/$path" || continue
    find "$root/$path" -type d -exec chmod 0755 {} +
    find "$root/$path" -type f -exec chmod 0644 {} +
  done
  chmod 0644 "$root/app.py" "$root/assets.json" "$root/asset-manifest.json"
  if [ -f "$root/requirements-production.txt" ]; then
    chmod 0644 "$root/requirements-production.txt"
  fi
}

normalize_candidate_modes() {
  local root="$1"
  local hit
  hit="$(find "$root" -xdev -type l -print -quit)"
  if [ -n "$hit" ]; then
    echo "candidate symlink is not allowed: $hit" >&2
    exit 1
  fi
  hit="$(find "$root" -xdev -type f \( \
    -name '.env' -o -name 'secrets.env' -o -name 'users.json' -o \
    -name 'secret.key' -o -name 'data.db' -o -name 'logins.jsonl' -o \
    -name 'alert_email.json' \) -print -quit)"
  if [ -n "$hit" ]; then
    echo "candidate contains private runtime state: $hit" >&2
    exit 1
  fi
  normalize_release_payload_modes "$root"
}

prune_snapshot_to_manifest() {
  local root="$1"
  "$PY" - "$root" <<'PY'
import json
import os
import sys
from pathlib import Path, PurePosixPath

root = Path(sys.argv[1]).resolve(strict=True)
manifest = json.loads((root / "asset-manifest.json").read_text(encoding="utf-8"))
allowed = set()
for entry in manifest.get("files", []):
    raw = entry.get("path")
    part = PurePosixPath(raw) if isinstance(raw, str) else None
    if part is None or part.is_absolute() or ".." in part.parts:
        raise SystemExit(f"unsafe rollback manifest path: {raw!r}")
    allowed.add(part.as_posix())

removed = []
for name in ("static", "tools", "cmdcenter", "public_evidence", "systemd"):
    base = root / name
    if not base.exists():
        continue
    for path in sorted(base.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink() or path.is_file():
            if rel not in allowed:
                path.unlink()
                removed.append(rel)
        elif path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass
print(json.dumps({"snapshot": str(root), "removed_unmanifested": sorted(removed)}))
PY
}

require_site32_runtime_prereqs() {
  local candidate_unit="$1"
  local candidate_root="$2"
  local state_db="/var/lib/xrd-cmdcenter/data.db"
  grep -q '^User=xrd-cmdcenter$' "$candidate_unit"
  grep -q '^Environment=XRD_CMD_DB_PATH=/var/lib/xrd-cmdcenter/data.db$' "$candidate_unit"
  getent passwd xrd-cmdcenter >/dev/null
  getent group xrd-cmdcenter >/dev/null
  getent group xrd-auth-readers >/dev/null
  id -nG xrd-cmdcenter | tr ' ' '\n' | grep -qx xrd-auth-readers
  sudo -n test -s /etc/xrd-cmdcenter/secrets.env
  sudo -n test -d /var/lib/xrd-cmdcenter
  sudo -n test -d /var/lib/xrd-cmdcenter/reports
  sudo -n test -s /var/lib/xrd-auth/users.json
  sudo -n test -e /var/log/xrd-auth/logins.jsonl
  sudo -n test -e /var/lib/xrd-auth/logins.jsonl
  sudo -n test -x "$PY"
  sudo -n -u xrd-cmdcenter test -w /var/lib/xrd-cmdcenter
  sudo -n -u xrd-cmdcenter test -w /var/lib/xrd-cmdcenter/reports
  sudo -n -u xrd-cmdcenter test -r /var/lib/xrd-auth/users.json
  sudo -n -u xrd-cmdcenter test -r /var/log/xrd-auth/logins.jsonl
  if test -s "$CD/data.db"; then
    sudo -n test -s "$state_db"
    sudo -n -u xrd-cmdcenter /usr/bin/python3 - "$state_db" <<'PY'
import sqlite3
import sys

connection = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
try:
    result = connection.execute("PRAGMA quick_check").fetchone()[0]
finally:
    connection.close()
if result != "ok":
    raise SystemExit(f"candidate state database quick_check failed: {result}")
PY
  fi
  sudo -n systemd-analyze verify "$candidate_unit" >/dev/null
  verify_site32_runtime_namespace "$candidate_root"
}

sync_exact_tree() {
  local source_dir="$1"
  local target_dir="$2"
  rsync -a --chmod=D755,F644 --delete --delete-excluded --delay-updates \
    --exclude='__pycache__/' --exclude='.pytest_cache/' \
    --exclude='*.pyc' --exclude='*.pyo' --exclude='*.swp' --exclude='*.tmp' \
    "$source_dir/" "$target_dir/"
}

case "$VER" in
  *[!A-Za-z0-9._-]*|''|'.'|'..') echo "invalid release version" >&2; exit 2 ;;
esac

command -v realpath >/dev/null
command -v rsync >/dev/null
test -x "$PY"
mkdir -p "$CD/_staging" "$CD/_releases"
STAGE_REAL="$(realpath -e -- "$STAGE")"
STAGE_ROOT="$(realpath -e -- "$CD/_staging")"
case "$STAGE_REAL/" in
  "$STAGE_ROOT"/*) ;;
  *) echo "stage must stay under $STAGE_ROOT" >&2; exit 2 ;;
esac

test -f "$STAGE_REAL/app.py"
test -f "$STAGE_REAL/assets.json"
test -f "$STAGE_REAL/asset-manifest.json"
test -f "$STAGE_REAL/static/index.html"
test -f "$STAGE_REAL/static/app.js"
test -f "$STAGE_REAL/static/style.css"
test -f "$STAGE_REAL/static/i18n.js"
test -f "$STAGE_REAL/static/sw.js"
test -f "$STAGE_REAL/tools/site31_asset_manifest.py"
test -f "$STAGE_REAL/tools/site31_gate_audit.py"
test -f "$STAGE_REAL/tools/site31_smoke.py"

MANIFEST_TOOL="$STAGE_REAL/tools/site31_asset_manifest.py"
GATE_TOOL="$STAGE_REAL/tools/site31_gate_audit.py"
SMOKE_TOOL="$STAGE_REAL/tools/site31_smoke.py"
STYLE_TOOL="$STAGE_REAL/tools/site32_style_audit.py"
ENVIRONMENT_TOOL="$STAGE_REAL/tools/site32_environment_matrix.py"
ENVIRONMENT_SNAPSHOT="$STAGE_REAL/static/quality/site32_production_snapshot.json"
ENVIRONMENT_OUTPUT="$STAGE_REAL/static/quality/site32_environment_matrix.json"
require_release_assets "$STAGE_REAL" "$VER"
normalize_candidate_modes "$STAGE_REAL"
if is_site32_release "$VER"; then
  require_site32_runtime_prereqs "$STAGE_REAL/systemd/xrd-cmdcenter.service" "$STAGE_REAL"
fi

# The candidate must already carry a release-bound browser evidence object.
XRD_CMD_TEST_MODE=1 "$PY" "$MANIFEST_TOOL" "$STAGE_REAL" \
  --verify --ignore-generated-gate --ignore-generated-style-audit >/dev/null
PY_COMPILE_TARGETS=("$STAGE_REAL/app.py" "$MANIFEST_TOOL" "$GATE_TOOL" "$SMOKE_TOOL")
if is_site32_release "$VER"; then
  PY_COMPILE_TARGETS+=(
    "$STYLE_TOOL"
    "$STAGE_REAL/tools/site32_state_bridge.py"
    "$STAGE_REAL/cmdcenter/config.py"
    "$STAGE_REAL/cmdcenter/public_dto.py"
    "$STAGE_REAL/cmdcenter/route_contract.py"
    "$STAGE_REAL/cmdcenter/storage.py"
  )
  if test -f "$STAGE_REAL/cmdcenter/research_search.py"; then
    PY_COMPILE_TARGETS+=("$STAGE_REAL/cmdcenter/research_search.py")
  fi
  if test -f "$STAGE_REAL/tools/site32_environment_matrix.py"; then
    PY_COMPILE_TARGETS+=("$STAGE_REAL/tools/site32_environment_matrix.py")
  fi
fi
XRD_CMD_TEST_MODE=1 "$PY" -m py_compile "${PY_COMPILE_TARGETS[@]}"
if command -v node >/dev/null 2>&1; then
  check_js_syntax "$STAGE_REAL/static/app.js"
  check_js_syntax "$STAGE_REAL/static/i18n.js"
  check_js_syntax "$STAGE_REAL/static/sw.js"
  if is_r4_release "$VER"; then
    check_js_syntax "$STAGE_REAL/static/r4.js"
    check_js_syntax "$STAGE_REAL/static/r4-performance.js"
    check_js_syntax "$STAGE_REAL/static/r4-accessibility.js"
  fi
  if is_site32_release "$VER"; then
    check_js_syntax "$STAGE_REAL/static/site32.js"
    while IFS= read -r -d '' module; do
      check_js_syntax "$module"
    done < <(find "$STAGE_REAL/static/src/site32" -type f -name '*.js' -print0)
  fi
fi
if is_site32_release "$VER"; then
  XRD_CMD_TEST_MODE=1 "$PY" "$STYLE_TOOL" --root "$STAGE_REAL" \
    --output "$STAGE_REAL/static/quality/site32_style_audit.json" >/dev/null
fi
# Generated quality artifacts are unbound from the content digest but remain
# bound by the manifest artifact hash. Refresh before every consumer validates.
XRD_CMD_TEST_MODE=1 "$PY" "$MANIFEST_TOOL" "$STAGE_REAL" --write >/dev/null
XRD_CMD_TEST_MODE=1 "$PY" "$GATE_TOOL" "$STAGE_REAL" \
  --phase preflight --output "$STAGE_REAL/static/quality/site31_gate_evidence.json"
XRD_CMD_TEST_MODE=1 "$PY" "$MANIFEST_TOOL" "$STAGE_REAL" --write >/dev/null
if is_site32_v15_or_newer_release "$VER"; then
  XRD_CMD_TEST_MODE=1 "$PY" "$ENVIRONMENT_TOOL" \
    --root "$STAGE_REAL" \
    --production-snapshot "$ENVIRONMENT_SNAPSHOT" \
    --output "$ENVIRONMENT_OUTPUT" >/dev/null
  XRD_CMD_TEST_MODE=1 "$PY" "$MANIFEST_TOOL" "$STAGE_REAL" --write >/dev/null
  "$PY" - "$ENVIRONMENT_OUTPUT" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except (OSError, UnicodeError, json.JSONDecodeError) as exc:
    raise SystemExit(f"Site32 environment matrix is unreadable: {exc}")
if payload.get("ready_for_promotion") is not True:
    conflicts = json.dumps(payload.get("conflicts", []), ensure_ascii=False, sort_keys=True)
    raise SystemExit(f"Site32 environment matrix is not ready for promotion: {conflicts}")
PY
fi
SMOKE_TMP="$(mktemp -d /tmp/site32-smoke.XXXXXX)"
cleanup_smoke_tmp() {
  rm -rf -- "$SMOKE_TMP"
}
trap cleanup_smoke_tmp EXIT
XRD_CMD_TEST_MODE=1 XRD_CMD_DB_PATH="$SMOKE_TMP/data.db" \
  "$PY" "$SMOKE_TOOL" "$STAGE_REAL"
cleanup_smoke_tmp
trap - EXIT
normalize_candidate_modes "$STAGE_REAL"

# Gate evidence is generated content: inventory its final preflight bytes now.
XRD_CMD_TEST_MODE=1 "$PY" "$MANIFEST_TOOL" "$STAGE_REAL" --write >/dev/null
STAGE_MANIFEST_JSON="$(XRD_CMD_TEST_MODE=1 "$PY" "$MANIFEST_TOOL" "$STAGE_REAL" --verify)"
EXPECTED_RELEASE="$(printf '%s' "$STAGE_MANIFEST_JSON" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["release"])')"
EXPECTED_DIGEST="$(printf '%s' "$STAGE_MANIFEST_JSON" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["manifest_digest"])')"
if [ "$EXPECTED_RELEASE" != "$VER" ]; then
  echo "stage release mismatch: requested=$VER manifest=$EXPECTED_RELEASE" >&2
  exit 1
fi

if [ -e "$BACKUP" ]; then
  echo "rollback snapshot already exists: $BACKUP" >&2
  exit 1
fi
mkdir "$BACKUP"
cp -a "$CD/app.py" "$BACKUP/app.py"
if [ -f "$CD/requirements-production.txt" ]; then
  cp -a "$CD/requirements-production.txt" "$BACKUP/requirements-production.txt"
else
  : > "$BACKUP/.requirements-production.absent"
fi
if [ -d "$CD/cmdcenter" ]; then
  cp -a "$CD/cmdcenter" "$BACKUP/cmdcenter"
else
  : > "$BACKUP/.cmdcenter.absent"
fi
if [ -d "$CD/systemd" ]; then
  cp -a "$CD/systemd" "$BACKUP/systemd"
else
  : > "$BACKUP/.systemd.absent"
fi
if [ -d "$CD/public_evidence" ]; then
  cp -a "$CD/public_evidence" "$BACKUP/public_evidence"
else
  : > "$BACKUP/.public-evidence.absent"
fi
if sudo -n test -f /etc/systemd/system/xrd-cmdcenter.service; then
  sudo -n cat /etc/systemd/system/xrd-cmdcenter.service > "$BACKUP/xrd-cmdcenter.service.active"
else
  : > "$BACKUP/.active-unit.absent"
fi
if [ -f "$CD/assets.json" ]; then
  cp -a "$CD/assets.json" "$BACKUP/assets.json"
else
  : > "$BACKUP/.assets.absent"
fi
if [ -f "$CD/asset-manifest.json" ]; then
  cp -a "$CD/asset-manifest.json" "$BACKUP/asset-manifest.json"
else
  : > "$BACKUP/.asset-manifest.absent"
fi
cp -a "$CD/static" "$BACKUP/static"
cp -a "$CD/tools" "$BACKUP/tools"
prune_snapshot_to_manifest "$BACKUP"
XRD_CMD_TEST_MODE=1 "$PY" "$MANIFEST_TOOL" "$BACKUP" --verify >/dev/null

restore_previous() {
  status=$?
  trap - ERR
  set +e
  cp -a "$BACKUP/app.py" "$CD/app.py"
  if [ -f "$BACKUP/requirements-production.txt" ]; then
    cp -a "$BACKUP/requirements-production.txt" "$CD/requirements-production.txt"
  else
    rm -f -- "$CD/requirements-production.txt"
  fi
  if [ -d "$BACKUP/cmdcenter" ]; then
    sync_exact_tree "$BACKUP/cmdcenter" "$CD/cmdcenter"
  else
    rm -rf -- "$CD/cmdcenter"
  fi
  if [ -d "$BACKUP/systemd" ]; then
    sync_exact_tree "$BACKUP/systemd" "$CD/systemd"
  else
    rm -rf -- "$CD/systemd"
  fi
  if [ -d "$BACKUP/public_evidence" ]; then
    sync_exact_tree "$BACKUP/public_evidence" "$CD/public_evidence"
  else
    rm -rf -- "$CD/public_evidence"
  fi
  if [ -f "$BACKUP/xrd-cmdcenter.service.active" ]; then
    sudo -n install -m 0644 "$BACKUP/xrd-cmdcenter.service.active" /etc/systemd/system/xrd-cmdcenter.service
    sudo -n systemctl daemon-reload
  fi
  if [ -f "$BACKUP/assets.json" ]; then
    cp -a "$BACKUP/assets.json" "$CD/assets.json"
  else
    rm -f -- "$CD/assets.json"
  fi
  if [ -f "$BACKUP/asset-manifest.json" ]; then
    cp -a "$BACKUP/asset-manifest.json" "$CD/asset-manifest.json"
  else
    rm -f -- "$CD/asset-manifest.json"
  fi
  sync_exact_tree "$BACKUP/static" "$CD/static"
  sync_exact_tree "$BACKUP/tools" "$CD/tools"
  sudo -n systemctl restart xrd-cmdcenter
  echo "deployment failed; restored rollback snapshot $BACKUP" >&2
  exit "$status"
}
trap restore_previous ERR

cp -a "$STAGE_REAL/app.py" "$CD/app.py"
cp -a "$STAGE_REAL/requirements-production.txt" "$CD/requirements-production.txt"
if [ -d "$STAGE_REAL/cmdcenter" ]; then
  sync_exact_tree "$STAGE_REAL/cmdcenter" "$CD/cmdcenter"
fi
sync_exact_tree "$STAGE_REAL/public_evidence" "$CD/public_evidence"
sync_exact_tree "$STAGE_REAL/systemd" "$CD/systemd"
cp -a "$STAGE_REAL/assets.json" "$CD/assets.json"
cp -a "$STAGE_REAL/asset-manifest.json" "$CD/asset-manifest.json"
sync_exact_tree "$STAGE_REAL/static" "$CD/static"
sync_exact_tree "$STAGE_REAL/tools" "$CD/tools"
XRD_CMD_TEST_MODE=1 "$PY" "$CD/tools/site31_asset_manifest.py" "$CD" --verify >/dev/null
sudo -n install -m 0644 "$CD/systemd/xrd-cmdcenter.service" /etc/systemd/system/xrd-cmdcenter.service
sudo -n systemctl daemon-reload
sudo -n systemctl restart xrd-cmdcenter

for _ in 1 2 3 4 5 6 7 8; do
  if curl -fsS --max-time 8 http://127.0.0.1:29100/api/public_status >/dev/null; then
    break
  fi
  sleep 1
done
STATUS_JSON="$(curl -fsS --max-time 8 http://127.0.0.1:29100/api/public_status)"
printf '%s' "$STATUS_JSON" | "$PY" -c \
  'import json,sys; expected=sys.argv[1]; data=json.load(sys.stdin); actual=data.get("release") or (data.get("summary") or {}).get("release"); assert actual==expected, {"expected": expected, "actual": actual}' "$VER"

# Post-deploy checks use the promoted candidate tools, not pre-existing live tools.
XRD_CMD_TEST_MODE=1 "$PY" "$CD/tools/site31_gate_audit.py" "$CD" --phase deployed \
  --base-url http://127.0.0.1:29100 \
  --output "$CD/static/quality/site31_gate_evidence.json"
XRD_CMD_TEST_MODE=1 "$PY" "$CD/tools/site31_asset_manifest.py" "$CD" --write >/dev/null
if is_site32_release "$VER"; then
  XRD_CMD_TEST_MODE=1 "$PY" "$CD/tools/site32_style_audit.py" --root "$CD" \
    --output "$CD/static/quality/site32_style_audit.json" >/dev/null
  XRD_CMD_TEST_MODE=1 "$PY" "$CD/tools/site31_asset_manifest.py" "$CD" --write >/dev/null
fi
XRD_CMD_TEST_MODE=1 "$PY" "$CD/tools/site31_smoke.py" "$CD"
XRD_CMD_TEST_MODE=1 "$PY" "$CD/tools/site31_asset_manifest.py" "$CD" --write >/dev/null
# Atomic quality writers may replace an existing file. Reassert the public
# release read contract before the sandboxed service performs its own scan.
normalize_release_payload_modes "$CD"
LIVE_MANIFEST_JSON="$(XRD_CMD_TEST_MODE=1 "$PY" "$CD/tools/site31_asset_manifest.py" "$CD" --verify)"
printf '%s' "$LIVE_MANIFEST_JSON" | "$PY" -c \
  'import json,sys; data=json.load(sys.stdin); assert data.get("release")==sys.argv[1], data; assert data.get("manifest_digest")==sys.argv[2], data' \
  "$VER" "$EXPECTED_DIGEST"

GATE_JSON="$("${REVIEW_CURL[@]}" http://127.0.0.1:29100/api/site31_gate_evidence)"
printf '%s' "$GATE_JSON" | "$PY" -c \
  'import json,sys; data=json.load(sys.stdin); assert data.get("valid") is True, data; assert data.get("gate")=="pass", data; assert data.get("release")==sys.argv[1], data; assert data.get("asset_manifest",{}).get("manifest_digest")==sys.argv[2], data' \
  "$VER" "$EXPECTED_DIGEST"
"${REVIEW_CURL[@]}" http://127.0.0.1:29100/api/site31_scorecard | "$PY" -c \
  'import json,sys; data=json.load(sys.stdin); assert data.get("gate")=="pass", data'

FILES="$(printf '%s' "$LIVE_MANIFEST_JSON" | "$PY" -c 'import json,sys; print(",".join(x["path"] for x in json.load(sys.stdin)["files"]))')"
SHA="${EXPECTED_DIGEST:0:16}"
sudo -n -u xrd-cmdcenter env XRD_RELEASE_DB=/var/lib/xrd-cmdcenter/data.db \
  /usr/bin/python3 - "$VER" "$NOTES" "$BY" "$SHA" "$FILES" <<'PY'
import sqlite3
import sys
import time

ver, notes, by, sha, files = sys.argv[1:6]
con = sqlite3.connect("/var/lib/xrd-cmdcenter/data.db", timeout=10)
con.execute("CREATE TABLE IF NOT EXISTS releases(id INTEGER PRIMARY KEY AUTOINCREMENT, ver TEXT, ts INTEGER, files TEXT, sha TEXT, notes TEXT, by TEXT)")
con.execute("INSERT INTO releases(ver,ts,files,sha,notes,by) VALUES(?,?,?,?,?,?)",
            (ver, int(time.time()), files, sha, notes, by))
con.commit()
con.close()
print("release recorded:", ver, sha)
PY

printf '%s\n' "$BACKUP" > "$CD/_releases/.prev"
trap - ERR
echo "deployed $VER with manifest $EXPECTED_DIGEST; rollback snapshot: $BACKUP"
