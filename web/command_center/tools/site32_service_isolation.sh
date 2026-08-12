#!/usr/bin/env bash
# Site32 auth/cmdcenter service-isolation migration helper.
#
# Default command is read-only preflight. Mutating commands require sudo -n and
# only touch the fixed Site32 service-isolation paths documented in the runbook.
set -euo pipefail

AUTH_ROOT="${XRD_AUTH_ROOT:-/home/rdk/auth}"
CMD_ROOT="${XRD_CMD_ROOT:-/home/rdk/cmdcenter}"
AUTH_HEALTH_URL="${XRD_AUTH_HEALTH_URL:-http://127.0.0.1:29000/healthz}"
CMD_HEALTH_URL="${XRD_CMD_HEALTH_URL:-http://127.0.0.1:29100/healthz}"
SNAPSHOT_ROOT="${XRD_SITE32_SNAPSHOT_ROOT:-/root/xrd-systemd-isolation}"
SUDO=(sudo -n)

AUTH_UNIT=/etc/systemd/system/xrd-auth.service
CMD_UNIT=/etc/systemd/system/xrd-cmdcenter.service
AUTH_DROPIN_DIR=/etc/systemd/system/xrd-auth.service.d
CMD_DROPIN_DIR=/etc/systemd/system/xrd-cmdcenter.service.d
AUTH_ROTATE_UNIT=/etc/systemd/system/xrd-auth-audit-rotate.service
AUTH_ROTATE_ALERT_UNIT=/etc/systemd/system/xrd-auth-audit-rotate-alert@.service
AUTH_ROTATE_TIMER=/etc/systemd/system/xrd-auth-audit-rotate.timer
AUTH_LOGROTATE=/etc/logrotate.d/xrd-auth
AUTH_ETC=/etc/xrd-auth
CMD_ETC=/etc/xrd-cmdcenter
AUTH_LIB=/var/lib/xrd-auth
AUTH_LOG=/var/log/xrd-auth
AUTH_ROTATE_STATE=/var/lib/xrd-auth-logrotate
CMD_LIB=/var/lib/xrd-cmdcenter

SNAPSHOT_OVERRIDE=""
AUTH_ACTIVATION_COMMITTED=1
AUTH_ACTIVATION_SNAPSHOT=""

die() {
  printf 'site32-service-isolation: %s\n' "$*" >&2
  exit 1
}

note() {
  printf '%s\n' "$*"
}

usage() {
  cat <<'EOF'
Usage:
  site32_service_isolation.sh [preflight]
  site32_service_isolation.sh prepare
  site32_service_isolation.sh activate-auth [--snapshot SNAPSHOT]
  site32_service_isolation.sh verify [--snapshot SNAPSHOT]
  site32_service_isolation.sh rollback-auth [--snapshot SNAPSHOT]

The helper never changes Caddy, UFW, Wi-Fi, VPN, routes, ARP, or SSH config.
EOF
}

parse_common_args() {
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --snapshot)
        [ "$#" -ge 2 ] || die "--snapshot requires a value"
        SNAPSHOT_OVERRIDE="$2"
        shift 2
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "unknown argument: $1"
        ;;
    esac
  done
}

require_abs() {
  case "${1:-}" in
    /*) ;;
    *) die "path must be absolute: ${1:-<empty>}" ;;
  esac
}

require_name_safe() {
  case "${1:-}" in
    *[!A-Za-z0-9._@+-]*|''|'.'|'..')
      die "unsafe path component: ${1:-<empty>}"
      ;;
  esac
}

resolve_existing() {
  local path="$1"
  require_abs "$path"
  "${SUDO[@]}" readlink -e -- "$path"
}

require_under() {
  local path="$1"
  local root="$2"
  require_abs "$path"
  require_abs "$root"
  case "$path/" in
    "$root"/*) ;;
    *) die "path escapes $root: $path" ;;
  esac
}

refuse_symlink_if_exists() {
  local path="$1"
  require_abs "$path"
  if "${SUDO[@]}" test -L "$path"; then
    die "refusing symlink path: $path"
  fi
}

require_existing_non_symlink() {
  local path="$1"
  refuse_symlink_if_exists "$path"
  "${SUDO[@]}" test -e "$path" || die "missing required path: $path"
  resolve_existing "$path" >/dev/null || die "unresolvable path: $path"
}

require_file_non_symlink() {
  local path="$1"
  require_existing_non_symlink "$path"
  "${SUDO[@]}" test -f "$path" || die "required path is not a regular file: $path"
}

require_dir_non_symlink() {
  local path="$1"
  require_existing_non_symlink "$path"
  "${SUDO[@]}" test -d "$path" || die "required path is not a directory: $path"
}

require_optional_non_symlink() {
  local path="$1"
  refuse_symlink_if_exists "$path"
  if "${SUDO[@]}" test -e "$path"; then
    resolve_existing "$path" >/dev/null || die "unresolvable optional path: $path"
  fi
}

require_known_write_path() {
  local path="$1"
  require_abs "$path"
  case "$path" in
    "$SNAPSHOT_ROOT"|"$SNAPSHOT_ROOT"/*|\
    "$AUTH_ETC"|"$AUTH_ETC"/*|\
    "$CMD_ETC"|"$CMD_ETC"/*|\
    "$AUTH_LIB"|"$AUTH_LIB"/*|\
    "$AUTH_LOG"|"$AUTH_LOG"/*|\
    "$AUTH_ROTATE_STATE"|"$AUTH_ROTATE_STATE"/*|\
    "$CMD_LIB"|"$CMD_LIB"/*|\
    "$AUTH_ROOT/logins.jsonl"|"$AUTH_ROOT"/logins.jsonl.*|\
    "$AUTH_ROOT/users.json"|"$AUTH_ROOT/secret.key"|\
    "$CMD_ROOT/data.db"|"$CMD_ROOT/data.db.rollback-new"|\
    "$AUTH_UNIT"|"$AUTH_DROPIN_DIR"|"$AUTH_DROPIN_DIR"/*|\
    "$AUTH_ROTATE_UNIT"|"$AUTH_ROTATE_ALERT_UNIT"|"$AUTH_ROTATE_TIMER"|"$AUTH_LOGROTATE")
      ;;
    *)
      die "refusing to write outside managed paths: $path"
      ;;
  esac
}

safe_install_dir() {
  local owner="$1"
  local group="$2"
  local mode="$3"
  local path="$4"
  require_known_write_path "$path"
  refuse_symlink_if_exists "$path"
  "${SUDO[@]}" install -d -o "$owner" -g "$group" -m "$mode" "$path"
}

safe_install_file() {
  local owner="$1"
  local group="$2"
  local mode="$3"
  local source="$4"
  local dest="$5"
  require_file_non_symlink "$source"
  require_known_write_path "$dest"
  refuse_symlink_if_exists "$dest"
  "${SUDO[@]}" install -o "$owner" -g "$group" -m "$mode" "$source" "$dest"
}

safe_install_empty() {
  local owner="$1"
  local group="$2"
  local mode="$3"
  local dest="$4"
  require_known_write_path "$dest"
  refuse_symlink_if_exists "$dest"
  "${SUDO[@]}" install -o "$owner" -g "$group" -m "$mode" /dev/null "$dest"
}

safe_move_known() {
  local source="$1"
  local dest="$2"
  require_known_write_path "$source"
  require_known_write_path "$dest"
  refuse_symlink_if_exists "$source"
  "${SUDO[@]}" test ! -e "$dest" || die "destination already exists: $dest"
  "${SUDO[@]}" mv -- "$source" "$dest"
}

safe_copy_tree() {
  local source="$1"
  local dest="$2"
  require_dir_non_symlink "$source"
  if "${SUDO[@]}" find "$source" -xdev -type l -print -quit | grep -q .; then
    die "refusing report tree containing symlinks: $source"
  fi
  require_known_write_path "$dest"
  refuse_symlink_if_exists "$dest"
  safe_install_dir xrd-cmdcenter xrd-cmdcenter 0750 "$dest"
  "${SUDO[@]}" cp -a "$source/." "$dest/"
}

assert_sudo_noninteractive() {
  "${SUDO[@]}" /usr/bin/true
}

validate_fixed_roots() {
  require_abs "$AUTH_ROOT"
  require_abs "$CMD_ROOT"
  require_abs "$SNAPSHOT_ROOT"
  case "$AUTH_ROOT" in /home/rdk/auth) ;; *) die "unexpected AUTH_ROOT: $AUTH_ROOT" ;; esac
  case "$CMD_ROOT" in /home/rdk/cmdcenter) ;; *) die "unexpected CMD_ROOT: $CMD_ROOT" ;; esac
  case "$SNAPSHOT_ROOT" in /root/xrd-systemd-isolation) ;; *) die "unexpected SNAPSHOT_ROOT: $SNAPSHOT_ROOT" ;; esac
  require_dir_non_symlink "$AUTH_ROOT"
  require_dir_non_symlink "$CMD_ROOT"
  [ "$(resolve_existing "$AUTH_ROOT")" = "$AUTH_ROOT" ] || die "AUTH_ROOT resolves outside fixed path"
  [ "$(resolve_existing "$CMD_ROOT")" = "$CMD_ROOT" ] || die "CMD_ROOT resolves outside fixed path"
}

validate_required_sources() {
  local path
  for path in \
    "$AUTH_ROOT/app.py" \
    "$AUTH_ROOT/.venv/bin/gunicorn" \
    "$AUTH_ROOT/users.json" \
    "$AUTH_ROOT/secret.key" \
    "$AUTH_ROOT/systemd/xrd-auth.service" \
    "$AUTH_ROOT/systemd/xrd-auth-audit-rotate.service" \
    "$AUTH_ROOT/systemd/xrd-auth-audit-rotate-alert@.service" \
    "$AUTH_ROOT/systemd/xrd-auth-audit-rotate.timer" \
    "$AUTH_ROOT/systemd/xrd-auth.logrotate" \
    "$CMD_ROOT/app.py" \
    "$CMD_ROOT/.venv/bin/gunicorn" \
    "$CMD_ROOT/secrets.env" \
    "$CMD_ROOT/systemd/xrd-cmdcenter.service"; do
    require_existing_non_symlink "$path"
  done
  "${SUDO[@]}" test -x /usr/sbin/logrotate || die "missing required executable: /usr/sbin/logrotate"
  "${SUDO[@]}" test -x "$AUTH_ROOT/.venv/bin/gunicorn" || die "auth gunicorn is not executable"
  "${SUDO[@]}" test -x "$CMD_ROOT/.venv/bin/gunicorn" || die "cmdcenter gunicorn is not executable"
  "${SUDO[@]}" test -s "$AUTH_ROOT/users.json" || die "empty auth users.json"
  "${SUDO[@]}" test -s "$AUTH_ROOT/secret.key" || die "empty auth secret.key"
  "${SUDO[@]}" test -s "$CMD_ROOT/secrets.env" || die "empty cmdcenter secrets.env"

  for path in \
    "$AUTH_ROOT/users.json" \
    "$AUTH_ROOT/secret.key" \
    "$CMD_ROOT/secrets.env" \
    "$AUTH_ROOT/logins.jsonl" \
    "$CMD_ROOT/data.db" \
    "$CMD_ROOT/alert_email.json" \
    "$CMD_ROOT/reports" \
    "$AUTH_LIB/users.json" \
    "$AUTH_LOG/logins.jsonl" \
    "$AUTH_ETC/secret.key" \
    "$CMD_ETC/secrets.env" \
    "$CMD_ETC/alert_email.json" \
    "$CMD_LIB/data.db" \
    "$CMD_LIB/reports"; do
    require_optional_non_symlink "$path"
  done
}

validate_active_units() {
  require_file_non_symlink "$AUTH_UNIT"
  require_file_non_symlink "$CMD_UNIT"
  if "${SUDO[@]}" test -d "$AUTH_DROPIN_DIR"; then
    require_dir_non_symlink "$AUTH_DROPIN_DIR"
  fi
  if "${SUDO[@]}" test -d "$CMD_DROPIN_DIR"; then
    require_dir_non_symlink "$CMD_DROPIN_DIR"
  fi
  local auth_fragment cmd_fragment
  auth_fragment="$(systemctl show xrd-auth -p FragmentPath --value)"
  cmd_fragment="$(systemctl show xrd-cmdcenter -p FragmentPath --value)"
  [ "$auth_fragment" = "$AUTH_UNIT" ] || die "unexpected xrd-auth FragmentPath: $auth_fragment"
  [ "$cmd_fragment" = "$CMD_UNIT" ] || die "unexpected xrd-cmdcenter FragmentPath: $cmd_fragment"
}

wait_for_url() {
  local url="$1"
  local attempts="$2"
  local delay="$3"
  local i
  for i in $(seq 1 "$attempts"); do
    if curl -fsS --max-time 5 "$url" >/dev/null; then
      return 0
    fi
    sleep "$delay"
  done
  curl -fsS --max-time 8 "$url" >/dev/null
}

sqlite_backup_checked() {
  local src="$1"
  local dst="$2"
  local tmp="${dst}.new.$$"
  require_file_non_symlink "$src"
  require_known_write_path "$dst"
  require_known_write_path "$tmp"
  refuse_symlink_if_exists "$dst"
  trap 'if [ -n "${tmp:-}" ]; then "${SUDO[@]}" rm -f -- "$tmp" 2>/dev/null || true; fi' RETURN
  "${SUDO[@]}" env XRD_DB_SRC="$src" XRD_DB_DST="$tmp" "$CMD_ROOT/.venv/bin/python" - <<'PY'
import os
import sqlite3

source = sqlite3.connect(os.environ["XRD_DB_SRC"])
target = sqlite3.connect(os.environ["XRD_DB_DST"])
try:
    source.backup(target)
    result = target.execute("PRAGMA quick_check").fetchone()[0]
    if result != "ok":
        raise SystemExit(f"SQLite quick_check failed: {result}")
finally:
    target.close()
    source.close()
PY
  "${SUDO[@]}" install -o xrd-cmdcenter -g xrd-cmdcenter -m 0640 "$tmp" "$dst"
  "${SUDO[@]}" rm -f -- "$tmp"
  trap - RETURN
}

sqlite_quick_check_ro() {
  local db="$1"
  require_file_non_symlink "$db"
  "${SUDO[@]}" -u xrd-cmdcenter /usr/bin/python3 - "$db" <<'PY'
import sqlite3
import sys

connection = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
try:
    result = connection.execute("PRAGMA quick_check").fetchone()[0]
finally:
    connection.close()
if result != "ok":
    raise SystemExit(f"SQLite quick_check failed: {result}")
PY
}

append_tar_path() {
  local tarball="$1"
  local path="$2"
  require_abs "$path"
  case "$path" in
    "$AUTH_ROOT"/*|"$CMD_ROOT"/*|"$AUTH_ETC"/*|"$CMD_ETC"/*|"$AUTH_LIB"/*|"$AUTH_LOG"/*|"$CMD_LIB"/*)
      ;;
    *)
      die "refusing to snapshot unmanaged path: $path"
      ;;
  esac
  refuse_symlink_if_exists "$path"
  if "${SUDO[@]}" test -e "$path"; then
    local rel="${path#/}"
    "${SUDO[@]}" tar --acls --xattrs -C / -rpf "$tarball" "$rel"
  fi
}

append_login_globs() {
  local tarball="$1"
  local dir="$2"
  local audit_file
  if ! "${SUDO[@]}" test -d "$dir"; then
    return 0
  fi
  require_dir_non_symlink "$dir"
  while IFS= read -r -d '' audit_file; do
    append_tar_path "$tarball" "$audit_file"
  done < <("${SUDO[@]}" find "$dir" -maxdepth 1 -type f -name 'logins.jsonl*' -print0)
}

write_audit_hashes() {
  local dir="$1"
  local out="$2"
  require_known_write_path "$out"
  if "${SUDO[@]}" test -d "$dir"; then
    require_dir_non_symlink "$dir"
    "${SUDO[@]}" sh -c 'cd "$1" && find . -maxdepth 1 -type f -name "logins.jsonl*" -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum' sh "$dir" |
      "${SUDO[@]}" tee "$out" >/dev/null
  else
    safe_install_empty root root 0600 "$out"
  fi
}

create_snapshot() {
  local stamp snapshot tarball
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  snapshot="$SNAPSHOT_ROOT/$stamp"
  require_known_write_path "$SNAPSHOT_ROOT"
  require_known_write_path "$snapshot"
  safe_install_dir root root 0700 "$SNAPSHOT_ROOT"
  "${SUDO[@]}" test ! -e "$snapshot" || die "snapshot already exists: $snapshot"
  safe_install_dir root root 0700 "$snapshot"

  safe_install_file root root 0600 "$AUTH_UNIT" "$snapshot/xrd-auth.service.before"
  safe_install_file root root 0600 "$CMD_UNIT" "$snapshot/xrd-cmdcenter.service.before"
  if "${SUDO[@]}" test -d "$AUTH_DROPIN_DIR"; then
    safe_copy_dropin "$AUTH_DROPIN_DIR" "$snapshot/xrd-auth.service.d.before"
  fi
  if "${SUDO[@]}" test -d "$CMD_DROPIN_DIR"; then
    safe_copy_dropin "$CMD_DROPIN_DIR" "$snapshot/xrd-cmdcenter.service.d.before"
  fi
  snapshot_optional_system_file "$AUTH_ROTATE_UNIT" "$snapshot/xrd-auth-audit-rotate.service.before"
  snapshot_optional_system_file "$AUTH_ROTATE_ALERT_UNIT" "$snapshot/xrd-auth-audit-rotate-alert@.service.before"
  snapshot_optional_system_file "$AUTH_ROTATE_TIMER" "$snapshot/xrd-auth-audit-rotate.timer.before"
  snapshot_optional_system_file "$AUTH_LOGROTATE" "$snapshot/xrd-auth.logrotate.before"

  systemctl cat xrd-auth | "${SUDO[@]}" tee "$snapshot/xrd-auth.effective.before" >/dev/null
  systemctl cat xrd-cmdcenter | "${SUDO[@]}" tee "$snapshot/xrd-cmdcenter.effective.before" >/dev/null
  (systemctl is-enabled xrd-auth-audit-rotate.timer 2>/dev/null || true) |
    "${SUDO[@]}" tee "$snapshot/xrd-auth-audit-rotate.timer.enabled.before" >/dev/null
  "${SUDO[@]}" chmod 0600 "$snapshot"/*.before "$snapshot"/*.effective.before "$snapshot"/*.enabled.before 2>/dev/null || true

  tarball="$snapshot/private-state.tar"
  "${SUDO[@]}" tar --acls --xattrs -C / -cpf "$tarball" \
    "${AUTH_ROOT#/}/users.json" \
    "${AUTH_ROOT#/}/secret.key" \
    "${CMD_ROOT#/}/secrets.env"

  for state_path in \
    "$CMD_ROOT/data.db" \
    "$CMD_ROOT/alert_email.json" \
    "$CMD_ROOT/reports" \
    "$AUTH_LIB/users.json" \
    "$AUTH_ETC/secret.key" \
    "$AUTH_LOG/logins.jsonl" \
    "$AUTH_LIB/logins.jsonl" \
    "$CMD_ETC/secrets.env" \
    "$CMD_ETC/alert_email.json" \
    "$CMD_LIB/data.db" \
    "$CMD_LIB/reports"; do
    append_tar_path "$tarball" "$state_path"
  done
  append_login_globs "$tarball" "$AUTH_ROOT"
  append_login_globs "$tarball" "$AUTH_LOG"
  write_audit_hashes "$AUTH_ROOT" "$snapshot/auth-audit-sha256.legacy.before"
  write_audit_hashes "$AUTH_LOG" "$snapshot/auth-audit-sha256.target.before"
  "${SUDO[@]}" sha256sum "$tarball" | "${SUDO[@]}" tee "$snapshot/private-state.tar.sha256" >/dev/null
  if "${SUDO[@]}" test -e "$SNAPSHOT_ROOT/PENDING"; then
    safe_move_known "$SNAPSHOT_ROOT/PENDING" "$snapshot/PENDING.previous"
  fi
  printf '%s\n' "$snapshot" | "${SUDO[@]}" tee "$SNAPSHOT_ROOT/PENDING.tmp" >/dev/null
  safe_move_known "$SNAPSHOT_ROOT/PENDING.tmp" "$SNAPSHOT_ROOT/PENDING"
  note "$snapshot"
}

safe_copy_dropin() {
  local source="$1"
  local dest="$2"
  require_dir_non_symlink "$source"
  require_known_write_path "$dest"
  "${SUDO[@]}" test ! -e "$dest" || die "snapshot drop-in destination exists: $dest"
  "${SUDO[@]}" cp -a "$source" "$dest"
  "${SUDO[@]}" chmod -R go-rwx "$dest"
}

snapshot_optional_system_file() {
  local source="$1"
  local dest="$2"
  if "${SUDO[@]}" test -e "$source"; then
    safe_install_file root root 0600 "$source" "$dest"
  fi
}

current_snapshot() {
  local snapshot=""
  if [ -n "$SNAPSHOT_OVERRIDE" ]; then
    snapshot="$SNAPSHOT_OVERRIDE"
  elif "${SUDO[@]}" test -s "$SNAPSHOT_ROOT/PENDING"; then
    snapshot="$("${SUDO[@]}" cat "$SNAPSHOT_ROOT/PENDING")"
  elif "${SUDO[@]}" test -s "$SNAPSHOT_ROOT/LAST_AUTH_SUCCESSFUL"; then
    snapshot="$("${SUDO[@]}" cat "$SNAPSHOT_ROOT/LAST_AUTH_SUCCESSFUL")"
  else
    die "no snapshot selected; run prepare first or pass --snapshot"
  fi
  require_abs "$snapshot"
  require_under "$snapshot" "$SNAPSHOT_ROOT"
  require_dir_non_symlink "$snapshot"
  printf '%s\n' "$snapshot"
}

create_identities_and_dirs() {
  "${SUDO[@]}" getent group xrd-auth-readers >/dev/null || "${SUDO[@]}" groupadd --system xrd-auth-readers
  "${SUDO[@]}" getent group xrd-auth >/dev/null || "${SUDO[@]}" groupadd --system xrd-auth
  "${SUDO[@]}" getent group xrd-cmdcenter >/dev/null || "${SUDO[@]}" groupadd --system xrd-cmdcenter

  if ! "${SUDO[@]}" id xrd-auth >/dev/null 2>&1; then
    "${SUDO[@]}" useradd --system --gid xrd-auth --home-dir "$AUTH_LIB" --no-create-home --shell /usr/sbin/nologin xrd-auth
  fi
  if ! "${SUDO[@]}" id xrd-cmdcenter >/dev/null 2>&1; then
    "${SUDO[@]}" useradd --system --gid xrd-cmdcenter --home-dir "$CMD_LIB" --no-create-home --shell /usr/sbin/nologin xrd-cmdcenter
  fi

  [ "$("${SUDO[@]}" id -gn xrd-auth)" = xrd-auth ] || die "xrd-auth primary group mismatch"
  [ "$("${SUDO[@]}" id -gn xrd-cmdcenter)" = xrd-cmdcenter ] || die "xrd-cmdcenter primary group mismatch"
  [ "$("${SUDO[@]}" getent passwd xrd-auth | cut -d: -f7)" = /usr/sbin/nologin ] || die "xrd-auth shell mismatch"
  [ "$("${SUDO[@]}" getent passwd xrd-cmdcenter | cut -d: -f7)" = /usr/sbin/nologin ] || die "xrd-cmdcenter shell mismatch"
  [ "$("${SUDO[@]}" id -u xrd-auth)" != "$("${SUDO[@]}" id -u xrd-cmdcenter)" ] || die "service users share uid"
  "${SUDO[@]}" usermod -a -G xrd-auth-readers xrd-auth
  "${SUDO[@]}" usermod -a -G xrd-auth-readers xrd-cmdcenter
  local unexpected_readers
  unexpected_readers="$("${SUDO[@]}" getent group xrd-auth-readers | cut -d: -f4 | tr ',' '\n' | grep -Ev '^(xrd-auth|xrd-cmdcenter)?$' || true)"
  [ -z "$unexpected_readers" ] || die "unexpected xrd-auth-readers members: $unexpected_readers"

  safe_install_dir root xrd-auth 0750 "$AUTH_ETC"
  safe_install_dir root xrd-cmdcenter 0750 "$CMD_ETC"
  safe_install_dir root xrd-auth-readers 0750 "$AUTH_LIB"
  safe_install_dir xrd-auth xrd-auth-readers 2750 "$AUTH_LOG"
  safe_install_dir root root 0700 "$AUTH_ROTATE_STATE"
  safe_install_dir xrd-cmdcenter xrd-cmdcenter 0750 "$CMD_LIB"
  safe_install_dir xrd-cmdcenter xrd-cmdcenter 0750 "$CMD_LIB/reports"
}

copy_auth_state_to_targets() {
  local audit_file audit_count=0
  safe_install_file root xrd-auth-readers 0640 "$AUTH_ROOT/users.json" "$AUTH_LIB/users.json"
  safe_install_file root xrd-auth 0640 "$AUTH_ROOT/secret.key" "$AUTH_ETC/secret.key"
  while IFS= read -r -d '' audit_file; do
    require_file_non_symlink "$audit_file"
    safe_install_file xrd-auth xrd-auth-readers 0640 "$audit_file" "$AUTH_LOG/$(basename "$audit_file")"
    audit_count=$((audit_count + 1))
  done < <("${SUDO[@]}" find "$AUTH_ROOT" -maxdepth 1 -type f -name 'logins.jsonl*' -print0)
  if [ "$audit_count" -eq 0 ]; then
    safe_install_empty xrd-auth xrd-auth-readers 0640 "$AUTH_LOG/logins.jsonl"
    safe_install_empty ubuntu ubuntu 0640 "$AUTH_ROOT/logins.jsonl"
  fi
  safe_install_empty root xrd-auth-readers 0640 "$AUTH_LIB/logins.jsonl"
}

copy_cmdcenter_state_to_targets() {
  safe_install_file root root 0600 "$CMD_ROOT/secrets.env" "$CMD_ETC/secrets.env"
  if "${SUDO[@]}" test -f "$CMD_ROOT/alert_email.json"; then
    safe_install_file root xrd-cmdcenter 0640 "$CMD_ROOT/alert_email.json" "$CMD_ETC/alert_email.json"
  fi
  if "${SUDO[@]}" test -f "$CMD_ROOT/data.db"; then
    sqlite_backup_checked "$CMD_ROOT/data.db" "$CMD_LIB/data.db"
  fi
  if "${SUDO[@]}" test -d "$CMD_ROOT/reports"; then
    safe_copy_tree "$CMD_ROOT/reports" "$CMD_LIB/reports"
    "${SUDO[@]}" chown -R xrd-cmdcenter:xrd-cmdcenter "$CMD_LIB/reports"
    "${SUDO[@]}" find "$CMD_LIB/reports" -type d -exec chmod 0750 {} +
    "${SUDO[@]}" find "$CMD_LIB/reports" -type f -exec chmod 0640 {} +
  fi
}

install_auth_rotation_assets() {
  safe_install_file root root 0644 "$AUTH_ROOT/systemd/xrd-auth.logrotate" "$AUTH_LOGROTATE"
  safe_install_file root root 0644 "$AUTH_ROOT/systemd/xrd-auth-audit-rotate.service" "$AUTH_ROTATE_UNIT"
  safe_install_file root root 0644 "$AUTH_ROOT/systemd/xrd-auth-audit-rotate-alert@.service" "$AUTH_ROTATE_ALERT_UNIT"
  safe_install_file root root 0644 "$AUTH_ROOT/systemd/xrd-auth-audit-rotate.timer" "$AUTH_ROTATE_TIMER"
}

verify_candidate_units() {
  "${SUDO[@]}" systemd-analyze verify \
    "$AUTH_ROOT/systemd/xrd-auth.service" \
    "$AUTH_ROOT/systemd/xrd-auth-audit-rotate.service" \
    "$AUTH_ROOT/systemd/xrd-auth-audit-rotate-alert@.service" \
    "$AUTH_ROOT/systemd/xrd-auth-audit-rotate.timer" \
    "$CMD_ROOT/systemd/xrd-cmdcenter.service" >/dev/null
  require_file_non_symlink "$AUTH_LOGROTATE"
  [ "$("${SUDO[@]}" stat -c '%a' "$AUTH_LOGROTATE")" = 644 ] ||
    die "installed auth logrotate config must be mode 0644"
  "${SUDO[@]}" /usr/sbin/logrotate --debug "$AUTH_LOGROTATE" >/dev/null
}

preflight() {
  assert_sudo_noninteractive
  validate_fixed_roots
  validate_required_sources
  validate_active_units
  command -v systemd-analyze >/dev/null || die "missing systemd-analyze"
  command -v curl >/dev/null || die "missing curl"
  command -v tar >/dev/null || die "missing tar"
  command -v sha256sum >/dev/null || die "missing sha256sum"
  findmnt -no TARGET,OPTIONS -T "$AUTH_ROOT"
  findmnt -no TARGET,OPTIONS -T "$CMD_ROOT"
  "${SUDO[@]}" -u ubuntu test -w "$AUTH_ROOT" || die "ubuntu cannot write AUTH_ROOT"
  "${SUDO[@]}" -u ubuntu test -w "$CMD_ROOT" || die "ubuntu cannot write CMD_ROOT"
  systemctl is-active --quiet xrd-auth || die "xrd-auth is not active"
  systemctl is-active --quiet xrd-cmdcenter || die "xrd-cmdcenter is not active"
  wait_for_url "$AUTH_HEALTH_URL" 1 1
  wait_for_url "$CMD_HEALTH_URL" 1 1
  systemctl show xrd-auth -p FragmentPath -p DropInPaths
  systemctl show xrd-cmdcenter -p FragmentPath -p DropInPaths
  "${SUDO[@]}" stat -c '%n %F %U:%G %a' "$AUTH_ROOT/users.json" "$AUTH_ROOT/secret.key" "$CMD_ROOT/secrets.env"
  note "preflight ok"
}

prepare() {
  assert_sudo_noninteractive
  validate_fixed_roots
  validate_required_sources
  validate_active_units
  if "${SUDO[@]}" test -e "$SNAPSHOT_ROOT/PENDING"; then
    die "pending migration already exists; verify or roll it back before preparing again"
  fi
  local snapshot
  snapshot="$(create_snapshot)"
  create_identities_and_dirs
  copy_auth_state_to_targets
  copy_cmdcenter_state_to_targets
  install_auth_rotation_assets
  verify_candidate_units
  write_audit_hashes "$AUTH_LOG" "$snapshot/auth-audit-sha256.target.after-prepare"
  note "prepare ok: $snapshot"
  note "services were not stopped and main units were not installed"
}

restore_auth_unit_from_snapshot() {
  local snapshot="$1"
  local suffix="$2"
  local saved_dropin="$snapshot/xrd-auth.service.d.before"
  local active_save="$snapshot/xrd-auth.service.d.$suffix"
  safe_install_file root root 0644 "$snapshot/xrd-auth.service.before" "$AUTH_UNIT"
  if "${SUDO[@]}" test -e "$AUTH_DROPIN_DIR"; then
    safe_move_known "$AUTH_DROPIN_DIR" "$active_save"
  fi
  if "${SUDO[@]}" test -d "$saved_dropin"; then
    safe_copy_dropin "$saved_dropin" "$AUTH_DROPIN_DIR"
  fi
  "${SUDO[@]}" systemctl daemon-reload
  "${SUDO[@]}" systemctl reset-failed xrd-auth || true
}

rollback_auth_activation_on_exit() {
  local status=$?
  local suffix
  trap - EXIT
  if [ "$AUTH_ACTIVATION_COMMITTED" -eq 0 ] && [ -n "$AUTH_ACTIVATION_SNAPSHOT" ]; then
    suffix="failed-activation-$(date -u +%Y%m%dT%H%M%SZ)"
    note "auth activation failed; restoring previous unit from $AUTH_ACTIVATION_SNAPSHOT" >&2
    set +e
    restore_auth_unit_from_snapshot "$AUTH_ACTIVATION_SNAPSHOT" "$suffix"
    "${SUDO[@]}" systemctl restart xrd-auth
    wait_for_url "$AUTH_HEALTH_URL" 20 1
    set -e
  fi
  if [ "$status" -eq 0 ]; then
    status=1
  fi
  exit "$status"
}

activate_auth() {
  assert_sudo_noninteractive
  validate_fixed_roots
  validate_required_sources
  local snapshot
  snapshot="$(current_snapshot)"
  require_file_non_symlink "$snapshot/xrd-auth.service.before"
  require_file_non_symlink "$AUTH_LIB/users.json"
  require_file_non_symlink "$AUTH_ETC/secret.key"
  require_file_non_symlink "$AUTH_LOG/logins.jsonl"
  AUTH_ACTIVATION_COMMITTED=0
  AUTH_ACTIVATION_SNAPSHOT="$snapshot"
  trap rollback_auth_activation_on_exit EXIT
  install_auth_rotation_assets
  verify_candidate_units
  if "${SUDO[@]}" test -e "$AUTH_DROPIN_DIR"; then
    safe_move_known "$AUTH_DROPIN_DIR" "$snapshot/xrd-auth.service.d.disabled-for-auth-activate"
  fi
  safe_install_file root root 0644 "$AUTH_ROOT/systemd/xrd-auth.service" "$AUTH_UNIT"
  "${SUDO[@]}" systemctl daemon-reload
  "${SUDO[@]}" systemctl reset-failed xrd-auth || true
  "${SUDO[@]}" systemctl restart xrd-auth
  wait_for_url "$AUTH_HEALTH_URL" 20 1
  "${SUDO[@]}" systemctl enable --now xrd-auth-audit-rotate.timer >/dev/null
  printf '%s\n' "$snapshot" | "${SUDO[@]}" tee "$SNAPSHOT_ROOT/LAST_AUTH_SUCCESSFUL" >/dev/null
  AUTH_ACTIVATION_COMMITTED=1
  AUTH_ACTIVATION_SNAPSHOT=""
  trap - EXIT
  note "activate-auth ok: $snapshot"
}

require_unit_property() {
  local unit="$1"
  local property="$2"
  local expected="$3"
  local actual
  actual="$(systemctl show "$unit" -p "$property" --value)"
  [ "$actual" = "$expected" ] || die "$unit $property expected '$expected' got '$actual'"
}

verify_loopback_only() {
  local port
  for port in 29000 29100; do
    "${SUDO[@]}" ss -ltnp | grep -E "127\\.0\\.0\\.1:${port}\\b" >/dev/null ||
      die "missing loopback listener on port $port"
  done
  if "${SUDO[@]}" ss -ltn | grep -Eq '(0\.0\.0\.0|\[::\]|\*):(29000|29100)\b'; then
    die "public listener found on auth/cmdcenter ports"
  fi
}

verify_cmdcenter_candidate_namespace() {
  local unit_name="site32-cmdcenter-prereq-$$"
  "${SUDO[@]}" systemd-run --quiet --wait --collect \
    --unit="$unit_name" \
    --property=Type=oneshot \
    --property=User=xrd-cmdcenter \
    --property=Group=xrd-cmdcenter \
    --property=SupplementaryGroups=xrd-auth-readers \
    --property=NoNewPrivileges=yes \
    --property=PrivateTmp=yes \
    --property=ProtectSystem=strict \
    --property=ProtectHome=tmpfs \
    --property="BindReadOnlyPaths=$CMD_ROOT $AUTH_LOG/logins.jsonl:$AUTH_LIB/logins.jsonl" \
    --property="BindPaths=$CMD_LIB/reports:$CMD_ROOT/reports" \
    --property="ReadOnlyPaths=$AUTH_LIB $AUTH_LOG $CMD_ETC" \
    --property="ReadWritePaths=$CMD_LIB $CMD_ROOT/reports" \
    /bin/sh -eu -c \
      'test -x /home/rdk/cmdcenter/.venv/bin/python
       test -r /home/rdk/cmdcenter/app.py
       test -r /var/lib/xrd-auth/users.json
       test -r /var/lib/xrd-auth/logins.jsonl
       test -w /var/lib/xrd-cmdcenter
       test -w /home/rdk/cmdcenter/reports'
}

verify_cmdcenter_deploy_prereqs() {
  local candidate_unit="$CMD_ROOT/systemd/xrd-cmdcenter.service"
  require_file_non_symlink "$candidate_unit"
  grep -q '^User=xrd-cmdcenter$' "$candidate_unit"
  grep -q '^Environment=XRD_CMD_DB_PATH=/var/lib/xrd-cmdcenter/data.db$' "$candidate_unit"
  "${SUDO[@]}" getent passwd xrd-cmdcenter >/dev/null
  "${SUDO[@]}" getent group xrd-cmdcenter >/dev/null
  "${SUDO[@]}" getent group xrd-auth-readers >/dev/null
  "${SUDO[@]}" id -nG xrd-cmdcenter | tr ' ' '\n' | grep -qx xrd-auth-readers
  "${SUDO[@]}" test -s "$CMD_ETC/secrets.env"
  "${SUDO[@]}" test -d "$CMD_LIB"
  "${SUDO[@]}" test -d "$CMD_LIB/reports"
  "${SUDO[@]}" test -s "$AUTH_LIB/users.json"
  "${SUDO[@]}" test -e "$AUTH_LOG/logins.jsonl"
  "${SUDO[@]}" test -e "$AUTH_LIB/logins.jsonl"
  "${SUDO[@]}" test -x "$CMD_ROOT/.venv/bin/python"
  "${SUDO[@]}" -u xrd-cmdcenter test -w "$CMD_LIB"
  "${SUDO[@]}" -u xrd-cmdcenter test -w "$CMD_LIB/reports"
  "${SUDO[@]}" -u xrd-cmdcenter test -r "$AUTH_LIB/users.json"
  "${SUDO[@]}" -u xrd-cmdcenter test -r "$AUTH_LOG/logins.jsonl"
  if "${SUDO[@]}" test -s "$CMD_LIB/data.db"; then
    sqlite_quick_check_ro "$CMD_LIB/data.db"
  elif "${SUDO[@]}" test -s "$CMD_ROOT/data.db"; then
    die "legacy cmdcenter data.db exists but target state database is absent"
  fi
  "${SUDO[@]}" systemd-analyze verify "$candidate_unit" >/dev/null
  verify_cmdcenter_candidate_namespace
}

verify() {
  assert_sudo_noninteractive
  validate_fixed_roots
  local snapshot auth_pid
  snapshot="$(current_snapshot)"
  require_dir_non_symlink "$snapshot"
  systemctl is-active --quiet xrd-auth || die "xrd-auth is not active"
  wait_for_url "$AUTH_HEALTH_URL" 1 1
  require_unit_property xrd-auth User xrd-auth
  require_unit_property xrd-auth Group xrd-auth
  require_unit_property xrd-auth SupplementaryGroups xrd-auth-readers
  require_unit_property xrd-auth ProtectSystem strict
  require_unit_property xrd-auth ProtectHome tmpfs
  require_unit_property xrd-auth PrivateTmp yes
  require_unit_property xrd-auth PrivateDevices yes
  require_unit_property xrd-auth NoNewPrivileges yes
  auth_pid="$(systemctl show -p MainPID --value xrd-auth)"
  [ -n "$auth_pid" ] && [ "$auth_pid" != "0" ] || die "xrd-auth has no MainPID"
  ps -o pid,user,group,args -p "$auth_pid"
  verify_loopback_only
  "${SUDO[@]}" -u xrd-cmdcenter test ! -r "$AUTH_ETC/secret.key"
  "${SUDO[@]}" -u xrd-auth test ! -r "$CMD_ETC/secrets.env"
  "${SUDO[@]}" -u xrd-auth test -r "$AUTH_LIB/users.json"
  "${SUDO[@]}" -u xrd-auth test -w "$AUTH_LOG/logins.jsonl"
  "${SUDO[@]}" -u xrd-auth test ! -w "$AUTH_ROOT/app.py"
  "${SUDO[@]}" nsenter -t "$auth_pid" -m -- findmnt "$AUTH_ROOT" >/dev/null
  "${SUDO[@]}" nsenter -t "$auth_pid" -m -- findmnt "$AUTH_ROOT/logins.jsonl" >/dev/null
  "${SUDO[@]}" /usr/sbin/logrotate --debug "$AUTH_LOGROTATE" >/dev/null
  verify_cmdcenter_deploy_prereqs
  printf '%s\n' "$snapshot" | "${SUDO[@]}" tee "$SNAPSHOT_ROOT/LAST_AUTH_SUCCESSFUL" >/dev/null
  note "verify ok: auth isolation and cmdcenter deploy prerequisites pass"
}

copy_latest_auth_state_back_to_legacy() {
  local snapshot="$1"
  local stamp audit_file old_dir
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  require_file_non_symlink "$AUTH_LIB/users.json"
  safe_install_file ubuntu ubuntu 0600 "$AUTH_LIB/users.json" "$AUTH_ROOT/users.json"
  require_file_non_symlink "$AUTH_ETC/secret.key"
  safe_install_file ubuntu ubuntu 0600 "$AUTH_ETC/secret.key" "$AUTH_ROOT/secret.key"

  old_dir="$snapshot/auth-audit-legacy-before-rollback-$stamp"
  safe_install_dir root root 0700 "$old_dir"
  while IFS= read -r -d '' audit_file; do
    local base
    base="$(basename "$audit_file")"
    require_name_safe "$base"
    safe_move_known "$audit_file" "$old_dir/$base"
  done < <("${SUDO[@]}" find "$AUTH_ROOT" -maxdepth 1 -type f -name 'logins.jsonl*' -print0)

  while IFS= read -r -d '' audit_file; do
    local base
    base="$(basename "$audit_file")"
    require_name_safe "$base"
    safe_install_file ubuntu ubuntu 0600 "$audit_file" "$AUTH_ROOT/$base"
  done < <("${SUDO[@]}" find "$AUTH_LOG" -maxdepth 1 -type f -name 'logins.jsonl*' -print0)

  write_audit_hashes "$AUTH_LOG" "$snapshot/auth-audit-sha256.rollback-source"
  write_audit_hashes "$AUTH_ROOT" "$snapshot/auth-audit-sha256.rollback-target"
  "${SUDO[@]}" cmp "$snapshot/auth-audit-sha256.rollback-source" "$snapshot/auth-audit-sha256.rollback-target"
}

restore_auth_rotation_from_snapshot() {
  local snapshot="$1"
  local suffix="$2"
  local item active saved save_to
  for item in \
    "xrd-auth-audit-rotate.service:$AUTH_ROTATE_UNIT" \
    "xrd-auth-audit-rotate-alert@.service:$AUTH_ROTATE_ALERT_UNIT" \
    "xrd-auth-audit-rotate.timer:$AUTH_ROTATE_TIMER"; do
    active="${item#*:}"
    saved="$snapshot/${item%%:*}.before"
    save_to="$snapshot/${item%%:*}.$suffix"
    if "${SUDO[@]}" test -e "$active"; then
      safe_move_known "$active" "$save_to"
    fi
    if "${SUDO[@]}" test -f "$saved"; then
      safe_install_file root root 0644 "$saved" "$active"
    fi
  done
  if "${SUDO[@]}" test -e "$AUTH_LOGROTATE"; then
    safe_move_known "$AUTH_LOGROTATE" "$snapshot/xrd-auth.logrotate.$suffix"
  fi
  if "${SUDO[@]}" test -f "$snapshot/xrd-auth.logrotate.before"; then
    safe_install_file root root 0644 "$snapshot/xrd-auth.logrotate.before" "$AUTH_LOGROTATE"
  fi
}

rollback_auth() {
  assert_sudo_noninteractive
  validate_fixed_roots
  local snapshot
  snapshot="$(current_snapshot)"
  require_file_non_symlink "$snapshot/xrd-auth.service.before"
  require_file_non_symlink "$snapshot/private-state.tar.sha256"
  "${SUDO[@]}" sha256sum -c "$snapshot/private-state.tar.sha256" >/dev/null
  "${SUDO[@]}" systemctl disable --now xrd-auth-audit-rotate.timer >/dev/null 2>&1 || true
  "${SUDO[@]}" systemctl stop xrd-auth
  copy_latest_auth_state_back_to_legacy "$snapshot"
  if "${SUDO[@]}" test -f "$AUTH_UNIT"; then
    safe_install_file root root 0600 "$AUTH_UNIT" "$snapshot/xrd-auth.service.after-migration"
  fi
  restore_auth_unit_from_snapshot "$snapshot" "after-migration"
  restore_auth_rotation_from_snapshot "$snapshot" "after-migration"
  "${SUDO[@]}" systemctl daemon-reload
  "${SUDO[@]}" systemctl reset-failed xrd-auth || true
  "${SUDO[@]}" systemctl start xrd-auth
  wait_for_url "$AUTH_HEALTH_URL" 20 1
  if "${SUDO[@]}" grep -qx enabled "$snapshot/xrd-auth-audit-rotate.timer.enabled.before" 2>/dev/null; then
    "${SUDO[@]}" systemctl enable --now xrd-auth-audit-rotate.timer >/dev/null
  fi
  printf '%s\n' "$snapshot" | "${SUDO[@]}" tee "$SNAPSHOT_ROOT/ROLLED_BACK_AUTH" >/dev/null
  note "rollback-auth ok: latest users/audit were copied back before restoring the old auth unit"
}

main() {
  local cmd="${1:-preflight}"
  if [ "$#" -gt 0 ]; then
    shift
  fi
  parse_common_args "$@"
  case "$cmd" in
    preflight) preflight ;;
    prepare) prepare ;;
    activate-auth) activate_auth ;;
    verify) verify ;;
    rollback-auth) rollback_auth ;;
    -h|--help|help) usage ;;
    *) die "unknown command: $cmd" ;;
  esac
}

main "$@"
