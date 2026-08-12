#!/usr/bin/env python3
"""Contract tests for the Site32 service-isolation migration helper."""

from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "site32_service_isolation.sh"


class Site32ServiceIsolationHelperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.script = SCRIPT.read_text(encoding="utf-8")

    def test_declares_fail_closed_commands_and_default_preflight(self) -> None:
        for token in (
            "preflight)",
            "prepare)",
            "activate-auth)",
            "verify)",
            "rollback-auth)",
            'local cmd="${1:-preflight}"',
            "set -euo pipefail",
        ):
            self.assertIn(token, self.script)

    def test_every_privileged_operation_uses_sudo_n_wrapper(self) -> None:
        self.assertIn("SUDO=(sudo -n)", self.script)
        self.assertIn('"${SUDO[@]}" /usr/bin/true', self._function_body("assert_sudo_noninteractive"))
        self.assertNotIn('"${SUDO[@]}" -v', self.script)
        self.assertNotRegex(self.script, r"(?<!\()(?<!-)\\bsudo\\s+(?!-n\\b)")
        for token in (
            '"${SUDO[@]}" systemd-analyze verify',
            '"${SUDO[@]}" systemctl daemon-reload',
            '"${SUDO[@]}" install',
            '"${SUDO[@]}" -u xrd-cmdcenter',
            '"${SUDO[@]}" -u xrd-auth',
        ):
            self.assertIn(token, self.script)

    def test_preflight_is_read_only_and_active_paths_are_strict(self) -> None:
        preflight = self._function_body("preflight")
        for forbidden in ('"${SUDO[@]}" install', '"${SUDO[@]}" mv', '"${SUDO[@]}" cp', '"${SUDO[@]}" tar', "useradd", "groupadd", "systemctl restart"):
            self.assertNotIn(forbidden, preflight)
        for token in (
            "validate_required_sources",
            "validate_active_units",
            "FragmentPath",
            "wait_for_url",
        ):
            self.assertIn(token, preflight)
        self.assertIn("refuse_symlink_if_exists", self._function_body("require_existing_non_symlink"))
        self.assertIn("refuse_symlink_if_exists", self._function_body("require_optional_non_symlink"))

    def test_prepare_snapshots_all_state_without_stopping_or_installing_main_units(self) -> None:
        prepare = self._function_body("prepare")
        create_snapshot = self._function_body("create_snapshot")
        for token in (
            "xrd-auth.service.before",
            "xrd-cmdcenter.service.before",
            "xrd-auth.service.d.before",
            "xrd-cmdcenter.service.d.before",
            "private-state.tar",
            "users.json",
            "secret.key",
            "secrets.env",
            "data.db",
            "reports",
            "append_login_globs \"$tarball\" \"$AUTH_ROOT\"",
            "append_login_globs \"$tarball\" \"$AUTH_LOG\"",
            "private-state.tar.sha256",
        ):
            self.assertIn(token, create_snapshot)
        for token in (
            "create_identities_and_dirs",
            "copy_auth_state_to_targets",
            "copy_cmdcenter_state_to_targets",
            "install_auth_rotation_assets",
            "verify_candidate_units",
        ):
            self.assertIn(token, prepare)
        self.assertIn("sqlite_backup_checked", self._function_body("copy_cmdcenter_state_to_targets"))
        verify_candidates = self._function_body("verify_candidate_units")
        self.assertIn('require_file_non_symlink "$AUTH_LOGROTATE"', verify_candidates)
        self.assertIn("installed auth logrotate config must be mode 0644", verify_candidates)
        self.assertNotIn('$AUTH_ROOT/systemd/xrd-auth.logrotate\" >/dev/null', verify_candidates)
        self.assertIn("pending migration already exists", prepare)
        self.assertNotIn("systemctl stop", prepare)
        self.assertNotIn('xrd-auth.service" "$AUTH_UNIT"', prepare)
        self.assertNotIn('xrd-cmdcenter.service" "$CMD_UNIT"', prepare)

    def test_activate_auth_is_transactional_and_does_not_switch_cmdcenter(self) -> None:
        body = self._function_body("activate_auth")
        rollback = self._function_body("rollback_auth_activation_on_exit")
        self.assertIn("safe_install_file root root 0644 \"$AUTH_ROOT/systemd/xrd-auth.service\" \"$AUTH_UNIT\"", body)
        self.assertIn("wait_for_url \"$AUTH_HEALTH_URL\" 20 1", body)
        self.assertIn("trap rollback_auth_activation_on_exit EXIT", body)
        self.assertIn("AUTH_ACTIVATION_COMMITTED=1", body)
        self.assertIn("trap - EXIT", body)
        self.assertIn("restore_auth_unit_from_snapshot", rollback)
        self.assertIn("systemctl restart xrd-auth", rollback)
        self.assertIn("wait_for_url \"$AUTH_HEALTH_URL\" 20 1", rollback)
        self.assertIn("xrd-auth.service.d.disabled-for-auth-activate", body)
        self.assertNotIn("xrd-cmdcenter.service", body)
        self.assertNotIn("systemctl restart xrd-cmdcenter", body)

    def test_verify_covers_auth_isolation_and_cmdcenter_deploy_prereqs(self) -> None:
        body = self._function_body("verify")
        prereq = self._function_body("verify_cmdcenter_deploy_prereqs")
        for token in (
            "require_unit_property xrd-auth User xrd-auth",
            "require_unit_property xrd-auth ProtectSystem strict",
            "verify_loopback_only",
            "test ! -r \"$AUTH_ETC/secret.key\"",
            "nsenter -t \"$auth_pid\" -m -- findmnt \"$AUTH_ROOT/logins.jsonl\"",
            "verify_cmdcenter_deploy_prereqs",
        ):
            self.assertIn(token, body)
        for token in (
            "getent passwd xrd-cmdcenter",
            "id -nG xrd-cmdcenter",
            "test -s \"$CMD_ETC/secrets.env\"",
            "test -s \"$AUTH_LIB/users.json\"",
            "test -e \"$AUTH_LOG/logins.jsonl\"",
            "sqlite_quick_check_ro",
            "systemd-analyze verify \"$candidate_unit\"",
            "verify_cmdcenter_candidate_namespace",
        ):
            self.assertIn(token, prereq)
        self.assertIn('"${SUDO[@]}" -u xrd-cmdcenter /usr/bin/python3', self._function_body("sqlite_quick_check_ro"))
        namespace = self._function_body("verify_cmdcenter_candidate_namespace")
        for token in (
            "systemd-run --quiet --wait --collect",
            "ProtectHome=tmpfs",
            "BindReadOnlyPaths=$CMD_ROOT",
            "test -x /home/rdk/cmdcenter/.venv/bin/python",
            "test -w /home/rdk/cmdcenter/reports",
        ):
            self.assertIn(token, namespace)

    def test_rollback_auth_preserves_latest_users_and_audit_before_restoring_old_unit(self) -> None:
        rollback = self._function_body("rollback_auth")
        copyback = self._function_body("copy_latest_auth_state_back_to_legacy")
        restore_index = rollback.index("restore_auth_unit_from_snapshot")
        copy_index = rollback.index("copy_latest_auth_state_back_to_legacy")
        self.assertLess(copy_index, restore_index)
        for token in (
            "safe_install_file ubuntu ubuntu 0600 \"$AUTH_LIB/users.json\" \"$AUTH_ROOT/users.json\"",
            "safe_install_file ubuntu ubuntu 0600 \"$AUTH_ETC/secret.key\" \"$AUTH_ROOT/secret.key\"",
            "find \"$AUTH_LOG\" -maxdepth 1 -type f -name 'logins.jsonl*'",
            "auth-audit-sha256.rollback-source",
            "auth-audit-sha256.rollback-target",
            "cmp \"$snapshot/auth-audit-sha256.rollback-source\"",
        ):
            self.assertIn(token, copyback)

    def test_no_unknown_destructive_or_network_perimeter_changes(self) -> None:
        forbidden = (
            "rm -rf",
            "git checkout",
            "git reset",
            "ufw ",
            "caddy ",
            "netsh ",
            "iptables",
            "nft ",
            "ip route",
            "arp ",
        )
        for token in forbidden:
            self.assertNotIn(token, self.script)
        self.assertIn("The helper never changes Caddy, UFW, Wi-Fi, VPN, routes, ARP, or SSH config.", self.script)

    def test_temp_fixture_contract_contains_every_state_family_and_rejects_symlink_pattern(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            auth = root / "home" / "ubuntu" / "auth"
            cmd = root / "home" / "ubuntu" / "cmdcenter"
            target_auth = root / "var" / "log" / "xrd-auth"
            reports = cmd / "reports"
            for directory in (auth, cmd, target_auth, reports):
                directory.mkdir(parents=True)
            for path in (
                auth / "users.json",
                auth / "secret.key",
                auth / "logins.jsonl",
                auth / "logins.jsonl.1",
                target_auth / "logins.jsonl",
                target_auth / "logins.jsonl.1",
                cmd / "secrets.env",
                cmd / "data.db",
                reports / "summary.json",
            ):
                path.write_text("fixture\n", encoding="utf-8")
            families = {
                "users": list(root.rglob("users.json")),
                "secret": list(root.rglob("secret.key")),
                "secrets": list(root.rglob("secrets.env")),
                "db": list(root.rglob("data.db")),
                "reports": list(root.rglob("reports")),
                "audit": list(root.rglob("logins.jsonl*")),
            }
            self.assertEqual({key for key, value in families.items() if value}, set(families))
            self.assertIn("refuse_symlink_if_exists", self.script)
            self.assertIn("require_optional_non_symlink", self.script)
            self.assertIn("require_existing_non_symlink", self.script)
            self.assertIn("refusing report tree containing symlinks", self._function_body("safe_copy_tree"))
            self.assertIn('resolve_existing "$AUTH_ROOT"', self._function_body("validate_fixed_roots"))

    def _function_body(self, name: str) -> str:
        pattern = re.compile(rf"^{re.escape(name)}\(\) \{{\n(?P<body>.*?)(?=^\}}\n)", re.M | re.S)
        match = pattern.search(self.script)
        self.assertIsNotNone(match, name)
        return match.group("body")


if __name__ == "__main__":
    unittest.main(verbosity=2)
