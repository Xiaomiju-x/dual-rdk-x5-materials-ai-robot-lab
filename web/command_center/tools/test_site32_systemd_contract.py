#!/usr/bin/env python3
"""Static fail-closed contract for the Site32 origin service sandbox."""

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUTH_ROOT = ROOT.parent / "auth"
UNIT = ROOT / "systemd" / "xrd-cmdcenter.service"
GATE = ROOT / "tools" / "site31_gate_audit.py"
DEPLOY = ROOT / "tools" / "deploy_staged.sh"
ROLLBACK = ROOT / "tools" / "rollback.sh"
RUNBOOK = ROOT / "docs" / "site32_systemd_service_isolation_migration.md"


class Site32SystemdContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.unit = UNIT.read_text(encoding="utf-8-sig")
        cls.gate = GATE.read_text(encoding="utf-8-sig")
        cls.deploy = DEPLOY.read_text(encoding="utf-8-sig")
        cls.rollback = ROLLBACK.read_text(encoding="utf-8-sig")
        cls.runbook = RUNBOOK.read_text(encoding="utf-8-sig")

    def test_dedicated_identity_and_loopback_only(self):
        for token in (
            "User=xrd-cmdcenter",
            "Group=xrd-cmdcenter",
            "SupplementaryGroups=xrd-auth-readers",
            "--bind 127.0.0.1:29100",
        ):
            self.assertIn(token, self.unit)
        self.assertNotIn("--bind 0.0.0.0:29100", self.unit)

    def test_state_and_secret_paths_are_separated_from_source(self):
        for token in (
            "XRD_CMD_DB_PATH=/var/lib/xrd-cmdcenter/data.db",
            "EnvironmentFile=/etc/xrd-cmdcenter/secrets.env",
            "BindReadOnlyPaths=/home/rdk/cmdcenter",
            "ReadWritePaths=/var/lib/xrd-cmdcenter /home/rdk/cmdcenter/reports",
            "InaccessiblePaths=/home/rdk/cmdcenter/secrets.env",
        ):
            self.assertIn(token, self.unit)

    def test_strict_sandbox_has_no_process_capabilities(self):
        for token in (
            "NoNewPrivileges=true",
            "PrivateTmp=true",
            "PrivateDevices=true",
            "PrivateIPC=true",
            "ProtectSystem=strict",
            "ProtectHome=tmpfs",
            "RestrictSUIDSGID=true",
            "RestrictNamespaces=true",
            "CapabilityBoundingSet=\n",
            "AmbientCapabilities=\n",
        ):
            self.assertIn(token, self.unit)

    def test_gate_requires_site32_strict_profile(self):
        for token in (
            'ASSET_VER.startswith("site32-")',
            '"User=xrd-cmdcenter"',
            '"ProtectSystem=strict"',
            '"ProtectHome=tmpfs"',
            '"CapabilityBoundingSet="',
            '"XRD_CMD_DB_PATH=/var/lib/xrd-cmdcenter/data.db"',
        ):
            self.assertIn(token, self.gate)

    def test_staged_deploy_fails_closed_before_installing_strict_unit(self):
        for token in (
            "require_site32_runtime_prereqs",
            "getent passwd xrd-cmdcenter",
            "sudo -n test -s /etc/xrd-cmdcenter/secrets.env",
            "sudo -n -u xrd-cmdcenter test -w /var/lib/xrd-cmdcenter",
            "sudo -n -u xrd-cmdcenter test -r /var/lib/xrd-auth/users.json",
            "verify_site32_runtime_namespace",
            "systemd-run --quiet --wait --collect",
            "ProtectHome=tmpfs",
            "BindReadOnlyPaths=$candidate_root:$CD",
            "from cmdcenter import RuntimeController, register_site32",
            "normalize_candidate_modes",
            "candidate contains private runtime state",
            "candidate symlink is not allowed",
            "--chmod=D755,F644",
            "systemd-analyze verify",
            "PRAGMA quick_check",
            "site32_state_bridge.py",
        ):
            self.assertIn(token, self.deploy)
        prereq = self.deploy.index('require_site32_runtime_prereqs "$STAGE_REAL/systemd/xrd-cmdcenter.service" "$STAGE_REAL"')
        install = self.deploy.index('install -m 0644 "$CD/systemd/xrd-cmdcenter.service"')
        self.assertLess(prereq, install)
        self.assertNotIn("reload-or-restart xrd-cmdcenter", self.deploy)
        self.assertGreaterEqual(self.deploy.count("sudo -n systemctl restart xrd-cmdcenter"), 2)

    def test_rollback_quiesces_service_and_bridges_state_before_release_switch(self):
        for token in (
            'STATE_BRIDGE="$SCRIPT_DIR/site32_state_bridge.py"',
            "prepare_target_state_db",
            "/var/lib/xrd-cmdcenter/data.db",
            'target_db="$CD/data.db"',
            "--owner \"$owner\" --group \"$group\" --mode 0600",
            "manifest_requires_asset",
            "required_critical_assets",
            "install_target_unit",
            "xrd-cmdcenter.service.active",
            '(data.get("summary") or {}).get("release")',
        ):
            self.assertIn(token, self.rollback)
        stop = self.rollback.index("sudo -n systemctl stop xrd-cmdcenter")
        bridge = self.rollback.index('prepare_target_state_db "$CURRENT_RELEASE" "$TARGET_RELEASE"')
        promote = self.rollback.index('cp -a "$PREV/app.py" "$CD/app.py"')
        self.assertLess(stop, bridge)
        self.assertLess(bridge, promote)

    def test_release_ledgers_follow_the_active_database_owner(self):
        self.assertNotIn('~/cmdcenter/data.db', self.deploy)
        self.assertNotIn('~/cmdcenter/data.db', self.rollback)
        self.assertIn('sudo -n -u xrd-cmdcenter', self.deploy)
        self.assertIn('/usr/bin/python3 - "$VER"', self.deploy)
        self.assertIn('/var/lib/xrd-cmdcenter/data.db', self.deploy)
        self.assertIn('LEDGER_RUN=(sudo -n -u xrd-cmdcenter)', self.rollback)
        self.assertIn('LEDGER_PY=/usr/bin/python3', self.rollback)

    def test_auth_rotation_refreshes_both_bind_mount_consumers(self):
        rotate = (AUTH_ROOT / "systemd" / "xrd-auth.logrotate").read_text(encoding="utf-8-sig")
        rotate_unit = (AUTH_ROOT / "systemd" / "xrd-auth-audit-rotate.service").read_text(encoding="utf-8-sig")
        alert_unit = AUTH_ROOT / "systemd" / "xrd-auth-audit-rotate-alert@.service"
        self.assertIn("try-restart xrd-auth.service", rotate)
        self.assertIn("try-restart xrd-cmdcenter.service", rotate)
        self.assertIn("OnFailure=xrd-auth-audit-rotate-alert@%n.service", rotate_unit)
        self.assertTrue(alert_unit.is_file())
        self.assertIn("auth.crit", alert_unit.read_text(encoding="utf-8-sig"))

    def test_runbook_preserves_every_audit_rotation_segment(self):
        for token in (
            "-name 'logins.jsonl*'",
            '"$AUTH_ROOT"/logins.jsonl*',
            "/var/log/xrd-auth/logins.jsonl*",
            "auth-audit-sha256.before",
            "auth-audit-sha256.rollback-source",
            "auth-audit-sha256.rollback-target",
            "xrd-auth-audit-rotate-alert@.service",
        ):
            self.assertIn(token, self.runbook)


if __name__ == "__main__":
    unittest.main(verbosity=2)
