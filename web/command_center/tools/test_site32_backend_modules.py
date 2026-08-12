#!/usr/bin/env python3
"""Contract tests for Site32 backend config and public DTO modules."""

from __future__ import annotations

import importlib
import sqlite3
import subprocess
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock


CMD_ROOT = Path(__file__).resolve().parents[1]
if str(CMD_ROOT) not in sys.path:
    sys.path.insert(0, str(CMD_ROOT))


class Site32BackendModuleTests(unittest.TestCase):
    def test_config_and_public_dto_import_without_runtime_side_effects(self) -> None:
        observations = {"sqlite": [], "threads": [], "subprocess": []}

        def blocked_connect(*args, **kwargs):
            observations["sqlite"].append(args[0] if args else None)
            raise AssertionError("sqlite3.connect should not run while importing backend modules")

        def blocked_thread_start(thread, *args, **kwargs):
            observations["threads"].append(thread.name)
            raise AssertionError("Thread.start should not run while importing backend modules")

        def blocked_subprocess_run(*args, **kwargs):
            observations["subprocess"].append(args[0] if args else None)
            raise AssertionError("subprocess.run should not run while importing backend modules")

        before = threading.active_count()
        with mock.patch.object(sqlite3, "connect", side_effect=blocked_connect), \
                mock.patch.object(threading.Thread, "start", new=blocked_thread_start), \
                mock.patch.object(subprocess, "run", side_effect=blocked_subprocess_run):
            importlib.reload(importlib.import_module("cmdcenter.config"))
            importlib.reload(importlib.import_module("cmdcenter.public_dto"))

        self.assertEqual(observations, {"sqlite": [], "threads": [], "subprocess": []})
        self.assertEqual(threading.active_count(), before)

    def test_config_parser_is_bounded_and_not_cached_by_environment(self) -> None:
        from cmdcenter import config

        first = config.load_config({
            "XRD_CMD_TEST_MODE": "yes",
            "XRD_WEBHOOK_HOST_ALLOWLIST": " Example.COM. ,hooks.test,, ",
            "XRD_SSE_MAX": "999",
            "XRD_SSE_LIFETIME_S": "1",
            "XRD_AUTH_DIR": "~/custom-auth",
            "DEEPSEEK_API_KEY": "  key-fixture  ",
            "DEEPSEEK_MODEL": "",
            "XRD_CMD_RUNTIME": "on",
            "PORT": "70000",
        })
        second = config.load_config({
            "XRD_CMD_TEST_MODE": "no",
            "XRD_SSE_MAX": "bad-int",
            "XRD_SSE_LIFETIME_S": "400",
            "PORT": "not-a-port",
        })

        self.assertTrue(first.cmd_test_mode)
        self.assertTrue(first.runtime_enabled)
        self.assertEqual(first.webhook_extra_hosts, frozenset({"example.com", "hooks.test"}))
        self.assertEqual(first.sse_max, 8)
        self.assertEqual(first.sse_lifetime_s, 15)
        self.assertEqual(first.llm_key, "key-fixture")
        self.assertEqual(first.llm_model, "deepseek-chat")
        self.assertEqual(first.port, 65535)

        self.assertFalse(second.cmd_test_mode)
        self.assertEqual(second.webhook_extra_hosts, frozenset())
        self.assertEqual(second.sse_max, 2)
        self.assertEqual(second.sse_lifetime_s, 300)
        self.assertEqual(second.port, 29100)

    def test_public_dto_redacts_assets_and_runbooks(self) -> None:
        from cmdcenter import public_dto

        group = {
            "key": "lab",
            "host": "gateway 192.0.2.103:29100",
            "ip": "198.51.100.103",
            "children": [
                {
                    "id": "arm-frpc-/dev/F407",
                    "name": "frpc GPIO PWM node",
                    "kind": "GPIO bridge",
                    "spec": "TIM3_CH1 on /dev/F407, token=secret-value",
                    "status": "open 18888/tcp",
                    "internal_note": "must be dropped",
                }
            ],
            "private": "must be dropped",
        }

        sanitized = public_dto.public_asset_group(group)
        rendered = str(sanitized)
        self.assertEqual(sanitized["ip"], "public-safe redacted")
        self.assertNotIn("192.0.2.103", rendered)
        self.assertNotIn("198.51.100.103", rendered)
        self.assertNotIn("/dev/F407", rendered)
        self.assertNotIn("GPIO", rendered)
        self.assertNotIn("secret-value", rendered)
        self.assertNotIn("18888/tcp", rendered)
        self.assertNotIn("internal_note", rendered)
        self.assertEqual(public_dto.public_redaction_scan(sanitized)["scan_pass"], True)
        self.assertEqual(
            public_dto.public_runbook_text("ssh host && systemctl status xrd-cmdcenter"),
            "operator_runbook_required",
        )

    def test_public_status_envelope_contract_is_reusable(self) -> None:
        from cmdcenter import public_dto

        fresh = public_dto.status_envelope(
            "live", "sampler", checked_at=100.0, ttl_s=90, confidence="high",
            release="site32-test", now=150.0,
        )
        stale = public_dto.status_envelope(
            "live", "sampler", checked_at=100.0, ttl_s=10,
            release="site32-test", now=150.0,
        )

        self.assertEqual(fresh["state"], "live")
        self.assertEqual(fresh["freshness"], "fresh")
        self.assertEqual(fresh["confidence"], "verified")
        self.assertEqual(fresh["checked_at"], "1970-01-01T00:01:40Z")
        self.assertEqual(fresh["release"], "site32-test")
        self.assertEqual(stale["state"], "stale")
        self.assertEqual(stale["freshness"], "stale")
        self.assertEqual(public_dto.status_from_serving("mirror")["source"], "mirror")
        self.assertEqual(public_dto.serving_source("real", age_s=301), "stale")
        self.assertEqual(public_dto.route_service("/api/materials/foo"), "research")
        self.assertEqual(public_dto.public_severity(503, test_mode=True), "critical")
        self.assertEqual(public_dto.public_severity(404, test_mode=True), "warning")
        self.assertEqual(public_dto.public_severity(200, test_mode=True), "info")

    def test_app_compatibility_names_still_point_to_extracted_contracts(self) -> None:
        import app
        from cmdcenter import config, public_dto

        self.assertEqual(app.ASSET_VER, config.ASSET_VER)
        self.assertEqual(app.RELEASED_AT, config.RELEASED_AT)
        self.assertEqual(app._public_safe_text("token=secret-value"), public_dto.public_safe_text("token=secret-value"))
        self.assertEqual(app._mask_ip("192.0.2.103"), public_dto.mask_ip("192.0.2.103"))
        self.assertEqual(app._status_meta("offline"), public_dto.status_meta("offline"))
        self.assertEqual(app._route_service("/api/predictions/trace"), public_dto.route_service("/api/predictions/trace"))
        self.assertEqual(app._status_envelope("live", "test", 100, ttl_s=1)["release"], app.ASSET_VER)


if __name__ == "__main__":
    unittest.main(verbosity=2)
