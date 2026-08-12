#!/usr/bin/env python3
"""Local regression tests for modern Site31/Site32 release integrity tools."""
from __future__ import annotations

import json
import re
import tempfile
import time
import unittest
from pathlib import Path

import site31_asset_manifest as manifest
import site32_style_audit as style_audit
from site31_gate_audit import (
    REQUIRED_BROWSER_CHECKS,
    REQUIRED_ORIGIN_CHECKS,
    _release_not_before,
    _validate_browser_evidence,
    _validate_origin_evidence,
    _validate_release_bindings,
    _validate_sw_cache_boundary,
)


class ReleaseToolsTest(unittest.TestCase):
    RELEASE = "site32-global-commercial-v1.4-20260711"

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "static" / "quality").mkdir(parents=True)
        (self.root / "tools").mkdir()
        (self.root / "cmdcenter").mkdir()
        (self.root / "systemd").mkdir()
        (self.root / "app.py").write_text(
            f'ASSET_VER = "{self.RELEASE}"\nRELEASED_AT = "2026-07-10T00:00:00Z"\n',
            encoding="utf-8-sig",
        )
        (self.root / "assets.json").write_text("{}\n", encoding="utf-8")
        (self.root / "requirements-production.txt").write_text("gunicorn==23.0.0\n", encoding="utf-8")
        (self.root / "systemd" / "xrd-cmdcenter.service").write_text("[Service]\n", encoding="utf-8")
        for name in (
            "__init__.py", "access.py", "config.py", "public_dto.py", "release.py",
            "route_contract.py", "runtime.py", "site32_blueprint.py", "storage.py",
        ):
            (self.root / "cmdcenter" / name).write_text("# fixture\n", encoding="utf-8")
        (self.root / "static" / "index.html").write_text("<!doctype html>\n", encoding="utf-8")
        (self.root / "static" / "app.js").write_text("'use strict';\n", encoding="utf-8")
        (self.root / "static" / "style.css").write_text("body{}\n", encoding="utf-8")
        (self.root / "static" / "i18n.js").write_text("'use strict';\n", encoding="utf-8")
        (self.root / "static" / "twin.js").write_text("'use strict';\n", encoding="utf-8")
        (self.root / "static" / "sw.js").write_text("'use strict';\n", encoding="utf-8")
        for name in ("r4.css", "r4.js", "r4-performance.js", "r4-accessibility.js"):
            (self.root / "static" / name).write_text("body{}\n" if name.endswith(".css") else "'use strict';\n", encoding="utf-8")
        (self.root / "static" / "src" / "site32").mkdir(parents=True)
        for name in ("site32.css", "site32.js"):
            (self.root / "static" / name).write_text("body{}\n" if name.endswith(".css") else "'use strict';\n", encoding="utf-8")
        for name in ("release.js", "runtime.js", "state.js"):
            (self.root / "static" / "src" / "site32" / name).write_text("export {};\n", encoding="utf-8")
        for name in (
            "deploy.sh", "deploy_staged.sh", "rollback.sh", "site31_asset_manifest.py",
            "site31_gate_audit.py", "site31_smoke.py", "site32_service_isolation.sh",
            "site32_state_bridge.py", "site32_style_audit.py",
            "site32_environment_matrix.py", "site32_round1_baseline.py",
            "test_site32_deploy_environment_gate.py", "test_site32_environment_matrix.py",
            "test_site32_gate_evidence_contract.py", "test_site32_public_research_contract.py",
            "test_site32_requirements_contract.py", "test_site32_rollback_contract.py",
            "test_site32_round1_baseline.py", "test_site32_runtime_manifest_integrity.py",
            "test_site32_sw_precache_contract.py",
        ):
            (self.root / "tools" / name).write_text("# fixture\n", encoding="utf-8")
        (self.root / "tools" / "release.sh").write_text("#!/bin/sh\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_evidence_and_manifest(self) -> dict:
        preliminary = manifest.build_manifest(self.root)
        evidence = {
            "schema_version": "site31.browser_evidence.v1",
            "release": self.RELEASE,
            "manifest_digest": preliminary["manifest_digest"],
            "completed_at": "2026-07-10T10:00:00+00:00",
            "base_url": "https://example.test/",
            "checks": [{"key": key, "state": "pass"} for key in REQUIRED_BROWSER_CHECKS],
        }
        (self.root / "static" / "quality" / "site31_browser_evidence.json").write_text(
            json.dumps(evidence), encoding="utf-8"
        )
        return manifest.write_manifest(self.root)

    def test_manifest_has_hash_size_mime_and_detects_tamper(self) -> None:
        payload = self._write_evidence_and_manifest()
        verified = manifest.verify_manifest(self.root)
        self.assertEqual(verified["manifest_digest"], payload["manifest_digest"])
        by_path = {entry["path"]: entry for entry in verified["files"]}
        self.assertEqual(by_path["static/app.js"]["mime"], "text/javascript")
        self.assertEqual(len(by_path["static/app.js"]["sha256"]), 64)
        self.assertEqual(by_path["static/app.js"]["size"], (self.root / "static" / "app.js").stat().st_size)
        self.assertFalse(by_path["static/quality/site31_browser_evidence.json"]["digest_bound"])
        required = set(verified["required_critical_assets"])
        self.assertNotIn("tools/deploy.sh", required)
        self.assertTrue({
            "static/r4.css", "static/r4.js", "static/r4-performance.js", "static/r4-accessibility.js",
            "static/site32.css", "static/site32.js", "static/src/site32/release.js",
            "static/src/site32/runtime.js", "static/src/site32/state.js",
            "requirements-production.txt",
            "systemd/xrd-cmdcenter.service",
            "cmdcenter/config.py", "cmdcenter/public_dto.py",
            "cmdcenter/route_contract.py", "cmdcenter/storage.py",
            "tools/deploy_staged.sh", "tools/rollback.sh",
            "tools/site31_asset_manifest.py", "tools/site31_gate_audit.py",
            "tools/site31_smoke.py", "tools/site32_service_isolation.sh",
            "tools/site32_state_bridge.py",
            "tools/site32_style_audit.py",
        }.issubset(required))
        self.assertEqual(len(verified["critical_assets_sha256"]), 64)
        summary = manifest._summary(verified, root=self.root, verified=True)
        self.assertEqual(summary["files"], verified["files"])
        self.assertEqual(summary["critical_assets"], verified["critical_assets"])
        (self.root / "static" / "app.js").write_text("tampered\n", encoding="utf-8")
        with self.assertRaises(manifest.ManifestError):
            manifest.verify_manifest(self.root)

    def test_manifest_reads_config_derived_release_without_importing_app(self) -> None:
        (self.root / "app.py").write_text(
            "from cmdcenter.config import load_config\n"
            "_CMD_CONFIG = load_config()\n"
            "ASSET_VER = _CMD_CONFIG.asset_version\n",
            encoding="utf-8-sig",
        )
        (self.root / "cmdcenter" / "config.py").write_text(
            f'ASSET_VER = "{self.RELEASE}"\n', encoding="utf-8"
        )
        payload = manifest.build_manifest(self.root)
        self.assertEqual(payload["release"], self.RELEASE)

    def test_site32_v15_requires_research_search_contract(self) -> None:
        release = "site32-global-commercial-v1.5-20260712"
        (self.root / "app.py").write_text(
            f'ASSET_VER = "{release}"\nRELEASED_AT = "2026-07-12T00:00:00Z"\n',
            encoding="utf-8-sig",
        )
        with self.assertRaises(manifest.ManifestError):
            manifest.build_manifest(self.root)
        (self.root / "cmdcenter" / "research_search.py").write_text("# fixture\n", encoding="utf-8")
        (self.root / "tools" / "test_site32_research_search.py").write_text("# fixture\n", encoding="utf-8")
        payload = manifest.build_manifest(self.root)
        self.assertEqual(payload["release"], release)
        for relative in (
            "cmdcenter/research_search.py",
            "tools/deploy.sh",
            "tools/site32_environment_matrix.py",
            "tools/site32_round1_baseline.py",
            "tools/test_site32_research_search.py",
            "tools/test_site32_runtime_manifest_integrity.py",
            "tools/test_site32_sw_precache_contract.py",
        ):
            self.assertIn(relative, payload["required_critical_assets"])

    def test_manifest_detects_unlisted_and_evidence_tamper(self) -> None:
        self._write_evidence_and_manifest()
        (self.root / "static" / "extra.txt").write_text("extra", encoding="utf-8")
        with self.assertRaises(manifest.ManifestError):
            manifest.verify_manifest(self.root)
        (self.root / "static" / "extra.txt").unlink()
        evidence_path = self.root / "static" / "quality" / "site31_browser_evidence.json"
        evidence_path.write_text("{}\n", encoding="utf-8")
        with self.assertRaises(manifest.ManifestError):
            manifest.verify_manifest(self.root)

    def test_site32_critical_assets_fail_closed_for_missing_and_tamper(self) -> None:
        (self.root / "cmdcenter" / "storage.py").unlink()
        with self.assertRaises(manifest.ManifestError):
            manifest.build_manifest(self.root)
        (self.root / "cmdcenter" / "storage.py").write_text("# fixture\n", encoding="utf-8")

        manifest.write_manifest(self.root)
        critical_paths = (
            "cmdcenter/config.py",
            "cmdcenter/public_dto.py",
            "cmdcenter/route_contract.py",
            "cmdcenter/storage.py",
            "tools/deploy_staged.sh",
            "tools/site32_service_isolation.sh",
            "tools/site32_state_bridge.py",
            "tools/site32_style_audit.py",
            "static/src/site32/state.js",
        )
        for relative in critical_paths:
            path = self.root / relative
            original = path.read_bytes()
            with self.subTest(relative=relative):
                path.write_bytes(original + b"# tamper\n")
                with self.assertRaises(manifest.ManifestError):
                    manifest.verify_manifest(self.root)
                path.write_bytes(original)
                manifest.verify_manifest(self.root)

        dynamic_module = self.root / "static" / "src" / "site32" / "state.js"
        original = dynamic_module.read_bytes()
        dynamic_module.unlink()
        with self.assertRaises(manifest.ManifestError):
            manifest.verify_manifest(self.root)
        dynamic_module.write_bytes(original)
        manifest.verify_manifest(self.root)

    def test_site32_v13_snapshot_does_not_require_future_state_bridge(self) -> None:
        legacy_release = "site32-global-commercial-v1.3-20260711"
        (self.root / "app.py").write_text(
            f'ASSET_VER = "{legacy_release}"\nRELEASED_AT = "2026-07-10T00:00:00Z"\n',
            encoding="utf-8-sig",
        )
        (self.root / "tools" / "site32_state_bridge.py").unlink()
        payload = manifest.write_manifest(self.root)
        self.assertNotIn(
            "tools/site32_state_bridge.py", payload["required_critical_assets"]
        )
        verified = manifest.verify_manifest(self.root)
        self.assertEqual(verified["release"], legacy_release)

    def test_site32_style_audit_failure_is_release_blocking(self) -> None:
        clean = style_audit.run_audit(
            self.root, style_audit.BUILTIN_BASELINE, "builtin"
        )
        self.assertTrue(clean["ok"], clean.get("violations"))
        site32_css = self.root / "static" / "site32.css"
        site32_css.write_text(".bad { transition: all 1s; }\n", encoding="utf-8")
        failed = style_audit.run_audit(
            self.root, style_audit.BUILTIN_BASELINE, "builtin"
        )
        self.assertFalse(failed["ok"])
        self.assertEqual(failed["exit_code"], style_audit.EXIT_GATE_FAILED)
        self.assertTrue(any(
            item["code"] == "site32_strict_zero"
            for item in failed["violations"]
        ))

    def test_browser_evidence_binds_release_digest_and_age(self) -> None:
        now = time.time()
        checks = sorted(REQUIRED_BROWSER_CHECKS)
        evidence = {
            "release": self.RELEASE,
            "manifest_digest": "a" * 64,
            "completed_at": now - 60,
            "base_url": "https://example.test/",
            "checks": [{"key": key, "state": "pass"} for key in checks],
        }
        valid, details = _validate_browser_evidence(
            evidence, release=self.RELEASE, manifest_digest="a" * 64, now=now, max_age_s=300
        )
        self.assertTrue(valid, details)
        evidence["completed_at"] = now - 301
        self.assertFalse(_validate_browser_evidence(
            evidence, release=self.RELEASE, manifest_digest="a" * 64, now=now, max_age_s=300
        )[0])
        evidence["completed_at"] = now - 60
        evidence["manifest_digest"] = "b" * 64
        self.assertFalse(_validate_browser_evidence(
            evidence, release=self.RELEASE, manifest_digest="a" * 64, now=now, max_age_s=300
        )[0])

    def test_r4_manifest_rejects_missing_asset_and_critical_hash_tamper(self) -> None:
        (self.root / "static" / "r4.js").unlink()
        with self.assertRaises(manifest.ManifestError):
            manifest.build_manifest(self.root)
        (self.root / "static" / "r4.js").write_text("'use strict';\n", encoding="utf-8")
        payload = manifest.write_manifest(self.root)
        payload["critical_assets_sha256"] = "0" * 64
        (self.root / manifest.MANIFEST_NAME).write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(manifest.ManifestError):
            manifest.verify_manifest(self.root)

    def test_release_bindings_cover_r4_queries_sw_and_hashes(self) -> None:
        payload = manifest.build_manifest(self.root)
        assets = ("style.css", "i18n.js", "app.js", "twin.js",
                  "r4.css", "r4.js", "r4-performance.js", "r4-accessibility.js",
                  "site32.css", "site32.js")
        html = "\n".join(
            f'<link href="/{name}?v={self.RELEASE}">' if name.endswith(".css")
            else f'<script src="/{name}?v={self.RELEASE}"></script>'
            for name in assets
        )
        app_text = f"const I18N_DEFAULT_VER='{self.RELEASE}';"
        sw_text = (
            f"const RELEASE = '{self.RELEASE}';\n"
            "const RELEASE_QUERY = `?v=${RELEASE}`;\n"
            + "\n".join(f"'/{name}'" for name in assets)
        )
        valid, detail = _validate_release_bindings(
            release=self.RELEASE,
            index_text=html,
            app_text=app_text,
            i18n_text=f"window.I18N_VERSION = '{self.RELEASE}';",
            sw_text=sw_text,
            manifest_payload=payload,
        )
        self.assertTrue(valid, detail)
        split_html = html.replace(self.RELEASE, "site31-global-commercial-r3.1-20260710", 1)
        self.assertFalse(_validate_release_bindings(
            release=self.RELEASE,
            index_text=split_html,
            app_text=app_text,
            i18n_text=f"window.I18N_VERSION = '{self.RELEASE}';",
            sw_text=sw_text,
            manifest_payload=payload,
        )[0])

    def test_browser_and_origin_evidence_must_postdate_release(self) -> None:
        release_not_before = _release_not_before(self.RELEASE, "2026-07-10T12:00:00Z")
        now = _release_not_before(self.RELEASE, "2026-07-10T13:00:00Z")
        browser = {
            "release": self.RELEASE,
            "manifest_digest": "a" * 64,
            "completed_at": "2026-07-10T11:59:59Z",
            "base_url": "https://example.test/",
            "checks": [{"key": key, "state": "pass"} for key in REQUIRED_BROWSER_CHECKS],
        }
        valid, detail = _validate_browser_evidence(
            browser,
            release=self.RELEASE,
            manifest_digest="a" * 64,
            now=now,
            max_age_s=86400,
            release_not_before_s=release_not_before,
        )
        self.assertFalse(valid)
        self.assertFalse(detail["newer_than_release"])

        origin = {
            "release": self.RELEASE,
            "manifest_digest": "a" * 64,
            "completed_at": "2026-07-10T12:30:00Z",
            "checks": [{"key": key, "state": "pass"} for key in REQUIRED_ORIGIN_CHECKS],
        }
        valid, detail = _validate_origin_evidence(
            origin,
            release=self.RELEASE,
            manifest_digest="a" * 64,
            now=now,
            max_age_s=86400,
            release_not_before_s=release_not_before,
        )
        self.assertTrue(valid, detail)
        origin["manifest_digest"] = "b" * 64
        self.assertFalse(_validate_origin_evidence(
            origin,
            release=self.RELEASE,
            manifest_digest="a" * 64,
            now=now,
            max_age_s=86400,
            release_not_before_s=release_not_before,
        )[0])

    def test_deploy_and_rollback_use_exact_tree_and_r4_contract(self) -> None:
        tools = Path(__file__).resolve().parent
        deploy = (tools / "deploy_staged.sh").read_text(encoding="utf-8")
        rollback = (tools / "rollback.sh").read_text(encoding="utf-8")
        for script in (deploy, rollback):
            self.assertIn("sync_exact_tree", script)
            self.assertIn("--delete", script)
            self.assertIn("--delete-excluded", script)
            for asset in ("r4.css", "r4.js", "r4-performance.js", "r4-accessibility.js"):
                self.assertIn(asset, script)
            for asset in ("site32.css", "site32.js"):
                self.assertIn(asset, script)
            for asset in ("route_contract.py", "storage.py", "site32_style_audit.py"):
                self.assertIn(asset, script)
            self.assertIn("static/src/site32", script)
        self.assertIn("systemd/xrd-cmdcenter.service", deploy)
        self.assertIn("auth/security_smoke.py", deploy)
        self.assertIn('(data.get("summary") or {}).get("release")', deploy)
        self.assertIn("--ignore-generated-style-audit", deploy)
        self.assertIn("site32_style_audit.json", deploy)
        self.assertIn("site32_state_bridge.py", deploy)
        self.assertIn('test -f "$root/public_evidence/rb_voe_r1_public.json"', deploy)
        self.assertIn('sync_exact_tree "$STAGE_REAL/public_evidence" "$CD/public_evidence"', deploy)
        self.assertIn('cp -a "$CD/public_evidence" "$BACKUP/public_evidence"', deploy)
        self.assertIn('sync_exact_tree "$BACKUP/public_evidence" "$CD/public_evidence"', deploy)
        self.assertIn("normalize_candidate_modes", deploy)
        self.assertIn("--chmod=D755,F644", deploy)
        self.assertIn('XRD_CMD_DB_PATH="$SMOKE_TMP/data.db"', deploy)
        self.assertGreaterEqual(deploy.count('normalize_candidate_modes "$STAGE_REAL"'), 2)
        self.assertIn("X-User: deploy-audit", deploy)
        self.assertIn("X-Role: judge", deploy)
        self.assertGreaterEqual(deploy.count('"${REVIEW_CURL[@]}"'), 2)
        self.assertIn("prune_snapshot_to_manifest", deploy)
        prune = deploy.index('prune_snapshot_to_manifest "$BACKUP"')
        verify_backup = deploy.index('"$MANIFEST_TOOL" "$BACKUP" --verify')
        promote = deploy.index('cp -a "$STAGE_REAL/app.py" "$CD/app.py"')
        self.assertLess(prune, verify_backup)
        self.assertLess(verify_backup, promote)
        self.assertIn("prepare_target_state_db", rollback)
        self.assertIn('STATE_BRIDGE="$SCRIPT_DIR/site32_state_bridge.py"', rollback)
        self.assertIn("install_target_unit", rollback)
        self.assertIn('xrd-cmdcenter.service.active', rollback)
        self.assertIn('if is_site32_release "$TARGET_RELEASE"', rollback)
        self.assertIn('(data.get("summary") or {}).get("release")', rollback)
        stop = rollback.index("sudo -n systemctl stop xrd-cmdcenter")
        state_bridge = rollback.index('prepare_target_state_db "$CURRENT_RELEASE" "$TARGET_RELEASE"')
        rollback_promote = rollback.index('cp -a "$PREV/app.py" "$CD/app.py"')
        self.assertLess(stop, state_bridge)
        self.assertLess(state_bridge, rollback_promote)
        self.assertIn('app_path.parent / "cmdcenter" / "config.py"', rollback)
        self.assertIn('test -f "$root/public_evidence/rb_voe_r1_public.json"', rollback)
        self.assertIn('cp -a "$CD/public_evidence" "$BACKOUT/public_evidence"', rollback)
        self.assertIn('sync_exact_tree "$PREV/public_evidence" "$CD/public_evidence"', rollback)
        self.assertIn('sync_exact_tree "$BACKOUT/public_evidence" "$CD/public_evidence"', rollback)
        style_write = deploy.index('--output "$STAGE_REAL/static/quality/site32_style_audit.json"')
        gate_run = deploy.index('"$GATE_TOOL" "$STAGE_REAL"')
        smoke_run = deploy.index('"$SMOKE_TOOL" "$STAGE_REAL"')
        manifest_writes = [match.start() for match in re.finditer(
            r'"\$MANIFEST_TOOL" "\$STAGE_REAL" --write', deploy
        )]
        self.assertTrue(any(style_write < offset < gate_run for offset in manifest_writes))
        self.assertTrue(any(gate_run < offset < smoke_run for offset in manifest_writes))

    def test_legacy_deploy_entry_fails_closed(self) -> None:
        deploy = (Path(__file__).resolve().parent / "deploy.sh").read_text(encoding="utf-8")
        self.assertIn("deploy_staged.sh", deploy)
        self.assertIn("exit 64", deploy)
        self.assertNotIn("systemctl restart", deploy)
        self.assertNotIn("cp -r", deploy)
        self.assertNotIn("sqlite3.connect", deploy)

    def test_gate_uses_an_isolated_schema_and_candidate_sidecars(self) -> None:
        gate = (Path(__file__).resolve().parent / "site31_gate_audit.py").read_text(encoding="utf-8")
        self.assertIn('TemporaryDirectory(prefix="site31-gate-db-")', gate)
        self.assertIn('release_module._init_db()', gate)
        self.assertIn('root / "auth"', gate)

    def test_r31_legacy_manifest_remains_verifiable_without_r4_assets(self) -> None:
        (self.root / "app.py").write_text(
            'ASSET_VER = "site31-global-commercial-r3.1-20260710"\n', encoding="utf-8-sig"
        )
        for name in ("r4.css", "r4.js", "r4-performance.js", "r4-accessibility.js"):
            (self.root / "static" / name).unlink()
        payload = manifest.build_manifest(self.root)
        legacy_paths = {entry["path"] for entry in payload["files"]}
        self.assertNotIn("requirements-production.txt", legacy_paths)
        self.assertFalse(any(path.startswith("cmdcenter/") for path in legacy_paths))
        self.assertFalse(any(path.startswith("systemd/") for path in legacy_paths))
        self.assertEqual(
            payload["digest_scope"],
            "app.py, assets.json, static/** and tools/** excluding static/quality/**",
        )
        for field in ("required_critical_assets", "critical_assets", "critical_assets_sha256"):
            payload.pop(field)
        unsigned = dict(payload)
        unsigned.pop("artifact_sha256", None)
        payload["artifact_sha256"] = manifest._canonical_sha256(unsigned)
        (self.root / manifest.MANIFEST_NAME).write_text(json.dumps(payload), encoding="utf-8")
        verified = manifest.verify_manifest(self.root)
        self.assertEqual(verified["release"], "site31-global-commercial-r3.1-20260710")

    def test_rollback_parser_accepts_utf8_bom_app(self) -> None:
        rollback = (Path(__file__).resolve().parent / "rollback.sh").read_text(encoding="utf-8")
        self.assertIn('read_text(encoding="utf-8-sig")', rollback)

    def test_r4_service_worker_cache_boundary_rejects_api_caching(self) -> None:
        secure = """
const CORE_PATHS = new Set(['/','/app.js']);
function isPrivateRequest(request, url) { return url.pathname.startsWith('/api/'); }
function responseIsPrivate(response) { return false; }
if (request.method !== 'GET' || url.origin !== self.location.origin) return;
if (isPrivateRequest(request, url)) return;
if (response.status !== 200 || response.redirected || responseIsPrivate(response)) return false;
"""
        valid, detail = _validate_sw_cache_boundary(secure, self.RELEASE)
        self.assertTrue(valid, detail)
        insecure = secure.replace("const CORE_PATHS = new Set(['/','/app.js']);", "const CORE_PATHS = new Set(['/api/config']);")
        self.assertFalse(_validate_sw_cache_boundary(insecure, self.RELEASE)[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
