#!/usr/bin/env python3
"""Regression tests for the Site32 anonymous research projection."""

from __future__ import annotations

import json
import unittest
from unittest import mock

from site31_r4_contract_test import _load_app_isolated


def _public_rows(count: int) -> list[dict]:
    return [
        {
            "id": f"public-{idx}",
            "formula": f"Host{idx}:Cr3+",
            "host": f"Host{idx}",
            "dopant": "Cr3+",
            "site": "M",
            "verdict": "REFERENCE",
            "lambda_em": 700.0 + idx,
            "confidence_interval": "",
            "band": "nir_i",
            "method": "curated public fixture",
            "source": "curated",
            "trace_id": "",
            "batch": "",
            "work_order": "",
            "stability_pct": None,
            "round": "public fixture",
            "state": "replay",
            "created": "",
        }
        for idx in range(count)
    ]


class Site32PublicResearchContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module, cls.observations, cls.tempdir, cls.import_error = _load_app_isolated()
        if cls.import_error is None:
            cls.module.app.config.update(TESTING=True)
            cls.client = cls.module.app.test_client()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tempdir.cleanup()

    def setUp(self) -> None:
        self.assertIsNone(self.import_error, f"isolated import failed: {self.import_error!r}")

    def test_internal_workorders_never_enter_anonymous_materials(self) -> None:
        private = dict(_public_rows(1)[0], id="private-workorder", work_order="WO-SECRET")
        with mock.patch.object(self.module, "_atlas_material_rows", return_value=[]), \
                mock.patch.object(self.module, "_observed_material_rows", return_value=[]), \
                mock.patch.object(self.module, "_seed_material_rows", return_value=_public_rows(2)), \
                mock.patch.object(self.module, "_workorder_material_rows", return_value=[private]) as workorders:
            rows = self.module._materials_all_rows()
        self.assertFalse(workorders.called, "anonymous projection must not read internal workorders")
        self.assertNotIn("private-workorder", {row.get("id") for row in rows})
        self.assertTrue(all(not row.get("work_order") for row in rows))

    def test_passport_and_bundle_ignore_request_query_truncation(self) -> None:
        rows = _public_rows(20)
        with mock.patch.object(self.module, "_materials_all_rows", return_value=rows), \
                mock.patch.object(self.module, "_public_api_doc_entries", return_value=[]), \
                mock.patch.object(self.module, "_public_status_components", return_value=[]):
            with self.module.app.test_request_context("/api/research_passport?limit=1&q=missing"):
                passport = self.module._research_passport_payload()
        evidence_card = next(item for item in passport["passport_cards"] if item["key"] == "evidence")
        self.assertIn("20 material rows", evidence_card["value"])

        with mock.patch.object(self.module, "_materials_all_rows", return_value=rows), \
                mock.patch.object(self.module, "_research_passport_payload", return_value=passport), \
                mock.patch.object(self.module, "_api_doc_entries", return_value=[]), \
                mock.patch.object(self.module, "_site31_evidence_objects", return_value=[]), \
                mock.patch.object(self.module, "_site31_trust_center_payload", return_value={"controls": []}):
            with self.module.app.test_request_context("/api/evidence_bundle.json?limit=1&q=missing"):
                bundle = self.module._evidence_bundle_payload()
        self.assertEqual(bundle["materials_summary"]["total"], 20)
        self.assertEqual(len(bundle["materials_sample"]), 12)

    def test_material_id_cannot_be_used_as_prediction_identity(self) -> None:
        rows = _public_rows(1)
        rows[0]["id"] = "seed-yag-cr3"
        with mock.patch.object(self.module, "_materials_all_rows", return_value=rows):
            response = self.client.get(
                "/api/predictions/seed-yag-cr3", headers={"Host": "xiaomiju.xyz"})
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.get_json()["kind"], "prediction")

    def test_evidence_objects_have_complete_english_boundaries(self) -> None:
        objects = self.module._site31_evidence_objects({"counts": {}, "downloads": []})
        self.assertTrue(objects)
        for item in objects:
            with self.subTest(evidence_id=item.get("evidence_id")):
                self.assertTrue(item.get("title_en"))
                self.assertTrue(item.get("description_en"))
                self.assertTrue(item.get("intended_use_en"))
                self.assertTrue(item.get("prohibited_use_en"))
                self.assertTrue(item.get("limitations_en"))
                self.assertTrue(item.get("citation", {}).get("text_en"))
                self.assertTrue(item.get("uncertainty", {}).get("statement_en"))
                self.assertTrue(all(claim.get("statement_en") for claim in item.get("claims", [])))

    def test_research_collections_are_public_safe_and_have_stable_detail(self) -> None:
        response = self.client.get(
            "/api/research_collections", headers={"Host": "xiaomiju.xyz"})
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["schema_version"], "site32.research_collections.v1")
        self.assertEqual(payload["count"], 5)
        self.assertEqual(payload["items"][0]["collection_id"], "rc:xrd:materials-atlas")
        rendered = str(payload).lower()
        for forbidden in (
            "work_order", "batch", "cmd_vel", "device_path", "private_prompt",
            "raw_log", "192.168.31.", "10.197.54.",
        ):
            self.assertNotIn(forbidden, rendered)

        detail = self.client.get(
            "/api/research_collections/rc:xrd:arm01-redundancy",
            headers={"Host": "xiaomiju.xyz"},
        )
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.get_json()["item"]["scope"], "arm01")
        missing = self.client.get(
            "/api/research_collections/rc:xrd:missing",
            headers={"Host": "xiaomiju.xyz"},
        )
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(missing.get_json()["error"], "collection_not_found")

    def test_public_bundle_manifest_and_sitemap_exclude_protected_surfaces(self) -> None:
        bundle = self.client.get(
            "/api/evidence_bundle.json", headers={"Host": "xiaomiju.xyz"}).get_json()
        protected = {
            "/api/site31_portal", "/api/site31_scorecard", "/api/site31_gate_evidence",
            "/api/releases", "/api/logs", "/api/config",
        }
        self.assertFalse(protected & set(bundle["public_endpoints"]))
        self.assertNotIn("/command", bundle["core_pages"])
        for row in bundle["materials_sample"]:
            self.assertNotIn("work_order", row)
            self.assertNotIn("batch", row)

        manifest = self.client.get(
            "/api/public_manifest", headers={"Host": "xiaomiju.xyz"}).get_json()
        hrefs = {item["href"] for item in manifest["exports"]}
        self.assertNotIn("/api/site31_portal", hrefs)
        self.assertNotIn("/api/site31_scorecard", hrefs)
        self.assertNotIn("/api/export/workorders.csv", hrefs)
        self.assertIn("/api/research_collections", hrefs)

        sitemap = self.client.get(
            "/sitemap.xml", headers={"Host": "xiaomiju.xyz"}).get_data(as_text=True)
        for path in ("/fleet", "/tasks", "/logs", "/command", "/studio"):
            self.assertNotIn(f"xiaomiju.xyz{path}<", sitemap)

    def test_assets_and_gate_evidence_hide_network_and_deployment_details(self) -> None:
        assets = self.client.get(
            "/api/assets", headers={"Host": "xiaomiju.xyz"})
        self.assertEqual(assets.status_code, 200)
        rendered_assets = json.dumps(assets.get_json(), ensure_ascii=False).lower()
        for forbidden in (
            "192.168.", ".136", "systemd", "socket activation", "frp",
            "two-hop", "deploy_staged", "forward_auth", "lb_policy",
        ):
            self.assertNotIn(forbidden, rendered_assets)

        raw_gate = {
            "schema_version": "site31.gate_evidence.v1",
            "release": self.module.ASSET_VER,
            "artifact_sha256": "c" * 64,
            "generated_at": 1784500000,
            "phase": "deployed",
            "valid": True,
            "gate": "pass",
            "dimensions": {"security": {"state": "verified"}},
            "summary": {"verified": 1, "failed": 0},
            "checks": [{
                "domain": "security", "key": "origin.boundary", "label": "Origin boundary",
                "state": "verified", "max_points": 1, "earned_points": 1,
                "critical": True, "evidence": "http://127.0.0.1:29100 deploy_staged.sh",
                "residual_risk": "systemd path /home/rdk/private",
            }],
            "asset_manifest": {
                "valid": True, "manifest_digest": "a" * 64,
                "critical_assets_sha256": "b" * 64,
                "critical_assets": [{"path": "/home/rdk/private/app.py"}],
            },
            "browser_evidence": {
                "valid": True, "completed_at": "2026-07-19T22:00:00Z",
                "manifest_matches": True, "base_url": "http://127.0.0.1:29100/",
            },
            "origin_evidence": {
                "valid": True, "completed_at": "2026-07-19T22:00:00Z",
                "manifest_matches": True, "path": "/home/rdk/private/origin.json",
            },
        }
        headers = {"Host": "xiaomiju.xyz", "X-User": "judge", "X-Role": "judge"}
        with mock.patch.object(self.module, "_site31_gate_evidence_payload", return_value=raw_gate):
            gate = self.client.get("/api/site31_gate_evidence", headers=headers)
        self.assertEqual(gate.status_code, 200)
        payload = gate.get_json()
        self.assertEqual(payload["artifact_sha256"], "c" * 64)
        self.assertEqual(payload["asset_manifest"]["manifest_digest"], "a" * 64)
        rendered_gate = json.dumps(payload, ensure_ascii=False).lower()
        for forbidden in ("127.0.0.1", "/home/rdk "deploy_staged", "systemd", "base_url"):
            self.assertNotIn(forbidden, rendered_gate)

        raw_quality = self.client.get(
            "/quality/site31_gate_evidence.json", headers=headers)
        self.assertEqual(raw_quality.status_code, 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
