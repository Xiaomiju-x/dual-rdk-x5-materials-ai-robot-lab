from __future__ import annotations

import copy
import json
import os
import sys
import unittest
from pathlib import Path

CMD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CMD_ROOT.parents[1]
if str(CMD_ROOT) not in sys.path:
    sys.path.insert(0, str(CMD_ROOT))

os.environ.setdefault("XRD_CMD_TEST_MODE", "1")
os.environ.setdefault("XRD_CMD_RUNTIME", "0")

import app as cmd_app
from cmdcenter.access import classify_request
from cmdcenter.rb_voe_public import RbVoePublicError, load_public_snapshot, validate_public_snapshot


class RbVoePublicContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.snapshot_path = CMD_ROOT / "public_evidence" / "rb_voe_r1_public.json"
        cls.snapshot = json.loads(cls.snapshot_path.read_text(encoding="utf-8"))
        cls.client = cmd_app.app.test_client()

    def test_snapshot_is_release_bound_and_authority_free(self) -> None:
        payload = load_public_snapshot(self.snapshot_path, site_release=cmd_app.ASSET_VER)
        self.assertEqual(payload["source"]["acceptance_status"], "PASS")
        self.assertTrue(payload["source"]["external_pin_verified"])
        self.assertEqual(payload["source"]["evidence_source"], "SIMULATED_COUNTERFACTUAL")
        self.assertFalse(payload["authority"]["execution_authority"])
        self.assertFalse(payload["authority"]["physical_closure_proven"])
        self.assertEqual(payload["authority"]["physical_risk_denominator_increment"], 0)
        self.assertEqual(payload["policy"]["decision"], "NEXT_EVIDENCE")
        self.assertEqual(payload["policy"]["risk"], 4)
        self.assertEqual(payload["hold_witness"]["reason"], "NO_FEASIBLE_OPTION")

    def test_digest_and_authority_mutations_are_rejected(self) -> None:
        digest_drift = copy.deepcopy(self.snapshot)
        digest_drift["policy"]["risk"] = 0
        with self.assertRaisesRegex(RbVoePublicError, "digest mismatch"):
            validate_public_snapshot(digest_drift)

        authority_drift = copy.deepcopy(self.snapshot)
        authority_drift["authority"]["execution_authority"] = True
        authority_drift.pop("public_snapshot_sha256")
        from cmdcenter.rb_voe_public import canonical_sha256

        authority_drift["public_snapshot_sha256"] = canonical_sha256(authority_drift)
        with self.assertRaisesRegex(RbVoePublicError, "authority boundary"):
            validate_public_snapshot(authority_drift)

    def test_public_api_and_access_contract_are_get_only(self) -> None:
        response = self.client.get("/api/rb_voe/explain", headers={"Host": "localhost"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("Cache-Control"), "no-store")
        payload = response.get_json()
        self.assertEqual(payload["site_release"], cmd_app.ASSET_VER)
        self.assertEqual(classify_request("/api/rb_voe/explain", "GET").scope, "public")
        self.assertEqual(classify_request("/api/rb_voe/explain", "POST").scope, "internal")

    def test_public_payload_contains_no_private_deployment_material(self) -> None:
        text = json.dumps(self.snapshot, ensure_ascii=True, sort_keys=True)
        for forbidden in ("192.168.", "/home/", "C:\\\\Users", "sk-"):
            self.assertNotIn(forbidden, text)
        keys = set()

        def collect(value):
            if isinstance(value, dict):
                keys.update(value)
                for item in value.values():
                    collect(item)
            elif isinstance(value, list):
                for item in value:
                    collect(item)

        collect(self.snapshot)
        self.assertTrue({"perturbation_id", "patches"}.isdisjoint(keys))

    def test_brain_frontend_fetches_and_renders_rb_voe(self) -> None:
        index = (CMD_ROOT / "static" / "index.html").read_text(encoding="utf-8")
        script = (CMD_ROOT / "static" / "app.js").read_text(encoding="utf-8")
        style = (CMD_ROOT / "static" / "site32.css").read_text(encoding="utf-8")
        self.assertIn('id="rbVoeExplain"', index)
        self.assertIn("/api/rb_voe/explain", script)
        self.assertIn("function rbVoeRender()", script)
        self.assertIn(".voe-shell", style)


if __name__ == "__main__":
    unittest.main()
