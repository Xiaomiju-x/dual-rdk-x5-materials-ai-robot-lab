from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from workstation.dual_arm_successor.runtime.campaign_index import build_index


class CampaignIndexTests(unittest.TestCase):
    def _result(self, status: str, mode: str = "EXECUTE") -> dict:
        return {
            "schema_version": "xrd-finals-part3-composed-v1",
            "mode": mode,
            "status": status,
            "events": [{"phase": "execute_preflight", "status": "PASS"}],
        }

    def test_indexes_only_composed_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, status in (("one", "CLOSED_LOOP_DONE"), ("two", "FAILED")):
                directory = root / name
                directory.mkdir()
                (directory / "result.json").write_text(
                    json.dumps(self._result(status)), encoding="utf-8"
                )
            other = root / "other"
            other.mkdir()
            (other / "result.json").write_text(
                json.dumps({"schema_version": "another-schema"}), encoding="utf-8"
            )
            index = build_index(root)
        self.assertEqual(index["summary"]["composed_results"], 2)
        self.assertEqual(index["summary"]["physical_success_results"], 1)
        self.assertEqual(index["summary"]["physical_failure_results"], 1)
        self.assertFalse(index["training_boundary"]["continuous_13d_policy_training_eligible"])
        self.assertEqual(index["authority"]["actuator_commands_issued"], 0)


if __name__ == "__main__":
    unittest.main()
