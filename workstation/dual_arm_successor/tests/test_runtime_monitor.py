from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from workstation.dual_arm_successor.runtime import skill_graph


def valid_result() -> dict:
    events = []
    for index, phase in enumerate(skill_graph.EXPECTED_PHASES):
        events.append(
            {
                "time": f"2026-07-20T05:{index:02d}:00+08:00",
                "phase": phase,
                "status": sorted(skill_graph.SUCCESS_STATUS[phase])[0],
            }
        )
    return {
        "status": "CLOSED_LOOP_DONE",
        "events": events,
        "apriltag": {
            "required_dict": "DICT_APRILTAG_36h11",
            "required_id": 2,
            "passed": True,
        },
        "overhead": {
            "cpu_authority": "BAG_PRESENT",
            "bpu_forward_executed": True,
        },
    }


class SkillGraphTests(unittest.TestCase):
    def _write(self, value: dict, directory: Path) -> Path:
        path = directory / "result.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_valid_run_agrees_without_motion_authority(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = self._write(valid_result(), Path(temp))
            receipt = skill_graph.evaluate(skill_graph.read_json(source), source)
        self.assertEqual(receipt["prediction"]["verdict"], "AGREE")
        self.assertFalse(receipt["authority"]["motion_authority"])
        self.assertFalse(receipt["authority"]["execution_allowed"])
        self.assertEqual(receipt["authority"]["actuator_commands_issued"], 0)
        self.assertEqual(receipt["data_scope"]["kind"], "STAGE_ONLY")

    def test_missing_phase_degrades(self) -> None:
        value = valid_result()
        value["events"] = [
            event for event in value["events"] if event["phase"] != "bag_release_visual_trigger"
        ]
        with tempfile.TemporaryDirectory() as temp:
            source = self._write(value, Path(temp))
            receipt = skill_graph.evaluate(skill_graph.read_json(source), source)
        self.assertEqual(receipt["prediction"]["verdict"], "SHADOW_DEGRADED")
        self.assertIn("MISSING_PHASE", receipt["evidence"]["hard_failures"])

    def test_wrong_apriltag_id_degrades(self) -> None:
        value = valid_result()
        value["apriltag"]["required_id"] = 8
        with tempfile.TemporaryDirectory() as temp:
            source = self._write(value, Path(temp))
            receipt = skill_graph.evaluate(skill_graph.read_json(source), source)
        self.assertEqual(receipt["prediction"]["verdict"], "SHADOW_DEGRADED")
        self.assertIn("APRILTAG_ID2_NOT_PROVEN", receipt["evidence"]["hard_failures"])


if __name__ == "__main__":
    unittest.main()
