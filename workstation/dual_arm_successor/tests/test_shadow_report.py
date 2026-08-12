from __future__ import annotations

import unittest

from workstation.dual_arm_successor.runtime.build_shadow_report import (
    build_html,
    require_no_motion_authority,
)


class ShadowReportTests(unittest.TestCase):
    def _receipt(self) -> dict:
        return {
            "authority": {
                "motion_authority": False,
                "execution_allowed": False,
                "actuator_commands_issued": 0,
            },
            "prediction": {"verdict": "AGREE", "current_phase": "DONE", "next_skill": "DONE"},
            "evidence": {
                "apriltag_id2": True,
                "bag_present_cpu_authority": True,
                "bpu_auxiliary_forward": True,
                "closed_loop_done": True,
            },
            "skill_graph": {
                "expected": ["observe", "drop"],
                "observed": ["observe", "drop"],
                "missing": [],
            },
            "source": {"sha256": "a" * 64},
        }

    def test_report_keeps_authority_boundary_visible(self) -> None:
        content = build_html(self._receipt(), [])
        self.assertIn("MOTION AUTHORITY = FROZEN V3", content)
        self.assertIn("SHADOW ACTUATOR COMMANDS = 0", content)
        self.assertIn("TRAINING_PENDING", content)

    def test_report_rejects_motion_authority(self) -> None:
        receipt = self._receipt()
        receipt["authority"]["motion_authority"] = True
        with self.assertRaises(ValueError):
            require_no_motion_authority(receipt)


if __name__ == "__main__":
    unittest.main()
