#!/usr/bin/env python3
"""Static contract for finals facts and public evidence boundaries."""

from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]


class Site32FinalsContentContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        active_paths = (
            ROOT / "app.py",
            ROOT / "assets.json",
            ROOT / "cmdcenter" / "research_collections.py",
            ROOT / "static" / "index.html",
            ROOT / "static" / "app.js",
            ROOT / "static" / "i18n.js",
        )
        cls.active = "\n".join(path.read_text(encoding="utf-8") for path in active_paths)
        public_repo = WORKSPACE / "embedded_contest" / "github_public_repo"
        cls.public = "\n".join(
            (public_repo / name).read_text(encoding="utf-8")
            for name in ("README.md", "README_cn.md", "workstation_public/skills.py")
        )

    def test_final_hardware_facts_are_present(self) -> None:
        for token in (
            "0.50m 里程计闭环",
            "下降放瓶与复位",
            "arm01 单臂视觉冗余",
            "arm02 并发四周期研磨",
            "CPU/OpenCV",
        ):
            self.assertIn(token, self.active)

    def test_stale_arm02_target_wording_is_absent(self) -> None:
        lowered = self.active.lower()
        for forbidden in (
            "arm02 复赛目标",
            "arm02 复赛协作目标",
            "arm02 planned",
            "arm02 finals collaboration target",
            "arm02 is labelled as finals collaboration target",
        ):
            self.assertNotIn(forbidden.lower(), lowered)

    def test_shadow_and_bpu_authority_boundaries_remain_explicit(self) -> None:
        self.assertIn("Lab-FSD 仍为 shadow/assist", self.active)
        self.assertIn("BPU 仅作辅助", self.active)
        self.assertIn("公网不下发动作", self.active)

    def test_public_repository_is_mock_only(self) -> None:
        lowered = self.public.lower()
        for forbidden in (
            "release_all_servos(",
            ".send_angles(",
            "peer_base",
            "_serial_lock",
            "http_json",
        ):
            self.assertNotIn(forbidden, lowered)
        self.assertIn("no_public_actuation", lowered)
        self.assertIn("mock-only", lowered)


if __name__ == "__main__":
    unittest.main(verbosity=2)
