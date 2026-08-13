from __future__ import annotations

import copy
import hashlib
import importlib.util
import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RENDERER_PATH = REPOSITORY_ROOT / "tools" / "publication" / "render_award_status.py"
SPEC = importlib.util.spec_from_file_location("render_award_status", RENDERER_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import infrastructure guard
    raise RuntimeError("could not load award renderer")
render_award_status = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = render_award_status
SPEC.loader.exec_module(render_award_status)


PROJECT_NAME_ZH = "基于双 RDK X5 异构协同的材料合成 AI 预测与多机具身实验助理机器人"
COMPETITION_ZH = "2026 全国大学生嵌入式芯片与系统设计竞赛"
DIVISION_ZH = "芯片应用赛道"
TOPIC_ZH = "地瓜机器人赛题"


class ReleaseMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.status = render_award_status.load_status(
            REPOSITORY_ROOT / "docs" / "competition" / "award_status.yaml"
        )

    def test_award_source_uses_first_prize_not_a_rank(self) -> None:
        render_award_status.validate(self.status, REPOSITORY_ROOT)
        self.assertEqual(COMPETITION_ZH, self.status["competition"]["name_zh"])
        self.assertEqual(DIVISION_ZH, self.status["competition"]["division"])
        self.assertEqual(TOPIC_ZH, self.status["competition"]["topic"])
        self.assertEqual("西南赛区", self.status["regional"]["region"])
        self.assertEqual("一等奖", self.status["regional"]["result"])

    def test_renderer_rejects_incorrect_regional_rank_wording(self) -> None:
        invalid = copy.deepcopy(self.status)
        invalid["regional"]["result"] = "第" + "1名"
        with self.assertRaisesRegex(ValueError, "西南赛区一等奖"):
            render_award_status.validate(invalid, REPOSITORY_ROOT)

    def test_generated_blocks_use_first_prize_wording(self) -> None:
        zh = render_award_status.zh_block(self.status, "事实边界")
        en = render_award_status.en_block(self.status)
        self.assertIn("| 西南赛区 | 一等奖 |", zh)
        self.assertIn("| Southwest Regional Contest | First Prize |", en)
        self.assertIn("| 全国总决赛 | 二等奖 |", zh)
        self.assertIn("| National final | Second Prize |", en)
        self.assertEqual("team_confirmed", self.status["national"]["status"])
        self.assertEqual("二等奖", self.status["national"]["result"])

    def test_readme_titles_use_the_formal_project_name(self) -> None:
        readme_zh = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
        readme_en = (REPOSITORY_ROOT / "README_en.md").read_text(encoding="utf-8")
        self.assertEqual(
            f"# {PROJECT_NAME_ZH}｜{COMPETITION_ZH}·{DIVISION_ZH}·{TOPIC_ZH}｜"
            "西南赛区一等奖·全国总决赛二等奖",
            readme_zh.splitlines()[0],
        )
        self.assertEqual(
            "# Material-Synthesis AI Prediction and Multi-Robot Embodied Laboratory "
            "Assistant Based on Dual-RDK X5 Heterogeneous Collaboration | 2026 "
            "National College Student Embedded Chip and System Design Competition · "
            "Chip Application Division · D-Robotics Topic | Southwest Regional First Prize · "
            "National Final Second Prize",
            readme_en.splitlines()[0],
        )
        self.assertIn(f"**Official Chinese project title:** {PROJECT_NAME_ZH}", readme_en)
        self.assertIn("西南赛区一等奖", readme_zh)
        self.assertIn("全国总决赛二等奖", readme_zh)
        self.assertIn("Southwest Regional First Prize", readme_en)
        self.assertIn("National Final Second Prize", readme_en)

    def test_readmes_link_all_three_public_demo_videos(self) -> None:
        video_paths = (
            "assets/media/videos/dashboard-xrd-pipeline.mp4",
            "assets/media/videos/embodied-assisted-workflow.mp4",
            "assets/media/videos/dual-arm-complete-hardware-demo.mp4",
        )
        for readme_name in ("README.md", "README_en.md"):
            text = (REPOSITORY_ROOT / readme_name).read_text(encoding="utf-8")
            for relative_path in video_paths:
                self.assertIn(relative_path, text)
                self.assertTrue((REPOSITORY_ROOT / relative_path).is_file())
                self.assertGreater((REPOSITORY_ROOT / relative_path).stat().st_size, 0)

    def test_readmes_show_all_six_latest_public_photos(self) -> None:
        photo_paths = (
            "assets/media/photos/project-overview-poster.webp",
            "assets/media/photos/dual-arm-workcell-full.webp",
            "assets/media/photos/embodied-platform-front-full.webp",
            "assets/media/photos/embodied-platform-sensor-deck-full.webp",
            "assets/media/photos/embodied-platform-three-quarter-full.webp",
            "assets/media/photos/team-dual-arm-integration-full.webp",
        )
        for readme_name in ("README.md", "README_en.md"):
            text = (REPOSITORY_ROOT / readme_name).read_text(encoding="utf-8")
            for relative_path in photo_paths:
                self.assertIn(relative_path, text)
                self.assertTrue((REPOSITORY_ROOT / relative_path).is_file())

    def test_obsolete_brand_and_repository_urls_are_absent(self) -> None:
        forbidden = (
            "XRD" + "智慧实验室",
            "XRD " + "Smart Lab",
            "github.com/" + "Xiaomiju-x/xrd",
            "github.com/" + "zhouLingxuan/xrd",
        )
        text_suffixes = {
            ".cff", ".css", ".html", ".js", ".json", ".md", ".py",
            ".service", ".toml", ".txt", ".yaml", ".yml",
        }
        findings: list[str] = []
        for path in REPOSITORY_ROOT.rglob("*"):
            if not path.is_file() or ".git" in path.parts or path.suffix.lower() not in text_suffixes:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(value in text for value in forbidden):
                findings.append(path.relative_to(REPOSITORY_ROOT).as_posix())
        self.assertEqual([], findings)

    def test_release_checksum_manifest_is_current(self) -> None:
        checksum_path = REPOSITORY_ROOT / "docs" / "releases" / "v1.0.1" / "SHA256SUMS.txt"
        entries = [line.split("  ", 1) for line in checksum_path.read_text(encoding="utf-8").splitlines()]
        self.assertGreaterEqual(len(entries), 16)
        seen: set[str] = set()
        for expected, relative_path in entries:
            self.assertNotIn(relative_path, seen)
            seen.add(relative_path)
            candidate = REPOSITORY_ROOT / relative_path
            self.assertTrue(candidate.is_file(), relative_path)
            if candidate.suffix.lower() in {".json", ".txt", ".yml"}:
                self.assertNotIn(b"\r\n", candidate.read_bytes(), relative_path)
            self.assertEqual(expected, hashlib.sha256(candidate.read_bytes()).hexdigest(), relative_path)


if __name__ == "__main__":
    unittest.main()
