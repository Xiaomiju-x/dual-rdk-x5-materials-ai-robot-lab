#!/usr/bin/env python3
"""Regression tests for the pure Site32 research-search kernel."""

from __future__ import annotations

import importlib.util
import sqlite3
import subprocess
import sys
import threading
import unittest
from pathlib import Path
from urllib.parse import parse_qs
from unittest import mock


CMD_ROOT = Path(__file__).resolve().parents[1]
if str(CMD_ROOT) not in sys.path:
    sys.path.insert(0, str(CMD_ROOT))
MODULE_PATH = CMD_ROOT / "cmdcenter" / "research_search.py"


def load_search_module(name: str = "site32_research_search_under_test"):
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Site32ResearchSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.search = load_search_module()

    def test_import_has_no_runtime_side_effects(self) -> None:
        observations = {"sqlite": [], "threads": [], "subprocess": []}

        def blocked_connect(*args, **kwargs):
            observations["sqlite"].append(args[0] if args else None)
            raise AssertionError("sqlite3.connect should not run while importing research_search")

        def blocked_thread_start(thread, *args, **kwargs):
            observations["threads"].append(thread.name)
            raise AssertionError("Thread.start should not run while importing research_search")

        def blocked_subprocess_run(*args, **kwargs):
            observations["subprocess"].append(args[0] if args else None)
            raise AssertionError("subprocess.run should not run while importing research_search")

        before = threading.active_count()
        with mock.patch.object(sqlite3, "connect", side_effect=blocked_connect), \
                mock.patch.object(threading.Thread, "start", new=blocked_thread_start), \
                mock.patch.object(subprocess, "run", side_effect=blocked_subprocess_run):
            load_search_module("site32_research_search_import_probe")

        self.assertEqual(observations, {"sqlite": [], "threads": [], "subprocess": []})
        self.assertEqual(threading.active_count(), before)

    def test_default_contract_is_stable_chinese(self) -> None:
        result = self.search.search_research({"q": "YAG"})
        self.assertEqual(result["schema_version"], self.search.SCHEMA_VERSION)
        self.assertEqual(result["default_language"], "zh-CN")
        self.assertEqual(result["schema"]["default_language"], "zh-CN")
        self.assertIn("YAG:Cr3+", result["default_suggestions"])
        self.assertEqual([group["key"] for group in result["facet_groups"]], ["kind", "status", "source"])
        self.assertEqual(result["facet_groups"][0]["label"], "类型")

    def test_unrelated_query_returns_zero_without_material_default_score(self) -> None:
        result = self.search.search_research({"q": "unrelated banana wafer"})
        self.assertEqual(result["total"], 0)
        self.assertEqual(result["items"], [])
        self.assertEqual(
            sum(option["count"] for group in result["facet_groups"] for option in group["options"]),
            0,
        )

    def test_yag_cr3_host_dopant_hits_seed_yag_cr3(self) -> None:
        result = self.search.search_research({"q": "YAG:Cr3+"})
        self.assertGreater(result["total"], 0)
        self.assertEqual(result["items"][0]["id"], "seed-yag-cr3")
        self.assertEqual(result["items"][0]["host"], "YAG")
        self.assertEqual(result["items"][0]["dopant"], "Cr3+")
        self.assertIn("host_dopant", result["items"][0]["matched_fields"])

    def test_ggg_ni2_host_dopant_hits_seed_ggg_ni2(self) -> None:
        result = self.search.search_research({"q": "GGG:Ni2+"})
        self.assertGreater(result["total"], 0)
        self.assertEqual(result["items"][0]["id"], "seed-ggg-ni2")
        self.assertEqual(result["items"][0]["host"], "GGG")
        self.assertEqual(result["items"][0]["dopant"], "Ni2+")

    def test_exact_ids_are_hit_at_one(self) -> None:
        material = self.search.search_research({"q": "seed-yag-cr3"})
        evidence = self.search.search_research({"q": "ev:xrd:materials"})

        self.assertEqual(material["items"][0]["id"], "seed-yag-cr3")
        self.assertEqual(material["items"][0]["matched_fields"][0], "id")
        self.assertGreaterEqual(material["items"][0]["score"], 9000)

        self.assertEqual(evidence["items"][0]["id"], "ev:xrd:materials")
        self.assertEqual(evidence["items"][0]["kind"], "evidence")
        self.assertEqual(evidence["items"][0]["matched_fields"][0], "id")
        self.assertGreaterEqual(evidence["items"][0]["score"], 9000)

    def test_chinese_and_english_natural_text_match(self) -> None:
        chinese = self.search.search_research({"q": "公开只读科研证据门户"})
        english = self.search.search_research({"q": "public NIR phosphor materials dataset"})

        self.assertGreater(chinese["total"], 0)
        self.assertIn(chinese["items"][0]["id"], {"ev:xrd:passport", "atlas"})
        self.assertGreater(english["total"], 0)
        self.assertEqual(english["items"][0]["id"], "ev:xrd:materials")

    def test_kind_status_source_filters_and_structured_facets(self) -> None:
        result = self.search.search_research({
            "q": "YAG:Cr3+",
            "kind": "material",
            "status": "replay",
            "source": "curated",
        })
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["items"][0]["id"], "seed-yag-cr3")
        self.assertEqual(
            result["query"]["filters"],
            {"kind": "material", "status": "replay", "source": "curated"},
        )
        by_key = {group["key"]: group for group in result["facet_groups"]}
        self.assertTrue(next(option for option in by_key["kind"]["options"] if option["value"] == "material")["selected"])
        self.assertTrue(next(option for option in by_key["status"]["options"] if option["value"] == "replay")["selected"])
        self.assertTrue(next(option for option in by_key["source"]["options"] if option["value"] == "curated")["selected"])

        excluded = self.search.search_research({"q": "YAG:Cr3+", "source": "mirror"})
        self.assertEqual(excluded["total"], 0)

    def test_share_query_roundtrip_preserves_plus_and_filters(self) -> None:
        params = {"q": "YAG:Cr3+", "kind": "material", "status": "replay", "source": "curated", "limit": "5"}
        query = self.search.build_share_query(params)
        parsed = parse_qs(query)

        self.assertEqual(parsed["q"], ["YAG:Cr3+"])
        self.assertEqual(parsed["kind"], ["material"])
        self.assertEqual(parsed["status"], ["replay"])
        self.assertEqual(parsed["source"], ["curated"])
        self.assertEqual(parsed["limit"], ["5"])

        roundtrip = self.search.search_research(parsed)
        self.assertEqual(roundtrip["total"], 1)
        self.assertEqual(roundtrip["items"][0]["id"], "seed-yag-cr3")
        self.assertEqual(roundtrip["query"]["share_query"], query)

    def test_malicious_values_are_cleaned_and_href_helpers_are_same_origin(self) -> None:
        bad = {
            "q": "<script>alert(1)</script>",
            "kind": "material<script>",
            "status": "replay\nSet-Cookie",
            "source": "javascript:alert(1)",
            "limit": "9999",
        }
        query = self.search.build_share_query(bad)
        normalized = self.search.normalize_search_params(bad)

        self.assertNotIn("<", query)
        self.assertNotIn(">", query)
        self.assertNotIn("javascript", query)
        self.assertNotIn("Set-Cookie", query)
        self.assertEqual(normalized["kind"], "")
        self.assertEqual(normalized["status"], "")
        self.assertEqual(normalized["source"], "")
        self.assertEqual(normalized["limit"], self.search.MAX_LIMIT)

        self.assertTrue(self.search.is_same_origin_href("/api/evidence_objects/ev%3Axrd%3Amaterials"))
        for href in ("https://evil.test/x", "//evil.test/x", "javascript:alert(1)", "/api/../admin", "/api/%2F%2Fevil"):
            self.assertFalse(self.search.is_same_origin_href(href), href)
        self.assertEqual(
            self.search.build_share_href("https://evil.test/x", bad).split("?", 1)[0],
            "/search",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
