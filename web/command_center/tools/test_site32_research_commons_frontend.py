#!/usr/bin/env python3
"""Static contract tests for the Site32 public research commons."""

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Site32ResearchCommonsFrontendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
        cls.app = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        cls.css = (ROOT / "static" / "site32.css").read_text(encoding="utf-8")
        cls.i18n = (ROOT / "static" / "i18n.js").read_text(encoding="utf-8")

    def test_commons_has_one_release_bound_discovery_band(self) -> None:
        self.assertEqual(self.html.count('id="researchCommons"'), 1)
        self.assertIn('id="researchCommonsList"', self.html)
        self.assertIn('id="researchCommonsDetail"', self.html)
        self.assertIn('href="/api/research_collections"', self.html)
        self.assertNotIn('id="researchCommonsSummary" role="status" aria-live="polite" data-i18n=', self.html)

    def test_loading_filter_detail_retry_and_deep_link_states_exist(self) -> None:
        for token in (
            "async function loadResearchCommons()",
            "function researchCommonsFilter(scope,trigger)",
            "function researchCollectionOpen(collectionId,updateUrl)",
            "function researchCommonsClose(updateUrl)",
            "data-prc-retry",
            "url.searchParams.set('collection',collectionId)",
            "url.searchParams.delete('collection')",
            "!items.some(item=>item.collection_id===_researchCommonsOpen)) researchCommonsClose(true)",
        ):
            self.assertIn(token, self.app)

    def test_dynamic_content_is_escaped_and_links_are_same_origin(self) -> None:
        self.assertIn("function researchCommonsHref(value,fallback)", self.app)
        self.assertIn("!href.startsWith('//')", self.app)
        self.assertIn("!href.includes('://')", self.app)
        self.assertIn("uiEsc(researchCommonsLabel(item,'title'))", self.app)

    def test_anonymous_role_hides_non_public_routes_and_work_orders(self) -> None:
        self.assertIn("const PUBLIC_VIEW_KEYS=new Set(['home','status','atlas','brain','models','assets','twin','more'])", self.app)
        self.assertIn("option[value=\"work_order\"]", self.app)
        self.assertIn("applyAudienceRole('public')", self.app)
        self.assertIn("Public visitor · read-only", self.app)
        self.assertIn("#moreMenu button[hidden]", self.css)
        self.assertIn("includes('/logout')", self.app)
        self.assertIn("includes('palOpen')", self.app)
        self.assertIn("if(!hasReviewerAccess())", self.app)

    def test_vivid_and_minimal_modes_share_the_same_content(self) -> None:
        self.assertIn("#overview .public-research-commons", self.css)
        self.assertIn('html[data-site32-visual-mode="minimal"] body #overview .public-research-commons', self.css)
        self.assertNotIn('html[data-site32-visual-mode="minimal"] body #overview .public-research-commons {\n  display: none', self.css)

    def test_commons_copy_has_english_contract(self) -> None:
        for key in (
            '"commons.title"',
            '"commons.lead"',
            '"commons.boundary"',
            '"commons.materials"',
            '"commons.embodied"',
        ):
            self.assertIn(key, self.i18n)


if __name__ == "__main__":
    unittest.main(verbosity=2)
