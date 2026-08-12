#!/usr/bin/env python3
"""Static fail-closed contract for Site32 full/defense/minimal modes."""

from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static"


class Site32VisualModeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.html = (STATIC / "index.html").read_text(encoding="utf-8")
        cls.entry = (STATIC / "site32.js").read_text(encoding="utf-8")
        cls.appearance = (STATIC / "src" / "site32" / "appearance.js").read_text(encoding="utf-8")
        cls.state = (STATIC / "src" / "site32" / "state.js").read_text(encoding="utf-8")
        cls.motion = (STATIC / "src" / "site32" / "motion.js").read_text(encoding="utf-8")
        cls.performance = (STATIC / "r4-performance.js").read_text(encoding="utf-8")
        cls.r4_css = (STATIC / "r4.css").read_text(encoding="utf-8")
        cls.style = (STATIC / "style.css").read_text(encoding="utf-8")
        cls.site32_css = (STATIC / "site32.css").read_text(encoding="utf-8")
        cls.full_css = (STATIC / "full-experience.css").read_text(encoding="utf-8")
        cls.full_js = (STATIC / "full-experience.js").read_text(encoding="utf-8")
        cls.service_worker = (STATIC / "sw.js").read_text(encoding="utf-8")
        cls.app = (STATIC / "app.js").read_text(encoding="utf-8")

    def test_vivid_is_default_and_bootstraps_before_styles(self) -> None:
        self.assertIn('data-site32-presentation-mode="vivid"', self.html)
        self.assertIn('data-site32-visual-mode="vivid"', self.html)
        self.assertIn("cmdcenter.visualMode.v3", self.html)
        bootstrap = self.html.index("cmdcenter.visualMode.v3")
        first_stylesheet = self.html.index('rel="stylesheet"')
        self.assertLess(bootstrap, first_stylesheet)
        self.assertIn("saved==='vivid'||saved==='defense'||saved==='minimal'", self.html)
        self.assertIn("cmdcenter.visualMode.v2", self.html)

    def test_native_persistent_three_option_control_exists(self) -> None:
        self.assertIn('<fieldset class="site32-appearance"', self.html)
        self.assertEqual(self.html.count('name="site32-visual-mode"'), 3)
        self.assertIn('value="vivid"', self.html)
        self.assertIn('value="defense"', self.html)
        self.assertIn('value="minimal"', self.html)
        self.assertIn('id="appearanceAnnouncer"', self.html)
        self.assertIn('id="site32ModeToggle"', self.html)
        self.assertIn('onclick="site32QuickModeToggle()"', self.html)

    def test_more_menu_uses_native_button_and_explicit_aria_state(self) -> None:
        self.assertIn('<button class="act more-btn" id="btnMore"', self.html)
        self.assertIn('aria-haspopup="menu" aria-expanded="false"', self.html)
        self.assertIn('id="moreMenu" role="menu" aria-hidden="true"', self.html)
        self.assertIn("trigger.setAttribute('aria-expanded',open?'true':'false')", self.app)
        self.assertIn("m.setAttribute('aria-hidden',open?'false':'true')", self.app)

    def test_appearance_is_a_first_class_module_and_state_scope(self) -> None:
        self.assertIn("./src/site32/appearance.js", self.entry)
        self.assertIn("installSite32Appearance({ state })", self.entry)
        self.assertIn("appearance,", self.entry)
        self.assertIn("'appearance'", self.entry)
        for token in ("requested: 'vivid'", "effective: 'vivid'", "transparencyReduced", "forcedColors"):
            self.assertIn(token, self.state)
        self.assertIn("export function installSite32Appearance", self.appearance)
        self.assertIn("localStorage.setItem(STORAGE_KEY, requested)", self.appearance)
        self.assertIn("windowRef.addEventListener('storage'", self.appearance)

    def test_visual_preference_and_render_tier_are_independent(self) -> None:
        self.assertIn("var visualMode='vivid'", self.performance)
        self.assertIn("setVisualMode:setVisualMode", self.performance)
        self.assertIn("visualMode==='defense'", self.performance)
        self.assertIn("visualMode==='minimal'", self.performance)
        self.assertNotIn("reduceMotion && reduceMotion.matches?'reduced-motion'", self.performance)
        self.assertNotIn("if(reduceMotion && reduceMotion.matches) setTier('static'", self.performance)
        self.assertIn("else if(forcedColors && forcedColors.matches) next='static'", self.performance)
        self.assertIn("else next='balanced'", self.performance)
        self.assertIn("'defense-static-background':'full-experience'", self.performance)
        self.assertIn("body.dataset.r4Render", self.motion)
        self.assertNotIn("r4RenderTier", self.motion)

    def test_flattening_is_mode_scoped_and_vivid_assets_remain_cached(self) -> None:
        self.assertNotIn("body .page:not(#overview) :is(\n  .card", self.r4_css)
        self.assertNotIn("Site31 performance gate: keep legacy optical layers static", self.style)
        self.assertIn('html[data-site32-visual-mode="vivid"]', self.site32_css)
        self.assertIn('html[data-site32-presentation-mode="defense"]', self.site32_css)
        self.assertIn('html[data-site32-visual-mode="minimal"]', self.site32_css)
        self.assertIn("'/src/site32/appearance.js'", self.service_worker)

    def test_complete_surface_is_default_and_minimal_owns_gui_reduction(self) -> None:
        self.assertIn("const minimalMode=document.documentElement.dataset.site32VisualMode==='minimal'", self.app)
        self.assertIn("const reduceSurface=publicOnly&&minimalMode", self.app)
        self.assertIn("element.hidden=reduceSurface&&!PUBLIC_VIEW_KEYS.has(key)", self.app)
        self.assertIn('html[data-site32-visual-mode="minimal"] body .ia-strip', self.site32_css)
        self.assertIn('html:not([data-site32-visual-mode="minimal"]) .stage', self.style)
        self.assertIn('body[data-r4-render="static"] .lg-stage', self.style)

    def test_full_experience_restores_complete_navigation_and_real_three(self) -> None:
        self.assertEqual(self.html.count('class="ia-chip'), 26)
        for route in (
            "home", "highlight", "defense", "benchmark", "brain", "lab", "mq",
            "studio", "atlas", "build", "fsd", "replay", "car", "arm", "fleet",
            "tasks", "command", "twin", "assets", "status", "ops", "obs", "logs",
            "traces", "sec", "release",
        ):
            self.assertIn(f'data-k="{route}"', self.html)
        self.assertIn('/three.min.js?v=site32-global-commercial-v1.13-20260720', self.html)
        self.assertLess(self.html.index('/three.min.js?'), self.html.index('/app.js?'))
        self.assertIn('/full-experience.css?v=site32-global-commercial-v1.13-20260720', self.html)
        self.assertIn('/full-experience.js?v=site32-global-commercial-v1.13-20260720', self.html)
        self.assertIn("window.ensureThreeLibrary", self.full_js)

    def test_more_menu_has_animation_independent_visible_terminal_state(self) -> None:
        self.assertIn(".more-menu.show", self.full_css)
        self.assertIn("grid-template-columns: repeat(2, minmax(0, 1fr))", self.full_css)
        self.assertIn("opacity: 1 !important", self.full_css)
        self.assertIn("visibility: visible !important", self.full_css)
        self.assertIn("animation: none !important", self.full_css)

    def test_vivid_background_matches_peak_density_without_route_pausing(self) -> None:
        for token in (
            "for(let i=0;i<32;i++)",
            "for(let i=0;i<22;i++)",
            "const max=mobile?14:30",
            "const max=mobile?9:20",
            "S.w<760?48:118",
        ):
            self.assertIn(token, self.app)
        self.assertNotIn("S.active=k==='home'", self.app)
        self.assertIn("S.active=presentationMode()!=='minimal'", self.app)
        self.assertIn("presentationMode()==='vivid'", self.app)
        self.assertIn("window.__site32BackgroundState", self.app)

    def test_defense_mode_freezes_background_only(self) -> None:
        self.assertIn("return mode!=='vivid'", self.app)
        self.assertIn("function frameTime(){ return animated()?performance.now():0; }", self.app)
        self.assertIn("visualMode==='defense'", self.performance)
        self.assertIn("visualMode==='minimal'?'minimal':'vivid'", self.performance)
        self.assertIn('.aurora::before', self.site32_css)
        self.assertNotIn('html[data-site32-presentation-mode="defense"] body .ia-strip {\n  display: none', self.site32_css)

    def test_home_canvas_has_one_renderer_owner(self) -> None:
        self.assertIn("typeof window.init3DRealRig!=='function'", self.app)
        self.assertIn("window.init3DRealRig();", self.app)
        self.assertNotIn("(window.init3DRealRig||init3D)();", self.app)


if __name__ == "__main__":
    unittest.main(verbosity=2)
