from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class AppearanceThemeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.python = (ROOT / "scripts" / "appearance_theme.py").read_text(
            encoding="utf-8"
        )
        cls.javascript = (
            ROOT / "javascript" / "appearance_theme.js"
        ).read_text(encoding="utf-8")
        cls.css = (ROOT / "style.css").read_text(encoding="utf-8")
        cls.tokens = (ROOT / "tokens.css").read_text(encoding="utf-8")
        cls.design = (ROOT / "design.md").read_text(encoding="utf-8")

    def test_settings_registers_all_requested_theme_choices(self):
        self.assertIn('OPT_APPEARANCE_THEME = "sam3_appearance_theme"', self.python)
        self.assertIn("script_callbacks.on_ui_settings(on_ui_settings)", self.python)
        for label in (
            "Forge Default",
            "Graphite Ember",
            "Obsidian Violet",
            "Warm Espresso",
            "OLED Mono",
        ):
            self.assertIn(f'"{label}"', self.python)
            self.assertIn(f'"{label}"', self.javascript)

    def test_settings_registers_fast_dropdown_visible_choice_limit(self):
        self.assertIn(
            'OPT_FAST_DROPDOWN_VISIBLE_CHOICES = '
            '"sam3_fast_dropdown_visible_choices"',
            self.python,
        )
        self.assertIn("DEFAULT_FAST_DROPDOWN_VISIBLE_CHOICES = 60", self.python)
        self.assertIn('"minimum": 10', self.python)
        self.assertIn('"maximum": 200', self.python)
        self.assertIn('"step": 5', self.python)

    def test_frontend_applies_options_and_cleans_up_for_forge_default(self):
        self.assertIn("onOptionsAvailable(applyConfiguredTheme)", self.javascript)
        self.assertIn("onOptionsChanged(applyConfiguredTheme)", self.javascript)
        self.assertIn("delete root.dataset.sam3Theme", self.javascript)
        self.assertIn("root.dataset.sam3Theme = currentSlug", self.javascript)
        self.assertIn("sam-extra.appearance-theme.v1", self.javascript)

    def test_style_imports_locked_tokens_and_has_hallmark_stamp(self):
        self.assertTrue(
            self.css.startswith("/* Hallmark · pre-emit critique:"),
            "style.css must keep the Hallmark app stamp as its first line",
        )
        self.assertIn('@import url("./tokens.css");', self.css)
        self.assertIn("design-system: design.md", self.css)
        self.assertIn("# Design — sam-extra Forge Appearance", self.design)

    def test_each_custom_palette_has_a_complete_semantic_contract(self):
        required = (
            "--sam3-color-paper:",
            "--sam3-color-paper-2:",
            "--sam3-color-paper-3:",
            "--sam3-color-paper-4:",
            "--sam3-color-ink:",
            "--sam3-color-muted:",
            "--sam3-color-rule:",
            "--sam3-color-accent:",
            "--sam3-color-accent-ink:",
            "--sam3-color-focus:",
            "--sam3-color-error:",
            "--sam3-color-success:",
        )
        for slug in (
            "graphite-ember",
            "obsidian-violet",
            "warm-espresso",
            "oled-mono",
        ):
            match = re.search(
                rf'data-sam3-theme="{re.escape(slug)}"\]\s*\{{(?P<body>.*?)\n\}}',
                self.tokens,
                flags=re.DOTALL,
            )
            self.assertIsNotNone(match, slug)
            body = match.group("body")
            for token in required:
                self.assertIn(token, body, f"{slug}: {token}")

    def test_new_theme_layer_avoids_known_css_slop_patterns(self):
        lowered = self.tokens.lower()
        self.assertNotIn("#000", lowered)
        self.assertNotIn("#fff", lowered)
        self.assertNotIn("rgb(", lowered)
        self.assertNotIn("rgba(", lowered)
        self.assertNotIn("linear-gradient", lowered)
        self.assertNotIn("radial-gradient", lowered)
        self.assertNotIn("transition: all", lowered)
        self.assertNotIn("z-index: 9999", lowered)

    def test_global_control_states_and_reduced_motion_are_present(self):
        for marker in (
            ":focus-visible",
            ":active",
            ":disabled",
            '[aria-busy="true"]',
            '[aria-invalid="true"]',
            '[data-state="success"]',
            "@media (prefers-reduced-motion: reduce)",
        ):
            self.assertIn(marker, self.css)

    def test_forge_default_is_an_exact_opt_out(self):
        self.assertNotIn('data-sam3-theme="forge-default"', self.tokens)
        self.assertIn(
            'if (slug === DEFAULT_SLUG) localStorage.removeItem(STORAGE_KEY)',
            self.javascript,
        )


if __name__ == "__main__":
    unittest.main()
