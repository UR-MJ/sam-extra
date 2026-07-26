"""Global Forge appearance themes owned by the SAM Extra extension.

The setting is persisted through Forge's normal options store. The frontend
reads it from the existing ``opts`` payload, so changing it does not require a
Forge core patch or an additional API route.
"""
from __future__ import annotations

import gradio as gr

from modules import script_callbacks, shared


OPT_APPEARANCE_THEME = "sam3_appearance_theme"
OPT_FAST_DROPDOWN_VISIBLE_CHOICES = "sam3_fast_dropdown_visible_choices"
DEFAULT_FAST_DROPDOWN_VISIBLE_CHOICES = 60

THEME_FORGE_DEFAULT = "Forge Default"
THEME_GRAPHITE_EMBER = "Graphite Ember"
THEME_OBSIDIAN_VIOLET = "Obsidian Violet"
THEME_WARM_ESPRESSO = "Warm Espresso"
THEME_OLED_MONO = "OLED Mono"

APPEARANCE_THEME_CHOICES = (
    THEME_FORGE_DEFAULT,
    THEME_GRAPHITE_EMBER,
    THEME_OBSIDIAN_VIOLET,
    THEME_WARM_ESPRESSO,
    THEME_OLED_MONO,
)


def on_ui_settings() -> None:
    section = ("sam3_appearance", "SAM Extra Appearance")
    shared.opts.add_option(
        OPT_APPEARANCE_THEME,
        shared.OptionInfo(
            THEME_FORGE_DEFAULT,
            "Forge UI 테마 (Settings 저장 후 즉시 적용)",
            gr.Dropdown,
            {"choices": list(APPEARANCE_THEME_CHOICES)},
            section=section,
        ),
    )
    shared.opts.add_option(
        OPT_FAST_DROPDOWN_VISIBLE_CHOICES,
        shared.OptionInfo(
            DEFAULT_FAST_DROPDOWN_VISIBLE_CHOICES,
            "빠른 드롭다운 최대 표시 항목 수",
            gr.Slider,
            {"minimum": 10, "maximum": 200, "step": 5},
            section=section,
        ).info(
            "목록이 이 값 이하면 전부 펼쳐 표시합니다. "
            "더 긴 목록은 여기까지 표시하고 검색으로 나머지를 찾습니다."
        ),
    )


script_callbacks.on_ui_settings(on_ui_settings)
