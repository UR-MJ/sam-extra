from __future__ import annotations

import unittest
from pathlib import Path

from fastapi import FastAPI

from sam3ext.lora_manager_core import (
    _BRIDGE_JS,
    LORA_CONFIG_PATH,
    LORA_SPAWN_PATH,
    lora_config_data,
    register_lora_routes,
)

ROOT = Path(__file__).resolve().parents[1]


class LoraRouteTests(unittest.TestCase):
    def test_routes_registered_idempotently(self):
        app = FastAPI()
        self.assertTrue(register_lora_routes(app))
        self.assertFalse(register_lora_routes(app))

        paths = {route.path for route in app.routes}
        self.assertIn(LORA_CONFIG_PATH, paths)
        self.assertIn(LORA_SPAWN_PATH, paths)

        for route in app.routes:
            if route.path in (LORA_CONFIG_PATH, LORA_SPAWN_PATH):
                self.assertIn("GET", route.methods)

    def test_config_payload_shape(self):
        data = lora_config_data()
        self.assertIn("available", data)
        self.assertIn("replace", data)
        self.assertIn("port", data)
        # Without a webui/settings environment the config is safe defaults.
        self.assertIsInstance(data["available"], bool)
        self.assertIsInstance(data["replace"], bool)
        self.assertIsInstance(data["port"], int)

    def test_bridge_intercepts_single_and_bulk_sends(self):
        # The injected iframe bridge must catch BOTH the single-card context menu
        # and the multi-select bulk submenu, so nothing routes to ComfyUI.
        self.assertIn("context-menu-item[data-action]", _BRIDGE_JS)
        self.assertIn("#bulkContextMenu", _BRIDGE_JS)
        self.assertIn(".model-card.selected", _BRIDGE_JS)
        self.assertIn("stopImmediatePropagation", _BRIDGE_JS)
        # Replace vs append is carried on the message.
        self.assertIn("replace:", _BRIDGE_JS)

    def test_forge_side_handles_bulk_and_replace(self):
        lora = (ROOT / "javascript" / "lora_manager.js").read_text(encoding="utf-8")
        # Insert helper honours the replace flag (strip existing <lora:...>).
        self.assertIn("function sam3InsertLora(text, replace)", lora)
        self.assertIn("<lora:[^>]*>", lora)
        self.assertIn("sam3InsertLora(d.text, !!d.replace)", lora)
        # The manager is injected directly into the one Forge page.
        self.assertIn("tryAllAndStop();", lora)
        self.assertNotIn("__sam3_live_workspace", lora)
        self.assertNotIn("inLiveChildFrame", lora)

    def test_dom_bootstrap_watchers_stop_after_injection(self):
        lora = (ROOT / "javascript" / "lora_manager.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('myBtn.classList.remove("selected")', lora)
        self.assertIn("function stopBootstrapWatchers()", lora)
        self.assertIn("if (tryAll()) stopBootstrapWatchers()", lora)
        self.assertNotIn(
            "new MutationObserver(function () { tryAll(); })",
            lora,
        )

    def test_manage_tab_visibility_survives_a_tab_strip_rerender(self):
        """Tab switching must not depend on nodes captured at injection.

        Gradio 4.40 keeps non-selected TabItems mounted and only writes an
        inline display, so nothing but this code can hide the synthetic manager
        pane. Binding to captured buttons means any re-render of the tab strip
        silently drops the listeners and the manager stays on screen forever, so
        the listener is delegated to the container and the nodes are resolved on
        every click.
        """
        lora = (ROOT / "javascript" / "lora_manager.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('container.addEventListener("click"', lora)
        self.assertIn("function liveButtons()", lora)
        self.assertIn("function livePanes()", lora)
        # No per-button listeners bound to the injection-time arrays.
        self.assertNotIn("allBtns.forEach(function (btn, idx)", lora)
        self.assertNotIn("var allPanes =", lora)

    def test_closing_manage_restores_the_previous_tab(self):
        """Gradio's state never changed while the manager was open.

        Its reactive block is therefore not dirty and it will not re-show the
        previous pane, so the restore is this code's job. Clearing the inline
        display instead would make a .tabitem fall back to display:block and
        show every pane at once.
        """
        lora = (ROOT / "javascript" / "lora_manager.js").read_text(
            encoding="utf-8"
        )
        self.assertIn("var savedNativeIndex = -1;", lora)
        self.assertNotIn('removeProperty("display")', lora)
        start = lora.index("function showManage(active)")
        end = lora.index("function selectTab(idx)", start)
        body = lora[start:end]
        # A real Gradio tab click wins; otherwise fall back to the saved tab.
        self.assertIn("selectedNow >= 0 ? selectedNow : savedNativeIndex", body)
        self.assertIn('pane.style.display = (index === target) ? "block" : "none"', body)

    def test_a_native_tab_click_closes_manage_without_waiting_for_gradio(self):
        """Clicking another tab must close the manager on that first click.

        Gradio flushes its tab switch asynchronously, and when it already
        considers the clicked tab selected it does not fire at all, so neither a
        deferred handler nor Gradio can be relied on. The click path closes
        synchronously and a container observer catches the async ordering.
        """
        lora = (ROOT / "javascript" / "lora_manager.js").read_text(
            encoding="utf-8"
        )
        start = lora.index('container.addEventListener("click"')
        end = lora.index("var navWatcher", start)
        body = lora[start:end]
        # Our own button opens (deferred is fine); a native button closes now.
        self.assertIn("setTimeout(function () { showManage(true); }, 0);", body)
        self.assertIn("showManage(false);", body)
        self.assertNotIn(
            "showManage(btn === myBtn || btn.hasAttribute", body
        )
        # The observer is the ordering safety net and cannot loop.
        self.assertIn("var manageActive = false;", lora)
        self.assertIn("if (!manageActive) return;", lora)
        self.assertIn("navWatcher.observe(container,", lora)


if __name__ == "__main__":
    unittest.main()
