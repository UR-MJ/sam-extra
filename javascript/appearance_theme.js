(function () {
    "use strict";

    var OPTION_KEY = "sam3_appearance_theme";
    var STORAGE_KEY = "sam-extra.appearance-theme.v1";
    var DEFAULT_SLUG = "forge-default";
    var THEME_SLUGS = Object.freeze({
        "Forge Default": DEFAULT_SLUG,
        "Graphite Ember": "graphite-ember",
        "Obsidian Violet": "obsidian-violet",
        "Warm Espresso": "warm-espresso",
        "OLED Mono": "oled-mono",
    });
    var currentSlug = DEFAULT_SLUG;
    var addedDarkClass = false;

    function normalizeTheme(value) {
        return THEME_SLUGS[value] || DEFAULT_SLUG;
    }

    function appRoot() {
        if (typeof gradioApp !== "function") return null;
        try {
            return gradioApp();
        } catch (_error) {
            return null;
        }
    }

    function collectRoots() {
        var roots = new Set();
        if (document.documentElement) roots.add(document.documentElement);
        if (document.body) roots.add(document.body);

        var host = document.querySelector("gradio-app");
        if (host) roots.add(host);

        var app = appRoot();
        if (app) {
            if (app.host) roots.add(app.host);
            else if (app.nodeType === 1) roots.add(app);
            if (typeof app.querySelectorAll === "function") {
                app.querySelectorAll(".dark").forEach(function (node) {
                    roots.add(node);
                });
            }
        }

        document.querySelectorAll(".dark").forEach(function (node) {
            roots.add(node);
        });
        return Array.from(roots);
    }

    // Every custom palette is dark, but Gradio gates a large amount of its own
    // and Forge's CSS on a `dark` class that Gradio puts on document.body only
    // when its resolved theme is dark (default is __theme=system). Without it a
    // light-mode session gets a hybrid: our near-white body text over
    // light-mode-only surfaces — pale toast backgrounds, light Prism colours on
    // a near-black code block, and `ul.options li.selected` painted with
    // --neutral-100. Add the class ourselves, and remember that we did so we
    // never strip a `dark` that Gradio owns when returning to Forge Default.
    function syncDarkClass() {
        var body = document.body;
        if (!body || !body.classList) return;
        if (currentSlug === DEFAULT_SLUG) {
            if (addedDarkClass) {
                body.classList.remove("dark");
                addedDarkClass = false;
            }
            return;
        }
        if (!body.classList.contains("dark")) {
            body.classList.add("dark");
            addedDarkClass = true;
        }
    }

    function syncThemeRoots() {
        syncDarkClass();
        collectRoots().forEach(function (root) {
            if (!root || !root.dataset) return;
            if (currentSlug === DEFAULT_SLUG) {
                delete root.dataset.sam3Theme;
            } else {
                root.dataset.sam3Theme = currentSlug;
            }
        });
    }

    function cacheTheme(slug) {
        try {
            if (slug === DEFAULT_SLUG) localStorage.removeItem(STORAGE_KEY);
            else localStorage.setItem(STORAGE_KEY, slug);
        } catch (_error) {
            // The Forge option remains authoritative when storage is blocked.
        }
    }

    function applyTheme(value, cache) {
        var slug = Object.prototype.hasOwnProperty.call(THEME_SLUGS, value)
            ? THEME_SLUGS[value]
            : Object.values(THEME_SLUGS).includes(value)
                ? value
                : DEFAULT_SLUG;
        currentSlug = slug;
        syncThemeRoots();
        if (cache !== false) cacheTheme(slug);
        document.dispatchEvent(new CustomEvent("sam3:appearance-theme", {
            detail: {theme: slug},
        }));
    }

    function configuredTheme() {
        try {
            if (typeof opts !== "undefined" && opts) {
                return opts[OPTION_KEY];
            }
        } catch (_error) {
            // opts is a global lexical binding and may not exist this early.
        }
        return null;
    }

    function applyConfiguredTheme() {
        applyTheme(configuredTheme() || "Forge Default", true);
    }

    try {
        var cached = localStorage.getItem(STORAGE_KEY);
        if (cached) applyTheme(cached, false);
    } catch (_error) {
        // Ignore storage-denied contexts; Forge opts will apply after load.
    }

    if (typeof onOptionsAvailable === "function") {
        onOptionsAvailable(applyConfiguredTheme);
    }
    if (typeof onOptionsChanged === "function") {
        onOptionsChanged(applyConfiguredTheme);
    }
    if (typeof onUiLoaded === "function") {
        onUiLoaded(syncThemeRoots);
    }
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", syncThemeRoots, {once: true});
    } else {
        syncThemeRoots();
    }

    window.sam3AppearanceTheme = Object.freeze({
        apply: applyTheme,
        current: function () {
            return currentSlug;
        },
        normalize: normalizeTheme,
    });
})();
