# Design — sam-extra Forge Appearance

A locked design system for the Forge Neo appearance layer shipped by this
extension. It changes presentation only: Forge routes, Gradio component
ownership, generation behavior, settings semantics, and extension DOM contracts
stay intact.

## Genre

Atmospheric, with the restraint of a technical workbench. The interface is dark
because it is used for long image-generation sessions, not because it needs
decorative glow, glass, gradients, or motion.

## Macrostructure family

- App pages: **Workbench** — preserve Forge's existing top-level tabs and dense
  control hierarchy. SAM3's txt2img layout may keep its asymmetric
  Parameters / Scripts / Gallery columns.
- Settings and extension pages: **Long Document** inside the same Workbench
  shell — existing section order and component ownership remain unchanged.
- Content/help surfaces: typography and dividers only; no decorative cards.

## Theme

The geometry, typography, spacing, and interaction language are shared. A user
may select one global palette at a time:

- **Forge Default** — no appearance override.
- **Graphite Ember** — neutral graphite surfaces with Forge's warm orange as
  the single signal colour.
- **Obsidian Violet** — violet-tinted near-black surfaces with a restrained
  violet signal colour.
- **Warm Espresso** — warm brown-black surfaces with amber signal colour.
- **OLED Mono** — near-black, low-chroma surfaces with an off-white signal.

The concrete OKLCH values and the Gradio variable bridge live in
[`tokens.css`](tokens.css). Palettes never mix between tabs.

## Typography

- Display: Source Sans Pro, weight 700, roman.
- Body: Source Sans Pro, weight 400.
- Mono: IBM Plex Mono, weight 400/600.
- Existing Forge font loading is retained to avoid an additional network
  dependency and layout shift.
- Data and numeric controls use tabular figures.

## Spacing

A named 4-point scale is defined in `tokens.css`. New appearance-layer rules use
the named tokens; existing Forge layout dimensions are not globally rewritten.

## Motion

- Easings: named exponential curves in `tokens.css`.
- State changes: colour and at most a one-pixel press translation.
- Page and tab content: no entrance animation.
- Focus indicators: instant.
- Reduced motion: spatial movement removed, state feedback retained.

## Microinteractions stance

- Silent success.
- No celebratory toasts.
- Hover is paired with keyboard focus.
- Focus rings are immediate and visible.
- Theme changes apply immediately after Settings are saved and persist through
  Forge's option store.

## CTA voice

- Primary: solid signal colour, no gradient, compact radius, one-line label.
- Secondary: elevated neutral surface and visible boundary.
- Disabled/loading/error/success states retain text or another non-colour
  signal.

## Per-page allowances

- App pages use no decorative enrichment; function carries the page.
- Existing preview images and generated galleries are content, not decoration.
- Third-party extensions may retain their own intentional brand surfaces, but
  common Gradio controls inherit this system.

## What pages MUST share

- Source Sans Pro / IBM Plex Mono.
- Surface elevation by lightness rather than shadow.
- The selected global palette and its one signal colour.
- Input, button, tab, focus, disabled, loading, error, and success language.
- Compact radii and the named spacing/motion tokens.

## What pages MAY differ on

- Existing Forge page composition and control density.
- Extension-specific information architecture.
- Generated-image and model-card content.

## Exports

### tokens.css

`tokens.css` is the canonical runtime export. It contains all colour, font,
spacing, radius, duration, easing, and elevation tokens plus the Gradio mapping.

### Tailwind v4 `@theme`

```css
@theme {
  --color-background: var(--sam3-color-paper);
  --color-foreground: var(--sam3-color-ink);
  --color-primary: var(--sam3-color-accent);
  --font-sans: var(--sam3-font-body);
  --font-mono: var(--sam3-font-mono);
  --spacing-md: var(--sam3-space-md);
  --radius-md: var(--sam3-radius-md);
}
```

### DTCG `tokens.json`

```json
{
  "color": {
    "paper": {"$value": "{sam3.color.paper}", "$type": "color"},
    "ink": {"$value": "{sam3.color.ink}", "$type": "color"},
    "accent": {"$value": "{sam3.color.accent}", "$type": "color"}
  },
  "space": {
    "md": {"$value": "1rem", "$type": "dimension"}
  }
}
```

### shadcn/ui CSS variables

```css
:root {
  --background: var(--sam3-color-paper);
  --foreground: var(--sam3-color-ink);
  --primary: var(--sam3-color-accent);
  --primary-foreground: var(--sam3-color-accent-ink);
  --muted: var(--sam3-color-paper-3);
  --muted-foreground: var(--sam3-color-muted);
  --border: var(--sam3-color-rule);
  --input: var(--sam3-color-rule);
  --ring: var(--sam3-color-focus);
}
```
