# Article artwork used in link previews

The final 1200×630 cards are assembled automatically by Quartz. This directory owns the **banner artwork** placed beneath each post's title; the selected PNGs live in `source/content/media/social/`. Frontmatter selects them with `socialImage` and describes them with `socialImageAlt`.

## Existing visuals

| Banner | Source | Capture / adaptation |
| --- | --- | --- |
| `kda-reference.png` | `content/media/kda/kda-future-animation.html` | The actual canvas decal, paused at reference token 8, `draw(0)`. |
| `ptq-dispatch.png` | `content/widgets/ptq-conveyor-board.html` | The actual dispatch canvas at `render(2400)`. |
| `ulp-binades.png` | `content/widgets/ulp-binades.html` | The existing overview SVG, including its original dark palette. |
| `clc-workers.png` | `content/widgets/clc-work-distribution.html` | Eight steps of the existing simulation; controls hidden and workers arranged in two columns for the banner. No work values changed. |
| `flex-determinism.png` | Two screenshots already in `content/media/`, dated 2025-10-17 at 11.34.06 AM and 12.11.58 PM | Cropped forward/backward result columns, with before/after labels. The SVG embeds the original pixels during rendering. |

Regenerate the four widget captures with an **ephemeral** Playwright install exposed through `NODE_PATH`, from `source/`:

```sh
NODE_PATH=/path/to/temp-playwright/node_modules node visuals/social/capture.cjs
```

The capture script opens the existing self-contained HTML files locally, uses a 1080×286 viewport at device scale 2, and writes PNGs. It does not change the widgets or require a running site. Playwright is deliberately not a blog dependency.

## New illustrations

`covers/*.svg` contains distinct, topic-specific artwork for the remaining posts, plus the FlexAttention screenshot composition. The pizza illustration follows biga → mixing → proofing → pizza. The other new illustrations use conceptual shapes, not measurements or benchmark plots.

Rasterize the SVG sources from `source/`:

```sh
node visuals/social/render.mjs
```

The renderer embeds local screenshot references, uses the bundled IBM Plex fonts without system-font discovery, and writes PNG banners. Run `npm run build` afterwards to refresh all final cards. Editing only a raw asset does not trigger a full rebuild in Quartz's default preview watcher; explicitly rebuild after regenerating banners.

The article bodies and their existing interactive graphics are unchanged. Collection pages retain the common Nuggets artwork; all 15 published posts have their own banner.
