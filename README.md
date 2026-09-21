# Nuggets Blog

Static site for publishing Obsidian notes through Quartz and GitHub Pages.

## Edit Content
- Update Markdown notes in `source/content`. Each note needs frontmatter with at least `title:` and `date:` (ISO `YYYY-MM-DD`); set `draft: true` to exclude one from the site.
- `source/content/All Notes.md` is auto-generated from frontmatter — do not edit by hand. A pre-commit hook regenerates and stages it on every commit, and CI regenerates it during builds.
- Place any hand-written HTML in `source/raw_html`.

The pre-commit hook lives at `scripts/git-hooks/pre-commit` and is wired in automatically by `npm install` (via the `prepare` script in `source/package.json`, which sets `core.hooksPath`). On a fresh clone, `cd source && npm install` is enough; if you ever bypass it, you can wire it manually with `git config core.hooksPath scripts/git-hooks`.

## Preview Locally
```bash
cd source
npm install
npm run dev
```
This regenerates `All Notes.md`, then starts the Quartz watcher. Open the printed URL to browse the site. Use `npm run build` for a one-shot production build.

## Interactive documentation helpers

Ported from `attention-gym` and adapted to Quartz: use a fenced block with a JSON object. Asset paths are relative to `source/content/`, must start with `media/`, and work on nested notes and the GitHub Pages `/nuggets/` base path. Titles are required; `height` is an optional positive integer in pixels.

### Perfetto traces

Store the trace and its screenshot under `source/content/media/traces/`:

````markdown
```perfetto
{
  "src": "media/traces/training.pftrace",
  "snapshot": "media/traces/training.png",
  "title": "Training step",
  "alt": "CPU launch activity above GPU kernel execution",
  "height": 680
}
```
````

The snapshot loads first. Clicking opens the interactive viewer, with fullscreen, return-to-snapshot, and download controls. The interactive viewer requires access to `ui.perfetto.dev`; the trace is fetched from this site and passed to that viewer in the browser. Only publish traces and screenshots you are comfortable making public.

### Plotly charts

Export a standalone HTML chart (for example, `fig.write_html("source/content/media/plots/results.html", include_plotlyjs=True, config={"responsive": True})`), then embed it:

````markdown
```plotly
{"src": "media/plots/results.html", "title": "Benchmark results", "height": 560}
```
````

Chart backgrounds, text, and Cartesian axes follow the site's light/dark theme. Data colors, titles, annotations, and layout are preserved. Plotly is loaded only inside chart frames, not globally on every note. Exports must expose `window.Plotly`, as ordinary standalone Plotly HTML exports do.

### Standalone HTML widgets

````markdown
```html-widget
{"src": "media/widgets/demo.html", "title": "Interactive demonstration", "height": 640}
```
````

Only embed trusted HTML: these same-origin frames run JavaScript with access to the site. Generic widgets own their own styling. These helpers do not change existing hand-written iframe embeds or import attention-gym's article content or benchmark assets.

Implementation: `source/quartz/plugins/transformers/docEmbeds.ts`, `source/quartz/components/scripts/docEmbeds.inline.ts`, and `source/quartz/static/widgets/perfetto-trace/`. Embed styles are imported by `source/quartz/styles/custom.scss`.

## Social link previews

Every published page gets a 1200×630 PNG during the normal Quartz build, including the homepage, posts, and tag/folder pages. Cards use the Nuggets mark, a light editorial palette, and the page title above a full-width, post-specific banner. Existing article decals/diagrams are captured where available; other posts have bespoke illustrations. Collection pages and posts without a cover use a branded fallback. The HTML advertises an X/Twitter large-image card and Open Graph title, description, image, dimensions, alt text, and canonical URL. These tags are server-rendered; crawlers don't need JavaScript.

- Set a short, plain-text `description:` in a note's frontmatter to control its preview copy. Without one, cards use `dek`, then Quartz's automatic excerpt. The note body and visible title are not rewritten.
- Set `socialImage: media/social/example.png` and `socialImageAlt:` in frontmatter to select the banner. Paths are relative to the content root, even on nested notes; PNG, JPEG, and WebP are supported. Covers stay local and symlinks cannot escape the content root. See `source/visuals/social/` for artwork sources and capture/render recipes.
- Run `cd source && npm run build` to regenerate. Generated cards live under `source/public/static/social/` and are not committed; selected banner assets under `source/content/media/social/` are committed. The card filename changes with preview copy or cover-image bytes, avoiding stale image URLs; X may still cache the page's metadata.
- The design lives in `source/quartz/plugins/emitters/socialImages.tsx`; metadata and the design-version cache key live in `source/quartz/util/social.ts`. Increment the version when changing the image design. `source/quartz/static/og-image.png` is the branded legacy fallback for older shared links, not the image used by new page cards.
- Satori and resvg render the images at build time. IBM Plex Sans regular/semibold fonts are bundled in `source/quartz/fonts/` with their SIL OFL license, from the [IBM Plex repository](https://github.com/IBM/plex/tree/master/packages/plex-sans/fonts/complete/ttf). Image generation needs no font downloads or system fonts in CI.
- Run `cd source && npm run test:social` for metadata, URL, and PNG-generation checks. Before sharing, inspect the actual composer preview: platform caching/cropping can differ from the source image, and previews aren't guaranteed on every surface.

## Deploy
- Push to `main` to trigger the GitHub Actions build.
- Published site: https://drisspg.github.io/nuggets/

## Analytics
- Minimal page-view analytics are tracked with GoatCounter: https://drisspg.goatcounter.com

## Styling
All site-wide styling is centralized so a single edit propagates everywhere.

- **Base palette** (light + dark mode primitives like `--light`, `--dark`, `--secondary`, etc.) lives in `source/quartz.config.ts` under `theme.colors`.
- **Syntax highlighting theme** (Shiki theme name per mode) lives in `source/quartz.config.ts` under `Plugin.SyntaxHighlighting`.
- **All other design tokens** (surfaces, rules, callouts, accent chips, prose rhythm, shadows) live at the top of `source/quartz/styles/custom.scss` in the `:root` and `:root[saved-theme="dark"]` blocks. The rest of `custom.scss` only references these tokens, never raw hex values.
- Optional third-party themes from [saberzero1/quartz-themes](https://github.com/saberzero1/quartz-themes) can be dropped into `source/quartz/styles/themes/<name>/` and forwarded from `source/quartz/styles/themes/_index.scss`. By default no external theme is forwarded.
- Rebuild locally with `cd source && npx quartz build --serve` to check the new styling before publishing.
