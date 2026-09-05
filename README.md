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
