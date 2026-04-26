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
