# Nuggets Blog

Static site for publishing Obsidian notes through Quartz and GitHub Pages.

## Edit Content
- Update Markdown notes in `source/content`.
- Place any hand-written HTML in `source/raw_html`.

## Preview Locally
```bash
cd source
npx quartz build --serve
```
Open the printed URL to browse the site.

## Deploy
- Push to `main` to trigger the GitHub Actions build.
- Published site: https://drisspg.github.io/nuggets/

## Styling
All site-wide styling is centralized so a single edit propagates everywhere.

- **Base palette** (light + dark mode primitives like `--light`, `--dark`, `--secondary`, etc.) lives in `source/quartz.config.ts` under `theme.colors`.
- **Syntax highlighting theme** (Shiki theme name per mode) lives in `source/quartz.config.ts` under `Plugin.SyntaxHighlighting`.
- **All other design tokens** (surfaces, rules, callouts, accent chips, prose rhythm, shadows) live at the top of `source/quartz/styles/custom.scss` in the `:root` and `:root[saved-theme="dark"]` blocks. The rest of `custom.scss` only references these tokens, never raw hex values.
- Optional third-party themes from [saberzero1/quartz-themes](https://github.com/saberzero1/quartz-themes) can be dropped into `source/quartz/styles/themes/<name>/` and forwarded from `source/quartz/styles/themes/_index.scss`. By default no external theme is forwarded.
- Rebuild locally with `cd source && npx quartz build --serve` to check the new styling before publishing.
