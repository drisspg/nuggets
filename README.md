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

## Theme Switching
- Theme overrides live in `source/quartz/styles/custom.scss`, which now imports `source/quartz/styles/themes/_index.scss`.
- Each theme from [saberzero1/quartz-themes](https://github.com/saberzero1/quartz-themes) belongs in its own folder under `source/quartz/styles/themes/` (e.g., `golden-topaz/_index.scss`).
- Switch themes by editing `source/quartz/styles/themes/_index.scss` to forward the desired folder (currently `@forward "./golden-topaz/_index.scss";`).
- Rebuild locally with `cd source && npx quartz build --serve` to check the new styling before publishing.
