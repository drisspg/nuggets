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
- Published site: https://defenderofbasic.github.io/obsidian-quartz-template
