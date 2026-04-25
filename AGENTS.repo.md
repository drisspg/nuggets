# Repository Guidelines

## Project Structure & Module Organization
- Root hosts repo-level metadata (`README.md`, `LICENSE`, `.github/workflows/ci.yaml`).
- Site sources live under `source/`: Quartz framework code in `quartz/`, configuration in `quartz.config.ts` & `quartz.layout.ts`, and deployable assets in `quartz/static/` (`icon.png`, `og-image.png`).
- Content you publish belongs in `source/content/` (Markdown) or `source/raw_html/` (custom embeds). Keep new images beside the content that uses them.
- Obsidian-specific settings remain in `.obsidian/`; do not edit unless you coordinate with note authors.

## Build, Test, and Development Commands
- `cd source && npx quartz build --serve` – build and serve the site locally with live reload at the printed URL.
- `cd source && npm run check` – TypeScript type check and Prettier verification.
- `cd source && npm test` – run the Quartz unit tests via `tsx`.
- `cd source && npm run format` – apply Prettier formatting to the entire codebase.
- Prefer `npm install` and the checked-in `package-lock.json`; remove `pnpm-lock.yaml` leftovers instead of reintroducing pnpm.

## Coding Style & Naming Conventions
- TypeScript and JSX files follow 2-space indentation (enforced by Prettier).
- Favor descriptive PascalCase for components (e.g., `Graph.tsx`) and camelCase for utilities and configuration keys.
- Store static assets under `source/quartz/static/`; name favicons `icon.png`, hero images `og-image.png`, and keep additional assets lowercase with hyphens (e.g., `graph-darkmode.svg`).
- Run `npm run format` before committing to avoid CI rejections.

## Testing Guidelines
- Unit tests live alongside implementation within `source/quartz/**` (e.g., `quartz/util/path.test.ts`).
- Name new test files with a `.test.ts` suffix and mirror the directory of the code under test.
- Execute `npm test` before opening a PR; aim to cover new branches or configuration toggles you introduce.
- If a feature relies on Markdown rendering, add a representative note in `source/content/` and reference it in PR notes instead of relying solely on screenshots.

## Commit & Pull Request Guidelines
- Write imperative commit messages (“Add custom favicon”) and group related changes per commit.
- Rebase onto `main` before opening a PR to keep history linear.
- Every PR description should include: summary of visible changes, testing evidence (command output or steps), and impacted content paths.
- Link to tracking issues when applicable, and attach screenshots or GIFs for visual changes to the rendered site.

## Security & Configuration Tips
- Keep environment-specific secrets out of the repo; GitHub Pages deployment uses the existing Action in `.github/workflows/ci.yaml`.
- When adding analytics or third-party embeds in `quartz.config.ts`, document required keys in `README.md` and gate them with environment variables where feasible.
