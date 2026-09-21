# Nuggets Project Instructions

## Note Migration

- When moving or copying notes into this repo, preserve the note body basically verbatim by default. Do not rewrite, polish, summarize, fix typos, change tone, or retitle the note unless the user explicitly asks for editorial changes.
- Only strip or quarantine clearly private publishing hazards by default, such as Workplace export metadata, private URLs/IDs, access-controlled links, named `cc` lists, and obviously internal-only blocks.
- Call out removed or questionable sections separately so the user can decide what to keep, edit, or publish.
- Keep filename/title changes minimal and source-faithful unless the user asks for a public-facing title.

## Social Cover Images

- Give each published post its own cover artwork. Reuse that post's existing decal, diagram, or reference image first; capture a deterministic still from animated widgets. Do not replace distinct article visuals with the same generic diagram on every card.
- If no suitable image exists, create a bespoke, topic-specific banner. Keep it recognizable at thumbnail size, with strong shapes and little embedded text. Illustrative timing bars/charts must not look like new measured results. Preserve original article bodies and titles unless editorial changes are requested.
- Use a 1200×630 final card with the Nuggets mark, a large readable title, and a full-width banner underneath. Maintain the paper/green/amber palette and safe crop margins. No bottom attribution, URL, or divider strip; the top branding is sufficient.
- Artwork sources and capture/render recipes live under `source/visuals/social/`; selected PNG banners live under `source/content/media/social/`. Use `socialImage: media/social/<name>.png`, `socialImageAlt:`, and a concise `description:` in frontmatter. The image path is content-root-relative, even for nested notes.
- From `source/`, run `node visuals/social/render.mjs` for SVG covers; use `visuals/social/capture.cjs` with an ephemeral Playwright install on `NODE_PATH` for existing widget stills. Then run `npm run build`. Generated final cards under `public/static/social/` are not committed. Cover-byte changes automatically invalidate card URLs; increment the design version in `quartz/util/social.ts` when changing the template.
- Inspect a contact sheet of all covers plus the featured card at desktop and phone-feed sizes. Check long titles, cropping, legibility, actual artwork identity, and before/after screenshots. Run `npm run test:social` and verify emitted HTML has absolute Open Graph/X image URLs, large-image-card metadata, image dimensions and alt text; fetch the PNGs as a crawler would, without JavaScript.
- X and other platforms may cache page metadata or crop previews differently. Verify deployed assets and distinguish that from an actual X composer check; don't promise a particular platform rendering without observing it.

## TouchDesigner Motion Graphics Workflow

- TouchDesigner source builders live under `source/visuals/touchdesigner/`; this is a repo-local visual generation workflow, not published blog content by itself.
- Final assets intended for posts live under `source/content/media/touchdesigner/` and can be embedded from notes with HTML video tags.
- Build a visual through the globally configured `twozero_td` MCP server with:
  `cd source/visuals/touchdesigner && python3 tools/run_visual.py visuals/neon_orbit/build.py`
- Record a web-friendly loop with:
  `cd source/visuals/touchdesigner && python3 tools/record.py --recorder /project1/neon_orbit/recorder --seconds 6`
- Publish a rendered asset into Quartz content media with:
  `cd source/visuals/touchdesigner && python3 tools/publish_asset.py dist/neon_orbit/neon-orbit-loop.mp4`
- Embed videos in Markdown notes with:
  `<video controls loop muted playsinline src="./media/touchdesigner/neon-orbit-loop.mp4"></video>`
- Keep heavy/generated TouchDesigner outputs (`dist/**/*.mov`, `.mp4`, `.toe`, `.tox`) out of git unless the user explicitly asks to publish or archive them. Commit the Python builders and selected compressed assets under `source/content/media/touchdesigner/`.

## Playwright Blog QA Workflow

- Self-contained transformer `externalResources()` bundles execute as classic scripts, outside the component `joinScripts` scope isolation. Wrap their compiled bodies in an IIFE and test that execution adds no globals (see the native-chart and code-annotation tests); otherwise independently minified helper names can collide across widgets.

- Use Playwright for rendered-blog checks when editing Quartz layout/components, CSS, navigation, search, graph/sidebar behavior, media embeds, or any note where Markdown rendering matters more than source text.
- Start the local Quartz preview from `source/` with `npm run dev -- --port 8080` or `npx quartz build --serve --port 8080`; the site should be at `http://localhost:8080/`. If port 8080 is busy, pass a different `--port` and use that URL in Playwright.
- This repo does not keep Playwright as a checked-in dependency by default. Prefer ephemeral commands such as `cd source && npx -y playwright@latest screenshot http://localhost:8080/ /tmp/nuggets-home.png` for one-off inspection, or `npx -y playwright@latest codegen http://localhost:8080/` when discovering selectors. Do not add Playwright dependencies or generated config unless the user asks for persistent browser tests.
- For repeatable checks, create an isolated temp Playwright project instead of adding dependencies to this repo: `rm -rf /tmp/nuggets-pw && mkdir /tmp/nuggets-pw && cd /tmp/nuggets-pw && npm init -y && npm install --save-dev @playwright/test@latest`, then put specs there and run `npx playwright test <spec>.ts --browser=chromium`. Keep screenshots, traces, and videos in `/tmp` unless the user explicitly wants artifacts committed.
- Prefer Chromium desktop checks first, then add mobile viewport checks for layout-sensitive work. Verify dark/light theme toggles, search overlay, internal links/backlinks, graph/sidebar rendering, headings/anchors, code blocks, math, and local media embeds when those areas are relevant to the change.
- When using Playwright to validate a content note, open the rendered page and compare visible output against the Markdown intent. For visual/layout bugs, capture before/after screenshots and mention the paths to the saved images in the final response.
