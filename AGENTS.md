# Nuggets Project Instructions

## Note Migration

- When moving or copying notes into this repo, preserve the note body basically verbatim by default. Do not rewrite, polish, summarize, fix typos, change tone, or retitle the note unless the user explicitly asks for editorial changes.
- Only strip or quarantine clearly private publishing hazards by default, such as Workplace export metadata, private URLs/IDs, access-controlled links, named `cc` lists, and obviously internal-only blocks.
- Call out removed or questionable sections separately so the user can decide what to keep, edit, or publish.
- Keep filename/title changes minimal and source-faithful unless the user asks for a public-facing title.

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
