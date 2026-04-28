# TouchDesigner visual workflow

Repo-local source workflow for generating motion graphics that can be embedded in Nuggets posts.

This folder is not a standalone post. It contains reusable visual builders and render helpers.

## Build

TouchDesigner must be running with twozero MCP listening at `http://localhost:40404/mcp`.

```bash
cd source/visuals/touchdesigner
python3 tools/run_visual.py visuals/neon_orbit/build.py
```

## Record

```bash
python3 tools/record.py --recorder /project1/neon_orbit/recorder --seconds 6
```

Generated files go to `dist/`. Heavy generated files are gitignored.

## Publish an asset for embedding

```bash
python3 tools/publish_asset.py dist/neon_orbit/neon-orbit-loop.mp4
```

This copies the compressed clip to `source/content/media/touchdesigner/`.

## Embed in a note

```html
<video controls loop muted playsinline src="./media/touchdesigner/neon-orbit-loop.mp4"></video>
```

Use `autoplay loop muted playsinline` for hero/background loops.
