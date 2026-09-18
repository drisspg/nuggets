---
title: Native Charts
draft: true
---

Native charts are SVGs inside the article, not iframes. They use the blog's fonts and theme, keep ordinary page scrolling, and support line plots, points, and horizontal or vertical uncertainty intervals. The same embed works in any note; no frontmatter class is needed. Existing `plotly` embeds remain supported.

## Add a chart

Save a versioned JSON specification under `source/content/media/`, then reference it from a note:

````md
```chart
{"src":"media/charts/experiment.json","title":"Held-out loss difference","height":300}
```
````

`src` is relative to the **content root**, including for notes in subfolders. Only local `media/*.json` paths are accepted; traversal, remote URLs, and symlinks escaping the content root are rejected. `title` is required and also names the chart for assistive technology. `height` is the SVG plot height in pixels (default 300, range 180–800); captions, controls, and data inspection use normal page flow outside it.

Everything in the data file is public and downloadable. Do not include credentials, private run URLs, or internal-only metadata.

Example `experiment.json`:

```json
{
  "version": 1,
  "x": { "label": "Training step" },
  "y": {
    "label": "Δ loss (nats/token)",
    "format": "scientific",
    "includeZero": true
  },
  "intervalLabel": "95% confidence interval supplied by the experiment",
  "series": [
    {
      "name": "Candidate − baseline",
      "mode": "line",
      "color": "green",
      "points": [
        { "x": 1000, "y": 0.04, "yLow": 0.01, "yHigh": 0.07 },
        { "x": 2000, "y": 0.02, "yLow": -0.01, "yHigh": 0.05 }
      ]
    }
  ]
}
```

These example numbers are illustrative. The renderer draws supplied interval endpoints; it does **not** infer standard errors, confidence levels, or statistical significance.

## Data contract

- `version` must be `1`. Unknown fields are errors, so misspelled options do not silently disappear.
- Axes have a nonempty `label`. Numeric axes optionally accept `format: "number" | "scientific"`, `includeZero`, and `domain: [minimum, maximum]`.
- Automatic domains include **all series and interval endpoints**, with 5% padding. Hiding a series does not silently rescale the chart. An explicit initial domain must contain all values, intervals, and any requested zero. Interactive zoom changes only the viewport: data and interval endpoints remain untouched.
- Each series has a unique `name`, a `mode` (`"line"` or `"points"`), and a nonempty `points` array. A series must contain at least one observation.
- Optional series styling: `color` is `blue`, `amber`, `green`, `red`, `gold`, or `purple`; `dash` is `solid` or `dash`; `marker` is `circle` or `diamond`. Colors adapt to the site theme. Use dashes/markers as well as color when comparing related series.
- Points have numeric `x` and numeric `y`. A numeric `y: null` is an explicit missing observation: it gets no marker and **breaks** a line instead of becoming zero or being interpolated across. Do not simply omit a missing point if the line must show a gap. Missing observations cannot carry interval bounds.
- Line-series x values must be strictly increasing. Points-only series can be unordered. Curves use straight segments, with no smoothing or resampling.
- Intervals use `xLow`/`xHigh` or `yLow`/`yHigh`. Both endpoints are required, must be finite, and must enclose the estimate. Any interval requires an `intervalLabel` explaining its meaning. Numeric strings, `NaN`, and infinity are rejected.
- Optional per-point `details` is a map of plain strings or finite numbers. It appears in inspection; it is never interpreted as HTML.

### Categorical rows / forest plots

Give the y axis a `categories` array instead of numeric options. The order is top to bottom, and points use a category string as `y`. Category charts require `mode: "points"`; horizontal intervals remain numeric:

```json
{
  "version": 1,
  "x": { "label": "Difference", "includeZero": true },
  "y": { "label": "Model", "categories": ["Small", "Large"] },
  "intervalLabel": "95% confidence interval supplied by the experiment",
  "series": [
    {
      "name": "Model comparison",
      "mode": "points",
      "color": "purple",
      "points": [
        { "x": 0.1, "y": "Small", "xLow": -0.1, "xHigh": 0.3 },
        { "x": 0.05, "y": "Large", "xLow": -0.05, "xHigh": 0.15 }
      ]
    }
  ]
}
```

## Interaction and accessibility

- Hover over the plot to inspect the nearest x coordinate, or the nearest categorical row. Dashed guides mark both coordinates of each inspected point (shared values reuse one guide). Inspection updates larger tabular numbers **directly in the legend**, with subtle series-colored underlines rather than boxes. An unsampled coordinate is shown as `—`; an explicit null observation is shown as `Missing`. Neither is interpolated.
- Dense line charts retain every vertex and inspectable value, but omit overlapping point markers. Markers reappear when the visible points have enough horizontal space; points-only series always retain their markers.
- Legend values use seven significant digits for compact display. Legend tooltips and accessible names retain the full-precision estimates, interval endpoints, and details. The source JSON retains the original numbers.
- Click or tap to **pin** the selection: moving the pointer no longer changes the crosshair or values. Click again to release it. A small top-right icon switches between cursor-following and pinned states; clicking it also toggles the pin. Pins survive resize and theme changes; hiding all observations at the pinned coordinate clears the pin.
- Focus the plot and use arrow keys, Home, or End to inspect observations, including while pinned. Enter or Space toggles pinning. Escape releases the pin, clears inspection, and returns keyboard scrolling to the page without changing its scroll position.
- Legend buttons toggle series and expose their state through `aria-pressed`. At least one series stays visible.
- Charts have no table or download controls in their normal view. Long category labels are abbreviated on the axis but remain complete in inspection.
- A source-data link appears only as a fallback if JavaScript is disabled or a runtime load fails. Invalid data fails the build; runtime failures display an error rather than an empty, apparently valid plot.
- The plot never captures wheel scrolling, even when pinned, or animates/interpolates the data. Updated legend values have a brief opacity accent; reduced-motion preferences disable it and the underline/status transitions.

### Zoom and pan

- **Drag a box** to zoom into that region. A nearly horizontal drag changes only X; a nearly vertical drag changes only Y. Categorical/forest charts zoom horizontally while preserving row order.
- **Shift-drag** to pan the selected window. Panning stays within the chart's initial full extent.
- **Double-click**, press **`0`**, or use the top-right reset icon to restore the full view. With the plot focused, **`+` / `-`** zoom around the view center and **Shift + arrow keys** pan.
- On touchscreens, activate the magnifier tool before dragging a selection. It returns to normal page scrolling after the selection; Escape or toggling the tool cancels it. Ordinary touch swipes and wheel gestures never zoom implicitly.
- A drag releases point pinning, and Escape cancels an in-progress drag without scrolling the article. View ranges survive resize and theme changes, but reset on page navigation.
- Lines and uncertainty bars are clipped at the viewport boundary, not discarded or recomputed. Inspection uses point centers in view (and explicit gaps). A region without point centers remains resettable rather than snapping to an off-screen point. Zoom is capped at one-millionth of the full span to keep projections numerically bounded.

Version 1 intentionally has no logarithmic/date axes, stacking, smoothing, animated data, or Plotly toolbar. Use the existing Plotly embed when those capabilities are needed. Do not add per-post rendering code for a feature that belongs in the shared renderer.

## KDA data and regeneration

The KDA article's five JSON files are derived from two metadata-free snapshots under `source/visuals/kda/`: `metrics.json` for checkpoint evaluations and `training-loss.json` for the full training history:

```sh
cd source && npm run charts:kda
```

The training-loss chart includes all 7,600 logged steps per arm, without smoothing or downsampling. These are training cross-entropies, not held-out evaluation losses.

The adapter preserves the logged values, paired versus sequence-level standard errors, and intervals computed as estimate ± 1.96 × logged SE. The two 1,024-sequence observations stay points-only; the four final-checkpoint categories keep their original order. Old Plotly exports and their Python renderer remain available for comparison, but KDA now embeds the native JSON files.

## Validation

From `source/`:

```sh
npm run test:charts
npm run build
```

The chart tests are also included in `npm test`. They cover schema/domain edge cases, real Markdown transforms, safe paths and escaping, and KDA data fidelity. Generation is checked against committed JSON, so stale assets fail tests.

Browser regression tests build isolated fixtures and run their own temporary server. They cover theme and resize behavior, error bars, keyboard/touch inspection, legend toggling, ordinary scrolling, missing data, unsafe text, failed fetches, nested deployed paths, and repeated SPA navigation. They do not restart the user's preview or add browser dependencies to this repo.

Use an isolated Playwright installation (or point `NODE_PATH` at one you already have):

```sh
pw_dir=$(mktemp -d) && npm install --prefix "$pw_dir" --no-save playwright@1.57.0 && "$pw_dir/node_modules/.bin/playwright" install chromium && NODE_PATH="$pw_dir/node_modules" npm run test:charts:browser
```

Screenshots and fixture output are retained in the printed temporary directory. Re-run both behavioral suites when changing the schema, renderer, or generation logic; a successful build alone does not check interaction or scientific fidelity.

Implementation: `quartz/util/chart.ts` (contract/domains), `quartz/util/chartViewport.ts` (bounded viewport arithmetic), `quartz/plugins/transformers/nativeCharts.ts` (Markdown/build validation), `quartz/components/scripts/nativeCharts.inline.ts` (D3 and lifecycle), and `quartz/components/styles/nativeCharts.scss` (scoped theme styling).
