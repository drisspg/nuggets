# KDA evaluation figures

These figures replot selected W&B evaluation metrics; they are not screenshots of the W&B UI. The snapshot includes no credentials, private dashboard URLs, hostnames, or run metadata.

Regenerate from the repository root:

```sh
uv run --with plotly==7.0.0 python source/visuals/kda/render.py
```

- `metrics.json`: exported checkpoint metrics at their logged precision. Series names distinguish the model/arm, paired evaluation size, or pilot seed.
- `render.py`: offline figure generation plus checks for checkpoint alignment, finite values, nonnegative standard errors, and agreement between logged gaps and differences of logged NLLs.
- `frame.html`: IBM Plex typography, Attention Gym docs' light/slate palettes, and hover-only modebar styling. Quartz's existing Plotly embed helper themes the chart chrome; the frame themes trace colors.
- Outputs under `source/content/media/kda/`: `scaled-loss.html`, `scaled-gap.html`, `paired-checkpoints.html`, and `paired-seeds.html`.

The article uses Quartz's existing `plotly` fenced embeds. Generation does not access W&B; viewing the HTML loads the versioned Plotly bundle and IBM Plex fonts from CDNs.

The embed owns the outer height. The exported plot fills the frame's usable height instead of fixing a second pixel height, which would overflow by the iframe border thickness. Ordinary wheel gestures use the existing `nuggets-widget-wheel` message to scroll the article; Ctrl/Cmd-wheel is left alone. Browser checks should compare both frame scroll dimensions with its viewport, not just check page-level horizontal overflow.

## Training versus inference

The article's `kda-token-modes` figure is a static inline SVG comparison, styled in `source/quartz/components/styles/kdaFigures.scss`. Training computes three next-token distributions together. Solid green arrows show the allowed causal prefixes; three red dashed arrows show all forbidden future-token influences in the example: x2 → p2, x3 → p2, and x3 → p3. Inference shows three successive prefixes: a distribution supplies the next sampled token before the next prediction. Dashed empty slots are unavailable future tokens. Inference uses red bars with different shapes to illustrate a possible prediction mismatch when future information used during training is absent. The bars are illustrative distributions, not experimental data or evidence that KDA learned to exploit leakage. Only the panel names and token/distribution symbols are visible; accessible descriptions explain the steps. Panels stack on mobile. This replaces the older, word-heavy prediction/loss boundary diagram.

## MatX source figure

`source/content/media/kda/matx-leaky-region.svg` is the original, unmodified figure from MatX's [Future leakage in block-quantized attention](https://matx.com/research/leaky_quantization) (January 9, 2026), downloaded from [the original SVG](https://matx.com/static/leaky_region_diagram.svg). The article credits and links MatX in its caption. SHA-256: `702b5746f92cc9e1ebfde261c914de235094c48c89ba411679f74f04024e7719`. Its black labels require a white image background in both themes; the source file itself is unchanged. The accompanying explanation distinguishes the risky diagonal blocks from the per-entry causal mask and connects shared-scale value quantization to KDA's different, gate-reference mechanism.

## Cover and reaction image

`source/content/media/kda/kda-future-animation.html` is the standalone canvas cover, styled after the PTQ dispatch-board animation. Two rows of 16 vector-like blocks surround an orange current-position pivot. The pivot aligns with the selected token column, highlighted in both rows. Selection starts at token 8 and sweeps back and forth with a short glide. Clicking a column or using Arrow keys/Home/End holds that selection for one 3.4-second interval before the sweep resumes from the chosen token. Focus alone does not stop the sweep. Arrows represent reference lookups: green toward earlier positions, red toward later ones, and orange for the selected position. This is conceptual, not measured data. The figure has no visible wording, caption, or play/pause button. An accessible canvas description remains; Space pauses or resumes the animation when the canvas is focused. It uses a `doc-widget widget-frame` iframe, follows the blog theme, and starts paused for reduced-motion preferences. The rejected static SVG cover and its generator were removed.

`source/content/media/kda/doc-brown-future.png` is the user-supplied reaction image, preserved unchanged. It is revealed from the “The mask does not fix this” phrase on hover or focus rather than displayed permanently in the article. Escape dismisses it; leaving and re-entering or refocusing can reveal it again.

## Measurement and uncertainty

NLLs and gaps are in nats/token, without rescaling. Gap axes use scientific notation; hover text shows the original decimal values and logged standard errors.

The scaled arm sweeps use 64 held-out sequences, evaluation length 256, and recurrent-KDA autoregressive evaluation. They cover eight checkpoints from step 1000 through 7600. The 1024-sequence paired scaled run has only checkpoints 4000 and 7600. The pilot comparisons have checkpoints 2000 and 4000, for seeds 42, 11, and 23.

Error bars are the estimate ± 1.96 times the **logged standard error**: sequence-level for arm gaps, paired for differences of gaps. These are approximate pointwise 95% intervals, not simultaneous intervals across checkpoints and not variation across training seeds. The script neither reconstructs per-sequence samples nor recomputes standard errors. The original analysis should be checked before publication.

No smoothing, interpolation of missing checkpoints, or averaging across training seeds is used. Connecting lines on the 64-sequence sweep only guide the eye between measured checkpoints. The precise private run-to-series mapping is retained in the agent-notes blog handoff rather than embedded in public assets.
