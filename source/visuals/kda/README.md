# KDA figures

The training and evaluation figures replot W&B metrics; they are not screenshots of the W&B UI. The snapshot includes no credentials, private dashboard URLs, hostnames, or run metadata.

Regenerate the native chart data from the repository root:

```sh
cd source && npm run charts:kda
```

- `training-loss.json`: full training history for both scaled arms, exported with `scan_history` rather than sampled history. Each arm has all 7,600 steps (1–7,600), with no duplicates or missing steps. `loss_metrics/global_avg_loss` is training cross-entropy averaged over valid tokens across ranks, in nats/token—not evaluation NLL. No smoothing or downsampling.
- `metrics.json`: exported checkpoint metrics at their logged precision. Series names distinguish the model/arm, paired evaluation size, or pilot seed.
- `native-charts.ts`: offline adapter from the snapshot to the shared versioned chart schema. It preserves logged values and SEs, computes ±1-SE error bars, and validates all five specifications before writing.
- `native-charts.test.ts`: exact snapshot/asset parity, independent final interval checks, checkpoint alignment, and preservation of modes, colors, markers, and category order. Runs in `npm test`.
- Outputs under `source/content/media/kda/`: `training-loss.json`, `scaled-loss.json`, `scaled-gap.json`, `paired-checkpoints.json`, and `paired-seeds.json`.

The article uses the reusable `chart` fenced embed described in `source/content/Native Charts.md`. Charts render directly in the page with D3/SVG and site CSS, without iframes or a Plotly CDN request. Hover, touch, and keyboard inspection update both-axis dashed guides and the legend; its tooltips expose full-precision values and uncertainty metadata. Ordinary wheel gestures scroll the article without forwarding messages between frames.

The old Plotly `.html` exports, `render.py`, and `frame.html` remain available for comparison and existing embeds. They retain their original ±1.96-SE intervals, unlike the native charts' ±1-SE bars. Regenerate those separately with `uv run --with plotly==7.0.0 python source/visuals/kda/render.py`. Their iframe-height, theme synchronization, and wheel-forwarding machinery are not used by the new native charts.

## Gate-factor range

`rebase-range.ts` generates the analytic `content/media/kda/rebase-range.json`, separately from the measured training/evaluation data. Regenerate it with `cd source && npx tsx visuals/kda/rebase-range.ts`.

For a first-row reference and every natural-log gate at the limiting value −5, an N-token window spans N−1 steps. The plotted exponents are ±5(N−1)log₂(e), not measured kernel outputs. The FP32 limits are chart `references` barriers, not series: dotted rules labelled “FP32 overflow (+128)” and “FP32 subnormal boundary (−126)” with no legend entry or readout; crossing the lower line means subnormal values, not necessarily zero. Point details include the unrounded factor and its FP32 cast with gradual underflow (kernels may flush subnormals earlier). Tests check asset parity, indexing, and FP32 overflow/underflow boundaries.

## TCGEN instruction-width measurement

`content/media/kda/tcgen-width-b200.json` records a separate instruction microbenchmark, not training data or end-to-end KDA performance. It uses the unchanged [upstream probe at `6cb0ca2`](https://github.com/drisspg/transformer_nuggets/blob/6cb0ca2687d259f699a6b4b98e252332305761f6/benchmarks/tcgen05_throughput.py), selecting its M64/N16 and M64/N32 cases with a common 128×128×128 work envelope. The source tile's K128 lowers into eight physical K16 instructions per panel. The selection was:

```python
probe.SWEEPS = (
    probe.Sweep(
        "kda_width",
        (probe.Shape(64, 16, 128, 128), probe.Shape(64, 32, 128, 128)),
    ),
)
```

- Hardware: one NVIDIA B200, 148 SMs, driver 615.71.09; PyTorch 2.15.0.dev20260920+cu132, CuTeDSL 4.7.1. Normal DVFS, no clock/power overrides. The sampled SM clock stayed at 1965 MHz through the last three seconds of warmup and both measured passes.
- Contract: zero-filled BF16 SMEM operands, FP32 accumulation, fixed-pointer CUDA Graph replay. This is a low-switching instruction-throughput result, not a cold-memory GEMM or realistic-input roofline. Each CTA uses 128 threads and 256 TMEM columns. The probe checks completion sentinels, not numerical TMEM results.
- Warmup: six seconds alternating long-repeat graphs, then the upstream five-call capture warmups. Two 31-round passes alternate/rotate measurement order. Repeat counts are 16, 64, 256, 512; a fitted slope removes the launch/setup intercept.
- Both shapes use the preselected **2×SM grid, 296 CTAs**, for the article comparison. The 1×/4× grids are sensitivity checks, not independently selected winners. Primary slopes are 3.1935847486702458 µs (N16) and 1.6660429229987248 µs (N32), a 1.916868× ratio. Confirmation gives 1.916754×. All grid fits and the primary-grid raw timing samples are retained in the JSON.
- Each work tile is `2 * 128 * 128 * 128` physical FLOPs per CTA/repeat. Throughput is `blocks * FLOPs / slope_us / 1e6` TFLOP/s. The implied amortized issue interval per SM is `slope_us * 1000 / (grid_multiplier * instructions_per_CTA_repeat)`: 12.475 ns versus 13.016 ns, not isolated instruction latency.
- Native SASS has 128 versus 64 UTCHMMA sites in the repeat body, before the completion commit/wait and sentinel store. Descriptors are `0x04042490` (M64/N16) and `0x04082490` (M64/N32); both use K16. Both kernels report 10 registers/thread, zero stack/local memory, with 20/24 KiB of operand SMEM respectively.

The measurements establish an instruction-width benefit. They do not show a 1.92× KDA/model speedup, validate a midpoint32 implementation, or remove triangular masked work.

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

Error bars are the estimate ± one **logged standard error**: sequence-level for arm gaps, paired for differences of gaps. The SE estimates uncertainty in the mean from variation across held-out documents, not variation across training seeds. These bars are not 95% confidence intervals; excluding zero does not establish significance at that level. The adapter neither reconstructs per-sequence samples nor recomputes standard errors.

No smoothing, interpolation of missing checkpoints, or averaging across training seeds is used. Connecting lines on the 64-sequence sweep only guide the eye between measured checkpoints. The precise private run-to-series mapping is retained in the agent-notes blog handoff rather than embedded in public assets.
