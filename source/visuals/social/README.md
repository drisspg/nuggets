# Social cover artwork

Standalone SVG banners used as the lower, full-width illustration in the 1200x630
social card. The nine original banners listed below are 1080x286, have no text
(the card template supplies the title), and use the site palette: paper `#f4f0e8`, ink `#1a2e22`, sage `#5f7f67`,
amber `#b98049`, muted red `#aa413a`, rule `#d0ccc4`.

These nine SVGs are self-contained (plain shapes, masks, and clip paths; no fonts
or external references). See [captures.md](captures.md) for the existing article
visuals, the pizza and FlexAttention compositions, and regeneration commands.

## covers/

Bespoke banners for posts without a usable existing image (other files in this
directory, e.g. `pizza.svg`, `flex-determinism.svg`, come from post-specific art):

| File | Post | Metaphor |
| --- | --- | --- |
| `two-attentions-split-ribbon.svg` | 2 attentions is all you need | One Q/K/V trunk forks into two ribbons: a dense, fully-ruled sage ribbon (SDPA) and an amber ribbon perforated by a block mask (FlexAttention). |
| `benchmark-trap-stopwatch.svg` | A Small PyTorch Benchmarking Trap | Stopwatch whose sweep has an extra dashed red segment; eight filled dynamo cache slots, a red `cache_size_limit` fence, a ninth slot that misses, and the timing sparkline jumping past the fence with the recompile looping back into the clock. |
| `least-leveraged-lever.svg` | Adding kernels to PyTorch core is the least leveraged thing | Lever on a rounded fulcrum: one amber abstraction pressing the long arm lifts a heavy stack of kernel bricks on the short arm. |
| `pipeline-hang-magnifier.svg` | Debugging pipeline hangs | Five-stage pipeline with tokens queued behind a red stalled token; downstream stages dashed and empty; a magnifier (coredump + cuda-gdb) shows the PC marker on the highlighted wait line with a spinning-wait arc. |
| `intrakernel-nested-spans.svg` | IntraKernel Profiling | Three profiled units, each an outer sage iteration span with nested amber compute / red store spans and ink sub-slices, over a clock-cycle ruler; one dashed red stub marks the atomic-index skew. |
| `micro-from-macro-trace.svg` | Microbenchmarks from Macrobenchmarks | A wide three-lane TorchBench trace ribbon; one amber SDPA call is outlined and funnelled into a small framed microbenchmark card with its shape bars; a `__torch_dispatch__` tap under the ribbon records the inputs. |
| `benchmark-toolkit-roll.svg` | My Python Tools for PyTorch bencmarking | A tool roll holding calipers around a kernel, a p05/p50/p95 dial gauge, a bell curve with median and bootstrap CI bracket, a CUDA-graph replay loop, and a clock lock, over a ruler. |
| `nightly-bisect-timeline.svg` | Utilizing the nightlies for bisect | A timeline of crescent-moon nightlies; binary-search arcs shrink toward the boundary, probes marked red (fail) or sage (pass), with an amber bracket on last-bad / first-good. |
| `qlora-nf4-adapter.svg` | QLoRA in Pure PyTorch | Normal curve with the 16 NF4 quantile levels; a block of weights squeezed into 4-bit nibbles plus an absmax scale chip and its double-quantized scale; frozen ink `W` + amber low-rank `A`/`B` adapter receiving the gradients. |
