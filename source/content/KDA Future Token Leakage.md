---
title: KDA Doesn't Care About the Future
date: 2026-09-15
enableToc: false
dek: A future-dependent rounding effect in KDA, and the experiments that separated numerical leakage from learned exploitation.
tags:
  - pytorch
cssclasses:
  - sidenotes
---

#### Written: September 15, 2026

<iframe class="doc-widget widget-frame" src="./media/kda/kda-future-animation.html" title="Conceptual reference directions from an orange midpoint: green toward earlier positions, red toward later positions." loading="lazy" style="height: 340px;"></iframe>

> [!note] Working draft
> This is a first pass through the experiment story. The evaluation figures use exported W&B metrics; the kernel and probe results come from my experiment notes. Public code links, the remaining probe figures, and full reproducibility details still need to be attached.

I was working on the intra-chunk kernels for Kimi Delta Attention (KDA) in [Attention Gym](https://github.com/meta-pytorch/attention-gym) and ran into something odd: an expression that is causal on paper was not quite causal on the GPU. I used [TorchTitan](https://github.com/pytorch/torchtitan) to test whether the model could learn to exploit it.

Changing a future token's gate could change an earlier output. The causal mask was still there. No future value vector had been accidentally included in the sum. The dependence came from a reference value used to rescale the operands before a matrix multiplication.

That led to two different questions:

1. Can the implementation's forward pass depend on future inputs?
2. Can training learn to use that dependence to predict future tokens?

For the midpoint-reference variant I tested, the first answer was **yes**. For the training experiments I ran, the second was **no exploitation detected**. Those are different results, and the difference is most of the story.

## What has to be causal?

For next-token prediction, the logits at position $t$ should depend on tokens $x_1,\ldots,x_t$, not on the tokens after them. Hold the model weights and valid past state fixed: changing the suffix should not change that prediction. Computing all positions in parallel does not relax this condition.

The target $x_{t+1}$ does enter the calculation—but **at the loss**, where we score the prediction, not upstream where we make it.

<figure class="kda-figure" aria-labelledby="kda-causality-title">
<div class="kda-figure-title" id="kda-causality-title">The boundary is the prediction, not the end of training</div>
<div class="kda-causal-row">
<div class="kda-node"><span class="kda-label">Available context</span><strong>Prefix x₁ … xₜ</strong>Fixed weights and state built from the allowed past.</div>
<div class="kda-node"><span class="kda-label">Forward</span><strong>Predict xₜ₊₁</strong>The logits must not depend on this sequence's future tokens.</div>
<div class="kda-node"><span class="kda-label">Loss</span><strong>Score the prediction</strong>The true target xₜ₊₁ is used here.</div>
</div>
<div class="kda-forbidden"><strong>Forbidden shortcut:</strong> current future tokens → a scale or reference → the current prediction.</div>
<div class="kda-backward"><strong>Different dependency rule after the loss:</strong> losses across positions → backward reductions → parameter update for subsequent training steps.</div>
<figcaption>A causal mask blocks direct attention to future positions. It cannot undo future information that already entered through operand preparation.</figcaption>
</figure>

### Why backward is different

The gradient of the training loss is allowed to combine information from different token positions. A weight-gradient reduction across tokens is not, by itself, a future leak into the forward prediction. The same distinction applies when discussing token-spanning quantization in backward: the important question for forward causality is whether the logits used to compute the loss already saw a forbidden signal.

That is **not** a license for arbitrary gradient error. Backward quantization can still introduce bias, noise, or instability, and must be validated on those terms. I would not turn this into a general rule that numerical bias is fine in forward but dangerous only in backward.

Delayed scaling from a previous batch is also a different dependency from reading this sequence's future. It may use permitted history, but that depends on the data order and the scale-state policy available at inference. “The optimizer already saw that batch” is not a proof that carrying extra state is redundant or harmless. Training–serving mismatch is the broader concern; forward future leakage is one way to create it, not a name for every numerical mismatch.

### The MatX example

MatX's [“Future leakage in block-quantized attention”](https://matx.com/research/leaky_quantization) gives a concrete example. When values share a quantization scale across token positions, a future outlier can increase that scale and make an earlier value underflow. An earlier query can still attend to that altered value even though future attention weights are masked. The future information arrived through the scale, not through an unmasked attention edge.

Their fix uses unquantized probabilities and values for the block-diagonal part of $PV$, while keeping block-quantized multiplications elsewhere. In two 1B-parameter C4 models using MXFP4 in attention and its gradients, they report parallel/autoregressive losses of **2.56/2.66** for the leaky model and **2.64/2.64** for the fixed model. The apparent parallel-mode advantage reversed when future tokens were unavailable.

That motivated the question here, but these are not matched experiments. MatX does not state a total training-token budget in the post. My scaled run used 1.45B parameters and 4B C4 tokens, and the numerical mechanism is different: **gate-reference rounding**, not block quantization of values. The experiment has to establish separately whether that different channel is usable by training.

## The forward pass, in broad strokes

KDA carries a recurrent state from earlier tokens. The chunked implementation reorganizes that recurrence into local matrix operations plus a state update across chunks. Here is the logical flow for the **64-token chunked path**—not a claim that each box is a separate GPU launch. Fusion and scheduling can combine these stages.

<figure class="kda-figure" aria-labelledby="kda-flow-title">
<div class="kda-figure-title" id="kda-flow-title">From Q/K/V inputs to chunk outputs</div>
<div class="kda-inputs"><span class="kda-label">Inputs</span><span>Q, K, V</span><span>per-token log gates</span><span>write gate β</span><span>state from earlier chunks</span></div>
<ol class="kda-flow">
<li class="kda-node"><span class="kda-label">01 · gates</span><strong>Prepare decay</strong>Accumulate log gates within each chunk and convert to base 2.</li>
<li class="kda-node kda-node--focus"><span class="kda-label">02 · local terms</span><strong>Build pairwise factors</strong>Compute Aqk and gated K/K interactions. <b>The reference choice enters here.</b></li>
<li class="kda-node"><span class="kda-label">03 · local solve</span><strong>Build W and U</strong>Use the lower-triangular system to express the local delta-rule updates.</li>
<li class="kda-node"><span class="kda-label">04 · recurrence</span><strong>Propagate state</strong>Combine local factors with the incoming state; produce corrected values and the next state.</li>
<li class="kda-node"><span class="kda-label">05 · outputs</span><strong>Compose token outputs</strong>Add the history contribution to Aqk times the corrected values.</li>
</ol>
<div class="kda-bypass"><strong>Two paths meet at the output:</strong> Aqk from stage 2 goes directly to output composition. The state path carries information from earlier chunks.</div>
<div class="kda-label">Inside one 64-token chunk: four 16-token diagonal subchunks</div>
<div class="kda-chunks"><span>0–15</span><span>16–31</span><span>32–47</span><span>48–63</span></div>
<figcaption>The experiment changes the gate reference used to build local factors. The causal mask, triangular solve, and state propagation are still present; a numerical dependency introduced upstream can survive them.</figcaption>
</figure>

The key distinction is between **local interactions within a chunk** and the **state inherited from previous chunks**. The output uses both. This map follows Attention Gym's [composed forward path](https://github.com/meta-pytorch/attention-gym/blob/52c9eaa31e87a28dc0d3d9464af2088e6384a483/attn_gym/linear/kda/fwd/cute/chunk_kda_fwd.py); it is an implementation map, not a record of the exact launch schedule of every training run.

## The Aqk step

This is not a derivation of all of KDA. I want to isolate the query/key interaction inside a chunk, which I'll call $A_{qk}$.

Let $q_i$ and $k_j$ be query and key vectors with $D$ channels. Let $g_{i,d}$ be the cumulative log-base-2 gate at token $i$, channel $d$, measured within the chunk.<span class="sidenote-ref" aria-hidden="true"></span><span class="sidenote" role="note">Attention Gym's KDA API uses natural-log gates. At the $-5$ floor—the strongest decay allowed by the tested gate transform—the multiplier is $e^{-5}\approx0.006738$: only about 0.67% of the previous state is retained before the other update terms.</span> Ignoring the usual query scaling, the causal entries are:

$$
A_{qk}[i,j] = \sum_{d=1}^{D} q_{i,d} k_{j,d} 2^{g_{i,d}-g_{j,d}}, \qquad j \leq i.
$$

Entries with $j>i$ are masked to zero. Because the per-token log gates are nonpositive, the cumulative gates decrease: for $j\leq i$, the decay factor is at most one.

The awkward part for a fast kernel is that the decay depends on the channel $d$. It lives **inside** the reduction. This is not a plain $QK^T$ followed by one scalar decay per matrix entry.

## The rebasing trick

Pick a reference gate $r_d$ for each channel and split the decay:

$$
2^{g_{i,d}-g_{j,d}} = 2^{g_{i,d}-r_d}\,2^{r_d-g_{j,d}}.
$$

Now define rescaled operands:

$$
\widetilde Q_{i,d}=q_{i,d}2^{g_{i,d}-r_d},
\qquad
\widetilde K_{j,d}=k_{j,d}2^{r_d-g_{j,d}}.
$$

The block can be computed as $\widetilde Q\widetilde K^T$, then causally masked. We have turned a channel-dependent decay into operand preparation plus a matrix multiplication.

In real arithmetic, the reference cancels. Pick the first row, pick the midpoint: same answer.

On the GPU, we materialize those operands separately. Exponent evaluation, multiplication, conversion, and the matrix multiplication all have finite precision. The cancellation is no longer an identity between the computed values. **The reference becomes part of the numerical result.**<span class="sidenote-ref" aria-hidden="true"></span><span class="sidenote" role="note">The BF16 CuTe engine casts the rescaled operands to BF16 before its matrix multiplication, with FP32 accumulation. The Triton diagonal kernel passes FP32 products to <code>tl.dot</code>; its effective dot precision needs to be checked separately. BF16 inputs do not imply identical rounding paths.</span>

## How a midpoint breaks causality

For a full 16-token diagonal subchunk, the midpoint reference is the gate at local row 8, using zero-based indexing:

<figure class="kda-figure" aria-labelledby="kda-pivot-title">
<div class="kda-figure-title" id="kda-pivot-title">One subchunk: row 8 supplies the midpoint reference</div>
<ol class="kda-tokens" aria-label="Local token rows zero through fifteen">
<li class="kda-token--early">0</li><li class="kda-token--early">1</li><li class="kda-token--early">2</li><li class="kda-token--early">3</li><li class="kda-token--early">4</li><li class="kda-token--early">5</li><li class="kda-token--early">6</li><li class="kda-token--early">7</li><li class="kda-token--pivot">8</li><li>9</li><li>10</li><li>11</li><li>12</li><li>13</li><li>14</li><li>15</li>
</ol>
<div class="kda-forbidden"><strong>Rows 0–7:</strong> operand scaling reads the cumulative gate at row 8, which includes future gate increments.</div>
<figcaption>Row 8 is highlighted as the reference. It is already available to rows 8–15, but is in the future for rows 0–7. The gate changes the rounding of retained causal entries; this is not an unmasked read of future values.</figcaption>
</figure>

For query rows 0–7, that reference depends on future gate increments. Changing those increments changes the scale factors, which can change the rounding of an otherwise causal dot product.

<span class="sidenote-hover"><button type="button" class="sidenote-trigger" aria-describedby="kda-mask-reaction">The mask does not fix this.</button><span id="kda-mask-reaction" class="sidenote sidenote--hover-media" role="note"><img src="./media/kda/doc-brown-future.png" alt="Doc Brown staring in disbelief." width="640" height="360" loading="lazy"><span class="sidenote-hover-caption">The causal mask was there the whole time.</span></span></span> It removes matrix entries with future key indices. It does not remove a future-dependent scale already used to compute a retained entry.

The test is simple: keep the tensor shape and a prefix fixed, change only suffix gate increments, and compare the prefix outputs. Keeping the shape fixed matters: shortening a sequence can also change dispatch, tiling, or reduction order.

An early isolated probe changed future gates in rows 8–15 while holding rows 0–7 fixed. It changed **11 of 512 BF16 Aqk values**, with a maximum absolute difference of **1.526e-5**. A later GB200 regression through the public `chunk_kda` path, perturbing future Q/K/V and gates together, recorded **851 of 9472 prefix output elements** changing, with maximum absolute difference **2.44e-4**. The causal-reference variant passed the bitwise prefix check.

These are different probes, not two measurements of the same tensor. Their magnitudes are workload-specific. The important result is that the numerical dependence exists, and the isolated gate probe identifies a route for it.

Using the **first row of each subchunk** as the reference removes that future read for the queries in the subchunk. But why use subchunks at all?

## Why 16 tokens, not one reference for all 64?

In the chunked path used for this experiment, the outer chunk is **64 tokens**. The diagonal rebasing subchunks are **16 tokens**. These are two different sizes.

Rebasing is also a dynamic-range problem. Even when the true pairwise decay is bounded by one, its split factors can be enormous and tiny:

$$
\text{small finite decay} = \text{tiny factor}\times\text{huge factor}.
$$

If we materialize them outside the available exponent range, that can become zero times infinity. A finite true answer does not save the computation.

The tested gate activation has a lower bound of $-5$ nats per token, equivalent to about $-7.2135$ log-base-2 units per token. At that bound, the largest absolute gate displacement from the reference is:

| Rebasing window | Reference | Maximum displacement, log-base-2 units |
| --- | --- | ---: |
| 16 tokens | First row | $15\times7.2135\approx108.2$ |
| 16 tokens | Midpoint, row 8 | $8\times7.2135\approx57.7$ |
| 64 tokens | First row | $63\times7.2135\approx454.4$ |

FP32 and BF16 have an upper exponent limit near 128. The 16-token first-row choice keeps the gate factors within that upper range under this bound; the 64-token choice does not. This is a factor-range calculation, not a blanket guarantee about arbitrary Q/K magnitudes, underflow, or every intermediate.

This was not just an extreme synthetic input. In the earlier kernel campaign, the training module initialized near **−2.5 nats/token**, rather than at the −5 bound. Even then, **64-token chunk-wide rebasing** exceeded the available exponent range for every measured chunk/head/channel triple. A 16-token window at the same per-token decay spans only about 54 base-2 log units, so this was not a failure of the bounded 16-token choice. The mild random gates in my optimization harness had missed the chunk-wide failure.

The midpoint buys extra exponent headroom, not extra mantissa bits. In a later comparison against an FP64 recurrent reference on the same BF16 inputs, the midpoint did not show a systematic accuracy advantage over the first-row reference across the tested gate ranges.

So there are two constraints to satisfy together: **do not read a future reference, and do not make the rebasing window too wide.**

## Does the model learn to use it?

A failed bitwise test is not evidence that a language model has learned to cheat.

I trained paired causal-reference and midpoint-reference variants, keeping the model configuration, initialization seed, data order, and other settings matched within each pair. The reference switch also followed forward recomputation.

Here, “causal” names the **forward reference choice**. Both arms retained the same backward intra kernel, which itself used a midpoint reference. Gradients can legitimately depend on future losses, so that is not evidence of forward leakage. It does mean this A/B isolates the forward reference under a shared backward policy; it is not a comparison of two entirely different forward-and-backward rebasing schemes.

For evaluation, I compared parallel evaluation with autoregressive evaluation. If the midpoint model learned to use a future-dependent signal available only in parallel, removing that signal should hurt it more than it hurts the causal control.

Loss uses the same log-base convention, but measures something different from a gate: a target token assigned probability $p$ has negative log-likelihood $-\ln p$, in **nats**. Averaging that over target tokens gives NLL in nats/token. Divide by $\ln 2$ to express the loss in bits/token instead; this changes the units, not the model or its predictions.

Define the evaluation gap, in nats per token:

$$
G = \mathrm{NLL}_{\mathrm{autoregressive}} - \mathrm{NLL}_{\mathrm{parallel}}.
$$

Then compare the gaps:

$$
\Delta G = G_{\mathrm{midpoint}} - G_{\mathrm{causal}}.
$$

A positive $\Delta G$ would mean the midpoint arm has a larger autoregressive penalty. Subtracting the causal arm matters: parallel and autoregressive implementations can differ numerically even without a future-dependent reference. I checked the recurrent-KDA evaluator against repeated prefix-only evaluation on trained pilot checkpoints rather than assuming the two were interchangeable.

### What the runs showed

| Experiment | Model size | Training tokens per arm | Result |
| --- | --- | --- | --- |
| Pilot, seeds 42, 11, 23 | 520M total / 208M non-embedding parameters | About 1.05B C4 tokens | No learned exploitation detected in any seed |
| Scaled pair, seed 42 | 1.45B total / about 830M non-embedding parameters | About 4.0B C4 tokens | No learned exploitation detected |

For the scaled pair's final checkpoint, evaluation on **1024 matched held-out sequences, with 256 evaluated positions per sequence**, using recurrent KDA for the autoregressive mode, gave $\Delta G$ of **+0.0000088 nats/token**, with a logged paired standard error of **0.0000512 nats/token**. An approximate 95% interval, computed as the estimate ± 1.96 standard errors, is **[-0.000092, +0.000109] nats/token**. That is a pointwise interval for this checkpoint and evaluation—not uncertainty across training seeds or a bound on all possible models.

I also separated positions before the midpoint from the remaining positions in each 16-token subchunk. The earlier rows, which can directly read a future reference, did not develop a distinct exploitation signal.

As a positive control, I explicitly added a scaled copy of the next token's value vector, $0.5\,v_{t+1}$, to the parallel KDA output. That model did learn to exploit the future, and the evaluator reported a large autoregressive penalty. This checks that the evaluation can detect an obvious leak; it does not guarantee sensitivity to every small channel.

The pilot replications also helped interpret small quality differences. Midpoint-versus-causal loss differences changed sign across seeds and metrics. A slightly better loss in one pair was not enough to conclude that midpoint rebasing helped.

```plotly
{"src":"media/kda/scaled-loss.html","title":"Scaled run: held-out NLL · 1.45B parameters · seed 42 · 64 sequences","height":460}
```

*Replotted from the logged W&B evaluation metrics, without smoothing. AR means autoregressive. The four curves nearly overlap; hover for exact values or click a legend entry to hide a trace. Lower NLL is better.*

```plotly
{"src":"media/kda/scaled-gap.html","title":"Zoom in: autoregressive minus parallel NLL · 64 sequences","height":460}
```

*Positive values mean an autoregressive penalty. Bars are estimate ± 1.96 times the logged sequence standard error. Separating the gap from the full loss curve makes the small differences visible.*

```plotly
{"src":"media/kda/paired-checkpoints.html","title":"Paired excess autoregressive penalty, ΔG · scaled run","height":460}
```

*The 64-sequence sweep and the two 1024-sequence evaluations are separate traces; the latter are not a full checkpoint sweep. Positive ΔG means a larger autoregressive penalty under midpoint rebasing.*

```plotly
{"src":"media/kda/paired-seeds.html","title":"Final checkpoints · 1,024 matched sequences per comparison","height":460}
```

*Pilot comparisons are at step 4000; the scaled comparison is at step 7600. Bars on both paired charts use the logged **paired** standard errors, not independently combined arm errors. Every displayed pointwise **ΔG interval** includes zero; this is consistent with no detected excess autoregressive penalty, not proof of exact equivalence.*

## Why is it hard to extract the future signal?

It is tempting to say “the perturbation is tiny, so the model cannot use it.” That is not a sufficient explanation. A tiny but reliable bit can carry useful information, and a model can sometimes amplify it. What matters is whether the perturbation contains **predictive structure beyond what the causal features already provide**, and whether training can discover that structure.

Here is a useful first-order model, not a claim that GPU errors are random. For one retained matrix entry, define the exact contribution of channel $d$ as:

$$
c_d = q_{i,d}k_{j,d}2^{g_{i,d}-g_{j,d}}.
$$

Away from overflow and underflow, write the computed rescaled operands as their exact values times $(1+\epsilon^Q_d(r))$ and $(1+\epsilon^K_d(r))$. The errors include operand preparation; their values depend on the reference and the actual inputs. To first order:

$$
\widehat A_{qk}(r)-A_{qk}
\approx
\sum_d c_d\bigl(\epsilon^Q_d(r)+\epsilon^K_d(r)\bigr)
+\epsilon_{\mathrm{dot}}(r),
$$

where $\epsilon_{\mathrm{dot}}$ is an additive error term for the dot product and output rounding. The future reference cancels out of each exact $c_d$. What remains is its influence on the errors.

That is very different from handing the model a future value vector:

- **The reference is not the next token.** It is a per-channel cumulative gate that may combine several future gate increments. Recovering a useful next-token feature from it is already a separate problem.
- **A changed output is not necessarily a predictive change.** The rounding pattern also depends on the queries, keys, decay regime, and where the operands lie relative to representable numbers. A large count of changed elements does not measure future information.
- **Channel contributions are mixed.** Signed contributions can cancel, reinforce, or be attenuated by later computation. Assuming independent, zero-mean rounding would make a simple noise model, but I did not establish that assumption here.
- **The backward pass is not an exact derivative of the rounding channel.** The ideal expression has zero derivative with respect to the cancelling reference. Ordinary kernel gradients do not explicitly model every rounding boundary. This makes “training will find the leak” less automatic, but does not prove that training cannot exploit or amplify it indirectly.

### Probe the residual, not just the loss

To investigate this, I captured exact KDA inputs from layers 0 and 1 of the trained causal 1.45B model on 1024 held-out sequences. I ran those same inputs through both reference variants and formed:

$$
R = O_{\mathrm{midpoint}} - O_{\mathrm{causal}}.
$$

This residual is a **reference-choice difference**, not a pure measurement of future information. Both variants round; changing references can change outputs even at rows whose reference is already available. The state can also carry differences into later strips.

About **45–49% of output elements** differed in this capture, and about **60% of those differences were exactly one BF16 ULP**. So there was plenty of measurable residual. The question was what could be read from it.

| Probe | Recorded result | What it checks |
| --- | --- | --- |
| Ridge regression for the future gate | $R^2$ approximately zero | No useful linear readout in this probe |
| MLP regression for the future gate | $R^2$ about 0.001, comparable to the current-gate control | A weak decay-regime signal is not necessarily future-specific |
| Next-token prediction with real versus shuffled residuals | Paired cross-entropy difference about zero | No detected incremental predictive benefit from the residual |

The next-token probe used the top-1024-token task and sequence-separated splits. The recorded paired cross-entropy difference was **0.000 ± 0.005 nats**; the uncertainty convention needs to be checked against the original report before publication. The same probe family extracted **0.4–0.7 nats** of predictive information from the ordinary causal features, so the probes were not simply incapable of learning anything.

Taken together, the experiments say more than “the training curves look similar”: the reference changes outputs, bounded readouts did not recover useful extra future information from those changes, and the training A/B did not show exploitation.

But this is not a proof of zero channel capacity. The probes covered two layers of one trained causal model with particular readouts. They do not exhaust nonlinear decoders, midpoint-trained representations, or changes the model might learn under another precision or training regime.

> [!todo] Residual figures
> Add the worklog's residual/ULP distribution and the future-gate and next-token probe comparisons, if available. Caption them as reference-choice residuals, not “percent of elements leaking the future.”

## Could midpoint rounding still be better?

There is another hypothesis worth separating from leakage: perhaps changing the pivot improves numerical behavior, and that changes training even if no future information is used.

**There is a clear range argument. There is not an automatic accuracy argument.**

For a channel in a rebasing window, let $g_{\min}$ and $g_{\max}$ be the smallest and largest cumulative gates. If we ignore Q/K magnitudes and only minimize the worst gate-factor exponent displacement, the best unrestricted reference is:

$$
r^* = \frac{g_{\max}+g_{\min}}{2}.
$$

It balances the two extreme distances. The first-row reference sits at $g_{\max}$ and uses the full gate span on one side; $r^*$ uses half on either side. This is a statement about factor headroom, not an optimal causal algorithm: computing these extrema over the whole window reads the future for early queries.

Also, **the midpoint token is not necessarily the midpoint gate value**. If per-token gate deltas are roughly constant, row 8 in a 16-token window is close to the halfway point in cumulative decay. If the deltas are bursty, most of the decay might occur before or after row 8. Different channels can have different patterns. Centering in token index then need not center the exponents.

The Q/K magnitudes matter too. Ignoring exact zeros, the operand log-magnitudes are:

$$
\log_2|\widetilde Q_{i,d}| = \log_2|q_{i,d}|+g_{i,d}-r_d,
\qquad
\log_2|\widetilde K_{j,d}| = \log_2|k_{j,d}|+r_d-g_{j,d}.
$$

So a reference that balances the gates alone need not balance the actual operands. Their joint distribution, proximity to underflow/overflow, and the conditioning of the final sum all affect numerical error.

Inside the normal range, BF16 does not gain relative precision merely because an operand moves closer to one. It has the same significand width in each normal binade. Rescaling changes which values round up or down; that can improve or worsen a particular answer without producing a systematic improvement. Near a range boundary, however, avoiding underflow or overflow is a qualitatively different benefit.

### What we measured for this hypothesis

The precision sweep compared both references against an FP64 recurrent reference, holding the BF16 inputs fixed. It used sequence length 1024, four heads, and per-token gates uniform in $[-s,0]$ nats for $s$ in $\{0.5,1,2.5,4,5\}$. The ratio of midpoint to causal **mean error** stayed between **0.99 and 1.02**; below one favors midpoint. This did not show a systematic accuracy benefit on that sweep. It does not cover every distribution of gate deltas.

The trained model's gates were not uniformly distributed: the recorded histograms were bimodal, near zero and near the $-5$ nats/token bound. In layers 0 and 1, respectively, **9.0% and 17.6%** of captured per-token gates were below $-4.9$ nats. Near-bound values are not necessarily exact point masses; the spread within each mode matters. I would not assume this distribution generates independent, uniformly distributed rounding errors.

A separate audit on the trained model did find better range headroom under midpoint rebasing. The causal-reference query operands reached into the subnormal range very rarely; the midpoint operands had no subnormal exposure in that capture. That supports a range advantage on those inputs, not a measured model-quality advantage.

This leaves a plausible, narrower hypothesis: **midpoint rebasing may help for distributions that put important operands near a range boundary, even though it did not improve accuracy or quality systematically in these runs.** To test it, vary the distribution of gate deltas and operand magnitudes separately, compare with the same high-precision reference, and measure boundary exposure alongside error. Keep the reference-choice experiment separate from changing the gate bound or subchunk size.

> [!todo] Precision figures
> Add the worklog's error-versus-gate-range and gate/operand-distribution plots, if available. Distinguish per-token deltas, cumulative gate spans, and actual rescaled operand exponents in the captions.

## What I take away

The midpoint reference creates **numerical future dependence**. The experiments did **not detect learned exploitation** of it. I would not describe the result as either “the model learned to see the future” or “there is no leakage.” Each drops half of what happened.

The implementation lesson is more general than KDA: a quantity that cancels in symbolic algebra can still carry information through finite-precision intermediates. Causal masking is not the whole causality contract. The provenance of normalization and scaling values matters too.

For this path, I kept the first-row reference as the default, with bounded rebasing windows and a prefix-invariance regression. Midpoint rebasing offered more range headroom, but these experiments did not establish either a useful future-information channel or a systematic accuracy or model-quality benefit.

The distinction I want to keep is: **a causality violation, a learnable predictive signal, and a better numerical approximation are three different claims.** A test for one does not settle the other two.

## Appendix: nats versus bits

A **nat** is a logarithmic unit using the natural logarithm, $\ln$, whose base is $e$. A **bit** uses $\log_2$. They describe the same quantity on different scales:

$$
\log_2 x = \frac{\ln x}{\ln 2} = \ln x\,\log_2 e.
$$

Attention Gym's public KDA API takes a per-token **natural-log decay**, not the multiplicative decay itself. If that log gate is $\ell$, the decay multiplier is $\alpha=e^{\ell}$. So a gate value of $-5$ means:

$$
\alpha=e^{-5}\approx0.006738,
\qquad
\log_2\alpha=-5\log_2 e\approx-7.2135.
$$

That channel's decay step retains about **0.67%** of the previous state, before KDA's other update terms. It is not $2^{-5}=1/32$, which would retain 3.125%. More-negative log gates mean stronger decay.

The bounded gate transform uses `lower_bound=-5` as its floor; it does not set every gate to $-5$. Attention Gym handles the cumulative sum and conversion to base-2 units internally, so the kernels can use `exp2`. In the equations above, $g$ denotes that **chunk-local cumulative base-2 log gate**, not the per-token natural-log value passed to the API.

## Still to add before publishing

- A minimal suffix-gate counterfactual and public, commit-pinned links to the tested kernels, including the earlier isolated Aqk probe and initialization gate-range measurements.
- The residual-probe and precision-distribution figures, which were not present in the retrieved W&B evaluation series.
- Exact run manifests and the paired-analysis code. The figure snapshot preserves the logged standard errors; verify their construction against the original per-sequence analysis before publication.

These results concern the implementation and configurations tested here, not every KDA implementation or production Kimi model.
