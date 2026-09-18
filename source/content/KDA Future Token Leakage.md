---
title: KDA Doesn't Care About the Future
date: 2026-09-15
enableToc: false
dek: A future-dependent rounding effect in KDA when enabling tensorcores and experiments investigating their affect
tags:
  - pytorch
cssclasses:
  - sidenotes
---

#### Written: September 15, 2026

<iframe class="doc-widget widget-frame" src="./media/kda/kda-future-animation.html" title="Conceptual reference directions from an orange midpoint: green toward earlier positions, red toward later positions." loading="lazy" style="height: 340px;"></iframe>

### Product pitch

TorchTitan has been working on enabling new models. As you most likely know, most new models have moved away from purely global causal attention and now do some combo of GCA (global causal attention) + local attention. We call these hybrid models. They optionally mix in sparse attention for some or all of the layers that used to be GCA. 

It is hard to be nimble in pytorch/pytorch - this is a good and a bad thing. We want to build out useful apis that enable researchers and implementers to get the most out of pytorch. Our Linear Attention apis have been severely lacking here. And our Sparse Apis have been decent through flex-attention but only when sparse granularity is largish (128,128)+. Soooo what are we to do! 

[Attention Gym](https://github.com/meta-pytorch/attention-gym) is changing! 

We are coalescing development of these fun new attention flavors here. We have built a number of primitives for gdn and kda and have been integrating them into torchtitan's training and RL(inference) stack. As well we have been adding more sparse primitives to enable performant DSv4 training in Titan.

"But Driss why would I not just use FLA" That is a great question insightful reader! My honest answer: we pytorch developers are humans. We need a place to explore ideas, find common abstractions, figure out what works and what doesn't. Attention gym is that place for me and others. Long term I would love to develop something as extensible as `Flex Linear Attention` but right now - I don't see it. As well AI has kind of thrown a wrench into this all generalization thing we like doing. If you want to have some influence on where we invest our time; use the repo, open issues and give us feedback. We are dogfooding in torchtitan but would love to hear from other voices.

Another more `polished` answer is that the components we are offering are more specialized to the latest hardware and this allows us to eek <span class="sidenote-pair"><span class="sidenote-ref" tabindex="0" aria-describedby="kda-performance-note">non trivial out performance</span>.<span id="kda-performance-note" class="sidenote" role="note">These performance gains can be very large, but numbers are numbers, and I don't want to include comparisons in this particular blog post.</span></span> We have fully integrated CuDNN's uber mega kernels, a robust CP implementation, paid special attention to making everything cuda-graphable. But if you are using FLA and it works for you and don't want to switch I get it. Its an awesome project and I personally have learned so much from it :)

#### Pitch done - TLDR

 I was working on the intra-chunk kernels for Kimi Delta Attention(KDA) and found that there was a subtle implementation choice that has the potential to break `causality`.  I used [TorchTitan](https://github.com/pytorch/torchtitan) to test whether the model could learn to exploit it. I wont bury the need but turns out not; at least at the scales I tested; and I do a lil math to show why it seems unlikely to do so at larger runs.

### What is causality anyways

I think this phrase is is a little to anthropomorphized. A better one is; training inference mismatch. Thats it. We call it causality because this particular form of mismatch is when you let a token at position $N$ receive information from token $M$, for some $M > N$. And in essence $N$ can see the future. This unsurprisingly really helps with the task of next token prediction. What are some ways this might happen;
1. you forget to invoke `F.scaled_dot_product_attention(... causal=True)`. That is an obvious one. There are some other more subtle forms; 
2. Expert choice routing
3. Blockwise scaling of inputs during training

etc etc

This can be subtle, because during training there isnt anything actually `wrong` with this. Either you have a massive information leak and you will see your loss decrease very very rapidly; or it will be a slow trickle. You might even think `damn i really did something with this datamix!`. Dont be fooled, the problems only show up when you try to serve this model using auto-regressive token generation. The model learned that it's only going to see batches of tokens and that this information from token $M$ will always be there to influence what it should predict at token $N$. That's what it learned during training, but at inference, token $M$ doesn't exist yet. We're building up the sequence one token at a time, and the result is you've trained this cracked model, but at inference time, it's going to underperform relative to what you saw during training!

<figure class="kda-figure kda-token-modes" aria-label="Possible prediction mismatch between parallel training and autoregressive inference">
<div class="kda-token-mode">
<div class="kda-label">Training</div>
<svg viewBox="0 0 360 240" role="img" aria-labelledby="kda-training-title kda-training-desc">
<title id="kda-training-title">Next-token distributions computed together</title>
<desc id="kda-training-desc">All three input tokens are available. Solid green arrows show allowed causal inputs: x1 predicts p2; x1 and x2 predict p3; x1 through x3 predict p4. Red dashed arrows show every forbidden future-token connection in this example: x2 into p2, and x3 into p2 and p3. These predictions should not depend on those future tokens. The distributions are computed in parallel. The leak and bars are schematic, not measured KDA results or evidence of learned exploitation.</desc>
<defs>
<marker id="kda-training-arrow" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L6 3L0 6Z" fill="currentColor" /></marker>
<marker id="kda-leak-arrow" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="5" markerHeight="5" orient="auto"><path class="kda-mode-leak-head" d="M0 0L6 3L0 6Z" /></marker>
<g id="kda-probability-bars"><rect x="0" y="21" width="7" height="11" rx="1" /><rect x="11" y="2" width="7" height="30" rx="1" /><rect x="22" y="13" width="7" height="19" rx="1" /><rect x="33" y="25" width="7" height="7" rx="1" /></g>
<g id="kda-probability-bars-next"><rect x="0" y="15" width="7" height="17" rx="1" /><rect x="11" y="25" width="7" height="7" rx="1" /><rect x="22" y="3" width="7" height="29" rx="1" /><rect x="33" y="18" width="7" height="14" rx="1" /></g>
<g id="kda-probability-bars-last"><rect x="0" y="24" width="7" height="8" rx="1" /><rect x="11" y="17" width="7" height="15" rx="1" /><rect x="22" y="22" width="7" height="10" rx="1" /><rect x="33" y="1" width="7" height="31" rx="1" /></g>
</defs>
<g class="kda-mode-token"><rect x="46" y="48" width="36" height="30" rx="3" /><text x="64" y="68">x₁</text><rect x="159" y="48" width="36" height="30" rx="3" /><text x="177" y="68">x₂</text><rect x="272" y="48" width="36" height="30" rx="3" /><text x="290" y="68">x₃</text></g>
<g class="kda-mode-arrows" marker-end="url(#kda-training-arrow)"><path d="M64 80V135" /><path d="M64 80C64 108 177 105 177 135" /><path d="M64 80C64 117 290 110 290 135" /><path d="M177 80V135" /><path d="M177 80C177 119 290 118 290 135" /><path d="M290 80V135" /></g>
<rect class="kda-mode-batch" x="29" y="139" width="298" height="69" rx="3" />
<g class="kda-mode-leak-arrows" marker-end="url(#kda-leak-arrow)"><path d="M177 80C177 106 82 114 76 143" /><path d="M290 80C290 128 123 165 87 165" /><path d="M290 80C290 104 192 112 188 143" /></g>
<g class="kda-mode-probs"><use href="#kda-probability-bars" x="44" y="147" /><use href="#kda-probability-bars-next" x="157" y="147" /><use href="#kda-probability-bars-last" x="270" y="147" /></g>
<g class="kda-mode-prob-label"><text x="64" y="198">p₂</text><text x="177" y="198">p₃</text><text x="290" y="198">p₄</text></g>
</svg>
</div>
<div class="kda-token-mode kda-token-mode--inference">
<div class="kda-label">Inference</div>
<svg viewBox="0 0 360 240" role="img" aria-labelledby="kda-inference-title kda-inference-desc">
<title id="kda-inference-title">Next-token distributions computed one at a time</title>
<desc id="kda-inference-desc">Start with x1 and compute p2. Sample x2, append it to the prefix, and compute p3. Sample x3, append it, and compute p4. Dashed empty boxes are tokens that do not exist yet; curved arrows feed a sampled token into the next step. Different bar shapes illustrate a possible prediction mismatch if training relied on future information that is unavailable at inference. These are illustrative distributions, not measured KDA results or evidence of learned exploitation.</desc>
<defs>
<marker id="kda-inference-arrow" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L6 3L0 6Z" fill="currentColor" /></marker>
<g id="kda-inference-bars"><rect x="0" y="8" width="7" height="24" rx="1" /><rect x="11" y="20" width="7" height="12" rx="1" /><rect x="22" y="23" width="7" height="9" rx="1" /><rect x="33" y="10" width="7" height="22" rx="1" /></g>
<g id="kda-inference-bars-next"><rect x="0" y="17" width="7" height="15" rx="1" /><rect x="11" y="7" width="7" height="25" rx="1" /><rect x="22" y="14" width="7" height="18" rx="1" /><rect x="33" y="23" width="7" height="9" rx="1" /></g>
<g id="kda-inference-bars-last"><rect x="0" y="6" width="7" height="26" rx="1" /><rect x="11" y="20" width="7" height="12" rx="1" /><rect x="22" y="14" width="7" height="18" rx="1" /><rect x="33" y="24" width="7" height="8" rx="1" /></g>
</defs>
<g class="kda-mode-token"><rect x="34" y="26" width="36" height="30" rx="3" /><text x="52" y="46">x₁</text><rect x="34" y="104" width="36" height="30" rx="3" /><text x="52" y="124">x₁</text><rect x="34" y="182" width="36" height="30" rx="3" /><text x="52" y="202">x₁</text><rect x="80" y="182" width="36" height="30" rx="3" /><text x="98" y="202">x₂</text></g>
<g class="kda-mode-token kda-mode-token--new"><rect x="80" y="104" width="36" height="30" rx="3" /><text x="98" y="124">x₂</text><rect x="126" y="182" width="36" height="30" rx="3" /><text x="144" y="202">x₃</text></g>
<g class="kda-mode-missing"><rect x="80" y="26" width="36" height="30" rx="3" /><rect x="126" y="26" width="36" height="30" rx="3" /><rect x="126" y="104" width="36" height="30" rx="3" /></g>
<g class="kda-mode-arrows" marker-end="url(#kda-inference-arrow)"><path d="M181 42H232" /><path d="M181 120H232" /><path d="M181 198H232" /><path class="kda-mode-feedback" d="M262 64C262 91 98 79 98 98" /><path class="kda-mode-feedback" d="M262 142C262 169 144 157 144 176" /></g>
<g class="kda-mode-probs"><use href="#kda-inference-bars" x="242" y="26" /><use href="#kda-inference-bars-next" x="242" y="104" /><use href="#kda-inference-bars-last" x="242" y="182" /></g>
<g class="kda-mode-prob-label"><text x="310" y="46">p₂</text><text x="310" y="124">p₃</text><text x="310" y="202">p₄</text></g>
</svg>
</div>
</figure>

### Don't just take my word for it

MatX's [“Future leakage in block-quantized attention”](https://matx.com/research/leaky_quantization) is a really really nice blog. The causal break it found has to do with low precision attention. When values share a quantization scale across token positions, a future outlier can increase that scale and make an earlier value underflow. Seems harmless but models are sneaky, trixy little hobbits and they use signals in remarkable ways! 

<figure class="kda-source-figure" aria-labelledby="matx-leaky-region-caption">
<img src="./media/kda/matx-leaky-region.svg" alt="MatX's query-by-value matrix, divided into 32-token blocks. Red hatched diagonal blocks can leak future information through shared value quantization scales; green earlier blocks are safe from this leak, and gray later blocks are masked." width="550" height="500" loading="lazy">
<figcaption id="matx-leaky-region-caption">Figure from MatX, <a href="https://matx.com/research/leaky_quantization">Future leakage in block-quantized attention</a> </figcaption>
</figure>

MX quantization shares one scale across 32 elements along the reduction dimension. In attention's second GEMM, $PV$, we multiply $(N_q \times N_{kv})$ by $(N_{kv} \times D_v)$. So the reduction runs across **token positions**.

In the picture, rows are queries and columns are values. Green blocks are entirely in the past -> all query indices > all  kv indices; gray blocks are fully masked. The **red diagonal blocks** have mixed sign. If we naively quantized there would be a path for info to flow form kv_index > q_index.

We shall see this future leakage ends up looking very similar to our KDA example but not through low precision quantization but a rescale factor.

MatX describes a really nice experimental process for finding this leak: train a small model and measure its performance on a held-out set, evaluating in two modes: parallel and autoregressive. If autoregressive performance is worse, that's a good sign your model's training setup isn't mirroring its inference setup. Their fix is also quite nice—>I encourage you to read the blog! We'll use this technique to see if our causal break is measurable.

## Chunkwise KDA in Broad Strokes

KDA is a delta-rule linear attention variant. Like many other linear attention variants it stores info in a recurrent state that is calculated earlier tokens:

$$
\begin{alignedat}{3}
\text{Definition:}\\
\text{(0)}&\quad D_i
&&=\operatorname{diag}(2^{g_i})
&\qquad&\text{form the per-channel decay operator},\\
\text{(1)}&\quad\widetilde S_i
&&=D_iS_{i-1}
&\qquad&\text{decay the previous state},\\
\text{(2)}&\quad\widehat v_i
&&=k_i^\top\widetilde S_i
&\qquad&\text{read the existing value associated with the current key},\\
\text{(3)}&\quad z_i
&&=\beta_i\left(v_i-\widehat v_i\right)
&\qquad&\text{given the current value, compute the scaled prediction error},\\
\text{(4)}&\quad S_i
&&=\widetilde S_i+k_iz_i^\top
&\qquad&\text{write that correction at the current key},\\
\text{(5)}&\quad o_i
&&=s q_i^\top S_i
&\qquad&\text{read the updated state with the current query}.
\end{alignedat}
$$


This recurrent form is great when we are decoding 1 token at a time but for training it is not efficient. If only there was some way to turn this memory bound problem into one that can use our tensorcores.. <span class="sidenote-hover"><button type="button" class="sidenote-trigger" aria-describedby="kda-chunking-reaction">CHUNKING!</button><span id="kda-chunking-reaction" class="sidenote sidenote--hover-media" role="note"><img src="./media/kda/kool-aid-man.gif" alt="The Kool-Aid Man bursts through a wall. Oh yeah!" width="420" height="310" loading="lazy"></span></span>

The chunked implementation reorganizes that recurrence into local matrix operations plus a state update across chunks, this shortens our sequential depth at the cost of explicitly computing pairwise terms within each chunk. 

The math is really fun but im hesitant to dive super deep here cause it can be distracting. SO stick with me and let's follow one of those pairwise terms: how query $i$ reads the correction written at token $j$; which comes from `step 5` in this recurrence.

## Chunkwise Aqk

Let's call this weight $A_{qk}[i,j]$. How much does query i read from token j's correction? That depends on how the query and key line up, and how much each channel has decayed between the two tokens.

From here on, $g_{i,d}=\sum_{t=0}^{i}\delta_{t,d}$ is the <span class="sidenote-pair"><span class="sidenote-ref" tabindex="0" aria-describedby="kda-gate-units-note">cumulative log-base-2 gate</span> at token $i$, <span class="sidenote-ref" tabindex="0" aria-describedby="kda-channel-gate-note">channel $d$</span>, measured within the chunk.<span class="sidenote" role="note"><span id="kda-gate-units-note">Attention Gym's KDA API uses natural-log gates. The lower bound is currently capped at $-5,$ that means means the strongest decay =  $e^{-5}\approx0.006738$: or in other words only about 0.67% of the previous state is retained before the other update terms.</span><br><br><span id="kda-channel-gate-note">Notice that extra $d$ index. GDN uses one decay per token per head; KDA gives every key channel its own. Seems like a small change, but as we'll see it makes a big difference to how we implement the chunkwise kernel.</span></span></span> The increments $\delta_{t,d}$ are the per-token log2 gates called $g$ in the recurrence above.

Suppose $j<i$. Token $j$ writes a correction into the state. By the time query $i$ reads it, that correction has been decayed at every step from $j+1$ through $i$. We start at $j+1$ because each token decays the existing state *before* adding its own correction.

With $D$ channels, and leaving out the usual query scale $s$, the weight on that correction is:

$$
A_{qk}[i,j] = \sum_{d=1}^{D} q_{i,d} k_{j,d} 2^{g_{i,d}-g_{j,d}}, \qquad j \leq i.
$$

Entries with $j>i$ are masked to zero. Because the per-token log gates are nonpositive, the cumulative gates decrease: for $j\leq i$, the decay factor is at most one.

<details>
<summary>Where did this matrix come from?</summary>

Unroll step (5) back to the chunk boundary. With $H_c$ as the incoming state, the output has two pieces:

$$
\begin{aligned}
o_i &= s(q_i\odot 2^{g_i})^\top H_c\\
&\quad + s\sum_{j\leq i}A_{qk}[i,j]z_j.
\end{aligned}
$$

The first reads history from earlier chunks. The second reads completed writes inside this chunk, including the current token's own write. The $s$ is outside Aqk here because we left it out of its definition above.

The lower-triangular system comes from steps (2)–(3): each key reads earlier writes to determine its new correction. That is a key/key dependency, separate from this query/key read. Attention Gym's [composed forward path](https://github.com/meta-pytorch/attention-gym/blob/52c9eaa31e87a28dc0d3d9464af2088e6384a483/attn_gym/linear/kda/fwd/cute/chunk_kda_fwd.py) puts those pieces together.

</details>

## The rebasing trick

The awkward part for a fast kernel is that the decay depends on the channel $d$. It lives **inside** the reduction. This is not a plain $QK^T$ followed by one scalar decay per matrix entry.

Luckily it separates: $2^{g_{i,d}-g_{j,d}}=2^{g_{i,d}}2^{-g_{j,d}}$. One factor belongs to the query, the other to the key. That is enough to make GEMM operands! But those factors can get very small and very large.

So we pick a common reference gate $r_d$ for each channel in the block to keep those factors in range, and split the decay around it:

$$
2^{g_{i,d}-g_{j,d}} = 2^{g_{i,d}-r_d}\,2^{r_d-g_{j,d}}.
$$

Now define rescaled operands:

$$
\widetilde Q_{i,d}=q_{i,d}2^{g_{i,d}-r_d},
\qquad
\widetilde K_{j,d}=k_{j,d}2^{r_d-g_{j,d}}.
$$

The left operand depends only on $(i,d)$; the right only on $(j,d)$, with the reference held fixed. So the block is $\widetilde Q\widetilde K^T$, then causally masked. There are our tensorcores.

In real arithmetic, the reference cancels. Pick the first row, pick the midpoint: same answer.

On the GPU, we materialize those operands separately. Exponent evaluation, multiplication, conversion, and the matrix multiplication all have finite precision. The cancellation is no longer an identity between the computed values. <span class="sidenote-pair"><strong class="sidenote-ref" tabindex="0" aria-describedby="kda-rounding-note">The reference can become part of the numerical result.</strong><span id="kda-rounding-note" class="sidenote" role="note">The BF16 CuTe engine casts the rescaled operands to BF16 before its matrix multiplication, with FP32 accumulation. The Triton diagonal kernel passes FP32 products to <code>tl.dot</code>; its effective dot precision needs to be checked separately. BF16 inputs do not imply identical rounding paths.</span></span>

And if that reference comes from a future token, an otherwise allowed Aqk entry can change when future gates change. No future values need to get through the mask.

<iframe class="doc-widget widget-frame" src="./media/kda/aqk-rescale.html" title="Following one retained causal Aqk entry, query 5 reading the write from token 2, through relative decay, operand rebasing, a midpoint reference, and separately rounded operands." loading="lazy" style="height: 900px;"></iframe>

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

```chart
{"src":"media/kda/scaled-loss.json","title":"Scaled run: held-out NLL · 1.45B parameters · seed 42 · 64 sequences","height":300}
```

*Replotted from the logged W&B evaluation metrics, without smoothing. AR means autoregressive. The four curves nearly overlap; hover to update the legend, or click a legend entry to hide a trace. Full-precision values are in the data table. Lower NLL is better.*

```chart
{"src":"media/kda/scaled-gap.json","title":"Zoom in: autoregressive minus parallel NLL · 64 sequences","height":300}
```

*Positive values mean an autoregressive penalty. Bars are estimate ± 1.96 times the logged sequence standard error. Separating the gap from the full loss curve makes the small differences visible.*

```chart
{"src":"media/kda/paired-checkpoints.json","title":"Paired excess autoregressive penalty, ΔG · scaled run","height":300}
```

*The 64-sequence sweep and the two 1024-sequence evaluations are separate traces; the latter are not a full checkpoint sweep. Positive ΔG means a larger autoregressive penalty under midpoint rebasing.*

```chart
{"src":"media/kda/paired-seeds.json","title":"Final checkpoints · 1,024 matched sequences per comparison","height":300}
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
