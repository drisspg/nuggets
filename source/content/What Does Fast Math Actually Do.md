---
title: What Does Fast Math Actually Do?
description: "Every float32 value through NVIDIA's fast-math instructions, with and without --use_fast_math. An interactive map of where the errors land."
socialImage: media/social/fast-math-sin.png
socialImageAlt: "A slice of the article's all-float32 error map for sin: the default CUDA build is almost uniformly correct while the fast-math build is red across nearly every binade."
date: 2026-10-27
enableToc: false
dek: Running all 2³² float32 values through every fast-math knob
tags:
  - pytorch
  - cuda
---

#### Written: October 27, 2026

"Fast math" is a weird term. It sounds like a free speed-up you forgot to turn on, but it is really a bundle of separate choices: flush tiny numbers to zero, swap in approximate hardware instructions, round division and square root differently. There are only 2³² float32 values, so instead of reading about it I ran every one of them through each choice and compared against the correctly rounded answer.

**How to read it.** Each heatmap is the whole float32 line folded into a grid: rows are exponents (tiny numbers at the bottom, huge at the top), columns are mantissa, `+x` on the left and `−x` on the right. Each cell covers 131,072 inputs and is colored by its worst error in ULPs (float32 steps). No fill means exact. Pick a function and two implementations, drag to zoom, or switch the color to **A vs B**.

<iframe class="doc-widget widget-frame" src="./media/fast-math/fast-math-explorer.html" title="Error maps over all float32 inputs for a chosen function, comparing two implementations against a correctly rounded reference, with source and PTX for each." loading="lazy" style="height: 2600px;"></iframe>

### Things worth clicking

- **sin**, CUDA default vs fast math: `sin.approx` is a low-accuracy approximation with no range reduction: 99% of inputs come back wrong, and large ones are garbage.
- **exp**: fast math computes `exp2(x · log₂e)`, mostly paying for rounding that product: up to 64 ULP near |x| ≈ 88.
- **relu**, then **Zoom to subnormals**: flush-to-zero turns `max` into `max.ftz`, so every positive subnormal becomes 0. No math function involved.
- **sqrt**: compare `inline PTX sqrt.approx` with CUDA fast math. The approximation is fine; the dramatic errors are all flush-to-zero.
- **x/10**, torch eager: on CUDA, PyTorch computes it as `x * 0.1`, which is 1 ULP off on about 20% of inputs.

Fast math is a bundle of instruction choices, and each one trades accuracy differently. It's worth knowing which one you're buying.
