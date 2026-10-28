---
title: A Visual Tour of Fast Math
description: "Every float32 value through NVIDIA's fast-math instructions, with and without --use_fast_math."
socialImage: media/social/fast-math-sin.png
socialImageAlt: "A slice of the article's all-float32 error map for sin: the default CUDA build is almost uniformly correct while the fast-math build is red across nearly every binade."
date: 2026-10-27
enableToc: false
dek: All 2³² float32 values through every fast-math knob
tags:
  - pytorch
  - cuda
---

#### Written: October 27, 2026

Fast math is a weird term. There are only 2³² float32s, so I just ran all of them.

<iframe class="doc-widget widget-frame" src="./media/fast-math/fast-math-explorer.html" title="Error maps over all float32 inputs for a chosen function, comparing two implementations against a correctly rounded reference, with source and PTX for each." loading="lazy" style="height: 2600px;"></iframe>
