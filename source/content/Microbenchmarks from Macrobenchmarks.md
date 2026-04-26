---
title: Microbenchmarks from Macrobenchmarks
date: 2024-08-11
tags:
  - tool-tips
---

#### Written: August 11, 2024

TLDR: Efficiently find real world operator inputs using a `__torch_dispatch__` mode to log inputs found in torchbench for arbritrary ops

## Motivation

torch.nn.functional.scaled_dot_product_attention currently has 3 different implementations, ranging in coverage and performance. There is a new kid on the block in CuDNN that promises up to 2x perf increases in the newest version of cuDNN. As well the FlashAttention2 implementation changes quite frequently, as of this PR: we are up to date with the origin but how do we make sure there aren't any perf regression.

The case for microbenchmarks. They are tricky but in general isolating at the per op level can be very helpful for tracking regressions and and improvements! This is magnified for common, compute bound ops; sdpa, matmul, convolution.

So how do we get the important inputs?

I think the standard way to this is to look at the popular models and work backward from the model definition. This is tenable for a some number of models, Im looking at you LLMs but there is a long tail of input shapes that slip under the radar.

Is there a way to do this empirically?

### Torchbench

[Torchbench](https://github.com/pytorch/benchmark) "is a collection of open source benchmarks used to evaluate PyTorch performance." I stores many popular model definitions and was designed for tracking and measuring PyTorch's eval and training perf on these models. In fact torchbench model definitions are used to track much of PT2's performance here:https://hud.pytorch.org/benchmark/compilers. This seems like a great place to find, important and represtantive inputs for ops of interest. The question is how to do this scalably?

### TorchDispatch

Torchdispatch is the one stop shop for extending pytorch! It enables this by hooking into one of PyTorch's core component, the dispatcher. Mandatory see Ed's blog here: http://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/. As well, Horace has a great dev-discuss post on some of the fundamentals of `__torch_dispatch__`: https://dev-discuss.pytorch.org/t/what-and-why-is-torch-dispatch/557.

I have been working a lot with tensor subclasses recently; [Float8Tensor](https://github.com/pytorch-labs/float8_experimental/blob/607ff7baa680b7e1e8a26c984005f30b546b3fa7/float8_experimental/float8_tensor.py#L146)is central to float8_experimental's design. As well the [NF4Tensor](https://github.com/pytorch-labs/ao/blob/9765bc60f8567b88d3282acb578ecdec4e6d89e8/torchao/dtypes/nf4tensor.py#L102) is utilized to enable QLoRa in native pytorch. Hence, this naturally seemed like the right tool for the job

Heavily inspired by the [LoggingMode](https://github.com/albanD/subclass_zoo/blob/276d2f005484d80ebbcd9e274d79685adb6a1da2/logging_mode.py#L39) in subclass zoo, I made a slightly modified version that can be found here: [ShapeLog](https://github.com/drisspg/transformer_nuggets/blob/c24053f77f59a9dff6b86c7df2a7f71ae5c88d45/transformer_nuggets/utils/shape_trace.py#L28). This will store input information into a Counter and save this to disk.

An example output:

```python
    'aten._scaled_dot_product_flash_attention.default': Counter({
        "query:bf16[4,12,512,64]|key:bf16[4,12,512,64]|value:bf16[4,12,512,64]|dropout_p:0.1|is_causal:False|return_debug_mask:False
|scale:0.125->('bf16[4,12,512,64]', 'f32[4,12,512]', None, None, 512, 512, 'i64[]', 'i64[]', 'bf16[0]')": 12,
        "query:bf16[4,12,512,64]|key:bf16[4,12,512,64]|value:bf16[4,12,512,64]|dropout_p:0.1|is_causal:True|return_debug_mask:False|
scale:0.125->('bf16[4,12,512,64]', 'f32[4,12,512]', None, None, 512, 512, 'i64[]', 'i64[]', 'bf16[0]')": 6,
```

I am also working on a [utility](https://github.com/drisspg/transformer_nuggets/blob/c24053f77f59a9dff6b86c7df2a7f71ae5c88d45/transformer_nuggets/utils/shape_trace.py#L105) to automatically generate the op inputs from this string format. I am sure I am re-inventing the wheel here and there is likely some FX format that I should be using. I will blame doc SEO for not using said format.

For example:

```python
aten = torch.ops.aten
op = aten.unsqueeze.default

logs = open_logs("..logs.pkl")
inpts = construct_input(logs, op, "cuda")
op(**inpts[0])
```

### Gluing it all together

I was able to craft a small script to loop over the available models in torchbench and grab operator stats for matmul ops, sdpa ops, and conv ops. Script and op logs can be found here: [GIST](https://gist.github.com/drisspg/402855d460dbf130013b9230b35699c2#file-torchbench_loop-py)

I plan to fold some of the SDPA op shapes into the existing SDPA microbenchmarks.
There are many more shape permutations for convolution and mm ops, and there might be some important shapes in there as well!

My usage of torchbench is far from ideal, for instance only about 50% of the models ran succesfully in bfloat16, I am sure there are other argumetns that I am missing as well.

#### Summary

This was a fun lil side project, that I think shows the power and flexibility of torch_dispatch. If you have an aten op that you are investigating feel free to use the subclass + torchbench to figure out how real world models are using it!
