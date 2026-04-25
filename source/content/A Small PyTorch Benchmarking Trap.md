---
title: A Small PyTorch Benchmarking Trap
date: 2024-08-11
---

#### Written: August 11, 2024

TLDR: A quick highlight of some of gotchas you might encounter when trying to benchmark pt2 code

### Intro

During my day-to-day development of PyTorch I very often need to measure the performance of different units of code at different levels of coarseness. Often this entails taking a cuda kernel in isolation and measuring its performance relative to another.

I do this so much in fact that I have some helper utilities defined here:
https://github.com/drisspg/transformer_nuggets/tree/main/transformer_nuggets/utils

The tools and data this generates is only valuable though if you can trust the numbers, and I was encountering something strange with this benchmark:

https://github.com/drisspg/transformer_nuggets/blob/main/benchmarks/fp8_sat_cast.py

### Whats the Problem?

In FP8 workloads it is common to utilize a technique known as "delayed scaling" to more optimally compute statistics needed for numerical stability during training. I wanted to write a small triton kernel to manually fuse the the operations that we have torch.compile do in this repo: https://github.com/pytorch-labs/float8_experimental

Okay so I we have a triton kernel, the eager code, and the compiled code and we want to compare them. Lets loop over different tensor shapes, saturation settings, and dtypes to get a sense of the performance charactersistics.

#### Slapping torch.compile over code block

It is not uncomon that you will want to microbenchmark some code that has graph breaks:
`compiled_pytorch_fn = torch.compile(eager_scaled_quant)`

This would produce the following numbers with our microbenchmark:
Besides a log about "cache_size_limit reached"

```
[2024-01-08 16:35:27,627] torch._dynamo.convert_frame: [WARNING]    function: 'eager_scaled_quant' (/home/drisspg/meta/transformer_nuggets/transformer_nuggets/fp8/scaled_quant.py:70)
```

The numbers look pretty reasonable (truncated for brevity): https://gist.github.com/drisspg/2bea2b1122793dc58c9ce66440bba12a#file-bad_output-md

```Shell
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [02:18<00:00,  4.31s/it]
   numel  high_precision_dtype    low_precision_dtype    saturated      triton_time    pytorch_time    compiled_pytorch_time
--------  ----------------------  ---------------------  -----------  -------------  --------------  -----------------------
 2097152  torch.bfloat16          torch.float8_e4m3fn    True               33.0514         55.897                   59.4491
 2097152  torch.bfloat16          torch.float8_e4m3fn    False              32.5205         27.1579                  22.5592
 2097152  torch.bfloat16          torch.float8_e5m2      True               32.6485         55.9769                  59.7592
 2097152  torch.bfloat16          torch.float8_e5m2      False              32.5321         27.3613                  22.6497
 2097152  torch.float32           torch.float8_e4m3fn    True               32.9345         53.8681                  57.449
```

And most importantly what you would clean is that it appears the "triton kernel" implementation is universally faster! Cool lets call this custom kernel from now and ship some wins!

#### Hold on a second!

That log was kinda weird, something about a cache... Well I think the code that I am attempting to compile should be 'fullgraph' compilable lets try setting that:
`compiled_pytorch_fn = torch.compile(eager_scaled_quant, fullgraph=True)`

Now we get an error!

```Python
  File "/home/drisspg/meta/pytorch/torch/_dynamo/convert_frame.py", line 353, in _convert_frame_assert
    unimplemented("cache_size_limit reached")
  File "/home/drisspg/meta/pytorch/torch/_dynamo/exc.py", line 193, in unimplemented
    raise Unsupported(msg)
torch._dynamo.exc.Unsupported: cache_size_limit reached
```

Crawling (asking voz) through the dynamo config you will find this argument:
`torch._dynamo.config.cache_size_limit`
with comment:

```Python
# controls the maximum number of cache entries with a guard on same ID_MATCH'd

# object. It also controls the maximum size of cache entries if they don't have

# any ID_MATCH'd guards.
```

Nothing about perf behavior in here but I know that for my example I iterating over 32 configs lets try setting this to something larger, like 1000 (go big or go home).

This gives now the following results:

```Shell
   numel  high_precision_dtype    low_precision_dtype    saturated      triton_time    pytorch_time    compiled_pytorch_time
--------  ----------------------  ---------------------  -----------  -------------  --------------  -----------------------
 2097152  torch.bfloat16          torch.float8_e4m3fn    True               33.2983         55.8643                  59.6987
 2097152  torch.bfloat16          torch.float8_e4m3fn    False              33.0075         26.9028                  22.0985
 2097152  torch.bfloat16          torch.float8_e5m2      True               33.1423         55.8323                  59.4687
 2097152  torch.bfloat16          torch.float8_e5m2      False              32.4296         26.9288                  22.1912
 2097152  torch.float32           torch.float8_e4m3fn    True               32.9567         53.8027                  58.6836
 2097152  torch.float32           torch.float8_e4m3fn    False              32.6021         25.575                   22.4197
 2097152  torch.float32           torch.float8_e5m2      True               33.1044         53.8684                  22.212
```

https://gist.github.com/drisspg/2bea2b1122793dc58c9ce66440bba12a#file-actual_numbers-md

#### Same experiment different conclusions

Re-running the same microbenchmark sweep now with fullgraph=True and increased dynamo cache size limit we come to the opposite conclusion compared to the naive benchmark. For all but a few cases the inductor generated code appears to be more performant. And we want the generated code of inductor and compare to the triton kernel.

I hope that this little note finds someone well, and will inspire you to not ignore logs when things seem slightly awry.
