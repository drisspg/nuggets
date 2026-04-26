---
title: My Python Tools for PyTorch bencmarking
date: 2026-04-26
tags:
  - tool-tips
  - pytorch
---

#### Written: April 26, 2026

# My Python Tools for PyTorch bencmarking

https://github.com/drisspg/transformer_nuggets/blob/main/transformer_nuggets/utils/benchmark.py

There are more main ones now:

```python
from transformer_nuggets.utils.benchmark import (
    benchmark_cuda_function_in_microseconds,
    benchmark_cuda_function_in_microseconds_triton,
    benchmark_cuda_function_stats,
)
```

I pretty much always try and use  `benchmark_cuda_function_in_microseconds`

From my past experience it was closest to what NCU reported. I typically run it as:

warmup(make sure compilation happens here) then time(which has its own warmup)
https://github.com/meta-pytorch/attention-gym/blob/main/benchmarks/bench_block_mask.py#L140-L147

That function now has a couple more knobs:

```Python
time_us = benchmark_cuda_function_in_microseconds(
    func,
    *args,
    NUM_ITERS=100,
    USE_CUDA_GRAPHS=True,
)
```

`USE_CUDA_GRAPHS=True` captures one static CUDA graph and times replay latency with CUDA events. This is often closer to NCU for tiny static kernels, where eager launch gaps can dominate what you think you are measuring. TBH I need to investigate this cause in theory kineto events should be more immune to this. Unless we have multiple kernels and their net time is inflated from launch overhead that you would expect to be hidden in real workloads.

There is also a stats version now:

```python
stats = benchmark_cuda_function_stats(
    func,
    *args,
    NUM_ITERS=100,
    USE_CUDA_GRAPHS=True,
)

print(stats.median_us)
print(stats.median_ci_us)
print(stats.p05_us, stats.p50_us, stats.p95_us)
```

This returns the raw samples, median, bootstrap confidence interval for the median, and p05/p50/p95. I like this when I am comparing variants and want to know whether the difference is real or just noise. I just used this for a big sweep in FA4. For CLC schedulers you going to naturally have more dynamic numbers because the process itself is dynamic. In that case comparing ranges with via pseduo statistical tests was helpful.

There is still the triton wrapper:

```python
time_us = benchmark_cuda_function_in_microseconds_triton(func, *args)
```

and all of these can take `LOCK_CLOCKS=True` if you want to lock GPU clocks for a more stable run. That requires root. This arg kinda works.. idk hit or mis sometimes.

I advocate for this usage (if you only care about cuda time)
