---
title: IntraKernel Profiling
date: 2025-12-05
tags:
  - tool-tips
  - pytorch
---

#### Written: December 5, 2025

# IntraKernel Profiling

[https://github.com/drisspg/transformer_nuggets/pull/63](https://github.com/drisspg/transformer_nuggets/pull/63)

I have been asking nvidia folks for this for a while, and I recently read gau-nernst's article t, he is the goat and he has a nice simple struct for doing the intra kernel timing trick. I liked the format he used super clean and seems simple enough.

I spent some time w/ mr codex and mr claude and decided to implement one.

~~So far it seems to be working as I expect. Big caveat is that I am not storing data to smem and then flushing at the end so there is extra overhead from writing to global at the region end w/ atomics. So overall kernel timings are likely pretty skewed but the regions should remain relatively correct.~~

I changed it so that it has two modes, 1 uses atomics to increment indexes and one that requires manual index bumping ( wanted the context manager per warp to handle this but tracing :(..

So basically the atomic mode is nice if you dont want to handle the incrementing. But it skews results if you have two nested profile regions since the inner one will invoke the atomic and this will cause artificial delays in the outer.

The manual is better but manual. You set a `max num events` and then we filter the tails of the buffers to only find actual recorded events.

Below is a trace w/ nested regions and user event_idx bumping

```python

class ProfiledKernelStatic(CuteOp):
    """Kernel using STATIC mode profiling."""

    def __init__(self, num_iterations: int = 4):
        super().__init__()
        self.num_iterations = num_iterations

    @cute.kernel
    def kernel(
        self,
        output: cute.Tensor,
        prof_buf: cute.Tensor,
        max_events_per_unit: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        bdim, _, _ = cute.arch.block_dim()

        prof_tid = bidx
        global_idx = bidx * bdim + tidx

        # STATIC MODE: Pass explicit event_idx using runtime expressions
        # Event layout:
        #   0: iteration (outer)
        #   1, 4, 7, 10: compute (1 + i*3)
        #   2, 5, 8, 11: store (2 + i*3)
        with profile_region(
            prof_buf,
            max_events_per_unit,
            TAG_ITERATION,
            prof_tid,
            event_idx=Int32(0),
        ):
            for i in cutlass.range(self.num_iterations):
                with profile_region(
                    prof_buf,
                    max_events_per_unit,
                    TAG_COMPUTE,
                    prof_tid,
                    event_idx=Int32(1) + i * Int32(3),  # 1, 4, 7, 10
                ):
                    with profile_region(
                        prof_buf,
                        max_events_per_unit,
                        TAG_STORE,
                        prof_tid,
                        event_idx=Int32(2) + i * Int32(3),  # 2, 5, 8, 11
                    ):
                        if global_idx < cute.size(output):
                            val = output[global_idx]
                            output[global_idx] = val + 1
```

### How useful is this?

Decently, 4/26/2026 Driss. I have used this to investigate FA4. I have used it for a few other things. But in general it has been a tool of last resort. I want to do a megakernels project when I get a sec and i think this will be very helpful.

#### Warpspec version

I have other examples in the nuggets lib. 

```python
TAG_PRODUCER = 0
TAG_CONSUMER = 1
NUM_BLOCKS = 4
THREADS_PER_BLOCK = 64
MAX_EVENTS_PER_UNIT = 4
PRODUCER_WORK_ITERS = 200_000
CONSUMER_WORK_ITERS = 120_000

class WarpSpecializedStaticProfile(CuteOp):
    """Profiles producer/consumer warps with static event indices."""

    @cute.kernel
    # pyrefly: ignore [bad-override]
    def kernel(
        self,
        output: cute.Tensor,
        prof_buf: cute.Tensor,
        max_events_per_unit: cutlass.Int32,
    ):
        bidx, _, _ = cute.arch.block_idx()
        bdim, _, _ = cute.arch.block_dim()
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()

        prof_tid = bidx * Int32(2) + warp_idx
        block_base = bidx * bdim

        if warp_idx == 0:
            with profile_region(
                prof_buf,
                max_events_per_unit,
                TAG_PRODUCER,
                prof_tid,
                target_warp=Int32(0),
                event_idx=Int32(0),
            ):
                idx = block_base + lane_idx
                if idx < cute.size(output):
                    # Do extra writes so producer spans a measurable interval.
                    # pyrefly: ignore [not-iterable]
                    for i in cutlass.range(Int32(PRODUCER_WORK_ITERS)):
                        output[idx + i] = Float32(i)

        if warp_idx == 1:
            with profile_region(
                prof_buf,
                max_events_per_unit,
                TAG_CONSUMER,
                prof_tid,
                target_warp=Int32(1),
                event_idx=Int32(1),
            ):
                idx = block_base + Int32(32) + lane_idx
                if idx < cute.size(output):
                    # Make consumer slightly shorter to show overlap.
                    # pyrefly: ignore [not-iterable]
                    for i in cutlass.range(Int32(CONSUMER_WORK_ITERS)):
                        output[CONSUMER_WORK_ITERS + idx + i] = Float32(i)
```
