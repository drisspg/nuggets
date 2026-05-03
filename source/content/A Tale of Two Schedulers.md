---
title: A Tale of Two Schedulers
date: 2026-05-03
tags:
  - pytorch
---

#### Written: May 3, 2026

# A Tale of Two Schedulers

![[media/scheduler-meme-wide.png]]

## TLDR

One mechanism for enabiling CLC with grouped_gemms. 

There were some interesting quirks of padded buffer layouts that made this a fun challenge, as well as a pattern for Cluster Launch Control (CLC) that I think is broadly applicable to other dynamic CUDA-graphable kernels.


## Background

Overlapping comms and compute is tricky, especially when it comes to gemms.

If a GEMM kernel starts before the comm kernel on a rank, the GEMM will use all available SMs, and the comm kernel won’t be able to start until the GEMM finishes, regardless of stream priority. This can create stragglers. On the other hand, if the comm kernel runs first, the GEMM won’t be able to start all blocks simultaneously, creating very bad wave quantization effects. We solve this by **limiting the number of SMs that persistent GEMM kernels can use**, when overlapped with communication kernels.

While this works and allows for the two different kernel types to overlap, efficiently partitioning your resources statically ahead of time can lead to suboptimal resource allocation. This pattern of partitioning SMs for communication and computation became best practice on Hopper GPUs and got the name `sm_carveout`. It was so popular, in fact, that we have a number of ways we tackle this problem in PyTorch:
 - [`torch.cuda.green_contexts.GreenContext`](https://github.com/pytorch/pytorch/blob/main/torch/cuda/green_contexts.py): the public CUDA-facing API for partitioning SM resources more generally
 - `torch._C._set_sm_carveout_experimental(...)`: a private experimental knob used by some matmul paths
 - backend-specific integrations that consume that carveout internally, including persistent matmul paths in Inductor/Triton, scaled matmul, and FA3

```python
torch._C._set_sm_carveout_experimental(27)
y = a @ b
torch._C._set_sm_carveout_experimental(None)
```

For Blackwell, Nvidia gave us programmers a little help and introduced Cluster Launch Control. The CUDA programming guide has a good overview here: [Cluster Launch Control](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cluster-launch-control).

The table below summarizes the tradeoffs between the three approaches:

| Capability | Fixed work per thread block | Fixed number of thread blocks | Cluster Launch Control |
|---|:---:|:---:|:---:|
| Reduced overheads | ❌ | ✅ | ✅ |
| Preemption | ✅ | ❌ | ✅ |
| Load balancing | ✅ | ❌ | ✅ |

Fixed work per thread block is a fancy way of saying a `normal` CUDA kernel, the ones you read about in PMPP, where you have many independent subproblems and you launch a grid that covers these problems.

A fixed number of thread blocks is synonymous with `persistent`, where you launch `num_sms` thread blocks and map all the independent problems onto your workers. In our case this would be roughly `num_sms - carveout_sms`.

CLC lets us have static workers, which we want so that we can overlap the epilogue work of one tile with the prologue of the next, but still get some of the benefits of normal CUDA kernels: higher priority kernels can supersede execution of lower priority ones.

Sounds too good to be true!!

## CLC was not designed for this..

There are two main APIs I want to focus on.

The first is the [`clusterlaunchcontrol.try_cancel`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-clusterlaunchcontrol-try-cancel) PTX instruction. The docs describe it as:

> The `clusterlaunchcontrol.try_cancel` instruction requests atomically cancelling the launch of a cluster that has not started running yet. It asynchronously writes an opaque response to shared memory indicating whether the operation succeeded or failed..

```cpp
if (cg::thread_block::thread_rank() == 0) {
    cg::invoke_one(cg::coalesced_threads(), [&]() {
        ptx::clusterlaunchcontrol_try_cancel(&result, &bar);
    });
    ptx::mbarrier_arrive_expect_tx(
        ptx::sem_relaxed,
        ptx::scope_cta,
        ptx::space_shared,
        &bar,
        sizeof(uint4));
}
```

The second is the [`clusterlaunchcontrol.query_cancel`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-clusterlaunchcontrol-query-cancel) PTX instruction. Again, the docs describe it as:

> Instruction `clusterlaunchcontrol.query_cancel` can be used to decode opaque response written by instruction `clusterlaunchcontrol.try_cancel`.

```cpp
bool success = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result);
if (success) {
    int bx = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x(result);
    int by = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_y(result);
    int bz = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_z(result);
}
```

I am omitting alot of the sync primitives for brevity, but I want to highlight that there are two distinct paired ops: asking for cancellation and checking the response. *This detail will be important later*.

For the Python-inclined like myself, the basic CLC GEMM structure looks like this:

- launch your grid as you would for a regular GEMM: 1 grid entry for every output tile
- instead of doing one K reduction loop and returning, structure the kernel body like a persistent GEMM
- keep an inner `while work.is_valid` loop that polls for new work and continues until there is no more work

But wait, I thought you said we launch a full grid? Indeed we do, but now a resident CTA is free to `steal` work from the launch grid with `try_cancel + query` before a CTA can be launched.

So you basically have two forms of launch-queue draining:

1. The OG path, where CTAs are spawned, consuming a slot.
2. An already resident CTA takes work before a fresh/new CTA is ever spawned.

This is a lil hard to wrap your head around. A log-based tracer is helpful for building intuition on how work gets assigned to each CTA: which tiles were picked up on launch vs. stolen through CLC.

![[widgets/clc-work-distribution.html|Interactive CLC work distribution]]

## Adding to Grouped Gemm

Okay, so how do we adapt this for grouped GEMM? There are two details that make this a lil harder:

1. We are doing a grouped GEMM, duh.
2. The ratio of fake work to real work can be very high.

Before CLC, let's look at how a static scheduler assigns work:

```python
# Assume we have G groups, each of which is doing an M_g x K @ K x N matrix multiply.
# We are using clusters of size (2, 1, 1), so each block_m is basically 2x as big.

# 1. How big is one output "block" of work?
block_size_m = 128 * cluster_shape_mn[0]
block_size_n = mma_tiler_n * cluster_shape_mn[1]

# 2. How many output blocks do we have total?
total_num_clusters = 0
for g in range(G):
    m_g = offs[g] - (offs[g - 1] if g > 0 else 0)
    n_g = N
    total_num_clusters += ceil_div(m_g, block_size_m) * ceil_div(n_g, block_size_n)

# 3. How many persistent workers do we launch?
cluster_size = cluster_shape_mn[0] * cluster_shape_mn[1]
num_workers = num_sms // cluster_size

# 4. What work does worker i get first?
worker_id = bidz
current_work_idx = worker_id

# 5. Static persistent assignment:
# each worker walks the work list with stride = num_workers.
while current_work_idx < total_num_clusters:
    do_cluster_tile(current_work_idx)
    current_work_idx += num_workers
```

The scheduler maps a flat `work_idx -> (group, m_tile, n_tile)`. Two things feed that mapping:

1. `total_num_clusters` — the stopping bound for the persistent work loop. We often do not know this on the host because it depends on runtime routing metadata, so a small prep kernel can walk the groups, write per-group shapes/pointers/strides into device tensors, and accumulate the cluster count.
2. `problem_sizes_mnkl` — the per-group shape table. The decoder needs this because decoding a flat index means knowing how many tiles each group contributes.

Given those, each call does roughly this walk:

```python
# inside the grouped-GEMM decoder, conceptually
local = linear_work_idx
for g in range(start_group_idx, G):
    tiles_g = ceil_div(M_g, block_size_m) * ceil_div(N, block_size_n)

    # Walk groups, subtracting each group's tile count.
    # When `local` fits inside the current group's tiles, we've found it.
    if local < tiles_g:
        mi = local % ceil_div(M_g, block_m)
        ni = local // ceil_div(M_g, block_m)
        return (
            g,
            mi * cluster_shape_m + cta_m_in_cluster,
            ni * cluster_shape_n + cta_n_in_cluster,
        )

    local -= tiles_g
```

This works for the persistent scheduler because the only thing we need at launch time is `num_workers`, and in the kernel body we just dynamically run until the work index gets past `total_num_clusters`.

For CLC, though, the launch grid is the work queue. The grid dimensions have to cover every tile the kernel might ever compute, and we have to pick those dimensions on the host without doing a device-to-host sync. The point is to stay CUDA-graphable, so reading the real `total_num_clusters` back on the host is no bueno.

The worst-case number of tiles is bounded by the shape of the output buffer. In this case we compute it by taking the packed output shape and computing an upper bound:

```python
max_tiles = (cdiv(total_M, block_m) + G - 1) * cdiv(N, block_n)
```

Side note: the default CLC grid layout would often put this tile count in `z`, but CUDA grid `z` is capped at 65535. Once the number of launch slots can exceed that cap, you need to fold the work dimension into `x` instead and reverse the remap in the scheduler.

So now we have our full launch grid, and we couple it with the decoder above to know which tile each work id maps to. At runtime the flow is:

1. CLC hands a resident CTA a linear work id.
2. We feed it through the grouped-GEMM decoder.
3. We get back `(group_idx, cta_m, cta_n)` plus the per-group shape.
4. We do the local matmul.

There is one catch, though. Since we sized the grid off a worst-case upper bound, most real workloads can have CLC handing out tiles that are past `total_num_clusters[0]` -> the actual work.

The hardware does not know which slots are "real" and which are padding. As far as CLC is concerned, every slot in the grid is a valid cluster id to dispatch. That means the kernel itself has to filter: before calling the decoder, we check `linear_work_idx < total_num_clusters[0]` and short-circuit to an invalid tile otherwise.

So now we have two forms of invalid tiles:

1. ones that the CLC scheduler tells us are invalid
2. the bulk of tiles that lie in the padded region needed for worst-case scheduling

## Problem Solved, Now We Profit

![[media/Pasted image 20260419150124.png]]

So I had run all my experiments on my local GB200, which has 2 connected GPUs. And this new scheduler was looking great... until it wasn't.

Lets do just a little bit of math to see why.
Say each rank starts with `M` local tokens, the model has `E` experts, the expert-parallel group has size `EP`, and each token routes to `K` experts.

Under uniform routing, the total routed tokens across the EP group is:

$$M \cdot EP \cdot K$$

Each expert gets about:

$$\frac{M \cdot EP \cdot K}{E}$$

Each rank owns `E / EP` experts, so the expected routed rows per rank is:

$$\frac{M \cdot EP \cdot K}{E} \cdot \frac{E}{EP} = M \cdot K$$

For a dropless backing buffer, though, the unlucky rank has to reserve room for the worst case. One token can route to at most `min(K, E / EP)` local experts on that rank, so the worst-case rows per rank are:

$$M \cdot EP \cdot \min(K, E / EP)$$

So the expected-over-worst ratio is:

$$\frac{M \cdot K}{M \cdot EP \cdot \min(K, E / EP)}$$

When `K <= E / EP`, this simplifies to about `1 / EP`.

So a small EP run hides the issue. With small padding factors, the extra CTAs are mostly harmless: they get stolen quickly, and the ratio of real work to padding is still reasonable.

At larger EP, the scheduler starts spending real time on fake work. The padded region does not run a matmul, but every CTA still pays the fixed CLC fetch/decode cost before the kernel can discover that the tile is past `total_num_clusters[0]`.

Here is the concrete microbenchmark I used to isolate that effect on a 4 gpu GB300 node. Take `2048` input tokens, route each token to `8` experts, and pack the resulting `16384` routed rows into `4` local groups of `4096` rows each. Then run the MXFP8 grouped GEMMs for the MoE projections:

- `w13` fprop/bprop: `4` groups of `4096 × 7168 @ 7168 × 4096`
- `w2` fprop/bprop: `4` groups of `4096 × 2048 @ 2048 × 7168`

For this sweep, the logical GEMM work stays fixed. Only the backing-buffer padding factor changes, which means the useful FLOPs are constant while the number of possible CLC-launched tiles grows. This is what CLC without tail cancellation looks like:

| pad | CLC w13 Fprop | CLC w13 Bprop | CLC w2 Fprop | CLC w2 Bprop |
|---:|---:|---:|---:|---:|
| 8  | 445.8 µs / 2158 TFLOP/s | 580.1 µs / 1658 TFLOP/s | 454.0 µs / 1060 TFLOP/s | 227.7 µs / 2113 TFLOP/s |
| 16 | 632.0 µs / 1522 TFLOP/s | 912.5 µs / 1054 TFLOP/s | 784.3 µs / 613 TFLOP/s | 324.0 µs / 1485 TFLOP/s |
| 32 | 1007.6 µs / 955 TFLOP/s | 1570.4 µs / 613 TFLOP/s | 1441.9 µs / 334 TFLOP/s | 511.8 µs / 940 TFLOP/s |
| 64 | 1759.6 µs / 547 TFLOP/s | 2892.3 µs / 333 TFLOP/s | 2760.9 µs / 174 TFLOP/s | 890.0 µs / 540 TFLOP/s |

Forward and backward get crushed once the scheduler has to chew through that much padding, while wgrad still comes out slightly ahead, which points to the fact that dynamic scheduling is beneficial.

So we need to somehow figure out how to do less fixed work per padded tile, or how to simply stop those empty clusters from being launched in the first place.

@Luca Wehrstedt brought up exactly this idea when we were talking at PTC!

## Superposition

This is my favorite code comment I have ever written, so I'm just going to drop the core of it here:

```python
@dsl_user_op
@cute.jit
def cancel_excess_tiles(self, cancel_mbar_ptrs, cancel_response_ptrs, *, loc=None, ip=None):
    if cutlass.const_expr(self.scheduling_mode == SchedulingMode.CLC):
        max_clc_cancel_attempts = GroupedGemmTileScheduler.MAX_CLC_CANCEL_ATTEMPTS

        # NOTE: CLC cluster tail cancel
        # We overestimate the CLC grid because we need to launch enough blocks to cover
        # the real tokens + padding. For large padding factors this can cause a large
        # slowdown to forward grouped GEMM. Instead of launching tiles that will not be
        # needed, we try to cancel as many blocks as possible to reduce the grid. We do
        # this in parallel across all the leader CTAs; this can race, but again this is
        # all excess not-needed work.
        #
        # Assumptions:
        # 1. Pseudo safe: tiles are pulled in increasing order.
        # 2. Schrodinger's Cat: re-calling try_cancel after it has returned invalid is
        #    okay iff we do not observe this invalid response.
        if self._backend._last_raw_work_is_valid:
            with cute.arch.elect_one():
                excess_cancel_budget = self._backend._total_num_clusters[1]
                real_cluster_count = self._backend._total_num_clusters[0]
                cancel_budget = cutlass.Int32(2)

                # Found via benchmarking padding to num cancel reqs.
                for tier_scale, tier_budget in (
                    (1, 8),
                    (2, 16),
                    (4, 32),
                    (8, max_clc_cancel_attempts),
                ):
                    if excess_cancel_budget >= real_cluster_count * cutlass.Int32(tier_scale):
                        cancel_budget = cutlass.Int32(tier_budget)

                for i in cutlass.range_constexpr(max_clc_cancel_attempts):
                    cute.arch.mbarrier_init(cancel_mbar_ptrs[i], 1)
                cute.arch.mbarrier_init_fence()

                for i in cutlass.range_constexpr(max_clc_cancel_attempts):
                    if cutlass.Int32(i) < cancel_budget:
                        nvvm.clusterlaunchcontrol_try_cancel(
                            cast(Any, cancel_response_ptrs[i]).llvm_ptr,
                            cast(Any, cancel_mbar_ptrs[i]).llvm_ptr,
                            loc=loc,
                            ip=ip,
                        )
```

As we saw from the PTX primitives CLC gives us, there are two distinct phases: one is `try_cancel`, and the other is actually decoding the response.

The way we map our linear index onto the problem space has a very nice property: any index greater than `total_num_clusters` will never be doing real work and will only be doing padding work.

One subtle but `VERY` important note from the docs is:

> Submitting another cancellation request after observing a previously failed request is undefined behavior.

So as long as we don't look in the box, we can submit cancel requests without observing whether they failed.

We capitalize on the fact that tiles are monotonically increasing in grid order. This is kinda sketchy, but also not sketchy enough for FA4/FA3 to critically rely on this fact for their deterministic impls not to deadlock. That means once any CLC fetch is performed and the decoded work is in the invalid region, we know no other fetch will ever return valid work, and the goal now is to shed the workload as fast as possible!

I tried many ways of doing this, including 1 CTA grabbing a lock and owning all the cancels, and all CTAs canceling all remaining work, but Codex and I found after many rounds of experiments the above mechanism, which I call the `capped spray and pray`. Basically, whenever a CTA gets a CLC decode that is in the padded region, launch up to a finite number of cancels and, CRUCIALLY, do not synchronize and decode.

Mechanically:

1. A CTA pulls a valid-but-padded tile from the CLC engine.
2. That breaks the main `while true` work loop.
3. In the clean epilogue phase, the CTA starts spray-and-praying cancels.
4. The CTA exits.
5. If padded tiles remain, they launch, quickly break the loop, and submit their own spray-and-prays.
6. This repeats until the grid is drained.

Soo did it work?

For the same fixed DSv3-style shape:

| pad | cancel | w13 Fprop | w13 Bprop | w2 Fprop | w2 Bprop |
|---:|:---:|---:|---:|---:|---:|
| 8  | off | 445.8 µs / 2158 TFLOP/s | 580.1 µs / 1658 TFLOP/s | 454.0 µs / 1060 TFLOP/s | 227.7 µs / 2113 TFLOP/s |
| 8  | on  | 311.0 µs / 3093 TFLOP/s | 324.1 µs / 2968 TFLOP/s | 189.2 µs / 2542 TFLOP/s | 162.6 µs / 2958 TFLOP/s |
| 16 | off | 632.0 µs / 1522 TFLOP/s | 912.5 µs / 1054 TFLOP/s | 784.3 µs / 613 TFLOP/s | 324.0 µs / 1485 TFLOP/s |
| 16 | on  | 315.5 µs / 3049 TFLOP/s | 330.3 µs / 2913 TFLOP/s | 198.2 µs / 2427 TFLOP/s | 168.3 µs / 2858 TFLOP/s |
| 32 | off | 1007.6 µs / 955 TFLOP/s | 1570.4 µs / 613 TFLOP/s | 1441.9 µs / 334 TFLOP/s | 511.8 µs / 940 TFLOP/s |
| 32 | on  | 322.7 µs / 2981 TFLOP/s | 346.9 µs / 2773 TFLOP/s | 213.6 µs / 2252 TFLOP/s | 167.9 µs / 2865 TFLOP/s |
| 64 | off | 1759.6 µs / 547 TFLOP/s | 2892.3 µs / 333 TFLOP/s | 2760.9 µs / 174 TFLOP/s | 890.0 µs / 540 TFLOP/s |
| 64 | on  | 339.3 µs / 2835 TFLOP/s | 382.5 µs / 2515 TFLOP/s | 254.1 µs / 1893 TFLOP/s | 180.2 µs / 2669 TFLOP/s |

Fprop speedup from cancellation:

| pad | w13 Fprop speedup | w2 Fprop speedup |
|---:|---:|---:|
| 8  | 1.4x | 2.4x |
| 16 | 2.0x | 4.0x |
| 32 | 3.1x | 6.8x |
| 64 | 5.2x | 10.9x |

The important part is the slope. Without cancellation, padding is in the hot path. With cancellation, padding mostly becomes a launch-tail problem.

## A Challenger Scheduler Has Entered the Chat

So the majority of the work was spent on getting CLC to work. The main reason is that we don't have to worry about picking an optimal carveout when overlapping with communication. CLC gives us dynamic workload balancing and pre-emption.

However, if you don't need communication overlap but still want dynamic workload balancing, there is an alternative mechanism.

You launch a static number of persistent workers. Each worker starts from an initial statically assigned tile, and then subsequent tiles are claimed dynamically from a global atomic work counter. That gives dynamic load balancing without relying on CLC's cancellation path.

Once the tile scheduling machinery is abstracted out for CLC, it is fairly straightforward to add this `Dynamic` scheduler.

The Dynamic scheduler is basically a better Static scheduler for the cases where you do not need CLC's pre-emption story:

- it keeps a fixed worker count
- it respects the carveout model
- it dynamically balances work instead of round-robinning fixed strides
- it avoids the padded-tail machinery that CLC needs

So the default choice becomes:

- use `Dynamic` for most grouped GEMMs
- use `CLC` when you specifically need communication overlap / pre-emption

![[widgets/clc-tail-cascade.html|Interactive CLC padded-tail cancellation cascade]]

This was alot of fun. I learned alot and I hope you enjoyed the read.