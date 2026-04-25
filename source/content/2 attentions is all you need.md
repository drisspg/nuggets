---
title: 2 attentions is all you need
date: 2025-10-01
---

#### Written: October 1, 2025

#### Update from April 25, 2026

Its funny how seemingly stale this doc is so quickly after writing. We have landed FA3 and FA4 support to sdpa (although you kinda need to know where to look), we host official FA3 wheels that are ABI stable. Landed the varlen API and it is now the default in torchtitan for all things where flex isn't needed. We fully landed a new FA4 backend for Flex on blackwell and hopper. AND YET, this feels arcane. Linear Attention is all the rage. DSv4 Dropped this past week and it has its own sparse impls that need alot of helpers (lightning indexers) that arent shipped as part of pytorch. I want flex-linear attention more than anyone else. Although life has gotten away from me and I havent found the time to really figure out how feasible this is. That beings said, I am reading papers :). Pytorch is big and beautiful and that means it can move slower than we like at times. It is the age of infra work, where it can be hard to find ways to empower your users and not compete against them. Suffice it so say striking a balance between innovation and stability is hard but we toil on regardless.

### OG Note

I will assume you know what FlashAttention is and why you should want your GPU implementation to utilize this optimization. With that said, you have your PyTorch modeling code setup and ready to go and now you want to figure out how to actually call this op. You have a few choices. I will start w/ the PyTorch "Native" solutions. This means that you won't need to run any other pip install commands besides what's listed https://pytorch.org/get-started/locally/.

### Choices

1. [F.scaled_dot_product_attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html#torch.nn.functional.scaled_dot_product_attention)
2. [torch.nn.attention.flex_attention](https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html#module-torch.nn.attention.flex_attention)

That's it. That's all you get. Technically you can write an unfused impl w/ mm + softmax + mm and torch.compile will match this to sdpa via the [fuse_attention](https://github.com/pytorch/pytorch/blob/1641606aa4d0eac4ec5422903bb101d590a32abc/torch/_inductor/fx_passes/fuse_attention.py) fx pattern matcher but all roads lead to Rome.

### How do I know what to choose?

The short answer is that if you look at SDPA's docs and you think that its limited feature set aligns w/ what you need -> **importantly you will not need an attn-mask arg**, then use SDPA. If you are reading this post then you likely really care about perf, so use the [kernel_context manager](https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html#torch.nn.attention.sdpa_kernel) to ensure you are getting a fused op. We do reserve the right to remove backends if we need to, this has never happened. We also reserve the right to add new backends. This has happened, notably for CuDNN which was recently added as the `priority` choice for the 2.9 release on H100 and B200. We saw problems where people baked in `sdpa_kernel(SDPBackend.FlashAttention)` and weren't getting the benefits of the new backends. So ideally set this while you are implementing your modeling code and then remove once you are sure you are well covered. I understand the last sentence might not be possible for determinism / code ownership reasons but I still think it is best practice.

#### I can't figure out how to call SDPA without using an attn_mask

Look to flex-attention. I designed the mascot of flex to be a jacked lil matrix but in reality I should have made him do a yoga pose because while fairly strong the premise for flex is its flexibility. Read the blog posts, check out the [attn-gym](https://github.com/meta-pytorch/attention-gym), or even look at the [vllm integration](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/flex_attention.py) which was the most complicated mask/score mods I have written. Most attention variants can be expressed w/ flex-attention. There are a few notable exceptions, mla w/ absorption (what is done for inference), linear attention variants, non softmax normalized, and likely many others because smart people like to do fun things. That being said score_mods + mask_mods + block_masks is quite covering.

#### Why shouldn't I just use FlexAttention

This is a very fair question, and mostly has to do w/ workflows and composability. The primary difference between SDPA + Flex is AOT vs JIT compilation. You know more about your workflow than I do. If you are already bought in to torch.compile then the flex workflow is probably fine. The CuDNN backend has jit compilation, but this is under the hood and less user visible. The runtime overhead is increased because we recommend compiling flex-attention w/ `mode="max-autotune-no-cudagraphs"` . Why? We are generating a new kernel that uses a different amount of hardware resources based on the specific impl, this includes register and smem usage. We have a set of default choices for kernel hyper parameters but they tend to fall short of optimal and autotuning can help guarantee you get better performance.

What's this composability thing you mention? Well PyTorch has always been an Onion, obligatory link to Ed's dispatcher post. The number of layers has increased and abstractions have leaked. Higher order operators + Selective AC + Compile + Export + AOTI + ... this list goes on. We have been and continue to burn down these composability issues for FlexAttention but there are still some edges and in general SDPA being a `dumb` c++ op means it by default has more composition since it is on a more well trodden path.

### Perf

SDPA also typically tends to have better performance than FlexAttention. There are some edge cases when you expand the cross product of hardware and input choices. But roughly speaking SDPA has a way easier job. The args are rigid the kernels are precompiled, the patterns are simple. Flex on A100 was near optimal in the fwd and we have had some mechanisms to allow users to give us more constraints: [KernelOptions](https://docs.pytorch.org/docs/main/nn.attention.flex_attention.html#torch.nn.attention.flex_attention.FlexKernelOptions). For H100 there is a delta between Flex and SOL (FAv3) and for B200 there is a larger delta between Flex and SOL (FAv4 / CuDNN). That being said we are actively working on this and look out for a future post on an exciting update on how we are tackling this problem.

### Variable Sequence Length

A number of people have been asking for variable sequence length support in PyTorch's attention offerings. What most don't know is that we have had this since 2022: https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html#nestedtensor-and-dense-tensor-support via NestedTensors. That being said NT has its own challenges w/ composability and it makes sense to have multiple offerings here. FlexAttention is indeed very flexible and this has been a popular use case for FlexAttention see: [document_masks.py](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/masks/document_mask.py). However, if people only want causal + var-seqlen there exist more specific solutions that fill this need. FlashAttention has had a `var_len` api since the start. I recently put up a very quick prototype showcasing how this can be done in PT https://github.com/pytorch/pytorch/pull/162326. Remember when I said 2 attentions is all you need... In all seriousness I think that it is time for PT to land an official and explicit implementation.

## Conclusion

The attention op has arguably become one of the most important subroutines in all of ML. There exist many variants, many runtimes, and many different requirements that individual users have when they want to invoke it. This is a small slice of that world, focusing on the intra PT options. There exists a wide world of options that compose w/ PT programs provided by third-party libs in the ecosystem. If you have any questions feel free to reach out :)
