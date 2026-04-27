---
title: Adding kernels to PyTorch core is the least leveraged thing
date: 2026-04-26
tags:
  - pytorch
---

#### Written: April 26, 2026

Addendum


This one is kinda interesting. The note below I wrote on 2025-02-26 right after DSv3 came out. And now its the sunday of DSv4's release. If anything we are less apt as PyTorch core to add all the new kernels needed for full efficiency. `We have won` is maybe less true. There are many amazing teams that are capable of doing immense model co-design, looking at you whales. My honest take, as someone who still feels relatively new to PT, PyTorch was never really designed for this regime. It was built for the age of arch exploration not for the age of scale. In those days installing cuda was a huge blocker - honestly still kinda is but people have gotten better at infra. To verify new research ideas they need to be tested at scale, and in order to do this performance is now a perquisite.  Gpus aint cheap! Kernels can and do really have immense power to unlock ideas. Flash Attention, Paged Attention, Cuda-Graphable Grouped Gemm, the list goes on.

I think as whole we are still figuring out how to provide better lower level abstractions that enable individual researchers and labs to scale alike. We have some nice newish things, symmetric memory, flex-attention, our investment in custom op authoring. To the original point of the post below - we actually have made meaningful strides! We recently added [torch.native._ops](https://github.com/pytorch/pytorch/tree/main/torch/_native). We like authoring in python dsl's as much as anyone and by making it easier to meet the kernel authoring community where it is heading PT can increase its velocity here. If we dont get reverted for every new impl we try and add 🙃

All of these ideas can provide value across the scaling lifecycle. But, Innovation is hard work. As for what drives me - good abstractions are out there, just waiting to be plucked from ether. 



# Adding kernels to PyTorch core is the least leveraged thing

My completely hot take is that adding kernels to PyTorch core is the least leveraged thing we can do as a framework - Yes I know I was the one who added FAv2 to core. What people are missing is that every new DeepSeek drop has fundamentally been written and designed to integrate w/ PyTorch. We have won.  We should take a second and learn from the places they had to work around our abstractions. IMO the main goal should be answering - How can we make their lives easier to create deepseeekv4. Of course, we ultimately want to benefit from these innovations and should work to democratize them for the broader community. This doesn't necessarily mean reshipping the same technique, but rather ensuring the external ecosystem receives the same benefits as core components. This approach creates the flywheel  we want, where new techniques are built, designed, and shipped with day-one PyTorch support. As a result, the path to production becomes as short as possible. My two cents. Time to get back to shipping FAv3 in core (ohh the irony 😉)
