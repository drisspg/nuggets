---
title: QLoRA in Pure PyTorch
date: 2024-08-11
---

#### Written: August 11, 2024

### TLDR:

I implemented the main memory savings component of the QLoRA paper in pure PyTorch and by using torch.compile I was able to see ~2x performance increase.

LLMs are everywhere. From [] to []. More and more individuals are wanting to take large foundational models and finetune on their own datasets. The paper [LoRA](https://arxiv.org/abs/2106.09685)which came out in October of 2021 proposed a mechanism for doing this. Instead of finetuning every weight of the base model, "freeze the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture". This technique is starting to gain more widespread adoption through the [PEFT](https://github.com/huggingface/peft)library from hugging face.

![[star-history-2023612.png]]

[QLoRA](https://arxiv.org/abs/2305.14314)was a paper released in May 2023 that builds upon this technique by adding quantization into the mix. The main idea being to statically quantize the pre-trained model weights to 4bit, capitalizing on the fact that model weights tend to be distributed normally. They then use LoRA finetuning to recover any performance degradation by quantizing the pre-trained weights.

The paper makes proposed two techniques for memory savings 4-bit NormalFloat Quantization and Double Quantization. In this note I implement both these techniques in pure PyTorch.

### NF4Tensor

[BitsandBytes](https://github.com/TimDettmers/bitsandbytes) has implemented the quantization techniques that are used in the [QloRA code](https://github.com/artidoro/qlora) and [PEFT](https://github.com/huggingface/peft). We can find the custom CUDA code written to support the [NormalFloat4](https://github.com/TimDettmers/bitsandbytes/blob/ac5550a0238286377ee3f58a85aeba1c40493e17/csrc/kernels.cu#L223) quantization type. I use this code as a comparison point for the pure PyTorch implementation. The implementation is 170 (uncommented) lines of python code. The code can be found here: [Here](https://github.com/drisspg/transformer_nuggets/blob/c1c932d0399a87c47442aa7c404d73d47776643e/transformer_nuggets/quant/qlora.py#L42)

The pseudo-code for converted the NF4 representation can be found below:

```Python
# 1.) Read full-precision pre-trained weight in blocks of 64
# 2.) Quantize each element in the block using a pre-computed set of int4 types.
# 3.) Store the the weight in int4. And dequantize on the fly to original dtype when doing computation.
```

The actual code for 2) and 3) can be found below:

```Python
@staticmethod
def convert_to_norm_float_weight(

inpt_tensor: torch.Tensor, n_blocks: int, block_size: int, nf4: torch.tensor

) -> torch.Tensor:

"""Convert a tensor to the normalized float weight format"""

flattened_tensor = inpt_tensor.flatten()

# Since we are using uint8 we will encode 2 entries per byte

numel = inpt_tensor.numel()

assert (

numel % 2 == 0

), "Number of elements must be even just to not have to think about the end"

# Reshape the flattened tensor into blocks of size self.block_size

blocks = flattened_tensor.view(n_blocks, block_size)

# Scale the blocks

scalers = get_block_absmax(inpt_tensor.flatten(), block_size)

scales = scalers.unsqueeze(-1).expand(n_blocks, block_size)

scaled_blocks = blocks / scales

quantized_blocks = NF4Tensor.quantize_tensor_nearest(scaled_blocks.flatten(), nf4)

# Combine the quantized elements into uint8 values

combined_blocks = quantized_blocks[::2] << 4 | quantized_blocks[1::2]

return combined_blocks.to(torch.uint8)
```

```Python
def get_original_weight(self) -> torch.Tensor:

"""Get the original weight from the normalized float weight format"""

# since we are using uint8 we will decode 2 entries per byte

# Shift elements down 4 and select out the bottom 4 bits

first_elements = (self.quantized_data >> 4).to(torch.long)

second_elements = (self.quantized_data & 0b1111).to(torch.long)


# Dequantize every element

dequantized_first = self.dequantize(first_elements, self.nf4)

dequantized_second = self.dequantize(second_elements, self.nf4)


# Build up matrix of scalers repeated for each element in the block

# Since first and second elements make up a full block, so

# we expand out to half the size of the full block

scalers = self.dequantize_scalers(

self.quantized_scalers, self.quantization_factor, self.scaler_block_size

)

repeated = scalers.unsqueeze(-1).expand(scalers.size(0), self.block_size // 2)



scaled_first = dequantized_first * repeated.flatten()

scaled_second = dequantized_second * repeated.flatten()


# Flip them to be vertical and them stack them together horizontally

# Upon flattening this will interleave the elements

scaled_first = scaled_first.unsqueeze(-1).transpose(0, 1)

scaled_second = scaled_second.unsqueeze(-1).transpose(0, 1)

return torch.stack([scaled_first, scaled_second], dim=-1).reshape(self.original_shape)
```

### DoubleQuantization

Since the block_size is relatively small for large tensors, for instance one weight in the llama 7b MLP will have n_blocks = 704,512. Storing the abs max scaling factors can quickly start add significant memory pressure. The trick was to quantize these scalers again and store the reduced precision format of these. I have also implemented that if interested feel free to take a look at the `double_quantize_scalers` method of NF4Tensor.

### Numerical Differences

I did my best to replicate what was implemented in the paper but my implementation is not bitwise identical to BitsandBytes(BnB) implementation.

1.  During quantization of the initial pre-trained tensor it is possible for an entry to be equi-distant to quantized representations. My implementation breaks ties by choosing the lower index while I found that BnB picks the higher index.
2.  For some entries I found that my implementation picks a quantized representation, that after de-quantization, is actually closer to the original value. I did not dig deeper into BnBs CUDA code to determine why that is.

That being said I wrote a few tests that measure the reconstruction accuracy between my implementation and BnB and both reconstruct the original tensor. The worst case de-quantization is approximately equal between the two implementations. The beauty of LoRA is that it can/should learn to account for the degradation of performance caused by quantization, so I was satisfied to stop going down this rabbit hole.

### Time for the Numbers

I did lots and lots of measuring. I measured individual de-quant performance, the de-quant performance on for a MLP(essentially 3 NF4Tensors) and on lit-llama that I modified to have every MLP use my NF4Tensor to back its weights.

I compared between, the "original" (no quantization), BnB's `bnb.nn.LinearNF4` class and my `NF4MLP`.

For Instance my NF4MLP class looks like

```Python
class NF4MLP(nn.Module):

def __init__(self, config: LLaMAConfig) -> None:

super().__init__()

weight1, weight2, weight3 = qlora.get_mlp_weights(config.n_embd)

self.w1 = qlora.NF4Tensor.from_tensor(weight1)

self.w2 = qlora.NF4Tensor.from_tensor(weight2)

self.w3 = qlora.NF4Tensor.from_tensor(weight3)


def forward(self, x: torch.Tensor) -> torch.Tensor:

x = F.silu(F.linear(x, self.w1.get_original_weight())) * F.linear(

x, self.w2.get_original_weight()

)

x = F.linear(x, self.w3.get_original_weight())

return x
```

Below is the plot of timings for the MLP in isolation. I swept over the embed dimensions found in the [Model Card](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md)

```Python
# https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md

# LLama 7b, 13b, 33b, 65b

embed_dims = [4096, 5120, 6656, 8192]

bszs = [8, 16, 32]

seqlens = [128, 256, 512]
```

![[qlora_mlp_embed_dim_4096.png]]

I have a full sweep of plots but they all essentially look like this. As you can see the MLP in full precision is fastest, there is no de-quantization that needs to be done. The eager NFTensor implementation in eager is much slower that bitsandbytes, especially in smaller batch_size \* seqlen sizes. This quickly becomes a big bottle neck when having to store the activations on device.

**But with torch.compile the pure PyTorch implementation is faster on every config!** The bitsandbytes implementation does not work with torch.compile due to it being a custom CUDA implementation so applying that this nn.Module does not provide any performance benefits.

### Testing on Lit-LLama

I was looking for a model to test end-to-end performance savings on and settled on [Lit-llama](https://github.com/Lightning-AI/lit-llama/blob/main/lit_llama/model.py)It is a fairly straightforward implementation, and it uses SDPA so I was biased 😏. I update the config to choose between one of 3 MLP implementations - Full Precision(Unchanged), BitsandBytes, and my NF4Mlp.

Again I have a slew of data for this one but I will attach the profiles below. The following timings can be found by inspecting the profile functions. This were generated on the llama 7B config with an input of batchsize=4 and seqlen =256

Original - 69.724 ms
BitsandBytes - 164.450 ms
NFTensor - 81.686 ms

_Note: These MLP do not actually have the adapater weights added. Since I was more curious about measuring the impact of the quantization and the adapters would be identical between models. This is a good follow up _

### Memory Profiling

You might be thinking, like I was, this is amazing, torch compile makes expressing QloRA both more readable, and more performant! I wanted to confirm that indeed we were seeing identical memory savings. Zachary Devito has some amazing write ups and tools on inspecting the memory usage of a PyTorch program. Below are images of the memory timelines for the three runs:

Original:
![[llama_original_compile_no_grad_compile_false.png]]

BitsandBytes:
![[llama_bnb_no_grad_compile_false.png]]

NF4Mlp:
![[llama_nf4_compile_True.png]]

Thats weird!! Both BnB and the original appear to have consistent memory usage with some little spikes. I was scratching my head on this one, thinking that maybe the GC was not kicking in, after much head scratching and a extremely helpful VC with Joel we were able to figure out that autograd was saving the full precision de-quantized tensor for backward! Crucially we can see that baseline for both BNB and NF4Mlp is around 7GB but the NF4MLP grows, at a rate of 86MiB per allocation - the size of each MLP weight 11008 _ 4096 _ 2 Bytes, to reach a maximum around 18Gb. In fact we overshoot the original model since we are storing the additional scalers + 5/4 the MLP weight size.

To test this theory I ran the model again under `torch.no_grad()` and low and behold:
![[llama_nf4_no_grad_compile_True.png]]

We see the correct total memory usage of this!

So what is the real solution, we should wrap the NF4Linears in their own autograd Fucntions:

```Python
class LinearNF4(torch.autograd.Function):

@staticmethod

def forward(ctx, input: torch.Tensor, weight: NF4Tensor):

ctx.nf4_weight = weight

return F.linear(input, weight.get_original_weight())



@staticmethod

def backward(ctx, grad_output):

weight: NF4Tensor = ctx.nf4_weight

return grad_output @ weight.get_original_weight(), None




def linear_nf4(input: torch.Tensor, weight: NF4Tensor) -> torch.Tensor:

return LinearNF4.apply(input, weight)
```

### Some Rough Edges:

While attempting to torch.compile this update version using torch.autograd.Function the following dynamo error is raise:
`torch._dynamo.exc.Unsupported: HigherOrderOperator with body that accepts non-Tensors as input`. I spoke with Richard Zou and says that supported types can be expanded. (Point to issue if exists or create one.I In theory this type can be fully passed from forward to backward using only torch.Tensors, ints, and torch.Size.

As is common in PEFT existing pre-trained models are updated using [module swapping](https://github.com/huggingface/peft/blob/189a6b8e357ecda05ccde13999e4c35759596a67/src/peft/tuners/lora.py#L225). I would argue that this is likely not the ideal user experience. And instead, subclassing torch.Tensor and replacing the weight parameter on torch.nn.Linear with this type would be a more seamless experience. That being said I think simply getting the autograd function working would be enough since users will need to add the adapter layers to run QLoRA.

### Future Work

I had no intentions of producing actually usable code. I was reading the QLoRA paper, taking notes and thought that it me beneficial to try and implement this as a learning experience. The debug class was my first attempt and as you can see very inefficient. With some more rounds of polish I was able to convert this class to something outperfoming the handrolled SOTA implementation.

That is the power of torch.compile. I had never written a quantization routine and in a matter of a few days I was able to match and exceed code from experts. (LITTLE HEAVY HANDED BUT KIND OF TRUE). That is saying nothing of its extensibility. If I wanted to change the NF4 to be pre-determined on my dataset, or quantize in 2 bits. I could do it in hours and best of all never look at a .cu file.

Concrete Items:

- Get torch.autograd.Function working with this type.
- Apply technique ore components (in-projection, out-projection of CausalAttention block.)
- Apply to more models - Project Blueberries.
- Work through a full end-to-end LoraFinetuning now that Ed has added this to torchbench. [PR](https://github.com/pytorch/benchmark/pull/1730)
- Submit upstream PRs to BitsandBytes? Lit-Llama?

#### Thanks

I want to say a quick thanks to Christina for pushing to make this component as performant as possible and identifying places to speedup. and Joel for being the best rubber duck ever!
