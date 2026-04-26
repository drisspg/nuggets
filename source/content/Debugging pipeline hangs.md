---
title: Debugging pipeline hangs
date: 2026-02-13
tags:
  - tool-tips
---

#### Written: February 13, 2026

I spent a long time yesterday reading code and spending alot of money on agents trying to find a kernel hang in fa4.

Ultimately I ended up working on this skill

````markdown
---
name: cuda-hang-dump
description: Reproduces a CUDA hang, triggers a user coredump, and extracts kernel PC to source line via cuda-gdb. Use when a GPU test hangs and Ctrl-C is ineffective.
---

# CUDA Hang Dump

Use this workflow when a CUDA kernel hangs and you need source-level evidence.

## Inputs

- Repro command (usually a single pytest target)
- Optional timeout seconds (default 180)

## Steps

1. Launch repro with coredump env:

```zsh
CUTE_DSL_LINEINFO=1 \
CUDA_ENABLE_USER_TRIGGERED_COREDUMP=1 \
CUDA_COREDUMP_PIPE="/tmp/cuda_hang_pipe_%h.%p.%t" \
CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 \
CUDA_COREDUMP_SHOW_PROGRESS=1 \
CUDA_COREDUMP_GENERATION_FLAGS='skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory' \
CUDA_COREDUMP_FILE="/tmp/cuda_hang_lineinfo_%h.%p.%t" \
<REPRO_COMMAND>
```

2. Identify child Python PID.
3. Resolve active pipe via `/proc/<pid>/fd`.
4. Trigger dump:

```zsh
dd if=/dev/zero bs=1M count=1 > <PIPE_PATH>
```

5. Open dump with cuda-gdb:

```zsh
/usr/local/cuda-13.1/bin/cuda-gdb -batch \
  -ex "set pagination off" \
  -ex "target cudacore <COREDUMP_FILE>" \
  -ex "where" \
  -ex "x/8i \$pc-32" \
  -ex "info line *\$pc" \
  -ex "info cuda kernels"
```

6. Report:
   - stuck kernel symbol
   - source file:line
   - current wait/spin instruction
7. Kill lingering repro/dd processes.

## Output Template

```markdown
Kernel: <symbol>
PC: <address>
Source: <file:line>
Instruction: <sass around pc>
Interpretation: <what wait/spin implies>
```
````

And with it me and opus 4.6 were able to pinpoint the source of the stall in ~20 mins

PR: [https://github.com/Dao-AILab/flash-attention/pull/2258](https://github.com/Dao-AILab/flash-attention/pull/2258)
