# LLF vs GGUF: Format Comparison

## Overview

yllm uses a custom model format called **LLF** (YLLM LLF, magic `YLLMLLF1`) instead of the standard GGUF format used by llama.cpp. This document explains the structural differences and the design advantages of LLF for yllm's use case.

## GGUF (llama.cpp standard)

- **Header**: magic + version + tensor/metadata counts
- **Metadata**: KV array (architecture, hyper-parameters, tokenizer, chat template, etc.)
- **Tensor table**: for each tensor — name string, dims, type, **offset relative to data_start**
- **Data section**: all tensors, aligned to `general.alignment` (usually 32 bytes)
- Tensors are located by **string name lookup** (e.g. `blk.3.attn_q.weight`)

```
[ header | metadata kv | tensor table | data_start ... tensor data ... ]
```

## LLF (yllm custom)

- **Fixed 128-byte header**: magic + arch + dims + max_seq + rms/rope params
- **Layer directory (dir)**: one 32-byte entry per layer → `{ offset, size, n_tensors }`
- **Tensor metadata**: **fixed 9 slots per layer** (norm1/Q/K/V/O/norm2/GATE/UP/DOWN); the slot index *is* the tensor semantic
- **Layer-aligned data**: each layer starts on a 4096-byte boundary (LLF_ALIGN)

```
[ header | layer dir | per-layer metas (9 slots) | layer 0 data | layer 1 data | ... ]
```

## Structural Differences

| Aspect | GGUF | LLF |
|---|---|---|
| Tensor location | string name → hash/linear lookup | **fixed slot index**, `base + mt[SLOT_Q].offset` |
| Layer boundary | inferred by walking tensors | **explicit dir entry** per layer |
| Alignment | 32 B inside tensors | 4096 B between layers |
| Parse cost | string comparisons per access | O(1) indexed |
| Loader dependency | generic spec | yllm-specific |

## Why LLF is Better for yllm

### 1. O(1) tensor addressing
GGUF stores tensor names as strings; loading requires string lookup per tensor. LLF uses fixed slot indices (`SLOT_Q=1`, `SLOT_K=2`, ...), so `mt[SLOT_Q].offset` directly gives the Q projection weights with zero parsing.

### 2. Layer-granular memory management
This is the core design goal. yllm streams weights via mmap and schedules **per-layer** prefetch/release (see `docs/design-mmap-layer-streaming.md`):

```c
/* engine.c: per-layer scheduling */
for (i = 0; i < m->n_layers; i++) {
    sched_ensure(ws, i);                    /* prefetch next layers */
    const uint8_t* base = map.base + m->dir[i].offset;  /* O(1) layer base */
    ...
    sched_release_budget(ws, i);            /* release budget-pressured layers */
}
```

With GGUF, the loader would need to reconstruct layer boundaries from scattered tensor offsets. LLF's `dir[layer]` gives the whole layer range in one read, enabling:
- **madvise(WILLNEED)** on a whole layer (page-granular prefetch)
- **madvise(DONTNEED)** to release layers under a memory budget
- layer-boundary alignment to 4096 → layer ranges align to OS pages

### 3. Fixed meta size → simpler mmap layout
Every layer's 9 tensors live in a fixed-size metadata block, so the meta table address is computed once (`header + n_layers * dir_entry_size`). No dynamic allocation or hash table needed at load.

### 4. Faster decode startup
The engine reads a handful of fixed structs; no scanning of thousands of tensor-name strings.

## Trade-offs

- **Not portable**: only yllm reads LLF; GGUF has ecosystem-wide support.
- **Fixed 9-slot layout** assumes a standard transformer block (norm1/Q/K/V/O/norm2/GATE/UP/DOWN). Non-standard architectures (e.g. extra tensors) need format extension.
- **4096-byte layer alignment** wastes some space for tiny models (usually negligible, < 1%).
- GGUF's `general.alignment` of 32 bytes is tighter, so GGUF files are marginally smaller.

## Summary

| | GGUF | LLF |
|---|---|---|
| Tensor addressing | string lookup | slot index |
| Layer streaming | not native | first-class |
| Alignment | 32 B | 4096 B |
| Ecosystem | broad | yllm only |
| Load complexity | moderate | minimal |
