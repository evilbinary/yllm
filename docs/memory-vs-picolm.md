# Memory Usage: yllm vs picolm

Measured on tinyllama-1.1b-chat Q4_K_M (667 MB file) on an Intel i5-10400.

## Runtime Memory (Windows, via Get-Process)

| | yllm (AVX2) | picolm |
|---|---|---|
| Working Set | 606 MB | 646 MB |
| Private Bytes | 49 MB | 50 MB |

Both mmap the model file, so the working set reflects the resident model pages and
both allocate a similar private working set (KV cache + activation buffers).

## Where the memory goes

| Component | Size |
|---|---|
| model file (mmap) | 667 MB (shared, demand-paged) |
| KV cache (fp16) | ~44 MB private |
| activations / scratch | ~5 MB private |
| total private | ~49 MB |

## The real difference: layer-granular scheduling

Both engines mmap the model. The difference is that **yllm is designed to
operate under a memory budget by streaming layers**:

```
engine.c, per forward:
  for each layer i:
    sched_ensure(ws, i)        /* prefetch upcoming layers (madvise WILLNEED) */
    ... compute layer i ...
    sched_release_budget(ws, i) /* release layers beyond the budget */
```

- `--budget NMB` limits resident layers; `ws_release` issues `madvise(DONTNEED)`
  on freed layers (Linux) so physical pages are reclaimed.
- On Windows, `ws_release` is currently a no-op (mmap pages cannot be returned
  without unmap), so the working set stays at the full model size. A future
  per-layer `UnmapViewOfFile`/remap or `madvise`-equivalent can reclaim pages.
- picolm loads the whole model via mmap and keeps all pages resident; it has no
  layer-level residency control.

## Budget behavior (Linux)

With `--budget 200MB`, yllm keeps only ~200 MB of model pages resident and
re-faults layers from disk on demand. picolm would keep all 667 MB resident.

## Residency tracking and adaptive budget (v2)

- **真实驻留位图**: 每个 token 用 `mincore()` 实测每层页缓存驻留状态(`ws.res[]`),
  取代"预取即驻留"的乐观记账。实测值与 `resident estimate` 输出一致。
- **自适应跳过**: 已实测驻留的层不再发 `madvise(WILLNEED)`(decode 稳态零预取 syscall)。
- **预算自适应**: 层数预算由字节预算折算,并按反馈浮动——本 token 发生缺页且
  系统空闲内存富余 → 多驻留一层;空闲内存告急 → 主动缩驻留。
- **回收下限**: `embed` / `final norm` / `lm_head` 为 hot 层恒驻留(豁免回收);
  因此驻留无法低于 hot 层之和(本模型 ~74MB),`--budget 50MB` 时实测驻留 ~86MB。

## Key takeaway

For memory-constrained devices the benefit is **control**: yllm can trade
speed for residency (`budget` knob), whereas picolm is all-or-nothing.
On this test both use the same total when unconstrained.
