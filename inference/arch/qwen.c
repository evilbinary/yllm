#include "arch.h"
#include "yllm.h"
#include "llf.h"

int arch_qwen_fwd_block(Engine* e, uint32_t layer, uint32_t pos)
{
    const uint8_t* base = (const uint8_t*)e->ws.map.base + e->ws.model.dir[layer].offset;
    return arch_llama_fwd_block_at(e, layer, pos, base, e->kv, 1);
}

int arch_qwen_fwd_block_batch(Engine* e, uint32_t layer, uint32_t pos_start, uint32_t B)
{
    return arch_llama_fwd_block_batch_rope(e, layer, pos_start, B, 1);
}

const ArchOps arch_qwen_ops = {
    .name = "qwen",
    .id = ARCH_QWEN,
    .cpu_batch_prefill = 0,
    .prefill_batch_min = 16,
    .gpu_fused = 1,
    .qwen_rope = 1,
    .fwd_block = arch_qwen_fwd_block,
    .fwd_block_batch = arch_qwen_fwd_block_batch,
};
