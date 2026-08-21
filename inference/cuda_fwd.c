/* cuda_fwd.c — CUDA 前向挂接
 *
 * P1: decode 用 load_weights 的权 blob + engine_fwd_block_at(CPU 算子)。
 * Prefill batch 仍走 CPU mmap 路径(权内容相同); KV 与 host 共享(shim)或后续 D2H。
 */
#include "device.h"
#include "yllm.h"

const uint8_t* cuda_layer_base(const Engine* e, uint32_t layer);

static int cuda_fwd_block(Engine* e, uint32_t layer, uint32_t pos)
{
    const uint8_t* base = cuda_layer_base(e, layer);
    uint16_t* kv = e->d_kv ? (uint16_t*)e->d_kv : e->kv;
    if (!base)
        base = (const uint8_t*)e->ws.map.base + e->ws.model.dir[layer].offset;
    return engine_fwd_block_at(e, layer, pos, base, kv);
}

void cuda_attach_fwd(Engine* e)
{
    e->fwd_block = cuda_fwd_block;
    /* fwd_block_batch 保持 CPU(mmap 权); 与 blob 数值一致, KV 走 e->d_kv 别名 */
}
