/* arch.h — 模型图(RoPE / 激活 / SWA / PLE / GDN)。算子在哪台设备上跑见 Device。 */
#ifndef YLLM_ARCH_H
#define YLLM_ARCH_H

#include <stdint.h>

struct Engine;
typedef struct Engine Engine;

typedef struct ArchOps {
    const char* name;
    uint32_t id;
    /* 1 = GPU prefill 失败时走 CPU 批(gemma4/qwen35); 0 = 退回逐 token */
    int cpu_batch_prefill;
    uint32_t prefill_batch_min; /* 0 视为 16 */

    int  (*alloc)(Engine* e);
    void (*free)(Engine* e);
    void (*after_embed)(Engine* e, uint32_t token);
    void (*after_embed_batch)(Engine* e, const uint32_t* tokens, uint32_t B);
    void (*refresh_ple_pp)(Engine* e, uint32_t token);
    void (*post_logits)(Engine* e);

    int (*fwd_block)(Engine* e, uint32_t layer, uint32_t pos);
    int (*fwd_block_batch)(Engine* e, uint32_t layer, uint32_t pos0, uint32_t B);
} ArchOps;

extern const ArchOps arch_llama_ops;
extern const ArchOps arch_qwen_ops;
extern const ArchOps arch_qwen35_ops;
extern const ArchOps arch_gemma4_ops;

const ArchOps* arch_lookup(uint32_t arch_id);

/* CPU graph lives in engine.c for now; later move into inference/arch. */
int  arch_llama_fwd_block(Engine* e, uint32_t layer, uint32_t pos);
int  arch_llama_fwd_block_batch(Engine* e, uint32_t layer, uint32_t pos0, uint32_t B);
int  arch_qwen35_fwd_block(Engine* e, uint32_t layer, uint32_t pos);
int  arch_qwen35_fwd_block_batch(Engine* e, uint32_t layer, uint32_t pos0, uint32_t B);
int  arch_gemma4_fwd_block_batch(Engine* e, uint32_t layer, uint32_t pos0, uint32_t B);
int  arch_gemma4_alloc(Engine* e);
int  arch_qwen35_alloc(Engine* e);
void arch_gemma4_after_embed(Engine* e, uint32_t token);
void arch_gemma4_after_embed_batch(Engine* e, const uint32_t* tokens, uint32_t B);
void arch_gemma4_refresh_ple_pp(Engine* e, uint32_t token);
void arch_gemma4_post_logits(Engine* e);

#endif /* YLLM_ARCH_H */
