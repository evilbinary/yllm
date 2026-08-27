#include "arch.h"
#include "llf.h"

const ArchOps arch_gemma4_ops = {
    .name = "gemma4",
    .id = ARCH_GEMMA4,
    .cpu_batch_prefill = 1,
    .after_embed = arch_gemma4_after_embed,
    .after_embed_batch = arch_gemma4_after_embed_batch,
    .refresh_ple_pp = arch_gemma4_refresh_ple_pp,
    .post_logits = arch_gemma4_post_logits,
    .fwd_block = arch_llama_fwd_block,
    .fwd_block_batch = arch_gemma4_fwd_block_batch,
};
