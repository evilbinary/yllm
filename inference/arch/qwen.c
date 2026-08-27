#include "arch.h"
#include "llf.h"

/* 块图与 llama 相同; RoPE 在 engine_fwd_block_at / CUDA 里按 header.arch 分支 */
const ArchOps arch_qwen_ops = {
    .name = "qwen",
    .id = ARCH_QWEN,
    .cpu_batch_prefill = 0,
    .prefill_batch_min = 16,
    .fwd_block = arch_llama_fwd_block,
    .fwd_block_batch = arch_llama_fwd_block_batch,
};
