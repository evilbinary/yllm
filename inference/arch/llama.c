#include "arch.h"
#include "llf.h"

const ArchOps arch_llama_ops = {
    .name = "llama",
    .id = ARCH_LLAMA,
    .cpu_batch_prefill = 0,
    .fwd_block = arch_llama_fwd_block,
    .fwd_block_batch = arch_llama_fwd_block_batch,
};
