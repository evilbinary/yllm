#include "arch.h"
#include "llf.h"

const ArchOps arch_qwen35_ops = {
    .name = "qwen35",
    .id = ARCH_QWEN35,
    .cpu_batch_prefill = 1,
    .fwd_block = arch_qwen35_fwd_block,
    .fwd_block_batch = arch_qwen35_fwd_block_batch,
};
