#include "arch.h"
#include "llf.h"

const ArchOps* arch_lookup(uint32_t arch_id)
{
    switch (arch_id) {
    case ARCH_QWEN:   return &arch_qwen_ops;
    case ARCH_QWEN35: return &arch_qwen35_ops;
    case ARCH_GEMMA4: return &arch_gemma4_ops;
    default:          return &arch_llama_ops;
    }
}
