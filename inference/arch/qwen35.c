#include "arch.h"
#include "yllm.h"
#include "llf.h"
#include <stdlib.h>

int arch_qwen35_alloc(Engine* e)
{
    LlModel* m = &e->ws.model;
    uint32_t n_gdn = 0;
    uint32_t conv_chan = 0, kwidth = 0;
    uint32_t n_vheads = 0, hvd = 0;
    uint32_t li;
    for (li = 1; li <= m->h.n_blocks; li++) {
        const LlfTensorMeta* mt = &m->metas[m->base_idx[li]];
        if (mt[SLOT_SSM_CONV1D].size > 0) {
            n_gdn++;
            if (conv_chan == 0) {
                conv_chan = mt[SLOT_SSM_CONV1D].shape[1];
                kwidth = mt[SLOT_SSM_CONV1D].shape[0];
                n_vheads = mt[SLOT_SSM_A].shape[0];
                hvd = mt[SLOT_SSM_NORM].shape[0];
            }
        }
    }
    if (n_gdn > 0 && conv_chan > 0 && n_vheads > 0 && hvd > 0) {
        e->ssm_state = (float*)ycalloc((size_t)n_gdn * n_vheads * hvd * hvd, 4);
        e->ssm_conv = (float*)ycalloc((size_t)n_gdn * (kwidth > 0 ? kwidth : 1) * conv_chan, 4);
        e->scratch = (float*)ymalloc(65536 * 4);
    }
    return 0;
}

const ArchOps arch_qwen35_ops = {
    .name = "qwen35",
    .id = ARCH_QWEN35,
    .cpu_batch_prefill = 1,
    .prefill_batch_min = 16,
    .alloc = arch_qwen35_alloc,
    .fwd_block = arch_qwen35_fwd_block,
    .fwd_block_batch = arch_qwen35_fwd_block_batch,
};
