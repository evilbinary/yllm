#include "yllm.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

const char* llf_dtype_name(uint32_t dtype)
{
    switch (dtype) {
    case DT_F16: return "f16";
    case DT_F32: return "f32";
    case DT_BF16: return "bf16";
    case DT_Q4K: return "q4_k";
    case DT_Q6K: return "q6_k";
    case DT_IQ4XS: return "iq4_xs";
    case DT_Q5K: return "q5_k";
    default: return "?";
    }
}

int llf_read(const WMap* map, LlModel* out)
{
    if (map->size < LLF_HEADER_SIZE) return -1;
    LlfHeader* h = (LlfHeader*)map->base;
    if (memcmp(h->magic, YLLM_MAGIC, 8) != 0 || h->version != YLLM_VERSION) return -1;
    if (h->file_size != map->size) return -1;
    uint32_t n_layers = h->n_blocks + 3;
    uint64_t dir_off = LLF_HEADER_SIZE;
    uint64_t need = dir_off + (uint64_t)n_layers * LLF_DIR_ENTRY_SIZE;
    if (need > map->size) return -1;
    out->h = *h;
    out->n_layers = n_layers;
    out->dir = (LlfLayerDir*)((uint8_t*)map->base + dir_off);
    out->base_idx = (uint32_t*)ymalloc((size_t)n_layers * 4);
    uint32_t i;
    for (i = 0; i < n_layers; i++) {
        out->base_idx[i] = i * BLOCK_TENSORS;
    }
    out->metas = (LlfTensorMeta*)((uint8_t*)map->base + dir_off + (uint64_t)n_layers * LLF_DIR_ENTRY_SIZE);
    uint32_t j;
    for (j = 0; j < n_layers; j++) {
        LlfLayerDir* d = &out->dir[j];
        if (d->offset % LLF_ALIGN != 0) { free(out->base_idx); return -1; }
        if (d->offset + d->size > map->size) { free(out->base_idx); return -1; }
    }
    return 0;
}

int llf_check(const char* path, char* err, size_t errlen)
{
    WMap map;
    if (wmap_open(path, &map) != 0) { snprintf(err, errlen, "cannot open %s", path); return -1; }
    int rc = 0;
    if (map.size < LLF_HEADER_SIZE) { snprintf(err, errlen, "file too small"); rc = -1; goto done; }
    {
        LlfHeader* h = (LlfHeader*)map.base;
        if (memcmp(h->magic, YLLM_MAGIC, 8) != 0) { snprintf(err, errlen, "bad magic"); rc = -1; goto done; }
        if (h->version != YLLM_VERSION) { snprintf(err, errlen, "bad version %u", h->version); rc = -1; goto done; }
        if (h->file_size != map.size) { snprintf(err, errlen, "size mismatch"); rc = -1; goto done; }
        printf("llf: blocks=%u vocab=%u hidden=%u heads=%u kv_heads=%u head_dim=%u max_seq=%u dtype=%s arch=%u\n",
               h->n_blocks, h->vocab, h->hidden, h->n_heads, h->n_kv_heads, h->head_dim, h->max_seq,
               llf_dtype_name(h->dtype), h->arch);
        printf("llf: file_size=%llu (%.2f MB), %u layers\n",
               (unsigned long long)map.size, (double)map.size / 1048576.0, h->n_blocks + 3);
    }
done:
    wmap_close(&map);
    return rc;
}
