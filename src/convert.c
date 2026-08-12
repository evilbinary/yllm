#include "yllm.h"
#include "convert.h"
#include <stdlib.h>
#include <string.h>

uint64_t align_up(uint64_t v, uint64_t a)
{
    return (v + a - 1) & ~(a - 1);
}

#if defined(_WIN32)
#define YFSEEK(f, off) _fseeki64((f), (__int64)(off), SEEK_SET)
#else
#define YFSEEK(f, off) fseeko((f), (off_t)(off), SEEK_SET)
#endif

static void write_at(FILE* f, uint64_t off, const void* data, size_t n)
{
    if (YFSEEK(f, off) != 0) { fprintf(stderr, "seek failed: offset=%llu\n", (unsigned long long)off); exit(1); }
    if (fwrite(data, 1, n, f) != n) { fprintf(stderr, "write failed\n"); exit(1); }
}

int conv_item_compare(const void* a, const void* b)
{
    const ConvItem* x = (const ConvItem*)a;
    const ConvItem* y = (const ConvItem*)b;
    if (x->layer != y->layer) return (int)(x->layer - y->layer);
    return (int)(x->slot - y->slot);
}

int llf_emit(const char* out_path, LlfHeader* h, ConvItem* items, int n,
             char* err, size_t errlen)
{
    if (n <= 0) { snprintf(err, errlen, "no tensors to write"); return -1; }
    qsort(items, (size_t)n, sizeof(ConvItem), conv_item_compare);
    uint32_t n_layers = h->n_blocks + 3;

    uint32_t* per = (uint32_t*)ycalloc(n_layers, 4);
    int i;
    for (i = 0; i < n; i++) {
        if (items[i].layer >= n_layers || items[i].slot >= BLOCK_TENSORS) {
            free(per);
            snprintf(err, errlen, "bad layer/slot in plan");
            return -1;
        }
        per[items[i].layer]++;
    }
    for (i = 0; i < (int)n_layers; i++) {
        if (per[i] == 0) {
            free(per);
            snprintf(err, errlen, "layer %d has no tensors", i);
            return -1;
        }
    }

    LlfTensorMeta* metas = (LlfTensorMeta*)ycalloc((size_t)n_layers * BLOCK_TENSORS, LLF_TENSOR_META_SIZE);
    LlfLayerDir* dir = (LlfLayerDir*)ycalloc(n_layers, LLF_DIR_ENTRY_SIZE);
    uint64_t dir_size = (uint64_t)n_layers * LLF_DIR_ENTRY_SIZE;
    uint64_t cursor = align_up(LLF_HEADER_SIZE + dir_size + (uint64_t)n_layers * BLOCK_TENSORS * LLF_TENSOR_META_SIZE, LLF_ALIGN);

    uint32_t last_li = (uint32_t)-1;
    for (i = 0; i < n; i++) {
        uint32_t li = items[i].layer;
        uint32_t slot = items[i].slot;
        if (li != last_li) {
            dir[li].offset = cursor;
            last_li = li;
        }
        LlfTensorMeta* tm = &metas[(size_t)li * BLOCK_TENSORS + slot];
        memset(tm, 0, sizeof(*tm));
        snprintf(tm->name, sizeof(tm->name), "%s", items[i].name);
        tm->dtype = items[i].dtype;
        tm->ndim = items[i].ndim;
        memcpy(tm->shape, items[i].shape, sizeof(tm->shape));
        tm->offset = cursor - dir[li].offset;
        tm->size = items[i].nbytes;
        cursor += items[i].nbytes;
    }
    for (i = 0; i < (int)n_layers; i++) {
        uint64_t first = dir[i].offset;
        uint64_t loff = align_up(first, LLF_ALIGN);
        uint64_t lend = 0;
        uint32_t s2;
        for (s2 = 0; s2 < per[i]; s2++) {
            LlfTensorMeta* tm = &metas[(size_t)i * BLOCK_TENSORS + s2];
            tm->offset += loff - first;
            if (loff + tm->offset + tm->size > lend) lend = loff + tm->offset + tm->size;
        }
        lend = align_up(lend, LLF_ALIGN);
        dir[i].offset = loff;
        dir[i].size = lend - loff;
        dir[i].n_tensors = per[i];
    }
    h->file_size = align_up(cursor, LLF_ALIGN);

    FILE* out = fopen(out_path, "wb");
    if (!out) {
        free(per); free(metas); free(dir);
        snprintf(err, errlen, "cannot write %s", out_path);
        return -1;
    }
    {
        uint8_t hb[LLF_HEADER_SIZE];
        memset(hb, 0, sizeof(hb));
        memcpy(hb, h, sizeof(*h));
        write_at(out, 0, hb, sizeof(hb));
    }
    write_at(out, LLF_HEADER_SIZE, dir, dir_size);
    write_at(out, LLF_HEADER_SIZE + dir_size, metas, (size_t)n_layers * BLOCK_TENSORS * LLF_TENSOR_META_SIZE);

    uint8_t* buf = (uint8_t*)ymalloc(1 << 22);
    for (i = 0; i < n; i++) {
        uint32_t li = items[i].layer;
        uint32_t slot = items[i].slot;
        LlfTensorMeta* tm = &metas[(size_t)li * BLOCK_TENSORS + slot];
        const uint8_t* sp = items[i].src + items[i].src_off;
        uint64_t done = 0;
        while (done < items[i].nbytes) {
            uint64_t take = items[i].nbytes - done;
            if (take > (1 << 22)) take = 1 << 22;
            memcpy(buf, sp + done, (size_t)take);
            write_at(out, dir[li].offset + tm->offset + done, buf, (size_t)take);
            done += take;
        }
    }
    free(buf);
    fclose(out);
    free(per); free(metas); free(dir);
    return 0;
}

int convert_model(const char* fmt, const char* in, const char* out, const char* vocab_out,
                  uint32_t max_seq, char* err, size_t errlen)
{
    if (strcmp(fmt, "safetensors") == 0) {
        return convert_safetensors(in, out, max_seq, err, errlen);
    }
    if (strcmp(fmt, "gguf") == 0) {
        return convert_gguf(in, out, vocab_out, max_seq, err, errlen);
    }
    snprintf(err, errlen, "unknown source format '%s'", fmt);
    return -1;
}

static uint16_t rnd_f16(uint64_t* s)
{
    uint64_t v = yrng(s);
    float f = ((float)(v >> 40) / 16777216.0f) * 0.02f - 0.01f;
    return f32_to_f16(f);
}

static void fill_random(uint64_t* rng, uint16_t* buf, uint64_t bytes)
{
    uint64_t c;
    for (c = 0; c < bytes / 2; c++) buf[c] = rnd_f16(rng);
}

int convert_dummy(const char* out_path, uint32_t blocks, uint32_t hidden, uint32_t heads,
                  uint32_t kv_heads, uint32_t vocab, uint32_t seq, uint32_t seed, char* err, size_t errlen)
{
    uint64_t rng = ysrand(seed);
    uint32_t inter = ((hidden * 8 / 3 + 63) / 64) * 64;
    uint32_t kv_dim = kv_heads * (hidden / heads);

    LlfHeader h;
    memset(&h, 0, sizeof(h));
    memcpy(h.magic, YLLM_MAGIC, 8);
    h.version = YLLM_VERSION;
    h.arch = ARCH_LLAMA;
    h.n_blocks = blocks;
    h.vocab = vocab;
    h.hidden = hidden;
    h.n_heads = heads;
    h.n_kv_heads = kv_heads;
    h.head_dim = hidden / heads;
    h.max_seq = seq;
    h.dtype = DT_F16;
    {
        float eps = 1e-5f;
        memcpy(&h.norm_eps_bits, &eps, 4);
        float theta = 10000.0f;
        memcpy(&h.rope_theta_bits, &theta, 4);
    }

    uint32_t n_layers = blocks + 3;
    uint64_t dir_size = (uint64_t)n_layers * LLF_DIR_ENTRY_SIZE;
    uint64_t cursor = align_up(LLF_HEADER_SIZE + dir_size + (uint64_t)n_layers * 9 * LLF_TENSOR_META_SIZE, LLF_ALIGN);
    LlfLayerDir* dir = (LlfLayerDir*)ycalloc(n_layers, LLF_DIR_ENTRY_SIZE);
    LlfTensorMeta* metas = (LlfTensorMeta*)ycalloc((size_t)n_layers * 9, LLF_TENSOR_META_SIZE);

    uint32_t sh[9][2] = {
        { vocab, hidden },
        { hidden, hidden },
        { kv_dim, hidden },
        { kv_dim, hidden },
        { hidden, hidden },
        { hidden, hidden },
        { inter, hidden },
        { inter, hidden },
        { hidden, inter },
    };
    static const char* nm[9] = { "norm1", "q", "k", "v", "o", "norm2", "gate", "up", "down" };

    uint32_t li;
    for (li = 0; li < n_layers; li++) {
        uint32_t nt;
        uint32_t r0;
        if (li == 0) { nt = 1; r0 = 0; }
        else if (li == blocks + 1) { nt = 1; r0 = 5; }
        else if (li == blocks + 2) { nt = 1; r0 = 0; }
        else { nt = 9; r0 = 0; }
        dir[li].offset = cursor;
        dir[li].n_tensors = nt;
        uint32_t s2;
        for (s2 = 0; s2 < nt; s2++) {
            uint32_t idx = (li == 0 || li == blocks + 1 || li == blocks + 2) ? r0 : s2;
            LlfTensorMeta* tm = &metas[(size_t)li * 9 + s2];
            memset(tm, 0, sizeof(*tm));
            snprintf(tm->name, sizeof(tm->name), "%s", li == blocks + 2 ? "output" : nm[idx]);
            tm->dtype = DT_F16;
            tm->ndim = 2;
            tm->shape[0] = sh[idx][0];
            tm->shape[1] = sh[idx][1];
            tm->offset = cursor - dir[li].offset;
            tm->size = (uint64_t)sh[idx][0] * sh[idx][1] * 2;
            cursor += tm->size;
        }
        cursor = align_up(cursor, LLF_ALIGN);
        dir[li].size = cursor - dir[li].offset;
    }
    h.file_size = cursor;

    FILE* out = fopen(out_path, "wb");
    if (!out) { free(dir); free(metas); snprintf(err, errlen, "cannot write %s", out_path); return -1; }
    {
        uint8_t hb[LLF_HEADER_SIZE];
        memset(hb, 0, sizeof(hb));
        memcpy(hb, &h, sizeof(h));
        write_at(out, 0, hb, sizeof(hb));
    }
    write_at(out, LLF_HEADER_SIZE, dir, dir_size);
    write_at(out, LLF_HEADER_SIZE + dir_size, metas, (size_t)n_layers * 9 * LLF_TENSOR_META_SIZE);

    uint16_t* buf = (uint16_t*)ymalloc(1 << 20);
    for (li = 0; li < n_layers; li++) {
        uint64_t total = 0;
        while (total < dir[li].size) {
            uint64_t take = dir[li].size - total;
            if (take > (1 << 20)) take = 1 << 20;
            fill_random(&rng, buf, take);
            write_at(out, dir[li].offset + total, buf, (size_t)take);
            total += take;
        }
    }
    free(buf);
    fclose(out);
    free(dir);
    free(metas);
    return 0;
}

int dummy_vocab(const char* out_path, uint32_t vocab, char* err, size_t errlen)
{
    FILE* f = fopen(out_path, "wb");
    if (!f) { snprintf(err, errlen, "cannot write %s", out_path); return -1; }
    fprintf(f, "%u\n", vocab);
    uint32_t i;
    for (i = 0; i < vocab; i++) {
        if (i < 256) fprintf(f, "%u\t<0x%02X>\n", i, i);
        else fprintf(f, "%u\t<t%u>\n", i, i);
    }
    fclose(f);
    return 0;
}
