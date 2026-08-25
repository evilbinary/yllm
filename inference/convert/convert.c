#include "yllm.h"
#include "convert.h"
#include "matvec.h"
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
        if (items[i].layer >= n_layers || items[i].slot >= BLOCK_TENSORS_MTP) {
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

    LlfTensorMeta* metas = (LlfTensorMeta*)ycalloc((size_t)n_layers * BLOCK_TENSORS_MTP, LLF_TENSOR_META_SIZE);
    LlfLayerDir* dir = (LlfLayerDir*)ycalloc(n_layers, LLF_DIR_ENTRY_SIZE);
    uint64_t dir_size = (uint64_t)n_layers * LLF_DIR_ENTRY_SIZE;
    uint64_t cursor = align_up(LLF_HEADER_SIZE + dir_size + (uint64_t)n_layers * BLOCK_TENSORS_MTP * LLF_TENSOR_META_SIZE, LLF_ALIGN);

    uint32_t last_li = (uint32_t)-1;
    for (i = 0; i < n; i++) {
        uint32_t li = items[i].layer;
        uint32_t slot = items[i].slot;
        if (li != last_li) {
            cursor = align_up(cursor, LLF_ALIGN);
            dir[li].offset = cursor;
            last_li = li;
        }
        LlfTensorMeta* tm = &metas[(size_t)li * BLOCK_TENSORS_MTP + slot];
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
            LlfTensorMeta* tm = &metas[(size_t)i * BLOCK_TENSORS_MTP + s2];
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
    write_at(out, LLF_HEADER_SIZE + dir_size, metas, (size_t)n_layers * BLOCK_TENSORS_MTP * LLF_TENSOR_META_SIZE);

    uint8_t* buf = (uint8_t*)ymalloc(1 << 22);
    for (i = 0; i < n; i++) {
        uint32_t li = items[i].layer;
        uint32_t slot = items[i].slot;
        LlfTensorMeta* tm = &metas[(size_t)li * BLOCK_TENSORS_MTP + slot];
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
    /* pad the file to the aligned size declared in the header */
    {
        uint64_t pos = align_up(cursor, LLF_ALIGN);
        if (pos > cursor) {
            uint64_t rest = pos - cursor;
            memset(buf, 0, 4096);
            uint64_t done = 0;
            while (done < rest) {
                uint64_t take = rest - done;
                if (take > 4096) take = 4096;
                write_at(out, cursor + done, buf, (size_t)take);
                done += take;
            }
        }
    }
    free(buf);
    fclose(out);
    free(per); free(metas); free(dir);
    return 0;
}

int convert_model(const char* fmt, const char* in, const char* out, const char* vocab_out,
                  uint32_t max_seq, uint32_t out_dtype, char* err, size_t errlen)
{
    if (strcmp(fmt, "safetensors") == 0) {
        (void)out_dtype; /* 仅 fp16 */
        return convert_safetensors(in, out, max_seq, err, errlen);
    }
    if (strcmp(fmt, "gguf") == 0) {
        return convert_gguf(in, out, vocab_out, max_seq, out_dtype, err, errlen);
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

static int llf_slot_linear_ok(uint32_t layer, uint32_t slot, uint32_t n_blocks)
{
    static const uint32_t k_slots[] = {
        SLOT_Q, SLOT_K, SLOT_V, SLOT_O, SLOT_GATE, SLOT_UP, SLOT_DOWN,
        SLOT_PLE_GATE, SLOT_PLE_PROJ, SLOT_PLE_MPROJ, SLOT_QGATE, SLOT_SSM_OUT
    };
    uint32_t s;
    if (layer == 0 && slot == 0) return 0;           /* embed */
    if (layer == n_blocks + 1) return 0;              /* final norm */
    if (layer == n_blocks + 2 && slot == 0) return 1; /* output */
    for (s = 0; s < sizeof(k_slots) / sizeof(k_slots[0]); s++)
        if (slot == k_slots[s]) return 1;
    return 0;
}

static int owned_push(uint8_t*** owned, int* n_owned, uint8_t* p)
{
    uint8_t** no = (uint8_t**)realloc(*owned, (size_t)(*n_owned + 1) * sizeof(uint8_t*));
    if (!no) return -1;
    *owned = no;
    (*owned)[(*n_owned)++] = p;
    return 0;
}

int conv_items_apply_dtype(ConvItem* items, int n, uint32_t n_blocks, uint32_t out_dtype,
                           uint8_t*** owned, int* n_owned, LlfHeader* h,
                           char* err, size_t errlen)
{
    int p2, n_remap = 0;
    if (out_dtype == DT_KEEP) return 0;
    if (out_dtype == DT_Q4K) {
        for (p2 = 0; p2 < n; p2++) {
            ConvItem* it = &items[p2];
            if (it->dtype == DT_W4B64 && it->ndim == 2 &&
                llf_slot_linear_ok(it->layer, it->slot, n_blocks)) {
                snprintf(err, errlen, "W4B64 -> Q4_K rempack not implemented (tensor %s)", it->name);
                return -1;
            }
        }
        if (h) h->dtype = DT_Q4K;
        return 0;
    }
    if (out_dtype != DT_W4B64) {
        snprintf(err, errlen, "unsupported --dtype (use q4km|w4)");
        return -1;
    }
    for (p2 = 0; p2 < n; p2++) {
        ConvItem* it = &items[p2];
        uint32_t out_r = 0, in_c = 0;
        uint8_t* packed;
        if (it->dtype != DT_Q4K || it->ndim != 2) continue;
        if (!llf_slot_linear_ok(it->layer, it->slot, n_blocks)) continue;
        {
            uint32_t a = it->shape[0], b = it->shape[1];
            uint64_t ra = (a % 256u == 0) ? (uint64_t)(a / 256) * 144 : 0;
            uint64_t rb = (b % 256u == 0) ? (uint64_t)(b / 256) * 144 : 0;
            if (ra && it->nbytes == (uint64_t)b * ra) { in_c = a; out_r = b; }
            else if (rb && it->nbytes == (uint64_t)a * rb) { in_c = b; out_r = a; }
            else continue;
        }
        if ((in_c % W4B64_BLK) != 0) continue;
        {
            size_t nbytes = w4b64_bytes(out_r, in_c);
            packed = (uint8_t*)ymalloc(nbytes);
            if (!packed) {
                snprintf(err, errlen, "oom packing %s", it->name);
                return -1;
            }
            if (w4b64_pack_mat_q4k(packed, it->src + it->src_off, out_r, in_c) != 0) {
                free(packed);
                continue;
            }
            if (owned_push(owned, n_owned, packed) != 0) {
                free(packed);
                snprintf(err, errlen, "oom");
                return -1;
            }
            it->src = packed;
            it->src_off = 0;
            it->dtype = DT_W4B64;
            it->nbytes = (uint64_t)nbytes;
            n_remap++;
        }
    }
    if (n_remap > 0 && h) h->dtype = DT_W4B64;
    return n_remap;
}

int convert_llf_repack(const char* in_path, const char* out_path, uint32_t out_dtype,
                       char* err, size_t errlen)
{
    WMap map;
    LlModel m;
    ConvItem* items = NULL;
    int n = 0, cap = 0, li, ti, rc, n_remap;
    uint8_t** owned = NULL;
    int n_owned = 0;
    LlfHeader h;
    if (out_dtype != DT_Q4K && out_dtype != DT_W4B64) {
        snprintf(err, errlen, "--llf rempack only supports --dtype q4km|w4");
        return -1;
    }
    if (wmap_open(in_path, &map) != 0) {
        snprintf(err, errlen, "cannot open %s", in_path);
        return -1;
    }
    if (llf_read(&map, &m) != 0) {
        wmap_close(&map);
        snprintf(err, errlen, "bad llf %s", in_path);
        return -1;
    }
    h = m.h;
    for (li = 0; li < (int)m.n_layers; li++) {
        for (ti = 0; ti < BLOCK_TENSORS_MTP; ti++) {
            LlfTensorMeta* tm = &m.metas[m.base_idx[li] + ti];
            ConvItem* it;
            if (!tm->name[0] || tm->size == 0) continue;
            if (n + 1 > cap) {
                int ncap = cap ? cap * 2 : 64;
                ConvItem* ni = (ConvItem*)realloc(items, (size_t)ncap * sizeof(ConvItem));
                if (!ni) {
                    free(items);
                    for (ti = 0; ti < n_owned; ti++) free(owned[ti]);
                    free(owned);
                    if (m.base_idx) free(m.base_idx);
                    wmap_close(&map);
                    snprintf(err, errlen, "oom");
                    return -1;
                }
                items = ni;
                cap = ncap;
            }
            it = &items[n];
            memset(it, 0, sizeof(*it));
            it->layer = (uint32_t)li;
            it->slot = (uint32_t)ti;
            it->dtype = tm->dtype;
            it->ndim = tm->ndim;
            memcpy(it->shape, tm->shape, sizeof(it->shape));
            it->nbytes = tm->size;
            snprintf(it->name, sizeof(it->name), "%s", tm->name);
            it->src = (const uint8_t*)map.base + m.dir[li].offset + tm->offset;
            it->src_off = 0;
            n++;
        }
    }
    n_remap = conv_items_apply_dtype(items, n, m.h.n_blocks, out_dtype, &owned, &n_owned, &h, err, errlen);
    if (n_remap < 0) {
        for (ti = 0; ti < n_owned; ti++) free(owned[ti]);
        free(owned); free(items);
        if (m.base_idx) free(m.base_idx);
        wmap_close(&map);
        return -1;
    }
    rc = llf_emit(out_path, &h, items, n, err, errlen);
    for (ti = 0; ti < n_owned; ti++) free(owned[ti]);
    free(owned);
    free(items);
    if (m.base_idx) free(m.base_idx);
    wmap_close(&map);
    if (rc == 0)
        printf("convert_llf_repack: remapped %d tensors, wrote %s\n", n_remap, out_path);
    return rc;
}
