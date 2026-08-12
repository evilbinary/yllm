#include "yllm.h"
#include "convert.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
    const uint8_t* p;
    const uint8_t* end;
    int err;
    int be; /* big-endian (gguf v3 files may be BE; detected from version field) */
} GB;

static uint32_t yb_bswap32(uint32_t v)
{
    return ((v & 0xFFu) << 24) | ((v & 0xFF00u) << 8) | ((v >> 8) & 0xFF00u) | (v >> 24);
}

static uint64_t yb_bswap64(uint64_t v)
{
    return ((uint64_t)yb_bswap32((uint32_t)v) << 32) | yb_bswap32((uint32_t)(v >> 32));
}

static uint32_t gb_u32(GB* b)
{
    uint32_t v;
    if (b->p + 4 > b->end) { b->err = 1; return 0; }
    memcpy(&v, b->p, 4);
    if (b->be) v = yb_bswap32(v);
    b->p += 4;
    return v;
}

static uint64_t gb_u64(GB* b)
{
    uint64_t v;
    if (b->p + 8 > b->end) { b->err = 1; return 0; }
    memcpy(&v, b->p, 8);
    if (b->be) v = yb_bswap64(v);
    b->p += 8;
    return v;
}

static double gb_f64(GB* b)
{
    uint64_t v = gb_u64(b);
    double d;
    memcpy(&d, &v, 8);
    return d;
}

static char* gb_str(GB* b)
{
    uint64_t len = gb_u64(b);
    if (b->err || b->p + len > b->end) { b->err = 1; return NULL; }
    char* s = (char*)ymalloc((size_t)len + 1);
    memcpy(s, b->p, (size_t)len);
    s[len] = 0;
    b->p += len;
    return s;
}

static void gb_skip(GB* b, uint64_t n)
{
    if (b->p + n > b->end) { b->err = 1; return; }
    b->p += n;
}

/* gguf value types */
enum {
    GVT_U8 = 0, GVT_I8 = 1, GVT_U16 = 2, GVT_I16 = 3, GVT_U32 = 4, GVT_I32 = 5,
    GVT_F32 = 6, GVT_BOOL = 7, GVT_STR = 8, GVT_ARRAY = 9, GVT_U64 = 10, GVT_I64 = 11, GVT_F64 = 12
};

typedef struct {
    char* arch;
    uint32_t n_blocks;
    uint32_t hidden;
    uint32_t heads;
    uint32_t kv_heads;
    float freq_base;
    float rms_eps;
    uint32_t context_len;
    uint32_t alignment;
    char** tokens;
    uint32_t n_tokens;
    uint32_t cap_tokens;
    float* scores;
    uint32_t n_scores;
    uint32_t cap_scores;
} GgufMeta;

static void gg_tokens_grow(GgufMeta* g, uint64_t need)
{
    if (need <= g->cap_tokens) return;
    uint32_t cap = g->cap_tokens ? g->cap_tokens : 1024;
    while (cap < need) cap *= 2;
    g->tokens = (char**)realloc(g->tokens, (size_t)cap * sizeof(char*));
    if (!g->tokens) exit(1);
    g->cap_tokens = cap;
}

static void gg_scores_grow(GgufMeta* g, uint64_t need)
{
    if (need <= g->cap_scores) return;
    uint32_t cap = g->cap_scores ? g->cap_scores : 1024;
    while (cap < need) cap *= 2;
    g->scores = (float*)realloc(g->scores, (size_t)cap * sizeof(float));
    if (!g->scores) exit(1);
    g->cap_scores = cap;
}

static void gg_kv_value(GB* b, const char* key, uint32_t type, GgufMeta* g)
{
    switch (type) {
    case GVT_U8:
    case GVT_I8:
    case GVT_BOOL:
        gb_skip(b, 1);
        break;
    case GVT_U16:
    case GVT_I16:
        gb_skip(b, 2);
        break;
    case GVT_U32:
    case GVT_I32:
    case GVT_F32: {
        uint32_t v = gb_u32(b);
        if (!strcmp(key, "llama.block_count")) g->n_blocks = v;
        else if (!strcmp(key, "llama.embedding_length")) g->hidden = v;
        else if (!strcmp(key, "llama.attention.head_count")) g->heads = v;
        else if (!strcmp(key, "llama.attention.head_count_kv")) g->kv_heads = v;
        else if (!strcmp(key, "llama.context_length")) g->context_len = v;
        else if (!strcmp(key, "general.alignment")) g->alignment = v;
        break;
    }
    case GVT_U64:
    case GVT_I64: {
        uint64_t v = gb_u64(b);
        if (!strcmp(key, "llama.block_count")) g->n_blocks = (uint32_t)v;
        else if (!strcmp(key, "llama.embedding_length")) g->hidden = (uint32_t)v;
        else if (!strcmp(key, "llama.attention.head_count")) g->heads = (uint32_t)v;
        else if (!strcmp(key, "llama.attention.head_count_kv")) g->kv_heads = (uint32_t)v;
        else if (!strcmp(key, "llama.context_length")) g->context_len = (uint32_t)v;
        break;
    }
    case GVT_F64: {
        double v = gb_f64(b);
        if (!strcmp(key, "llama.rope.freq_base")) g->freq_base = (float)v;
        else if (!strcmp(key, "llama.attention.layer_norm_rms_epsilon")) g->rms_eps = (float)v;
        break;
    }
    case GVT_STR: {
        char* s = gb_str(b);
        if (s) {
            if (!strcmp(key, "general.architecture")) {
                free(g->arch);
                g->arch = s;
            } else {
                free(s);
            }
        }
        break;
    }
    case GVT_ARRAY: {
        uint32_t at = gb_u32(b);
        uint64_t n = gb_u64(b);
        int is_tokens = !strcmp(key, "tokenizer.ggml.tokens");
        int is_scores = !strcmp(key, "tokenizer.ggml.scores");
        if (is_tokens) gg_tokens_grow(g, (uint64_t)g->n_tokens + n);
        if (is_scores) gg_scores_grow(g, (uint64_t)g->n_scores + n);
        uint64_t i;
        for (i = 0; i < n && !b->err; i++) {
            if (is_tokens && at == GVT_STR) {
                char* s = gb_str(b);
                if (!s) break;
                g->tokens[g->n_tokens++] = s;
            } else if (is_scores && at == GVT_F32) {
                uint32_t v = gb_u32(b);
                float f;
                memcpy(&f, &v, 4);
                g->scores[g->n_scores++] = f;
            } else {
                switch (at) {
                case GVT_U8: case GVT_I8: case GVT_BOOL: gb_skip(b, 1); break;
                case GVT_U16: case GVT_I16: gb_skip(b, 2); break;
                case GVT_U32: case GVT_I32: case GVT_F32: gb_skip(b, 4); break;
                case GVT_U64: case GVT_I64: case GVT_F64: gb_skip(b, 8); break;
                case GVT_STR: {
                    char* s = gb_str(b);
                    free(s);
                    break;
                }
                default: b->err = 1;
                }
            }
        }
        break;
    }
    default:
        b->err = 1;
    }
}

typedef struct {
    char* name;
    uint32_t gtype;
    uint32_t ndims;
    uint64_t dims[4];
    uint64_t offset;
    uint64_t nbytes;
} GGTensor;

typedef struct {
    GGTensor* t;
    int n;
    int cap;
} GGList;

static void gg_add(GGList* l, GGTensor* v)
{
    if (l->n == l->cap) {
        l->cap = l->cap ? l->cap * 2 : 64;
        l->t = (GGTensor*)realloc(l->t, (size_t)l->cap * sizeof(GGTensor));
        if (!l->t) exit(1);
    }
    l->t[l->n++] = *v;
}

/* probe per-type byte layout from actual tensor sizes; map[gtype] -> llf dtype,
   or 255 for unsupported.  Some quantizers emit non-standard ggml type ids,
   so the fixed enum cannot be trusted. */
static void gg_probe_layout(GGList* l, const uint8_t* dptr, uint64_t data_start, uint64_t fsize, uint8_t map[256])
{
    static const struct { uint64_t nb; uint32_t dt; } cand[] = {
        { 210, DT_Q6K },
        { 144, DT_Q4K },
        { 144, DT_IQ4XS },
        { 66, 0 },
        { 176, 0 },
        { 80, 0 },
        { 110, 0 },
        { 36, 0 },
    };
    size_t i;
    unsigned* idx = (unsigned*)ymalloc((size_t)l->n * sizeof(unsigned));
    for (i = 0; i < (size_t)l->n; i++) idx[i] = (unsigned)i;
    /* sort by offset to measure each tensor's real size */
    size_t a, b2;
    for (a = 1; a < (size_t)l->n; a++) {
        unsigned k = idx[a];
        for (b2 = a; b2 > 0 && l->t[idx[b2 - 1]].offset > l->t[k].offset; b2--) idx[b2] = idx[b2 - 1];
        idx[b2] = k;
    }
    for (i = 0; i < (size_t)l->n; i++) {
        GGTensor* t = &l->t[idx[i]];
        t->nbytes = i + 1 < (size_t)l->n
                        ? l->t[idx[i + 1]].offset - t->offset
                        : fsize - (data_start + t->offset);
        if (i + 1 >= (size_t)l->n && t->nbytes > (uint64_t)0 - data_start) t->nbytes = 0;
    }
    free(idx);
    for (a = 0; a < 256; a++) map[a] = 255;
    map[0] = DT_F32;
    map[1] = DT_F16;
    for (a = 2; a < 256; a++) {
        int seen = 0;
        size_t c;
        for (c = 0; c < sizeof(cand) / sizeof(cand[0]); c++) {
            if (cand[c].dt == 0) continue;
            int ok = 1;
            for (i = 0; i < (size_t)l->n; i++) {
                const GGTensor* t = &l->t[i];
                if (t->gtype != a) continue;
                seen = 1;
                uint64_t nelem = 1;
                uint32_t d;
                for (d = 0; d < t->ndims; d++) nelem *= t->dims[d];
                if (nelem == 0 || nelem % 256 != 0 || t->nbytes != nelem / 256 * cand[c].nb) { ok = 0; break; }
            }
            if (ok && seen) {
                map[a] = (uint8_t)cand[c].dt;
                if (cand[c].nb == 144) {
                    /* 144 B/block matches both Q4_K and IQ4_XS: inspect content.
                       Q4_K has fp16 dmin at bytes 2..3 (should be a sane small
                       value); IQ4_XS starts quants right after d. */
                    const GGTensor* t0 = NULL;
                    for (i = 0; i < (size_t)l->n; i++) { if (l->t[i].gtype == a) { t0 = &l->t[i]; break; } }
                    if (t0) {
                        uint16_t u16;
                        memcpy(&u16, dptr + data_start + t0->offset + 2, 2);
                        float dmin = f16_to_f32(u16);
                        printf("probe: type %u first=%s offset=%llu dmin=%g\n",
                            (unsigned)a, t0->name, (unsigned long long)t0->offset, (double)dmin);
                        if (dmin < -0.5f || dmin > 0.5f) map[a] = DT_IQ4XS;
                    }
                }
                break;
            }
        }
        if (seen && map[a] == 255) {
            const GGTensor* t0 = NULL;
            for (i = 0; i < (size_t)l->n; i++) { if (l->t[i].gtype == a) { t0 = &l->t[i]; break; } }
            if (t0 && t0->nbytes) {
                uint64_t nelem = 1;
                uint32_t d;
                for (d = 0; d < t0->ndims; d++) nelem *= t0->dims[d];
                printf("gguf: unsupported quant type %u (%.2f bytes per 256 elems)\n",
                    (unsigned)a, (double)t0->nbytes / (double)(nelem / 256));
            }
        }
    }
}

enum { SP_EMBED = -2, SP_FINALNORM = -3, SP_OUTPUT = -4 };

static int gg_slot_for(const char* name, int* layer)
{
    if (strcmp(name, "token_embd.weight") == 0) { *layer = 0; return SP_EMBED; }
    if (strcmp(name, "output_norm.weight") == 0) { *layer = 0; return SP_FINALNORM; }
    if (strcmp(name, "output.weight") == 0) { *layer = 0; return SP_OUTPUT; }
    if (strncmp(name, "blk.", 4) == 0) {
        const char* p = name + 4;
        int n = 0;
        while (*p >= '0' && *p <= '9') { n = n * 10 + (*p - '0'); p++; }
        if (*p != '.') return -1;
        p++;
        static const struct { const char* suf; int slot; } tab[] = {
            { "attn_norm.weight", SLOT_NORM1 },
            { "attn_q.weight", SLOT_Q },
            { "attn_k.weight", SLOT_K },
            { "attn_v.weight", SLOT_V },
            { "attn_output.weight", SLOT_O },
            { "ffn_norm.weight", SLOT_NORM2 },
            { "ffn_gate.weight", SLOT_GATE },
            { "ffn_up.weight", SLOT_UP },
            { "ffn_down.weight", SLOT_DOWN },
        };
        size_t i;
        for (i = 0; i < sizeof(tab) / sizeof(tab[0]); i++) {
            if (strcmp(p, tab[i].suf) == 0) { *layer = n; return tab[i].slot; }
        }
    }
    return -1;
}

int convert_gguf(const char* in_path, const char* out_path, const char* vocab_out,
                 uint32_t max_seq, char* err, size_t errlen)
{
    uint64_t fsize = 0;
    if (yfile_size(in_path, &fsize) != 0) { snprintf(err, errlen, "cannot open %s", in_path); return -1; }
    uint8_t* data = (uint8_t*)ymalloc((size_t)fsize);
    FILE* f = fopen(in_path, "rb");
    if (!f) { free(data); snprintf(err, errlen, "cannot open %s", in_path); return -1; }
    if (fread(data, 1, (size_t)fsize, f) != fsize) { fclose(f); free(data); snprintf(err, errlen, "read failed"); return -1; }
    fclose(f);

    GB b;
    b.p = data;
    b.end = data + fsize;
    b.err = 0;
    b.be = 0;
    if (fsize < 8 || memcmp(b.p, "GGUF", 4) != 0) {
        free(data);
        snprintf(err, errlen, "not a gguf file");
        return -1;
    }
    b.p += 4;
    uint32_t ver = gb_u32(&b);
    if ((ver & 0xFFFF) == 0) {
        /* gguf v3 supports big-endian files; the version field read with the
           wrong byte order has all zero low 16 bits, use that to detect BE */
        b.be = 1;
        ver = gb_u32(&b);
    }
    if (ver < 1 || ver > 3) {
        free(data);
        snprintf(err, errlen, "unsupported gguf version %u (need 1, 2 or 3)", ver);
        return -1;
    }
    uint64_t n_tensors, n_kv;
    if (ver == 1) {
        /* v1: counts and string lengths are uint32 */
        n_tensors = gb_u32(&b);
        n_kv = gb_u32(&b);
    } else {
        /* v2+: counts are uint64 */
        n_tensors = gb_u64(&b);
        n_kv = gb_u64(&b);
    }

    GgufMeta g;
    memset(&g, 0, sizeof(g));
    g.freq_base = 10000.0f;
    g.rms_eps = 1e-5f;

    uint64_t i;
    for (i = 0; i < n_kv && !b.err; i++) {
        char* key = gb_str(&b);
        if (!key) break;
        uint32_t type = gb_u32(&b);
        gg_kv_value(&b, key, type, &g);
        free(key);
    }
    if (b.err) {
        free(g.arch);
        for (i = 0; i < g.n_tokens; i++) free(g.tokens[i]);
        free(g.tokens);
        free(g.scores);
        free(data);
        snprintf(err, errlen, "bad gguf kv section");
        return -1;
    }
    if (!g.arch || strcmp(g.arch, "llama") != 0) {
        snprintf(err, errlen, "unsupported architecture '%s' (only 'llama' supported)", g.arch ? g.arch : "?");
        for (i = 0; i < g.n_tokens; i++) free(g.tokens[i]);
        free(g.tokens);
        free(g.scores);
        free(g.arch); free(data);
        return -1;
    }
    free(g.arch);
    if (g.n_blocks == 0 || g.hidden == 0 || g.heads == 0) {
        for (i = 0; i < g.n_tokens; i++) free(g.tokens[i]);
        free(g.tokens);
        free(g.scores);
        free(data);
        snprintf(err, errlen, "missing model dims in metadata");
        return -1;
    }
    if (g.kv_heads == 0) g.kv_heads = g.heads;
    if (g.hidden % g.heads != 0) {
        for (i = 0; i < g.n_tokens; i++) free(g.tokens[i]);
        free(g.tokens);
        free(g.scores);
        free(data);
        snprintf(err, errlen, "hidden %u not divisible by heads %u", g.hidden, g.heads);
        return -1;
    }
    if (g.hidden % 256 != 0) {
        for (i = 0; i < g.n_tokens; i++) free(g.tokens[i]);
        free(g.tokens);
        free(g.scores);
        free(data);
        snprintf(err, errlen, "hidden %u not divisible by 256 (required for K-quants)", g.hidden);
        return -1;
    }

    GGList list;
    memset(&list, 0, sizeof(list));
    for (i = 0; i < n_tensors && !b.err; i++) {
        GGTensor t;
        memset(&t, 0, sizeof(t));
        t.name = gb_str(&b);
        if (!t.name) break;
        uint32_t ndims = gb_u32(&b);
        if (ndims == 0 || ndims > 4) { free(t.name); b.err = 1; break; }
        t.ndims = ndims;
        uint64_t nelem = 1;
        uint32_t d;
        for (d = 0; d < ndims; d++) {
            t.dims[d] = gb_u64(&b);
            nelem *= t.dims[d];
        }
        t.gtype = gb_u32(&b);
        t.offset = gb_u64(&b);
        gg_add(&list, &t);
    }
    if (b.err) {
        for (i = 0; i < (uint64_t)list.n; i++) free(list.t[i].name);
        free(list.t);
        free(data);
        snprintf(err, errlen, "bad gguf tensor section");
        return -1;
    }
    uint64_t meta_end = (uint64_t)(b.p - data);
    if (g.alignment && (g.alignment < 8 || (g.alignment & (g.alignment - 1)) != 0)) {
        for (i = 0; i < (uint64_t)list.n; i++) free(list.t[i].name);
        free(list.t);
        free(data);
        snprintf(err, errlen, "bad general.alignment %u (must be >=8 and a power of two)", g.alignment);
        return -1;
    }
    /* v1 has no data alignment; v2/v3 pad the data section to general.alignment
       (default 32) so tensors can be memory-mapped */
    uint32_t alignment = g.alignment ? g.alignment : 32;
    uint64_t data_start = ver == 1 ? meta_end : align_up(meta_end, alignment);
    if (data_start > fsize) {
        for (i = 0; i < (uint64_t)list.n; i++) free(list.t[i].name);
        free(list.t);
        free(data);
        snprintf(err, errlen, "gguf data section out of range");
        return -1;
    }
    printf("gguf: version %u%s alignment=%u data_start=%llu\n", ver,
           b.be ? " (big-endian)" : "", alignment, (unsigned long long)data_start);

    uint8_t type_map[256];
    gg_probe_layout(&list, data, data_start, fsize, type_map);
    {
        static const char* dn[8] = { "f16", "f32", "bf16", "q4_k", "q6_k", "iq4_xs" };
        int a;
        for (a = 0; a < 256; a++) {
            int has = 0;
            for (i = 0; i < (uint64_t)list.n; i++) if (list.t[i].gtype == (uint32_t)a) { has = 1; break; }
            if (has) printf("probe: gguf type %d -> %s\n", a, type_map[a] < 6 ? dn[type_map[a]] : "?");
        }
    }
    {
        /* compute nbytes and drop unsupported tensors */
        GGList keep;
        memset(&keep, 0, sizeof(keep));
        for (i = 0; i < (uint64_t)list.n; i++) {
            const GGTensor* t = &list.t[i];
            uint64_t nelem = 1;
            uint32_t d;
            for (d = 0; d < t->ndims; d++) nelem *= t->dims[d];
            uint32_t dt = type_map[t->gtype];
            if (dt == 255) { free(t->name); continue; }
            GGTensor c;
            c = *t;
            if (dt == DT_F32) c.nbytes = nelem * 4;
            else if (dt == DT_F16) c.nbytes = nelem * 2;
            else {
                uint64_t nb = (dt == DT_Q6K) ? 210 : 144;
                c.nbytes = nelem / 256 * nb;
            }
            gg_add(&keep, &c);
        }
        free(list.t);
        list = keep;
    }

    ConvItem* items = (ConvItem*)ymalloc((size_t)list.n * sizeof(ConvItem));
    int n = 0;
    for (i = 0; i < (uint64_t)list.n; i++) {
        int layer;
        int slot = gg_slot_for(list.t[i].name, &layer);
        if (slot == SP_EMBED) { items[n].layer = 0; items[n].slot = 0; }
        else if (slot == SP_FINALNORM) { items[n].layer = g.n_blocks + 1; items[n].slot = 0; }
        else if (slot == SP_OUTPUT) { items[n].layer = g.n_blocks + 2; items[n].slot = 0; }
        else if (slot >= SLOT_NORM1 && slot <= SLOT_DOWN) { items[n].layer = (uint32_t)layer + 1; items[n].slot = (uint32_t)slot; }
        else continue;
        const GGTensor* t = &list.t[i];
        items[n].dtype = type_map[t->gtype];
        items[n].ndim = t->ndims;
        uint32_t d;
        for (d = 0; d < t->ndims; d++) items[n].shape[d] = (uint32_t)t->dims[d];
        items[n].nbytes = t->nbytes;
        snprintf(items[n].name, sizeof(items[n].name), "%s", t->name);
        items[n].src = data + data_start + t->offset;
        items[n].src_off = 0;
        n++;
    }
    if (n == 0) { free(items); free(list.t); free(data); snprintf(err, errlen, "no recognized tensors"); return -1; }

    {
        int p2;
        int missing = 0;
        static const int all[BLOCK_TENSORS] = { SLOT_NORM1, SLOT_Q, SLOT_K, SLOT_V, SLOT_O, SLOT_NORM2, SLOT_GATE, SLOT_UP, SLOT_DOWN };
        for (i = 1; i <= g.n_blocks && !missing; i++) {
            int s2;
            for (s2 = 0; s2 < BLOCK_TENSORS; s2++) {
                int found = 0;
                for (p2 = 0; p2 < n; p2++) {
                    if (items[p2].layer == i && items[p2].slot == (uint32_t)all[s2]) { found = 1; break; }
                }
                if (!found) { missing = 1; break; }
            }
        }
        if (missing) {
            for (i = 0; i < (uint64_t)list.n; i++) free(list.t[i].name);
            free(list.t); free(items); free(data);
            snprintf(err, errlen, "some transformer tensors missing from gguf");
            return -1;
        }
    }

    LlfHeader h;
    memset(&h, 0, sizeof(h));
    memcpy(h.magic, YLLM_MAGIC, 8);
    h.version = YLLM_VERSION;
    h.arch = ARCH_LLAMA;
    h.n_blocks = g.n_blocks;
    h.hidden = g.hidden;
    h.n_heads = g.heads;
    h.n_kv_heads = g.kv_heads;
    h.head_dim = g.hidden / g.heads;
    if (max_seq == 0 || (g.context_len > 0 && g.context_len < max_seq)) max_seq = g.context_len ? g.context_len : max_seq;
    h.max_seq = max_seq;
    h.dtype = DT_Q4K;
    memcpy(&h.norm_eps_bits, &g.rms_eps, 4);
    memcpy(&h.rope_theta_bits, &g.freq_base, 4);
    {
        uint64_t vocab = 0;
        int p2;
        for (p2 = 0; p2 < n; p2++) {
            if (items[p2].layer == 0 && items[p2].slot == 0) {
                vocab = items[p2].shape[1];
                break;
            }
        }
        if (vocab == 0) {
            for (p2 = 0; p2 < n; p2++) {
                if (items[p2].layer == g.n_blocks + 2) { vocab = items[p2].shape[1]; break; }
            }
        }
        h.vocab = (uint32_t)vocab;
    }
    if (h.vocab == 0) {
        for (i = 0; i < (uint64_t)list.n; i++) free(list.t[i].name);
        free(list.t); free(items); free(data);
        snprintf(err, errlen, "cannot determine vocab size");
        return -1;
    }

    int rc = llf_emit(out_path, &h, items, n, err, errlen);

    if (rc == 0 && vocab_out) {
        FILE* vf = fopen(vocab_out, "wb");
        if (vf) {
            fprintf(vf, "%u\n", g.n_tokens);
            for (i = 0; i < g.n_tokens; i++) {
                fprintf(vf, "%u\t", (uint32_t)i);
                const char* s = g.tokens[i];
                for (; *s; s++) {
                    if (*s == '\n') fputs("\\n", vf);
                    else if (*s == '\r') fputs("\\r", vf);
                    else if (*s == '\t') fputs("\\t", vf);
                    else if (*s == '\\') fputs("\\\\", vf);
                    else fputc(*s, vf);
                }
                fputc('\n', vf);
            }
            if (g.n_scores > 0) {
                fprintf(vf, "#SCORES#\n%u\n", g.n_scores);
                for (i = 0; i < g.n_scores; i++) {
                    fprintf(vf, "%.9g\n", (double)g.scores[i]);
                }
            }
            fclose(vf);
            printf("vocab written: %s (%u pieces, %u scores)\n", vocab_out, g.n_tokens, g.n_scores);
        } else {
            snprintf(err, errlen, "cannot write %s", vocab_out);
            rc = -1;
        }
    }
    for (i = 0; i < g.n_tokens; i++) free(g.tokens[i]);
    free(g.tokens);
    free(g.scores);
    for (i = 0; i < (uint64_t)list.n; i++) free(list.t[i].name);
    free(list.t);
    free(items);
    free(data);
    return rc;
}