#include "yllm.h"
#include "convert.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
    const uint8_t* p;
    const uint8_t* end;
    int err;
} GB;

static uint32_t gb_u32(GB* b)
{
    uint32_t v;
    if (b->p + 4 > b->end) { b->err = 1; return 0; }
    memcpy(&v, b->p, 4);
    b->p += 4;
    return v;
}

static uint64_t gb_u64(GB* b)
{
    uint64_t v;
    if (b->p + 8 > b->end) { b->err = 1; return 0; }
    memcpy(&v, b->p, 8);
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
    char** tokens;
    uint32_t n_tokens;
    uint32_t cap_tokens;
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
        if (is_tokens) gg_tokens_grow(g, (uint64_t)g->n_tokens + n);
        uint64_t i;
        for (i = 0; i < n && !b->err; i++) {
            if (is_tokens && at == GVT_STR) {
                char* s = gb_str(b);
                if (!s) break;
                g->tokens[g->n_tokens++] = s;
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

/* ggml type -> (llf dtype, nbytes) for supported types */
static int gg_type_info(uint32_t gtype, uint64_t nelem, uint32_t* dt, uint64_t* nbytes)
{
    switch (gtype) {
    case 0:
        if (dt) *dt = DT_F32;
        if (nbytes) *nbytes = nelem * 4;
        return 0;
    case 1:
        if (dt) *dt = DT_F16;
        if (nbytes) *nbytes = nelem * 2;
        return 0;
    case 10:
        if (dt) *dt = DT_Q4K;
        if (nbytes) *nbytes = nelem / 256 * 144;
        return 0;
    case 12:
        if (dt) *dt = DT_Q6K;
        if (nbytes) *nbytes = nelem / 256 * 210;
        return 0;
    default:
        return -1;
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
    if (fsize < 8 || memcmp(b.p, "GGUF", 4) != 0) {
        free(data);
        snprintf(err, errlen, "not a gguf file");
        return -1;
    }
    b.p += 4;
    uint32_t ver = gb_u32(&b);
    if (ver != 2 && ver != 3) {
        free(data);
        snprintf(err, errlen, "unsupported gguf version %u (need 2 or 3)", ver);
        return -1;
    }
    uint64_t n_tensors = gb_u64(&b);
    uint64_t n_kv = gb_u64(&b);

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
    if (b.err) { free(g.arch); free(data); snprintf(err, errlen, "bad gguf kv section"); return -1; }
    if (!g.arch || strcmp(g.arch, "llama") != 0) {
        snprintf(err, errlen, "unsupported architecture '%s' (only 'llama' supported)", g.arch ? g.arch : "?");
        free(g.arch); free(data);
        return -1;
    }
    free(g.arch);
    if (g.n_blocks == 0 || g.hidden == 0 || g.heads == 0) {
        free(data);
        snprintf(err, errlen, "missing model dims in metadata");
        return -1;
    }
    if (g.kv_heads == 0) g.kv_heads = g.heads;
    if (g.hidden % g.heads != 0) { free(data); snprintf(err, errlen, "hidden %u not divisible by heads %u", g.hidden, g.heads); return -1; }
    if (g.hidden % 256 != 0) { free(data); snprintf(err, errlen, "hidden %u not divisible by 256 (required for K-quants)", g.hidden); return -1; }

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
        {
            uint32_t dt;
            if (gg_type_info(t.gtype, nelem, &dt, &t.nbytes) != 0) {
                free(t.name);
                continue;
            }
        }
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
    uint64_t data_start = align_up(meta_end, 32);
    if (data_start > fsize) {
        for (i = 0; i < (uint64_t)list.n; i++) free(list.t[i].name);
        free(list.t);
        free(data);
        snprintf(err, errlen, "gguf data section out of range");
        return -1;
    }

    ConvItem* items = (ConvItem*)ymalloc((size_t)list.n * sizeof(ConvItem));
    int n = 0;
    for (i = 0; i < (uint64_t)list.n; i++) {
        int layer;
        int slot = gg_slot_for(list.t[i].name, &layer);
        if (slot == SP_EMBED) { items[n].layer = 0; items[n].slot = 0; }
        else if (slot == SP_FINALNORM) { items[n].layer = g.n_blocks + 1; items[n].slot = 0; }
        else if (slot == SP_OUTPUT) { items[n].layer = g.n_blocks + 2; items[n].slot = 0; }
        else if (slot >= SLOT_NORM1 && slot <= SLOT_DOWN) { items[n].layer = (uint32_t)layer; items[n].slot = (uint32_t)slot; }
        else continue;
        const GGTensor* t = &list.t[i];
        gg_type_info(t->gtype, 0, &items[n].dtype, NULL);
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
                fprintf(vf, "%u\t%s\n", (uint32_t)i, g.tokens[i]);
            }
            fclose(vf);
            printf("vocab written: %s (%u pieces)\n", vocab_out, g.n_tokens);
        } else {
            snprintf(err, errlen, "cannot write %s", vocab_out);
            rc = -1;
        }
    }
    for (i = 0; i < g.n_tokens; i++) free(g.tokens[i]);
    free(g.tokens);
    for (i = 0; i < (uint64_t)list.n; i++) free(list.t[i].name);
    free(list.t);
    free(items);
    free(data);
    return rc;
}