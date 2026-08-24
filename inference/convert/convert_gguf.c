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
    uint32_t key_length;
    uint32_t alignment;
    char** tokens;
    uint32_t n_tokens;
    uint32_t cap_tokens;
    float* scores;
    uint32_t n_scores;
    uint32_t cap_scores;
    char** merges;
    uint32_t n_merges;
    uint32_t cap_merges;
    char* chat_template;
    int add_bos;
    int eos_id;
    int bos_id;
    /* gemma4 专用 */
    float attn_logit_cap;    /* attention logit soft-cap (默认 50.0) */
    float final_logit_cap;   /* final logit soft-cap (默认 30.0) */
    uint32_t swa_window;     /* sliding window size (默认 4096) */
    uint32_t swa_pattern;    /* 每 N 层为全局层 (默认 6) */
    uint32_t n_kv_shared_layers; /* shared KV 层数 (默认 1) */
    uint32_t n_embd_per_layer;   /* PLE 每层宽度 */
    uint64_t swa_mask;           /* bit i = is_swa(i); 0 = 用 swa_pattern */
} GgufMeta;

typedef struct {
    uint32_t attn_logit_cap_bits;
    uint32_t final_logit_cap_bits;
    uint32_t swa_window;
    uint32_t swa_pattern;
    uint32_t n_kv_shared_layers;
    uint32_t n_embd_per_layer;
    uint64_t swa_mask;
} LlfGemma4Ext;

static void gg_merges_grow(GgufMeta* g, uint64_t need)
{
    if (need <= g->cap_merges) return;
    uint32_t cap = g->cap_merges ? g->cap_merges : 1024;
    while (cap < need) cap *= 2;
    g->merges = (char**)realloc(g->merges, (size_t)cap * sizeof(char*));
    if (!g->merges) exit(1);
    g->cap_merges = cap;
}

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

/* KV key 去掉 "<arch>." 前缀后与公共名比较(如 llama.block_count / qwen2.block_count) */
static const char* gk_suffix(const char* key)
{
    const char* p = strchr(key, '.');
    return p ? p + 1 : key;
}

static void gg_kv_value(GB* b, const char* key, uint32_t type, GgufMeta* g)
{
    const char* k = gk_suffix(key);
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
        if (!strcmp(k, "block_count")) g->n_blocks = v;
        else if (!strcmp(k, "embedding_length")) g->hidden = v;
        else if (!strcmp(k, "attention.head_count")) g->heads = v;
        else if (!strcmp(k, "attention.head_count_kv")) g->kv_heads = v;
        else if (!strcmp(k, "attention.key_length")) g->key_length = v;
        else if (!strcmp(k, "context_length")) g->context_len = v;
        else if (!strcmp(k, "rope.freq_base")) { float f; memcpy(&f, &v, 4); g->freq_base = f; }
        else if (!strcmp(k, "attention.layer_norm_rms_epsilon")) { float f; memcpy(&f, &v, 4); g->rms_eps = f; }
        else if (!strcmp(key, "general.alignment")) g->alignment = v;
        else if (!strcmp(key, "tokenizer.ggml.add_bos_token")) g->add_bos = (int)v;
        else if (!strcmp(key, "tokenizer.ggml.eos_token_id")) g->eos_id = (int)v;
        else if (!strcmp(key, "tokenizer.ggml.bos_token_id")) g->bos_id = (int)v;
        /* gemma4 */
        else if (!strcmp(k, "attn_logit_softcapping")) { float f; memcpy(&f, &v, 4); g->attn_logit_cap = f; }
        else if (!strcmp(k, "final_logit_softcapping")) { float f; memcpy(&f, &v, 4); g->final_logit_cap = f; }
        else if (!strcmp(k, "attention.sliding_window")) g->swa_window = v;
        else if (!strcmp(k, "attention.sliding_window_pattern")) g->swa_pattern = v;
        else if (!strcmp(k, "attention.shared_kv_layers")) g->n_kv_shared_layers = v;
        else if (!strcmp(k, "embedding_length_per_layer_input")) g->n_embd_per_layer = v;
        break;
    }
    case GVT_U64:
    case GVT_I64: {
        uint64_t v = gb_u64(b);
        if (!strcmp(k, "block_count")) g->n_blocks = (uint32_t)v;
        else if (!strcmp(k, "embedding_length")) g->hidden = (uint32_t)v;
        else if (!strcmp(k, "attention.head_count")) g->heads = (uint32_t)v;
        else if (!strcmp(k, "attention.head_count_kv")) g->kv_heads = (uint32_t)v;
        else if (!strcmp(k, "attention.key_length")) g->key_length = (uint32_t)v;
        else if (!strcmp(k, "context_length")) g->context_len = (uint32_t)v;
        else if (!strcmp(key, "tokenizer.ggml.add_bos_token")) g->add_bos = (int)v;
        else if (!strcmp(key, "tokenizer.ggml.eos_token_id")) g->eos_id = (int)v;
        else if (!strcmp(key, "tokenizer.ggml.bos_token_id")) g->bos_id = (int)v;
        else if (!strcmp(k, "embedding_length_per_layer_input")) g->n_embd_per_layer = (uint32_t)v;
        else if (!strcmp(k, "attention.shared_kv_layers")) g->n_kv_shared_layers = (uint32_t)v;
        break;
    }
    case GVT_F64: {
        double v = gb_f64(b);
        if (!strcmp(k, "rope.freq_base")) g->freq_base = (float)v;
        else if (!strcmp(k, "attention.layer_norm_rms_epsilon")) g->rms_eps = (float)v;
        else if (!strcmp(k, "attn_logit_softcapping")) g->attn_logit_cap = (float)v;
        else if (!strcmp(k, "final_logit_softcapping")) g->final_logit_cap = (float)v;
        break;
    }
    case GVT_STR: {
        char* s = gb_str(b);
        if (s) {
            if (!strcmp(key, "general.architecture")) {
                free(g->arch);
                g->arch = s;
            } else if (!strcmp(key, "tokenizer.chat_template")) {
                free(g->chat_template);
                g->chat_template = s;
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
        int is_merges = !strcmp(key, "tokenizer.ggml.merges");
        int is_swa_arr = !strcmp(k, "attention.sliding_window_pattern");
        if (is_tokens) gg_tokens_grow(g, (uint64_t)g->n_tokens + n);
        if (is_scores) gg_scores_grow(g, (uint64_t)g->n_scores + n);
        if (is_merges) gg_merges_grow(g, (uint64_t)g->n_merges + n);
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
            } else if (is_merges && at == GVT_STR) {
                char* s = gb_str(b);
                if (!s) break;
                g->merges[g->n_merges++] = s;
            } else if (is_swa_arr && i < 64 &&
                       (at == GVT_U8 || at == GVT_I8 || at == GVT_BOOL ||
                        at == GVT_U32 || at == GVT_I32)) {
                uint32_t v = 0;
                if (at == GVT_U32 || at == GVT_I32) v = gb_u32(b);
                else { v = (uint32_t)(unsigned char)b->p[0]; gb_skip(b, 1); }
                if (v) g->swa_mask |= (1ull << i);
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
        { 176, DT_Q5K },
        { 144, DT_Q4K },
        { 144, DT_IQ4XS },
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
    /* ggml type 30 = BF16: 2 bytes/elem */
    {
        int seen = 0, ok = 1;
        for (i = 0; i < (size_t)l->n; i++) {
            const GGTensor* t = &l->t[i];
            if (t->gtype != 30) continue;
            seen = 1;
            uint64_t nelem = 1;
            uint32_t d;
            for (d = 0; d < t->ndims; d++) nelem *= t->dims[d];
            if (nelem == 0 || t->nbytes != nelem * 2) { ok = 0; break; }
        }
        if (seen && ok) map[30] = DT_BF16;
    }
}

enum { SP_EMBED = -2, SP_FINALNORM = -3, SP_OUTPUT = -4,
       SP_PLE_TOK = -10, SP_PLE_MPROJ = -11, SP_PLE_PNORM = -12,
       SP_MTP_EH = 100, SP_MTP_ENORM = 101, SP_MTP_HNORM = 102, SP_MTP_HEAD_NORM = 103 };

/* Q8_0 去量化成 F16: 每 32 元素 = 1 fp16 scale + 32 int8。
 * 返回 malloc 的 F16 缓冲(nelem*2 字节), 调用方负责释放。 */
static uint8_t* q8_0_to_f16(const uint8_t* src, uint64_t nelem)
{
    uint8_t* out = (uint8_t*)ymalloc((size_t)nelem * 2);
    const uint8_t* p = src;
    uint64_t i = 0;
    while (i + 32 <= nelem) {
        float s = f16_to_f32((uint16_t)(p[0] | (p[1] << 8)));
        const int8_t* q = (const int8_t*)(p + 2);
        uint32_t k;
        for (k = 0; k < 32; k++) {
            float v = s * (float)q[k];
            uint16_t h = f32_to_f16(v);
            out[(i + k) * 2] = (uint8_t)(h & 0xFF);
            out[(i + k) * 2 + 1] = (uint8_t)(h >> 8);
        }
        p += 2 + 32;
        i += 32;
    }
    return out;
}

static int gg_slot_for(const char* name, int* layer)
{
    /* qwen2 风格命名: model.embed_tokens / model.layers.{i}.self_attn.{q,k,v,o}_proj
       / model.layers.{i}.mlp.{gate,up,down}_proj / model.norm / lm_head */
    if (strcmp(name, "model.embed_tokens.weight") == 0) { *layer = 0; return SP_EMBED; }
    if (strcmp(name, "model.norm.weight") == 0) { *layer = 0; return SP_FINALNORM; }
    if (strcmp(name, "lm_head.weight") == 0) { *layer = 0; return SP_OUTPUT; }
    if (strncmp(name, "model.layers.", 13) == 0) {
        const char* p = name + 13;
        int n = 0;
        while (*p >= '0' && *p <= '9') { n = n * 10 + (*p - '0'); p++; }
        if (*p != '.') return -1;
        p++;
        static const struct { const char* suf; int slot; } tab_qwen[] = {
            { "input_layernorm.weight", SLOT_NORM1 },
            { "self_attn.q_proj.weight", SLOT_Q },
            { "self_attn.k_proj.weight", SLOT_K },
            { "self_attn.v_proj.weight", SLOT_V },
            { "self_attn.o_proj.weight", SLOT_O },
            { "self_attn.q_norm.weight", SLOT_QNORM },
            { "self_attn.k_norm.weight", SLOT_KNORM },
            { "post_attention_layernorm.weight", SLOT_NORM2 },
            { "mlp.gate_proj.weight", SLOT_GATE },
            { "mlp.up_proj.weight", SLOT_UP },
            { "mlp.down_proj.weight", SLOT_DOWN },
        };
        size_t i;
        for (i = 0; i < sizeof(tab_qwen) / sizeof(tab_qwen[0]); i++) {
            if (strcmp(p, tab_qwen[i].suf) == 0) { *layer = n; return tab_qwen[i].slot; }
        }
        return -1;
    }
    if (strcmp(name, "token_embd.weight") == 0) { *layer = 0; return SP_EMBED; }
    if (strcmp(name, "output_norm.weight") == 0) { *layer = 0; return SP_FINALNORM; }
    if (strcmp(name, "output.weight") == 0) { *layer = 0; return SP_OUTPUT; }
    if (strcmp(name, "per_layer_token_embd.weight") == 0) { *layer = 0; return SP_PLE_TOK; }
    if (strcmp(name, "per_layer_model_proj.weight") == 0) { *layer = 0; return SP_PLE_MPROJ; }
    if (strcmp(name, "per_layer_proj_norm.weight") == 0) { *layer = 0; return SP_PLE_PNORM; }
    if (strncmp(name, "blk.", 4) == 0) {
        const char* p = name + 4;
        int n = 0;
        while (*p >= '0' && *p <= '9') { n = n * 10 + (*p - '0'); p++; }
        if (*p != '.') return -1;
        p++;
        static const struct { const char* suf; int slot; } tab_blk[] = {
            { "nextn.eh_proj.weight", SP_MTP_EH },
            { "nextn.enorm.weight", SP_MTP_ENORM },
            { "nextn.hnorm.weight", SP_MTP_HNORM },
            { "nextn.shared_head_norm.weight", SP_MTP_HEAD_NORM },
            { "attn_norm.weight", SLOT_NORM1 },
            { "attn_q.weight", SLOT_Q },
            { "attn_k.weight", SLOT_K },
            { "attn_v.weight", SLOT_V },
            { "attn_output.weight", SLOT_O },
            { "ffn_norm.weight", SLOT_NORM2 },
            { "post_attention_norm.weight", SLOT_NORM3 },
            { "post_ffw_norm.weight", SLOT_NORM4 },
            { "layer_output_scale.weight", SLOT_LAYER_SCALE },
            { "inp_gate.weight", SLOT_PLE_GATE },
            { "proj.weight", SLOT_PLE_PROJ },
            { "post_norm.weight", SLOT_PLE_POST },
            { "post_attention_layernorm.weight", SLOT_NORM2 },
            { "ffn_gate.weight", SLOT_GATE },
            { "ffn_up.weight", SLOT_UP },
            { "ffn_down.weight", SLOT_DOWN },
            { "attn_q.bias", SLOT_QBIAS },
            { "attn_k.bias", SLOT_KBIAS },
            { "attn_v.bias", SLOT_VBIAS },
            { "attn_q_norm.weight", SLOT_QNORM },
            { "attn_k_norm.weight", SLOT_KNORM },
            { "attn_qkv.weight", SLOT_QKV },
            { "attn_gate.weight", SLOT_GATE_ATTN },
            { "ssm_conv1d.weight", SLOT_SSM_CONV1D },
            { "ssm_a", SLOT_SSM_A },
            { "ssm_dt.bias", SLOT_SSM_DT },
            { "ssm_alpha.weight", SLOT_SSM_ALPHA },
            { "ssm_beta.weight", SLOT_SSM_BETA },
            { "ssm_norm.weight", SLOT_SSM_NORM },
            { "ssm_out.weight", SLOT_SSM_OUT },
        };
        size_t i;
        for (i = 0; i < sizeof(tab_blk) / sizeof(tab_blk[0]); i++) {
            if (strcmp(p, tab_blk[i].suf) == 0) { *layer = n; return tab_blk[i].slot; }
        }
    }
    return -1;
}

int convert_gguf(const char* in_path, const char* out_path, const char* vocab_out,
                 uint32_t max_seq, char* err, size_t errlen)
{
    /* 省内存模式: 只读 mmap 整个 gguf, 不整读进堆(17GB 模型避免 OOM) */
    WMap gmap;
    if (wmap_open(in_path, &gmap) != 0) { snprintf(err, errlen, "cannot open %s", in_path); return -1; }
    uint8_t* data = (uint8_t*)gmap.base;
    uint64_t fsize = gmap.size;

    GB b;
    b.p = data;
    b.end = data + fsize;
    b.err = 0;
    b.be = 0;
    if (fsize < 8 || memcmp(b.p, "GGUF", 4) != 0) {
        wmap_close(&gmap);
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
        wmap_close(&gmap);
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
    g.add_bos = 1; /* llama architecture default */
    g.bos_id = -1;
    g.eos_id = -1;
    g.attn_logit_cap = 50.0f;
    g.final_logit_cap = 30.0f;
    g.swa_window = 4096;
    g.swa_pattern = 6;
    g.n_kv_shared_layers = 0;

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
        free(g.chat_template);
        wmap_close(&gmap);
        snprintf(err, errlen, "bad gguf kv section");
        return -1;
    }
    if (!g.arch || (strcmp(g.arch, "llama") != 0 && strcmp(g.arch, "qwen2") != 0 &&
                    strcmp(g.arch, "qwen3") != 0 && strcmp(g.arch, "qwen35") != 0 &&
                    strcmp(g.arch, "gemma4") != 0)) {
        snprintf(err, errlen, "unsupported architecture '%s' (only 'llama', 'qwen2/qwen3/qwen35' and 'gemma4' supported)", g.arch ? g.arch : "?");
        for (i = 0; i < g.n_tokens; i++) free(g.tokens[i]);
        free(g.tokens);
        free(g.scores);
        free(g.arch); wmap_close(&gmap);
        return -1;
    }
    /* qwen2/qwen3/qwen35: 同构(interleaved RoPE, 无 head 输出 bias; qwen3 连 attention bias 都没有,
     * 代码按 tensor 是否存在自动跳过) */
    int is_qwen = g.arch && (strcmp(g.arch, "qwen2") == 0 || strcmp(g.arch, "qwen3") == 0);
    int is_qwen35 = g.arch && strcmp(g.arch, "qwen35") == 0;
    int is_gemma4 = g.arch && strcmp(g.arch, "gemma4") == 0;
    free(g.arch);
    if (g.n_blocks == 0 || g.hidden == 0 || g.heads == 0) {
        for (i = 0; i < g.n_tokens; i++) free(g.tokens[i]);
        free(g.tokens);
        free(g.scores);
        free(g.chat_template);
        wmap_close(&gmap);
        snprintf(err, errlen, "missing model dims in metadata");
        return -1;
    }
    if (g.kv_heads == 0) g.kv_heads = g.heads;
    if (!is_qwen35 && g.hidden % g.heads != 0) {
        for (i = 0; i < g.n_tokens; i++) free(g.tokens[i]);
        free(g.tokens);
        free(g.scores);
        free(g.chat_template);
        wmap_close(&gmap);
        snprintf(err, errlen, "hidden %u not divisible by heads %u", g.hidden, g.heads);
        return -1;
    }
    if (g.hidden % 256 != 0) {
        for (i = 0; i < g.n_tokens; i++) free(g.tokens[i]);
        free(g.tokens);
        free(g.scores);
        free(g.chat_template);
        wmap_close(&gmap);
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
        wmap_close(&gmap);
        snprintf(err, errlen, "bad gguf tensor section");
        return -1;
    }
    uint64_t meta_end = (uint64_t)(b.p - data);
    if (g.alignment && (g.alignment < 8 || (g.alignment & (g.alignment - 1)) != 0)) {
        for (i = 0; i < (uint64_t)list.n; i++) free(list.t[i].name);
        free(list.t);
        wmap_close(&gmap);
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
        wmap_close(&gmap);
        snprintf(err, errlen, "gguf data section out of range");
        return -1;
    }
    printf("gguf: version %u%s alignment=%u data_start=%llu\n", ver,
           b.be ? " (big-endian)" : "", alignment, (unsigned long long)data_start);

    uint8_t type_map[256];
    gg_probe_layout(&list, data, data_start, fsize, type_map);
    {
        static const char* dn[8] = { "f16", "f32", "bf16", "q4_k", "q6_k", "iq4_xs", "q5_k" };
        int a;
        for (a = 0; a < 256; a++) {
            int has = 0;
            for (i = 0; i < (uint64_t)list.n; i++) if (list.t[i].gtype == (uint32_t)a) { has = 1; break; }
            if (has) printf("probe: gguf type %d -> %s\n", a, type_map[a] < 7 ? dn[type_map[a]] : "?");
        }
    }
    {
        /* compute nbytes and drop unsupported tensors; validate data bounds */
        GGList keep;
        memset(&keep, 0, sizeof(keep));
        for (i = 0; i < (uint64_t)list.n; i++) {
            const GGTensor* t = &list.t[i];
            uint64_t nelem = 1;
            uint32_t d;
            for (d = 0; d < t->ndims; d++) nelem *= t->dims[d];
            if (strstr(t->name, "nextn")) {
                /* MTP 权重: 保留, 走 gg_slot_for 映射到 output 层 MTP 槽 */
                printf("gguf: MTP '%s' nd=%u nelem=%llu nbytes=%llu gtype=%u\n",
                       t->name, t->ndims,
                       (unsigned long long)nelem,
                       (unsigned long long)t->nbytes, (unsigned)t->gtype);
            }
            uint32_t dt = type_map[t->gtype];
            if (dt == 255 && !strstr(t->name, "nextn")) {
                printf("gguf: DROP '%s' gtype=%u\n", t->name, (unsigned)t->gtype); free(t->name); continue;
            }
            if (strstr(t->name, "nextn") && t->gtype == 8) {
                /* MTP eh_proj Q8_0: dt 未知但转换时去量化成 F16, 放行;
                 * 源数据是 Q8_0(272B/256元素), 非 F16 */
                dt = DT_F16;
            }
            GGTensor c;
            c = *t;
            if (strstr(t->name, "nextn") && t->gtype == 8) {
                c.nbytes = (nelem / 256) * 272;   /* Q8_0 块大小 */
            } else if (dt == DT_F32) c.nbytes = nelem * 4;
            else if (dt == DT_F16 || dt == DT_BF16) c.nbytes = nelem * 2;
            else {
                uint64_t nb = (dt == DT_Q6K) ? 210 : (dt == DT_Q5K) ? 176 : 144;
                c.nbytes = (nelem / 256) * nb;
            }
            /* reject truncated/corrupt file: tensor data must fit inside the file */
            if (c.offset > fsize - data_start || c.nbytes > fsize - data_start - c.offset) {
                for (int k = 0; k < keep.n; k++) free(keep.t[k].name);
                free(keep.t);
                while (i < (uint64_t)list.n) free(list.t[i++].name);
                free(list.t);
                wmap_close(&gmap);
                snprintf(err, errlen, "gguf tensor '%s' data out of range (truncated file?)", t->name);
                return -1;
            }
            gg_add(&keep, &c);
        }
        free(list.t);
        list = keep;
    }

    ConvItem* items = (ConvItem*)ymalloc((size_t)list.n * 2 * sizeof(ConvItem));
    uint8_t* qg_bufs[32] = { 0 };  /* attn_q 交错重排临时缓冲(llf_emit 后释放) */
    int qg_n = 0;
    int n = 0;
    for (i = 0; i < (uint64_t)list.n; i++) {
        int layer;
        int slot = gg_slot_for(list.t[i].name, &layer);
        if (slot == SP_EMBED) {
            items[n].layer = 0; items[n].slot = 0;
            /* gemma4 tied embedding: embed == lm_head, 先写 embed, 然后额外写 output */
        }
        else if (slot == SP_FINALNORM) { items[n].layer = g.n_blocks + 1; items[n].slot = 0; }
        else if (slot == SP_OUTPUT) { items[n].layer = g.n_blocks + 2; items[n].slot = 0; }
        else if (slot == SP_MTP_EH) { items[n].layer = g.n_blocks + 2; items[n].slot = SLOT_MTP_EH; }
        else if (slot == SP_MTP_ENORM) { items[n].layer = g.n_blocks + 2; items[n].slot = SLOT_MTP_ENORM; }
        else if (slot == SP_MTP_HNORM) { items[n].layer = g.n_blocks + 2; items[n].slot = SLOT_MTP_HNORM; }
        else if (slot == SP_MTP_HEAD_NORM) { items[n].layer = g.n_blocks + 2; items[n].slot = SLOT_MTP_HEAD_NORM; }
        else if (slot == SP_PLE_TOK) { items[n].layer = 0; items[n].slot = SLOT_PLE_TOK; }
        else if (slot == SP_PLE_MPROJ) { items[n].layer = 0; items[n].slot = SLOT_PLE_MPROJ; }
        else if (slot == SP_PLE_PNORM) { items[n].layer = 0; items[n].slot = SLOT_PLE_PNORM; }
        else if (slot >= SLOT_NORM1 && slot <= SLOT_KNORM) { items[n].layer = (uint32_t)layer + 1; items[n].slot = (uint32_t)slot; }
        else if (slot >= SLOT_QKV && slot <= SLOT_LAYER_SCALE) { items[n].layer = (uint32_t)layer + 1; items[n].slot = (uint32_t)slot; }
        else { printf("gguf: SKIP '%s'\n", list.t[i].name); continue; }
        const GGTensor* t = &list.t[i];
        items[n].dtype = type_map[t->gtype];
        items[n].ndim = t->ndims;
        uint32_t d;
        for (d = 0; d < t->ndims; d++) items[n].shape[d] = (uint32_t)t->dims[d];
        items[n].nbytes = t->nbytes;
        snprintf(items[n].name, sizeof(items[n].name), "%s", t->name);
        items[n].src = data + data_start + t->offset;
        items[n].src_off = 0;
        /* MTP 权重(小, 55MB): 量化(Q8_0=gtype8)直接去量化成 F16 存 llf, 便于 engine 使用 */
        if (slot >= SP_MTP_EH && slot <= SP_MTP_HEAD_NORM && t->gtype == 8) {
            uint64_t mnelem = 1;
            for (d = 0; d < t->ndims; d++) mnelem *= t->dims[d];
            items[n].src = q8_0_to_f16(items[n].src, mnelem);
            items[n].src_off = 0;
            items[n].dtype = DT_F16;
            items[n].nbytes = mnelem * 2;
        }
        if (items[n].dtype == DT_BF16) {
            uint64_t mnelem = 1;
            for (d = 0; d < t->ndims; d++) mnelem *= t->dims[d];
            uint8_t* nb = (uint8_t*)ymalloc((size_t)mnelem * 2);
            bf16_to_f16_buf((const uint16_t*)items[n].src, (uint16_t*)nb, (size_t)mnelem);
            items[n].src = nb;
            items[n].src_off = 0;
            items[n].dtype = DT_F16;
            items[n].nbytes = mnelem * 2;
            if (qg_n < 32) qg_bufs[qg_n++] = nb;
        }
        n++;
        /* gemma4 tied embedding: 写完 embed 后额外写一份 output(lm_head) 指向同一数据 */
        if (is_gemma4 && slot == SP_EMBED) {
            items[n] = items[n - 1];
            items[n].layer = g.n_blocks + 2;
            snprintf(items[n].name, sizeof(items[n].name), "output.weight");
            n++;
        }
        if (is_qwen35 && slot == SLOT_Q && t->ndims == 2 &&
            t->dims[1] == 2 * (uint64_t)g.heads * (g.key_length ? g.key_length : g.hidden / g.heads)) {
            /* qwen35 attention 层: attn_q = [q|gate] 拼接(输出维 2×qdim)。
               每 head 输出 512 行 = [q 256 | gate 256] 交错。
               转换期按 head 交错重排: q 半部(每 head 前 256 行) → SLOT_Q,
               gate 半部(每 head 后 256 行) → SLOT_QGATE */
            uint64_t qdim = (uint64_t)g.heads * (g.key_length ? g.key_length : g.hidden / g.heads);
            uint32_t qper = qdim / g.heads;                 /* 256 */
            uint64_t rowb = t->nbytes / t->dims[1];         /* 每输出行字节 */
            uint8_t* nb = (uint8_t*)ymalloc((size_t)t->nbytes);
            const uint8_t* src = data + data_start + t->offset;
            uint32_t h, r;
            for (h = 0; h < g.heads; h++) {
                for (r = 0; r < qper; r++) {
                    uint64_t q_src = (uint64_t)h * 2 * qper + r;
                    uint64_t g_src = (uint64_t)h * 2 * qper + qper + r;
                    uint64_t q_dst = (uint64_t)h * qper + r;
                    uint64_t g_dst = (uint64_t)g.heads * qper + (uint64_t)h * qper + r;
                    memcpy(nb + q_dst * rowb, src + q_src * rowb, (size_t)rowb);
                    memcpy(nb + g_dst * rowb, src + g_src * rowb, (size_t)rowb);
                }
            }
            if (qg_n < 32) qg_bufs[qg_n++] = nb;
            items[n - 1].shape[1] = (uint32_t)qdim;
            items[n - 1].nbytes = qdim * rowb;
            items[n - 1].src = nb;
            items[n - 1].src_off = 0;
            items[n].layer = items[n - 1].layer;
            items[n].slot = SLOT_QGATE;
            items[n].dtype = items[n - 1].dtype;
            items[n].ndim = 2;
            items[n].shape[0] = g.hidden;
            items[n].shape[1] = (uint32_t)qdim;
            items[n].nbytes = qdim * rowb;
            snprintf(items[n].name, sizeof(items[n].name), "%s.gate", t->name);
            items[n].src = nb + qdim * rowb;
            items[n].src_off = 0;
            n++;
        }
    }
    if (n == 0) { free(items); free(list.t); wmap_close(&gmap); snprintf(err, errlen, "no recognized tensors"); return -1; }

    if (is_gemma4 && g.n_embd_per_layer == 0) {
        int p2;
        for (p2 = 0; p2 < n; p2++) {
            if (items[p2].layer == 0 && items[p2].slot == SLOT_PLE_TOK && g.n_blocks > 0 &&
                items[p2].shape[0] % g.n_blocks == 0)
                g.n_embd_per_layer = items[p2].shape[0] / g.n_blocks;
        }
    }

    {
        int p2;
        int missing = 0;
        /* gemma4: 每层必需 4 norm + Q/K/V/O + Gate/Up/Down */
        if (is_gemma4) {
        static const int g4req[] = { SLOT_NORM1, SLOT_NORM2, SLOT_NORM3, SLOT_NORM4,
                                         SLOT_Q, SLOT_O,
                                         SLOT_GATE, SLOT_UP, SLOT_DOWN };
            uint32_t kv_from = g.n_blocks;
            if (g.n_kv_shared_layers > 0 && g.n_kv_shared_layers < g.n_blocks)
                kv_from = g.n_blocks - g.n_kv_shared_layers;
            for (i = 1; i <= g.n_blocks && !missing; i++) {
                size_t si;
                for (si = 0; si < sizeof(g4req)/sizeof(g4req[0]); si++) {
                    int found = 0;
                    for (p2 = 0; p2 < n; p2++) {
                        if (items[p2].layer == i && items[p2].slot == (uint32_t)g4req[si]) { found = 1; break; }
                    }
                    if (!found) { missing = 1; break; }
                }
                if (!missing && (i - 1) < kv_from) {
                    int has_k = 0, has_v = 0;
                    for (p2 = 0; p2 < n; p2++) {
                        if (items[p2].layer == i && items[p2].slot == SLOT_K) has_k = 1;
                        if (items[p2].layer == i && items[p2].slot == SLOT_V) has_v = 1;
                    }
                    if (!has_k || !has_v) missing = 1;
                }
                if (!missing && g.n_embd_per_layer > 0) {
                    int has_g = 0, has_p = 0, has_n = 0;
                    for (p2 = 0; p2 < n; p2++) {
                        if (items[p2].layer == i && items[p2].slot == SLOT_PLE_GATE) has_g = 1;
                        if (items[p2].layer == i && items[p2].slot == SLOT_PLE_PROJ) has_p = 1;
                        if (items[p2].layer == i && items[p2].slot == SLOT_PLE_POST) has_n = 1;
                    }
                    if (!has_g || !has_p || !has_n) missing = 1;
                }
            }
            if (!missing && g.n_embd_per_layer > 0) {
                int has_tok = 0, has_mp = 0, has_pn = 0;
                for (p2 = 0; p2 < n; p2++) {
                    if (items[p2].layer == 0 && items[p2].slot == SLOT_PLE_TOK) has_tok = 1;
                    if (items[p2].layer == 0 && items[p2].slot == SLOT_PLE_MPROJ) has_mp = 1;
                    if (items[p2].layer == 0 && items[p2].slot == SLOT_PLE_PNORM) has_pn = 1;
                }
                if (!has_tok || !has_mp || !has_pn) missing = 1;
            }
        } else {
        /* 公共必需: ffn 三件套 + 两个 norm */
        static const int common[5] = { SLOT_NORM1, SLOT_NORM2, SLOT_GATE, SLOT_UP, SLOT_DOWN };
        for (i = 1; i <= g.n_blocks && !missing; i++) {
            int s2;
            for (s2 = 0; s2 < 5; s2++) {
                int found = 0;
                for (p2 = 0; p2 < n; p2++) {
                    if (items[p2].layer == i && items[p2].slot == (uint32_t)common[s2]) { found = 1; break; }
                }
                if (!found) { missing = 1; break; }
            }
            if (missing) break;
            if (is_qwen35) {
                /* 混合层: 每层必须有 GDN 的 QKV 或 attention 的 Q(择一) */
                int has_qkv = 0, has_q = 0;
                for (p2 = 0; p2 < n; p2++) {
                    if (items[p2].layer == i && items[p2].slot == SLOT_QKV) has_qkv = 1;
                    if (items[p2].layer == i && items[p2].slot == SLOT_Q)   has_q = 1;
                }
                if (!has_qkv && !has_q) missing = 1;
            } else {
                static const int attn[4] = { SLOT_Q, SLOT_K, SLOT_V, SLOT_O };
                for (s2 = 0; s2 < 4 && !missing; s2++) {
                    int found = 0;
                    for (p2 = 0; p2 < n; p2++) {
                        if (items[p2].layer == i && items[p2].slot == (uint32_t)attn[s2]) { found = 1; break; }
                    }
                    if (!found) missing = 1;
                }
            }
        }
        } /* end !is_gemma4 */
        if (missing) {
            for (i = 0; i < (uint64_t)list.n; i++) free(list.t[i].name);
            free(list.t); free(items); wmap_close(&gmap);
            for (i = 0; i < (uint32_t)qg_n; i++) free(qg_bufs[i]);
            snprintf(err, errlen, "some transformer tensors missing from gguf");
            return -1;
        }
    }

    LlfHeader h;
    memset(&h, 0, sizeof(h));
    memcpy(h.magic, YLLM_MAGIC, 8);
    h.version = YLLM_VERSION;
    h.arch = is_qwen35 ? ARCH_QWEN35 : (is_qwen ? ARCH_QWEN : (is_gemma4 ? ARCH_GEMMA4 : ARCH_LLAMA));
    h.n_blocks = g.n_blocks;
    h.hidden = g.hidden;
    h.n_heads = g.heads;
    h.n_kv_heads = g.kv_heads;
    h.head_dim = g.key_length ? g.key_length : (g.hidden / g.heads);
    if (max_seq == 0 || (g.context_len > 0 && g.context_len < max_seq)) max_seq = g.context_len ? g.context_len : max_seq;
    h.max_seq = max_seq;
    h.dtype = DT_Q4K;
    memcpy(&h.norm_eps_bits, &g.rms_eps, 4);
    memcpy(&h.rope_theta_bits, &g.freq_base, 4);
    h.ext_ptr = 0;
    if (is_gemma4) {
        LlfGemma4Ext ext;
        memset(&ext, 0, sizeof(ext));
        memcpy(&ext.attn_logit_cap_bits, &g.attn_logit_cap, 4);
        memcpy(&ext.final_logit_cap_bits, &g.final_logit_cap, 4);
        ext.swa_window = g.swa_window;
        ext.swa_pattern = g.swa_pattern;
        ext.n_kv_shared_layers = g.n_kv_shared_layers;
        ext.n_embd_per_layer = g.n_embd_per_layer;
        ext.swa_mask = g.swa_mask;
        memcpy(h.reserved, &ext, sizeof(ext));
    }
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
        free(list.t); free(items); wmap_close(&gmap);
        for (i = 0; i < (uint32_t)qg_n; i++) free(qg_bufs[i]);
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
            if (g.n_merges > 0) {
                fprintf(vf, "#MERGES#\n%u\n", g.n_merges);
                for (i = 0; i < g.n_merges; i++) {
                    const char* s = g.merges[i];
                    for (; *s; s++) {
                        if (*s == '\n') fputs("\\n", vf);
                        else if (*s == '\r') fputs("\\r", vf);
                        else if (*s == '\\') fputs("\\\\", vf);
                        else fputc(*s, vf);
                    }
                    fputc('\n', vf);
                }
            }
            fprintf(vf, "#CHAT#\n");
            fprintf(vf, "add_bos=%d\n", g.add_bos);
            fprintf(vf, "eos_id=%d\n", g.eos_id);
            fprintf(vf, "bos_id=%d\n", g.bos_id);
            fprintf(vf, "template=");
            if (g.chat_template) {
                const char* s = g.chat_template;
                for (; *s; s++) {
                    if (*s == '\n') fputs("\\n", vf);
                    else if (*s == '\r') fputs("\\r", vf);
                    else if (*s == '\\') fputs("\\\\", vf);
                    else fputc(*s, vf);
                }
            }
            fputc('\n', vf);
            fclose(vf);
            printf("vocab written: %s (%u pieces, %u scores, add_bos=%d)\n",
                   vocab_out, g.n_tokens, g.n_scores, g.add_bos);
        } else {
            snprintf(err, errlen, "cannot write %s", vocab_out);
            rc = -1;
        }
    }
    for (i = 0; i < g.n_tokens; i++) free(g.tokens[i]);
    free(g.tokens);
    free(g.scores);
    free(g.chat_template);
    for (i = 0; i < (uint64_t)list.n; i++) free(list.t[i].name);
    free(list.t);
    for (i = 0; i < (uint32_t)qg_n; i++) free(qg_bufs[i]);
    free(items);
    wmap_close(&gmap);
    return rc;
}