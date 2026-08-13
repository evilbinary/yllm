#include "yllm.h"
#include "convert.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
    const uint8_t* p;
    const uint8_t* end;
} JP;

static void jp_ws(JP* j)
{
    while (j->p < j->end) {
        uint8_t c = *j->p;
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') j->p++;
        else break;
    }
}

static int jp_match(JP* j, const char* s)
{
    jp_ws(j);
    size_t n = strlen(s);
    if ((size_t)(j->end - j->p) < n) return 0;
    if (memcmp(j->p, s, n) != 0) return 0;
    j->p += n;
    return 1;
}

static int jp_str(JP* j, char* out, size_t max)
{
    jp_ws(j);
    if (j->p >= j->end || *j->p != '"') return -1;
    j->p++;
    size_t n = 0;
    while (j->p < j->end && *j->p != '"') {
        if (*j->p == '\\') {
            j->p++;
            if (j->p >= j->end) return -1;
            if (n + 1 < max) out[n++] = *j->p;
            j->p++;
        } else {
            if (n + 1 < max) out[n++] = *j->p;
            j->p++;
        }
    }
    if (j->p >= j->end) return -1;
    j->p++;
    out[n] = 0;
    return 0;
}

static int jp_num(JP* j, int64_t* out)
{
    jp_ws(j);
    int neg = 0;
    if (j->p < j->end && *j->p == '-') { neg = 1; j->p++; }
    int64_t v = 0;
    int any = 0;
    while (j->p < j->end && *j->p >= '0' && *j->p <= '9') { v = v * 10 + (*j->p - '0'); j->p++; any = 1; }
    if (!any) return -1;
    *out = neg ? -v : v;
    return 0;
}

static int jp_skip_value(JP* j)
{
    jp_ws(j);
    if (j->p >= j->end) return -1;
    uint8_t c = *j->p;
    if (c == '{') {
        j->p++;
        if (jp_match(j, "}")) return 0;
        for (;;) {
            if (jp_str(j, (char[1]){0}, 1) != 0) return -1;
            if (!jp_match(j, ":")) return -1;
            if (jp_skip_value(j) != 0) return -1;
            if (jp_match(j, "}")) return 0;
            if (!jp_match(j, ",")) return -1;
        }
    }
    if (c == '[') {
        j->p++;
        if (jp_match(j, "]")) return 0;
        for (;;) {
            if (jp_skip_value(j) != 0) return -1;
            if (jp_match(j, "]")) return 0;
            if (!jp_match(j, ",")) return -1;
        }
    }
    if (c == '"') { char t[1]; return jp_str(j, t, 1); }
    if (jp_match(j, "true")) return 0;
    if (jp_match(j, "false")) return 0;
    if (jp_match(j, "null")) return 0;
    {
        int64_t v;
        return jp_num(j, &v);
    }
}

static int jp_array_of_int(JP* j, int64_t* out, int maxn)
{
    jp_ws(j);
    if (!jp_match(j, "[")) return -1;
    if (jp_match(j, "]")) return 0;
    int n = 0;
    for (;;) {
        if (n >= maxn) return -1;
        if (jp_num(j, &out[n++]) != 0) return -1;
        if (jp_match(j, "]")) return 0;
        if (!jp_match(j, ",")) return -1;
    }
}

typedef struct {
    char* name;
    uint32_t dtype;
    uint32_t ndim;
    uint32_t shape[4];
    uint64_t data_off;
    uint64_t nbytes;
} STTensor;

typedef struct {
    STTensor* t;
    int n;
    int cap;
    uint64_t hlen;
} STList;

static void st_add(STList* l, STTensor* v)
{
    if (l->n == l->cap) {
        l->cap = l->cap ? l->cap * 2 : 64;
        l->t = (STTensor*)realloc(l->t, (size_t)l->cap * sizeof(STTensor));
        if (!l->t) exit(1);
    }
    l->t[l->n++] = *v;
}

static int parse_safetensors_header(const uint8_t* data, uint64_t size, STList* list)
{
    if (size < 8) return -1;
    memcpy(&list->hlen, data, 8);
    if (8 + list->hlen > size) return -1;
    JP j;
    j.p = data + 8;
    j.end = data + 8 + list->hlen;
    if (!jp_match(&j, "{")) return -1;
    if (jp_match(&j, "}")) return 0;
    for (;;) {
        char key[512];
        if (jp_str(&j, key, sizeof(key)) != 0) return -1;
        if (!jp_match(&j, ":")) return -1;
        if (strcmp(key, "__metadata__") == 0) {
            if (jp_skip_value(&j) != 0) return -1;
        } else {
            STTensor t;
            memset(&t, 0, sizeof(t));
            t.name = ystrdup(key);
            if (!jp_match(&j, "{")) { free(t.name); return -1; }
            char k[64], v[64];
            for (;;) {
                if (jp_str(&j, k, sizeof(k)) != 0) { free(t.name); return -1; }
                if (!jp_match(&j, ":")) { free(t.name); return -1; }
                if (strcmp(k, "dtype") == 0) {
                    if (jp_str(&j, v, sizeof(v)) != 0) { free(t.name); return -1; }
                    if (strcmp(v, "F16") == 0 || strcmp(v, "f16") == 0) t.dtype = DT_F16;
                    else if (strcmp(v, "F32") == 0 || strcmp(v, "f32") == 0) t.dtype = DT_F32;
                    else if (strcmp(v, "BF16") == 0 || strcmp(v, "bf16") == 0) t.dtype = DT_BF16;
                    else { free(t.name); return -1; }
                } else if (strcmp(k, "shape") == 0) {
                    int64_t sh[4] = {0, 0, 0, 0};
                    int nn;
                    if (jp_array_of_int(&j, sh, 4) != 0) { free(t.name); return -1; }
                    nn = 4;
                    while (nn > 0 && sh[nn - 1] == 0) nn--;
                    t.ndim = (uint32_t)nn;
                    int d;
                    for (d = 0; d < 4; d++) t.shape[d] = (uint32_t)sh[d];
                } else if (strcmp(k, "data_offsets") == 0) {
                    int64_t offs[2];
                    if (jp_array_of_int(&j, offs, 2) != 0) { free(t.name); return -1; }
                    t.data_off = (uint64_t)offs[0];
                    t.nbytes = (uint64_t)(offs[1] - offs[0]);
                } else {
                    if (jp_skip_value(&j) != 0) { free(t.name); return -1; }
                }
                if (jp_match(&j, "}")) break;
                if (!jp_match(&j, ",")) { free(t.name); return -1; }
            }
            st_add(list, &t);
        }
        if (jp_match(&j, "}")) return 0;
        if (!jp_match(&j, ",")) return -1;
    }
}

enum { SP_EMBED = -2, SP_FINALNORM = -3, SP_OUTPUT = -4 };

static int st_slot_for(const char* name, int* layer)
{
    static const struct { const char* key; int slot; } tab[] = {
        { "model.embed_tokens.weight", SLOT_EMBED },
        { "model.norm.weight", SP_FINALNORM },
        { "model.layers.*.input_layernorm.weight", SLOT_NORM1 },
        { "model.layers.*.post_attention_layernorm.weight", SLOT_NORM2 },
        { "model.layers.*.self_attn.q_proj.weight", SLOT_Q },
        { "model.layers.*.self_attn.k_proj.weight", SLOT_K },
        { "model.layers.*.self_attn.v_proj.weight", SLOT_V },
        { "model.layers.*.self_attn.o_proj.weight", SLOT_O },
        { "model.layers.*.mlp.gate_proj.weight", SLOT_GATE },
        { "model.layers.*.mlp.up_proj.weight", SLOT_UP },
        { "model.layers.*.mlp.down_proj.weight", SLOT_DOWN },
        { "lm_head.weight", SP_OUTPUT },
    };
    size_t i;
    for (i = 0; i < sizeof(tab) / sizeof(tab[0]); i++) {
        const char* p = tab[i].key;
        const char* star = strstr(p, "*");
        int slot = (int)tab[i].slot;
        if (slot == SLOT_EMBED) slot = SP_EMBED;
        if (!star) {
            if (strcmp(name, p) == 0) {
                *layer = slot;
                return slot;
            }
            continue;
        }
        size_t l1 = (size_t)(star - p);
        const char* q = star + 2;
        size_t l2 = strlen(q);
        if (strncmp(name, p, l1) != 0) continue;
        if (strcmp(name + strlen(name) - l2, q) != 0) continue;
        size_t mid = strlen(name) - l1 - l2;
        if (mid == 0 || mid > 8) continue;
        char nb[16];
        size_t j;
        for (j = 0; j < mid; j++) nb[j] = name[l1 + j];
        nb[mid] = 0;
        *layer = atoi(nb);
        return (int)tab[i].slot;
    }
    return -1;
}

static const STTensor* find_tensor(const STList* l, const char* name)
{
    int i;
    for (i = 0; i < l->n; i++) {
        if (strcmp(l->t[i].name, name) == 0) return &l->t[i];
    }
    return NULL;
}

int convert_safetensors(const char* in_path, const char* out_path, uint32_t max_seq, char* err, size_t errlen)
{
    uint64_t fsize = 0;
    if (yfile_size(in_path, &fsize) != 0) { snprintf(err, errlen, "cannot open %s", in_path); return -1; }
    uint8_t* data = (uint8_t*)ymalloc((size_t)fsize);
    FILE* f = fopen(in_path, "rb");
    if (!f) { free(data); snprintf(err, errlen, "cannot open %s", in_path); return -1; }
    if (fread(data, 1, (size_t)fsize, f) != fsize) { fclose(f); free(data); snprintf(err, errlen, "read failed"); return -1; }
    fclose(f);

    STList list;
    memset(&list, 0, sizeof(list));
    if (parse_safetensors_header(data, fsize, &list) != 0) {
        free(data);
        snprintf(err, errlen, "bad safetensors header");
        return -1;
    }

    const STTensor* tt = find_tensor(&list, "model.embed_tokens.weight");
    if (!tt) { free(data); snprintf(err, errlen, "no embedding tensor"); return -1; }
    uint32_t vocab = tt->shape[0];
    uint32_t hidden = tt->shape[1];

    const STTensor* th = find_tensor(&list, "lm_head.weight");
    if (th) vocab = th->shape[0];

    uint32_t n_blocks = 0;
    int i;
    for (i = 0; i < list.n; i++) {
        int layer;
        int slot = st_slot_for(list.t[i].name, &layer);
        if (slot >= SLOT_NORM1 && slot <= SLOT_DOWN && layer + 1 > (int)n_blocks) n_blocks = (uint32_t)layer + 1;
    }
    if (n_blocks == 0) { free(data); snprintf(err, errlen, "no transformer blocks found"); return -1; }

    const STTensor* q = find_tensor(&list, "model.layers.0.self_attn.q_proj.weight");
    const STTensor* k = find_tensor(&list, "model.layers.0.self_attn.k_proj.weight");
    if (!q || !k) { free(data); snprintf(err, errlen, "no q/k proj"); return -1; }
    uint32_t heads = q->shape[0] / hidden;
    uint32_t kv_heads = k->shape[0] / hidden;
    uint32_t head_dim = q->shape[0] / heads;

    LlfHeader h;
    memset(&h, 0, sizeof(h));
    memcpy(h.magic, YLLM_MAGIC, 8);
    h.version = YLLM_VERSION;
    h.arch = ARCH_LLAMA;
    h.n_blocks = n_blocks;
    h.vocab = vocab;
    h.hidden = hidden;
    h.n_heads = heads;
    h.n_kv_heads = kv_heads;
    h.head_dim = head_dim;
    h.max_seq = max_seq;
    h.dtype = DT_F16;
    {
        float eps = 1e-5f;
        memcpy(&h.norm_eps_bits, &eps, 4);
        float theta = 10000.0f;
        memcpy(&h.rope_theta_bits, &theta, 4);
    }

    ConvItem* items = (ConvItem*)ymalloc((size_t)list.n * sizeof(ConvItem));
    const STTensor** stmap = (const STTensor**)ymalloc((size_t)list.n * sizeof(STTensor*));
    int n = 0;
    for (i = 0; i < list.n; i++) {
        int layer;
        int slot = st_slot_for(list.t[i].name, &layer);
        if (slot == SP_EMBED) { items[n].layer = 0; items[n].slot = 0; }
        else if (slot == SP_FINALNORM) { items[n].layer = n_blocks + 1; items[n].slot = 0; }
        else if (slot == SP_OUTPUT) { items[n].layer = n_blocks + 2; items[n].slot = 0; }
        else if (slot >= SLOT_NORM1 && slot <= SLOT_DOWN) { items[n].layer = (uint32_t)layer + 1; items[n].slot = (uint32_t)slot; }
        else continue;
        const STTensor* t = &list.t[i];
        items[n].dtype = DT_F16;
        items[n].ndim = t->ndim;
        memcpy(items[n].shape, t->shape, sizeof(items[n].shape));
        items[n].nbytes = (uint64_t)t->shape[0] * (t->ndim > 1 ? t->shape[1] : 1) * 2;
        /* reject truncated/corrupt file: tensor data must fit inside the file */
        {
            uint64_t soff = (uint64_t)8 + list.hlen + t->data_off;
            if (soff > fsize || items[n].nbytes > fsize - soff) {
                free(stmap); free(items); free(data);
                snprintf(err, errlen, "safetensors tensor '%s' data out of range (truncated file?)", t->name);
                return -1;
            }
        }
        snprintf(items[n].name, sizeof(items[n].name), "%s", t->name);
        items[n].src = data + 8 + list.hlen + t->data_off;
        items[n].src_off = 0;
        stmap[n] = t;
        n++;
    }
    if (n == 0) { free(stmap); free(items); free(data); snprintf(err, errlen, "no recognized tensors"); return -1; }

    uint8_t** bufs = (uint8_t**)ycalloc((size_t)n, sizeof(uint8_t*));
    for (i = 0; i < n; i++) {
        const STTensor* t = stmap[i];
        if (t->dtype == DT_BF16) {
            bufs[i] = (uint8_t*)ymalloc(t->nbytes / 2 * 2);
            bf16_to_f16_buf((const uint16_t*)items[i].src, (uint16_t*)bufs[i], t->nbytes / 2);
            items[i].src = bufs[i];
        } else if (t->dtype == DT_F32) {
            bufs[i] = (uint8_t*)ymalloc(t->nbytes / 4 * 2);
            f32_to_f16_buf((const float*)items[i].src, (uint16_t*)bufs[i], t->nbytes / 4);
            items[i].src = bufs[i];
        }
    }

    int rc = llf_emit(out_path, &h, items, n, err, errlen);
    for (i = 0; i < n; i++) free(bufs[i]);
    free(bufs);
    free(stmap);
    free(items);
    for (i = 0; i < list.n; i++) free(list.t[i].name);
    free(list.t);
    free(data);
    return rc;
}
