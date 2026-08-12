#include "yllm.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static uint64_t align_up(uint64_t v, uint64_t a)
{
    return (v + a - 1) & ~(a - 1);
}

static void write_at(FILE* f, uint64_t off, const void* data, size_t n)
{
    if (fseeko(f, (off_t)off, SEEK_SET) != 0) { fprintf(stderr, "seek failed\n"); exit(1); }
    if (fwrite(data, 1, n, f) != n) { fprintf(stderr, "write failed\n"); exit(1); }
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

typedef struct {
    uint32_t layer;
    int slot;
    const STTensor* t;
} PlanItem;

static int plan_compare(const void* a, const void* b)
{
    const PlanItem* x = (const PlanItem*)a;
    const PlanItem* y = (const PlanItem*)b;
    if (x->layer != y->layer) return (int)(x->layer - y->layer);
    return x->slot - y->slot;
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
        if (slot >= SLOT_NORM1 && slot <= SLOT_DOWN && layer > (int)n_blocks) n_blocks = (uint32_t)layer;
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

    PlanItem* plan = (PlanItem*)ymalloc((size_t)list.n * sizeof(PlanItem));
    int nplan = 0;
    for (i = 0; i < list.n; i++) {
        int layer;
        int slot = st_slot_for(list.t[i].name, &layer);
        if (slot == SP_EMBED) { plan[nplan].layer = 0; plan[nplan].slot = 0; }
        else if (slot == SP_FINALNORM) { plan[nplan].layer = n_blocks + 1; plan[nplan].slot = 0; }
        else if (slot == SP_OUTPUT) { plan[nplan].layer = n_blocks + 2; plan[nplan].slot = 0; }
        else if (slot >= SLOT_NORM1 && slot <= SLOT_DOWN) { plan[nplan].layer = (uint32_t)layer; plan[nplan].slot = slot; }
        else continue;
        plan[nplan].t = &list.t[i];
        nplan++;
    }
    if (nplan == 0) { free(plan); free(data); snprintf(err, errlen, "no recognized tensors"); return -1; }
    qsort(plan, (size_t)nplan, sizeof(PlanItem), plan_compare);

    uint32_t n_layers = n_blocks + 3;
    uint32_t* per = (uint32_t*)ycalloc(n_layers, 4);
    for (i = 0; i < nplan; i++) per[plan[i].layer]++;

    LlfTensorMeta* metas = (LlfTensorMeta*)ycalloc((size_t)n_layers * 9, LLF_TENSOR_META_SIZE);
    LlfLayerDir* dir = (LlfLayerDir*)ycalloc(n_layers, LLF_DIR_ENTRY_SIZE);
    uint64_t dir_size = (uint64_t)n_layers * LLF_DIR_ENTRY_SIZE;
    uint64_t cursor = align_up(LLF_HEADER_SIZE + dir_size, LLF_ALIGN);

    for (i = 0; i < nplan; i++) {
        uint32_t li = plan[i].layer;
        int slot = plan[i].slot;
        LlfTensorMeta* tm = &metas[(size_t)li * 9 + slot];
        const STTensor* t = plan[i].t;
        memset(tm, 0, sizeof(*tm));
        snprintf(tm->name, sizeof(tm->name), "%s", t->name);
        tm->dtype = DT_F16;
        tm->ndim = t->ndim;
        memcpy(tm->shape, t->shape, sizeof(tm->shape));
        if (per[li] == 0) dir[li].offset = cursor;
        tm->offset = cursor - dir[li].offset;
        tm->size = (uint64_t)t->shape[0] * (t->ndim > 1 ? t->shape[1] : 1) * 2;
        cursor += tm->size;
        per[li]++;
    }
    for (i = 0; i < (int)n_layers; i++) {
        uint64_t first = dir[i].offset;
        uint64_t loff = align_up(first, LLF_ALIGN);
        uint64_t lend = 0;
        uint32_t s2;
        for (s2 = 0; s2 < per[i]; s2++) {
            LlfTensorMeta* tm = &metas[(size_t)i * 9 + s2];
            tm->offset += loff - first;
            if (tm->offset + tm->size > lend) lend = tm->offset + tm->size;
        }
        lend = align_up(lend, LLF_ALIGN);
        dir[i].offset = loff;
        dir[i].size = lend - loff;
        dir[i].n_tensors = per[i];
    }
    h.file_size = align_up(cursor, LLF_ALIGN);

    FILE* out = fopen(out_path, "wb");
    if (!out) {
        free(per); free(metas); free(dir); free(plan); free(data);
        snprintf(err, errlen, "cannot write %s", out_path);
        return -1;
    }
    {
        uint8_t hb[LLF_HEADER_SIZE];
        memset(hb, 0, sizeof(hb));
        memcpy(hb, &h, sizeof(h));
        write_at(out, 0, hb, sizeof(hb));
    }
    write_at(out, LLF_HEADER_SIZE, dir, dir_size);
    write_at(out, LLF_HEADER_SIZE + dir_size, metas, (size_t)n_layers * 9 * LLF_TENSOR_META_SIZE);

    uint8_t* buf = (uint8_t*)ymalloc(1 << 22);
    for (i = 0; i < nplan; i++) {
        const STTensor* t = plan[i].t;
        uint32_t li = plan[i].layer;
        int slot = plan[i].slot;
        LlfTensorMeta* tm = &metas[(size_t)li * 9 + slot];
        const uint8_t* sp = data + 8 + list.hlen + t->data_off;
        if (t->dtype == DT_F16) {
            memcpy(buf, sp, t->nbytes);
        } else if (t->dtype == DT_BF16) {
            bf16_to_f16_buf((const uint16_t*)sp, (uint16_t*)buf, t->nbytes / 2);
        } else {
            f32_to_f16_buf((const float*)sp, (uint16_t*)buf, t->nbytes / 4);
        }
        write_at(out, dir[li].offset + tm->offset, buf, tm->size);
    }
    free(buf);
    fclose(out);
    free(per); free(metas); free(dir); free(plan); free(data);
    return 0;
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
    uint64_t cursor = align_up(LLF_HEADER_SIZE + dir_size, LLF_ALIGN);
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
        else if (li == blocks + 2) { nt = 1; r0 = 1; }
        else { nt = 9; r0 = 0; }
        dir[li].offset = cursor;
        dir[li].n_tensors = nt;
        uint32_t s2;
        for (s2 = 0; s2 < nt; s2++) {
            uint32_t idx = (li == 0 || li == blocks + 1 || li == blocks + 2) ? r0 : s2;
            LlfTensorMeta* tm = &metas[(size_t)li * 9 + s2];
            memset(tm, 0, sizeof(*tm));
            snprintf(tm->name, sizeof(tm->name), "%s", nm[idx]);
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
