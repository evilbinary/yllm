/* MiniCPM-V 4.6: 直接 mmap clip mmproj GGUF, CPU 前向对齐 llama.cpp clip_graph_minicpmv4_6 */
#include "vision.h"
#include "yllm.h"
#include "matvec.h"
#include "log.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_HDR
#define STBI_NO_PIC
#include "stb_image.h"
#pragma GCC diagnostic pop

#define CLIP_MAX_LAYERS 32
#define CLIP_MAX_POS 1024

typedef struct {
    const uint8_t* p;
    uint64_t off; /* tensor blob offset until resolved */
    uint32_t dtype; /* 0=f32 1=f16 */
    uint32_t ndim;
    uint64_t dims[4];
    char name[128];
} ClipT;

typedef struct {
    ClipT ln1_w, ln1_b, ln2_w, ln2_b;
    ClipT q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b;
    ClipT up_w, up_b, down_w, down_b;
} ClipLayer;

struct Vision {
    WMap map;
    ClipT* ts;
    int n_t;
    uint32_t image_size, patch, n_embd, n_ff, n_layer, n_head, n_out_embd;
    uint32_t insert_lid, n_merge;
    float mean[3], std[3], eps;
    ClipT patch_w, patch_b, pos_embd, post_ln_w, post_ln_b;
    ClipT mm_in_w, mm_in_b, mm_up_w, mm_up_b, mm_down_w, mm_down_b;
    ClipT vm_ln1_w, vm_ln1_b, vm_q_w, vm_q_b, vm_k_w, vm_k_b, vm_v_w, vm_v_b, vm_o_w, vm_o_b;
    ClipT vm_ds_ln_w, vm_ds_ln_b, vm_ds_up_w, vm_ds_up_b, vm_ds_down_w, vm_ds_down_b;
    ClipLayer layers[CLIP_MAX_LAYERS];
    float *x, *res, *tmp, *q, *k, *v, *attn, *ff;
};

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
static void gb_skip(GB* b, uint64_t n)
{
    if (b->p + n > b->end) { b->err = 1; return; }
    b->p += (size_t)n;
}

static void skip_val(GB* b, uint32_t t)
{
    uint64_t i, n;
    uint32_t at;
    switch (t) {
    case 0: case 1: case 7: gb_skip(b, 1); break;
    case 2: case 3: gb_skip(b, 2); break;
    case 4: case 5: case 6: gb_skip(b, 4); break;
    case 10: case 11: case 12: gb_skip(b, 8); break;
    case 8: n = gb_u64(b); gb_skip(b, n); break;
    case 9:
        at = gb_u32(b); n = gb_u64(b);
        for (i = 0; i < n && !b->err; i++) skip_val(b, at);
        break;
    default: b->err = 1;
    }
}

static ClipT* find_t(Vision* v, const char* name)
{
    int i;
    for (i = 0; i < v->n_t; i++)
        if (strcmp(v->ts[i].name, name) == 0) return &v->ts[i];
    return NULL;
}

static int req_t(Vision* v, ClipT* dst, const char* name)
{
    ClipT* t = find_t(v, name);
    if (!t) return -1;
    *dst = *t;
    return 0;
}

static uint64_t align_up_u(uint64_t x, uint64_t a)
{
    return (x + a - 1) / a * a;
}

/* ClipT.dtype: 0=F32 1=F16. yllm DT_F16=0, DT_F32=1. */
static uint32_t ydtype(const ClipT* t)
{
    return (t && t->dtype == 0) ? DT_F32 : DT_F16;
}

static float tload(const ClipT* t, uint64_t i)
{
    if (!t || !t->p) return 0.f;
    if (t->dtype == 0) return ((const float*)t->p)[i];
    return f16_to_f32(((const uint16_t*)t->p)[i]);
}

static void add_bias(float* y, const ClipT* b, uint32_t n)
{
    uint32_t i;
    if (!b || !b->p) return;
    for (i = 0; i < n; i++) y[i] += tload(b, i);
}

static void layernorm(float* y, const float* x, const ClipT* w, const ClipT* b, uint32_t n, float eps)
{
    double m = 0, v = 0;
    uint32_t i;
    for (i = 0; i < n; i++) m += x[i];
    m /= (double)n;
    for (i = 0; i < n; i++) {
        double d = x[i] - m;
        v += d * d;
    }
    v = 1.0 / sqrt(v / (double)n + (double)eps);
    for (i = 0; i < n; i++) {
        float z = (float)((x[i] - m) * v);
        if (w && w->p) z *= tload(w, i);
        if (b && b->p) z += tload(b, i);
        y[i] = z;
    }
}

static void gemm_lin(float* y, const float* x, const ClipT* w, const ClipT* bias,
                     uint32_t M, uint32_t out, uint32_t in)
{
    uint32_t dtype = ydtype(w);
    uint32_t m;
    /* 不用 collapse/嵌套 omp: MinGW libgomp 上容易崩 */
    for (m = 0; m < M; m++) {
        matmul(y + (size_t)m * out, x + (size_t)m * in, w->p, out, in, dtype);
        add_bias(y + (size_t)m * out, bias, out);
    }
}

static float gelu_erf(float x)
{
    return 0.5f * x * (1.0f + erff(x * 0.70710678118f));
}

static void gelu_erf_inplace(float* y, uint32_t n)
{
    uint32_t i;
    for (i = 0; i < n; i++) y[i] = gelu_erf(y[i]);
}

static void attn_full(float* out, const float* q, const float* k, const float* v,
                     uint32_t n, uint32_t n_head, uint32_t dh, float scale)
{
    uint32_t t;
    for (t = 0; t < n; t++) {
        uint32_t h, j, d;
        for (h = 0; h < n_head; h++) {
            const float* qt = q + ((size_t)t * n_head + h) * dh;
            float mx = -1e30f, sum = 0.f;
            float sc[CLIP_MAX_POS];
            for (j = 0; j < n; j++) {
                const float* kj = k + ((size_t)j * n_head + h) * dh;
                float acc = 0.f;
                for (d = 0; d < dh; d++) acc += qt[d] * kj[d];
                sc[j] = acc * scale;
                if (sc[j] > mx) mx = sc[j];
            }
            for (j = 0; j < n; j++) {
                sc[j] = expf(sc[j] - mx);
                sum += sc[j];
            }
            {
                float inv = 1.f / sum;
                float* o = out + ((size_t)t * n_head + h) * dh;
                memset(o, 0, (size_t)dh * 4);
                for (j = 0; j < n; j++) {
                    float a = sc[j] * inv;
                    const float* vj = v + ((size_t)j * n_head + h) * dh;
                    for (d = 0; d < dh; d++) o[d] += a * vj[d];
                }
            }
        }
    }
}

/* 2×2 窗口: 重排后每 4 token 一组自注意力 */
static void attn_win4(float* out, const float* q, const float* k, const float* v,
                     uint32_t n, uint32_t n_head, uint32_t dh, float scale)
{
    uint32_t g, ng = n / 4;
    for (g = 0; g < ng; g++) {
        uint32_t base = g * 4, ii, h, j, d;
        for (ii = 0; ii < 4; ii++) {
            uint32_t t = base + ii;
            for (h = 0; h < n_head; h++) {
                const float* qt = q + ((size_t)t * n_head + h) * dh;
                float sc[4], mx = -1e30f, sum = 0.f;
                for (j = 0; j < 4; j++) {
                    const float* kj = k + ((size_t)(base + j) * n_head + h) * dh;
                    float acc = 0.f;
                    for (d = 0; d < dh; d++) acc += qt[d] * kj[d];
                    sc[j] = acc * scale;
                    if (sc[j] > mx) mx = sc[j];
                }
                for (j = 0; j < 4; j++) { sc[j] = expf(sc[j] - mx); sum += sc[j]; }
                {
                    float inv = 1.f / sum;
                    float* o = out + ((size_t)t * n_head + h) * dh;
                    memset(o, 0, (size_t)dh * 4);
                    for (j = 0; j < 4; j++) {
                        float a = sc[j] * inv;
                        const float* vj = v + ((size_t)(base + j) * n_head + h) * dh;
                        for (d = 0; d < dh; d++) o[d] += a * vj[d];
                    }
                }
            }
        }
    }
}

static void vit_attn(Vision* vis, const ClipT* qw, const ClipT* qb, const ClipT* kw, const ClipT* kb,
                    const ClipT* vw, const ClipT* vb, const ClipT* ow, const ClipT* ob,
                    uint32_t n, int win4)
{
    uint32_t e = vis->n_embd, nh = vis->n_head, dh = e / nh;
    float scale = 1.f / sqrtf((float)dh);
    gemm_lin(vis->q, vis->tmp, qw, qb, n, e, e);
    gemm_lin(vis->k, vis->tmp, kw, kb, n, e, e);
    gemm_lin(vis->v, vis->tmp, vw, vb, n, e, e);
    if (win4) attn_win4(vis->attn, vis->q, vis->k, vis->v, n, nh, dh, scale);
    else attn_full(vis->attn, vis->q, vis->k, vis->v, n, nh, dh, scale);
    gemm_lin(vis->tmp, vis->attn, ow, ob, n, e, e);
}

static void vit_ffn(Vision* vis, const ClipT* up_w, const ClipT* up_b,
                    const ClipT* down_w, const ClipT* down_b, uint32_t n, uint32_t in, int erf)
{
    uint32_t n_ff = (uint32_t)up_w->dims[1];
    gemm_lin(vis->ff, vis->tmp, up_w, up_b, n, n_ff, in);
    if (erf) {
        uint32_t i;
        for (i = 0; i < n * n_ff; i++) vis->ff[i] = gelu_erf(vis->ff[i]);
    } else {
        gelu_inplace(vis->ff, n * n_ff);
    }
    gemm_lin(vis->tmp, vis->ff, down_w, down_b, n, vis->n_embd, n_ff);
}

static void vit_block(Vision* vis, uint32_t il, uint32_t n)
{
    ClipLayer* L = &vis->layers[il];
    uint32_t e = vis->n_embd, t;
    memcpy(vis->res, vis->x, (size_t)n * e * 4);
    for (t = 0; t < n; t++)
        layernorm(vis->tmp + (size_t)t * e, vis->x + (size_t)t * e, &L->ln1_w, &L->ln1_b, e, vis->eps);
    vit_attn(vis, &L->q_w, &L->q_b, &L->k_w, &L->k_b, &L->v_w, &L->v_b, &L->o_w, &L->o_b, n, 0);
    for (t = 0; t < n * e; t++) vis->x[t] = vis->res[t] + vis->tmp[t];
    memcpy(vis->res, vis->x, (size_t)n * e * 4);
    for (t = 0; t < n; t++)
        layernorm(vis->tmp + (size_t)t * e, vis->x + (size_t)t * e, &L->ln2_w, &L->ln2_b, e, vis->eps);
    vit_ffn(vis, &L->up_w, &L->up_b, &L->down_w, &L->down_b, n, e, 0);
    for (t = 0; t < n * e; t++) vis->x[t] = vis->res[t] + vis->tmp[t];
}

static int load_layer(Vision* v, uint32_t il)
{
    ClipLayer* L = &v->layers[il];
    char n[96];
#define LREQ(field, fmt) do { snprintf(n, sizeof(n), fmt, il); if (req_t(v, &L->field, n) != 0) return -1; } while (0)
    LREQ(ln1_w, "v.blk.%u.ln1.weight");
    LREQ(ln1_b, "v.blk.%u.ln1.bias");
    LREQ(ln2_w, "v.blk.%u.ln2.weight");
    LREQ(ln2_b, "v.blk.%u.ln2.bias");
    LREQ(q_w, "v.blk.%u.attn_q.weight");
    LREQ(q_b, "v.blk.%u.attn_q.bias");
    LREQ(k_w, "v.blk.%u.attn_k.weight");
    LREQ(k_b, "v.blk.%u.attn_k.bias");
    LREQ(v_w, "v.blk.%u.attn_v.weight");
    LREQ(v_b, "v.blk.%u.attn_v.bias");
    LREQ(o_w, "v.blk.%u.attn_out.weight");
    LREQ(o_b, "v.blk.%u.attn_out.bias");
    LREQ(up_w, "v.blk.%u.ffn_up.weight");
    LREQ(up_b, "v.blk.%u.ffn_up.bias");
    LREQ(down_w, "v.blk.%u.ffn_down.weight");
    LREQ(down_b, "v.blk.%u.ffn_down.bias");
#undef LREQ
    return 0;
}

Vision* vision_load(const char* path, char* err, size_t errlen)
{
    Vision* v = (Vision*)ycalloc(1, sizeof(*v));
    const uint8_t* data;
    uint64_t fsize, data_start, n_kv, n_tensors, i;
    uint32_t ver, alignment = 32;
    GB b;
    if (!v) { if (err) snprintf(err, errlen, "oom"); return NULL; }
    if (wmap_open(path, &v->map) != 0) {
        if (err) snprintf(err, errlen, "cannot mmap %s", path);
        free(v); return NULL;
    }
    data = (const uint8_t*)v->map.base;
    fsize = v->map.size;
    if (fsize < 24 || memcmp(data, "GGUF", 4) != 0) {
        if (err) snprintf(err, errlen, "not gguf");
        vision_free(v); return NULL;
    }
    b.p = data + 4; b.end = data + fsize; b.err = 0;
    ver = gb_u32(&b);
    n_tensors = gb_u64(&b);
    n_kv = gb_u64(&b);
    v->image_size = 448; v->patch = 14; v->n_embd = 1152; v->n_ff = 4304;
    v->n_layer = 27; v->n_head = 16; v->n_out_embd = 1024; v->insert_lid = 6;
    v->n_merge = 4; v->eps = 1e-6f;
    v->mean[0] = v->mean[1] = v->mean[2] = 0.5f;
    v->std[0] = v->std[1] = v->std[2] = 0.5f;
    (void)ver;
    for (i = 0; i < n_kv && !b.err; i++) {
        uint64_t klen = gb_u64(&b);
        char key[256];
        uint32_t typ;
        if (klen >= sizeof(key) || b.p + klen > b.end) { b.err = 1; break; }
        memcpy(key, b.p, (size_t)klen); key[klen] = 0; b.p += (size_t)klen;
        typ = gb_u32(&b);
        /* gguf stores ints as UINT32 or INT32 */
        if ((typ == 4 || typ == 5) &&
            (!strcmp(key, "clip.vision.image_size") ||
             !strcmp(key, "clip.vision.patch_size") ||
             !strcmp(key, "clip.vision.embedding_length") ||
             !strcmp(key, "clip.vision.feed_forward_length") ||
             !strcmp(key, "clip.vision.block_count") ||
             !strcmp(key, "clip.vision.attention.head_count") ||
             !strcmp(key, "clip.vision.projection_dim") ||
             !strcmp(key, "clip.vision.projector.scale_factor") ||
             !strcmp(key, "general.alignment"))) {
            uint32_t u = gb_u32(&b);
            if (!strcmp(key, "clip.vision.image_size")) v->image_size = u;
            else if (!strcmp(key, "clip.vision.patch_size")) v->patch = u;
            else if (!strcmp(key, "clip.vision.embedding_length")) v->n_embd = u;
            else if (!strcmp(key, "clip.vision.feed_forward_length")) v->n_ff = u;
            else if (!strcmp(key, "clip.vision.block_count")) v->n_layer = u;
            else if (!strcmp(key, "clip.vision.attention.head_count")) v->n_head = u;
            else if (!strcmp(key, "clip.vision.projection_dim")) v->n_out_embd = u;
            else if (!strcmp(key, "clip.vision.projector.scale_factor")) v->n_merge = u;
            else alignment = u;
        } else if (!strcmp(key, "clip.vision.attention.layer_norm_epsilon") && typ == 6) {
            uint32_t u = gb_u32(&b); memcpy(&v->eps, &u, 4);
        } else if (!strcmp(key, "clip.vision.wa_layer_indexes") && typ == 9) {
            uint32_t at = gb_u32(&b); uint64_t n = gb_u64(&b), j;
            for (j = 0; j < n && !b.err; j++) {
                if (at == 4 || at == 5) {
                    uint32_t u = gb_u32(&b);
                    if (j == 0) v->insert_lid = u;
                } else skip_val(&b, at);
            }
        } else if ((!strcmp(key, "clip.vision.image_mean") || !strcmp(key, "clip.vision.image_std")) && typ == 9) {
            uint32_t at = gb_u32(&b); uint64_t n = gb_u64(&b); uint64_t j;
            float* dst = strstr(key, "mean") ? v->mean : v->std;
            for (j = 0; j < n && !b.err; j++) {
                if (at == 6) {
                    uint32_t u = gb_u32(&b); float f; memcpy(&f, &u, 4);
                    if (j < 3) dst[j] = f;
                } else skip_val(&b, at);
            }
        } else skip_val(&b, typ);
    }
    if (b.err) { if (err) snprintf(err, errlen, "bad mmproj kv"); vision_free(v); return NULL; }
    v->ts = (ClipT*)ycalloc((size_t)n_tensors, sizeof(ClipT));
    if (!v->ts) { vision_free(v); if (err) snprintf(err, errlen, "oom"); return NULL; }
    for (i = 0; i < n_tensors && !b.err; i++) {
        uint64_t nlen = gb_u64(&b);
        uint32_t nd, d, gtype;
        ClipT* t = &v->ts[v->n_t];
        if (nlen >= sizeof(t->name) || b.p + nlen > b.end) { b.err = 1; break; }
        memcpy(t->name, b.p, (size_t)nlen); t->name[nlen] = 0; b.p += (size_t)nlen;
        nd = gb_u32(&b);
        if (nd == 0 || nd > 4) { b.err = 1; break; }
        t->ndim = nd;
        for (d = 0; d < nd; d++) t->dims[d] = gb_u64(&b);
        gtype = gb_u32(&b);
        t->off = gb_u64(&b);
        t->dtype = (gtype == 0) ? 0 : 1;
        t->p = NULL;
        v->n_t++;
    }
    if (b.err) { if (err) snprintf(err, errlen, "bad mmproj tensors"); vision_free(v); return NULL; }
    data_start = align_up_u((uint64_t)(b.p - data), alignment ? alignment : 32);
    for (i = 0; i < (uint64_t)v->n_t; i++) {
        uint64_t ne = 1, d, nb;
        v->ts[i].p = data + data_start + v->ts[i].off;
        for (d = 0; d < v->ts[i].ndim; d++) ne *= v->ts[i].dims[d];
        nb = ne * (v->ts[i].dtype == 0 ? 4ull : 2ull);
        if (v->ts[i].p < data || v->ts[i].p + nb > data + fsize) {
            if (err) snprintf(err, errlen, "tensor %s out of file", v->ts[i].name);
            vision_free(v); return NULL;
        }
    }
    if (req_t(v, &v->patch_w, "v.patch_embd.weight") != 0 ||
        req_t(v, &v->patch_b, "v.patch_embd.bias") != 0 ||
        req_t(v, &v->pos_embd, "v.position_embd.weight") != 0 ||
        req_t(v, &v->post_ln_w, "v.post_ln.weight") != 0 ||
        req_t(v, &v->post_ln_b, "v.post_ln.bias") != 0 ||
        req_t(v, &v->mm_in_w, "mm.input_norm.weight") != 0 ||
        req_t(v, &v->mm_in_b, "mm.input_norm.bias") != 0 ||
        req_t(v, &v->mm_up_w, "mm.up.weight") != 0 ||
        req_t(v, &v->mm_up_b, "mm.up.bias") != 0 ||
        req_t(v, &v->mm_down_w, "mm.down.weight") != 0 ||
        req_t(v, &v->mm_down_b, "mm.down.bias") != 0 ||
        req_t(v, &v->vm_ln1_w, "v.vit_merger.ln1.weight") != 0 ||
        req_t(v, &v->vm_ln1_b, "v.vit_merger.ln1.bias") != 0 ||
        req_t(v, &v->vm_q_w, "v.vit_merger.attn_q.weight") != 0 ||
        req_t(v, &v->vm_q_b, "v.vit_merger.attn_q.bias") != 0 ||
        req_t(v, &v->vm_k_w, "v.vit_merger.attn_k.weight") != 0 ||
        req_t(v, &v->vm_k_b, "v.vit_merger.attn_k.bias") != 0 ||
        req_t(v, &v->vm_v_w, "v.vit_merger.attn_v.weight") != 0 ||
        req_t(v, &v->vm_v_b, "v.vit_merger.attn_v.bias") != 0 ||
        req_t(v, &v->vm_o_w, "v.vit_merger.attn_out.weight") != 0 ||
        req_t(v, &v->vm_o_b, "v.vit_merger.attn_out.bias") != 0 ||
        req_t(v, &v->vm_ds_ln_w, "v.vit_merger.ds_ln.weight") != 0 ||
        req_t(v, &v->vm_ds_ln_b, "v.vit_merger.ds_ln.bias") != 0 ||
        req_t(v, &v->vm_ds_up_w, "v.vit_merger.ds_ffn_up.weight") != 0 ||
        req_t(v, &v->vm_ds_up_b, "v.vit_merger.ds_ffn_up.bias") != 0 ||
        req_t(v, &v->vm_ds_down_w, "v.vit_merger.ds_ffn_down.weight") != 0 ||
        req_t(v, &v->vm_ds_down_b, "v.vit_merger.ds_ffn_down.bias") != 0) {
        if (err) snprintf(err, errlen, "mmproj missing tensors");
        vision_free(v); return NULL;
    }
    if (v->n_layer > CLIP_MAX_LAYERS) { if (err) snprintf(err, errlen, "too many vit layers"); vision_free(v); return NULL; }
    for (i = 0; i < v->n_layer; i++)
        if (load_layer(v, (uint32_t)i) != 0) {
            if (err) snprintf(err, errlen, "missing vit layer %u", (unsigned)i);
            vision_free(v); return NULL;
        }
    {
        uint32_t np = (v->image_size / v->patch) * (v->image_size / v->patch);
        size_t cap = (size_t)np * v->n_embd;
        size_t ffcap = (size_t)np * 17216; /* merger ds up */
        if (ffcap < (size_t)np * v->n_ff) ffcap = (size_t)np * v->n_ff;
        v->x = (float*)ymalloc(cap * 4);
        v->res = (float*)ymalloc(cap * 4);
        v->tmp = (float*)ymalloc(cap * 4);
        v->q = (float*)ymalloc(cap * 4);
        v->k = (float*)ymalloc(cap * 4);
        v->v = (float*)ymalloc(cap * 4);
        v->attn = (float*)ymalloc(cap * 4);
        v->ff = (float*)ymalloc(ffcap * 4);
        if (!v->x || !v->res || !v->tmp || !v->q || !v->k || !v->v || !v->attn || !v->ff) {
            if (err) snprintf(err, errlen, "oom vis buf");
            vision_free(v); return NULL;
        }
    }
    ylog_info("vision: clip %ux%u patch=%u layers=%u embd=%u out=%u insert=%u mean=%.3f std=%.3f",
              v->image_size, v->image_size, v->patch, v->n_layer, v->n_embd, v->n_out_embd,
              v->insert_lid, (double)v->mean[0], (double)v->std[0]);
    return v;
}

void vision_free(Vision* v)
{
    if (!v) return;
    free(v->x); free(v->res); free(v->tmp); free(v->q); free(v->k); free(v->v); free(v->attn); free(v->ff);
    free(v->ts);
    wmap_close(&v->map);
    free(v);
}

int vision_n_tokens(const Vision* v)
{
    uint32_t g;
    if (!v) return 0;
    g = v->image_size / v->patch;
    g /= 4; /* 16x: 2x2 vit merger + 2x2 final */
    return (int)(g * g);
}

int vision_hidden(const Vision* v)
{
    return v ? (int)v->n_out_embd : 0;
}

static void patch_embed(Vision* vis, const float* chw, uint32_t gh, uint32_t gw)
{
    uint32_t ps = vis->patch, e = vis->n_embd, oc, py, px, ic, ky, kx;
    /* w layout ggml: [ps, ps, 3, oc]. 不用 OpenMP: MinGW libgomp 在此核上 SIGSEGV */
    for (py = 0; py < gh; py++) {
        for (px = 0; px < gw; px++) {
            uint32_t t = py * gw + px;
            float* dst = vis->x + (size_t)t * e;
            for (oc = 0; oc < e; oc++) {
                float acc = tload(&vis->patch_b, oc);
                for (ic = 0; ic < 3; ic++)
                    for (ky = 0; ky < ps; ky++)
                        for (kx = 0; kx < ps; kx++) {
                            uint32_t ix = px * ps + kx, iy = py * ps + ky;
                            float pix = chw[(size_t)ic * vis->image_size * vis->image_size + iy * vis->image_size + ix];
                            uint64_t wi = (uint64_t)kx + (uint64_t)ps * (ky + ps * (ic + 3 * oc));
                            acc += pix * tload(&vis->patch_w, wi);
                        }
                dst[oc] = acc;
            }
        }
    }
}

static void add_pos(Vision* vis, uint32_t gh, uint32_t gw)
{
    uint32_t e = vis->n_embd, i, j, t = 0;
    for (i = 0; i < gh; i++) {
        int bh = (int)floor(70.0 * (double)i / (double)gh);
        for (j = 0; j < gw; j++, t++) {
            int bw = (int)floor(70.0 * (double)j / (double)gw);
            uint32_t row = (uint32_t)(bh * 70 + bw);
            float* dst = vis->x + (size_t)t * e;
            uint32_t c;
            for (c = 0; c < e; c++) dst[c] += tload(&vis->pos_embd, (uint64_t)row * e + c);
        }
    }
}

static void gather_2x2(const float* src, float* dst, uint32_t gh, uint32_t gw, uint32_t e,
                       int off_r, int off_c)
{
    uint32_t i, j, c, ds_h = gh / 2, ds_w = gw / 2;
    for (i = 0; i < ds_h; i++)
        for (j = 0; j < ds_w; j++) {
            uint32_t s = (2 * i + (uint32_t)off_r) * gw + (2 * j + (uint32_t)off_c);
            const float* sp = src + (size_t)s * e;
            float* dp = dst + ((size_t)i * ds_w + j) * e;
            for (c = 0; c < e; c++) dp[c] = sp[c];
        }
}

static int encode_448(Vision* vis, const float* chw, float* out)
{
    uint32_t gh = vis->image_size / vis->patch, gw = gh;
    uint32_t n = gh * gw, e = vis->n_embd, il, t;
    uint32_t half = gh / 2;
    patch_embed(vis, chw, gh, gw);
    add_pos(vis, gh, gw);
    for (il = 0; il <= vis->insert_lid && il < vis->n_layer; il++)
        vit_block(vis, il, n);

    /* window reorder + window attn + residual */
    {
        float* re = vis->res; /* reuse as scratch order */
        uint32_t k = 0, wi, wj, idx;
        int32_t widx[CLIP_MAX_POS], inv[CLIP_MAX_POS];
        for (wi = 0; wi < half; wi++)
            for (wj = 0; wj < half; wj++) {
                widx[k++] = (int32_t)((2 * wi) * gw + (2 * wj));
                widx[k++] = (int32_t)((2 * wi) * gw + (2 * wj + 1));
                widx[k++] = (int32_t)((2 * wi + 1) * gw + (2 * wj));
                widx[k++] = (int32_t)((2 * wi + 1) * gw + (2 * wj + 1));
            }
        for (idx = 0; idx < n; idx++) inv[widx[idx]] = (int32_t)idx;
        memcpy(vis->res, vis->x, (size_t)n * e * 4);
        for (t = 0; t < n; t++)
            layernorm(vis->tmp + (size_t)t * e, vis->x + (size_t)t * e,
                      &vis->vm_ln1_w, &vis->vm_ln1_b, e, vis->eps);
        for (t = 0; t < n; t++)
            memcpy(vis->q + (size_t)t * e, vis->tmp + (size_t)widx[t] * e, (size_t)e * 4);
        memcpy(vis->tmp, vis->q, (size_t)n * e * 4);
        vit_attn(vis, &vis->vm_q_w, &vis->vm_q_b, &vis->vm_k_w, &vis->vm_k_b,
                 &vis->vm_v_w, &vis->vm_v_b, &vis->vm_o_w, &vis->vm_o_b, n, 1);
        for (t = 0; t < n; t++)
            memcpy(vis->q + (size_t)t * e, vis->tmp + (size_t)inv[t] * e, (size_t)e * 4);
        for (t = 0; t < n * e; t++) vis->x[t] = vis->res[t] + vis->q[t];
        (void)re;
    }

    /* 2x2 downsample MLP (tanh gelu) + mean residual */
    {
        uint32_t ds = half * half, c;
        float *p0 = vis->q, *p1 = vis->k, *p2 = vis->v, *p3 = vis->attn;
        float* cat = vis->ff;
        gather_2x2(vis->x, p0, gh, gw, e, 0, 0);
        gather_2x2(vis->x, p1, gh, gw, e, 0, 1);
        gather_2x2(vis->x, p2, gh, gw, e, 1, 0);
        gather_2x2(vis->x, p3, gh, gw, e, 1, 1);
        for (t = 0; t < ds; t++) {
            float* d = vis->res + (size_t)t * e;
            for (c = 0; c < e; c++)
                d[c] = 0.25f * (p0[t * e + c] + p1[t * e + c] + p2[t * e + c] + p3[t * e + c]);
            memcpy(cat + (size_t)t * (e * 4), p0 + (size_t)t * e, (size_t)e * 4);
            memcpy(cat + (size_t)t * (e * 4) + e, p1 + (size_t)t * e, (size_t)e * 4);
            memcpy(cat + (size_t)t * (e * 4) + 2 * e, p2 + (size_t)t * e, (size_t)e * 4);
            memcpy(cat + (size_t)t * (e * 4) + 3 * e, p3 + (size_t)t * e, (size_t)e * 4);
            layernorm(vis->tmp + (size_t)t * (e * 4), cat + (size_t)t * (e * 4),
                      &vis->vm_ds_ln_w, &vis->vm_ds_ln_b, e * 4, vis->eps);
        }
        {
            uint32_t n_ff = (uint32_t)vis->vm_ds_up_w.dims[1];
            gemm_lin(vis->ff, vis->tmp, &vis->vm_ds_up_w, &vis->vm_ds_up_b, ds, n_ff, e * 4);
            gelu_inplace(vis->ff, ds * n_ff);
            gemm_lin(vis->tmp, vis->ff, &vis->vm_ds_down_w, &vis->vm_ds_down_b, ds, e, n_ff);
        }
        for (t = 0; t < ds * e; t++) vis->x[t] = vis->tmp[t] + vis->res[t];
        n = ds; gh = half; gw = half;
    }

    for (il = vis->insert_lid + 1; il < vis->n_layer; il++)
        vit_block(vis, il, n);
    for (t = 0; t < n; t++)
        layernorm(vis->tmp + (size_t)t * e, vis->x + (size_t)t * e, &vis->post_ln_w, &vis->post_ln_b, e, vis->eps);
    memcpy(vis->x, vis->tmp, (size_t)n * e * 4);

    /* final 2x2 merger GELU-erf → 1024 */
    {
        uint32_t half2 = gh / 2, ds = half2 * half2, c;
        float *p0 = vis->q, *p1 = vis->k, *p2 = vis->v, *p3 = vis->attn;
        gather_2x2(vis->x, p0, gh, gw, e, 0, 0);
        gather_2x2(vis->x, p1, gh, gw, e, 0, 1);
        gather_2x2(vis->x, p2, gh, gw, e, 1, 0);
        gather_2x2(vis->x, p3, gh, gw, e, 1, 1);
        for (t = 0; t < ds; t++) {
            memcpy(vis->ff + (size_t)t * (e * 4), p0 + (size_t)t * e, (size_t)e * 4);
            memcpy(vis->ff + (size_t)t * (e * 4) + e, p1 + (size_t)t * e, (size_t)e * 4);
            memcpy(vis->ff + (size_t)t * (e * 4) + 2 * e, p2 + (size_t)t * e, (size_t)e * 4);
            memcpy(vis->ff + (size_t)t * (e * 4) + 3 * e, p3 + (size_t)t * e, (size_t)e * 4);
            layernorm(vis->tmp + (size_t)t * (e * 4), vis->ff + (size_t)t * (e * 4),
                      &vis->mm_in_w, &vis->mm_in_b, e * 4, vis->eps);
        }
        {
            uint32_t n_up = (uint32_t)vis->mm_up_w.dims[1];
            gemm_lin(vis->ff, vis->tmp, &vis->mm_up_w, &vis->mm_up_b, ds, n_up, e * 4);
            gelu_erf_inplace(vis->ff, ds * n_up);
            gemm_lin(out, vis->ff, &vis->mm_down_w, &vis->mm_down_b, ds, vis->n_out_embd, n_up);
        }
        (void)c;
    }
    return 0;
}

int vision_encode_image(Vision* v, const char* image_path, float* out, int max_tok, char* err, size_t errlen)
{
    int w = 0, h = 0, c = 0, x, y, ic;
    unsigned char* img;
    float* chw;
    uint32_t S;
    int ntok;
    if (!v || !image_path || !out) { if (err) snprintf(err, errlen, "bad vision args"); return -1; }
    ntok = vision_n_tokens(v);
    if (max_tok < ntok) { if (err) snprintf(err, errlen, "out tokens %d < %d", max_tok, ntok); return -1; }
    img = stbi_load(image_path, &w, &h, &c, 3);
    if (!img) { if (err) snprintf(err, errlen, "cannot load image %s", image_path); return -1; }
    S = v->image_size;
    chw = (float*)ymalloc((size_t)3 * S * S * 4);
    if (!chw) { stbi_image_free(img); if (err) snprintf(err, errlen, "oom"); return -1; }
    for (y = 0; y < (int)S; y++) {
        float fy = ((float)y + 0.5f) * (float)h / (float)S - 0.5f;
        int y0 = (int)floorf(fy), y1 = y0 + 1;
        float wy = fy - (float)y0;
        if (y0 < 0) { y0 = 0; wy = 0; }
        if (y1 >= h) { y1 = h - 1; }
        if (y0 >= h) y0 = h - 1;
        for (x = 0; x < (int)S; x++) {
            float fx = ((float)x + 0.5f) * (float)w / (float)S - 0.5f;
            int x0 = (int)floorf(fx), x1 = x0 + 1;
            float wx = fx - (float)x0;
            if (x0 < 0) { x0 = 0; wx = 0; }
            if (x1 >= w) x1 = w - 1;
            if (x0 >= w) x0 = w - 1;
            for (ic = 0; ic < 3; ic++) {
                float p00 = img[(y0 * w + x0) * 3 + ic];
                float p10 = img[(y0 * w + x1) * 3 + ic];
                float p01 = img[(y1 * w + x0) * 3 + ic];
                float p11 = img[(y1 * w + x1) * 3 + ic];
                float p = (p00 * (1 - wx) + p10 * wx) * (1 - wy) + (p01 * (1 - wx) + p11 * wx) * wy;
                chw[(size_t)ic * S * S + (uint32_t)y * S + (uint32_t)x] = (p / 255.f - v->mean[ic]) / v->std[ic];
            }
        }
    }
    stbi_image_free(img);
    ylog_info("vision: encoding %ux%u (this is slow on CPU)", S, S);
    encode_448(v, chw, out);
    free(chw);
    ylog_info("vision: encoded %dx%d -> %d tokens hidden=%u", w, h, ntok, v->n_out_embd);
    return ntok;
}
