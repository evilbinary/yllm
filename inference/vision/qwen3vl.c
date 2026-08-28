/* Qwen3-VL CLIP: 对齐 llama.cpp clip_graph_qwen3vl (静帧) */
#include "vision_impl.h"
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
#ifdef __AVX2__
#include <immintrin.h>
#endif
#ifdef _WIN32
#include <malloc.h>
#define vis_alloca _alloca
#else
#include <alloca.h>
#define vis_alloca alloca
#endif
#include "stb_image.h"

#define Q3V_MAX_LAYERS 32
#define Q3V_MAX_DS 8

typedef struct {
    const uint8_t* p;
    uint64_t off;
    uint32_t dtype; /* 0=f32 1=f16 */
    uint32_t gtype;
    uint32_t owned; /* p 为 load 时解出的堆缓冲 */
    uint32_t ndim;
    uint64_t dims[4];
    char name[128];
} ClipT;

typedef struct {
    ClipT ln1_w, ln1_b, ln2_w, ln2_b;
    ClipT qkv_w, qkv_b, o_w, o_b;
    ClipT up_w, up_b, gate_w, gate_b, down_w, down_b;
    int has_gate;
    int has_ds;
    ClipT ds_ln_w, ds_ln_b, ds_fc1_w, ds_fc1_b, ds_fc2_w, ds_fc2_b;
} ClipLayer;

struct Q3v {
    WMap map;
    ClipT* ts;
    int n_t;
    uint32_t image_size, patch, n_embd, n_ff, n_layer, n_head, n_out_embd, n_merge;
    float mean[3], std[3], eps;
    int ffn_quick; /* 1 = gelu_quick, 0 = tanh gelu */
    int n_ds;
    uint32_t ds_il[Q3V_MAX_DS];
    ClipT patch_w, patch_w1, patch_b, pos_embd;
    ClipT pre_ln_w, pre_ln_b, post_ln_w, post_ln_b;
    ClipT mm0_w, mm0_b, mm2_w, mm2_b;
    int has_pre_ln, has_post_ln, has_patch1;
    ClipLayer layers[Q3V_MAX_LAYERS];
    float *x, *res, *tmp, *q, *k, *v, *attn, *ff, *qkv, *sc;
    float *patch_wf;
    float *pos_f;
    uint32_t pos_side, pos_n;
    float *gemm_row;
    uint32_t gemm_nthr, gemm_in_cap, npos_max, sc_nthr;
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
    memcpy(&v, b->p, 4); b->p += 4; return v;
}
static uint64_t gb_u64(GB* b)
{
    uint64_t v;
    if (b->p + 8 > b->end) { b->err = 1; return 0; }
    memcpy(&v, b->p, 8); b->p += 8; return v;
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
static uint64_t align_up_u(uint64_t x, uint64_t a)
{
    return (x + a - 1) / a * a;
}
static float tload(const ClipT* t, uint64_t i)
{
    if (!t || !t->p) return 0.f;
    if (t->dtype == 0) return ((const float*)t->p)[i];
    return f16_to_f32(((const uint16_t*)t->p)[i]);
}

#ifdef __AVX2__
static float hsum8(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
}
static void f16row_to_f32(float* d, const uint16_t* s, uint32_t n)
{
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i h = _mm_loadu_si128((const __m128i*)(s + i));
        _mm256_storeu_ps(d + i, _mm256_cvtph_ps(h));
    }
    for (; i < n; i++) d[i] = f16_to_f32(s[i]);
}
static float dot_f32(const float* a, const float* b, uint32_t n)
{
    __m256 s = _mm256_setzero_ps();
    uint32_t i = 0;
    for (; i + 16 <= n; i += 16) {
        s = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s);
        s = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s);
    }
    for (; i + 8 <= n; i += 8)
        s = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s);
    {
        float acc = hsum8(s);
        for (; i < n; i++) acc += a[i] * b[i];
        return acc;
    }
}
#else
static void f16row_to_f32(float* d, const uint16_t* s, uint32_t n)
{
    uint32_t i;
    for (i = 0; i < n; i++) d[i] = f16_to_f32(s[i]);
}
static float dot_f32(const float* a, const float* b, uint32_t n)
{
    float acc = 0.f;
    uint32_t i;
    for (i = 0; i < n; i++) acc += a[i] * b[i];
    return acc;
}
#endif

static void add_bias_rows(float* y, const float* b, uint32_t M, uint32_t out)
{
    uint32_t m, i;
    if (!b) return;
    for (m = 0; m < M; m++) {
        float* row = y + (size_t)m * out;
        for (i = 0; i < out; i++) row[i] += b[i];
    }
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

static void gemm_lin(Q3v* vis, float* y, const float* x, const ClipT* w, const ClipT* bias,
                     uint32_t M, uint32_t out, uint32_t in)
{
    uint32_t oo;
    float* bf = NULL;
    if (!w || !w->p) { memset(y, 0, (size_t)M * out * 4); return; }
    if (bias && bias->p) {
        bf = (float*)vis_alloca((size_t)out * 4);
        for (oo = 0; oo < out; oo++) bf[oo] = tload(bias, oo);
    }
    if (w->dtype == 0) {
#pragma omp parallel for schedule(static)
        for (oo = 0; oo < out; oo++) {
            const float* wr = (const float*)w->p + (size_t)oo * in;
            uint32_t m;
            for (m = 0; m < M; m++) {
                float acc = dot_f32(x + (size_t)m * in, wr, in);
                if (bf) acc += bf[oo];
                y[(size_t)m * out + oo] = acc;
            }
        }
        return;
    }
    if (!vis->gemm_row || in > vis->gemm_in_cap) {
        uint32_t m;
        for (m = 0; m < M; m++) {
            matmul(y + (size_t)m * out, x + (size_t)m * in, w->p, out, in, DT_F16);
            if (bf) {
                uint32_t i;
                for (i = 0; i < out; i++) y[(size_t)m * out + i] += bf[i];
            }
        }
        return;
    }
#pragma omp parallel for schedule(static)
    for (oo = 0; oo < out; oo++) {
        int tid = 0;
        float* wr;
        uint32_t m;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        if (tid < 0 || (uint32_t)tid >= vis->gemm_nthr) tid = 0;
        wr = vis->gemm_row + (size_t)tid * vis->gemm_in_cap;
        f16row_to_f32(wr, (const uint16_t*)w->p + (size_t)oo * in, in);
        for (m = 0; m < M; m++) {
            float acc = dot_f32(x + (size_t)m * in, wr, in);
            if (bf) acc += bf[oo];
            y[(size_t)m * out + oo] = acc;
        }
    }
}

static void gemm_f32(float* y, const float* x, const float* w, const float* bias,
                     uint32_t M, uint32_t out, uint32_t in)
{
    matmul_batch(y, x, (const uint8_t*)w, out, in, DT_F32, M);
    add_bias_rows(y, bias, M, out);
}

static float gelu_quick(float x)
{
    return x / (1.0f + expf(-1.702f * x));
}
static float gelu_tanh_f(float x)
{
    float c = 0.7978845608f;
    float inner = c * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}
static void act_inplace(float* y, uint32_t n, int quick)
{
    uint32_t i;
    if (quick) for (i = 0; i < n; i++) y[i] = gelu_quick(y[i]);
    else for (i = 0; i < n; i++) y[i] = gelu_tanh_f(y[i]);
}

static uint64_t gguf_nbytes(uint32_t gtype, uint64_t ne)
{
    if (gtype == 0) return ne * 4ull;
    if (gtype == 1) return ne * 2ull;
    if (gtype == 8) return ((ne + 31ull) / 32ull) * 34ull; /* Q8_0 */
    return 0;
}

static float* q8_0_to_f32(const uint8_t* src, uint64_t nelem)
{
    float* out = (float*)ymalloc((size_t)nelem * 4);
    const uint8_t* p = src;
    uint64_t i = 0;
    if (!out) return NULL;
    while (i + 32 <= nelem) {
        float s = f16_to_f32((uint16_t)(p[0] | ((uint16_t)p[1] << 8)));
        const int8_t* q = (const int8_t*)(p + 2);
        uint32_t k;
        for (k = 0; k < 32; k++) out[i + k] = s * (float)q[k];
        p += 34;
        i += 32;
    }
    for (; i < nelem; i++) out[i] = 0.f;
    return out;
}

static ClipT* find_t(Q3v* v, const char* name)
{
    int i;
    for (i = 0; i < v->n_t; i++)
        if (strcmp(v->ts[i].name, name) == 0) return &v->ts[i];
    return NULL;
}
static int req_t(Q3v* v, ClipT* dst, const char* name)
{
    ClipT* t = find_t(v, name);
    if (!t) return -1;
    *dst = *t;
    return 0;
}
static int opt_t(Q3v* v, ClipT* dst, const char* name)
{
    ClipT* t = find_t(v, name);
    if (!t) { memset(dst, 0, sizeof(*dst)); return 0; }
    *dst = *t;
    return 1;
}

static void unpack_patch(const ClipT* tw, float* dst, uint32_t e, uint32_t K, uint32_t ps, int add)
{
    uint32_t oc, ic, ky, kx;
    for (oc = 0; oc < e; oc++)
        for (ic = 0; ic < 3; ic++)
            for (ky = 0; ky < ps; ky++)
                for (kx = 0; kx < ps; kx++) {
                    uint64_t wi = (uint64_t)kx + (uint64_t)ps * (ky + ps * (ic + 3 * oc));
                    uint32_t k = ic * ps * ps + ky * ps + kx;
                    float val = tload(tw, wi);
                    if (add) dst[(size_t)oc * K + k] += val;
                    else dst[(size_t)oc * K + k] = val;
                }
}

static void spatial_merge(const float* src, float* dst, uint32_t gh, uint32_t gw, uint32_t e)
{
    uint32_t ty, tx, dy, dx, c, k = 0;
    uint32_t th = gh / 2, tw = gw / 2;
    for (ty = 0; ty < th; ty++)
        for (tx = 0; tx < tw; tx++)
            for (dy = 0; dy < 2; dy++)
                for (dx = 0; dx < 2; dx++) {
                    const float* s = src + ((size_t)(2 * ty + dy) * gw + (2 * tx + dx)) * e;
                    float* d = dst + (size_t)k * e;
                    for (c = 0; c < e; c++) d[c] = s[c];
                    k++;
                }
}

/* VISION M-RoPE: n_dims=dh/2, sections=dh/4, NEOX pairs over full head */
static void vision_rope(float* x, const int32_t* pos, uint32_t n, uint32_t nh, uint32_t dh)
{
    uint32_t t, nd = dh / 2, sec = dh / 4;
    float theta_scale = powf(10000.f, -2.f / (float)nd);
#pragma omp parallel for schedule(static)
    for (t = 0; t < n; t++) {
        float cache[256];
        float pt = (float)pos[t], ph = (float)pos[n + t];
        float tt = pt, tht = ph;
        uint32_t i0, h, sector;
        if (dh > 256) continue;
        for (i0 = 0; i0 < dh; i0 += 2) {
            sector = i0 / 2;
            if (sector == 0) tt = pt;
            else if (sector == sec) tht = ph;
            {
                float theta = (sector < sec) ? tt : tht;
                cache[i0] = cosf(theta);
                cache[i0 + 1] = sinf(theta);
            }
            tt *= theta_scale;
            tht *= theta_scale;
        }
        for (h = 0; h < nh; h++) {
            float* row = x + ((size_t)t * nh + h) * dh;
            for (i0 = 0; i0 < dh; i0 += 2) {
                uint32_t ic = i0 / 2;
                float c = cache[i0], s = cache[i0 + 1];
                float x0 = row[ic], x1 = row[ic + nd];
                row[ic] = x0 * c - x1 * s;
                row[ic + nd] = x0 * s + x1 * c;
            }
        }
    }
}

static void pack_heads(float* dst, const float* src, uint32_t n, uint32_t nh, uint32_t dh)
{
    uint32_t h;
#pragma omp parallel for schedule(static)
    for (h = 0; h < nh; h++) {
        uint32_t j;
        for (j = 0; j < n; j++)
            memcpy(dst + ((size_t)h * n + j) * dh,
                   src + ((size_t)j * nh + h) * dh, (size_t)dh * 4);
    }
}

static void attn_full(Q3v* vis, float* out, const float* q, const float* kpk, const float* vpk,
                     uint32_t n, uint32_t n_head, uint32_t dh, float scale)
{
    uint32_t t;
#pragma omp parallel for schedule(static)
    for (t = 0; t < n; t++) {
        uint32_t h, j;
        int tid = 0;
        float* sc;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        if (tid < 0 || (uint32_t)tid >= vis->sc_nthr) tid = 0;
        sc = vis->sc + (size_t)tid * vis->npos_max;
        for (h = 0; h < n_head; h++) {
            const float* qt = q + ((size_t)t * n_head + h) * dh;
            const float* kh = kpk + (size_t)h * n * dh;
            const float* vh = vpk + (size_t)h * n * dh;
            float mx = -1e30f, sum = 0.f;
            uint32_t d;
            for (j = 0; j < n; j++) {
                sc[j] = dot_f32(qt, kh + (size_t)j * dh, dh) * scale;
                if (sc[j] > mx) mx = sc[j];
            }
            for (j = 0; j < n; j++) {
                sc[j] = expf(sc[j] - mx);
                sum += sc[j];
            }
            {
                float inv = 1.f / (sum > 0.f ? sum : 1.f);
                float* o = out + ((size_t)t * n_head + h) * dh;
                memset(o, 0, (size_t)dh * 4);
                for (j = 0; j < n; j++) {
                    float a = sc[j] * inv;
                    const float* vj = vh + (size_t)j * dh;
#ifdef __AVX2__
                    __m256 as = _mm256_set1_ps(a);
                    for (d = 0; d + 8 <= dh; d += 8) {
                        __m256 ov = _mm256_loadu_ps(o + d);
                        ov = _mm256_fmadd_ps(as, _mm256_loadu_ps(vj + d), ov);
                        _mm256_storeu_ps(o + d, ov);
                    }
                    for (; d < dh; d++) o[d] += a * vj[d];
#else
                    for (d = 0; d < dh; d++) o[d] += a * vj[d];
#endif
                }
            }
        }
    }
}

static void split_qkv(Q3v* vis, uint32_t n)
{
    uint32_t e = vis->n_embd, t;
#pragma omp parallel for schedule(static)
    for (t = 0; t < n; t++) {
        const float* row = vis->qkv + (size_t)t * 3 * e;
        memcpy(vis->q + (size_t)t * e, row, (size_t)e * 4);
        memcpy(vis->k + (size_t)t * e, row + e, (size_t)e * 4);
        memcpy(vis->v + (size_t)t * e, row + 2 * e, (size_t)e * 4);
    }
}

static void vit_ffn(Q3v* vis, ClipLayer* L, uint32_t n, uint32_t in)
{
    uint32_t n_ff = (uint32_t)L->up_w.dims[1];
    if (n_ff == 0 || n_ff == vis->n_embd) n_ff = vis->n_ff;
    gemm_lin(vis, vis->ff, vis->tmp, &L->up_w, L->up_b.p ? &L->up_b : NULL, n, n_ff, in);
    if (L->has_gate) {
        uint32_t i;
        gemm_lin(vis, vis->qkv, vis->tmp, &L->gate_w, L->gate_b.p ? &L->gate_b : NULL, n, n_ff, in);
        for (i = 0; i < n * n_ff; i++) {
            float g = vis->qkv[i], u = vis->ff[i];
            if (vis->ffn_quick) vis->ff[i] = gelu_quick(g) * u;
            else vis->ff[i] = gelu_tanh_f(g) * u;
        }
    } else {
        act_inplace(vis->ff, n * n_ff, vis->ffn_quick);
    }
    gemm_lin(vis, vis->tmp, vis->ff, &L->down_w, L->down_b.p ? &L->down_b : NULL, n, vis->n_embd, n_ff);
}

static int load_layer(Q3v* v, uint32_t il)
{
    ClipLayer* L = &v->layers[il];
    char n[96];
    memset(L, 0, sizeof(*L));
#define LREQ(field, fmt) do { snprintf(n, sizeof(n), fmt, il); if (req_t(v, &L->field, n) != 0) return -1; } while (0)
#define LOPT(field, fmt) do { snprintf(n, sizeof(n), fmt, il); opt_t(v, &L->field, n); } while (0)
    LREQ(ln1_w, "v.blk.%u.ln1.weight");
    LOPT(ln1_b, "v.blk.%u.ln1.bias");
    LREQ(ln2_w, "v.blk.%u.ln2.weight");
    LOPT(ln2_b, "v.blk.%u.ln2.bias");
    LREQ(qkv_w, "v.blk.%u.attn_qkv.weight");
    LOPT(qkv_b, "v.blk.%u.attn_qkv.bias");
    LREQ(o_w, "v.blk.%u.attn_out.weight");
    LOPT(o_b, "v.blk.%u.attn_out.bias");
    LREQ(up_w, "v.blk.%u.ffn_up.weight");
    LOPT(up_b, "v.blk.%u.ffn_up.bias");
    LREQ(down_w, "v.blk.%u.ffn_down.weight");
    LOPT(down_b, "v.blk.%u.ffn_down.bias");
#undef LREQ
    snprintf(n, sizeof(n), "v.blk.%u.ffn_gate.weight", il);
    L->has_gate = opt_t(v, &L->gate_w, n);
    snprintf(n, sizeof(n), "v.blk.%u.ffn_gate.bias", il);
    opt_t(v, &L->gate_b, n);
    snprintf(n, sizeof(n), "v.deepstack.%u.norm.weight", il);
    if (opt_t(v, &L->ds_ln_w, n)) {
        snprintf(n, sizeof(n), "v.deepstack.%u.norm.bias", il);
        opt_t(v, &L->ds_ln_b, n);
        snprintf(n, sizeof(n), "v.deepstack.%u.fc1.weight", il);
        if (opt_t(v, &L->ds_fc1_w, n)) {
            snprintf(n, sizeof(n), "v.deepstack.%u.fc1.bias", il);
            opt_t(v, &L->ds_fc1_b, n);
            snprintf(n, sizeof(n), "v.deepstack.%u.fc2.weight", il);
            if (opt_t(v, &L->ds_fc2_w, n)) {
                snprintf(n, sizeof(n), "v.deepstack.%u.fc2.bias", il);
                opt_t(v, &L->ds_fc2_b, n);
                L->has_ds = 1;
                if (v->n_ds < Q3V_MAX_DS) v->ds_il[v->n_ds++] = il;
            }
        }
    }
    return 0;
}

void q3v_free(Q3v* v);

static void interp_pos(Q3v* vis, float* dst, uint32_t gh, uint32_t gw)
{
    uint32_t e = vis->n_embd, y, x, c;
    uint32_t ns = vis->pos_side;
    if (ns == gh && ns == gw) {
        memcpy(dst, vis->pos_f, (size_t)gh * gw * e * 4);
        return;
    }
    for (y = 0; y < gh; y++) {
        float fy = (gh <= 1) ? 0.f : (float)y * (float)(ns - 1) / (float)(gh - 1);
        int y0 = (int)floorf(fy), y1 = y0 + 1;
        float wy = fy - (float)y0;
        if (y1 >= (int)ns) y1 = (int)ns - 1;
        for (x = 0; x < gw; x++) {
            float fx = (gw <= 1) ? 0.f : (float)x * (float)(ns - 1) / (float)(gw - 1);
            int x0 = (int)floorf(fx), x1 = x0 + 1;
            float wx = fx - (float)x0;
            float* d = dst + ((size_t)y * gw + x) * e;
            if (x1 >= (int)ns) x1 = (int)ns - 1;
            for (c = 0; c < e; c++) {
                float p00 = vis->pos_f[((size_t)y0 * ns + (uint32_t)x0) * e + c];
                float p10 = vis->pos_f[((size_t)y0 * ns + (uint32_t)x1) * e + c];
                float p01 = vis->pos_f[((size_t)y1 * ns + (uint32_t)x0) * e + c];
                float p11 = vis->pos_f[((size_t)y1 * ns + (uint32_t)x1) * e + c];
                d[c] = (p00 * (1 - wx) + p10 * wx) * (1 - wy) + (p01 * (1 - wx) + p11 * wx) * wy;
            }
        }
    }
}

static void patch_embed(Q3v* vis, const float* chw, uint32_t gh, uint32_t gw, uint32_t sw, uint32_t sh)
{
    uint32_t ps = vis->patch, e = vis->n_embd, K = 3 * ps * ps;
    uint32_t n = gh * gw, t;
    float* col = vis->ff;
#pragma omp parallel for schedule(static)
    for (t = 0; t < n; t++) {
        uint32_t py = t / gw, px = t % gw, ic, ky, kx, k = 0;
        float* c = col + (size_t)t * K;
        for (ic = 0; ic < 3; ic++)
            for (ky = 0; ky < ps; ky++)
                for (kx = 0; kx < ps; kx++) {
                    uint32_t ix = px * ps + kx, iy = py * ps + ky;
                    c[k++] = chw[(size_t)ic * sh * sw + iy * sw + ix];
                }
    }
    gemm_f32(vis->x, col, vis->patch_wf, NULL, n, e, K);
}

static void smart_hw(int w, int h, int factor, int min_px, int max_px, int* ow, int* oh)
{
    int wb, hb;
    if (w < 1) w = 1;
    if (h < 1) h = 1;
    if (factor < 1) factor = 1;
    wb = ((w + factor / 2) / factor) * factor;
    hb = ((h + factor / 2) / factor) * factor;
    if (wb < factor) wb = factor;
    if (hb < factor) hb = factor;
    if (max_px > 0 && (int64_t)hb * wb > max_px) {
        float beta = sqrtf((float)h * (float)w / (float)max_px);
        hb = (int)(floorf((float)h / beta / (float)factor) * (float)factor);
        wb = (int)(floorf((float)w / beta / (float)factor) * (float)factor);
        if (hb < factor) hb = factor;
        if (wb < factor) wb = factor;
    } else if (min_px > 0 && (int64_t)hb * wb < min_px) {
        float beta = sqrtf((float)min_px / ((float)h * (float)w));
        hb = (int)(ceilf((float)h * beta / (float)factor) * (float)factor);
        wb = (int)(ceilf((float)w * beta / (float)factor) * (float)factor);
    }
    *ow = wb;
    *oh = hb;
}

Q3v* q3v_load(const char* path, char* err, size_t errlen)
{
    Q3v* v = (Q3v*)ycalloc(1, sizeof(*v));
    const uint8_t* data;
    uint64_t fsize, data_start, n_kv, n_tensors, i;
    uint32_t ver, alignment = 32;
    GB b;
    int use_gelu = 0, use_silu = 0;
    if (!v) { if (err) snprintf(err, errlen, "oom"); return NULL; }
    if (wmap_open(path, &v->map) != 0) {
        if (err) snprintf(err, errlen, "cannot mmap %s", path);
        free(v); return NULL;
    }
    data = (const uint8_t*)v->map.base;
    fsize = v->map.size;
    if (fsize < 24 || memcmp(data, "GGUF", 4) != 0) {
        if (err) snprintf(err, errlen, "not gguf");
        q3v_free(v); return NULL;
    }
    b.p = data + 4; b.end = data + fsize; b.err = 0;
    ver = gb_u32(&b);
    n_tensors = gb_u64(&b);
    n_kv = gb_u64(&b);
    v->image_size = 768; v->patch = 16; v->n_embd = 1024; v->n_ff = 4096;
    v->n_layer = 24; v->n_head = 16; v->n_out_embd = 2048; v->n_merge = 2;
    v->eps = 1e-6f; v->ffn_quick = 1;
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
        if ((typ == 4 || typ == 5) &&
            (!strcmp(key, "clip.vision.image_size") ||
             !strcmp(key, "clip.vision.patch_size") ||
             !strcmp(key, "clip.vision.embedding_length") ||
             !strcmp(key, "clip.vision.feed_forward_length") ||
             !strcmp(key, "clip.vision.block_count") ||
             !strcmp(key, "clip.vision.attention.head_count") ||
             !strcmp(key, "clip.vision.projection_dim") ||
             !strcmp(key, "clip.vision.spatial_merge_size") ||
             !strcmp(key, "general.alignment"))) {
            uint32_t u = gb_u32(&b);
            if (!strcmp(key, "clip.vision.image_size")) v->image_size = u;
            else if (!strcmp(key, "clip.vision.patch_size")) v->patch = u;
            else if (!strcmp(key, "clip.vision.embedding_length")) v->n_embd = u;
            else if (!strcmp(key, "clip.vision.feed_forward_length")) v->n_ff = u;
            else if (!strcmp(key, "clip.vision.block_count")) v->n_layer = u;
            else if (!strcmp(key, "clip.vision.attention.head_count")) v->n_head = u;
            else if (!strcmp(key, "clip.vision.projection_dim")) v->n_out_embd = u;
            else if (!strcmp(key, "clip.vision.spatial_merge_size")) v->n_merge = u;
            else alignment = u;
        } else if (!strcmp(key, "clip.vision.attention.layer_norm_epsilon") && typ == 6) {
            uint32_t u = gb_u32(&b); memcpy(&v->eps, &u, 4);
        } else if ((!strcmp(key, "clip.use_gelu") || !strcmp(key, "clip.use_silu")) &&
                   (typ == 0 || typ == 1 || typ == 7)) {
            uint8_t u = 0;
            if (b.p < b.end) { u = *b.p; gb_skip(&b, 1); }
            if (strstr(key, "gelu")) use_gelu = u;
            else use_silu = u;
        } else if ((!strcmp(key, "clip.vision.image_mean") || !strcmp(key, "clip.vision.image_std")) && typ == 9) {
            uint32_t at = gb_u32(&b); uint64_t nn = gb_u64(&b); uint64_t j;
            float* dst = strstr(key, "mean") ? v->mean : v->std;
            for (j = 0; j < nn && !b.err; j++) {
                if (at == 6) {
                    uint32_t u = gb_u32(&b); float f; memcpy(&f, &u, 4);
                    if (j < 3) dst[j] = f;
                } else skip_val(&b, at);
            }
        } else skip_val(&b, typ);
    }
    if (b.err) { if (err) snprintf(err, errlen, "bad mmproj kv"); q3v_free(v); return NULL; }
    (void)use_silu;
    if (use_gelu) v->ffn_quick = 0;
    v->ts = (ClipT*)ycalloc((size_t)n_tensors, sizeof(ClipT));
    if (!v->ts) { q3v_free(v); if (err) snprintf(err, errlen, "oom"); return NULL; }
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
        t->gtype = gtype;
        t->owned = 0;
        t->dtype = (gtype == 0) ? 0 : 1;
        t->p = NULL;
        v->n_t++;
    }
    if (b.err) { if (err) snprintf(err, errlen, "bad mmproj tensors"); q3v_free(v); return NULL; }
    data_start = align_up_u((uint64_t)(b.p - data), alignment ? alignment : 32);
    for (i = 0; i < (uint64_t)v->n_t; i++) {
        uint64_t ne = 1, d, nb;
        uint32_t gt = v->ts[i].gtype;
        const uint8_t* src;
        for (d = 0; d < v->ts[i].ndim; d++) ne *= v->ts[i].dims[d];
        nb = gguf_nbytes(gt, ne);
        src = data + data_start + v->ts[i].off;
        if (!nb || src < data || src + nb > data + fsize) {
            if (err) snprintf(err, errlen, "tensor %s type=%u out of file", v->ts[i].name, gt);
            q3v_free(v); return NULL;
        }
        if (gt == 8) {
            float* f = q8_0_to_f32(src, ne);
            if (!f) { if (err) snprintf(err, errlen, "oom q8 dequant %s", v->ts[i].name); q3v_free(v); return NULL; }
            v->ts[i].p = (const uint8_t*)f;
            v->ts[i].dtype = 0;
            v->ts[i].owned = 1;
        } else if (gt == 1) {
            float* f = (float*)ymalloc((size_t)ne * 4);
            uint64_t k;
            if (!f) { if (err) snprintf(err, errlen, "oom f16 dequant %s", v->ts[i].name); q3v_free(v); return NULL; }
            for (k = 0; k < ne; k++) f[k] = f16_to_f32(((const uint16_t*)src)[k]);
            v->ts[i].p = (const uint8_t*)f;
            v->ts[i].dtype = 0;
            v->ts[i].owned = 1;
        } else if (gt == 0) {
            v->ts[i].p = src;
            v->ts[i].dtype = 0;
        } else {
            if (err) snprintf(err, errlen, "unsupported mmproj dtype %u (%s)", gt, v->ts[i].name);
            q3v_free(v); return NULL;
        }
    }
    if (req_t(v, &v->patch_w, "v.patch_embd.weight") != 0 ||
        req_t(v, &v->patch_b, "v.patch_embd.bias") != 0 ||
        req_t(v, &v->pos_embd, "v.position_embd.weight") != 0 ||
        req_t(v, &v->mm0_w, "mm.0.weight") != 0 ||
        req_t(v, &v->mm0_b, "mm.0.bias") != 0 ||
        req_t(v, &v->mm2_w, "mm.2.weight") != 0 ||
        req_t(v, &v->mm2_b, "mm.2.bias") != 0) {
        if (err) snprintf(err, errlen, "qwen3vl mmproj missing tensors");
        q3v_free(v); return NULL;
    }
    v->has_patch1 = opt_t(v, &v->patch_w1, "v.patch_embd.weight.1");
    v->has_pre_ln = opt_t(v, &v->pre_ln_w, "v.pre_ln.weight");
    opt_t(v, &v->pre_ln_b, "v.pre_ln.bias");
    v->has_post_ln = opt_t(v, &v->post_ln_w, "v.post_ln.weight");
    opt_t(v, &v->post_ln_b, "v.post_ln.bias");
    if (v->mm2_b.p && v->mm2_b.dims[0]) v->n_out_embd = (uint32_t)v->mm2_b.dims[0];
    if (v->n_layer > Q3V_MAX_LAYERS) { if (err) snprintf(err, errlen, "too many vit layers"); q3v_free(v); return NULL; }
    v->n_ds = 0;
    for (i = 0; i < v->n_layer; i++)
        if (load_layer(v, (uint32_t)i) != 0) {
            if (err) snprintf(err, errlen, "missing vit layer %u", (unsigned)i);
            q3v_free(v); return NULL;
        }
    {
        uint32_t np = (v->image_size / v->patch) * (v->image_size / v->patch);
        uint32_t ps = v->patch, e = v->n_embd, K = 3 * ps * ps, oc;
        size_t cap = (size_t)np * e;
        size_t ffcap = (size_t)np * K;
        uint32_t mm_ff = (uint32_t)v->mm0_w.dims[1];
        if (mm_ff < v->n_ff) mm_ff = v->n_ff;
        if (ffcap < (size_t)np * 3 * e) ffcap = (size_t)np * 3 * e;
        if (ffcap < (size_t)np * v->n_ff) ffcap = (size_t)np * v->n_ff;
        if (ffcap < (size_t)(np / 4) * (e * 4 + mm_ff)) ffcap = (size_t)(np / 4) * (e * 4 + mm_ff);
        v->npos_max = np;
        v->x = (float*)ymalloc(cap * 4);
        v->res = (float*)ymalloc(cap * 4);
        v->tmp = (float*)ymalloc(cap * 4);
        v->q = (float*)ymalloc(cap * 4);
        v->k = (float*)ymalloc(cap * 4);
        v->v = (float*)ymalloc(cap * 4);
        v->attn = (float*)ymalloc(cap * 4);
        v->qkv = (float*)ymalloc((size_t)np * (3 * e > v->n_ff ? 3 * e : v->n_ff) * 4);
        v->ff = (float*)ymalloc(ffcap * 4);
        v->patch_wf = (float*)ymalloc((size_t)e * K * 4);
        v->pos_n = (uint32_t)v->pos_embd.dims[1];
        if (!v->pos_n) v->pos_n = np;
        v->pos_side = (uint32_t)(sqrt((double)v->pos_n) + 0.5);
        v->pos_f = (float*)ymalloc((size_t)v->pos_n * e * 4);
        v->gemm_nthr = 1;
#ifdef _OPENMP
        v->gemm_nthr = (uint32_t)omp_get_max_threads();
        if (v->gemm_nthr < 1) v->gemm_nthr = 1;
#endif
        v->sc_nthr = v->gemm_nthr;
        v->sc = (float*)ymalloc((size_t)v->sc_nthr * np * 4);
        v->gemm_in_cap = e * 4;
        if (v->gemm_in_cap < v->n_ff) v->gemm_in_cap = v->n_ff;
        if (v->gemm_in_cap < 3 * e) v->gemm_in_cap = 3 * e;
        if (v->gemm_in_cap < mm_ff) v->gemm_in_cap = mm_ff;
        v->gemm_row = (float*)ymalloc((size_t)v->gemm_nthr * v->gemm_in_cap * 4);
        if (v->patch_wf) {
            unpack_patch(&v->patch_w, v->patch_wf, e, K, ps, 0);
            if (v->has_patch1) unpack_patch(&v->patch_w1, v->patch_wf, e, K, ps, 1);
        }
        if (v->pos_f) {
            uint32_t pi;
            for (pi = 0; pi < v->pos_n * e; pi++)
                v->pos_f[pi] = tload(&v->pos_embd, pi);
        }
        (void)oc;
        if (!v->x || !v->res || !v->tmp || !v->q || !v->k || !v->v || !v->attn || !v->ff ||
            !v->qkv || !v->patch_wf || !v->pos_f || !v->gemm_row || !v->sc) {
            if (err) snprintf(err, errlen, "oom vis buf");
            q3v_free(v); return NULL;
        }
    }
    ylog_info("vision qwen3vl: %ux%u patch=%u layers=%u embd=%u out=%u ds=%d merge=%u",
              v->image_size, v->image_size, v->patch, v->n_layer, v->n_embd, v->n_out_embd,
              v->n_ds, v->n_merge);
    return v;
}

void q3v_free(Q3v* v)
{
    int i;
    if (!v) return;
    if (v->ts) {
        for (i = 0; i < v->n_t; i++)
            if (v->ts[i].owned) free((void*)v->ts[i].p);
    }
    free(v->x); free(v->res); free(v->tmp); free(v->q); free(v->k); free(v->v);
    free(v->attn); free(v->ff); free(v->qkv); free(v->sc);
    free(v->patch_wf); free(v->pos_f); free(v->gemm_row);
    free(v->ts);
    wmap_close(&v->map);
    free(v);
}

int q3v_n_tokens(const Q3v* v)
{
    uint32_t g;
    if (!v) return 0;
    g = v->image_size / v->patch / 2;
    return (int)(g * g);
}

int q3v_hidden(const Q3v* v)
{
    return v ? (int)v->n_out_embd : 0;
}

int q3v_n_deepstack(const Q3v* v)
{
    return v ? v->n_ds : 0;
}

static int encode_run(Q3v* vis, const float* chw, uint32_t sw, uint32_t sh, float* out, float* ds)
{
    uint32_t gh = sh / vis->patch, gw = sw / vis->patch;
    uint32_t n = gh * gw, e = vis->n_embd, il, t;
    uint32_t n4 = n / 4, nh = vis->n_head, dh = e / nh;
    float scale = 1.f / sqrtf((float)dh);
    int32_t* pos;
    uint32_t ptr, y, x, dy, dx;
    int dsi = 0;
    uint32_t oc;
    float* pbias;

    if (n == 0 || n > vis->npos_max || (n % 4) != 0) return -1;
    patch_embed(vis, chw, gh, gw, sw, sh);
    spatial_merge(vis->x, vis->res, gh, gw, e);
    memcpy(vis->x, vis->res, (size_t)n * e * 4);
    pbias = (float*)vis_alloca((size_t)e * 4);
    for (oc = 0; oc < e; oc++) pbias[oc] = tload(&vis->patch_b, oc);
    for (t = 0; t < n; t++)
        add_inplace(vis->x + (size_t)t * e, pbias, e);
    interp_pos(vis, vis->tmp, gh, gw);
    spatial_merge(vis->tmp, vis->res, gh, gw, e);
    add_inplace(vis->x, vis->res, n * e);

    if (vis->has_pre_ln) {
#pragma omp parallel for schedule(static)
        for (t = 0; t < n; t++)
            layernorm(vis->tmp + (size_t)t * e, vis->x + (size_t)t * e,
                      &vis->pre_ln_w, vis->pre_ln_b.p ? &vis->pre_ln_b : NULL, e, vis->eps);
        memcpy(vis->x, vis->tmp, (size_t)n * e * 4);
    }

    pos = (int32_t*)ymalloc((size_t)n * 4 * 4);
    if (!pos) return -1;
    ptr = 0;
    for (y = 0; y < gh; y += 2)
        for (x = 0; x < gw; x += 2)
            for (dy = 0; dy < 2; dy++)
                for (dx = 0; dx < 2; dx++) {
                    pos[ptr] = (int32_t)(y + dy);
                    pos[n + ptr] = (int32_t)(x + dx);
                    pos[2 * n + ptr] = (int32_t)(y + dy);
                    pos[3 * n + ptr] = (int32_t)(x + dx);
                    ptr++;
                }

    for (il = 0; il < vis->n_layer; il++) {
        ClipLayer* L = &vis->layers[il];
        memcpy(vis->res, vis->x, (size_t)n * e * 4);
#pragma omp parallel for schedule(static)
        for (t = 0; t < n; t++)
            layernorm(vis->tmp + (size_t)t * e, vis->x + (size_t)t * e,
                      &L->ln1_w, L->ln1_b.p ? &L->ln1_b : NULL, e, vis->eps);
        gemm_lin(vis, vis->qkv, vis->tmp, &L->qkv_w, L->qkv_b.p ? &L->qkv_b : NULL, n, 3 * e, e);
        split_qkv(vis, n);
        vision_rope(vis->q, pos, n, nh, dh);
        vision_rope(vis->k, pos, n, nh, dh);
        pack_heads(vis->ff, vis->k, n, nh, dh);
        pack_heads(vis->qkv, vis->v, n, nh, dh);
        attn_full(vis, vis->attn, vis->q, vis->ff, vis->qkv, n, nh, dh, scale);
        gemm_lin(vis, vis->tmp, vis->attn, &L->o_w, L->o_b.p ? &L->o_b : NULL, n, e, e);
        memcpy(vis->x, vis->res, (size_t)n * e * 4);
        add_inplace(vis->x, vis->tmp, n * e);
        memcpy(vis->res, vis->x, (size_t)n * e * 4);
#pragma omp parallel for schedule(static)
        for (t = 0; t < n; t++)
            layernorm(vis->tmp + (size_t)t * e, vis->x + (size_t)t * e,
                      &L->ln2_w, L->ln2_b.p ? &L->ln2_b : NULL, e, vis->eps);
        vit_ffn(vis, L, n, e);
        memcpy(vis->x, vis->res, (size_t)n * e * 4);
        add_inplace(vis->x, vis->tmp, n * e);

        if (L->has_ds) {
            uint32_t in4 = e * 4, n_ff, i;
#pragma omp parallel for schedule(static)
            for (t = 0; t < n4; t++)
                layernorm(vis->tmp + (size_t)t * in4, vis->x + (size_t)t * in4,
                          &L->ds_ln_w, L->ds_ln_b.p ? &L->ds_ln_b : NULL, in4, vis->eps);
            n_ff = (uint32_t)L->ds_fc1_w.dims[1];
            gemm_lin(vis, vis->ff, vis->tmp, &L->ds_fc1_w, L->ds_fc1_b.p ? &L->ds_fc1_b : NULL, n4, n_ff, in4);
            for (i = 0; i < n4 * n_ff; i++) vis->ff[i] = gelu_tanh_f(vis->ff[i]);
            gemm_lin(vis, vis->tmp, vis->ff, &L->ds_fc2_w, L->ds_fc2_b.p ? &L->ds_fc2_b : NULL,
                     n4, vis->n_out_embd, n_ff);
            if (ds && dsi < vis->n_ds)
                memcpy(ds + (size_t)dsi * n4 * vis->n_out_embd, vis->tmp,
                       (size_t)n4 * vis->n_out_embd * 4);
            dsi++;
        }
    }
    free(pos);

    if (vis->has_post_ln) {
#pragma omp parallel for schedule(static)
        for (t = 0; t < n; t++)
            layernorm(vis->tmp + (size_t)t * e, vis->x + (size_t)t * e,
                      &vis->post_ln_w, vis->post_ln_b.p ? &vis->post_ln_b : NULL, e, vis->eps);
        memcpy(vis->x, vis->tmp, (size_t)n * e * 4);
    }
    {
        uint32_t in4 = e * 4, n_ff = (uint32_t)vis->mm0_w.dims[1], i;
        gemm_lin(vis, vis->ff, vis->x, &vis->mm0_w, vis->mm0_b.p ? &vis->mm0_b : NULL, n4, n_ff, in4);
        for (i = 0; i < n4 * n_ff; i++) vis->ff[i] = gelu_tanh_f(vis->ff[i]);
        gemm_lin(vis, out, vis->ff, &vis->mm2_w, vis->mm2_b.p ? &vis->mm2_b : NULL,
                 n4, vis->n_out_embd, n_ff);
    }
    return (int)n4;
}

int q3v_encode(Q3v* v, const char* image_path, float* out, float* ds, int max_tok,
               char* err, size_t errlen)
{
    int w = 0, h = 0, c = 0, x, y, ic, ntok, sw, sh;
    unsigned char* img;
    float* chw;
    int factor, min_px, max_px;
    if (!v || !image_path || !out) { if (err) snprintf(err, errlen, "bad vision args"); return -1; }
    img = stbi_load(image_path, &w, &h, &c, 3);
    if (!img) { if (err) snprintf(err, errlen, "cannot load image %s", image_path); return -1; }
    factor = (int)(v->patch * (v->n_merge ? v->n_merge : 2));
    if (factor < 2) factor = (int)(v->patch * 2);
    max_px = (int)(v->image_size * v->image_size);
    min_px = factor * factor; /* 至少 1 个 merge tile, 小图不拉到 image_size */
    smart_hw(w, h, factor, min_px, max_px, &sw, &sh);
    ntok = (sw / (int)v->patch / 2) * (sh / (int)v->patch / 2);
    if (ntok < 1) { stbi_image_free(img); if (err) snprintf(err, errlen, "image too small"); return -1; }
    if (max_tok < ntok) {
        stbi_image_free(img);
        if (err) snprintf(err, errlen, "out tokens %d < %d", max_tok, ntok);
        return -1;
    }
    chw = (float*)ymalloc((size_t)3 * (size_t)sh * (size_t)sw * 4);
    if (!chw) { stbi_image_free(img); if (err) snprintf(err, errlen, "oom"); return -1; }
    for (y = 0; y < sh; y++) {
        float fy = ((float)y + 0.5f) * (float)h / (float)sh - 0.5f;
        int y0 = (int)floorf(fy), y1 = y0 + 1;
        float wy = fy - (float)y0;
        if (y0 < 0) { y0 = 0; wy = 0; }
        if (y1 >= h) y1 = h - 1;
        if (y0 >= h) y0 = h - 1;
        for (x = 0; x < sw; x++) {
            float fx = ((float)x + 0.5f) * (float)w / (float)sw - 0.5f;
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
                chw[(size_t)ic * sh * sw + (uint32_t)y * sw + (uint32_t)x] = (p / 255.f - v->mean[ic]) / v->std[ic];
            }
        }
    }
    stbi_image_free(img);
    ylog_info("vision qwen3vl: encoding %dx%d (src %dx%d)", sw, sh, w, h);
    {
        uint64_t t0 = ynow_ms();
        if (encode_run(v, chw, (uint32_t)sw, (uint32_t)sh, out, ds) < 0) {
            free(chw);
            if (err) snprintf(err, errlen, "encode failed");
            return -1;
        }
        ylog_info("vision qwen3vl: %dx%d -> %d tokens hidden=%u ds=%d in %.2f s",
                  w, h, ntok, v->n_out_embd, v->n_ds, (double)(ynow_ms() - t0) / 1000.0);
    }
    free(chw);
    return ntok;
}
