/* Gemma 4 vision (E2B / E4B 同一 projector_type=gemma4v, 对齐 llama.cpp clip_graph_gemma4v) */
#include "vision_impl.h"
#include "yllm.h"
#include "matvec.h"
#include "log.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef __AVX2__
#include <immintrin.h>
#endif
#if defined(__aarch64__)
#include <arm_neon.h>
#endif
#include "stb_image.h"

#define G4V_MAX_LAYERS 48

typedef struct {
    const uint8_t* p;
    uint64_t off;
    uint32_t dtype;
    uint32_t ndim;
    uint64_t dims[4];
    char name[128];
    float imin, imax, omin, omax;
} ClipT;

typedef struct {
    ClipT ln1_w, ln2_w, q_w, k_w, v_w, o_w;
    ClipT q_b, k_b, v_b, o_b;
    ClipT q_norm, k_norm;
    ClipT up_w, gate_w, down_w;
    ClipT up_b, gate_b, down_b;
    ClipT ls1, ls2, ls_out, attn_post, ff_post;
    int has_gate, has_q_norm, has_k_norm;
} ClipLayer;

struct G4v {
    WMap map;
    ClipT* ts;
    int n_t;
    uint32_t image_size, patch, n_embd, n_ff, n_layer, n_head, n_out_embd, n_merge;
    uint32_t ntok_min, ntok_max, npos_max;
    float mean[3], std[3], eps, rope_theta;
    int ffn_op; /* 0 quick 1 gelu 2 silu */
    ClipT patch_w, pos_embd, pre_ln_w, mm_proj;
    ClipT std_bias, std_scale;
    int has_pre_ln, has_std;
    ClipLayer layers[G4V_MAX_LAYERS];
    float *x, *res, *tmp, *q, *k, *v, *attn, *ff, *ffg, *sc;
    float *patch_wf, *pos_x, *pos_y, *gemm_row;
    uint32_t pos_n, gemm_nthr, gemm_in_cap;
};

void g4v_free(G4v* v);

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
static ClipT* find_t(G4v* v, const char* name)
{
    int i;
    for (i = 0; i < v->n_t; i++)
        if (strcmp(v->ts[i].name, name) == 0) return &v->ts[i];
    return NULL;
}
static int req_t(G4v* v, ClipT* dst, const char* name)
{
    ClipT* t = find_t(v, name);
    if (!t) return -1;
    *dst = *t;
    return 0;
}
static int opt_t(G4v* v, ClipT* dst, const char* name)
{
    ClipT* t = find_t(v, name);
    if (!t) { memset(dst, 0, sizeof(*dst)); return 0; }
    *dst = *t;
    return 0;
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
static void f16row_to_f32(float* d, const uint16_t* s, uint32_t n)
{
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i h = _mm_loadu_si128((const __m128i*)(s + i));
        _mm256_storeu_ps(d + i, _mm256_cvtph_ps(h));
    }
    for (; i < n; i++) d[i] = f16_to_f32(s[i]);
}
#elif defined(__aarch64__)
static float hsum4(float32x4_t v) { return vaddvq_f32(v); }
static float dot_f32(const float* a, const float* b, uint32_t n)
{
    float32x4_t s0 = vdupq_n_f32(0.f), s1 = vdupq_n_f32(0.f);
    uint32_t i = 0;
    for (; i + 16 <= n; i += 16) {
        s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
        s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        s0 = vfmaq_f32(s0, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
        s1 = vfmaq_f32(s1, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    for (; i + 4 <= n; i += 4)
        s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
    {
        float acc = hsum4(vaddq_f32(s0, s1));
        for (; i < n; i++) acc += a[i] * b[i];
        return acc;
    }
}
static void f16row_to_f32(float* d, const uint16_t* s, uint32_t n)
{
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float16x8_t h = vld1q_f16((const __fp16*)(s + i));
        vst1q_f32(d + i, vcvt_f32_f16(vget_low_f16(h)));
        vst1q_f32(d + i + 4, vcvt_f32_f16(vget_high_f16(h)));
    }
    for (; i < n; i++) d[i] = f16_to_f32(s[i]);
#else
    uint32_t i;
    for (i = 0; i < n; i++) d[i] = f16_to_f32(s[i]);
#endif
}
#else
static float dot_f32(const float* a, const float* b, uint32_t n)
{
    float acc = 0.f;
    uint32_t i;
    for (i = 0; i < n; i++) acc += a[i] * b[i];
    return acc;
}
static void f16row_to_f32(float* d, const uint16_t* s, uint32_t n)
{
    uint32_t i;
    for (i = 0; i < n; i++) d[i] = f16_to_f32(s[i]);
}
#endif

static void clamp_buf(float* x, uint32_t n, float lo, float hi)
{
    uint32_t i;
    if (lo <= -1e20f && hi >= 1e20f) return;
    for (i = 0; i < n; i++) {
        float v = x[i];
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        x[i] = v;
    }
}

static void gemm_lin(G4v* vis, float* y, const float* x, const ClipT* w, const ClipT* bias,
                     uint32_t M, uint32_t out, uint32_t in)
{
    uint32_t oo;
    float *xc = NULL;
    /* 只按 out 维并行, 内层串行扫 M. 禁止再调 matmul(): 其内部也有 omp, MinGW 嵌套会 SIGSEGV */
    if (!w || !w->p) { memset(y, 0, (size_t)M * out * 4); return; }
    if (w->imin > -1e20f || w->imax < 1e20f) {
        xc = (float*)ymalloc((size_t)M * in * 4);
        if (!xc) return;
        memcpy(xc, x, (size_t)M * in * 4);
        clamp_buf(xc, M * in, w->imin, w->imax);
        x = xc;
    }
    if (w->dtype == 0) {
#pragma omp parallel for schedule(static)
        for (oo = 0; oo < out; oo++) {
            const float* wr = (const float*)w->p + (size_t)oo * in;
            float b = (bias && bias->p) ? tload(bias, oo) : 0.f;
            uint32_t m;
            for (m = 0; m < M; m++)
                y[(size_t)m * out + oo] = dot_f32(x + (size_t)m * in, wr, in) + b;
        }
    } else {
#pragma omp parallel for schedule(static)
        for (oo = 0; oo < out; oo++) {
            int tid = 0;
            float* wr;
            float b;
            uint32_t m;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            if (tid < 0 || (uint32_t)tid >= vis->gemm_nthr) tid = 0;
            wr = vis->gemm_row + (size_t)tid * vis->gemm_in_cap;
            f16row_to_f32(wr, (const uint16_t*)w->p + (size_t)oo * in, in);
            b = (bias && bias->p) ? tload(bias, oo) : 0.f;
            for (m = 0; m < M; m++)
                y[(size_t)m * out + oo] = dot_f32(x + (size_t)m * in, wr, in) + b;
        }
    }
    if (w->omin > -1e20f || w->omax < 1e20f)
        clamp_buf(y, M * out, w->omin, w->omax);
    free(xc);
}

static float gelu_quick(float x)
{
    return x / (1.0f + expf(-1.702f * x));
}
static void act_gate(float* y, const float* gate, const float* up, uint32_t n, int op)
{
    uint32_t i;
    if (op == 2) {
        for (i = 0; i < n; i++) {
            float g = gate[i];
            y[i] = (g / (1.f + expf(-g))) * up[i];
        }
        return;
    }
    if (op == 1) {
        geglu(y, gate, up, n);
        return;
    }
    for (i = 0; i < n; i++) y[i] = gelu_quick(gate[i]) * up[i];
}

static void rms_w(float* y, const float* x, const ClipT* w, uint32_t n, float eps)
{
    if (w && w->p && w->dtype == 0)
        rmsnorm(y, x, w->p, n, eps, DT_F32);
    else {
        rmsnorm_unit(y, x, n, eps);
        if (w && w->p) {
            uint32_t i;
            for (i = 0; i < n; i++) y[i] *= tload(w, i);
        }
    }
}

static void scale_vec(float* y, const ClipT* s, uint32_t n)
{
    uint32_t i, ne = 1, d;
    if (!s || !s->p) return;
    for (d = 0; d < s->ndim; d++) ne *= (uint32_t)s->dims[d];
    if (ne <= 1) {
        float a = tload(s, 0);
        for (i = 0; i < n; i++) y[i] *= a;
        return;
    }
    if (s->dtype == 0)
        for (i = 0; i < n; i++) y[i] *= ((const float*)s->p)[i];
    else
        for (i = 0; i < n; i++) y[i] *= tload(s, i);
}

/* NEOX rope on a half-head (len = n_dim), theta typically 100 */
static void rope_neox(float* row, uint32_t n_dim, int32_t pos, float theta)
{
    uint32_t half = n_dim / 2, i;
    float inv;
    if (n_dim < 2) return;
    inv = 2.f / (float)n_dim;
    for (i = 0; i < half; i++) {
        float freq = (float)pos * powf(theta, -(float)i * inv);
        float c = cosf(freq), s = sinf(freq);
        float x0 = row[i], x1 = row[i + half];
        row[i] = x0 * c - x1 * s;
        row[i + half] = x0 * s + x1 * c;
    }
}

static void attn_full(G4v* vis, float* out, const float* q, const float* k, const float* v,
                     uint32_t n, uint32_t nh, uint32_t dh, float scale)
{
    uint32_t t;
#pragma omp parallel for schedule(static)
    for (t = 0; t < n; t++) {
        uint32_t h, j, d;
        int tid = 0;
        float* sc;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        if (tid < 0 || (uint32_t)tid >= vis->gemm_nthr) tid = 0;
        sc = vis->sc + (size_t)tid * vis->npos_max;
        for (h = 0; h < nh; h++) {
            const float* qt = q + ((size_t)t * nh + h) * dh;
            float mx = -1e30f, sum = 0.f;
            float* o = out + ((size_t)t * nh + h) * dh;
            for (j = 0; j < n; j++) {
                sc[j] = dot_f32(qt, k + ((size_t)j * nh + h) * dh, dh) * scale;
                if (sc[j] > mx) mx = sc[j];
            }
            for (j = 0; j < n; j++) {
                sc[j] = expf(sc[j] - mx);
                sum += sc[j];
            }
            {
                float inv = 1.f / (sum > 0.f ? sum : 1.f);
                memset(o, 0, (size_t)dh * 4);
                for (j = 0; j < n; j++) {
                    float a = sc[j] * inv;
                    const float* vj = v + ((size_t)j * nh + h) * dh;
                    for (d = 0; d < dh; d++) o[d] += a * vj[d];
                }
            }
        }
    }
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

static int load_layer(G4v* v, uint32_t il)
{
    ClipLayer* L = &v->layers[il];
    char n[96];
    memset(L, 0, sizeof(*L));
#define OPT(field, fmt) do { snprintf(n, sizeof(n), fmt, il); opt_t(v, &L->field, n); } while (0)
    OPT(ln1_w, "v.blk.%u.ln1.weight");
    OPT(ln2_w, "v.blk.%u.ln2.weight");
    OPT(q_w, "v.blk.%u.attn_q.weight");
    OPT(k_w, "v.blk.%u.attn_k.weight");
    OPT(v_w, "v.blk.%u.attn_v.weight");
    OPT(o_w, "v.blk.%u.attn_out.weight");
    OPT(q_b, "v.blk.%u.attn_q.bias");
    OPT(k_b, "v.blk.%u.attn_k.bias");
    OPT(v_b, "v.blk.%u.attn_v.bias");
    OPT(o_b, "v.blk.%u.attn_out.bias");
    OPT(q_norm, "v.blk.%u.attn_q_norm.weight");
    OPT(k_norm, "v.blk.%u.attn_k_norm.weight");
    OPT(up_w, "v.blk.%u.ffn_up.weight");
    OPT(gate_w, "v.blk.%u.ffn_gate.weight");
    OPT(down_w, "v.blk.%u.ffn_down.weight");
    OPT(up_b, "v.blk.%u.ffn_up.bias");
    OPT(gate_b, "v.blk.%u.ffn_gate.bias");
    OPT(down_b, "v.blk.%u.ffn_down.bias");
    OPT(ls1, "v.blk.%u.ls1.weight");
    OPT(ls2, "v.blk.%u.ls2.weight");
    OPT(ls_out, "v.blk.%u.out_scale.weight");
    OPT(attn_post, "v.blk.%u.attn_post_norm.weight");
    OPT(ff_post, "v.blk.%u.ffn_post_norm.weight");
#undef OPT
    L->has_gate = L->gate_w.p != NULL;
    L->has_q_norm = L->q_norm.p != NULL;
    L->has_k_norm = L->k_norm.p != NULL;
    if (!L->ln1_w.p || !L->q_w.p || !L->o_w.p || !L->up_w.p || !L->down_w.p) return -1;
    return 0;
}

static void bind_clamp(G4v* v)
{
    int i;
    for (i = 0; i < v->n_t; i++) {
        ClipT* t = &v->ts[i];
        size_t n = strlen(t->name);
        char base[160], a[176], b[176], c[176], d[176];
        ClipT *tmax, *tmin, *omax, *omin;
        t->imin = -FLT_MAX; t->imax = FLT_MAX;
        t->omin = -FLT_MAX; t->omax = FLT_MAX;
        if (n < 8 || strcmp(t->name + n - 7, ".weight") != 0) continue;
        memcpy(base, t->name, n - 7); base[n - 7] = 0;
        snprintf(a, sizeof(a), "%s.input_max", base);
        snprintf(b, sizeof(b), "%s.input_min", base);
        snprintf(c, sizeof(c), "%s.output_max", base);
        snprintf(d, sizeof(d), "%s.output_min", base);
        tmax = find_t(v, a); tmin = find_t(v, b);
        omax = find_t(v, c); omin = find_t(v, d);
        if (tmin && tmin->p) t->imin = tload(tmin, 0);
        if (tmax && tmax->p) t->imax = tload(tmax, 0);
        if (omin && omin->p) t->omin = tload(omin, 0);
        if (omax && omax->p) t->omax = tload(omax, 0);
    }
}

G4v* g4v_load(const char* path, char* err, size_t errlen)
{
    G4v* v = (G4v*)ycalloc(1, sizeof(*v));
    const uint8_t* data;
    uint64_t fsize, data_start, n_kv, n_tensors, i;
    uint32_t ver, alignment = 32;
    GB b;
    int use_gelu = 0, use_silu = 0;
    uint32_t ps, merge, pa;
    if (!v) { if (err) snprintf(err, errlen, "oom"); return NULL; }
    if (wmap_open(path, &v->map) != 0) {
        if (err) snprintf(err, errlen, "cannot mmap %s", path);
        free(v); return NULL;
    }
    data = (const uint8_t*)v->map.base;
    fsize = v->map.size;
    if (fsize < 24 || memcmp(data, "GGUF", 4) != 0) {
        if (err) snprintf(err, errlen, "not gguf");
        g4v_free(v); return NULL;
    }
    b.p = data + 4; b.end = data + fsize; b.err = 0;
    ver = gb_u32(&b);
    n_tensors = gb_u64(&b);
    n_kv = gb_u64(&b);
    (void)ver;
    v->image_size = 896; v->patch = 14; v->n_embd = 1152; v->n_ff = 4304;
    v->n_layer = 27; v->n_head = 16; v->n_out_embd = 1536; v->n_merge = 3;
    v->eps = 1e-6f; v->rope_theta = 100.f;
    v->mean[0] = v->mean[1] = v->mean[2] = 0.5f;
    v->std[0] = v->std[1] = v->std[2] = 0.5f;
    v->ntok_min = 40; v->ntok_max = 280; v->ffn_op = 0;
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
            else if (!strcmp(key, "general.alignment")) alignment = u;
        } else if (typ == 6 && !strcmp(key, "clip.vision.attention.layer_norm_epsilon")) {
            uint32_t u = gb_u32(&b); memcpy(&v->eps, &u, 4);
        } else if (typ == 7 && (!strcmp(key, "clip.use_gelu") || !strcmp(key, "clip.use_silu"))) {
            uint8_t u = 0;
            if (b.p < b.end) { u = *b.p; b.p++; }
            if (!strcmp(key, "clip.use_gelu")) use_gelu = u;
            else use_silu = u;
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
    if (use_silu) v->ffn_op = 2;
    else if (use_gelu) v->ffn_op = 1;
    if (v->n_merge == 0) v->n_merge = 3;
    if (b.err) { if (err) snprintf(err, errlen, "bad mmproj kv"); g4v_free(v); return NULL; }
    v->ts = (ClipT*)ycalloc((size_t)n_tensors, sizeof(ClipT));
    if (!v->ts) { g4v_free(v); if (err) snprintf(err, errlen, "oom"); return NULL; }
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
    if (b.err) { if (err) snprintf(err, errlen, "bad mmproj tensors"); g4v_free(v); return NULL; }
    data_start = align_up_u((uint64_t)(b.p - data), alignment ? alignment : 32);
    for (i = 0; i < (uint64_t)v->n_t; i++) {
        uint64_t ne = 1, d, nb;
        v->ts[i].p = data + data_start + v->ts[i].off;
        for (d = 0; d < v->ts[i].ndim; d++) ne *= v->ts[i].dims[d];
        nb = ne * (v->ts[i].dtype == 0 ? 4ull : 2ull);
        if (v->ts[i].p < data || v->ts[i].p + nb > data + fsize) {
            if (err) snprintf(err, errlen, "tensor %s out of file", v->ts[i].name);
            g4v_free(v); return NULL;
        }
    }
    bind_clamp(v);
    if (req_t(v, &v->patch_w, "v.patch_embd.weight") != 0 ||
        req_t(v, &v->pos_embd, "v.position_embd.weight") != 0 ||
        req_t(v, &v->mm_proj, "mm.input_projection.weight") != 0) {
        if (err) snprintf(err, errlen, "gemma4v mmproj missing tensors");
        g4v_free(v); return NULL;
    }
    opt_t(v, &v->pre_ln_w, "v.pre_ln.weight");
    opt_t(v, &v->std_bias, "v.std_bias");
    opt_t(v, &v->std_scale, "v.std_scale");
    v->has_pre_ln = v->pre_ln_w.p != NULL;
    v->has_std = v->std_bias.p != NULL && v->std_scale.p != NULL;
    /* ggml: ne[0]=in, ne[1]=out */
    if (v->mm_proj.dims[1]) v->n_out_embd = (uint32_t)v->mm_proj.dims[1];
    if (v->n_layer > G4V_MAX_LAYERS) { if (err) snprintf(err, errlen, "too many vit layers"); g4v_free(v); return NULL; }
    for (i = 0; i < v->n_layer; i++)
        if (load_layer(v, (uint32_t)i) != 0) {
            if (err) snprintf(err, errlen, "missing vit layer %u", (unsigned)i);
            g4v_free(v); return NULL;
        }
    if (v->n_ff == 0 && v->layers[0].up_w.dims[1])
        v->n_ff = (uint32_t)v->layers[0].up_w.dims[1];
    ps = v->patch; merge = v->n_merge ? v->n_merge : 3;
    pa = ps * ps * merge * merge;
    v->npos_max = (v->ntok_max + 8) * merge * merge;
    if (v->npos_max < 4096) v->npos_max = 4096;
    {
        uint32_t e = v->n_embd, K = 3 * ps * ps, oc, ic, ky, kx;
        size_t cap = (size_t)v->npos_max * e;
        size_t ffcap = cap;
        if (ffcap < (size_t)v->npos_max * v->n_ff) ffcap = (size_t)v->npos_max * v->n_ff;
        v->x = (float*)ymalloc(cap * 4);
        v->res = (float*)ymalloc(cap * 4);
        v->tmp = (float*)ymalloc(cap * 4);
        v->q = (float*)ymalloc(cap * 4);
        v->k = (float*)ymalloc(cap * 4);
        v->v = (float*)ymalloc(cap * 4);
        v->attn = (float*)ymalloc(cap * 4);
        v->ff = (float*)ymalloc(ffcap * 4);
        v->ffg = (float*)ymalloc(ffcap * 4);
        v->patch_wf = (float*)ymalloc((size_t)e * K * 4);
        if (v->pos_embd.ndim >= 3)
            v->pos_n = (uint32_t)v->pos_embd.dims[1];
        else
            v->pos_n = v->pos_embd.dims[1] ? (uint32_t)(v->pos_embd.dims[1] / 2) : 0;
        if (!v->pos_n && v->pos_embd.dims[0]) v->pos_n = (uint32_t)(v->pos_embd.dims[0] / 2);
        v->pos_x = (float*)ymalloc((size_t)v->pos_n * e * 4);
        v->pos_y = (float*)ymalloc((size_t)v->pos_n * e * 4);
        v->gemm_nthr = 1;
#ifdef _OPENMP
        v->gemm_nthr = (uint32_t)omp_get_max_threads();
        if (v->gemm_nthr < 1) v->gemm_nthr = 1;
#endif
        v->sc = (float*)ymalloc((size_t)v->gemm_nthr * v->npos_max * 4);
        v->gemm_in_cap = e;
        if (v->gemm_in_cap < v->n_ff) v->gemm_in_cap = v->n_ff;
        if (v->gemm_in_cap < K) v->gemm_in_cap = K;
        v->gemm_row = (float*)ymalloc((size_t)v->gemm_nthr * v->gemm_in_cap * 4);
        if (v->patch_wf) {
            for (oc = 0; oc < e; oc++)
                for (ic = 0; ic < 3; ic++)
                    for (ky = 0; ky < ps; ky++)
                        for (kx = 0; kx < ps; kx++) {
                            uint64_t wi = (uint64_t)kx + (uint64_t)ps * (ky + ps * (ic + 3 * oc));
                            uint32_t k = ic * ps * ps + ky * ps + kx;
                            v->patch_wf[(size_t)oc * K + k] = tload(&v->patch_w, wi);
                        }
        }
        if (v->pos_x && v->pos_y && v->pos_n) {
            uint32_t p, j;
            /* GGUF [n_embd, 2*pos_n]: row p at p*n_embd */
            for (p = 0; p < v->pos_n; p++)
                for (j = 0; j < e; j++) {
                    v->pos_x[(size_t)p * e + j] = tload(&v->pos_embd, (uint64_t)p * e + j);
                    v->pos_y[(size_t)p * e + j] = tload(&v->pos_embd, (uint64_t)(p + v->pos_n) * e + j);
                }
        }
        if (!v->x || !v->res || !v->tmp || !v->q || !v->k || !v->v || !v->attn || !v->ff ||
            !v->ffg || !v->patch_wf || !v->pos_x || !v->pos_y || !v->sc || !v->gemm_row) {
            if (err) snprintf(err, errlen, "oom vis buf");
            g4v_free(v); return NULL;
        }
        (void)pa;
    }
    ylog_info("vision gemma4v: patch=%u layers=%u embd=%u out=%u merge=%u heads=%u ntok=%u..%u",
              v->patch, v->n_layer, v->n_embd, v->n_out_embd, v->n_merge, v->n_head,
              v->ntok_min, v->ntok_max);
    return v;
}

void g4v_free(G4v* v)
{
    if (!v) return;
    free(v->x); free(v->res); free(v->tmp); free(v->q); free(v->k); free(v->v);
    free(v->attn); free(v->ff); free(v->ffg); free(v->sc); free(v->patch_wf); free(v->pos_x); free(v->pos_y);
    free(v->gemm_row);
    free(v->ts);
    wmap_close(&v->map);
    free(v);
}

int g4v_n_tokens(const G4v* v)
{
    return v ? (int)v->ntok_max : 0;
}
int g4v_hidden(const G4v* v)
{
    return v ? (int)v->n_out_embd : 0;
}

static int encode_run(G4v* vis, const float* chw, uint32_t sw, uint32_t sh, float* out)
{
    uint32_t gh = sh / vis->patch, gw = sw / vis->patch;
    uint32_t n = gh * gw, e = vis->n_embd, il, t, merge = vis->n_merge;
    uint32_t nh = vis->n_head, dh = e / nh, n_ff = vis->n_ff;
    uint32_t ox, oy, out_x, out_y, nout;
    float scale = 1.f; /* gemma4v kq_scale */
    if (n == 0 || n > vis->npos_max || dh == 0 || (gw % merge) || (gh % merge)) return -1;

    /* patch embed */
    {
        uint32_t ps = vis->patch, K = 3 * ps * ps;
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
        {
            ClipT w = vis->patch_w;
            w.p = (const uint8_t*)vis->patch_wf;
            w.dtype = 0;
            gemm_lin(vis, vis->x, col, &w, NULL, n, e, K);
        }
    }
    for (t = 0; t < n; t++) {
        uint32_t py = t / gw, px = t % gw;
        if (px >= vis->pos_n) px = vis->pos_n - 1;
        if (py >= vis->pos_n) py = vis->pos_n - 1;
        add_inplace(vis->x + (size_t)t * e, vis->pos_x + (size_t)px * e, e);
        add_inplace(vis->x + (size_t)t * e, vis->pos_y + (size_t)py * e, e);
    }
    if (vis->has_pre_ln) {
#pragma omp parallel for schedule(static)
        for (t = 0; t < n; t++)
            rms_w(vis->tmp + (size_t)t * e, vis->x + (size_t)t * e, &vis->pre_ln_w, e, vis->eps);
        memcpy(vis->x, vis->tmp, (size_t)n * e * 4);
    }

    for (il = 0; il < vis->n_layer; il++) {
        ClipLayer* L = &vis->layers[il];
        memcpy(vis->res, vis->x, (size_t)n * e * 4);
#pragma omp parallel for schedule(static)
        for (t = 0; t < n; t++)
            rms_w(vis->tmp + (size_t)t * e, vis->x + (size_t)t * e, &L->ln1_w, e, vis->eps);
        gemm_lin(vis, vis->q, vis->tmp, &L->q_w, L->q_b.p ? &L->q_b : NULL, n, e, e);
        gemm_lin(vis, vis->k, vis->tmp, &L->k_w, L->k_b.p ? &L->k_b : NULL, n, e, e);
        gemm_lin(vis, vis->v, vis->tmp, &L->v_w, L->v_b.p ? &L->v_b : NULL, n, e, e);
        if (L->has_q_norm) {
            uint32_t qn = (uint32_t)L->q_norm.dims[0];
            if (qn == dh) {
#pragma omp parallel for schedule(static)
                for (t = 0; t < n; t++) {
                    uint32_t h;
                    for (h = 0; h < nh; h++) {
                        float* r = vis->q + ((size_t)t * nh + h) * dh;
                        rms_w(r, r, &L->q_norm, dh, vis->eps);
                    }
                }
            } else {
#pragma omp parallel for schedule(static)
                for (t = 0; t < n; t++)
                    rms_w(vis->ff + (size_t)t * e, vis->q + (size_t)t * e, &L->q_norm, e, vis->eps);
                memcpy(vis->q, vis->ff, (size_t)n * e * 4);
            }
        }
        if (L->has_k_norm) {
            uint32_t kn = (uint32_t)L->k_norm.dims[0];
            if (kn == dh) {
#pragma omp parallel for schedule(static)
                for (t = 0; t < n; t++) {
                    uint32_t h;
                    for (h = 0; h < nh; h++) {
                        float* r = vis->k + ((size_t)t * nh + h) * dh;
                        rms_w(r, r, &L->k_norm, dh, vis->eps);
                    }
                }
            } else {
#pragma omp parallel for schedule(static)
                for (t = 0; t < n; t++)
                    rms_w(vis->ff + (size_t)t * e, vis->k + (size_t)t * e, &L->k_norm, e, vis->eps);
                memcpy(vis->k, vis->ff, (size_t)n * e * 4);
            }
        }
#pragma omp parallel for schedule(static)
        for (t = 0; t < n; t++) {
            uint32_t h, px = t % gw, py = t / gw;
            for (h = 0; h < nh; h++) {
                float* qr = vis->q + ((size_t)t * nh + h) * dh;
                float* kr = vis->k + ((size_t)t * nh + h) * dh;
                float* vr = vis->v + ((size_t)t * nh + h) * dh;
                rope_neox(qr, dh / 2, (int32_t)px, vis->rope_theta);
                rope_neox(qr + dh / 2, dh / 2, (int32_t)py, vis->rope_theta);
                rope_neox(kr, dh / 2, (int32_t)px, vis->rope_theta);
                rope_neox(kr + dh / 2, dh / 2, (int32_t)py, vis->rope_theta);
                rmsnorm_unit(vr, vr, dh, vis->eps);
            }
        }
        attn_full(vis, vis->attn, vis->q, vis->k, vis->v, n, nh, dh, scale);
        gemm_lin(vis, vis->tmp, vis->attn, &L->o_w, L->o_b.p ? &L->o_b : NULL, n, e, e);
        if (L->ls1.p) {
            for (t = 0; t < n; t++) scale_vec(vis->tmp + (size_t)t * e, &L->ls1, e);
        }
        if (L->attn_post.p) {
#pragma omp parallel for schedule(static)
            for (t = 0; t < n; t++)
                rms_w(vis->ff + (size_t)t * e, vis->tmp + (size_t)t * e, &L->attn_post, e, vis->eps);
            memcpy(vis->tmp, vis->ff, (size_t)n * e * 4);
        }
        for (t = 0; t < n * e; t++) vis->x[t] = vis->res[t] + vis->tmp[t];
        memcpy(vis->res, vis->x, (size_t)n * e * 4);
#pragma omp parallel for schedule(static)
        for (t = 0; t < n; t++)
            rms_w(vis->tmp + (size_t)t * e, vis->x + (size_t)t * e, &L->ln2_w, e, vis->eps);
        gemm_lin(vis, vis->ff, vis->tmp, &L->up_w, L->up_b.p ? &L->up_b : NULL, n, n_ff, e);
        if (L->has_gate) {
            gemm_lin(vis, vis->ffg, vis->tmp, &L->gate_w, L->gate_b.p ? &L->gate_b : NULL, n, n_ff, e);
            act_gate(vis->ff, vis->ffg, vis->ff, n * n_ff, vis->ffn_op);
        } else {
            uint32_t u;
            for (u = 0; u < n * n_ff; u++) vis->ff[u] = gelu_quick(vis->ff[u]);
        }
        gemm_lin(vis, vis->tmp, vis->ff, &L->down_w, L->down_b.p ? &L->down_b : NULL, n, e, n_ff);
        if (L->ff_post.p) {
#pragma omp parallel for schedule(static)
            for (t = 0; t < n; t++)
                rms_w(vis->ff + (size_t)t * e, vis->tmp + (size_t)t * e, &L->ff_post, e, vis->eps);
            memcpy(vis->tmp, vis->ff, (size_t)n * e * 4);
        }
        if (L->ls2.p) {
            for (t = 0; t < n; t++) scale_vec(vis->tmp + (size_t)t * e, &L->ls2, e);
        }
        for (t = 0; t < n * e; t++) vis->x[t] = vis->res[t] + vis->tmp[t];
        if (L->ls_out.p) {
            for (t = 0; t < n; t++) scale_vec(vis->x + (size_t)t * e, &L->ls_out, e);
        }
    }

    out_x = gw / merge; out_y = gh / merge; nout = out_x * out_y;
    {
        float s = sqrtf((float)e), inv = 1.f / (float)(merge * merge);
        for (oy = 0; oy < out_y; oy++)
            for (ox = 0; ox < out_x; ox++) {
                uint32_t ky, kx, c;
                float* dst = vis->tmp + ((size_t)oy * out_x + ox) * e;
                memset(dst, 0, (size_t)e * 4);
                for (ky = 0; ky < merge; ky++)
                    for (kx = 0; kx < merge; kx++) {
                        const float* src = vis->x + ((size_t)(oy * merge + ky) * gw + (ox * merge + kx)) * e;
                        for (c = 0; c < e; c++) dst[c] += src[c];
                    }
                for (c = 0; c < e; c++) dst[c] = dst[c] * inv * s;
            }
    }
    if (vis->has_std) {
        for (t = 0; t < nout; t++) {
            float* row = vis->tmp + (size_t)t * e;
            uint32_t c;
            for (c = 0; c < e; c++)
                row[c] = (row[c] - tload(&vis->std_bias, c)) * tload(&vis->std_scale, c);
        }
    }
#pragma omp parallel for schedule(static)
    for (t = 0; t < nout; t++)
        rmsnorm_unit(vis->x + (size_t)t * e, vis->tmp + (size_t)t * e, e, vis->eps);
    gemm_lin(vis, out, vis->x, &vis->mm_proj, NULL, nout, vis->n_out_embd, e);
    return (int)nout;
}

static float cubic_cr(float a, float b, float c, float d, float t)
{
    return b + 0.5f * t * (c - a + t * (2.f * a - 5.f * b + 4.f * c - d + t * (3.f * (b - c) + d - a)));
}

static float samp3(const unsigned char* img, int w, int h, int x, int y, int ic)
{
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= w) x = w - 1;
    if (y >= h) y = h - 1;
    return (float)img[(y * w + x) * 3 + ic];
}

int g4v_encode(G4v* v, const char* image_path, float* out, int max_tok, char* err, size_t errlen)
{
    int w = 0, h = 0, c = 0, x, y, ic, ntok, sw, sh, factor, min_px, max_px;
    unsigned char* img;
    float* chw;
    if (!v || !image_path || !out) { if (err) snprintf(err, errlen, "bad vision args"); return -1; }
    img = stbi_load(image_path, &w, &h, &c, 3);
    if (!img) { if (err) snprintf(err, errlen, "cannot load image %s", image_path); return -1; }
    factor = (int)(v->patch * (v->n_merge ? v->n_merge : 3));
    min_px = (int)(v->ntok_min * v->patch * v->patch * v->n_merge * v->n_merge);
    max_px = (int)(v->ntok_max * v->patch * v->patch * v->n_merge * v->n_merge);
    smart_hw(w, h, factor, min_px, max_px, &sw, &sh);
    ntok = (sw / (int)v->patch / (int)v->n_merge) * (sh / (int)v->patch / (int)v->n_merge);
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
        int y0 = (int)floorf(fy);
        float wy = fy - (float)y0;
        if (y0 < 0) { y0 = 0; wy = 0; }
        if (y0 >= h) y0 = h - 1;
        for (x = 0; x < sw; x++) {
            float fx = ((float)x + 0.5f) * (float)w / (float)sw - 0.5f;
            int x0 = (int)floorf(fx);
            float wx = fx - (float)x0;
            if (x0 < 0) { x0 = 0; wx = 0; }
            if (x0 >= w) x0 = w - 1;
            for (ic = 0; ic < 3; ic++) {
                float col[4], p, z;
                int yy, xx;
                for (yy = 0; yy < 4; yy++) {
                    float row[4];
                    int sy = y0 + yy - 1;
                    for (xx = 0; xx < 4; xx++)
                        row[xx] = samp3(img, w, h, x0 + xx - 1, sy, ic);
                    col[yy] = cubic_cr(row[0], row[1], row[2], row[3], wx);
                }
                p = cubic_cr(col[0], col[1], col[2], col[3], wy);
                if (p < 0.f) p = 0.f;
                if (p > 255.f) p = 255.f;
                z = (p / 255.f - v->mean[ic]) / v->std[ic];
                chw[(size_t)ic * sh * sw + (uint32_t)y * sw + (uint32_t)x] = z * 2.f - 1.f;
            }
        }
    }
    stbi_image_free(img);
    ylog_info("vision gemma4v: encoding %dx%d (src %dx%d)", sw, sh, w, h);
    {
        uint64_t t0 = ynow_ms();
        int got = encode_run(v, chw, (uint32_t)sw, (uint32_t)sh, out);
        free(chw);
        if (got < 0) {
            if (err) snprintf(err, errlen, "encode failed");
            return -1;
        }
        ylog_info("vision gemma4v: %dx%d -> %d tokens hidden=%u in %.2f s",
                  w, h, got, v->n_out_embd, (double)(ynow_ms() - t0) / 1000.0);
        return got;
    }
}
