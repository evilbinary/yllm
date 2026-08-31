/* MiniCPM-V 4.6: 直接 mmap clip mmproj GGUF, CPU 前向对齐 llama.cpp clip_graph_minicpmv4_6 */
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
#if defined(__aarch64__)
#include <arm_neon.h>
#endif
#ifdef _WIN32
#include <malloc.h>
#define vis_alloca _alloca
#else
#include <alloca.h>
#define vis_alloca alloca
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

struct Mcpv {
    WMap map;
    ClipT* ts;
    int n_t;
    uint32_t image_size, patch, n_embd, n_ff, n_layer, n_head, n_out_embd;
    uint32_t insert_lid, n_merge;
    int downsample; /* 16 或 4 (--opt downsample_mode) */
    float mean[3], std[3], eps;
    ClipT patch_w, patch_b, pos_embd, post_ln_w, post_ln_b;
    ClipT mm_in_w, mm_in_b, mm_up_w, mm_up_b, mm_down_w, mm_down_b;
    ClipT vm_ln1_w, vm_ln1_b, vm_q_w, vm_q_b, vm_k_w, vm_k_b, vm_v_w, vm_v_b, vm_o_w, vm_o_b;
    ClipT vm_ds_ln_w, vm_ds_ln_b, vm_ds_up_w, vm_ds_up_b, vm_ds_down_w, vm_ds_down_b;
    ClipLayer layers[CLIP_MAX_LAYERS];
    float *x, *res, *tmp, *q, *k, *v, *attn, *ff;
    float *patch_wf; /* [n_embd, 3*ps*ps] 行主序 f32 */
    float *pos_f;    /* [70*70, n_embd] */
    float *gemm_row; /* nthr * gemm_in_cap */
    uint32_t gemm_nthr, gemm_in_cap;
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

static ClipT* find_t(Mcpv* v, const char* name)
{
    int i;
    for (i = 0; i < v->n_t; i++)
        if (strcmp(v->ts[i].name, name) == 0) return &v->ts[i];
    return NULL;
}

static int req_t(Mcpv* v, ClipT* dst, const char* name)
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
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
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
#if defined(__aarch64__)
static float hsum4(float32x4_t v)
{
    return vaddvq_f32(v);
}
static float dot_f32(const float* a, const float* b, uint32_t n)
{
    float32x4_t s0 = vdupq_n_f32(0.f), s1 = vdupq_n_f32(0.f);
    uint32_t i = 0;
    for (; i + 16 <= n; i += 16) {
        s0 = vfmaq_f32(s0, vld1q_f32(a + i),      vld1q_f32(b + i));
        s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4),  vld1q_f32(b + i + 4));
        s0 = vfmaq_f32(s0, vld1q_f32(a + i + 8),  vld1q_f32(b + i + 8));
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
#else
static float dot_f32(const float* a, const float* b, uint32_t n)
{
    float acc = 0.f;
    uint32_t i;
    for (i = 0; i < n; i++) acc += a[i] * b[i];
    return acc;
}
#endif
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

static void gemm_nn(float* y, const float* x, const float* w, const float* bias,
                    uint32_t M, uint32_t out, uint32_t in)
{
    uint32_t g, ng = (out + 3u) / 4u;
#pragma omp parallel for schedule(static) if(out > 8)
    for (g = 0; g < ng; g++) {
        uint32_t oo = g * 4, nout = out - oo, m, i, k;
        const float* w0 = w + (size_t)oo * in;
        const float* w1 = w0 + in;
        const float* w2 = w1 + in;
        const float* w3 = w2 + in;
        float b0 = 0.f, b1 = 0.f, b2 = 0.f, b3 = 0.f;
        if (nout > 4) nout = 4;
        if (bias) {
            b0 = bias[oo];
            if (nout > 1) b1 = bias[oo + 1];
            if (nout > 2) b2 = bias[oo + 2];
            if (nout > 3) b3 = bias[oo + 3];
        }
#ifdef __AVX2__
        if (nout == 4) {
            m = 0;
            for (; m + 2 <= M; m += 2) {
                const float* xr0 = x + (size_t)m * in;
                const float* xr1 = xr0 + in;
                __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
                __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
                __m256 c0 = _mm256_setzero_ps(), c1 = _mm256_setzero_ps();
                __m256 c2 = _mm256_setzero_ps(), c3 = _mm256_setzero_ps();
                float* y0 = y + (size_t)m * out + oo;
                float* y1 = y0 + out;
                for (i = 0; i + 8 <= in; i += 8) {
                    __m256 wv0 = _mm256_loadu_ps(w0 + i), wv1 = _mm256_loadu_ps(w1 + i);
                    __m256 wv2 = _mm256_loadu_ps(w2 + i), wv3 = _mm256_loadu_ps(w3 + i);
                    __m256 x0 = _mm256_loadu_ps(xr0 + i), x1 = _mm256_loadu_ps(xr1 + i);
                    a0 = _mm256_fmadd_ps(x0, wv0, a0); a1 = _mm256_fmadd_ps(x0, wv1, a1);
                    a2 = _mm256_fmadd_ps(x0, wv2, a2); a3 = _mm256_fmadd_ps(x0, wv3, a3);
                    c0 = _mm256_fmadd_ps(x1, wv0, c0); c1 = _mm256_fmadd_ps(x1, wv1, c1);
                    c2 = _mm256_fmadd_ps(x1, wv2, c2); c3 = _mm256_fmadd_ps(x1, wv3, c3);
                }
                {
                    float s0 = hsum8(a0), s1 = hsum8(a1), s2 = hsum8(a2), s3 = hsum8(a3);
                    float t0 = hsum8(c0), t1 = hsum8(c1), t2 = hsum8(c2), t3 = hsum8(c3);
                    for (; i < in; i++) {
                        float v0 = xr0[i], v1 = xr1[i];
                        s0 += v0 * w0[i]; s1 += v0 * w1[i]; s2 += v0 * w2[i]; s3 += v0 * w3[i];
                        t0 += v1 * w0[i]; t1 += v1 * w1[i]; t2 += v1 * w2[i]; t3 += v1 * w3[i];
                    }
                    y0[0] = s0 + b0; y0[1] = s1 + b1; y0[2] = s2 + b2; y0[3] = s3 + b3;
                    y1[0] = t0 + b0; y1[1] = t1 + b1; y1[2] = t2 + b2; y1[3] = t3 + b3;
                }
            }
            for (; m < M; m++) {
                const float* xr = x + (size_t)m * in;
                __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
                __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
                float* yr = y + (size_t)m * out + oo;
                i = 0;
                for (; i + 8 <= in; i += 8) {
                    __m256 xv = _mm256_loadu_ps(xr + i);
                    a0 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w0 + i), a0);
                    a1 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w1 + i), a1);
                    a2 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w2 + i), a2);
                    a3 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w3 + i), a3);
                }
                {
                    float s0 = hsum8(a0), s1 = hsum8(a1), s2 = hsum8(a2), s3 = hsum8(a3);
                    for (; i < in; i++) {
                        float xv = xr[i];
                        s0 += xv * w0[i]; s1 += xv * w1[i]; s2 += xv * w2[i]; s3 += xv * w3[i];
                    }
                    yr[0] = s0 + b0; yr[1] = s1 + b1; yr[2] = s2 + b2; yr[3] = s3 + b3;
                }
            }
        } else
#elif defined(__aarch64__)
        if (nout == 4) {
            m = 0;
            for (; m + 4 <= M; m += 4) {
                const float* xr0 = x + (size_t)m * in;
                const float* xr1 = xr0 + in, *xr2 = xr1 + in, *xr3 = xr2 + in;
                float32x4_t a00 = vdupq_n_f32(0.f), a01 = vdupq_n_f32(0.f);
                float32x4_t a02 = vdupq_n_f32(0.f), a03 = vdupq_n_f32(0.f);
                float32x4_t a10 = vdupq_n_f32(0.f), a11 = vdupq_n_f32(0.f);
                float32x4_t a12 = vdupq_n_f32(0.f), a13 = vdupq_n_f32(0.f);
                float32x4_t a20 = vdupq_n_f32(0.f), a21 = vdupq_n_f32(0.f);
                float32x4_t a22 = vdupq_n_f32(0.f), a23 = vdupq_n_f32(0.f);
                float32x4_t a30 = vdupq_n_f32(0.f), a31 = vdupq_n_f32(0.f);
                float32x4_t a32 = vdupq_n_f32(0.f), a33 = vdupq_n_f32(0.f);
                float* y0 = y + (size_t)m * out + oo;
                for (i = 0; i + 4 <= in; i += 4) {
                    float32x4_t wv0 = vld1q_f32(w0 + i), wv1 = vld1q_f32(w1 + i);
                    float32x4_t wv2 = vld1q_f32(w2 + i), wv3 = vld1q_f32(w3 + i);
                    float32x4_t x0 = vld1q_f32(xr0 + i), x1 = vld1q_f32(xr1 + i);
                    float32x4_t x2 = vld1q_f32(xr2 + i), x3 = vld1q_f32(xr3 + i);
                    a00 = vfmaq_f32(a00, x0, wv0); a01 = vfmaq_f32(a01, x0, wv1);
                    a02 = vfmaq_f32(a02, x0, wv2); a03 = vfmaq_f32(a03, x0, wv3);
                    a10 = vfmaq_f32(a10, x1, wv0); a11 = vfmaq_f32(a11, x1, wv1);
                    a12 = vfmaq_f32(a12, x1, wv2); a13 = vfmaq_f32(a13, x1, wv3);
                    a20 = vfmaq_f32(a20, x2, wv0); a21 = vfmaq_f32(a21, x2, wv1);
                    a22 = vfmaq_f32(a22, x2, wv2); a23 = vfmaq_f32(a23, x2, wv3);
                    a30 = vfmaq_f32(a30, x3, wv0); a31 = vfmaq_f32(a31, x3, wv1);
                    a32 = vfmaq_f32(a32, x3, wv2); a33 = vfmaq_f32(a33, x3, wv3);
                }
                {
                    float s00 = hsum4(a00), s01 = hsum4(a01), s02 = hsum4(a02), s03 = hsum4(a03);
                    float s10 = hsum4(a10), s11 = hsum4(a11), s12 = hsum4(a12), s13 = hsum4(a13);
                    float s20 = hsum4(a20), s21 = hsum4(a21), s22 = hsum4(a22), s23 = hsum4(a23);
                    float s30 = hsum4(a30), s31 = hsum4(a31), s32 = hsum4(a32), s33 = hsum4(a33);
                    for (; i < in; i++) {
                        float v0 = xr0[i], v1 = xr1[i], v2 = xr2[i], v3 = xr3[i];
                        s00 += v0 * w0[i]; s01 += v0 * w1[i]; s02 += v0 * w2[i]; s03 += v0 * w3[i];
                        s10 += v1 * w0[i]; s11 += v1 * w1[i]; s12 += v1 * w2[i]; s13 += v1 * w3[i];
                        s20 += v2 * w0[i]; s21 += v2 * w1[i]; s22 += v2 * w2[i]; s23 += v2 * w3[i];
                        s30 += v3 * w0[i]; s31 += v3 * w1[i]; s32 += v3 * w2[i]; s33 += v3 * w3[i];
                    }
                    y0[0] = s00 + b0; y0[1] = s01 + b1; y0[2] = s02 + b2; y0[3] = s03 + b3;
                    y0[out + 0] = s10 + b0; y0[out + 1] = s11 + b1; y0[out + 2] = s12 + b2; y0[out + 3] = s13 + b3;
                    y0[2 * out + 0] = s20 + b0; y0[2 * out + 1] = s21 + b1; y0[2 * out + 2] = s22 + b2; y0[2 * out + 3] = s23 + b3;
                    y0[3 * out + 0] = s30 + b0; y0[3 * out + 1] = s31 + b1; y0[3 * out + 2] = s32 + b2; y0[3 * out + 3] = s33 + b3;
                }
            }
            for (; m < M; m++) {
                const float* xr = x + (size_t)m * in;
                float32x4_t a0 = vdupq_n_f32(0.f), a1 = vdupq_n_f32(0.f);
                float32x4_t a2 = vdupq_n_f32(0.f), a3 = vdupq_n_f32(0.f);
                float* yr = y + (size_t)m * out + oo;
                i = 0;
                for (; i + 4 <= in; i += 4) {
                    float32x4_t xv = vld1q_f32(xr + i);
                    a0 = vfmaq_f32(a0, xv, vld1q_f32(w0 + i));
                    a1 = vfmaq_f32(a1, xv, vld1q_f32(w1 + i));
                    a2 = vfmaq_f32(a2, xv, vld1q_f32(w2 + i));
                    a3 = vfmaq_f32(a3, xv, vld1q_f32(w3 + i));
                }
                {
                    float s0 = hsum4(a0), s1 = hsum4(a1), s2 = hsum4(a2), s3 = hsum4(a3);
                    for (; i < in; i++) {
                        float xv = xr[i];
                        s0 += xv * w0[i]; s1 += xv * w1[i]; s2 += xv * w2[i]; s3 += xv * w3[i];
                    }
                    yr[0] = s0 + b0; yr[1] = s1 + b1; yr[2] = s2 + b2; yr[3] = s3 + b3;
                }
            }
        } else
#endif
        {
            for (m = 0; m < M; m++) {
                const float* xr = x + (size_t)m * in;
                float* yr = y + (size_t)m * out + oo;
                for (k = 0; k < nout; k++)
                    yr[k] = dot_f32(xr, w0 + (size_t)k * in, in) + (k == 0 ? b0 : k == 1 ? b1 : k == 2 ? b2 : b3);
            }
        }
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

/* y[M,out] = x[M,in] · W[out,in]^T */
static void gemm_lin(Mcpv* vis, float* y, const float* x, const ClipT* w, const ClipT* bias,
                     uint32_t M, uint32_t out, uint32_t in)
{
    uint32_t oo;
    float* bf = NULL;
    if (bias && bias->p) {
        bf = (float*)vis_alloca((size_t)out * 4);
        for (oo = 0; oo < out; oo++) bf[oo] = tload(bias, oo);
    }
    if (w->dtype == 0) {
        gemm_nn(y, x, (const float*)w->p, bf, M, out, in);
        return;
    }
    if (vis->gemm_row && (uint64_t)4 * in + (uint64_t)M * 4 <= vis->gemm_in_cap) {
        uint32_t g, ng = (out + 3u) / 4u;
#pragma omp parallel for schedule(static)
        for (g = 0; g < ng; g++) {
            int tid = 0;
            uint32_t o0 = g * 4, nout = out - o0, k, m;
            float* wr;
            float* tmp;
            float b4[4];
            if (nout > 4) nout = 4;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            if (tid < 0 || (uint32_t)tid >= vis->gemm_nthr) tid = 0;
            wr = vis->gemm_row + (size_t)tid * vis->gemm_in_cap;
            tmp = wr + (size_t)4 * in;
            for (k = 0; k < 4; k++) {
                if (k < nout)
                    f16row_to_f32(wr + (size_t)k * in, (const uint16_t*)w->p + (size_t)(o0 + k) * in, in);
                else
                    memset(wr + (size_t)k * in, 0, (size_t)in * 4);
                b4[k] = (bf && k < nout) ? bf[o0 + k] : 0.f;
            }
            gemm_nn(tmp, x, wr, b4, M, 4, in);
            for (m = 0; m < M; m++)
                memcpy(y + (size_t)m * out + o0, tmp + (size_t)m * 4, (size_t)nout * 4);
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
    gemm_nn(y, x, w, bias, M, out, in);
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
#pragma omp parallel for schedule(static)
    for (t = 0; t < n; t++) {
        uint32_t h, j;
        for (h = 0; h < n_head; h++) {
            const float* qt = q + ((size_t)t * n_head + h) * dh;
            float mx = -1e30f, sum = 0.f;
            float sc[CLIP_MAX_POS];
            for (j = 0; j < n; j++) {
                const float* kj = k + ((size_t)j * n_head + h) * dh;
                sc[j] = dot_f32(qt, kj, dh) * scale;
                if (sc[j] > mx) mx = sc[j];
            }
            for (j = 0; j < n; j++) {
                sc[j] = expf(sc[j] - mx);
                sum += sc[j];
            }
            {
                float inv = 1.f / (sum > 0.f ? sum : 1.f);
                float* o = out + ((size_t)t * n_head + h) * dh;
                uint32_t d;
                memset(o, 0, (size_t)dh * 4);
                for (j = 0; j < n; j++) {
                    float a = sc[j] * inv;
                    const float* vj = v + ((size_t)j * n_head + h) * dh;
#ifdef __AVX2__
                    __m256 as = _mm256_set1_ps(a);
                    for (d = 0; d + 8 <= dh; d += 8) {
                        __m256 ov = _mm256_loadu_ps(o + d);
                        ov = _mm256_fmadd_ps(as, _mm256_loadu_ps(vj + d), ov);
                        _mm256_storeu_ps(o + d, ov);
                    }
                    for (; d < dh; d++) o[d] += a * vj[d];
#elif defined(__aarch64__)
                    {
                        float32x4_t as = vdupq_n_f32(a);
                        for (d = 0; d + 4 <= dh; d += 4)
                            vst1q_f32(o + d, vfmaq_f32(vld1q_f32(o + d), as, vld1q_f32(vj + d)));
                        for (; d < dh; d++) o[d] += a * vj[d];
                    }
#else
                    for (d = 0; d < dh; d++) o[d] += a * vj[d];
#endif
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
#pragma omp parallel for schedule(static)
    for (g = 0; g < ng; g++) {
        uint32_t base = g * 4, ii, h, j, d;
        for (ii = 0; ii < 4; ii++) {
            uint32_t t = base + ii;
            for (h = 0; h < n_head; h++) {
                const float* qt = q + ((size_t)t * n_head + h) * dh;
                float sc[4], mx = -1e30f, sum = 0.f;
                for (j = 0; j < 4; j++) {
                    const float* kj = k + ((size_t)(base + j) * n_head + h) * dh;
                    sc[j] = dot_f32(qt, kj, dh) * scale;
                    if (sc[j] > mx) mx = sc[j];
                }
                for (j = 0; j < 4; j++) { sc[j] = expf(sc[j] - mx); sum += sc[j]; }
                {
                    float inv = 1.f / (sum > 0.f ? sum : 1.f);
                    float* o = out + ((size_t)t * n_head + h) * dh;
                    memset(o, 0, (size_t)dh * 4);
                    for (j = 0; j < 4; j++) {
                        float a = sc[j] * inv;
                        const float* vj = v + ((size_t)(base + j) * n_head + h) * dh;
#if defined(__aarch64__)
                        {
                            float32x4_t as = vdupq_n_f32(a);
                            for (d = 0; d + 4 <= dh; d += 4)
                                vst1q_f32(o + d, vfmaq_f32(vld1q_f32(o + d), as, vld1q_f32(vj + d)));
                            for (; d < dh; d++) o[d] += a * vj[d];
                        }
#else
                        for (d = 0; d < dh; d++) o[d] += a * vj[d];
#endif
                    }
                }
            }
        }
    }
}

static void vit_attn(Mcpv* vis, const ClipT* qw, const ClipT* qb, const ClipT* kw, const ClipT* kb,
                    const ClipT* vw, const ClipT* vb, const ClipT* ow, const ClipT* ob,
                    uint32_t n, int win4)
{
    uint32_t e = vis->n_embd, nh = vis->n_head, dh = e / nh;
    float scale = 1.f / sqrtf((float)dh);
    gemm_lin(vis, vis->q, vis->tmp, qw, qb, n, e, e);
    gemm_lin(vis, vis->k, vis->tmp, kw, kb, n, e, e);
    gemm_lin(vis, vis->v, vis->tmp, vw, vb, n, e, e);
    if (win4) attn_win4(vis->attn, vis->q, vis->k, vis->v, n, nh, dh, scale);
    else attn_full(vis->attn, vis->q, vis->k, vis->v, n, nh, dh, scale);
    gemm_lin(vis, vis->tmp, vis->attn, ow, ob, n, e, e);
}

static void vit_ffn(Mcpv* vis, const ClipT* up_w, const ClipT* up_b,
                    const ClipT* down_w, const ClipT* down_b, uint32_t n, uint32_t in, int erf)
{
    uint32_t n_ff = (uint32_t)up_w->dims[1];
    gemm_lin(vis, vis->ff, vis->tmp, up_w, up_b, n, n_ff, in);
    if (erf) {
        uint32_t i;
        for (i = 0; i < n * n_ff; i++) vis->ff[i] = gelu_erf(vis->ff[i]);
    } else {
        gelu_inplace(vis->ff, n * n_ff);
    }
    gemm_lin(vis, vis->tmp, vis->ff, down_w, down_b, n, vis->n_embd, n_ff);
}

static void vit_block(Mcpv* vis, uint32_t il, uint32_t n)
{
    ClipLayer* L = &vis->layers[il];
    uint32_t e = vis->n_embd, t;
    memcpy(vis->res, vis->x, (size_t)n * e * 4);
#pragma omp parallel for schedule(static)
    for (t = 0; t < n; t++)
        layernorm(vis->tmp + (size_t)t * e, vis->x + (size_t)t * e, &L->ln1_w, &L->ln1_b, e, vis->eps);
    vit_attn(vis, &L->q_w, &L->q_b, &L->k_w, &L->k_b, &L->v_w, &L->v_b, &L->o_w, &L->o_b, n, 0);
    memcpy(vis->x, vis->res, (size_t)n * e * 4);
    add_inplace(vis->x, vis->tmp, n * e);
    memcpy(vis->res, vis->x, (size_t)n * e * 4);
#pragma omp parallel for schedule(static)
    for (t = 0; t < n; t++)
        layernorm(vis->tmp + (size_t)t * e, vis->x + (size_t)t * e, &L->ln2_w, &L->ln2_b, e, vis->eps);
    vit_ffn(vis, &L->up_w, &L->up_b, &L->down_w, &L->down_b, n, e, 0);
    memcpy(vis->x, vis->res, (size_t)n * e * 4);
    add_inplace(vis->x, vis->tmp, n * e);
}

static int load_layer(Mcpv* v, uint32_t il)
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

Mcpv* mcpv_load(const char* path, char* err, size_t errlen)
{
    Mcpv* v = (Mcpv*)ycalloc(1, sizeof(*v));
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
        mcpv_free(v); return NULL;
    }
    b.p = data + 4; b.end = data + fsize; b.err = 0;
    ver = gb_u32(&b);
    n_tensors = gb_u64(&b);
    n_kv = gb_u64(&b);
    v->image_size = 448; v->patch = 14; v->n_embd = 1152; v->n_ff = 4304;
    v->n_layer = 27; v->n_head = 16; v->n_out_embd = 1024; v->insert_lid = 6;
    v->n_merge = 4; v->eps = 1e-6f;
    v->downsample = 4; /* 默认 4x(细); 16x 用 --opt downsample_mode=16x */
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
    if (b.err) { if (err) snprintf(err, errlen, "bad mmproj kv"); mcpv_free(v); return NULL; }
    /* gguf scale_factor=4 对应官方 16x 默认; 细粒度任务 4x 明显更好, 默认走 4x */
    v->ts = (ClipT*)ycalloc((size_t)n_tensors, sizeof(ClipT));
    if (!v->ts) { mcpv_free(v); if (err) snprintf(err, errlen, "oom"); return NULL; }
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
    if (b.err) { if (err) snprintf(err, errlen, "bad mmproj tensors"); mcpv_free(v); return NULL; }
    data_start = align_up_u((uint64_t)(b.p - data), alignment ? alignment : 32);
    for (i = 0; i < (uint64_t)v->n_t; i++) {
        uint64_t ne = 1, d, nb;
        v->ts[i].p = data + data_start + v->ts[i].off;
        for (d = 0; d < v->ts[i].ndim; d++) ne *= v->ts[i].dims[d];
        nb = ne * (v->ts[i].dtype == 0 ? 4ull : 2ull);
        if (v->ts[i].p < data || v->ts[i].p + nb > data + fsize) {
            if (err) snprintf(err, errlen, "tensor %s out of file", v->ts[i].name);
            mcpv_free(v); return NULL;
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
        mcpv_free(v); return NULL;
    }
    if (v->n_layer > CLIP_MAX_LAYERS) { if (err) snprintf(err, errlen, "too many vit layers"); mcpv_free(v); return NULL; }
    for (i = 0; i < v->n_layer; i++)
        if (load_layer(v, (uint32_t)i) != 0) {
            if (err) snprintf(err, errlen, "missing vit layer %u", (unsigned)i);
            mcpv_free(v); return NULL;
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
        {
            uint32_t ps = v->patch, e = v->n_embd, K = 3 * ps * ps, oc, ic, ky, kx, pi;
            v->patch_wf = (float*)ymalloc((size_t)e * K * 4);
            v->pos_f = (float*)ymalloc((size_t)70 * 70 * e * 4);
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
            if (v->pos_f)
                for (pi = 0; pi < 70u * 70u * e; pi++)
                    v->pos_f[pi] = tload(&v->pos_embd, pi);
            v->gemm_nthr = 1;
#ifdef _OPENMP
            v->gemm_nthr = (uint32_t)omp_get_max_threads();
            if (v->gemm_nthr < 1) v->gemm_nthr = 1;
#endif
            v->gemm_in_cap = 65536;
            v->gemm_row = (float*)ymalloc((size_t)v->gemm_nthr * v->gemm_in_cap * 4);
        }
        if (!v->x || !v->res || !v->tmp || !v->q || !v->k || !v->v || !v->attn || !v->ff ||
            !v->patch_wf || !v->pos_f || !v->gemm_row) {
            if (err) snprintf(err, errlen, "oom vis buf");
            mcpv_free(v); return NULL;
        }
    }
    ylog_info("vision: clip %ux%u patch=%u layers=%u embd=%u out=%u insert=%u downsample=%dx mean=%.3f std=%.3f",
              v->image_size, v->image_size, v->patch, v->n_layer, v->n_embd, v->n_out_embd,
              v->insert_lid, v->downsample, (double)v->mean[0], (double)v->std[0]);
    return v;
}

void mcpv_free(Mcpv* v)
{
    if (!v) return;
    free(v->x); free(v->res); free(v->tmp); free(v->q); free(v->k); free(v->v); free(v->attn); free(v->ff);
    free(v->patch_wf); free(v->pos_f); free(v->gemm_row);
    free(v->ts);
    wmap_close(&v->map);
    free(v);
}

int mcpv_n_tokens(const Mcpv* v)
{
    uint32_t g, div;
    if (!v) return 0;
    g = v->image_size / v->patch;
    div = (v->downsample == 4) ? 2u : 4u;
    g /= div;
    return (int)(g * g);
}

int mcpv_hidden(const Mcpv* v)
{
    return v ? (int)v->n_out_embd : 0;
}

int mcpv_apply_opt(Mcpv* v, int downsample, int max_slice, char* err, size_t errlen)
{
    if (!v) return -1;
    if (downsample > 0) {
        if (downsample != 16 && downsample != 4) {
            if (err) snprintf(err, errlen, "downsample_mode want 16x|4x");
            return -1;
        }
        v->downsample = downsample;
        ylog_info("vision minicpmv: downsample=%dx -> %d tokens", v->downsample, mcpv_n_tokens(v));
    }
    if (max_slice > 0 && max_slice != 1) {
        if (err) snprintf(err, errlen, "max_slice_nums=%d not implemented (only 1)", max_slice);
        return -1;
    }
    return 0;
}

static void patch_embed(Mcpv* vis, const float* chw, uint32_t gh, uint32_t gw)
{
    uint32_t ps = vis->patch, e = vis->n_embd, K = 3 * ps * ps;
    uint32_t n = gh * gw, S = vis->image_size, t;
    float* col = vis->ff;
    float* bias = (float*)vis_alloca((size_t)e * 4);
    uint32_t oc;
    for (oc = 0; oc < e; oc++) bias[oc] = tload(&vis->patch_b, oc);
#pragma omp parallel for schedule(static)
    for (t = 0; t < n; t++) {
        uint32_t py = t / gw, px = t % gw, ic, ky, kx, k = 0;
        float* c = col + (size_t)t * K;
        for (ic = 0; ic < 3; ic++)
            for (ky = 0; ky < ps; ky++)
                for (kx = 0; kx < ps; kx++) {
                    uint32_t ix = px * ps + kx, iy = py * ps + ky;
                    c[k++] = chw[(size_t)ic * S * S + iy * S + ix];
                }
    }
    gemm_f32(vis->x, col, vis->patch_wf, bias, n, e, K);
}

static void add_pos(Mcpv* vis, uint32_t gh, uint32_t gw)
{
    uint32_t e = vis->n_embd, i, j, t = 0;
    for (i = 0; i < gh; i++) {
        int bh = (int)floor(70.0 * (double)i / (double)gh);
        for (j = 0; j < gw; j++, t++) {
            int bw = (int)floor(70.0 * (double)j / (double)gw);
            uint32_t row = (uint32_t)(bh * 70 + bw);
            add_inplace(vis->x + (size_t)t * e, vis->pos_f + (size_t)row * e, e);
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

static int encode_448(Mcpv* vis, const float* chw, float* out)
{
    uint32_t gh = vis->image_size / vis->patch, gw = gh;
    uint32_t n = gh * gw, e = vis->n_embd, il, t;
    uint32_t half = gh / 2;
    int is_4x = (vis->downsample == 4);
    patch_embed(vis, chw, gh, gw);
    add_pos(vis, gh, gw);
    if (is_4x) {
        for (il = 0; il < vis->n_layer; il++)
            vit_block(vis, il, n);
    } else {
        for (il = 0; il <= vis->insert_lid && il < vis->n_layer; il++)
            vit_block(vis, il, n);
    }

    if (!is_4x) {
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
            gemm_lin(vis, vis->ff, vis->tmp, &vis->vm_ds_up_w, &vis->vm_ds_up_b, ds, n_ff, e * 4);
            gelu_inplace(vis->ff, ds * n_ff);
            gemm_lin(vis, vis->tmp, vis->ff, &vis->vm_ds_down_w, &vis->vm_ds_down_b, ds, e, n_ff);
        }
        for (t = 0; t < ds * e; t++) vis->x[t] = vis->tmp[t] + vis->res[t];
        n = ds; gh = half; gw = half;
    }

    for (il = vis->insert_lid + 1; il < vis->n_layer; il++)
        vit_block(vis, il, n);
    } /* !is_4x: skip vit window merger */

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
            gemm_lin(vis, vis->ff, vis->tmp, &vis->mm_up_w, &vis->mm_up_b, ds, n_up, e * 4);
            gelu_erf_inplace(vis->ff, ds * n_up);
            gemm_lin(vis, out, vis->ff, &vis->mm_down_w, &vis->mm_down_b, ds, vis->n_out_embd, n_up);
        }
        (void)c;
    }
    return 0;
}

int mcpv_encode_image(Mcpv* v, const char* image_path, float* out, int max_tok, char* err, size_t errlen)
{
    int w = 0, h = 0, c = 0, x, y, ic;
    unsigned char* img;
    float* chw;
    uint32_t S;
    int ntok;
    if (!v || !image_path || !out) { if (err) snprintf(err, errlen, "bad vision args"); return -1; }
    ntok = mcpv_n_tokens(v);
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
    {
        uint64_t t0 = ynow_ms();
        encode_448(v, chw, out);
        ylog_info("vision: encoded %dx%d -> %d tokens hidden=%u downsample=%dx in %.2f s",
                  w, h, ntok, v->n_out_embd, v->downsample, (double)(ynow_ms() - t0) / 1000.0);
    }
    free(chw);
    return ntok;
}
