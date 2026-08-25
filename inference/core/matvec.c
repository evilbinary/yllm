#include "yllm.h"
#include "matvec.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#ifdef _WIN32
#include <malloc.h>
#else
#include <alloca.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
static inline float hsum_avx2(__m256 v)
{
    __m128 low = _mm256_extractf128_ps(v, 0);
    __m128 high = _mm256_extractf128_ps(v, 1);
    low = _mm_add_ps(low, high);
    __m128 shuf = _mm_movehl_ps(low, low);
    __m128 sum = _mm_add_ps(low, shuf);
    shuf = _mm_shuffle_ps(sum, sum, 1);
    sum = _mm_add_ss(sum, shuf);
    return _mm_cvtss_f32(sum);
}
#endif

#if defined(__aarch64__)
#include <arm_neon.h>
static inline int32x4_t neon_dotacc_u8s8(int32x4_t acc, uint8x16_t q, int8x16_t x)
{
#if defined(__ARM_FEATURE_DOTPROD)
    return vdotq_s32(acc, vreinterpretq_s8_u8(q), x);
#else
    int16x8_t p0 = vmull_s8(vreinterpret_s8_u8(vget_low_u8(q)), vget_low_s8(x));
    int16x8_t p1 = vmull_high_s8(vreinterpretq_s8_u8(q), x);
    return vaddq_s32(acc, vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)));
#endif
}

static inline int32x4_t neon_sumacc_s8(int32x4_t acc, int8x16_t x)
{
#if defined(__ARM_FEATURE_DOTPROD)
    return vdotq_s32(acc, vdupq_n_s8(1), x);
#else
    return vaddq_s32(acc, vpaddlq_s16(vpaddlq_s8(x)));
#endif
}

static inline float neon_hsum_f32(float32x4_t v)
{
    return vaddvq_f32(v);
}
#endif

static void gsm_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m)
{
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

#define get_scale_min_k4 gsm_k4

void q4k_block(float* y, const uint8_t* blk, uint32_t stride)
{
    float d = f16_to_f32(((const uint16_t*)blk)[0]);
    float min = f16_to_f32(((const uint16_t*)blk)[1]);
    const uint8_t* qs = blk + 16;
    uint32_t g;
    for (g = 0; g < 8; g++) {
        uint8_t sc, m;
        gsm_k4((int)g, blk + 4, &sc, &m);
        float d1 = d * (float)sc;
        float m1 = min * (float)m;
        const uint8_t* q = qs + (g >> 1) * 32;
        uint32_t e;
        for (e = 0; e < 32; e++) {
            uint8_t nib = q[e];
            nib = (g & 1) ? (nib >> 4) : (nib & 0xF);
            y[(size_t)g * 32 + e] = d1 * (float)nib - m1;
        }
    }
    (void)stride;
}

float q6k_val(const uint8_t* blk, uint32_t e)
{
    float d = f16_to_f32(((const uint16_t*)blk)[104]);
    uint32_t half = e >> 7;
    uint32_t k = e & 0x7f;
    uint32_t quad = k >> 5;
    uint32_t ll = k & 31;
    uint8_t ql = blk[half * 64 + (quad & 1) * 32 + ll];
    uint8_t qh = blk[128 + half * 32 + ll];
    uint32_t bits = ((quad & 2) ? (ql >> 4) : (ql & 0xF)) | (((qh >> (quad * 2)) & 3) << 4);
    int8_t q = (int8_t)bits - 32;
    int8_t s = ((const int8_t*)(blk + 192))[half * 8 + quad * 2 + (ll >> 4)];
    return d * (float)s * (float)q;
}

static const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

static void iq4xs_block(float* y, const uint8_t* blk, float d)
{
    const uint8_t* qs = blk + 2;
    const uint8_t* sc_h = blk + 130;
    const uint8_t* sc_l = blk + 134;
    uint32_t ib;
    for (ib = 0; ib < 8; ib++) {
        int ls = ((sc_l[ib / 2] >> 4 * (ib % 2)) & 0xF) | (((sc_h[ib / 2] >> 2 * (ib % 2)) & 3) << 4);
        float dl = d * (float)(ls - 32);
        uint32_t j;
        for (j = 0; j < 16; j++) {
            y[ib * 32 + j] = dl * (float)kvalues_iq4nl[qs[ib * 16 + j] & 0xF];
            y[ib * 32 + j + 16] = dl * (float)kvalues_iq4nl[qs[ib * 16 + j] >> 4];
        }
    }
}

static void matmul_f32(float* y, const float* x, const uint8_t* w, uint32_t out, uint32_t in)
{
    const float* wp = (const float*)w;
    uint32_t oo;
    #pragma omp parallel for schedule(static)
    for (oo = 0; oo < out; oo++) {
        float acc = 0.0f;
        uint32_t ii;
        for (ii = 0; ii < in; ii++) acc += x[ii] * wp[(size_t)oo * in + ii];
        y[oo] = acc;
    }
}

static void matmul_f16(float* y, const float* x, const uint8_t* w, uint32_t out, uint32_t in)
{
    uint32_t oo;
    const uint16_t* wp = (const uint16_t*)w;
    #pragma omp parallel for schedule(static)
    for (oo = 0; oo < out; oo++) {
        float acc = 0.0f;
        uint32_t ii;
        for (ii = 0; ii < in; ii++) acc += x[ii] * f16_to_f32(wp[(size_t)oo * in + ii]);
        y[oo] = acc;
    }
}

void embed_q4k(float* y, const uint8_t* w, uint32_t row, uint32_t hidden)
{
    uint32_t nb = hidden / 256;
    const uint8_t* r = w + (size_t)row * nb * 144;
    uint32_t b;
    for (b = 0; b < nb; b++) {
        q4k_block(y + (size_t)b * 256, r + (size_t)b * 144, 0);
    }
}

void embed_q6k(float* y, const uint8_t* w, uint32_t row, uint32_t hidden)
{
    uint32_t nb = hidden / 256;
    const uint8_t* r = w + (size_t)row * nb * 210;
    uint32_t b;
    for (b = 0; b < nb; b++) {
        const uint8_t* blk = r + (size_t)b * 210;
        uint32_t e;
        for (e = 0; e < 256; e++) y[b * 256 + e] = q6k_val(blk, e);
    }
}

void embed_f32(float* y, const uint8_t* w, uint32_t row, uint32_t hidden)
{
    const float* wp = (const float*)w + (size_t)row * hidden;
    memcpy(y, wp, (size_t)hidden * 4);
}

void rmsnorm(float* y, const float* x, const uint8_t* w, uint32_t n, float eps, uint32_t dtype)
{
    float s = 0.0f;
    uint32_t i = 0;
#if defined(__aarch64__)
    float32x4_t sv = vdupq_n_f32(0.0f);
    for (; i + 16 <= n; i += 16) {
        float32x4_t v0 = vld1q_f32(x + i);
        float32x4_t v1 = vld1q_f32(x + i + 4);
        float32x4_t v2 = vld1q_f32(x + i + 8);
        float32x4_t v3 = vld1q_f32(x + i + 12);
        sv = vfmaq_f32(sv, v0, v0);
        sv = vfmaq_f32(sv, v1, v1);
        sv = vfmaq_f32(sv, v2, v2);
        sv = vfmaq_f32(sv, v3, v3);
    }
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        sv = vfmaq_f32(sv, v, v);
    }
    s = neon_hsum_f32(sv);
#endif
    for (; i < n; i++) s += x[i] * x[i];
    float inv = 1.0f / sqrtf(s / (float)n + eps);
    i = 0;
    if (dtype == DT_F32) {
        const float* wp = (const float*)w;
#if defined(__aarch64__)
        float32x4_t vinv = vdupq_n_f32(inv);
        for (; i + 16 <= n; i += 16) {
            vst1q_f32(y + i, vmulq_f32(vmulq_f32(vld1q_f32(x + i), vld1q_f32(wp + i)), vinv));
            vst1q_f32(y + i + 4, vmulq_f32(vmulq_f32(vld1q_f32(x + i + 4), vld1q_f32(wp + i + 4)), vinv));
            vst1q_f32(y + i + 8, vmulq_f32(vmulq_f32(vld1q_f32(x + i + 8), vld1q_f32(wp + i + 8)), vinv));
            vst1q_f32(y + i + 12, vmulq_f32(vmulq_f32(vld1q_f32(x + i + 12), vld1q_f32(wp + i + 12)), vinv));
        }
        for (; i + 4 <= n; i += 4)
            vst1q_f32(y + i, vmulq_f32(vmulq_f32(vld1q_f32(x + i), vld1q_f32(wp + i)), vinv));
#endif
        for (; i < n; i++) y[i] = x[i] * wp[i] * inv;
    } else {
        const uint16_t* wp = (const uint16_t*)w;
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        float32x4_t vinv = vdupq_n_f32(inv);
        for (; i + 8 <= n; i += 8) {
            float16x8_t h = vld1q_f16((const __fp16*)(wp + i));
            float32x4_t w0 = vcvt_f32_f16(vget_low_f16(h));
            float32x4_t w1 = vcvt_f32_f16(vget_high_f16(h));
            vst1q_f32(y + i, vmulq_f32(vmulq_f32(vld1q_f32(x + i), w0), vinv));
            vst1q_f32(y + i + 4, vmulq_f32(vmulq_f32(vld1q_f32(x + i + 4), w1), vinv));
        }
#endif
        for (; i < n; i++) y[i] = x[i] * f16_to_f32(wp[i]) * inv;
    }
}

/* Q6_K 块反量化到 tmp[256](批量路径用, 无逐元素函数调用) */
static void q6k_block(float* y, const uint8_t* blk)
{
    float d = f16_to_f32(((const uint16_t*)blk)[104]);
    uint32_t half, e;
    for (half = 0; half < 2; half++) {
        const uint8_t* ql = blk + half * 64;
        const uint8_t* qh = blk + 128 + half * 32;
        const int8_t* sc = (const int8_t*)(blk + 192) + half * 8;
        for (e = 0; e < 128; e++) {
            uint32_t quad = e >> 5;
            uint32_t ll = e & 31;
            uint8_t qb = ql[(quad & 1) * 32 + ll];
            uint8_t bits = (uint8_t)(((quad & 2) ? (qb >> 4) : (qb & 0xF)) | (((qh[ll] >> (quad * 2)) & 3) << 4));
            int8_t q = (int8_t)bits - 32;
            int8_t s = sc[quad * 2 + (ll >> 4)];
            y[half * 128 + e] = d * (float)s * (float)q;
        }
    }
}

/* Q5_K 块反量化到 y[256] (176 B/256). 布局: d(f16,0) dmin(f16,2) scales(4,12B)
   qh(16,32B) qs(48,128B). 每 64 元素一轮, 用 2 个 6-bit scale/min. */
static void q5k_block(float* y, const uint8_t* blk)
{
    const float d = f16_to_f32(((const uint16_t*)blk)[0]);
    const float min = f16_to_f32(((const uint16_t*)blk)[1]);
    const uint8_t* sc = blk + 4;
    const uint8_t* qh = blk + 16;
    const uint8_t* qs = blk + 48;
    uint32_t is = 0;
    uint32_t u1 = 1, u2 = 2;
    uint32_t r;
    for (r = 0; r < 4; r++) {
        uint8_t dsc, dm;
        float d1, m1, d2, m2;
        if (is + 0 < 4) { dsc = sc[is + 0] & 63; dm = sc[is + 4] & 63; }
        else { dsc = (sc[is + 4] & 0xF) | ((sc[is - 4] >> 6) << 4); dm = (sc[is + 4] >> 4) | ((sc[is] >> 6) << 4); }
        d1 = d * (float)dsc; m1 = min * (float)dm;
        if (is + 1 < 4) { dsc = sc[is + 1] & 63; dm = sc[is + 5] & 63; }
        else { dsc = (sc[is + 5] & 0xF) | ((sc[is - 3] >> 6) << 4); dm = (sc[is + 5] >> 4) | ((sc[is + 1] >> 6) << 4); }
        d2 = d * (float)dsc; m2 = min * (float)dm;
        uint32_t l;
        for (l = 0; l < 32; l++) {
            float qv1 = (float)((qs[l] & 0xF) + ((qh[l] & u1) ? 16 : 0));
            float qv2 = (float)((qs[l] >> 4) + ((qh[l] & u2) ? 16 : 0));
            y[r * 64 + l]     = d1 * qv1 - m1;
            y[r * 64 + l + 32] = d2 * qv2 - m2;
        }
        qs += 32; is += 2; u1 <<= 2; u2 <<= 2;
    }
}

/* Q8_K 量化激活(与 llama.cpp vec_dot 一致):
 * 每 256 元素 block: d = amax/127, q = clamp(round(x/d)) 还原为 float */
static void q8k_quant(const float* x, float* xq, uint32_t n)
{
    uint32_t nb = n / 256;
    uint32_t b, i;
    for (b = 0; b < nb; b++) {
        const float* xb = x + (size_t)b * 256;
        float* xqb = xq + (size_t)b * 256;
        float amax = 0.0f;
        for (i = 0; i < 256; i++) { float a = fabsf(xb[i]); if (a > amax) amax = a; }
        float d = amax / 127.0f;
        if (d <= 0.0f) { memset(xqb, 0, 256 * 4); continue; }
        float inv = 1.0f / d;
        for (i = 0; i < 256; i++) {
            float q = roundf(xb[i] * inv);
            if (q > 127.0f) q = 127.0f;
            else if (q < -128.0f) q = -128.0f;
            xqb[i] = q * d;
        }
    }
}

void matvec_q8k_quant(const float* x, float* xq, uint32_t n)
{
    q8k_quant(x, xq, n);
}

/* int8 量化 x + 每 256 元素块缩放(供 int8 maddubs 点积) */
static void q8k_quant_i8(const float* x, int8_t* xq, float* xs, uint32_t n)
{
    uint32_t nb = n / 256;
    uint32_t b, i;
    for (b = 0; b < nb; b++) {
        const float* xb = x + (size_t)b * 256;
        int8_t* xqb = xq + (size_t)b * 256;
        float amax = 0.0f;
#if defined(__aarch64__)
        {
            float32x4_t am = vdupq_n_f32(0.0f);
            for (i = 0; i < 256; i += 16) {
                am = vmaxq_f32(am, vabsq_f32(vld1q_f32(xb + i)));
                am = vmaxq_f32(am, vabsq_f32(vld1q_f32(xb + i + 4)));
                am = vmaxq_f32(am, vabsq_f32(vld1q_f32(xb + i + 8)));
                am = vmaxq_f32(am, vabsq_f32(vld1q_f32(xb + i + 12)));
            }
            amax = vmaxvq_f32(am);
        }
#else
        for (i = 0; i < 256; i++) { float a = fabsf(xb[i]); if (a > amax) amax = a; }
#endif
        float d = amax / 127.0f;
        xs[b] = d;
        if (d <= 0.0f) { memset(xqb, 0, 256); continue; }
        float inv = 1.0f / d;
#if defined(__aarch64__)
        {
            float32x4_t vinv = vdupq_n_f32(inv);
            int32x4_t lo = vdupq_n_s32(-128);
            int32x4_t hi = vdupq_n_s32(127);
            for (i = 0; i < 256; i += 16) {
                int32x4_t q0 = vcvtq_s32_f32(vrndnq_f32(vmulq_f32(vld1q_f32(xb + i), vinv)));
                int32x4_t q1 = vcvtq_s32_f32(vrndnq_f32(vmulq_f32(vld1q_f32(xb + i + 4), vinv)));
                int32x4_t q2 = vcvtq_s32_f32(vrndnq_f32(vmulq_f32(vld1q_f32(xb + i + 8), vinv)));
                int32x4_t q3 = vcvtq_s32_f32(vrndnq_f32(vmulq_f32(vld1q_f32(xb + i + 12), vinv)));
                q0 = vmaxq_s32(vminq_s32(q0, hi), lo);
                q1 = vmaxq_s32(vminq_s32(q1, hi), lo);
                q2 = vmaxq_s32(vminq_s32(q2, hi), lo);
                q3 = vmaxq_s32(vminq_s32(q3, hi), lo);
                int16x8_t p0 = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
                int16x8_t p1 = vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3));
                vst1q_s8(xqb + i, vcombine_s8(vqmovn_s16(p0), vqmovn_s16(p1)));
            }
        }
#else
        for (i = 0; i < 256; i++) {
            float q = roundf(xb[i] * inv);
            if (q > 127.0f) q = 127.0f;
            else if (q < -128.0f) q = -128.0f;
            xqb[i] = (int8_t)q;
        }
#endif
    }
}

void matmul_iq4xs(float* y, const float* x, const uint8_t* w, uint32_t out, uint32_t in)
{
    uint32_t nb = in / 256;
    uint32_t rowb = nb * 144;
    float* xq = (float*)alloca((size_t)in * 4);
    q8k_quant(x, xq, in);
    uint32_t oo;
    #pragma omp parallel for schedule(static)
    for (oo = 0; oo < out; oo++) {
        const uint8_t* row = w + (size_t)oo * rowb;
        float acc = 0.0f;
        float tmp[256];  /* 每线程私有, 避免多线程共享 tmp 数据竞争 */
        uint32_t b;
        for (b = 0; b < nb; b++) {
            float d = f16_to_f32(((const uint16_t*)row)[0]);
            iq4xs_block(tmp, row + (size_t)b * 144, d);
            const float* xb = xq + (size_t)b * 256;
            uint32_t e;
            for (e = 0; e < 256; e++) acc += xb[e] * tmp[e];
        }
        y[oo] = acc;
    }
}

void embed_iq4xs(float* y, const uint8_t* w, uint32_t row, uint32_t hidden)
{
    uint32_t nb = hidden / 256;
    const uint8_t* r = w + (size_t)row * nb * 144;
    uint32_t b;
    for (b = 0; b < nb; b++) {
        float d = f16_to_f32(((const uint16_t*)r)[0]);
        iq4xs_block(y + (size_t)b * 256, r + (size_t)b * 144, d);
    }
}

void matmul_q5k(float* y, const float* x, const uint8_t* w, uint32_t out, uint32_t in)
{
    uint32_t nb = in / 256;
    uint32_t rowb = nb * 176;
    float* xq = (float*)alloca((size_t)in * 4);
    q8k_quant(x, xq, in);
#ifdef __AVX2__
    #pragma omp parallel for schedule(static)
    for (uint32_t oo = 0; oo < out; oo++) {
        const uint8_t* row = w + (size_t)oo * rowb;
        __m256 acc_v = _mm256_setzero_ps();
        uint32_t b;
        for (b = 0; b < nb; b++) {
            const uint8_t* blk = row + (size_t)b * 176;
            const float* xp = xq + (size_t)b * 256;
            float d    = f16_to_f32(((const uint16_t*)blk)[0]);
            float dmin = f16_to_f32(((const uint16_t*)blk)[1]);
            const uint8_t* scp = blk + 4;
            const uint8_t* qh = blk + 16;
            const uint8_t* qs = blk + 48;
            if (b + 1 < nb) _mm_prefetch((const char*)(blk + 176), _MM_HINT_T0);
            uint32_t r;
            for (r = 0; r < 4; r++) {
                uint32_t is = 2 * r;
                uint8_t dsc, dm;
                get_scale_min_k4(is, scp, &dsc, &dm);
                float d1 = d * (float)dsc, m1 = dmin * (float)dm;
                get_scale_min_k4(is + 1, scp, &dsc, &dm);
                float d2 = d * (float)dsc, m2 = dmin * (float)dm;
                uint8_t bit1 = (uint8_t)(1u << (2 * r));
                uint8_t bit2 = (uint8_t)(2u << (2 * r));
                __m256 d1v = _mm256_set1_ps(d1), m1v = _mm256_set1_ps(m1);
                __m256 d2v = _mm256_set1_ps(d2), m2v = _mm256_set1_ps(m2);
                __m128i b1m = _mm_set1_epi8((char)bit1);
                __m128i b2m = _mm_set1_epi8((char)bit2);
                __m128i s16 = _mm_set1_epi8(16);
                uint32_t l;
                for (l = 0; l < 32; l += 8) {
                    __m128i qs8 = _mm_loadu_si128((const __m128i*)(qs + l));
                    __m128i qh8 = _mm_loadu_si128((const __m128i*)(qh + l));
                    /* qv1 = (qs&0xF) + (bit1 set ? 16 : 0), qv2 = (qs>>4) + (bit2 ? 16:0) */
                    __m128i lo = _mm_and_si128(qs8, _mm_set1_epi8(0x0F));
                    __m128i hi = _mm_and_si128(_mm_srli_epi16(qs8, 4), _mm_set1_epi8(0x0F));
                    __m128i b1 = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(qh8, b1m), b1m), s16);
                    __m128i b2 = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(qh8, b2m), b2m), s16);
                    __m128i qv1_8 = _mm_add_epi8(lo, b1);
                    __m128i qv2_8 = _mm_add_epi8(hi, b2);
                    /* xp 布局: qv1 -> r*64+l, qv2 -> r*64+32+l */
                    __m256 x1 = _mm256_loadu_ps(xp + r * 64 + l);
                    __m256 x2 = _mm256_loadu_ps(xp + r * 64 + 32 + l);
                    __m256 q1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(qv1_8));
                    __m256 q2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(qv2_8));
                    acc_v = _mm256_fmadd_ps(d1v, _mm256_mul_ps(q1, x1), acc_v);
                    acc_v = _mm256_fnmadd_ps(m1v, x1, acc_v);
                    acc_v = _mm256_fmadd_ps(d2v, _mm256_mul_ps(q2, x2), acc_v);
                    acc_v = _mm256_fnmadd_ps(m2v, x2, acc_v);
                }
                qs += 32;  /* 每 64 元素轮换 qs 偏移(与 q5k_block/llama.cpp 一致) */
            }
        }
        y[oo] = hsum_avx2(acc_v);
    }
#else
    uint32_t oo;
    #pragma omp parallel for schedule(static)
    for (oo = 0; oo < out; oo++) {
        const uint8_t* row = w + (size_t)oo * rowb;
        float acc = 0.0f;
        float tmp[256];  /* 每线程私有, 避免多线程共享 tmp 数据竞争 */
        uint32_t b;
        for (b = 0; b < nb; b++) {
            q5k_block(tmp, row + (size_t)b * 176);
            const float* xb = xq + (size_t)b * 256;
            uint32_t e;
            for (e = 0; e < 256; e++) acc += xb[e] * tmp[e];
        }
        y[oo] = acc;
    }
#endif
}

void embed_q5k(float* y, const uint8_t* w, uint32_t row, uint32_t hidden)
{
    uint32_t nb = hidden / 256;
    const uint8_t* r = w + (size_t)row * nb * 176;
    uint32_t b;
    for (b = 0; b < nb; b++) {
        q5k_block(y + (size_t)b * 256, r + (size_t)b * 176);
    }
}

int dequant_mat_f16(uint16_t* dst, const uint8_t* w, uint32_t out, uint32_t in, uint32_t dtype)
{
    float* row = (float*)malloc((size_t)in * sizeof(float));
    if (!row) return -1;
    uint32_t oo;
    uint32_t nb = in / 256;
    for (oo = 0; oo < out; oo++) {
        uint16_t* drow = dst + (size_t)oo * in;
        if (dtype == DT_F16) {
            memcpy(drow, (const uint16_t*)w + (size_t)oo * in, (size_t)in * 2);
            continue;
        }
        if (dtype == DT_F32) {
            f32_to_f16_buf((const float*)w + (size_t)oo * in, drow, in);
            continue;
        }
        if (dtype == DT_BF16) {
            uint32_t i;
            const uint16_t* bp = (const uint16_t*)w + (size_t)oo * in;
            for (i = 0; i < in; i++) row[i] = f16_to_f32(bf16_to_f16(bp[i]));
            f32_to_f16_buf(row, drow, in);
            continue;
        }
        if (in % 256 != 0) { free(row); return -1; }
        uint32_t b;
        if (dtype == DT_Q4K) {
            const uint8_t* r = w + (size_t)oo * nb * 144;
            for (b = 0; b < nb; b++) q4k_block(row + (size_t)b * 256, r + (size_t)b * 144, 0);
        } else if (dtype == DT_Q6K) {
            const uint8_t* r = w + (size_t)oo * nb * 210;
            for (b = 0; b < nb; b++) q6k_block(row + (size_t)b * 256, r + (size_t)b * 210);
        } else if (dtype == DT_Q5K) {
            const uint8_t* r = w + (size_t)oo * nb * 176;
            for (b = 0; b < nb; b++) q5k_block(row + (size_t)b * 256, r + (size_t)b * 176);
        } else if (dtype == DT_IQ4XS) {
            const uint8_t* r = w + (size_t)oo * nb * 144;
            for (b = 0; b < nb; b++) {
                const uint8_t* blk = r + (size_t)b * 144;
                iq4xs_block(row + (size_t)b * 256, blk, f16_to_f32(((const uint16_t*)blk)[0]));
            }
        } else {
            free(row);
            return -1;
        }
        f32_to_f16_buf(row, drow, in);
    }
    free(row);
    return 0;
}

int dequant_mat_f32(float* dst, const uint8_t* w, uint32_t out, uint32_t in, uint32_t dtype)
{
    uint32_t oo;
    uint32_t nb = in / 256;
    for (oo = 0; oo < out; oo++) {
        float* drow = dst + (size_t)oo * in;
        if (dtype == DT_F32) {
            memcpy(drow, (const float*)w + (size_t)oo * in, (size_t)in * 4);
            continue;
        }
        if (dtype == DT_F16) {
            const uint16_t* wp = (const uint16_t*)w + (size_t)oo * in;
            uint32_t i;
            for (i = 0; i < in; i++) drow[i] = f16_to_f32(wp[i]);
            continue;
        }
        if (dtype == DT_BF16) {
            const uint16_t* bp = (const uint16_t*)w + (size_t)oo * in;
            uint32_t i;
            for (i = 0; i < in; i++) drow[i] = f16_to_f32(bf16_to_f16(bp[i]));
            continue;
        }
        if (in % 256 != 0) return -1;
        uint32_t b;
        if (dtype == DT_Q4K) {
            const uint8_t* r = w + (size_t)oo * nb * 144;
            for (b = 0; b < nb; b++) q4k_block(drow + (size_t)b * 256, r + (size_t)b * 144, 0);
        } else if (dtype == DT_Q6K) {
            const uint8_t* r = w + (size_t)oo * nb * 210;
            for (b = 0; b < nb; b++) q6k_block(drow + (size_t)b * 256, r + (size_t)b * 210);
        } else if (dtype == DT_Q5K) {
            const uint8_t* r = w + (size_t)oo * nb * 176;
            for (b = 0; b < nb; b++) q5k_block(drow + (size_t)b * 256, r + (size_t)b * 176);
        } else if (dtype == DT_IQ4XS) {
            const uint8_t* r = w + (size_t)oo * nb * 144;
            for (b = 0; b < nb; b++) {
                const uint8_t* blk = r + (size_t)b * 144;
                iq4xs_block(drow + (size_t)b * 256, blk, f16_to_f32(((const uint16_t*)blk)[0]));
            }
        } else {
            return -1;
        }
    }
    return 0;
}

void matmul_f32_t(float* y, const float* x, const uint8_t* w, uint32_t in, uint32_t out)
{
    const float* wp = (const float*)w;
    uint32_t oo;
    #pragma omp parallel for schedule(static)
    for (oo = 0; oo < out; oo++) {
        float acc = 0.0f;
        uint32_t ii;
        for (ii = 0; ii < in; ii++) acc += x[ii] * wp[(size_t)oo * in + ii];
        y[oo] = acc;
    }
}

void matmul_f16_t(float* y, const float* x, const uint8_t* w, uint32_t in, uint32_t out)
{
    const uint16_t* wp = (const uint16_t*)w;
    uint32_t oo;
    #pragma omp parallel for schedule(static)
    for (oo = 0; oo < out; oo++) {
        float acc = 0.0f;
        uint32_t ii;
        for (ii = 0; ii < in; ii++) acc += x[ii] * f16_to_f32(wp[(size_t)oo * in + ii]);
        y[oo] = acc;
    }
}

void matmul_q4k(float* y, const float* x, const uint8_t* w, uint32_t out, uint32_t in)
{
    uint32_t nb = in / 256;
    uint32_t rowb = nb * 144;
    uint32_t oo;
#ifdef __AVX2__
    int8_t* xq8 = (int8_t*)alloca((size_t)in);
    float* xs = (float*)alloca((size_t)nb * 4);
    q8k_quant_i8(x, xq8, xs, in);
    const __m256i ones_u8 = _mm256_set1_epi8(1);
    const __m256i ones_i16 = _mm256_set1_epi16(1);
    const __m128i lowmask = _mm_set1_epi8(0x0F);
    /* 预计算每组 Σxq 的 8 个 4 元素部分和(仅依赖 x, 全部输出行共享):
     * 与行内 maddubs(ones,xv)→madd 的逐车道结果逐位一致, 只是上移一次。 */
    float* sxg = (float*)alloca((size_t)nb * 8 * 8 * 4);
    for (uint32_t b = 0; b < nb; b++) {
        const int8_t* xp = xq8 + (size_t)b * 256;
        for (uint32_t g = 0; g < 8; g++) {
            __m256i xv = _mm256_loadu_si256((const __m256i*)(xp + g * 32));
            __m256i s32 = _mm256_madd_epi16(_mm256_maddubs_epi16(ones_u8, xv), ones_i16);
            _mm256_storeu_ps(sxg + ((size_t)b * 8 + g) * 8, _mm256_cvtepi32_ps(s32));
        }
    }
    #pragma omp parallel for schedule(static)
    for (oo = 0; oo < out; oo++) {
        const uint8_t* row = w + (size_t)oo * rowb;
        __m256 acc_v = _mm256_setzero_ps();
        uint32_t b;
        for (b = 0; b < nb; b++) {
            const uint8_t* blk = row + (size_t)b * 144;
            const int8_t* xp = xq8 + (size_t)b * 256;
            const uint8_t* q = blk + 16;
            const uint8_t* scp = blk + 4;
            if (b + 1 < nb) _mm_prefetch((const char*)(blk + 144), _MM_HINT_T0);
#ifdef __F16C__
            __m128i hb = _mm_loadl_epi64((const __m128i*)blk);
            __m128 fd = _mm_cvtph_ps(hb);
            float d = _mm_cvtss_f32(fd);
            float dmin = _mm_cvtss_f32(_mm_shuffle_ps(fd, fd, _MM_SHUFFLE(1, 1, 1, 1)));
#else
            float d    = f16_to_f32(((const uint16_t*)blk)[0]);
            float dmin = f16_to_f32(((const uint16_t*)blk)[1]);
#endif
            float dsc = d * xs[b];      /* 折叠块内 x 的 int8 缩放 */
            float msc = dmin * xs[b];

            /* 展开解 8 组 scale/min(无分支, 替代逐组 gsm_k4 调用) */
            uint8_t sc8[8], mn8[8];
            sc8[0] = scp[0] & 63;  mn8[0] = scp[4] & 63;
            sc8[1] = scp[1] & 63;  mn8[1] = scp[5] & 63;
            sc8[2] = scp[2] & 63;  mn8[2] = scp[6] & 63;
            sc8[3] = scp[3] & 63;  mn8[3] = scp[7] & 63;
            sc8[4] = (scp[8] & 15) | ((scp[0] >> 6) << 4);
            mn8[4] = (scp[8] >> 4) | ((scp[4] >> 6) << 4);
            sc8[5] = (scp[9] & 15) | ((scp[1] >> 6) << 4);
            mn8[5] = (scp[9] >> 4) | ((scp[5] >> 6) << 4);
            sc8[6] = (scp[10] & 15) | ((scp[2] >> 6) << 4);
            mn8[6] = (scp[10] >> 4) | ((scp[6] >> 6) << 4);
            sc8[7] = (scp[11] & 15) | ((scp[3] >> 6) << 4);
            mn8[7] = (scp[11] >> 4) | ((scp[7] >> 6) << 4);
            float d1[8], m1[8];
            for (int ii = 0; ii < 8; ii++) { d1[ii] = dsc * sc8[ii]; m1[ii] = msc * mn8[ii]; }

            int is = 0;
            for (int j = 0; j < 4; j++) {
                /* 32B qs: byte k 的低 nibble → x[k](组 is), 高 nibble → x[k+32](组 is+1) */
                __m128i q0 = _mm_loadu_si128((const __m128i*)(q));
                __m128i q1 = _mm_loadu_si128((const __m128i*)(q + 16));
                __m128i lo0 = _mm_and_si128(q0, lowmask);
                __m128i hi0 = _mm_and_si128(_mm_srli_epi16(q0, 4), lowmask);
                __m128i lo1 = _mm_and_si128(q1, lowmask);
                __m128i hi1 = _mm_and_si128(_mm_srli_epi16(q1, 4), lowmask);
                __m256i qv1 = _mm256_inserti128_si256(_mm256_castsi128_si256(lo0), lo1, 1);
                __m256i qv2 = _mm256_inserti128_si256(_mm256_castsi128_si256(hi0), hi1, 1);

                __m256i xv1 = _mm256_loadu_si256((const __m256i*)(xp));
                __m256i xv2 = _mm256_loadu_si256((const __m256i*)(xp + 32));
                __m256i p1 = _mm256_maddubs_epi16(qv1, xv1);   /* u8×s8 → 16×s16 */
                __m256i p2 = _mm256_maddubs_epi16(qv2, xv2);

                __m256 dot1 = _mm256_cvtepi32_ps(_mm256_madd_epi16(p1, ones_i16));
                __m256 dot2 = _mm256_cvtepi32_ps(_mm256_madd_epi16(p2, ones_i16));
                __m256 sx1 = _mm256_loadu_ps(sxg + ((size_t)b * 8 + is) * 8);
                __m256 sx2 = _mm256_loadu_ps(sxg + ((size_t)b * 8 + is + 1) * 8);

                acc_v = _mm256_fmadd_ps(dot1, _mm256_set1_ps(d1[is]), acc_v);
                acc_v = _mm256_fnmadd_ps(sx1, _mm256_set1_ps(m1[is]), acc_v);
                acc_v = _mm256_fmadd_ps(dot2, _mm256_set1_ps(d1[is + 1]), acc_v);
                acc_v = _mm256_fnmadd_ps(sx2, _mm256_set1_ps(m1[is + 1]), acc_v);

                xp += 64;
                q  += 32;
                is += 2;
            }
        }
        float acc = hsum_avx2(acc_v);
        y[oo] = acc;
    }
#elif defined(__aarch64__)
    /* decode 热路径: 与 AVX2 对齐 — 激活 int8 量化一次, 预计算各组 Σxq,
     * 双输出行共用 x 向量(减带宽/ALU)。 */
    int8_t* xq8 = (int8_t*)alloca((size_t)in);
    float* xs = (float*)alloca((size_t)nb * 4);
    float* sxg = (float*)alloca((size_t)nb * 8 * 4);
    q8k_quant_i8(x, xq8, xs, in);
    {
        uint32_t b, g;
        for (b = 0; b < nb; b++) {
            const int8_t* xp = xq8 + (size_t)b * 256;
            for (g = 0; g < 8; g++) {
                int32x4_t z = vdupq_n_s32(0);
                z = neon_sumacc_s8(z, vld1q_s8(xp + g * 32));
                z = neon_sumacc_s8(z, vld1q_s8(xp + g * 32 + 16));
                sxg[(size_t)b * 8 + g] = (float)vaddvq_s32(z);
            }
        }
    }
    const uint8x16_t nibble = vdupq_n_u8(0x0F);
    uint32_t n2 = out & ~1u;
    #pragma omp parallel for schedule(static)
    for (oo = 0; oo < n2; oo += 2) {
        const uint8_t* row0 = w + (size_t)oo * rowb;
        const uint8_t* row1 = row0 + rowb;
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        float c0 = 0.0f, c1 = 0.0f;
        uint32_t b;
        if (oo + 3 < out) {
            __builtin_prefetch(row1 + rowb, 0, 3);
            __builtin_prefetch(row1 + 2 * rowb, 0, 3);
        }
        for (b = 0; b < nb; b++) {
            const uint8_t* blk0 = row0 + (size_t)b * 144;
            const uint8_t* blk1 = row1 + (size_t)b * 144;
            const int8_t* xp = xq8 + (size_t)b * 256;
            const float* sxb = sxg + (size_t)b * 8;
            const uint8_t* q0 = blk0 + 16;
            const uint8_t* q1 = blk1 + 16;
            float d0, dm0, d1, dm1;
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
            {
                float16x4_t h0 = vld1_f16((const __fp16*)blk0);
                float16x4_t h1 = vld1_f16((const __fp16*)blk1);
                float32x4_t f0 = vcvt_f32_f16(h0);
                float32x4_t f1 = vcvt_f32_f16(h1);
                d0 = vgetq_lane_f32(f0, 0); dm0 = vgetq_lane_f32(f0, 1);
                d1 = vgetq_lane_f32(f1, 0); dm1 = vgetq_lane_f32(f1, 1);
            }
#else
            d0 = f16_to_f32(((const uint16_t*)blk0)[0]);
            dm0 = f16_to_f32(((const uint16_t*)blk0)[1]);
            d1 = f16_to_f32(((const uint16_t*)blk1)[0]);
            dm1 = f16_to_f32(((const uint16_t*)blk1)[1]);
#endif
            float dsc0 = d0 * xs[b], msc0 = dm0 * xs[b];
            float dsc1 = d1 * xs[b], msc1 = dm1 * xs[b];
            float ds0[8], ms0[8], ds1[8], ms1[8];
            {
                const uint8_t* s0 = blk0 + 4;
                const uint8_t* s1 = blk1 + 4;
                int ii;
                uint8_t sc0[8], mn0[8], sc1[8], mn1[8];
                sc0[0] = s0[0] & 63; mn0[0] = s0[4] & 63;
                sc0[1] = s0[1] & 63; mn0[1] = s0[5] & 63;
                sc0[2] = s0[2] & 63; mn0[2] = s0[6] & 63;
                sc0[3] = s0[3] & 63; mn0[3] = s0[7] & 63;
                sc0[4] = (s0[8] & 15) | ((s0[0] >> 6) << 4);
                mn0[4] = (s0[8] >> 4) | ((s0[4] >> 6) << 4);
                sc0[5] = (s0[9] & 15) | ((s0[1] >> 6) << 4);
                mn0[5] = (s0[9] >> 4) | ((s0[5] >> 6) << 4);
                sc0[6] = (s0[10] & 15) | ((s0[2] >> 6) << 4);
                mn0[6] = (s0[10] >> 4) | ((s0[6] >> 6) << 4);
                sc0[7] = (s0[11] & 15) | ((s0[3] >> 6) << 4);
                mn0[7] = (s0[11] >> 4) | ((s0[7] >> 6) << 4);
                sc1[0] = s1[0] & 63; mn1[0] = s1[4] & 63;
                sc1[1] = s1[1] & 63; mn1[1] = s1[5] & 63;
                sc1[2] = s1[2] & 63; mn1[2] = s1[6] & 63;
                sc1[3] = s1[3] & 63; mn1[3] = s1[7] & 63;
                sc1[4] = (s1[8] & 15) | ((s1[0] >> 6) << 4);
                mn1[4] = (s1[8] >> 4) | ((s1[4] >> 6) << 4);
                sc1[5] = (s1[9] & 15) | ((s1[1] >> 6) << 4);
                mn1[5] = (s1[9] >> 4) | ((s1[5] >> 6) << 4);
                sc1[6] = (s1[10] & 15) | ((s1[2] >> 6) << 4);
                mn1[6] = (s1[10] >> 4) | ((s1[6] >> 6) << 4);
                sc1[7] = (s1[11] & 15) | ((s1[3] >> 6) << 4);
                mn1[7] = (s1[11] >> 4) | ((s1[7] >> 6) << 4);
                for (ii = 0; ii < 8; ii++) {
                    ds0[ii] = dsc0 * sc0[ii]; ms0[ii] = msc0 * mn0[ii];
                    ds1[ii] = dsc1 * sc1[ii]; ms1[ii] = msc1 * mn1[ii];
                }
            }
            int is = 0;
            int j;
            for (j = 0; j < 4; j++) {
                int8x16_t x0 = vld1q_s8(xp);
                int8x16_t x1 = vld1q_s8(xp + 16);
                int8x16_t x2 = vld1q_s8(xp + 32);
                int8x16_t x3 = vld1q_s8(xp + 48);
                uint8x16_t qa0 = vld1q_u8(q0);
                uint8x16_t qa1 = vld1q_u8(q0 + 16);
                uint8x16_t qb0 = vld1q_u8(q1);
                uint8x16_t qb1 = vld1q_u8(q1 + 16);
                uint8x16_t lo_a0 = vandq_u8(qa0, nibble);
                uint8x16_t lo_a1 = vandq_u8(qa1, nibble);
                uint8x16_t hi_a0 = vshrq_n_u8(qa0, 4);
                uint8x16_t hi_a1 = vshrq_n_u8(qa1, 4);
                uint8x16_t lo_b0 = vandq_u8(qb0, nibble);
                uint8x16_t lo_b1 = vandq_u8(qb1, nibble);
                uint8x16_t hi_b0 = vshrq_n_u8(qb0, 4);
                uint8x16_t hi_b1 = vshrq_n_u8(qb1, 4);
                int32x4_t z = vdupq_n_s32(0);
                int32x4_t dlo0 = neon_dotacc_u8s8(neon_dotacc_u8s8(z, lo_a0, x0), lo_a1, x1);
                int32x4_t dhi0 = neon_dotacc_u8s8(neon_dotacc_u8s8(z, hi_a0, x2), hi_a1, x3);
                int32x4_t dlo1 = neon_dotacc_u8s8(neon_dotacc_u8s8(z, lo_b0, x0), lo_b1, x1);
                int32x4_t dhi1 = neon_dotacc_u8s8(neon_dotacc_u8s8(z, hi_b0, x2), hi_b1, x3);
                acc0 = vfmaq_n_f32(acc0, vcvtq_f32_s32(dlo0), ds0[is]);
                acc0 = vfmaq_n_f32(acc0, vcvtq_f32_s32(dhi0), ds0[is + 1]);
                acc1 = vfmaq_n_f32(acc1, vcvtq_f32_s32(dlo1), ds1[is]);
                acc1 = vfmaq_n_f32(acc1, vcvtq_f32_s32(dhi1), ds1[is + 1]);
                /* min 修正是标量 Σxq；不可 vdup 到 4 lane 再 hsum(会×4) */
                c0 -= sxb[is] * ms0[is] + sxb[is + 1] * ms0[is + 1];
                c1 -= sxb[is] * ms1[is] + sxb[is + 1] * ms1[is + 1];
                xp += 64;
                q0 += 32;
                q1 += 32;
                is += 2;
            }
        }
        y[oo] = neon_hsum_f32(acc0) + c0;
        y[oo + 1] = neon_hsum_f32(acc1) + c1;
    }
    if (out & 1u) {
        oo = out - 1u;
        const uint8_t* row = w + (size_t)oo * rowb;
        float32x4_t acc = vdupq_n_f32(0.0f);
        float c = 0.0f;
        uint32_t b;
        for (b = 0; b < nb; b++) {
            const uint8_t* blk = row + (size_t)b * 144;
            const int8_t* xp = xq8 + (size_t)b * 256;
            const float* sxb = sxg + (size_t)b * 8;
            const uint8_t* q = blk + 16;
            const uint8_t* scp = blk + 4;
            float d = f16_to_f32(((const uint16_t*)blk)[0]);
            float dmin = f16_to_f32(((const uint16_t*)blk)[1]);
            float dsc = d * xs[b], msc = dmin * xs[b];
            float ds[8], ms[8];
            uint8_t sc8[8], mn8[8];
            int ii, j, is = 0;
            sc8[0] = scp[0] & 63; mn8[0] = scp[4] & 63;
            sc8[1] = scp[1] & 63; mn8[1] = scp[5] & 63;
            sc8[2] = scp[2] & 63; mn8[2] = scp[6] & 63;
            sc8[3] = scp[3] & 63; mn8[3] = scp[7] & 63;
            sc8[4] = (scp[8] & 15) | ((scp[0] >> 6) << 4);
            mn8[4] = (scp[8] >> 4) | ((scp[4] >> 6) << 4);
            sc8[5] = (scp[9] & 15) | ((scp[1] >> 6) << 4);
            mn8[5] = (scp[9] >> 4) | ((scp[5] >> 6) << 4);
            sc8[6] = (scp[10] & 15) | ((scp[2] >> 6) << 4);
            mn8[6] = (scp[10] >> 4) | ((scp[6] >> 6) << 4);
            sc8[7] = (scp[11] & 15) | ((scp[3] >> 6) << 4);
            mn8[7] = (scp[11] >> 4) | ((scp[7] >> 6) << 4);
            for (ii = 0; ii < 8; ii++) { ds[ii] = dsc * sc8[ii]; ms[ii] = msc * mn8[ii]; }
            for (j = 0; j < 4; j++) {
                uint8x16_t qq0 = vld1q_u8(q);
                uint8x16_t qq1 = vld1q_u8(q + 16);
                int8x16_t x0 = vld1q_s8(xp);
                int8x16_t x1 = vld1q_s8(xp + 16);
                int8x16_t x2 = vld1q_s8(xp + 32);
                int8x16_t x3 = vld1q_s8(xp + 48);
                int32x4_t z = vdupq_n_s32(0);
                int32x4_t dlo = neon_dotacc_u8s8(
                    neon_dotacc_u8s8(z, vandq_u8(qq0, nibble), x0), vandq_u8(qq1, nibble), x1);
                int32x4_t dhi = neon_dotacc_u8s8(
                    neon_dotacc_u8s8(z, vshrq_n_u8(qq0, 4), x2), vshrq_n_u8(qq1, 4), x3);
                acc = vfmaq_n_f32(acc, vcvtq_f32_s32(dlo), ds[is]);
                acc = vfmaq_n_f32(acc, vcvtq_f32_s32(dhi), ds[is + 1]);
                c -= sxb[is] * ms[is] + sxb[is + 1] * ms[is + 1];
                xp += 64;
                q += 32;
                is += 2;
            }
        }
        y[oo] = neon_hsum_f32(acc) + c;
    }
#else
    float* xq = (float*)alloca((size_t)in * 4);
    q8k_quant(x, xq, in);
    #pragma omp parallel for schedule(static)
    for (oo = 0; oo < out; oo++) {
        const uint8_t* row = w + (size_t)oo * rowb;
        float acc = 0.0f;
        uint32_t b;
        for (b = 0; b < nb; b++) {
            const uint8_t* blk = row + (size_t)b * 144;
            const float* xb = xq + (size_t)b * 256;
            float d = f16_to_f32(((const uint16_t*)blk)[0]);
            float min = f16_to_f32(((const uint16_t*)blk)[1]);
            const uint8_t* qs = blk + 16;
            /* 分开累加 qx 与 x,组末才做 d*sum_qx - min*sum_x(每元素少一次 mul/sub) */
            float sq0 = 0, sx0 = 0, sq1 = 0, sx1 = 0;
            float sq2 = 0, sx2 = 0, sq3 = 0, sx3 = 0;
            uint32_t g, e;
            for (g = 0; g < 8; g += 4) {
                uint8_t sc, m;
                float dg[4], mg[4];
                uint32_t gg;
                for (gg = 0; gg < 4; gg++) {
                    gsm_k4((int)(g + gg), blk + 4, &sc, &m);
                    dg[gg] = d * (float)sc;
                    mg[gg] = min * (float)m;
                }
                const uint8_t* qa = qs + (g >> 1) * 32;
                const uint8_t* qb = qa + 32;
                for (e = 0; e < 32; e++) {
                    uint8_t ba = qa[e], bb = qb[e];
                    float x0 = xb[g * 32 + e];
                    float x1 = xb[g * 32 + e + 32];
                    float x2 = xb[g * 32 + e + 64];
                    float x3 = xb[g * 32 + e + 96];
                    sq0 += (float)(ba & 0xF) * x0; sx0 += x0;
                    sq1 += (float)(ba >> 4) * x1; sx1 += x1;
                    sq2 += (float)(bb & 0xF) * x2; sx2 += x2;
                    sq3 += (float)(bb >> 4) * x3; sx3 += x3;
                }
                acc += dg[0] * sq0 - mg[0] * sx0
                     + dg[1] * sq1 - mg[1] * sx1
                     + dg[2] * sq2 - mg[2] * sx2
                     + dg[3] * sq3 - mg[3] * sx3;
                sq0 = sx0 = sq1 = sx1 = sq2 = sx2 = sq3 = sx3 = 0.0f;
            }
        }
        y[oo] = acc;
    }
#endif
}

void matmul_q6k(float* y, const float* x, const uint8_t* w, uint32_t out, uint32_t in)
{
    uint32_t nb = in / 256;
    uint32_t rowb = nb * 210;
    uint32_t oo;
    float* xq = (float*)alloca((size_t)in * 4);
    q8k_quant(x, xq, in);
#ifdef __AVX2__
    #pragma omp parallel for schedule(static)
    for (oo = 0; oo < out; oo++) {
        const uint8_t* row = w + (size_t)oo * rowb;
        float acc = 0.0f;
        uint32_t b;
        for (b = 0; b < nb; b++) {
            const uint8_t* blk = row + (size_t)b * 210;
            const float* xb = xq + (size_t)b * 256;
            float d = f16_to_f32(((const uint16_t*)blk)[104]);
            const uint8_t* ql = blk;
            const uint8_t* qh = blk + 128;
            const int8_t* sc = (const int8_t*)(blk + 192);
            float sums[16];
            memset(sums, 0, sizeof(sums));
            for (int chunk = 0; chunk < 2; chunk++) {
                const uint8_t* ql_c = ql + chunk * 64;
                const uint8_t* qh_c = qh + chunk * 32;
                const float* xp_c = xb + chunk * 128;
                int is = chunk * 8;
                __m128i ql0 = _mm_loadu_si128((const __m128i*)ql_c);
                __m128i ql1 = _mm_loadu_si128((const __m128i*)(ql_c + 16));
                __m128i ql2 = _mm_loadu_si128((const __m128i*)(ql_c + 32));
                __m128i ql3 = _mm_loadu_si128((const __m128i*)(ql_c + 48));
                __m128i qh0 = _mm_loadu_si128((const __m128i*)qh_c);
                __m128i qh1 = _mm_loadu_si128((const __m128i*)(qh_c + 16));
                __m128i mF = _mm_set1_epi8(0x0F);
                __m128i m3 = _mm_set1_epi8(3);
                /* even groups */
                __m128i e_q1 = _mm_or_si128(_mm_and_si128(ql0, mF), _mm_slli_epi16(_mm_and_si128(qh0, m3), 4));
                __m128i e_q2 = _mm_or_si128(_mm_and_si128(ql2, mF), _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh0, 2), m3), 4));
                __m128i e_q3 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(ql0, 4), mF), _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh0, 4), m3), 4));
                __m128i e_q4 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(ql2, 4), mF), _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh0, 6), m3), 4));
                /* odd groups */
                __m128i o_q1 = _mm_or_si128(_mm_and_si128(ql1, mF), _mm_slli_epi16(_mm_and_si128(qh1, m3), 4));
                __m128i o_q2 = _mm_or_si128(_mm_and_si128(ql3, mF), _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh1, 2), m3), 4));
                __m128i o_q3 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(ql1, 4), mF), _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh1, 4), m3), 4));
                __m128i o_q4 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(ql3, 4), mF), _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh1, 6), m3), 4));
#define Q6_GROUP_SUM(qvals, xpos) ({ \
    __m256i _e16 = _mm256_cvtepu8_epi16(qvals); \
    __m128i _l = _mm256_castsi256_si128(_e16); \
    __m128i _h = _mm256_extracti128_si256(_e16, 1); \
    __m256 _fl = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_l)), _mm256_set1_ps(32.0f)); \
    __m256 _fh = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_h)), _mm256_set1_ps(32.0f)); \
    __m256 _p = _mm256_fmadd_ps(_fl, _mm256_loadu_ps(xp_c + (xpos)), \
               _mm256_mul_ps(_fh, _mm256_loadu_ps(xp_c + (xpos) + 8))); \
    hsum_avx2(_p); \
})
                sums[is+0] = Q6_GROUP_SUM(e_q1, 0);
                sums[is+2] = Q6_GROUP_SUM(e_q2, 32);
                sums[is+4] = Q6_GROUP_SUM(e_q3, 64);
                sums[is+6] = Q6_GROUP_SUM(e_q4, 96);
                sums[is+1] = Q6_GROUP_SUM(o_q1, 16);
                sums[is+3] = Q6_GROUP_SUM(o_q2, 48);
                sums[is+5] = Q6_GROUP_SUM(o_q3, 80);
                sums[is+7] = Q6_GROUP_SUM(o_q4, 112);
#undef Q6_GROUP_SUM
            }
            for (int j = 0; j < 16; j++) acc += d * (float)sc[j] * sums[j];
        }
        y[oo] = acc;
    }
#else
    #pragma omp parallel for schedule(static)
    for (oo = 0; oo < out; oo++) {
        const uint8_t* row = w + (size_t)oo * rowb;
        float acc = 0.0f;
        uint32_t b;
        for (b = 0; b < nb; b++) {
            const uint8_t* blk = row + (size_t)b * 210;
            const float* xb = xq + (size_t)b * 256;
            float d = f16_to_f32(((const uint16_t*)blk)[104]);
            /* 每次 3 个 load 解出 4 个 6-bit 权重(无 per-element 索引运算) */
            const uint8_t* ql = blk;
            const uint8_t* qh = blk + 128;
            const int8_t* sc = (const int8_t*)(blk + 192);
            float sums[16];
            memset(sums, 0, sizeof(sums));
            int chunk;
            for (chunk = 0; chunk < 2; chunk++) {
                int is = chunk * 8;
                const uint8_t* ql_c = ql + chunk * 64;
                const uint8_t* qh_c = qh + chunk * 32;
                const float* xp_c = xb + chunk * 128;
                int l;
                for (l = 0; l < 16; l++) {
                    int q1 = (int)((ql_c[l] & 0xF) | (((qh_c[l] >> 0) & 3) << 4)) - 32;
                    int q2 = (int)((ql_c[l + 32] & 0xF) | (((qh_c[l] >> 2) & 3) << 4)) - 32;
                    int q3 = (int)((ql_c[l] >> 4) | (((qh_c[l] >> 4) & 3) << 4)) - 32;
                    int q4 = (int)((ql_c[l + 32] >> 4) | (((qh_c[l] >> 6) & 3) << 4)) - 32;
                    sums[is + 0] += (float)q1 * xp_c[l];
                    sums[is + 2] += (float)q2 * xp_c[l + 32];
                    sums[is + 4] += (float)q3 * xp_c[l + 64];
                    sums[is + 6] += (float)q4 * xp_c[l + 96];
                }
                for (l = 16; l < 32; l++) {
                    int q1 = (int)((ql_c[l] & 0xF) | (((qh_c[l] >> 0) & 3) << 4)) - 32;
                    int q2 = (int)((ql_c[l + 32] & 0xF) | (((qh_c[l] >> 2) & 3) << 4)) - 32;
                    int q3 = (int)((ql_c[l] >> 4) | (((qh_c[l] >> 4) & 3) << 4)) - 32;
                    int q4 = (int)((ql_c[l + 32] >> 4) | (((qh_c[l] >> 6) & 3) << 4)) - 32;
                    sums[is + 1] += (float)q1 * xp_c[l];
                    sums[is + 3] += (float)q2 * xp_c[l + 32];
                    sums[is + 5] += (float)q3 * xp_c[l + 64];
                    sums[is + 7] += (float)q4 * xp_c[l + 96];
                }
            }
            int j;
            for (j = 0; j < 16; j++) acc += d * (float)sc[j] * sums[j];
        }
        y[oo] = acc;
    }
#endif
}

void matmul(float* y, const float* x, const uint8_t* w, uint32_t out, uint32_t in, uint32_t dtype)
{
    switch (dtype) {
    case DT_F32: matmul_f32(y, x, w, out, in); break;
    case DT_Q4K: matmul_q4k(y, x, w, out, in); break;
    case DT_Q6K: matmul_q6k(y, x, w, out, in); break;
    case DT_Q5K: matmul_q5k(y, x, w, out, in); break;
    case DT_IQ4XS: matmul_iq4xs(y, x, w, out, in); break;
    default: matmul_f16(y, x, w, out, in); break;
    }
}

void matmul_rows(float* y, const float* x, const uint8_t* w,
                 uint32_t row_begin, uint32_t n_rows, uint32_t in, uint32_t out, uint32_t dtype)
{
    (void)out;
    switch (dtype) {
    case DT_Q4K:   matmul_q4k(y, x, w + (size_t)row_begin * ((size_t)(in / 256) * 144), n_rows, in); break;
    case DT_Q6K:   matmul_q6k(y, x, w + (size_t)row_begin * ((size_t)(in / 256) * 210), n_rows, in); break;
    case DT_IQ4XS: matmul_iq4xs(y, x, w + (size_t)row_begin * ((size_t)(in / 256) * 144), n_rows, in); break;
    default:       matmul(y, x, w, n_rows, in, dtype); break;
    }
}

size_t matmul_row_bytes(uint32_t dtype, uint32_t in)
{
    switch (dtype) {
    case DT_Q4K:
    case DT_IQ4XS: return (size_t)(in / 256) * 144;
    case DT_Q6K:   return (size_t)(in / 256) * 210;
    default:       return 0;   /* F16/F32 列主序, 行不连续, 不支持行分块 */
    }
}

/* ---- 批量 matmul(批量 prefill 用): y[B×out] = x[B×in] · W^T
 * 每个输出行 oo: 同时算 B 个 token 的点积, 权重行只读一次(每 4 token 一组 SIMD)。 */
/* ---- 批量 matmul(批量 prefill 用): y[B×out] = x[B×in] · W^T
 * 每个 256 元素块只反量化一次(tmp[256]), 权重对整批 token 共享(内存带宽 ÷B)。
 * Q4K/Q6K/IQ4XS 用各自的反量化; f32/f16 直接拷贝。AVX2 构建用 SIMD 点积,
 * 基础构建用标量点积(编译器 -O2 自动向量化)。 */
void matmul_batch(float* y, const float* x, const uint8_t* w, uint32_t out, uint32_t in,
                  uint32_t dtype, uint32_t B)
{
    uint32_t nb = in / 256;
    uint32_t oo;
    const uint32_t blk = (dtype == DT_Q6K) ? 210 : (dtype == DT_Q5K) ? 176 : 144;
    if (dtype == DT_F32 || dtype == DT_F16) {
        /* f32/f16: 逐 token 点积 */
        #pragma omp parallel for schedule(static)
        for (oo = 0; oo < out; oo++) {
            uint32_t g, i;
            for (g = 0; g < B; g++) {
                const float* xg = x + (size_t)g * in;
                float acc = 0.0f;
                if (dtype == DT_F32) {
                    const float* wp = (const float*)w + (size_t)oo * in;
                    for (i = 0; i < in; i++) acc += xg[i] * wp[i];
                } else {
                    const uint16_t* wp = (const uint16_t*)w + (size_t)oo * in;
                    for (i = 0; i < in; i++) acc += xg[i] * f16_to_f32(wp[i]);
                }
                y[(size_t)g * out + oo] = acc;
            }
        }
        return;
    }
    /* 量化 dtype(Q4K/Q6K/IQ4XS): 块反量化共享 + 批量点积 */
    #pragma omp parallel for schedule(static)
    for (oo = 0; oo < out; oo++) {
        const uint8_t* row = w + (size_t)oo * nb * blk;
        float acc[64];     /* B ≤ 64, 栈上避免每行 malloc/free */
        float tmp[256];   /* 每线程独立, 避免共享数组踩踏 */
        uint32_t b, g;
        for (g = 0; g < B; g++) acc[g] = 0.0f;
        for (b = 0; b < nb; b++) {
            const uint8_t* bp = row + (size_t)b * blk;
            if (dtype == DT_Q4K) q4k_block(tmp, bp, 0);
            else if (dtype == DT_Q6K) q6k_block(tmp, bp);
            else if (dtype == DT_Q5K) q5k_block(tmp, bp);
            else iq4xs_block(tmp, bp, f16_to_f32(((const uint16_t*)bp)[0]));
            for (g = 0; g < B; g++) {
                const float* xg = x + (size_t)g * in + (size_t)b * 256;
#ifdef __AVX2__
                __m256 sv = _mm256_setzero_ps();
                uint32_t i;
                for (i = 0; i < 256; i += 8)
                    sv = _mm256_fmadd_ps(_mm256_loadu_ps(tmp + i), _mm256_loadu_ps(xg + i), sv);
                acc[g] += hsum_avx2(sv);
#else
                uint32_t i;
                for (i = 0; i < 256; i++) acc[g] += tmp[i] * xg[i];
#endif
            }
        }
        for (g = 0; g < B; g++) y[(size_t)g * out + oo] = acc[g];
    }
}

void embed_f16(float* y, const uint8_t* w, uint32_t row, uint32_t hidden)
{
    const uint16_t* wp = (const uint16_t*)w + (size_t)row * hidden;
    uint32_t i;
    for (i = 0; i < hidden; i++) y[i] = f16_to_f32(wp[i]);
}

void rope_inplace(float* v, uint32_t d, uint32_t pos, float theta)
{
    uint32_t half = d / 2;
    uint32_t j;
    for (j = 0; j < half; j++) {
        float freq = powf(theta, -2.0f * (float)j / (float)d);
        float ang = freq * (float)pos;
        float c = cosf(ang);
        float s = sinf(ang);
        float a = v[2 * j];
        float b = v[2 * j + 1];
        v[2 * j] = a * c - b * s;
        v[2 * j + 1] = a * s + b * c;
    }
}

void rope_inplace_qwen_ff(float* v, uint32_t d, uint32_t pos, float theta, const float* freq_factors)
{
    uint32_t half = d / 2;
    uint32_t j;
    for (j = 0; j < half; j++) {
        float freq = powf(theta, -2.0f * (float)j / (float)d);
        float ff = 1.0f;
        if (freq_factors) {
            ff = freq_factors[j];
            if (!(ff > 0.0f)) ff = 1.0f;
        }
        float ang = freq * (float)pos / ff;
        float c = cosf(ang);
        float s = sinf(ang);
        float a = v[j];
        float b = v[j + half];
        v[j] = a * c - b * s;
        v[j + half] = a * s + b * c;
    }
}

void rope_inplace_neox_if(float* v, uint32_t d, uint32_t pos, const float* inv_freq)
{
    uint32_t half, j;
    float pf;
    if (pos == 0 || !inv_freq) return;
    half = d / 2;
    pf = (float)pos;
    for (j = 0; j < half; j++) {
        float ang = inv_freq[j] * pf;
        float c = cosf(ang);
        float s = sinf(ang);
        float a = v[j];
        float b = v[j + half];
        v[j] = a * c - b * s;
        v[j + half] = a * s + b * c;
    }
}

/* qwen2 interleaved RoPE: v[i] 与 v[i+half] 配对(llama 是 v[2j]/v[2j+1]) */
void rope_inplace_qwen(float* v, uint32_t d, uint32_t pos, float theta)
{
    rope_inplace_qwen_ff(v, d, pos, theta, NULL);
}

/* M-RoPE(qwen35 纯文本): 只作用前 n_dims 维, interleaved pair (v[i], v[i+n_dims/2]) */
void rope_inplace_mrope(float* v, uint32_t head_dim, uint32_t n_dims, uint32_t pos, float theta)
{
    uint32_t n_offset = n_dims / 2;  /* 32 */
    uint32_t pairs = n_dims / 2;     /* 32 */
    uint32_t j;
    for (j = 0; j < pairs; j++) {
        float freq = powf(theta, -2.0f * (float)j / (float)n_dims);
        float ang = freq * (float)pos;
        float c = cosf(ang);
        float s = sinf(ang);
        float a = v[j];
        float b = v[j + n_offset];
        v[j] = a * c - b * s;
        v[j + n_offset] = a * s + b * c;
    }
}

/* L2 归一化(就地): y = x / max(sqrt(Σx²), eps) */
void l2norm_inplace(float* v, uint32_t n, float eps)
{
    float s = 0.0f;
    uint32_t i;
    for (i = 0; i < n; i++) s += v[i] * v[i];
    float scale = 1.0f / fmaxf(sqrtf(s), eps);
    for (i = 0; i < n; i++) v[i] *= scale;
}

/* conv1d 更新(qwen35 GDN, 因果无 bias):
 * state 为 [kwidth × dim] 延迟线(末行 = 当前输入 x);
 * 先把 x 滑入末行, 再输出 out[ch] = Σ_i state[i*dim+ch] × w[ch*kwidth+i] 写回 x。
 * 注意: gguf 权重布局为 [dim, kwidth](ne0=kwidth 最内层, 每 channel 的 taps 连续)。
 * x 已被拷入 state 末行, 故可安全就地写回。 */
void conv1d_update(float* state, const float* x, const uint8_t* w, uint32_t dim, uint32_t kwidth)
{
    uint32_t i, ch;
    memmove(state, state + dim, (size_t)(kwidth - 1) * dim * 4);
    memcpy(state + (size_t)(kwidth - 1) * dim, x, (size_t)dim * 4);
    const float* wf = (const float*)w;
    for (ch = 0; ch < dim; ch++) {
        float acc = 0.0f;
        for (i = 0; i < kwidth; i++)
            acc += state[(size_t)i * dim + ch] * wf[(size_t)ch * kwidth + i];
        ((float*)x)[ch] = acc;
    }
}

/* GDN 递归状态更新 + 注意力输出(qwen35):
 * 每 v_head h 复用 k_head = h % num_k_heads 的 q/k;
 * S 存储转置 S[j*hv+i] = Smat[i][j];
 * g=exp(gate[h]); S*=g; delta[j]=(v[j]-dot(S列j,k))*beta[h]; S列j+=k*delta[j];
 * out[j] = dot(S列j,q) * (1/sqrt(hv))。 */
void gdn_state_update(float* S, float* out, const float* q, const float* k,
                      const float* v, const float* gate, const float* beta,
                      uint32_t num_k_heads, uint32_t hk, uint32_t n_vheads, uint32_t hv)
{
    const float scale = 1.0f / sqrtf((float)hv);
    uint32_t h, j, ii;
    for (h = 0; h < n_vheads; h++) {
        uint32_t kh = h % num_k_heads;
        const float* qh = q + (size_t)kh * hk;
        const float* khv = k + (size_t)kh * hk;
        const float* vh = v + (size_t)h * hv;
        float* Sh = S + (size_t)h * hk * hv;
        float* oh = out + (size_t)h * hv;
        float g = expf(gate[h]);
        float b = beta[h];
        for (j = 0; j < hv; j++) {
            float* Sj = Sh + (size_t)j * hv;
            float dot = 0.0f, del;
            for (ii = 0; ii < hk; ii++) { Sj[ii] *= g; dot += Sj[ii] * khv[ii]; }
            del = (vh[j] - dot) * b;
            for (ii = 0; ii < hk; ii++) Sj[ii] += khv[ii] * del;
        }
        for (j = 0; j < hv; j++) {
            const float* Sj = Sh + (size_t)j * hv;
            float dot = 0.0f;
            for (ii = 0; ii < hk; ii++) dot += Sj[ii] * qh[ii];
            oh[j] = dot * scale;
        }
    }
}

/* rmsnorm_gated: y = rmsnorm(x, w) * silu(z), 逐维; y 可与 x 同缓冲 */
void rmsnorm_gated(float* y, const float* x, const uint8_t* w, const float* z,
                   uint32_t n, float eps, uint32_t dtype)
{
    float s = 0.0f;
    uint32_t i;
    for (i = 0; i < n; i++) s += x[i] * x[i];
    float inv = 1.0f / sqrtf(s / (float)n + eps);
    const float* wf = (dtype == DT_F32) ? (const float*)w : NULL;
    const uint16_t* wh = (dtype == DT_F32) ? NULL : (const uint16_t*)w;
    for (i = 0; i < n; i++) {
        float wgt = wf ? wf[i] : f16_to_f32(wh[i]);
        float zg = z[i];
        y[i] = x[i] * wgt * inv * (zg / (1.0f + expf(-zg)));
    }
}

void softmax(float* v, uint32_t n)
{
    float m = v[0];
    uint32_t i;
    for (i = 1; i < n; i++) if (v[i] > m) m = v[i];
    float s = 0.0f;
    for (i = 0; i < n; i++) { v[i] = expf(v[i] - m); s += v[i]; }
    for (i = 0; i < n; i++) v[i] /= s;
}

void swiglu(float* y, const float* gate, const float* up, uint32_t n)
{
    uint32_t i;
    for (i = 0; i < n; i++) {
        float g = gate[i];
        y[i] = g / (1.0f + expf(-g)) * up[i];
    }
}

/* GeGLU: y[i] = gelu(gate[i]) * up[i]  (gemma4 FFN 激活) */
static float gelu_approx(float x)
{
    /* tanh approx: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3))) */
    float c = 0.7978845608f; /* sqrt(2/pi) */
    float inner = c * (x + 0.044715f * x * x * x);
    float t = tanhf(inner);
    return 0.5f * x * (1.0f + t);
}

void geglu(float* y, const float* gate, const float* up, uint32_t n)
{
    uint32_t i;
    for (i = 0; i < n; i++)
        y[i] = gelu_approx(gate[i]) * up[i];
}

void gelu_inplace(float* y, uint32_t n)
{
    uint32_t i;
    for (i = 0; i < n; i++)
        y[i] = gelu_approx(y[i]);
}

void rmsnorm_unit(float* y, const float* x, uint32_t n, float eps)
{
    float s = 0.0f;
    uint32_t i;
    for (i = 0; i < n; i++) s += x[i] * x[i];
    float inv = 1.0f / sqrtf(s / (float)n + eps);
    for (i = 0; i < n; i++) y[i] = x[i] * inv;
}

void add_inplace(float* y, const float* x, uint32_t n)
{
    uint32_t i = 0;
#if defined(__aarch64__)
    for (; i + 16 <= n; i += 16) {
        vst1q_f32(y + i, vaddq_f32(vld1q_f32(y + i), vld1q_f32(x + i)));
        vst1q_f32(y + i + 4, vaddq_f32(vld1q_f32(y + i + 4), vld1q_f32(x + i + 4)));
        vst1q_f32(y + i + 8, vaddq_f32(vld1q_f32(y + i + 8), vld1q_f32(x + i + 8)));
        vst1q_f32(y + i + 12, vaddq_f32(vld1q_f32(y + i + 12), vld1q_f32(x + i + 12)));
    }
    for (; i + 4 <= n; i += 4)
        vst1q_f32(y + i, vaddq_f32(vld1q_f32(y + i), vld1q_f32(x + i)));
#endif
    for (; i < n; i++) y[i] += x[i];
}

float vec_dot_f32_f16(const float* a, const uint16_t* b, uint32_t n)
{
    uint32_t i = 0;
    float s = 0.0f;
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (; i + 8 <= n; i += 8) {
        float16x8_t h = vld1q_f16((const __fp16*)(b + i));
        acc = vfmaq_f32(acc, vld1q_f32(a + i), vcvt_f32_f16(vget_low_f16(h)));
        acc = vfmaq_f32(acc, vld1q_f32(a + i + 4), vcvt_f32_f16(vget_high_f16(h)));
    }
    s = neon_hsum_f32(acc);
#endif
    for (; i < n; i++) s += a[i] * f16_to_f32(b[i]);
    return s;
}

void vec_axpy_f16(float* y, const uint16_t* x, float a, uint32_t n)
{
    uint32_t i = 0;
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    float32x4_t va = vdupq_n_f32(a);
    for (; i + 8 <= n; i += 8) {
        float16x8_t h = vld1q_f16((const __fp16*)(x + i));
        vst1q_f32(y + i, vfmaq_f32(vld1q_f32(y + i), vcvt_f32_f16(vget_low_f16(h)), va));
        vst1q_f32(y + i + 4, vfmaq_f32(vld1q_f32(y + i + 4), vcvt_f32_f16(vget_high_f16(h)), va));
    }
#endif
    for (; i < n; i++) y[i] += a * f16_to_f32(x[i]);
}
