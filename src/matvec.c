#include "yllm.h"
#include "matvec.h"
#include <math.h>
#include <string.h>

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
    uint32_t i;
    for (i = 0; i < n; i++) s += x[i] * x[i];
    float inv = 1.0f / sqrtf(s / (float)n + eps);
    if (dtype == DT_F32) {
        const float* wp = (const float*)w;
        for (i = 0; i < n; i++) y[i] = x[i] * wp[i] * inv;
    } else {
        const uint16_t* wp = (const uint16_t*)w;
        for (i = 0; i < n; i++) y[i] = x[i] * f16_to_f32(wp[i]) * inv;
    }
}

void matmul_iq4xs(float* y, const float* x, const uint8_t* w, uint32_t out, uint32_t in)
{
    uint32_t nb = in / 256;
    uint32_t rowb = nb * 144;
    float tmp[256];
    uint32_t oo;
    for (oo = 0; oo < out; oo++) {
        const uint8_t* row = w + (size_t)oo * rowb;
        float acc = 0.0f;
        uint32_t b;
        for (b = 0; b < nb; b++) {
            float d = f16_to_f32(((const uint16_t*)row)[0]);
            iq4xs_block(tmp, row + (size_t)b * 144, d);
            const float* xb = x + (size_t)b * 256;
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

void matmul_f32_t(float* y, const float* x, const uint8_t* w, uint32_t in, uint32_t out)
{
    const float* wp = (const float*)w;
    uint32_t oo;
    for (oo = 0; oo < out; oo++) {
        float acc = 0.0f;
        uint32_t ii;
        for (ii = 0; ii < in; ii++) acc += x[ii] * wp[(size_t)ii * out + oo];
        y[oo] = acc;
    }
}

void matmul_q4k(float* y, const float* x, const uint8_t* w, uint32_t out, uint32_t in)
{
    uint32_t nb = in / 256;
    uint32_t rowb = nb * 144;
    uint32_t oo;
#ifdef __AVX2__
    for (oo = 0; oo < out; oo++) {
        const uint8_t* row = w + (size_t)oo * rowb;
        float acc = 0.0f;
        uint32_t b;
        for (b = 0; b < nb; b++) {
            const uint8_t* blk = row + (size_t)b * 144;
            const float* xb = x + (size_t)b * 256;
            float d = f16_to_f32(((const uint16_t*)blk)[0]);
            float dmin = f16_to_f32(((const uint16_t*)blk)[1]);
            const uint8_t* q = blk + 16;
            const uint8_t* scp = blk + 4;
            int is = 0;
            for (int j = 0; j < 4; j++) {
                uint8_t sc, mn;
                gsm_k4(is, scp, &sc, &mn);
                float d1 = d * (float)sc;
                float m1 = dmin * (float)mn;
                gsm_k4(is + 1, scp, &sc, &mn);
                float d2 = d * (float)sc;
                float m2 = dmin * (float)mn;

                __m256 sum_qx1 = _mm256_setzero_ps();
                __m256 sum_x1 = _mm256_setzero_ps();
                __m256 sum_qx2 = _mm256_setzero_ps();
                __m256 sum_x2 = _mm256_setzero_ps();
                for (int l = 0; l < 32; l += 16) {
                    /* 16 bytes: bytes 0..15 -> low nibbles for x[l..l+15],
                       high nibbles for x[l+32..l+47] */
                    __m128i q16 = _mm_loadu_si128((const __m128i*)(q + l));
                    __m128i lo = _mm_and_si128(q16, _mm_set1_epi8(0x0F));
                    __m128i hi = _mm_and_si128(_mm_srli_epi16(q16, 4), _mm_set1_epi8(0x0F));
                    __m256i lo16 = _mm256_cvtepu8_epi16(lo);
                    __m256 lo_f0 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(lo16)));
                    __m256 lo_f1 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(lo16, 1)));
                    __m256 x_l0 = _mm256_loadu_ps(xb + l);
                    __m256 x_l1 = _mm256_loadu_ps(xb + l + 8);
                    sum_qx1 = _mm256_fmadd_ps(lo_f0, x_l0, sum_qx1);
                    sum_qx1 = _mm256_fmadd_ps(lo_f1, x_l1, sum_qx1);
                    sum_x1 = _mm256_add_ps(sum_x1, _mm256_add_ps(x_l0, x_l1));

                    __m256i hi16 = _mm256_cvtepu8_epi16(hi);
                    __m256 hi_f0 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(hi16)));
                    __m256 hi_f1 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(hi16, 1)));
                    __m256 x_h0 = _mm256_loadu_ps(xb + l + 32);
                    __m256 x_h1 = _mm256_loadu_ps(xb + l + 40);
                    sum_qx2 = _mm256_fmadd_ps(hi_f0, x_h0, sum_qx2);
                    sum_qx2 = _mm256_fmadd_ps(hi_f1, x_h1, sum_qx2);
                    sum_x2 = _mm256_add_ps(sum_x2, _mm256_add_ps(x_h0, x_h1));
                }
                acc += d1 * hsum_avx2(sum_qx1) - m1 * hsum_avx2(sum_x1)
                     + d2 * hsum_avx2(sum_qx2) - m2 * hsum_avx2(sum_x2);
                q += 32;
                is += 2;
            }
        }
        y[oo] = acc;
    }
#else
    for (oo = 0; oo < out; oo++) {
        const uint8_t* row = w + (size_t)oo * rowb;
        float acc = 0.0f;
        uint32_t b;
        for (b = 0; b < nb; b++) {
            const uint8_t* blk = row + (size_t)b * 144;
            const float* xb = x + (size_t)b * 256;
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
                    acc += xb[(size_t)g * 32 + e] * (d1 * (float)nib - m1);
                }
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
#ifdef __AVX2__
    for (oo = 0; oo < out; oo++) {
        const uint8_t* row = w + (size_t)oo * rowb;
        float acc = 0.0f;
        uint32_t b;
        for (b = 0; b < nb; b++) {
            const uint8_t* blk = row + (size_t)b * 210;
            const float* xb = x + (size_t)b * 256;
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
    for (oo = 0; oo < out; oo++) {
        const uint8_t* row = w + (size_t)oo * rowb;
        float acc = 0.0f;
        uint32_t b;
        for (b = 0; b < nb; b++) {
            const uint8_t* blk = row + (size_t)b * 210;
            const float* xb = x + (size_t)b * 256;
            float d = f16_to_f32(((const uint16_t*)blk)[104]);
            uint32_t e;
            for (e = 0; e < 256; e++) {
                uint32_t half = e >> 7;
                uint32_t k = e & 0x7f;
                uint32_t quad = k >> 5;
                uint32_t ll = k & 31;
                uint8_t ql = blk[half * 64 + (quad & 1) * 32 + ll];
                uint8_t qh = blk[128 + half * 32 + ll];
                uint32_t bits = ((quad & 2) ? (ql >> 4) : (ql & 0xF)) | (((qh >> (quad * 2)) & 3) << 4);
                int8_t q = (int8_t)bits - 32;
                int8_t s = ((const int8_t*)(blk + 192))[half * 8 + quad * 2 + (ll >> 4)];
                acc += xb[e] * d * (float)s * (float)q;
            }
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
    case DT_IQ4XS: matmul_iq4xs(y, x, w, out, in); break;
    default: matmul_f16(y, x, w, out, in); break;
    }
}

void matmul_f16_t(float* y, const float* x, const uint8_t* w, uint32_t in, uint32_t out)
{
    uint32_t oo;
    const uint16_t* wp = (const uint16_t*)w;
    for (oo = 0; oo < out; oo++) {
        float acc = 0.0f;
        uint32_t ii;
        for (ii = 0; ii < in; ii++) acc += x[ii] * f16_to_f32(wp[(size_t)ii * out + oo]);
        y[oo] = acc;
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
