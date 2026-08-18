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

void matmul_iq4xs(float* y, const float* x, const uint8_t* w, uint32_t out, uint32_t in)
{
    uint32_t nb = in / 256;
    uint32_t rowb = nb * 144;
    float tmp[256];
    uint32_t oo;
    #pragma omp parallel for schedule(static)
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
    #pragma omp parallel for schedule(static)
    for (oo = 0; oo < out; oo++) {
        float acc = 0.0f;
        uint32_t ii;
        for (ii = 0; ii < in; ii++) acc += x[ii] * wp[(size_t)ii * out + oo];
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
        for (ii = 0; ii < in; ii++) acc += x[ii] * f16_to_f32(wp[(size_t)ii * out + oo]);
        y[oo] = acc;
    }
}

void matmul_q4k(float* y, const float* x, const uint8_t* w, uint32_t out, uint32_t in)
{
    uint32_t nb = in / 256;
    uint32_t rowb = nb * 144;
    uint32_t oo;
#ifdef __AVX2__
    #pragma omp parallel for schedule(static)
    for (oo = 0; oo < out; oo++) {
        const uint8_t* row = w + (size_t)oo * rowb;
        __m256 acc_v = _mm256_setzero_ps();
        uint32_t b;
        for (b = 0; b < nb; b++) {
            const uint8_t* blk = row + (size_t)b * 144;
            const float* xp = x + (size_t)b * 256;
            float d    = f16_to_f32(((const uint16_t*)blk)[0]);
            float dmin = f16_to_f32(((const uint16_t*)blk)[1]);
            const uint8_t* q = blk + 16;
            const uint8_t* scp = blk + 4;
            if (b + 1 < nb) _mm_prefetch((const char*)(blk + 144), _MM_HINT_T0);

            int is = 0;
            for (int j = 0; j < 4; j++) {
                uint8_t sc, mn;
                get_scale_min_k4(is, scp, &sc, &mn);
                float d1 = d * (float)sc;
                float m1 = dmin * (float)mn;
                get_scale_min_k4(is + 1, scp, &sc, &mn);
                float d2 = d * (float)sc;
                float m2 = dmin * (float)mn;

                __m256 sum_qx1_v = _mm256_setzero_ps();
                __m256 sum_x1_v  = _mm256_setzero_ps();
                __m256 sum_qx2_v = _mm256_setzero_ps();
                __m256 sum_x2_v  = _mm256_setzero_ps();

                for (int l = 0; l < 32; l += 16) {
                    __m128i q16 = _mm_loadu_si128((const __m128i*)(q + l));

                    __m128i lo16 = _mm_and_si128(q16, _mm_set1_epi8(0x0F));
                    __m128i hi16 = _mm_and_si128(_mm_srli_epi16(q16, 4), _mm_set1_epi8(0x0F));

                    /* Low nibbles: bytes 0-7 -> x[l..l+7], 8-15 -> x[l+8..l+15] */
                    __m256i lo16_0 = _mm256_cvtepu8_epi16(lo16);
                    __m256 lo_f_a0 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(lo16_0)));
                    __m256 lo_f_a1 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(lo16_0, 1)));
                    __m256 x_l0 = _mm256_loadu_ps(xp + l);
                    __m256 x_l1 = _mm256_loadu_ps(xp + l + 8);
                    sum_qx1_v = _mm256_fmadd_ps(lo_f_a0, x_l0, sum_qx1_v);
                    sum_qx1_v = _mm256_fmadd_ps(lo_f_a1, x_l1, sum_qx1_v);
                    sum_x1_v  = _mm256_add_ps(sum_x1_v, _mm256_add_ps(x_l0, x_l1));

                    /* High nibbles: bytes 0-7 -> x[l+32..l+39], 8-15 -> x[l+40..l+47] */
                    __m256i hi16_0 = _mm256_cvtepu8_epi16(hi16);
                    __m256 hi_f_a0 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(hi16_0)));
                    __m256 hi_f_a1 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(hi16_0, 1)));
                    __m256 x_h0 = _mm256_loadu_ps(xp + l + 32);
                    __m256 x_h1 = _mm256_loadu_ps(xp + l + 40);
                    sum_qx2_v = _mm256_fmadd_ps(hi_f_a0, x_h0, sum_qx2_v);
                    sum_qx2_v = _mm256_fmadd_ps(hi_f_a1, x_h1, sum_qx2_v);
                    sum_x2_v  = _mm256_add_ps(sum_x2_v, _mm256_add_ps(x_h0, x_h1));
                }

                /* 每块 8 次水平求和的替代: 缩放后并入行级向量累加, 整行只 hsum 一次 */
                acc_v = _mm256_fmadd_ps(sum_qx1_v, _mm256_set1_ps(d1), acc_v);
                acc_v = _mm256_fmadd_ps(sum_x1_v, _mm256_set1_ps(-m1), acc_v);
                acc_v = _mm256_fmadd_ps(sum_qx2_v, _mm256_set1_ps(d2), acc_v);
                acc_v = _mm256_fmadd_ps(sum_x2_v, _mm256_set1_ps(-m2), acc_v);

                xp += 64;
                q  += 32;
                is += 2;
            }
        }
        float acc = hsum_avx2(acc_v);
        y[oo] = acc;
    }
#else
    #pragma omp parallel for schedule(static)
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
#ifdef __AVX2__
    #pragma omp parallel for schedule(static)
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
    #pragma omp parallel for schedule(static)
    for (oo = 0; oo < out; oo++) {
        const uint8_t* row = w + (size_t)oo * rowb;
        float acc = 0.0f;
        uint32_t b;
        for (b = 0; b < nb; b++) {
            const uint8_t* blk = row + (size_t)b * 210;
            const float* xb = x + (size_t)b * 256;
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
    const uint32_t blk = (dtype == DT_Q6K) ? 210 : 144;
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

/* qwen2 interleaved RoPE: v[i] 与 v[i+half] 配对(llama 是 v[2j]/v[2j+1]) */
void rope_inplace_qwen(float* v, uint32_t d, uint32_t pos, float theta)
{
    uint32_t half = d / 2;
    uint32_t j;
    for (j = 0; j < half; j++) {
        float freq = powf(theta, -2.0f * (float)j / (float)d);
        float ang = freq * (float)pos;
        float c = cosf(ang);
        float s = sinf(ang);
        float a = v[j];
        float b = v[j + half];
        v[j] = a * c - b * s;
        v[j + half] = a * s + b * c;
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
