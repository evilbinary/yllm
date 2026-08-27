#include "yllm.h"
#include "matvec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int g_fail = 0;
static int g_pass = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; } \
    else { g_fail++; printf("FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); } \
} while (0)

#define CHECK_NEAR(a, b, eps, msg) do { \
    double _a = (double)(a), _b = (double)(b); \
    int _ok = 0; \
    if (isinf(_a) || isinf(_b)) _ok = (isinf(_a) && isinf(_b) && (signbit(_a) == signbit(_b))); \
    else _ok = (fabs(_a - _b) <= (eps)); \
    if (_ok) { g_pass++; } \
    else { g_fail++; printf("FAIL: %s: %.9g vs %.9g (%s:%d)\n", msg, _a, _b, __FILE__, __LINE__); } \
} while (0)

/* ---- golden reference data (from llama.cpp/gguf dequantize) ---- */
#include "ref_data.h"

/* ---- f16_to_f32 regression tests ---- *
 * Fixed bugs:
 *  1. denormal exponent was 125-sh instead of 113-sh (4096x amplification)
 *  2. denormal branch dropped the sign bit (negative -> positive)
 */
static void test_f16(void)
{
    CHECK_NEAR(f16_to_f32(0x0000), 0.0, 1e-9, "f16 0x0000 = 0");
    CHECK_NEAR(f16_to_f32(0x8000), -0.0, 1e-9, "f16 0x8000 = -0");
    CHECK_NEAR(f16_to_f32(0x3c00), 1.0, 1e-9, "f16 0x3c00 = 1");
    CHECK_NEAR(f16_to_f32(0xbc00), -1.0, 1e-9, "f16 0xbc00 = -1");
    CHECK_NEAR(f16_to_f32(0x0001), 5.960464477539063e-8, 1e-12, "f16 denormal 0x0001");
    CHECK_NEAR(f16_to_f32(0x8001), -5.960464477539063e-8, 1e-12, "f16 negative denormal 0x8001");
    CHECK_NEAR(f16_to_f32(0x03f7), 6.049871444702148e-5, 1e-12, "f16 denormal 0x03f7 (was 4096x too big)");
    CHECK_NEAR(f16_to_f32(0x83f7), -6.049871444702148e-5, 1e-12, "f16 negative denormal 0x83f7 (sign was dropped)");
    CHECK_NEAR(f16_to_f32(0x0400), 6.103515625e-5, 1e-12, "f16 min normal 0x0400");
    CHECK_NEAR(f16_to_f32(0x7c00), (double)INFINITY, 1e-9, "f16 +inf");
    CHECK_NEAR(f16_to_f32(0xfc00), -(double)INFINITY, 1e-9, "f16 -inf");
    /* round trip */
    {
        const uint16_t vals[] = { 0x0000, 0x3c00, 0xbc00, 0x0400, 0x7bff, 0x03ff, 0x8003, 0x3555, 0xc555 };
        size_t i;
        for (i = 0; i < sizeof(vals)/sizeof(vals[0]); i++) {
            uint16_t back = f32_to_f16(f16_to_f32(vals[i]));
            CHECK(back == vals[i], "f16 round trip");
        }
    }
    CHECK_NEAR(f32_to_f16(f16_to_f32(0x03f7)), 0x03f7, 1e-12, "f16 round trip denormal");
}

/* ---- Q4_K decode regression ---- */
static void test_q4k_decode(void)
{
    float out[512];
    uint32_t b;
    for (b = 0; b < 2; b++) {
        q4k_block(out + b * 256, K4_RAW + b * 144, 0);
    }
    uint32_t i;
    for (i = 0; i < 512; i++) {
        CHECK_NEAR(out[i], K4_REF[i], 1e-6, "q4k decode value");
    }
}

/* ---- Q6_K decode regression (block5 has negative d) ---- */
static void test_q6k_decode(void)
{
    float out[256], blk[256];
    uint32_t e;
    for (e = 0; e < 256; e++) {
        out[e] = q6k_val(V6_RAW, e);
    }
    for (e = 0; e < 256; e++) {
        CHECK_NEAR(out[e], V6_REF[e], 1e-6, "q6k decode value");
    }
    CHECK(dequant_mat_f32(blk, V6_RAW, 1, 256, DT_Q6K) == 0, "dequant_mat_f32 q6k");
    for (e = 0; e < 256; e++) {
        CHECK_NEAR(blk[e], V6_REF[e], 1e-5, "q6k_block vs golden");
    }
}

/* ---- embed_q4k regression (token 15043 embedding) ---- */
static void test_embed_q4k(void)
{
    float out[2048];
    uint8_t row[2 * 1152];
    uint32_t i;
    /* construct 2 rows: d=1, dmin=0, scales=1, qs=0x12 -> values 2 */
    for (i = 0; i < 2 * 8; i++) {
        uint8_t* blk = row + i * 144;
        memset(blk, 0, 144);
        blk[0] = 0x00; blk[1] = 0x3c;  /* d = 1.0 */
        blk[2] = 0x00; blk[3] = 0x00;  /* dmin = 0 */
        uint32_t s;
        for (s = 0; s < 12; s++) blk[4 + s] = 0x01;  /* scales = 1, mins = 1? */
        for (s = 0; s < 128; s++) blk[16 + s] = 0x12; /* low nibble 2, high nibble 1 */
    }
    embed_q4k(out, row, 0, 2048);
    CHECK_NEAR(out[0], 2.0, 1e-6, "embed_q4k synthetic d1 v0");
    CHECK_NEAR(out[1], 2.0, 1e-6, "embed_q4k synthetic v1");
    CHECK_NEAR(out[2], 2.0, 1e-6, "embed_q4k synthetic v2");
    /* row 1 differs: low nibble 4, high nibble 3 */
    memset(row + 1 * 1152 + 16, 0x34, 128);
    embed_q4k(out, row, 1, 2048);
    CHECK_NEAR(out[0], 4.0, 1e-6, "embed_q4k synthetic row1 v0");
}

/* ---- matmul_q4k: 2-block weight, known input ---- */
static void test_matmul_q4k(void)
{
    /* weight: 2 rows x 512 cols using synthetic q4k blocks:
       row0: d=1, dmin=0, scales=1, qs=0x12 -> all values 2 (low nibble)
       row1: d=1, dmin=0, scales=1, qs=0x01 -> low nibble 1, high nibble 0 -> 1 */
    uint8_t w[2 * 2 * 144];
    uint32_t r, b, s;
    for (r = 0; r < 2; r++) {
        for (b = 0; b < 2; b++) {
            uint8_t* blk = w + r * 2 * 144 + b * 144;
            memset(blk, 0, 144);
            blk[0] = 0x00; blk[1] = 0x3c;
            blk[2] = 0x00; blk[3] = 0x00;
            for (s = 0; s < 12; s++) blk[4 + s] = 0x01;
            for (s = 0; s < 128; s++) blk[16 + s] = (uint8_t)(r == 0 ? 0x12 : 0x01);
        }
    }
    float x[512];
    for (s = 0; s < 512; s++) x[s] = 1.0f;
    float y[2];
    matmul_q4k(y, x, w, 2, 512);
    /* row0: qs=0x12 -> low nibble 2 (256 elems) + high nibble 1 (256 elems) = 768 */
    CHECK_NEAR(y[0], 2.0 * 256 + 1.0 * 256, 1e-3, "matmul_q4k row0 (2/1 mix)");
    /* row1: qs=0x01 -> low nibble 1 (256) + high nibble 0 (256) = 256 */
    CHECK_NEAR(y[1], 1.0 * 256 + 0.0 * 256, 1e-3, "matmul_q4k row1 (1/0 mix)");
    /* same call through generic matmul dispatch */
    float y2[2];
    matmul(y2, x, w, 2, 512, DT_Q4K);
    CHECK_NEAR(y2[0], y[0], 1e-6, "matmul dispatch q4k");
}

/* ---- matmul_q5k: 2-block weight, known input ----
 * 每 block 176 字节: d@[0], min@[1], sc@[4..15], qh@[16..47], qs@[48..175]。
 * 4 组: 每组 qv1 = (qs&0xF)+(qh&u1?16:0), qv2 = (qs>>4)+(qh&u2?16:0)。
 * 此处 d=1, min=0, 所有 sc=1(缩放 1, min 项因 min=0 归零):
 *   row0: qs=0x11, qh=0 -> 每元素 1
 *   row1: qs=0x21, qh=0 -> low nibble 1, high nibble 2
 */
static void test_matmul_q5k(void)
{
    uint8_t w[2 * 2 * 176];
    uint32_t r, b, i;
    for (r = 0; r < 2; r++) {
        for (b = 0; b < 2; b++) {
            uint8_t* blk = w + r * 2 * 176 + b * 176;
            memset(blk, 0, 176);
            blk[0] = 0x00; blk[1] = 0x3c; /* d = 1.0 */
            blk[2] = 0x00; blk[3] = 0x00; /* min = 0 */
            for (i = 0; i < 12; i++) blk[4 + i] = 0x01; /* scales=1, mins->0 */
            for (i = 0; i < 128; i++) blk[48 + i] = (uint8_t)(r == 0 ? 0x11 : 0x21);
            /* qh(blk+16..47) 保持 0 */
        }
    }
    float x[512];
    for (i = 0; i < 512; i++) x[i] = 1.0f;
    float y[2];
    matmul_q5k(y, x, w, 2, 512);
    /* 每 block 256 元素: row0 全 1, row1 128*1 + 128*2; x=1 */
    CHECK_NEAR(y[0], 256.0 * 2, 1e-3, "matmul_q5k row0 (all 1)");
    CHECK_NEAR(y[1], (128.0 * 1 + 128.0 * 2) * 2, 1e-3, "matmul_q5k row1 (1/2 mix)");
    /* 通过通用 dispatch */
    float y2[2];
    matmul(y2, x, w, 2, 512, DT_Q5K);
    CHECK_NEAR(y2[0], y[0], 1e-6, "matmul dispatch q5k row0");
    CHECK_NEAR(y2[1], y[1], 1e-6, "matmul dispatch q5k row1");
}

/* ---- matmul_q6k: synthetic blocks ---- */
static void test_matmul_q6k(void)
{
    /* block: d=1, ql all 0x11 (low nibble 1, high nibble 1), qh all 0,
       scales all 1 -> q = 1 - 32 = -31, value = d*1*(-31) = -31 */
    uint8_t w[2 * 210];
    uint32_t b, i;
    for (b = 0; b < 2; b++) {
        uint8_t* blk = w + b * 210;
        memset(blk, 0, 210);
        blk[208] = 0x00; blk[209] = 0x3c; /* d = 1.0 */
        for (i = 0; i < 128; i++) blk[i] = 0x11;
        for (i = 0; i < 16; i++) blk[192 + i] = 1;
    }
    float x[512];
    for (i = 0; i < 512; i++) x[i] = 1.0f;
    float y[1];
    matmul_q6k(y, x, w, 1, 512);
    CHECK_NEAR(y[0], -31.0 * 512, 1e-3, "matmul_q6k synthetic");
    {
        uint8_t w2[2 * 2 * 210];
        float y2[2];
        memcpy(w2, w, 2 * 210);
        memcpy(w2 + 2 * 210, w, 2 * 210);
        matmul_q6k(y2, x, w2, 2, 512);
        CHECK_NEAR(y2[0], y[0], 1e-3, "matmul_q6k dual row0");
        CHECK_NEAR(y2[1], y[0], 1e-3, "matmul_q6k dual row1");
    }
    {
        float wr[512], xq[512], ref = 0.0f;
        uint32_t i;
        CHECK(dequant_mat_f32(wr, w, 1, 512, DT_Q6K) == 0, "dequant q6k row");
        matvec_q8k_quant(x, xq, 512);
        for (i = 0; i < 512; i++) ref += wr[i] * xq[i];
        CHECK_NEAR(y[0], ref, 1e-2, "matmul_q6k vs dequant dot");
    }
    {
        float xv[256], xq[256], wr[256], yg[1], ref = 0.0f;
        uint32_t i;
        for (i = 0; i < 256; i++) xv[i] = (float)((int)i - 128) * 0.03125f;
        CHECK(dequant_mat_f32(wr, V6_RAW, 1, 256, DT_Q6K) == 0, "dequant V6_RAW");
        matvec_q8k_quant(xv, xq, 256);
        for (i = 0; i < 256; i++) ref += wr[i] * xq[i];
        matmul_q6k(yg, xv, V6_RAW, 1, 256);
        CHECK_NEAR(yg[0], ref, 2e-2, "matmul_q6k golden vs dequant");
    }
}

static void test_q8k_quant(void)
{
    float x[512], y[512], ref[512];
    uint32_t i, b;
    for (i = 0; i < 512; i++) x[i] = (float)((int)i - 256) * 0.07f;
    matvec_q8k_quant(x, y, 512);
    for (b = 0; b < 2; b++) {
        const float* xb = x + b * 256;
        float* rb = ref + b * 256;
        float amax = 0.0f;
        for (i = 0; i < 256; i++) {
            float a = fabsf(xb[i]);
            if (a > amax) amax = a;
        }
        float d = amax / 127.0f;
        float inv = 1.0f / d;
        for (i = 0; i < 256; i++) {
            float q = roundf(xb[i] * inv);
            if (q > 127.0f) q = 127.0f;
            else if (q < -128.0f) q = -128.0f;
            rb[i] = q * d;
        }
    }
    for (i = 0; i < 512; i++)
        CHECK_NEAR(y[i], ref[i], 1e-5, "q8k_quant vs scalar");
}

/* ---- rmsnorm ---- */
static void test_rmsnorm(void)
{
    float x[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    float wn[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    float y[4];
    rmsnorm(y, x, (const uint8_t*)wn, 4, 1e-5f, DT_F32);
    /* x / sqrt(mean(x^2)+eps) = x / sqrt(30/4) */
    float inv = 1.0f / sqrtf(7.5f + 1e-5f);
    CHECK_NEAR(y[0], 1.0f * inv, 1e-5, "rmsnorm v0");
    CHECK_NEAR(y[3], 4.0f * inv, 1e-5, "rmsnorm v3");
}

/* ---- rope_inplace: pos=0 identity, pos>0 rotation ---- */
static void test_rope(void)
{
    float v[4] = { 1.0f, 0.0f, 0.0f, 1.0f };
    rope_inplace(v, 4, 0, 10000.0f);
    CHECK_NEAR(v[0], 1.0f, 1e-6, "rope pos0 v0");
    CHECK_NEAR(v[1], 0.0f, 1e-6, "rope pos0 v1");
    CHECK_NEAR(v[2], 0.0f, 1e-6, "rope pos0 v2");
    CHECK_NEAR(v[3], 1.0f, 1e-6, "rope pos0 v3");
    /* pos=1, pair (a,b)=(1,0): angle = 10000^(-0) = 1 rad? freq = theta^(-2*0/4)=1
       -> rotated by 1 rad */
    v[0] = 1.0f; v[1] = 0.0f; v[2] = 0.0f; v[3] = 1.0f;
    rope_inplace(v, 4, 1, 10000.0f);
    float c = cosf(1.0f), s = sinf(1.0f);
    CHECK_NEAR(v[0], c, 1e-5, "rope pos1 v0 (pairwise style)");
    CHECK_NEAR(v[1], s, 1e-5, "rope pos1 v1 (pairwise style)");
}

/* ---- softmax ---- */
static void test_softmax(void)
{
    float v[3] = { 1.0f, 2.0f, 3.0f };
    softmax(v, 3);
    float s = expf(1) + expf(2) + expf(3);
    CHECK_NEAR(v[0], expf(1) / s, 1e-6, "softmax v0");
    CHECK_NEAR(v[2], expf(3) / s, 1e-6, "softmax v2");
    CHECK_NEAR(v[0] + v[1] + v[2], 1.0, 1e-6, "softmax sum");
}

/* 对照旧路径: 每 head 独立 softmax + axpy */
static void attn_ref(float* out, const float* q,
                     const uint16_t* kcache, const uint16_t* vcache,
                     uint32_t s0, uint32_t pos,
                     uint32_t n_heads, uint32_t n_kv_heads, uint32_t hd, uint32_t kv_dim,
                     float inv_d, float attn_cap)
{
    uint32_t n = pos + 1 - s0;
    float* att = (float*)malloc((size_t)n * 4);
    uint32_t hh, s;
    for (hh = 0; hh < n_heads; hh++) {
        uint32_t kv_head = hh * n_kv_heads / n_heads;
        const float* qh = q + (size_t)hh * hd;
        float* og = out + (size_t)hh * hd;
        for (s = s0; s <= pos; s++) {
            const uint16_t* kh = kcache + (size_t)s * kv_dim + (size_t)kv_head * hd;
            float sc = vec_dot_f32_f16(qh, kh, hd) * inv_d;
            if (attn_cap > 0.0f) sc = attn_cap * tanhf(sc / attn_cap);
            att[s - s0] = sc;
        }
        softmax(att, n);
        memset(og, 0, (size_t)hd * 4);
        for (s = s0; s <= pos; s++) {
            const uint16_t* vh = vcache + (size_t)s * kv_dim + (size_t)kv_head * hd;
            vec_axpy_f16(og, vh, att[s - s0], hd);
        }
    }
    free(att);
}

static void test_attn_kv_f16(void)
{
    const uint32_t n_heads = 4, n_kv = 2, hd = 8, kv_dim = n_kv * hd;
    const uint32_t pos = 11, s0 = 3;
    float q[4 * 8];
    float out[4 * 8], ref[4 * 8];
    uint16_t kcache[12 * 16], vcache[12 * 16];
    uint32_t i, t;
    float inv_d = 1.0f / sqrtf((float)hd);
    float cap = 20.0f;
    for (i = 0; i < n_heads * hd; i++)
        q[i] = sinf((float)i * 0.17f) * 0.5f;
    for (t = 0; t <= pos; t++) {
        for (i = 0; i < kv_dim; i++) {
            float kf = cosf((float)(t * 17 + i) * 0.03f);
            float vf = sinf((float)(t * 13 + i) * 0.05f);
            kcache[t * kv_dim + i] = f32_to_f16(kf);
            vcache[t * kv_dim + i] = f32_to_f16(vf);
        }
    }
    attn_kv_f16(out, q, kcache, vcache, s0, pos, n_heads, n_kv, hd, kv_dim, inv_d, 0.0f);
    attn_ref(ref, q, kcache, vcache, s0, pos, n_heads, n_kv, hd, kv_dim, inv_d, 0.0f);
    for (i = 0; i < n_heads * hd; i++)
        CHECK_NEAR(out[i], ref[i], 2e-3, "attn_kv_f16 vs two-pass (swa)");
    attn_kv_f16(out, q, kcache, vcache, 0, pos, n_heads, n_kv, hd, kv_dim, 1.0f, cap);
    attn_ref(ref, q, kcache, vcache, 0, pos, n_heads, n_kv, hd, kv_dim, 1.0f, cap);
    for (i = 0; i < n_heads * hd; i++)
        CHECK_NEAR(out[i], ref[i], 2e-3, "attn_kv_f16 vs two-pass (cap)");
    {
        const uint32_t nh = 8, nkv1 = 1, hd2 = 8, kvd = nkv1 * hd2;
        float q2[8 * 8], o2[8 * 8], r2[8 * 8];
        uint16_t k2[12 * 8], v2[12 * 8];
        uint32_t j;
        for (j = 0; j < nh * hd2; j++) q2[j] = sinf((float)j * 0.11f);
        for (t = 0; t <= pos; t++) {
            for (j = 0; j < kvd; j++) {
                k2[t * kvd + j] = f32_to_f16(cosf((float)(t + j) * 0.07f));
                v2[t * kvd + j] = f32_to_f16(sinf((float)(t + j) * 0.09f));
            }
        }
        attn_kv_f16(o2, q2, k2, v2, 0, pos, nh, nkv1, hd2, kvd, inv_d, 0.0f);
        attn_ref(r2, q2, k2, v2, 0, pos, nh, nkv1, hd2, kvd, inv_d, 0.0f);
        for (j = 0; j < nh * hd2; j++)
            CHECK_NEAR(o2[j], r2[j], 2e-3, "attn_kv_f16 GQA 8:1");
    }
}

/* ---- swiglu ---- */
static void test_swiglu(void)
{
    float g[2] = { 0.0f, 1.0f };
    float u[2] = { 2.0f, 3.0f };
    float y[2];
    swiglu(y, g, u, 2);
    CHECK_NEAR(y[0], 0.0f, 1e-6, "swiglu sigmoid(0)=0.5*2? g=0 -> 0");
    CHECK_NEAR(y[1], (1.0f / (1.0f + expf(-1.0f))) * 3.0f, 1e-6, "swiglu g=1 u=3");
}

int main(void)
{
    printf("running test_f16...\n"); fflush(stdout);
    test_f16();
    printf("running test_q4k_decode...\n"); fflush(stdout);
    test_q4k_decode();
    printf("running test_q6k_decode...\n"); fflush(stdout);
    test_q6k_decode();
    printf("running test_embed_q4k...\n"); fflush(stdout);
    test_embed_q4k();
    printf("running test_matmul_q4k...\n"); fflush(stdout);
    test_matmul_q4k();
    printf("running test_matmul_q5k...\n"); fflush(stdout);
    test_matmul_q5k();
    printf("running test_matmul_q6k...\n"); fflush(stdout);
    test_matmul_q6k();
    printf("running test_q8k_quant...\n"); fflush(stdout);
    test_q8k_quant();
    printf("running test_rmsnorm...\n"); fflush(stdout);
    test_rmsnorm();
    printf("running test_rope...\n"); fflush(stdout);
    test_rope();
    printf("running test_softmax...\n"); fflush(stdout);
    test_softmax();
    printf("running test_attn_kv_f16...\n"); fflush(stdout);
    test_attn_kv_f16();
    printf("running test_swiglu...\n"); fflush(stdout);
    test_swiglu();
    printf("matvec tests: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
