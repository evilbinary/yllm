/* vulkan_fwd.c — native: RMSNorm(F32) 走 GPU, 其余仍 CPU */
#include "device.h"
#include "yllm.h"
#include "matvec.h"
#include "llf.h"
#include "vulkan_ctx.h"
#include "vulkan_compute.h"
#include "log.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>

int vulkan_selftest_rmsnorm(VulkanCtx* ctx)
{
    if (!ctx || !ctx->compute_ready || ctx->hidden == 0) return -1;
    uint32_t n = ctx->hidden;
    float* x = (float*)malloc((size_t)n * 4);
    float* w = (float*)malloc((size_t)n * 4);
    float* yg = (float*)malloc((size_t)n * 4);
    float* yc = (float*)malloc((size_t)n * 4);
    if (!x || !w || !yg || !yc) {
        free(x); free(w); free(yg); free(yc);
        return -1;
    }
    uint32_t i;
    for (i = 0; i < n; i++) {
        x[i] = 0.01f * (float)((int)(i % 97) - 48);
        w[i] = 1.0f + 0.001f * (float)(i % 13);
    }
    float eps = 1e-6f;
    if (vulkan_k_rmsnorm(ctx, yg, x, w, n, eps) != 0) {
        free(x); free(w); free(yg); free(yc);
        return -1;
    }
    {
        float ss = 0.0f;
        for (i = 0; i < n; i++) ss += x[i] * x[i];
        float inv = 1.0f / sqrtf(ss / (float)n + eps);
        for (i = 0; i < n; i++) yc[i] = x[i] * inv * w[i];
    }
    float maxe = 0.0f;
    for (i = 0; i < n; i++) {
        float d = fabsf(yg[i] - yc[i]);
        if (d > maxe) maxe = d;
    }
    free(x); free(w); free(yg); free(yc);
    ylog_info("vulkan: rmsnorm selftest max_abs_err=%.6g (n=%u)", (double)maxe, n);
    return maxe < 1e-3f ? 0 : -1;
}

/* 合成一小块 Q4_K 与 CPU matmul_q4k 对比 */
int vulkan_selftest_gemv_q4k(VulkanCtx* ctx)
{
    if (!ctx || !ctx->gemv_ready) return -1;
    uint32_t in = 256;
    uint32_t out = 8;
    if (in > ctx->max_in || out > ctx->max_out) return -1;
    size_t wbytes = (size_t)out * 144;
    if (wbytes > ctx->wq_bytes) return -1;

    float* x = (float*)malloc((size_t)in * 4);
    float* yg = (float*)malloc((size_t)out * 4);
    float* yc = (float*)malloc((size_t)out * 4);
    uint8_t* w = (uint8_t*)calloc(1, wbytes);
    if (!x || !yg || !yc || !w) {
        free(x); free(yg); free(yc); free(w);
        return -1;
    }
    uint32_t i;
    for (i = 0; i < in; i++)
        x[i] = 0.02f * (float)((int)(i % 17) - 8);
    /* 每行一块: d=1, dmin=0, scales=1, qs 填递增 nibble */
    for (i = 0; i < out; i++) {
        uint8_t* blk = w + (size_t)i * 144;
        ((uint16_t*)blk)[0] = 0x3c00; /* f16 1.0 */
        ((uint16_t*)blk)[1] = 0x0000; /* f16 0.0 */
        memset(blk + 4, 0x01, 12);    /* sc/min 低 6bit ≈1 */
        uint32_t e;
        for (e = 0; e < 128; e++)
            blk[16 + e] = (uint8_t)((e & 0xF) | (((e + 3) & 0xF) << 4));
    }
    if (vulkan_k_gemv_q4k_host(ctx, yg, x, w, out, in) != 0) {
        free(x); free(yg); free(yc); free(w);
        return -1;
    }
    /* 参考用反量化+点积(勿用 AVX Q8 路径, 会引入额外量化误差) */
    {
        float deq[256];
        uint32_t o, j;
        for (o = 0; o < out; o++) {
            float acc = 0.0f;
            const uint8_t* row = w + (size_t)o * 144;
            q4k_block(deq, row, 0);
            for (j = 0; j < 256; j++) acc += x[j] * deq[j];
            yc[o] = acc;
        }
    }
    float maxe = 0.0f;
    for (i = 0; i < out; i++) {
        float d = fabsf(yg[i] - yc[i]);
        if (d > maxe) maxe = d;
    }
    free(x); free(yg); free(yc); free(w);
    ylog_info("vulkan: gemv_q4k selftest max_abs_err=%.6g (out=%u in=%u)",
              (double)maxe, out, in);
    return maxe < 1e-3f ? 0 : -1;
}

static int load_norm_f32(VulkanCtx* ctx, const uint8_t* wbytes, uint32_t n, uint32_t dtype)
{
    if (!ctx || !ctx->host_w || (size_t)n * 4 > ctx->wn_bytes) return -1;
    if (dtype == DT_F32) {
        memcpy(ctx->host_w, wbytes, (size_t)n * 4);
        return 0;
    }
    if (dtype == DT_F16) {
        uint32_t i;
        const uint16_t* wh = (const uint16_t*)wbytes;
        for (i = 0; i < n; i++) ctx->host_w[i] = f16_to_f32(wh[i]);
        return 0;
    }
    return -1;
}

static uint64_t wq_off(VulkanCtx* ctx, uint32_t layer, uint32_t slot)
{
    if (!ctx || !ctx->wq_off) return (uint64_t)~0ull;
    return ctx->wq_off[(size_t)layer * ctx->wq_nslot + slot];
}

static void vk_or_cpu_rmsnorm(VulkanCtx* ctx,
                              float* y, const float* x,
                              const uint8_t* wbytes, uint32_t n, float eps, uint32_t dtype)
{
    if (ctx && ctx->compute_ready && (size_t)n * 4 <= ctx->x_bytes &&
        (size_t)n * 4 <= ctx->wn_bytes) {
        if (dtype == DT_F32) {
            if (vulkan_k_rmsnorm(ctx, y, x, (const float*)wbytes, n, eps) == 0)
                return;
        } else if (dtype == DT_F16 && ctx->host_w) {
            uint32_t i;
            const uint16_t* wh = (const uint16_t*)wbytes;
            for (i = 0; i < n; i++) ctx->host_w[i] = f16_to_f32(wh[i]);
            if (vulkan_k_rmsnorm(ctx, y, x, ctx->host_w, n, eps) == 0)
                return;
        }
    }
    rmsnorm(y, x, wbytes, n, eps, dtype);
}

static void vk_or_cpu_matmul(VulkanCtx* ctx, float* y, const float* x,
                             const uint8_t* w, uint32_t out, uint32_t in, uint32_t dtype,
                             uint32_t layer, uint32_t slot)
{
    if (ctx && ctx->gemv_ready && dtype == DT_Q4K) {
        if (ctx->wq_resident && ctx->wq_off) {
            uint64_t off = wq_off(ctx, layer, slot);
            if (off != (uint64_t)~0ull &&
                vulkan_k_gemv_q4k(ctx, y, x, out, in, off) == 0)
                return;
        }
        if (vulkan_k_gemv_q4k_host(ctx, y, x, w, out, in) == 0)
            return;
    }
    matmul(y, x, w, out, in, dtype);
}

static int try_fused_qkv(VulkanCtx* ctx, const float* x, float* q, float* k, float* v,
                         uint32_t hidden, uint32_t kv_dim, float eps,
                         const uint8_t* base, const LlfTensorMeta* mt, uint32_t layer)
{
    if (!ctx || !ctx->fuse_ready || !ctx->wq_resident) return -1;
    if (mt[SLOT_Q].dtype != DT_Q4K || mt[SLOT_K].dtype != DT_Q4K || mt[SLOT_V].dtype != DT_Q4K)
        return -1;
    if (load_norm_f32(ctx, base + mt[SLOT_NORM1].offset, hidden, mt[SLOT_NORM1].dtype) != 0)
        return -1;
    uint64_t oq = wq_off(ctx, layer, SLOT_Q);
    uint64_t ok = wq_off(ctx, layer, SLOT_K);
    uint64_t ov = wq_off(ctx, layer, SLOT_V);
    if (oq == (uint64_t)~0ull || ok == (uint64_t)~0ull || ov == (uint64_t)~0ull)
        return -1;
    return vulkan_fused_norm_qkv(ctx, x, ctx->host_w, hidden, eps,
                                 q, k, v, kv_dim, oq, ok, ov);
}

static int try_fused_ffn(VulkanCtx* ctx, const float* x, float* out,
                         uint32_t hidden, uint32_t inter, float eps,
                         const uint8_t* base, const LlfTensorMeta* mt, uint32_t layer)
{
    if (!ctx || !ctx->fuse_ready || !ctx->swi_ready || !ctx->wq_resident) return -1;
    if (mt[SLOT_GATE].dtype != DT_Q4K || mt[SLOT_UP].dtype != DT_Q4K ||
        mt[SLOT_DOWN].dtype != DT_Q4K)
        return -1;
    if (load_norm_f32(ctx, base + mt[SLOT_NORM2].offset, hidden, mt[SLOT_NORM2].dtype) != 0)
        return -1;
    uint64_t og = wq_off(ctx, layer, SLOT_GATE);
    uint64_t ou = wq_off(ctx, layer, SLOT_UP);
    uint64_t od = wq_off(ctx, layer, SLOT_DOWN);
    if (og == (uint64_t)~0ull || ou == (uint64_t)~0ull || od == (uint64_t)~0ull)
        return -1;
    return vulkan_fused_ffn(ctx, x, ctx->host_w, hidden, eps, out, inter, og, ou, od);
}

static int try_fused_attn_block(VulkanCtx* ctx, const float* x, float* out,
                                uint32_t hidden, uint32_t kv_dim, float eps, float theta,
                                uint32_t rope_mode, uint32_t layer, uint32_t pos,
                                const uint8_t* base, const LlfTensorMeta* mt,
                                uint16_t* host_k_row, uint16_t* host_v_row)
{
    if (!ctx || !ctx->rope_ready || !ctx->attn_ready || !ctx->fuse_ready || !ctx->wq_resident)
        return -1;
    if (mt[SLOT_Q].dtype != DT_Q4K || mt[SLOT_K].dtype != DT_Q4K || mt[SLOT_V].dtype != DT_Q4K)
        return -1;
    if (load_norm_f32(ctx, base + mt[SLOT_NORM1].offset, hidden, mt[SLOT_NORM1].dtype) != 0)
        return -1;
    uint64_t oq = wq_off(ctx, layer, SLOT_Q);
    uint64_t ok = wq_off(ctx, layer, SLOT_K);
    uint64_t ov = wq_off(ctx, layer, SLOT_V);
    if (!ctx->attn_o_ready || mt[SLOT_O].dtype != DT_Q4K) return -1;
    uint64_t oo = wq_off(ctx, layer, SLOT_O);
    if (oq == (uint64_t)~0ull || ok == (uint64_t)~0ull || ov == (uint64_t)~0ull ||
        oo == (uint64_t)~0ull)
        return -1;

    const float* bq = mt[SLOT_QBIAS].size > 0 ? (const float*)(base + mt[SLOT_QBIAS].offset) : NULL;
    const float* bk = mt[SLOT_KBIAS].size > 0 ? (const float*)(base + mt[SLOT_KBIAS].offset) : NULL;
    const float* bv = mt[SLOT_VBIAS].size > 0 ? (const float*)(base + mt[SLOT_VBIAS].offset) : NULL;
    const uint8_t* qn = mt[SLOT_QNORM].size > 0 ? base + mt[SLOT_QNORM].offset : NULL;
    const uint8_t* kn = mt[SLOT_KNORM].size > 0 ? base + mt[SLOT_KNORM].offset : NULL;

    return vulkan_fused_qkv_rope_attn(ctx, x, ctx->host_w, hidden, eps,
                                      out, kv_dim, oq, ok, ov, oo,
                                      layer, pos, rope_mode, theta,
                                      bq, bk, bv,
                                      qn, mt[SLOT_QNORM].dtype,
                                      kn, mt[SLOT_KNORM].dtype,
                                      host_k_row, host_v_row);
}

static int vulkan_fwd_block(Engine* e, uint32_t layer, uint32_t pos)
{
    VulkanCtx* ctx = (VulkanCtx*)e->w_dev;
    const uint8_t* base = (const uint8_t*)e->ws.map.base + e->ws.model.dir[layer].offset;
    uint16_t* kv = e->kv;

    Ws* ws = &e->ws;
    LlModel* m = &ws->model;
    const LlfHeader* h = &m->h;
    uint32_t hidden = h->hidden;
    uint32_t kv_dim = h->n_kv_heads * h->head_dim;
    float eps, theta;
    memcpy(&eps, &h->norm_eps_bits, 4);
    memcpy(&theta, &h->rope_theta_bits, 4);
    float* x = e->x;
    float* x2 = e->hb;
    float* q = e->hb2;
    float* k = e->hb2 + hidden;
    float* v = e->hb2 + hidden + kv_dim;
    float* att_out = e->hb2 + hidden + 2 * kv_dim;
    uint32_t hidx = m->base_idx[layer];
    const LlfTensorMeta* mt = &m->metas[hidx];
    uint32_t inter = mt[SLOT_GATE].shape[0] * mt[SLOT_GATE].shape[1] / hidden;
    uint16_t* kcache = kv + (size_t)layer * e->max_seq * kv_dim;
    uint16_t* vcache = kv + (size_t)(h->n_blocks + layer) * e->max_seq * kv_dim;
    uint64_t kvp = (uint64_t)pos * kv_dim;
    uint32_t hh, j;
    uint32_t rope_mode = (h->arch == ARCH_QWEN) ? 1u : 0u;

    /* 快路径: QKV+rope+attn(+O) 少 host 往返 */
    if (try_fused_attn_block(ctx, x, att_out, hidden, kv_dim, eps, theta, rope_mode,
                             layer, pos, base, mt, kcache + kvp, vcache + kvp) == 0) {
        for (j = 0; j < hidden; j++) x[j] += att_out[j];
        if (try_fused_ffn(ctx, x, att_out, hidden, inter, eps, base, mt, layer) == 0) {
            for (j = 0; j < hidden; j++) x[j] += att_out[j];
            return 0;
        }
        {
            float* fg = e->ffn;
            float* fu = e->ffn + inter;
            vk_or_cpu_rmsnorm(ctx, x2, x, base + mt[SLOT_NORM2].offset,
                              hidden, eps, mt[SLOT_NORM2].dtype);
            vk_or_cpu_matmul(ctx, fg, x2, base + mt[SLOT_GATE].offset, inter, hidden, mt[SLOT_GATE].dtype,
                             layer, SLOT_GATE);
            vk_or_cpu_matmul(ctx, fu, x2, base + mt[SLOT_UP].offset, inter, hidden, mt[SLOT_UP].dtype,
                             layer, SLOT_UP);
            swiglu(x2, fg, fu, inter);
            vk_or_cpu_matmul(ctx, att_out, x2, base + mt[SLOT_DOWN].offset, hidden, inter, mt[SLOT_DOWN].dtype,
                             layer, SLOT_DOWN);
            for (j = 0; j < hidden; j++) x[j] += att_out[j];
        }
        return 0;
    }

    if (try_fused_qkv(ctx, x, q, k, v, hidden, kv_dim, eps, base, mt, layer) != 0) {
        vk_or_cpu_rmsnorm(ctx, x2, x, base + mt[SLOT_NORM1].offset,
                          hidden, eps, mt[SLOT_NORM1].dtype);
        vk_or_cpu_matmul(ctx, q, x2, base + mt[SLOT_Q].offset, hidden, hidden, mt[SLOT_Q].dtype,
                         layer, SLOT_Q);
        vk_or_cpu_matmul(ctx, k, x2, base + mt[SLOT_K].offset, kv_dim, hidden, mt[SLOT_K].dtype,
                         layer, SLOT_K);
        vk_or_cpu_matmul(ctx, v, x2, base + mt[SLOT_V].offset, kv_dim, hidden, mt[SLOT_V].dtype,
                         layer, SLOT_V);
    }
    if (mt[SLOT_QBIAS].size > 0) {
        const float* bq = (const float*)(base + mt[SLOT_QBIAS].offset);
        for (j = 0; j < hidden; j++) q[j] += bq[j];
    }
    if (mt[SLOT_KBIAS].size > 0) {
        const float* bk = (const float*)(base + mt[SLOT_KBIAS].offset);
        for (j = 0; j < kv_dim; j++) k[j] += bk[j];
    }
    if (mt[SLOT_VBIAS].size > 0) {
        const float* bv = (const float*)(base + mt[SLOT_VBIAS].offset);
        for (j = 0; j < kv_dim; j++) v[j] += bv[j];
    }
    if (mt[SLOT_QNORM].size > 0) {
        for (hh = 0; hh < h->n_heads; hh++)
            rmsnorm(q + (size_t)hh * h->head_dim, q + (size_t)hh * h->head_dim,
                    base + mt[SLOT_QNORM].offset, h->head_dim, eps, mt[SLOT_QNORM].dtype);
    }
    if (mt[SLOT_KNORM].size > 0) {
        for (hh = 0; hh < h->n_kv_heads; hh++)
            rmsnorm(k + (size_t)hh * h->head_dim, k + (size_t)hh * h->head_dim,
                    base + mt[SLOT_KNORM].offset, h->head_dim, eps, mt[SLOT_KNORM].dtype);
    }

    for (hh = 0; hh < h->n_heads; hh++) {
        if (rope_mode)
            rope_inplace_qwen(q + (size_t)hh * h->head_dim, h->head_dim, pos, theta);
        else
            rope_inplace(q + (size_t)hh * h->head_dim, h->head_dim, pos, theta);
    }
    for (hh = 0; hh < h->n_kv_heads; hh++) {
        if (rope_mode)
            rope_inplace_qwen(k + (size_t)hh * h->head_dim, h->head_dim, pos, theta);
        else
            rope_inplace(k + (size_t)hh * h->head_dim, h->head_dim, pos, theta);
    }

    int attn_ok = 0;
    int attn_fused_o = 0;
    if (ctx && ctx->attn_ready) {
        uint64_t off_o = (uint64_t)~0ull;
        if (ctx->attn_o_ready && ctx->wq_resident && mt[SLOT_O].dtype == DT_Q4K) {
            uint64_t oo = wq_off(ctx, layer, SLOT_O);
            if (oo != (uint64_t)~0ull) {
                off_o = oo;
                attn_fused_o = 1;
            }
        }
        attn_ok = (vulkan_k_attn_decode(ctx, q, k, v, att_out, layer, pos,
                                        kcache + kvp, vcache + kvp, off_o) == 0);
        if (!attn_ok) attn_fused_o = 0;
    }
    if (!attn_ok) {
        uint32_t j2;
        for (j2 = 0; j2 < kv_dim; j2++) {
            kcache[kvp + j2] = f32_to_f16(k[j2]);
            vcache[kvp + j2] = f32_to_f16(v[j2]);
        }
        float* att = e->att;
        float inv_d = 1.0f / sqrtf((float)h->head_dim);
        #pragma omp parallel for schedule(static)
        for (hh = 0; hh < h->n_heads; hh++) {
            float* att_h = att + (size_t)hh * e->max_seq;
            uint32_t kv_head = hh * h->n_kv_heads / h->n_heads;
            const float* qh = q + (size_t)hh * h->head_dim;
            uint32_t s, jj;
            for (s = 0; s <= pos; s++) {
                const uint16_t* kh = kcache + (size_t)s * kv_dim + (size_t)kv_head * h->head_dim;
                float acc = 0.0f;
                for (jj = 0; jj < h->head_dim; jj++) acc += qh[jj] * f16_to_f32(kh[jj]);
                att_h[s] = acc * inv_d;
            }
            softmax(att_h, pos + 1);
            float* outh = att_out + (size_t)hh * h->head_dim;
            memset(outh, 0, (size_t)h->head_dim * 4);
            for (s = 0; s <= pos; s++) {
                const uint16_t* vh = vcache + (size_t)s * kv_dim + (size_t)kv_head * h->head_dim;
                float a = att_h[s];
                for (jj = 0; jj < h->head_dim; jj++) outh[jj] += a * f16_to_f32(vh[jj]);
            }
        }
    }
    if (!attn_fused_o) {
        memcpy(x2, att_out, (size_t)hidden * 4);
        vk_or_cpu_matmul(ctx, att_out, x2, base + mt[SLOT_O].offset, hidden, hidden, mt[SLOT_O].dtype,
                         layer, SLOT_O);
    }
    for (j = 0; j < hidden; j++) x[j] += att_out[j];

    if (try_fused_ffn(ctx, x, att_out, hidden, inter, eps, base, mt, layer) == 0) {
        for (j = 0; j < hidden; j++) x[j] += att_out[j];
        return 0;
    }
    {
        float* fg = e->ffn;
        float* fu = e->ffn + inter;
        vk_or_cpu_rmsnorm(ctx, x2, x, base + mt[SLOT_NORM2].offset,
                          hidden, eps, mt[SLOT_NORM2].dtype);
        vk_or_cpu_matmul(ctx, fg, x2, base + mt[SLOT_GATE].offset, inter, hidden, mt[SLOT_GATE].dtype,
                         layer, SLOT_GATE);
        vk_or_cpu_matmul(ctx, fu, x2, base + mt[SLOT_UP].offset, inter, hidden, mt[SLOT_UP].dtype,
                         layer, SLOT_UP);
        swiglu(x2, fg, fu, inter);
        vk_or_cpu_matmul(ctx, att_out, x2, base + mt[SLOT_DOWN].offset, hidden, inter, mt[SLOT_DOWN].dtype,
                         layer, SLOT_DOWN);
        for (j = 0; j < hidden; j++) x[j] += att_out[j];
    }
    return 0;
}

void vulkan_attach_fwd(Engine* e)
{
    engine_attach_cpu_fwd(e);
    if (e->device_mode == DEV_MODE_VULKAN)
        e->fwd_block = vulkan_fwd_block;
}

int vulkan_final_norm(Engine* e)
{
    if (!e || e->device_mode != DEV_MODE_VULKAN) return -1;
    VulkanCtx* ctx = (VulkanCtx*)e->w_dev;
    if (!ctx || !ctx->compute_ready) return -1;
    LlModel* m = &e->ws.model;
    uint32_t layer = m->h.n_blocks + 1;
    if (layer >= m->n_layers) return -1;
    const uint8_t* base = (const uint8_t*)e->ws.map.base + m->dir[layer].offset;
    const LlfTensorMeta* tm = &m->metas[m->base_idx[layer]];
    uint32_t hidden = m->h.hidden;
    float eps = ctx->norm_eps;
    if (load_norm_f32(ctx, base + tm->offset, hidden, tm->dtype) != 0) return -1;
    if (vulkan_k_rmsnorm(ctx, e->x, e->x, ctx->host_w, hidden, eps) != 0) return -1;
    return 0;
}

int vulkan_lm_head(Engine* e)
{
    if (!e || e->device_mode != DEV_MODE_VULKAN) return -1;
    VulkanCtx* ctx = (VulkanCtx*)e->w_dev;
    if (!ctx || !ctx->gemv_ready || !ctx->lm_ready || !ctx->wq_resident) return -1;
    if (ctx->lm_in != ctx->hidden || ctx->lm_out == 0) return -1;
    if ((ctx->lm_in % 256) != 0) return -1;

    uint32_t vocab = ctx->lm_out;
    uint32_t hidden = ctx->lm_in;
    size_t row_bytes = (size_t)(hidden / 256) * 144;
    uint32_t chunk = ctx->max_out;
    if (chunk == 0) chunk = hidden;
    if (chunk > vocab) chunk = vocab;

    uint32_t rows = 0;
    while (rows < vocab) {
        uint32_t n = vocab - rows;
        if (n > chunk) n = chunk;
        uint64_t off = ctx->lm_off + (uint64_t)rows * row_bytes;
        if (vulkan_k_gemv_q4k(ctx, e->logits + rows, e->x, n, hidden, off) != 0)
            return -1;
        rows += n;
    }
    return 0;
}
