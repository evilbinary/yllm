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

static void vk_or_cpu_rmsnorm(VulkanCtx* ctx,
                              float* y, const float* x,
                              const uint8_t* wbytes, uint32_t n, float eps, uint32_t dtype)
{
    if (ctx && ctx->compute_ready && (size_t)n * 4 <= ctx->buf_bytes) {
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

    vk_or_cpu_rmsnorm(ctx, x2, x, base + mt[SLOT_NORM1].offset,
                      hidden, eps, mt[SLOT_NORM1].dtype);
    matmul(q, x2, base + mt[SLOT_Q].offset, hidden, hidden, mt[SLOT_Q].dtype);
    matmul(k, x2, base + mt[SLOT_K].offset, kv_dim, hidden, mt[SLOT_K].dtype);
    matmul(v, x2, base + mt[SLOT_V].offset, kv_dim, hidden, mt[SLOT_V].dtype);
    if (mt[SLOT_QBIAS].size > 0) {
        const float* bq = (const float*)(base + mt[SLOT_QBIAS].offset);
        uint32_t j;
        for (j = 0; j < hidden; j++) q[j] += bq[j];
    }
    if (mt[SLOT_KBIAS].size > 0) {
        const float* bk = (const float*)(base + mt[SLOT_KBIAS].offset);
        uint32_t j;
        for (j = 0; j < kv_dim; j++) k[j] += bk[j];
    }
    if (mt[SLOT_VBIAS].size > 0) {
        const float* bv = (const float*)(base + mt[SLOT_VBIAS].offset);
        uint32_t j;
        for (j = 0; j < kv_dim; j++) v[j] += bv[j];
    }
    if (mt[SLOT_QNORM].size > 0) {
        uint32_t hh;
        for (hh = 0; hh < h->n_heads; hh++)
            rmsnorm(q + (size_t)hh * h->head_dim, q + (size_t)hh * h->head_dim,
                    base + mt[SLOT_QNORM].offset, h->head_dim, eps, mt[SLOT_QNORM].dtype);
    }
    if (mt[SLOT_KNORM].size > 0) {
        uint32_t hh;
        for (hh = 0; hh < h->n_kv_heads; hh++)
            rmsnorm(k + (size_t)hh * h->head_dim, k + (size_t)hh * h->head_dim,
                    base + mt[SLOT_KNORM].offset, h->head_dim, eps, mt[SLOT_KNORM].dtype);
    }

    uint16_t* kcache = kv + (size_t)layer * e->max_seq * kv_dim;
    uint16_t* vcache = kv + (size_t)(h->n_blocks + layer) * e->max_seq * kv_dim;
    uint64_t kvp = (uint64_t)pos * kv_dim;
    uint32_t hh, j;
    for (hh = 0; hh < h->n_heads; hh++) {
        if (h->arch == ARCH_QWEN)
            rope_inplace_qwen(q + (size_t)hh * h->head_dim, h->head_dim, pos, theta);
        else
            rope_inplace(q + (size_t)hh * h->head_dim, h->head_dim, pos, theta);
    }
    for (hh = 0; hh < h->n_kv_heads; hh++) {
        if (h->arch == ARCH_QWEN)
            rope_inplace_qwen(k + (size_t)hh * h->head_dim, h->head_dim, pos, theta);
        else
            rope_inplace(k + (size_t)hh * h->head_dim, h->head_dim, pos, theta);
    }
    for (j = 0; j < kv_dim; j++) {
        kcache[kvp + j] = f32_to_f16(k[j]);
        vcache[kvp + j] = f32_to_f16(v[j]);
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
        float* out = att_out + (size_t)hh * h->head_dim;
        memset(out, 0, (size_t)h->head_dim * 4);
        for (s = 0; s <= pos; s++) {
            const uint16_t* vh = vcache + (size_t)s * kv_dim + (size_t)kv_head * h->head_dim;
            float a = att_h[s];
            for (jj = 0; jj < h->head_dim; jj++) out[jj] += a * f16_to_f32(vh[jj]);
        }
    }
    memcpy(x2, att_out, (size_t)hidden * 4);
    matmul(att_out, x2, base + mt[SLOT_O].offset, hidden, hidden, mt[SLOT_O].dtype);
    for (j = 0; j < hidden; j++) x[j] += att_out[j];
    vk_or_cpu_rmsnorm(ctx, x2, x, base + mt[SLOT_NORM2].offset,
                      hidden, eps, mt[SLOT_NORM2].dtype);
    float* fg = e->ffn;
    float* fu = e->ffn + inter;
    matmul(fg, x2, base + mt[SLOT_GATE].offset, inter, hidden, mt[SLOT_GATE].dtype);
    matmul(fu, x2, base + mt[SLOT_UP].offset, inter, hidden, mt[SLOT_UP].dtype);
    swiglu(x2, fg, fu, inter);
    matmul(att_out, x2, base + mt[SLOT_DOWN].offset, hidden, inter, mt[SLOT_DOWN].dtype);
    for (j = 0; j < hidden; j++) x[j] += att_out[j];
    return 0;
}

void vulkan_attach_fwd(Engine* e)
{
    engine_attach_cpu_fwd(e);
    if (e->device_mode == DEV_MODE_VULKAN)
        e->fwd_block = vulkan_fwd_block;
}
