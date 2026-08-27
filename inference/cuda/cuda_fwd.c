/* cuda_fwd.c — CUDA 前向挂接
 *
 * host-shim: blob + engine_fwd_block_at(CPU)
 * 真 GPU: FP16 权 + cublasGemmEx; 激活常驻; 批 prefill
 */
#include "device.h"
#include "yllm.h"
#include "matvec.h"
#include "cuda_ctx.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(YLLM_CUDA) && !defined(YLLM_CUDA_HOST)
#include "cuda_kernels.h"
#endif

const uint8_t* cuda_layer_base(const Engine* e, uint32_t layer);
CudaCtx* cuda_get_ctx(Engine* e);
const uint8_t* cuda_tensor_q4k(const Engine* e, uint32_t layer, uint32_t slot,
                               uint32_t* out, uint32_t* in);
const uint16_t* cuda_tensor_f16w(const Engine* e, uint32_t layer, uint32_t slot,
                                 uint32_t* out, uint32_t* in);
const float* cuda_tensor_f32(const Engine* e, uint32_t layer, uint32_t slot, uint32_t* n);

static int cuda_fwd_block_shim(Engine* e, uint32_t layer, uint32_t pos)
{
    const uint8_t* base = cuda_layer_base(e, layer);
    uint16_t* kv = e->d_kv ? (uint16_t*)e->d_kv : e->kv;
    if (!base)
        base = (const uint8_t*)e->ws.map.base + e->ws.model.dir[layer].offset;
    return engine_fwd_block_at(e, layer, pos, base, kv);
}

#if defined(YLLM_CUDA) && !defined(YLLM_CUDA_HOST)
static int ensure_stream_layer(Engine* e, uint32_t layer)
{
    if (!e->dev || !e->dev->prefetch_layer) return 0;
    CudaCtx* ctx = cuda_get_ctx(e);
    if (!ctx || !ctx->stream_w) return 0;
    return e->dev->prefetch_layer(e, layer);
}

static int gemv(Engine* e, CudaCtx* ctx, float* y, uint32_t layer, uint32_t slot,
                const float* x, char* err, size_t errlen)
{
    uint32_t out = 0, in = 0;
    const uint8_t* wq = cuda_tensor_q4k(e, layer, slot, &out, &in);
    if (wq)
        return cuda_k_gemv_q4k(y, wq, x, out, in, err, errlen);
    const uint16_t* w = cuda_tensor_f16w(e, layer, slot, &out, &in);
    if (!w) {
        if (err && errlen) snprintf(err, errlen, "missing w L%u S%u", layer, slot);
        return -1;
    }
    return cuda_k_gemv_f16(ctx->cublas, y, w, x, ctx->d_xf16, out, in, err, errlen);
}

static int gemm(Engine* e, CudaCtx* ctx, float* y, uint32_t layer, uint32_t slot,
                const float* x, uint32_t B, char* err, size_t errlen)
{
    uint32_t out = 0, in = 0;
    const uint8_t* wq = cuda_tensor_q4k(e, layer, slot, &out, &in);
    if (wq)
        return cuda_k_gemm_q4k(y, wq, x, out, in, B, err, errlen);
    const uint16_t* w = cuda_tensor_f16w(e, layer, slot, &out, &in);
    if (!w) {
        if (err && errlen) snprintf(err, errlen, "missing w L%u S%u", layer, slot);
        return -1;
    }
    return cuda_k_gemm_f16(ctx->cublas, y, w, x, ctx->d_xf16, out, in, B, err, errlen);
}

static void qk_norm_batch(CudaCtx* ctx, float* q, float* k, const float* qn, const float* kn,
                          uint32_t B)
{
    uint32_t b, hh;
    if (qn) {
        for (b = 0; b < B; b++)
            for (hh = 0; hh < ctx->n_heads; hh++)
                cuda_k_rmsnorm(q + (size_t)b * ctx->hidden + (size_t)hh * ctx->head_dim,
                               q + (size_t)b * ctx->hidden + (size_t)hh * ctx->head_dim,
                               qn, ctx->head_dim, ctx->eps);
    }
    if (kn) {
        for (b = 0; b < B; b++)
            for (hh = 0; hh < ctx->n_kv_heads; hh++)
                cuda_k_rmsnorm(k + (size_t)b * ctx->kv_dim + (size_t)hh * ctx->head_dim,
                               k + (size_t)b * ctx->kv_dim + (size_t)hh * ctx->head_dim,
                               kn, ctx->head_dim, ctx->eps);
    }
}

static int cuda_fwd_block_gpu(Engine* e, uint32_t layer, uint32_t pos)
{
    CudaCtx* ctx = cuda_get_ctx(e);
    if (!ctx || e->device_mode != DEV_MODE_CUDA) return -1;
    if (ensure_stream_layer(e, layer) != 0) return -1;
    char err[256];
    uint32_t hidden = ctx->hidden;
    uint32_t kv_dim = ctx->kv_dim;
    float* x = ctx->d_x;
    float* x2 = ctx->d_hb;
    float* q = ctx->d_hb2;
    float* k = ctx->d_hb2 + hidden;
    float* v = ctx->d_hb2 + hidden + kv_dim;
    float* att_out = ctx->d_hb2 + hidden + 2 * kv_dim;
    float* fg = ctx->d_ffn;
    float* fu = ctx->d_ffn + ctx->inter;

    if (!ctx->x_on_dev) {
        if (cuda_k_memcpy_h2d(x, e->x, (size_t)hidden * 4) != 0) return -1;
        ctx->x_on_dev = 1;
    }

    uint32_t n = 0;
    const float* wn1 = cuda_tensor_f32(e, layer, SLOT_NORM1, &n);
    if (!wn1) return -1;
    cuda_k_rmsnorm(x2, x, wn1, hidden, ctx->eps);

    if (gemv(e, ctx, q, layer, SLOT_Q, x2, err, sizeof(err)) != 0) return -1;
    if (gemv(e, ctx, k, layer, SLOT_K, x2, err, sizeof(err)) != 0) return -1;
    if (gemv(e, ctx, v, layer, SLOT_V, x2, err, sizeof(err)) != 0) return -1;

    const float* bq = cuda_tensor_f32(e, layer, SLOT_QBIAS, &n);
    if (bq) cuda_k_add_bias(q, bq, hidden);
    const float* bk = cuda_tensor_f32(e, layer, SLOT_KBIAS, &n);
    if (bk) cuda_k_add_bias(k, bk, kv_dim);
    const float* bv = cuda_tensor_f32(e, layer, SLOT_VBIAS, &n);
    if (bv) cuda_k_add_bias(v, bv, kv_dim);

    const float* qn = cuda_tensor_f32(e, layer, SLOT_QNORM, &n);
    const float* kn = cuda_tensor_f32(e, layer, SLOT_KNORM, &n);
    qk_norm_batch(ctx, q, k, qn, kn, 1);

    if (ctx->arch == ARCH_QWEN) {
        cuda_k_rope_qwen_heads(q, ctx->n_heads, ctx->head_dim, pos, ctx->theta);
        cuda_k_rope_qwen_heads(k, ctx->n_kv_heads, ctx->head_dim, pos, ctx->theta);
    } else {
        cuda_k_rope_llama_heads(q, ctx->n_heads, ctx->head_dim, pos, ctx->theta);
        cuda_k_rope_llama_heads(k, ctx->n_kv_heads, ctx->head_dim, pos, ctx->theta);
    }

    uint16_t* kv = ctx->kv_blob;
    uint16_t* kcache = kv + (size_t)layer * e->max_seq * kv_dim;
    uint16_t* vcache = kv + (size_t)(ctx->n_blocks + layer) * e->max_seq * kv_dim;
    cuda_k_store_kv(kcache, vcache, k, v, pos, kv_dim);

    cuda_k_attn_decode(att_out, q, kcache, vcache, pos,
                       ctx->n_heads, ctx->n_kv_heads, ctx->head_dim, kv_dim);

    if (cuda_k_memcpy_d2d(x2, att_out, (size_t)hidden * 4) != 0) return -1;
    if (gemv(e, ctx, att_out, layer, SLOT_O, x2, err, sizeof(err)) != 0) return -1;
    cuda_k_add(x, x, att_out, hidden);

    const float* wn2 = cuda_tensor_f32(e, layer, SLOT_NORM2, &n);
    if (!wn2) return -1;
    cuda_k_rmsnorm(x2, x, wn2, hidden, ctx->eps);

    if (gemv(e, ctx, fg, layer, SLOT_GATE, x2, err, sizeof(err)) != 0) return -1;
    if (gemv(e, ctx, fu, layer, SLOT_UP, x2, err, sizeof(err)) != 0) return -1;
    cuda_k_swiglu(x2, fg, fu, ctx->inter);
    if (gemv(e, ctx, att_out, layer, SLOT_DOWN, x2, err, sizeof(err)) != 0) return -1;
    cuda_k_add(x, x, att_out, hidden);

    ctx->x_on_dev = 1;
    return 0;
}

static int cuda_fwd_block_batch_gpu(Engine* e, uint32_t layer, uint32_t pos_start, uint32_t B)
{
    CudaCtx* ctx = cuda_get_ctx(e);
    if (!ctx || e->device_mode != DEV_MODE_CUDA || B == 0 || B > ctx->pb_cap) return -1;
    if (ensure_stream_layer(e, layer) != 0) return -1;

    char err[256];
    uint32_t hidden = ctx->hidden;
    uint32_t kv_dim = ctx->kv_dim;
    float* x = ctx->d_pb;
    float* x2 = ctx->d_pb2;
    float* q = ctx->d_pbq;
    float* k = ctx->d_pbk;
    float* v = ctx->d_pbv;
    float* fg = ctx->d_pbg;
    float* fu = ctx->d_pbu;
    uint32_t n = 0;

    const float* wn1 = cuda_tensor_f32(e, layer, SLOT_NORM1, &n);
    if (!wn1) return -1;
    cuda_k_rmsnorm_batch(x2, x, wn1, hidden, ctx->eps, B);

    if (gemm(e, ctx, q, layer, SLOT_Q, x2, B, err, sizeof(err)) != 0) return -1;
    if (gemm(e, ctx, k, layer, SLOT_K, x2, B, err, sizeof(err)) != 0) return -1;
    if (gemm(e, ctx, v, layer, SLOT_V, x2, B, err, sizeof(err)) != 0) return -1;

    const float* bq = cuda_tensor_f32(e, layer, SLOT_QBIAS, &n);
    if (bq) cuda_k_add_bias_batch(q, bq, hidden, B);
    const float* bk = cuda_tensor_f32(e, layer, SLOT_KBIAS, &n);
    if (bk) cuda_k_add_bias_batch(k, bk, kv_dim, B);
    const float* bv = cuda_tensor_f32(e, layer, SLOT_VBIAS, &n);
    if (bv) cuda_k_add_bias_batch(v, bv, kv_dim, B);

    const float* qn = cuda_tensor_f32(e, layer, SLOT_QNORM, &n);
    const float* kn = cuda_tensor_f32(e, layer, SLOT_KNORM, &n);
    qk_norm_batch(ctx, q, k, qn, kn, B);

    if (ctx->arch == ARCH_QWEN) {
        cuda_k_rope_qwen_heads_batch(q, ctx->n_heads, ctx->head_dim, pos_start, B, ctx->theta);
        cuda_k_rope_qwen_heads_batch(k, ctx->n_kv_heads, ctx->head_dim, pos_start, B, ctx->theta);
    } else {
        cuda_k_rope_llama_heads_batch(q, ctx->n_heads, ctx->head_dim, pos_start, B, ctx->theta);
        cuda_k_rope_llama_heads_batch(k, ctx->n_kv_heads, ctx->head_dim, pos_start, B, ctx->theta);
    }

    uint16_t* kv = ctx->kv_blob;
    uint16_t* kcache = kv + (size_t)layer * e->max_seq * kv_dim;
    uint16_t* vcache = kv + (size_t)(ctx->n_blocks + layer) * e->max_seq * kv_dim;
    cuda_k_store_kv_batch(kcache, vcache, k, v, pos_start, B, kv_dim);

    cuda_k_attn_prefill(x2, q, kcache, vcache, pos_start, B,
                        ctx->n_heads, ctx->n_kv_heads, ctx->head_dim, kv_dim, hidden);

    if (gemm(e, ctx, q, layer, SLOT_O, x2, B, err, sizeof(err)) != 0) return -1;
    cuda_k_add_batch(x, x, q, hidden, B);

    const float* wn2 = cuda_tensor_f32(e, layer, SLOT_NORM2, &n);
    if (!wn2) return -1;
    cuda_k_rmsnorm_batch(x2, x, wn2, hidden, ctx->eps, B);

    if (gemm(e, ctx, fg, layer, SLOT_GATE, x2, B, err, sizeof(err)) != 0) return -1;
    if (gemm(e, ctx, fu, layer, SLOT_UP, x2, B, err, sizeof(err)) != 0) return -1;
    cuda_k_swiglu_batch(fg, fg, fu, ctx->inter, B);
    if (gemm(e, ctx, q, layer, SLOT_DOWN, fg, B, err, sizeof(err)) != 0) return -1;
    cuda_k_add_batch(x, x, q, hidden, B);
    return 0;
}
#endif

static int cuda_fwd_block(Engine* e, uint32_t layer, uint32_t pos)
{
#if defined(YLLM_CUDA) && !defined(YLLM_CUDA_HOST)
    if (e->device_mode == DEV_MODE_CUDA)
        return cuda_fwd_block_gpu(e, layer, pos);
#endif
    return cuda_fwd_block_shim(e, layer, pos);
}

static int cuda_fwd_block_batch(Engine* e, uint32_t layer, uint32_t pos_start, uint32_t B)
{
#if defined(YLLM_CUDA) && !defined(YLLM_CUDA_HOST)
    if (e->device_mode == DEV_MODE_CUDA)
        return cuda_fwd_block_batch_gpu(e, layer, pos_start, B);
#endif
    /* shim: 逐 token */
    uint32_t b;
    for (b = 0; b < B; b++) {
        if (cuda_fwd_block_shim(e, layer, pos_start + b) != 0) return -1;
    }
    return 0;
}

void cuda_sync_x_to_host(Engine* e)
{
#if defined(YLLM_CUDA) && !defined(YLLM_CUDA_HOST)
    CudaCtx* ctx = cuda_get_ctx(e);
    if (!ctx || e->device_mode != DEV_MODE_CUDA || !ctx->x_on_dev || !ctx->d_x || !e->x) return;
    cuda_k_memcpy_d2h(e->x, ctx->d_x, (size_t)ctx->hidden * 4);
#else
    (void)e;
#endif
}

void cuda_mark_x_host(Engine* e)
{
#if defined(YLLM_CUDA) && !defined(YLLM_CUDA_HOST)
    CudaCtx* ctx = cuda_get_ctx(e);
    if (ctx) ctx->x_on_dev = 0;
#else
    (void)e;
#endif
}

int cuda_embed(Engine* e, uint32_t token)
{
#if defined(YLLM_CUDA) && !defined(YLLM_CUDA_HOST)
    if (e->device_mode != DEV_MODE_CUDA) return -1;
    if (ensure_stream_layer(e, 0) != 0) return -1;
    CudaCtx* ctx = cuda_get_ctx(e);
    uint32_t out = 0, in = 0;
    const uint8_t* eq = cuda_tensor_q4k(e, 0, SLOT_EMBED, &out, &in);
    if (eq) {
        cuda_k_embed_q4k(ctx->d_x, eq, token, ctx->hidden);
        ctx->x_on_dev = 1;
        return 0;
    }
    const uint16_t* emb = cuda_tensor_f16w(e, 0, SLOT_EMBED, &out, &in);
    if (!emb) return -1;
    cuda_k_embed_f16(ctx->d_x, emb, token, ctx->hidden);
    ctx->x_on_dev = 1;
    return 0;
#else
    (void)e;
    (void)token;
    return -1;
#endif
}

int cuda_final_norm(Engine* e)
{
#if defined(YLLM_CUDA) && !defined(YLLM_CUDA_HOST)
    if (e->device_mode != DEV_MODE_CUDA) return -1;
    CudaCtx* ctx = cuda_get_ctx(e);
    uint32_t layer = ctx->n_blocks + 1;
    if (ensure_stream_layer(e, layer) != 0) return -1;
    uint32_t n = 0;
    const float* w = cuda_tensor_f32(e, layer, 0, &n);
    if (!w) return -1;
    if (!ctx->x_on_dev) {
        if (cuda_k_memcpy_h2d(ctx->d_x, e->x, (size_t)ctx->hidden * 4) != 0) return -1;
        ctx->x_on_dev = 1;
    }
    cuda_k_rmsnorm(ctx->d_x, ctx->d_x, w, ctx->hidden, ctx->eps);
    ctx->x_on_dev = 1;
    return 0;
#else
    (void)e;
    return -1;
#endif
}

int cuda_lm_head(Engine* e)
{
#if defined(YLLM_CUDA) && !defined(YLLM_CUDA_HOST)
    if (e->device_mode != DEV_MODE_CUDA) return -1;
    CudaCtx* ctx = cuda_get_ctx(e);
    char err[256];
    uint32_t layer = ctx->n_blocks + 2;
    if (ensure_stream_layer(e, layer) != 0) return -1;
    uint32_t out = 0, in = 0;
    if (!ctx->x_on_dev) {
        if (cuda_k_memcpy_h2d(ctx->d_x, e->x, (size_t)ctx->hidden * 4) != 0) return -1;
        ctx->x_on_dev = 1;
    }
    const uint8_t* wq = cuda_tensor_q4k(e, layer, 0, &out, &in);
    if (wq) {
        if (cuda_k_gemv_q4k(ctx->d_logits, wq, ctx->d_x, out, in, err, sizeof(err)) != 0)
            return -1;
    } else {
        const uint16_t* w = cuda_tensor_f16w(e, layer, 0, &out, &in);
        if (!w) return -1;
        if (cuda_k_gemv_f16(ctx->cublas, ctx->d_logits, w, ctx->d_x, ctx->d_xf16, out, in, err, sizeof(err)) != 0)
            return -1;
    }
    if (cuda_k_memcpy_d2h(e->logits, ctx->d_logits, (size_t)ctx->vocab * 4) != 0) return -1;
    cuda_sync_x_to_host(e);
    return 0;
#else
    (void)e;
    return -1;
#endif
}

int cuda_prefill(Engine* e, const uint32_t* tokens, int n, int start_pos)
{
#if defined(YLLM_CUDA) && !defined(YLLM_CUDA_HOST)
    if (e->device_mode != DEV_MODE_CUDA || !tokens || n <= 0) return -1;
    /* 单进程层切混合: 全 GPU 批路径不适用, 交回逐 token */
    if (e->gpu_layer_end) return -1;
    CudaCtx* ctx = cuda_get_ctx(e);
    if (!ctx || !ctx->d_pb || ctx->pb_cap == 0) return -1;

    uint32_t out = 0, in = 0;
    const uint8_t* eq = cuda_tensor_q4k(e, 0, SLOT_EMBED, &out, &in);
    const uint16_t* emb = eq ? NULL : cuda_tensor_f16w(e, 0, SLOT_EMBED, &out, &in);
    if (!eq && !emb) return -1;

        uint32_t trunk = ctx->n_blocks - (e->mtp_layer ? 1u : 0u);
        uint32_t layer_lo = e->layer_begin ? e->layer_begin : 1;
        uint32_t layer_hi = e->layer_end;
        if (layer_hi > trunk + 1) layer_hi = trunk + 1;
        int off = 0;
        while (off < n) {
            uint32_t nb = (uint32_t)(n - off);
            if (nb > ctx->pb_cap) nb = ctx->pb_cap;

            if (cuda_k_memcpy_h2d(ctx->d_tokens, tokens + off, (size_t)nb * sizeof(uint32_t)) != 0)
                return -1;
            if (eq)
                cuda_k_embed_q4k_batch(ctx->d_pb, eq, ctx->d_tokens, nb, ctx->hidden);
            else
                cuda_k_embed_f16_batch(ctx->d_pb, emb, ctx->d_tokens, nb, ctx->hidden);

            uint32_t layer;
            for (layer = layer_lo; layer < layer_hi && layer <= trunk; layer++) {
                if (layer == 0) continue;
                if (cuda_fwd_block_batch_gpu(e, layer, (uint32_t)(start_pos + off), nb) != 0)
                    return -1;
            }

        if (e->mtp_h) {
            if (cuda_k_memcpy_d2h(e->mtp_h,
                                  ctx->d_pb + (size_t)(nb - 1) * ctx->hidden,
                                  (size_t)ctx->hidden * 4) != 0)
                return -1;
            e->mtp_h_ready = 1;
        }

        if (off + (int)nb == n) {
            if (cuda_k_memcpy_d2d(ctx->d_x, ctx->d_pb + (size_t)(nb - 1) * ctx->hidden,
                                  (size_t)ctx->hidden * 4) != 0)
                return -1;
            ctx->x_on_dev = 1;
            if (cuda_final_norm(e) != 0) return -1;
            if (cuda_lm_head(e) != 0) return -1;
        }
        off += (int)nb;
    }
    return 0;
#else
    (void)e;
    (void)tokens;
    (void)n;
    (void)start_pos;
    return -1;
#endif
}

int cuda_forward_batch_x(Engine* e, const float* xin, int n, uint32_t pos,
                         float* x_out, float* logits_out)
{
#if defined(YLLM_CUDA) && !defined(YLLM_CUDA_HOST)
    if (e->device_mode != DEV_MODE_CUDA || !xin || n < 1) return -1;
    if (e->gpu_layer_end) return -1; /* 混合层切: 走逐 token 回退 */
    CudaCtx* ctx = cuda_get_ctx(e);
    if (!ctx || !ctx->d_pb || (uint32_t)n > ctx->pb_cap) return -1;

    uint32_t hidden = ctx->hidden;
    uint32_t nb = (uint32_t)n;
    if (cuda_k_memcpy_h2d(ctx->d_pb, xin, (size_t)nb * hidden * 4) != 0) return -1;

    uint32_t trunk = ctx->n_blocks - (e->mtp_layer ? 1u : 0u);
    uint32_t layer;
    uint32_t begin = e->layer_begin ? e->layer_begin : 1;
    uint32_t end = e->layer_end;
    if (end > trunk + 1) end = trunk + 1; /* 先跑块; norm/head 另处理 */
    for (layer = begin; layer < end && layer <= trunk; layer++) {
        if (layer == 0) continue;
        if (cuda_fwd_block_batch_gpu(e, layer, pos, nb) != 0) return -1;
    }

    if (x_out) {
        if (cuda_k_memcpy_d2h(x_out, ctx->d_pb, (size_t)nb * hidden * 4) != 0) return -1;
    }

    if (logits_out) {
        /* 末段: 最后 token → final norm + lm_head */
        if (e->layer_end <= ctx->n_blocks) return -1;
        if (cuda_k_memcpy_d2d(ctx->d_x, ctx->d_pb + (size_t)(nb - 1) * hidden,
                              (size_t)hidden * 4) != 0)
            return -1;
        ctx->x_on_dev = 1;
        if (cuda_final_norm(e) != 0) return -1;
        if (cuda_lm_head(e) != 0) return -1;
        memcpy(logits_out, e->logits, (size_t)ctx->vocab * 4);
    }
    return 0;
#else
    (void)e;
    (void)xin;
    (void)n;
    (void)pos;
    (void)x_out;
    (void)logits_out;
    return -1;
#endif
}

void cuda_attach_fwd(Engine* e)
{
    if (!e || !e->dev) return;
    e->dev->fwd_block = cuda_fwd_block;
#if defined(YLLM_CUDA) && !defined(YLLM_CUDA_HOST)
    if (e->device_mode == DEV_MODE_CUDA)
        e->dev->fwd_block_batch = cuda_fwd_block_batch;
#else
    e->dev->fwd_block_batch = NULL;
    (void)cuda_fwd_block_batch;
#endif
}
