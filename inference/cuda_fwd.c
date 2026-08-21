/* cuda_fwd.c — CUDA 前向挂接
 *
 * host-shim: blob + engine_fwd_block_at(CPU)
 * 真 GPU: FP16 权 + cublasGemmEx; 激活常驻 d_x(层间不来回拷)
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
static int gemv(Engine* e, CudaCtx* ctx, float* y, uint32_t layer, uint32_t slot,
                const float* x, char* err, size_t errlen)
{
    uint32_t out = 0, in = 0;
    const uint16_t* w = cuda_tensor_f16w(e, layer, slot, &out, &in);
    if (!w) {
        if (err && errlen) snprintf(err, errlen, "missing f16w L%u S%u", layer, slot);
        return -1;
    }
    return cuda_k_gemv_f16(ctx->cublas, y, w, x, ctx->d_xf16, out, in, err, errlen);
}

static int cuda_fwd_block_gpu(Engine* e, uint32_t layer, uint32_t pos)
{
    CudaCtx* ctx = cuda_get_ctx(e);
    if (!ctx || e->device_mode != DEV_MODE_CUDA) return -1;
    char err[256];
    uint32_t hidden = ctx->hidden;
    uint32_t kv_dim = ctx->kv_dim;
    float* x = ctx->d_x;
    float* x2 = ctx->d_hb;
    float* q = ctx->d_hb2;
    float* k = ctx->d_hb2 + hidden;
    float* v = ctx->d_hb2 + hidden + kv_dim;
    float* att_out = ctx->d_hb2 + hidden + 2 * kv_dim;
    float* att = ctx->d_att;
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
    if (qn) {
        uint32_t hh;
        for (hh = 0; hh < ctx->n_heads; hh++)
            cuda_k_rmsnorm(q + (size_t)hh * ctx->head_dim, q + (size_t)hh * ctx->head_dim,
                           qn, ctx->head_dim, ctx->eps);
    }
    const float* kn = cuda_tensor_f32(e, layer, SLOT_KNORM, &n);
    if (kn) {
        uint32_t hh;
        for (hh = 0; hh < ctx->n_kv_heads; hh++)
            cuda_k_rmsnorm(k + (size_t)hh * ctx->head_dim, k + (size_t)hh * ctx->head_dim,
                           kn, ctx->head_dim, ctx->eps);
    }

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

    cuda_k_attn_decode(att_out, att, q, kcache, vcache, pos,
                       ctx->n_heads, ctx->n_kv_heads, ctx->head_dim, kv_dim, e->max_seq);

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

    /* 激活留在 d_x; 需要 host 时再 cuda_sync_x_to_host */
    ctx->x_on_dev = 1;
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

int cuda_embed(Engine* e, uint32_t token)
{
#if defined(YLLM_CUDA) && !defined(YLLM_CUDA_HOST)
    if (e->device_mode != DEV_MODE_CUDA) return -1;
    CudaCtx* ctx = cuda_get_ctx(e);
    uint32_t out = 0, in = 0;
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
    uint32_t out = 0, in = 0;
    const uint16_t* w = cuda_tensor_f16w(e, layer, 0, &out, &in);
    if (!w) return -1;
    if (!ctx->x_on_dev) {
        if (cuda_k_memcpy_h2d(ctx->d_x, e->x, (size_t)ctx->hidden * 4) != 0) return -1;
        ctx->x_on_dev = 1;
    }
    if (cuda_k_gemv_f16(ctx->cublas, ctx->d_logits, w, ctx->d_x, ctx->d_xf16, out, in, err, sizeof(err)) != 0)
        return -1;
    if (cuda_k_memcpy_d2h(e->logits, ctx->d_logits, (size_t)ctx->vocab * 4) != 0) return -1;
    /* 采样只用 logits; 顺带把 x 拉回以免后续 host 读脏 */
    cuda_sync_x_to_host(e);
    return 0;
#else
    (void)e;
    return -1;
#endif
}

void cuda_attach_fwd(Engine* e)
{
    e->fwd_block = cuda_fwd_block;
}
