/* device_cuda.c — CUDA 设备后端
 *
 * YLLM_CUDA_HOST=1: 权镜像主机堆 + CPU 算子
 * 真 CUDA: 线性权解量化 FP16 上卡 + cublas decode(见 cuda_fwd / cuda_kernels)
 */
#include "device.h"
#include "yllm.h"
#include "matvec.h"
#include "log.h"
#include "cuda_ctx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(YLLM_CUDA) && !defined(YLLM_CUDA_HOST)
#include <cuda_runtime.h>
#include "cuda_kernels.h"
#define CUDA_OK(call, err, errlen) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        if (err && errlen) snprintf(err, errlen, "CUDA: %s", cudaGetErrorString(_e)); \
        return -1; \
    } \
} while (0)
#endif

void cuda_attach_fwd(Engine* e);

static uint32_t slot_count(const LlModel* m, uint32_t layer)
{
    uint32_t nt = m->dir[layer].n_tensors;
    if (nt > BLOCK_TENSORS_MTP) nt = BLOCK_TENSORS_MTP;
    return nt;
}

static int tensor_out_in(const LlfTensorMeta* mt, uint32_t* out, uint32_t* in)
{
    if (mt->ndim < 2 || mt->shape[0] == 0 || mt->shape[1] == 0) return -1;
    uint32_t a = mt->shape[0], b = mt->shape[1];
    *out = a;
    *in = b;
    if (mt->dtype == DT_Q4K || mt->dtype == DT_IQ4XS) {
        size_t e0 = (size_t)a * (b / 256) * 144;
        size_t e1 = (size_t)b * (a / 256) * 144;
        if (e0 != mt->size && e1 == mt->size) { *out = b; *in = a; }
        else if (e0 != mt->size && e1 != mt->size) return -1;
    } else if (mt->dtype == DT_Q6K) {
        size_t e0 = (size_t)a * (b / 256) * 210;
        size_t e1 = (size_t)b * (a / 256) * 210;
        if (e0 != mt->size && e1 == mt->size) { *out = b; *in = a; }
        else if (e0 != mt->size && e1 != mt->size) return -1;
    } else if (mt->dtype == DT_Q5K) {
        size_t e0 = (size_t)a * (b / 256) * 176;
        size_t e1 = (size_t)b * (a / 256) * 176;
        if (e0 != mt->size && e1 == mt->size) { *out = b; *in = a; }
        else if (e0 != mt->size && e1 != mt->size) return -1;
    } else if (mt->dtype == DT_F16 || mt->dtype == DT_BF16) {
        size_t e0 = (size_t)a * b * 2;
        size_t e1 = (size_t)b * a * 2;
        if (e0 != mt->size && e1 == mt->size) { *out = b; *in = a; }
    } else if (mt->dtype == DT_F32) {
        size_t e0 = (size_t)a * b * 4;
        size_t e1 = (size_t)b * a * 4;
        if (e0 != mt->size && e1 == mt->size) { *out = b; *in = a; }
    }
    return 0;
}

/* 按引擎 matmul 约定固定 out/in, 再用文件 size 校验 */
static int slot_out_in(uint32_t layer, uint32_t slot, const Engine* e,
                       const LlfTensorMeta* mt, uint32_t* out, uint32_t* in)
{
    const LlfHeader* h = &e->ws.model.h;
    uint32_t hidden = h->hidden;
    uint32_t kv_dim = h->n_kv_heads * h->head_dim;
    uint32_t inter = e->inter;
    uint32_t vocab = h->vocab;
    if (layer == 0 && slot == SLOT_EMBED) {
        *out = vocab;
        *in = hidden;
    } else if (layer >= 1 && layer <= h->n_blocks) {
        switch (slot) {
        case SLOT_Q: case SLOT_O: *out = hidden; *in = hidden; break;
        case SLOT_K: case SLOT_V: *out = kv_dim; *in = hidden; break;
        case SLOT_GATE: case SLOT_UP: *out = inter; *in = hidden; break;
        case SLOT_DOWN: *out = hidden; *in = inter; break;
        default:
            return tensor_out_in(mt, out, in);
        }
    } else if (layer == h->n_blocks + 2) {
        *out = vocab;
        *in = hidden;
    } else {
        return tensor_out_in(mt, out, in);
    }
    /* 校验量化行字节与 size 一致 */
    if (mt->dtype == DT_Q4K || mt->dtype == DT_IQ4XS) {
        size_t expect = (size_t)(*out) * (*in / 256) * 144;
        if (expect != mt->size) return -1;
    } else if (mt->dtype == DT_Q6K) {
        size_t expect = (size_t)(*out) * (*in / 256) * 210;
        if (expect != mt->size) return -1;
    } else if (mt->dtype == DT_Q5K) {
        size_t expect = (size_t)(*out) * (*in / 256) * 176;
        if (expect != mt->size) return -1;
    }
    return 0;
}

static int is_linear_slot(uint32_t layer, uint32_t slot, const LlfHeader* h)
{
    if (layer == 0) return slot == SLOT_EMBED;
    if (layer >= 1 && layer <= h->n_blocks)
        return slot == SLOT_Q || slot == SLOT_K || slot == SLOT_V || slot == SLOT_O ||
               slot == SLOT_GATE || slot == SLOT_UP || slot == SLOT_DOWN;
    if (layer == h->n_blocks + 2) return 1; /* lm_head (+ MTP linears) */
    return 0;
}

static int is_f32vec_slot(uint32_t layer, uint32_t slot, const LlfHeader* h, const LlfTensorMeta* mt)
{
    if (mt->size == 0) return 0;
    if (layer >= 1 && layer <= h->n_blocks) {
        if (slot == SLOT_NORM1 || slot == SLOT_NORM2 ||
            slot == SLOT_QBIAS || slot == SLOT_KBIAS || slot == SLOT_VBIAS ||
            slot == SLOT_QNORM || slot == SLOT_KNORM)
            return 1;
    }
    if (layer == h->n_blocks + 1) return 1; /* final norm */
    (void)mt;
    return 0;
}

static void cuda_ctx_clear(CudaCtx* ctx)
{
#if defined(YLLM_CUDA) && !defined(YLLM_CUDA_HOST)
    if (!ctx->host_shim) {
        if (ctx->w_blob) cudaFree(ctx->w_blob);
        if (ctx->w_q4) cudaFree(ctx->w_q4);
        if (ctx->w_f16) cudaFree(ctx->w_f16);
        if (ctx->w_f32) cudaFree(ctx->w_f32);
        if (ctx->kv_blob) cudaFree(ctx->kv_blob);
        if (ctx->d_x) cudaFree(ctx->d_x);
        if (ctx->d_hb) cudaFree(ctx->d_hb);
        if (ctx->d_hb2) cudaFree(ctx->d_hb2);
        if (ctx->d_ffn) cudaFree(ctx->d_ffn);
        if (ctx->d_logits) cudaFree(ctx->d_logits);
        if (ctx->d_xf16) cudaFree(ctx->d_xf16);
        if (ctx->d_pb) cudaFree(ctx->d_pb);
        if (ctx->d_pb2) cudaFree(ctx->d_pb2);
        if (ctx->d_pbq) cudaFree(ctx->d_pbq);
        if (ctx->d_pbk) cudaFree(ctx->d_pbk);
        if (ctx->d_pbv) cudaFree(ctx->d_pbv);
        if (ctx->d_pbg) cudaFree(ctx->d_pbg);
        if (ctx->d_pbu) cudaFree(ctx->d_pbu);
        if (ctx->d_tokens) cudaFree(ctx->d_tokens);
        if (ctx->cublas) cuda_k_cublas_destroy(ctx->cublas);
    } else
#endif
    {
        free(ctx->w_blob);
        free(ctx->kv_blob);
    }
    free(ctx->layer_off);
    free(ctx->off_q4);
    free(ctx->off_f16);
    free(ctx->off_f32);
    free(ctx->host_off_q4);
    free(ctx->host_off_f16);
    free(ctx->host_off_f32);
    free(ctx->h_q4);
    free(ctx->h_f16);
    free(ctx->h_f32);
    free(ctx->dim_out);
    free(ctx->dim_in);
    memset(ctx, 0, sizeof(*ctx));
}

#if defined(YLLM_CUDA) && !defined(YLLM_CUDA_HOST)
static int load_weights_gpu(Engine* e, CudaCtx* ctx, char* err, size_t errlen)
{
    LlModel* m = &e->ws.model;
    const LlfHeader* h = &m->h;
    uint32_t begin = e->layer_begin;
    uint32_t end = e->layer_end;
    /* 单进程混合: 只把 [begin, gpu_layer_end) 上卡, 其余走 mmap CPU */
    if (e->gpu_layer_end && e->gpu_layer_end < end) end = e->gpu_layer_end;
    if (end > m->n_layers) end = m->n_layers;
    if (begin > end) begin = end;

    uint32_t nslot = BLOCK_TENSORS_MTP;
    size_t tab = (size_t)m->n_layers * nslot;
    ctx->n_layers = m->n_layers;
    ctx->off_q4 = (uint64_t*)malloc(tab * sizeof(uint64_t));
    ctx->off_f16 = (uint64_t*)malloc(tab * sizeof(uint64_t));
    ctx->off_f32 = (uint64_t*)malloc(tab * sizeof(uint64_t));
    ctx->dim_out = (uint32_t*)calloc(tab, sizeof(uint32_t));
    ctx->dim_in = (uint32_t*)calloc(tab, sizeof(uint32_t));
    if (!ctx->off_q4 || !ctx->off_f16 || !ctx->off_f32 || !ctx->dim_out || !ctx->dim_in) {
        if (err && errlen) snprintf(err, errlen, "oom tensor tables");
        return -1;
    }
    uint32_t i, s;
    for (i = 0; i < tab; i++) {
        ctx->off_q4[i] = CUDA_OFF_NONE;
        ctx->off_f16[i] = CUDA_OFF_NONE;
        ctx->off_f32[i] = CUDA_OFF_NONE;
    }

    /* 统计 Q4_K 字节 / FP16 元素 / F32 向量 */
    uint64_t n_q4 = 0, n_f16 = 0, n_f32 = 0;
    const uint8_t* map = (const uint8_t*)e->ws.map.base;
    for (i = begin; i < end; i++) {
        uint32_t nt = slot_count(m, i);
        const LlfTensorMeta* mt0 = &m->metas[m->base_idx[i]];
        for (s = 0; s < nt; s++) {
            const LlfTensorMeta* mt = &mt0[s];
            if (mt->size == 0) continue;
            size_t idx = (size_t)i * nslot + s;
            if (is_linear_slot(i, s, h) && mt->ndim >= 2) {
                uint32_t out = 0, in = 0;
                if (slot_out_in(i, s, e, mt, &out, &in) != 0) {
                    if (err && errlen)
                        snprintf(err, errlen, "bad linear shape layer %u slot %u size=%llu",
                                 i, s, (unsigned long long)mt->size);
                    return -1;
                }
                ctx->dim_out[idx] = out;
                ctx->dim_in[idx] = in;
                /* FP16 模式: 全部解量化; AUTO/Q4K: 仅 DT_Q4K 走原生 */
                if (e->cuda_wmode != CUDA_W_FP16 && mt->dtype == DT_Q4K) {
                    if (in % 256 != 0) {
                        if (err && errlen)
                            snprintf(err, errlen, "Q4_K in%%256!=0 L%u S%u in=%u", i, s, in);
                        return -1;
                    }
                    ctx->off_q4[idx] = n_q4;
                    n_q4 += (uint64_t)out * (in / 256) * 144;
                } else {
                    ctx->off_f16[idx] = n_f16;
                    n_f16 += (uint64_t)out * in;
                }
            } else if (is_f32vec_slot(i, s, h, mt)) {
                uint32_t n = mt->shape[0] ? mt->shape[0] : (uint32_t)(mt->size / 4);
                if (mt->dtype == DT_F16 || mt->dtype == DT_BF16)
                    n = (uint32_t)(mt->size / 2);
                else if (mt->dtype == DT_F32)
                    n = (uint32_t)(mt->size / 4);
                ctx->off_f32[idx] = n_f32;
                ctx->dim_out[idx] = n;
                ctx->dim_in[idx] = 1;
                n_f32 += n;
            }
        }
    }

    uint8_t* h_q4 = n_q4 ? (uint8_t*)malloc((size_t)n_q4) : NULL;
    uint16_t* h_f16 = n_f16 ? (uint16_t*)malloc((size_t)n_f16 * 2) : NULL;
    float* h_f32 = n_f32 ? (float*)malloc((size_t)n_f32 * 4) : NULL;
    if ((n_q4 && !h_q4) || (n_f16 && !h_f16) || (n_f32 && !h_f32)) {
        free(h_q4); free(h_f16); free(h_f32);
        if (err && errlen) snprintf(err, errlen, "oom host weights");
        return -1;
    }

    for (i = begin; i < end; i++) {
        uint32_t nt = slot_count(m, i);
        const LlfTensorMeta* mt0 = &m->metas[m->base_idx[i]];
        const uint8_t* lbase = map + m->dir[i].offset;
        for (s = 0; s < nt; s++) {
            const LlfTensorMeta* mt = &mt0[s];
            size_t idx = (size_t)i * nslot + s;
            if (ctx->off_q4[idx] != CUDA_OFF_NONE) {
                uint32_t out = ctx->dim_out[idx], in = ctx->dim_in[idx];
                size_t nbytes = (size_t)out * (in / 256) * 144;
                memcpy(h_q4 + ctx->off_q4[idx], lbase + mt->offset, nbytes);
            } else if (ctx->off_f16[idx] != CUDA_OFF_NONE) {
                uint32_t out = ctx->dim_out[idx], in = ctx->dim_in[idx];
                if (dequant_mat_f16(h_f16 + ctx->off_f16[idx], lbase + mt->offset,
                                    out, in, mt->dtype) != 0) {
                    free(h_q4); free(h_f16); free(h_f32);
                    if (err && errlen)
                        snprintf(err, errlen, "dequant layer %u slot %u dtype %u", i, s, mt->dtype);
                    return -1;
                }
            } else if (ctx->off_f32[idx] != CUDA_OFF_NONE) {
                uint32_t n = ctx->dim_out[idx];
                float* dst = h_f32 + ctx->off_f32[idx];
                if (mt->dtype == DT_F32) {
                    memcpy(dst, lbase + mt->offset, (size_t)n * 4);
                } else {
                    const uint16_t* src = (const uint16_t*)(lbase + mt->offset);
                    uint32_t j;
                    for (j = 0; j < n; j++) {
                        uint16_t h16 = (mt->dtype == DT_BF16) ? bf16_to_f16(src[j]) : src[j];
                        dst[j] = f16_to_f32(h16);
                    }
                }
            }
        }
    }

    ctx->stream_w = e->cuda_stream_w ? 1 : 0;
    ctx->stream_layer = ~0u;
    if (ctx->stream_w) {
        /* 权常驻 host; 设备只开单层峰值缓冲, 由 prefetch_layer 上卡 */
        size_t tab = (size_t)m->n_layers * nslot;
        ctx->host_off_q4 = (uint64_t*)malloc(tab * sizeof(uint64_t));
        ctx->host_off_f16 = (uint64_t*)malloc(tab * sizeof(uint64_t));
        ctx->host_off_f32 = (uint64_t*)malloc(tab * sizeof(uint64_t));
        if (!ctx->host_off_q4 || !ctx->host_off_f16 || !ctx->host_off_f32) {
            free(h_q4); free(h_f16); free(h_f32);
            if (err && errlen) snprintf(err, errlen, "oom host_off");
            return -1;
        }
        memcpy(ctx->host_off_q4, ctx->off_q4, tab * sizeof(uint64_t));
        memcpy(ctx->host_off_f16, ctx->off_f16, tab * sizeof(uint64_t));
        memcpy(ctx->host_off_f32, ctx->off_f32, tab * sizeof(uint64_t));
        for (i = 0; i < tab; i++) {
            ctx->off_q4[i] = CUDA_OFF_NONE;
            ctx->off_f16[i] = CUDA_OFF_NONE;
            ctx->off_f32[i] = CUDA_OFF_NONE;
        }
        ctx->h_q4 = h_q4;
        ctx->h_f16 = h_f16;
        ctx->h_f32 = h_f32;
        h_q4 = NULL;
        h_f16 = NULL;
        h_f32 = NULL;

        uint64_t max_q4 = 0, max_f16 = 0, max_f32 = 0;
        for (i = begin; i < end; i++) {
            uint64_t lq = 0, lf = 0, l32 = 0;
            uint32_t nt = slot_count(m, i);
            for (s = 0; s < nt; s++) {
                size_t idx = (size_t)i * nslot + s;
                if (ctx->host_off_q4[idx] != CUDA_OFF_NONE) {
                    uint32_t out = ctx->dim_out[idx], in = ctx->dim_in[idx];
                    lq += (uint64_t)out * (in / 256) * 144;
                } else if (ctx->host_off_f16[idx] != CUDA_OFF_NONE) {
                    lf += (uint64_t)ctx->dim_out[idx] * ctx->dim_in[idx];
                } else if (ctx->host_off_f32[idx] != CUDA_OFF_NONE) {
                    l32 += ctx->dim_out[idx];
                }
            }
            if (lq > max_q4) max_q4 = lq;
            if (lf > max_f16) max_f16 = lf;
            if (l32 > max_f32) max_f32 = l32;
        }
        ctx->max_layer_q4 = max_q4;
        ctx->max_layer_f16 = max_f16;
        ctx->max_layer_f32 = max_f32;
        ctx->n_q4 = max_q4;
        ctx->n_f16 = max_f16;
        ctx->n_f32 = max_f32;
        if (max_q4)
            CUDA_OK(cudaMalloc((void**)&ctx->w_q4, (size_t)max_q4), err, errlen);
        if (max_f16)
            CUDA_OK(cudaMalloc((void**)&ctx->w_f16, (size_t)max_f16 * 2), err, errlen);
        if (max_f32)
            CUDA_OK(cudaMalloc((void**)&ctx->w_f32, (size_t)max_f32 * 4), err, errlen);
    } else {
        if (n_q4)
            CUDA_OK(cudaMalloc((void**)&ctx->w_q4, (size_t)n_q4), err, errlen);
        if (n_f16)
            CUDA_OK(cudaMalloc((void**)&ctx->w_f16, (size_t)n_f16 * 2), err, errlen);
        if (n_f32)
            CUDA_OK(cudaMalloc((void**)&ctx->w_f32, (size_t)n_f32 * 4), err, errlen);
        if (n_q4)
            CUDA_OK(cudaMemcpy(ctx->w_q4, h_q4, (size_t)n_q4, cudaMemcpyHostToDevice), err, errlen);
        if (n_f16)
            CUDA_OK(cudaMemcpy(ctx->w_f16, h_f16, (size_t)n_f16 * 2, cudaMemcpyHostToDevice), err, errlen);
        if (n_f32)
            CUDA_OK(cudaMemcpy(ctx->w_f32, h_f32, (size_t)n_f32 * 4, cudaMemcpyHostToDevice), err, errlen);
        free(h_q4);
        free(h_f16);
        free(h_f32);
        ctx->n_q4 = n_q4;
        ctx->n_f16 = n_f16;
        ctx->n_f32 = n_f32;
    }

    ctx->hidden = h->hidden;
    ctx->kv_dim = h->n_kv_heads * h->head_dim;
    ctx->max_seq = e->max_seq;
    ctx->inter = e->inter;
    ctx->vocab = h->vocab;
    ctx->n_heads = h->n_heads;
    ctx->n_kv_heads = h->n_kv_heads;
    ctx->head_dim = h->head_dim;
    ctx->n_blocks = h->n_blocks;
    ctx->arch = h->arch;
    memcpy(&ctx->eps, &h->norm_eps_bits, 4);
    memcpy(&ctx->theta, &h->rope_theta_bits, 4);

    ctx->kv_bytes = (size_t)(2 * h->n_blocks + 1) * e->max_seq * ctx->kv_dim * 2;
    CUDA_OK(cudaMalloc((void**)&ctx->kv_blob, ctx->kv_bytes), err, errlen);
    CUDA_OK(cudaMemset(ctx->kv_blob, 0, ctx->kv_bytes), err, errlen);

    CUDA_OK(cudaMalloc((void**)&ctx->d_x, (size_t)ctx->hidden * 4), err, errlen);
    CUDA_OK(cudaMalloc((void**)&ctx->d_hb, (size_t)ctx->hidden * 4 * 9), err, errlen);
    CUDA_OK(cudaMalloc((void**)&ctx->d_hb2, (size_t)ctx->hidden * 4 * 9), err, errlen);
    CUDA_OK(cudaMalloc((void**)&ctx->d_ffn, (size_t)2 * ctx->inter * 4), err, errlen);
    CUDA_OK(cudaMalloc((void**)&ctx->d_logits, (size_t)h->vocab * 4), err, errlen);
    {
        uint32_t B = e->pb_cap ? e->pb_cap : 64;
        uint32_t xf = ctx->hidden > ctx->inter ? ctx->hidden : ctx->inter;
        ctx->pb_cap = B;
        CUDA_OK(cudaMalloc((void**)&ctx->d_xf16, (size_t)B * xf * 2), err, errlen);
        CUDA_OK(cudaMalloc((void**)&ctx->d_pb, (size_t)B * ctx->hidden * 4), err, errlen);
        CUDA_OK(cudaMalloc((void**)&ctx->d_pb2, (size_t)B * ctx->hidden * 4), err, errlen);
        CUDA_OK(cudaMalloc((void**)&ctx->d_pbq, (size_t)B * ctx->hidden * 4), err, errlen);
        CUDA_OK(cudaMalloc((void**)&ctx->d_pbk, (size_t)B * ctx->kv_dim * 4), err, errlen);
        CUDA_OK(cudaMalloc((void**)&ctx->d_pbv, (size_t)B * ctx->kv_dim * 4), err, errlen);
        CUDA_OK(cudaMalloc((void**)&ctx->d_pbg, (size_t)B * ctx->inter * 4), err, errlen);
        CUDA_OK(cudaMalloc((void**)&ctx->d_pbu, (size_t)B * ctx->inter * 4), err, errlen);
        CUDA_OK(cudaMalloc((void**)&ctx->d_tokens, (size_t)B * sizeof(uint32_t)), err, errlen);
    }

    if (cuda_k_cublas_create(&ctx->cublas, err, errlen) != 0) return -1;

    e->w_dev = ctx;
    e->d_kv = ctx->kv_blob;
    e->weights_ready = 1;
    e->device_mode = DEV_MODE_CUDA;
    cuda_attach_fwd(e);
    cuda_k_sync();
    ylog_info("cuda: load_weights GPU mode=%s%s q4=%.2f MB f16w=%.2f MB f32v=%.2f MB kv=%.2f MB pb=%u layers=[%u,%u) gpu=%d",
              e->cuda_wmode == CUDA_W_FP16 ? "fp16" :
              (e->cuda_wmode == CUDA_W_Q4K ? "q4k" : "auto"),
              ctx->stream_w ? "+stream" : "",
              (double)ctx->n_q4 / 1048576.0, (double)ctx->n_f16 * 2 / 1048576.0, (double)ctx->n_f32 * 4 / 1048576.0,
              (double)ctx->kv_bytes / 1048576.0, ctx->pb_cap, begin, end, ctx->device_id);
    return 0;
}
#endif

static int load_weights_shim(Engine* e, CudaCtx* ctx, char* err, size_t errlen)
{
    LlModel* m = &e->ws.model;
    uint32_t begin = e->layer_begin;
    uint32_t end = e->layer_end;
    if (e->gpu_layer_end && e->gpu_layer_end < end) end = e->gpu_layer_end;
    if (end > m->n_layers) end = m->n_layers;
    if (begin > end) begin = end;

    uint64_t total = 0;
    uint32_t i;
    for (i = begin; i < end; i++) total += m->dir[i].size;

    ctx->n_layers = m->n_layers;
    ctx->layer_off = (uint64_t*)malloc((size_t)m->n_layers * sizeof(uint64_t));
    if (!ctx->layer_off) {
        if (err && errlen) snprintf(err, errlen, "oom layer_off");
        return -1;
    }
    for (i = 0; i < m->n_layers; i++) ctx->layer_off[i] = CUDA_OFF_NONE;

    ctx->w_blob = (uint8_t*)calloc(1, (size_t)total);
    if (total && !ctx->w_blob) {
        if (err && errlen) snprintf(err, errlen, "oom w_blob");
        return -1;
    }
    ctx->w_bytes = total;
    uint64_t cursor = 0;
    const uint8_t* map = (const uint8_t*)e->ws.map.base;
    for (i = begin; i < end; i++) {
        uint64_t sz = m->dir[i].size;
        ctx->layer_off[i] = cursor;
        if (sz) memcpy(ctx->w_blob + cursor, map + m->dir[i].offset, (size_t)sz);
        cursor += sz;
    }
    e->w_dev = ctx;
    e->d_kv = e->kv;
    e->weights_ready = 1;
    e->device_mode = DEV_MODE_CUDA_HOST;
    cuda_attach_fwd(e);
    ylog_info("cuda: load_weights shim layers=[%u,%u) blob=%.2f MB",
              begin, end, (double)total / 1048576.0);
    return 0;
}

static int cuda_load_weights(Engine* e, char* err, size_t errlen)
{
    Device* d = e->dev;
    CudaCtx* ctx = d ? (CudaCtx*)d->handle : NULL;
    if (!ctx) {
        if (err && errlen) snprintf(err, errlen, "null cuda ctx");
        return -1;
    }
    if (e->ops && !e->ops->gpu_fused) {
        ylog_info("cuda: skip GPU transformer for %s", e->ops->name);
        e->w_dev = ctx;
        e->d_kv = e->kv;
        e->weights_ready = 1;
        e->device_mode = DEV_MODE_CPU;
        if (e->dev) {
            e->dev->fwd_block = NULL;
            e->dev->fwd_block_batch = NULL;
        }
        return 0;
    }
    int device_id = ctx->device_id;
    int host_shim = ctx->host_shim;
    cuda_ctx_clear(ctx);
    ctx->device_id = device_id;
    ctx->host_shim = host_shim;

#if defined(YLLM_CUDA) && !defined(YLLM_CUDA_HOST)
    if (!ctx->host_shim)
        return load_weights_gpu(e, ctx, err, errlen);
#endif
    return load_weights_shim(e, ctx, err, errlen);
}

static void cuda_free_dev(Engine* e)
{
    Device* d = e->dev;
    CudaCtx* ctx = d ? (CudaCtx*)d->handle : NULL;
    if (ctx) {
        int device_id = ctx->device_id;
        int host_shim = ctx->host_shim;
        cuda_ctx_clear(ctx);
        ctx->device_id = device_id;
        ctx->host_shim = host_shim;
    }
    e->w_dev = NULL;
    e->d_kv = NULL;
    e->weights_ready = 0;
    e->device_mode = DEV_MODE_CPU;
}

const uint8_t* cuda_layer_base(const Engine* e, uint32_t layer)
{
    const CudaCtx* ctx = e && e->w_dev ? (const CudaCtx*)e->w_dev : NULL;
    if (!ctx || !ctx->w_blob || layer >= ctx->n_layers) return NULL;
    if (ctx->layer_off[layer] == CUDA_OFF_NONE) return NULL;
    return ctx->w_blob + ctx->layer_off[layer];
}

void cuda_after_prefill(Engine* e, uint32_t n_pos)
{
#if defined(YLLM_CUDA) && !defined(YLLM_CUDA_HOST)
    CudaCtx* ctx = e && e->w_dev ? (CudaCtx*)e->w_dev : NULL;
    if (!e || e->device_mode != DEV_MODE_CUDA || !ctx || !ctx->kv_blob || !e->kv || n_pos == 0) return;
    (void)n_pos;
    /* 仅当 prefill 走了 CPU batch 时需要; GPU 逐 token prefill 已写 d_kv */
    cuda_k_memcpy_h2d(ctx->kv_blob, e->kv, ctx->kv_bytes);
    cuda_k_memcpy_h2d(ctx->d_x, e->x, (size_t)ctx->hidden * 4);
    ctx->x_on_dev = 1;
    cuda_k_sync();
#else
    (void)e;
    (void)n_pos;
#endif
}

const uint8_t* cuda_tensor_q4k(const Engine* e, uint32_t layer, uint32_t slot,
                               uint32_t* out, uint32_t* in)
{
    const CudaCtx* ctx = e && e->w_dev ? (const CudaCtx*)e->w_dev : NULL;
    if (!ctx || !ctx->w_q4 || !ctx->off_q4) return NULL;
    size_t idx = (size_t)layer * BLOCK_TENSORS_MTP + slot;
    if (layer >= ctx->n_layers || ctx->off_q4[idx] == CUDA_OFF_NONE) return NULL;
    if (out) *out = ctx->dim_out[idx];
    if (in) *in = ctx->dim_in[idx];
    return ctx->w_q4 + ctx->off_q4[idx];
}

const uint16_t* cuda_tensor_f16w(const Engine* e, uint32_t layer, uint32_t slot,
                                 uint32_t* out, uint32_t* in)
{
    const CudaCtx* ctx = e && e->w_dev ? (const CudaCtx*)e->w_dev : NULL;
    if (!ctx || !ctx->w_f16 || !ctx->off_f16) return NULL;
    size_t idx = (size_t)layer * BLOCK_TENSORS_MTP + slot;
    if (layer >= ctx->n_layers || ctx->off_f16[idx] == CUDA_OFF_NONE) return NULL;
    if (out) *out = ctx->dim_out[idx];
    if (in) *in = ctx->dim_in[idx];
    return ctx->w_f16 + ctx->off_f16[idx];
}

const float* cuda_tensor_f32(const Engine* e, uint32_t layer, uint32_t slot, uint32_t* n)
{
    const CudaCtx* ctx = e && e->w_dev ? (const CudaCtx*)e->w_dev : NULL;
    if (!ctx || !ctx->w_f32 || !ctx->off_f32) return NULL;
    size_t idx = (size_t)layer * BLOCK_TENSORS_MTP + slot;
    if (layer >= ctx->n_layers || ctx->off_f32[idx] == CUDA_OFF_NONE) return NULL;
    if (n) *n = ctx->dim_out[idx];
    return ctx->w_f32 + ctx->off_f32[idx];
}

CudaCtx* cuda_get_ctx(Engine* e)
{
    return e && e->w_dev ? (CudaCtx*)e->w_dev : NULL;
}

#if defined(YLLM_CUDA) && !defined(YLLM_CUDA_HOST)
static void stream_invalidate_layer(CudaCtx* ctx, uint32_t layer)
{
    if (!ctx || layer >= ctx->n_layers || !ctx->off_q4) return;
    uint32_t s;
    for (s = 0; s < BLOCK_TENSORS_MTP; s++) {
        size_t idx = (size_t)layer * BLOCK_TENSORS_MTP + s;
        ctx->off_q4[idx] = CUDA_OFF_NONE;
        ctx->off_f16[idx] = CUDA_OFF_NONE;
        ctx->off_f32[idx] = CUDA_OFF_NONE;
    }
}

static int cuda_prefetch_layer(Engine* e, uint32_t layer)
{
    CudaCtx* ctx = cuda_get_ctx(e);
    if (!ctx || !ctx->stream_w) return 0;
    if (layer >= ctx->n_layers) return -1;
    if (ctx->stream_layer == layer) return 0;

    if (ctx->stream_layer != ~0u)
        stream_invalidate_layer(ctx, ctx->stream_layer);

    uint64_t dq = 0, df = 0, d32 = 0;
    uint32_t s;
    for (s = 0; s < BLOCK_TENSORS_MTP; s++) {
        size_t idx = (size_t)layer * BLOCK_TENSORS_MTP + s;
        if (ctx->host_off_q4 && ctx->host_off_q4[idx] != CUDA_OFF_NONE) {
            uint32_t out = ctx->dim_out[idx], in = ctx->dim_in[idx];
            size_t nbytes = (size_t)out * (in / 256) * 144;
            if (dq + nbytes > ctx->max_layer_q4) return -1;
            if (cudaMemcpy(ctx->w_q4 + dq, ctx->h_q4 + ctx->host_off_q4[idx],
                           nbytes, cudaMemcpyHostToDevice) != cudaSuccess)
                return -1;
            ctx->off_q4[idx] = dq;
            dq += nbytes;
        } else if (ctx->host_off_f16 && ctx->host_off_f16[idx] != CUDA_OFF_NONE) {
            uint64_t ne = (uint64_t)ctx->dim_out[idx] * ctx->dim_in[idx];
            if (df + ne > ctx->max_layer_f16) return -1;
            if (cudaMemcpy(ctx->w_f16 + df, ctx->h_f16 + ctx->host_off_f16[idx],
                           (size_t)ne * 2, cudaMemcpyHostToDevice) != cudaSuccess)
                return -1;
            ctx->off_f16[idx] = df;
            df += ne;
        } else if (ctx->host_off_f32 && ctx->host_off_f32[idx] != CUDA_OFF_NONE) {
            uint32_t n = ctx->dim_out[idx];
            if (d32 + n > ctx->max_layer_f32) return -1;
            if (cudaMemcpy(ctx->w_f32 + d32, ctx->h_f32 + ctx->host_off_f32[idx],
                           (size_t)n * 4, cudaMemcpyHostToDevice) != cudaSuccess)
                return -1;
            ctx->off_f32[idx] = d32;
            d32 += n;
        }
    }
    ctx->stream_layer = layer;
    return 0;
}

static void cuda_release_layer(Engine* e, uint32_t layer)
{
    CudaCtx* ctx = cuda_get_ctx(e);
    if (!ctx || !ctx->stream_w) return;
    if (ctx->stream_layer == layer) {
        stream_invalidate_layer(ctx, layer);
        ctx->stream_layer = ~0u;
    }
}
#else
static int cuda_prefetch_layer(Engine* e, uint32_t layer)
{
    (void)e;
    (void)layer;
    return 0;
}
static void cuda_release_layer(Engine* e, uint32_t layer)
{
    (void)e;
    (void)layer;
}
#endif

Device* device_create_cuda(int device_id, char* err, size_t errlen)
{
#ifndef YLLM_CUDA
    (void)device_id;
    if (err && errlen)
        snprintf(err, errlen, "CUDA backend not built (rebuild with YLLM_CUDA=1)");
    return NULL;
#else
    int host_shim = 0;
#ifdef YLLM_CUDA_HOST
    host_shim = 1;
#else
    host_shim = 0;
    {
        cudaError_t ce = cudaSetDevice(device_id);
        if (ce != cudaSuccess) {
            if (err && errlen)
                snprintf(err, errlen, "cudaSetDevice(%d): %s", device_id, cudaGetErrorString(ce));
            return NULL;
        }
    }
#endif
    Device* d = (Device*)calloc(1, sizeof(Device));
    CudaCtx* ctx = (CudaCtx*)calloc(1, sizeof(CudaCtx));
    if (!d || !ctx) {
        free(d);
        free(ctx);
        if (err && errlen) snprintf(err, errlen, "oom");
        return NULL;
    }
    ctx->device_id = device_id;
    ctx->host_shim = host_shim;
    d->kind = DEV_CUDA;
    d->id = device_id;
    d->handle = ctx;
    d->load_weights = cuda_load_weights;
    d->free_dev = cuda_free_dev;
    d->prefetch_layer = cuda_prefetch_layer;
    d->release_layer = cuda_release_layer;
    d->embed = cuda_embed;
    d->after_cpu_embed = cuda_mark_x_host;
    d->final_norm = cuda_final_norm;
    d->lm_head = cuda_lm_head;
    d->prefill = cuda_prefill;
    d->sync_x = cuda_sync_x_to_host;
    d->mark_x_host = cuda_mark_x_host;
    d->forward_batch_x = cuda_forward_batch_x;
    if (host_shim)
        ylog_info("cuda: host-shim device (weights mirrored in RAM; CPU compute)");
    else
        ylog_info("cuda: device=%d (Q4_K native + FP16 fallback)", device_id);
    return d;
#endif
}
