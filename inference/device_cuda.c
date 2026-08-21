/* device_cuda.c — CUDA 设备后端
 *
 * YLLM_CUDA_HOST=1: 权拷到主机堆(无 GPU 也可测)
 * 真 CUDA: cudaMallocManaged(统一内存) — CPU 算子过渡期可读; 后续 kernel 直接用同指针
 */
#include "device.h"
#include "yllm.h"
#include "log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(YLLM_CUDA) && !defined(YLLM_CUDA_HOST)
#include <cuda_runtime.h>
#define CUDA_OK(call, err, errlen) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        if (err && errlen) snprintf(err, errlen, "CUDA: %s", cudaGetErrorString(_e)); \
        return -1; \
    } \
} while (0)
#endif

typedef struct {
    int device_id;
    int host_shim;          /* 1 = malloc 模拟设备 */
    uint8_t* w_blob;        /* 本 rank 层权连续存放 */
    uint64_t* layer_off;    /* 全局层号 → blob 内偏移; 未覆盖层为 UINT64_MAX */
    uint32_t n_layers;
    uint64_t w_bytes;
    uint16_t* kv_blob;      /* 设备 KV(与 Engine.kv 同布局); shim 不用 */
    size_t kv_bytes;
} CudaCtx;

/* cuda_fwd.c */
void cuda_attach_fwd(Engine* e);

static void cuda_ctx_clear(CudaCtx* ctx)
{
#if defined(YLLM_CUDA) && !defined(YLLM_CUDA_HOST)
    if (!ctx->host_shim) {
        if (ctx->w_blob) cudaFree(ctx->w_blob);
        if (ctx->kv_blob) cudaFree(ctx->kv_blob);
        ctx->w_blob = NULL;
        ctx->kv_blob = NULL;
    } else
#endif
    {
        free(ctx->w_blob);
        free(ctx->kv_blob);
        ctx->w_blob = NULL;
        ctx->kv_blob = NULL;
    }
    free(ctx->layer_off);
    ctx->layer_off = NULL;
    ctx->w_bytes = 0;
    ctx->kv_bytes = 0;
    ctx->n_layers = 0;
}

static int cuda_alloc(void** p, size_t n, int host_shim, char* err, size_t errlen)
{
    *p = NULL;
    if (n == 0) return 0;
#if defined(YLLM_CUDA) && !defined(YLLM_CUDA_HOST)
    if (!host_shim) {
        /* Managed: CPU 过渡算子与后续 GPU kernel 共用同一指针 */
        CUDA_OK(cudaMallocManaged(p, n, cudaMemAttachGlobal), err, errlen);
        return 0;
    }
#else
    (void)host_shim;
#endif
    *p = calloc(1, n);
    if (!*p) {
        if (err && errlen) snprintf(err, errlen, "oom (%zu bytes)", n);
        return -1;
    }
    return 0;
}

static int cuda_copy_h2d(void* dst, const void* src, size_t n, int host_shim, char* err, size_t errlen)
{
    if (n == 0) return 0;
#if defined(YLLM_CUDA) && !defined(YLLM_CUDA_HOST)
    if (!host_shim) {
        /* managed 可用 memcpy; 用 cudaMemcpy 便于以后改成纯 device 内存 */
        CUDA_OK(cudaMemcpy(dst, src, n, cudaMemcpyDefault), err, errlen);
        return 0;
    }
#else
    (void)err;
    (void)errlen;
    (void)host_shim;
#endif
    memcpy(dst, src, n);
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
    if (e->arch == ARCH_QWEN35) {
        if (err && errlen)
            snprintf(err, errlen, "CUDA path does not support ARCH_QWEN35 yet; use --device cpu");
        return -1;
    }

    cuda_ctx_clear(ctx);

    LlModel* m = &e->ws.model;
    uint32_t begin = e->layer_begin;
    uint32_t end = e->layer_end;
    if (end > m->n_layers) end = m->n_layers;
    if (begin > end) begin = end;

    uint64_t total = 0;
    uint32_t i;
    for (i = begin; i < end; i++)
        total += m->dir[i].size;

    ctx->n_layers = m->n_layers;
    ctx->layer_off = (uint64_t*)malloc((size_t)m->n_layers * sizeof(uint64_t));
    if (!ctx->layer_off) {
        if (err && errlen) snprintf(err, errlen, "oom layer_off");
        return -1;
    }
    for (i = 0; i < m->n_layers; i++)
        ctx->layer_off[i] = (uint64_t)~0ULL;

    if (cuda_alloc((void**)&ctx->w_blob, (size_t)total, ctx->host_shim, err, errlen) != 0)
        return -1;
    ctx->w_bytes = total;

    /* 逐层拷贝到连续 blob(层内 offset 仍相对层基址, 与 mmap 布局一致) */
    uint64_t cursor = 0;
    const uint8_t* map = (const uint8_t*)e->ws.map.base;
    for (i = begin; i < end; i++) {
        uint64_t sz = m->dir[i].size;
        ctx->layer_off[i] = cursor;
        if (sz > 0) {
            if (cuda_copy_h2d(ctx->w_blob + cursor, map + m->dir[i].offset, (size_t)sz,
                              ctx->host_shim, err, errlen) != 0) {
                cuda_ctx_clear(ctx);
                return -1;
            }
        }
        cursor += sz;
    }

    ctx->kv_bytes = 0;
    ctx->kv_blob = NULL;

    e->w_dev = ctx;
    /* 过渡期: KV 仍用 host e->kv, 与 prefill batch 一致。
     * 真 GPU kernel 落地后再给 KV 单独 managed/device 缓冲。 */
    e->d_kv = e->kv;
    if (ctx->kv_blob) {
#if defined(YLLM_CUDA) && !defined(YLLM_CUDA_HOST)
        if (!ctx->host_shim) cudaFree(ctx->kv_blob);
        else
#endif
            free(ctx->kv_blob);
        ctx->kv_blob = NULL;
        ctx->kv_bytes = 0;
    }
    e->weights_ready = 1;
    cuda_attach_fwd(e);
#if defined(YLLM_CUDA) && !defined(YLLM_CUDA_HOST)
    if (!ctx->host_shim) {
        cudaDeviceSynchronize();
        ylog_info("cuda: load_weights layers=[%u,%u) blob=%.2f MB (managed) gpu=%d",
                  begin, end, (double)total / 1048576.0, ctx->device_id);
    } else
#endif
    ylog_info("cuda: load_weights layers=[%u,%u) blob=%.2f MB kv=shared-host shim=%d gpu=%d",
              begin, end, (double)total / 1048576.0,
              ctx->host_shim, ctx->device_id);
    return 0;
}

static void cuda_free_dev(Engine* e)
{
    Device* d = e->dev;
    CudaCtx* ctx = d ? (CudaCtx*)d->handle : NULL;
    if (ctx) cuda_ctx_clear(ctx);
    e->w_dev = NULL;
    e->d_kv = NULL;
    e->weights_ready = 0;
}

const uint8_t* cuda_layer_base(const Engine* e, uint32_t layer)
{
    const CudaCtx* ctx = e && e->w_dev ? (const CudaCtx*)e->w_dev : NULL;
    if (!ctx || !ctx->w_blob || layer >= ctx->n_layers) return NULL;
    if (ctx->layer_off[layer] == (uint64_t)~0ULL) return NULL;
    return ctx->w_blob + ctx->layer_off[layer];
}

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
    d->prefetch_layer = NULL;
    d->release_layer = NULL;
    if (host_shim)
        ylog_info("cuda: host-shim device (weights mirrored in RAM; CPU compute)");
    else
        ylog_info("cuda: device=%d (managed memory weights; CPU compute until kernels)", device_id);
    return d;
#endif
}
