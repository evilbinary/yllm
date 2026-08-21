/* device_vulkan.c — Vulkan 设备后端(Android / iOS·MoltenVK / PC) */
#include "device.h"
#include "yllm.h"
#include "llf.h"
#include "log.h"
#include "vulkan_ctx.h"
#include "vulkan_load.h"
#include "vulkan_compute.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void vulkan_attach_fwd(Engine* e);
int vulkan_selftest_rmsnorm(VulkanCtx* ctx);
int vulkan_selftest_gemv_q4k(VulkanCtx* ctx);

static int tensor_out_in(const LlfTensorMeta* mt, uint32_t* out, uint32_t* in)
{
    if (mt->ndim < 2 || mt->shape[0] == 0 || mt->shape[1] == 0) return -1;
    uint32_t a = mt->shape[0], b = mt->shape[1];
    *out = a;
    *in = b;
    if (mt->dtype == DT_Q4K) {
        size_t e0 = (size_t)a * (b / 256) * 144;
        size_t e1 = (size_t)b * (a / 256) * 144;
        if (e0 != mt->size && e1 == mt->size) { *out = b; *in = a; }
        else if (e0 != mt->size && e1 != mt->size) return -1;
    }
    return 0;
}

static void scan_block_q4k(const LlModel* m, uint32_t* max_in, uint32_t* max_out,
                           size_t* total_wq)
{
    uint32_t hidden = m->h.hidden;
    *max_in = hidden;
    *max_out = hidden;
    *total_wq = 0;
    uint32_t li;
    for (li = 0; li < m->h.n_blocks; li++) {
        uint32_t layer = li + 1;
        if (layer >= m->n_layers) continue;
        uint32_t hidx = m->base_idx[layer];
        uint32_t nt = m->dir[layer].n_tensors;
        if (nt > BLOCK_TENSORS) nt = BLOCK_TENSORS;
        uint32_t s;
        for (s = 0; s < nt; s++) {
            const LlfTensorMeta* mt = &m->metas[hidx + s];
            if (mt->dtype != DT_Q4K || mt->size == 0) continue;
            uint32_t o, i;
            if (tensor_out_in(mt, &o, &i) != 0) continue;
            if ((i % 256) != 0) continue;
            if (o > *max_out) *max_out = o;
            if (i > *max_in) *max_in = i;
            *total_wq += (size_t)o * ((size_t)(i / 256) * 144);
        }
    }
}

/* 打包块内 Q4_K → 连续 blob, 填 wq_off[layer*nslot+slot] */
static int pack_upload_q4k(Engine* e, VulkanCtx* ctx)
{
    LlModel* m = &e->ws.model;
    uint32_t nslot = ctx->wq_nslot;
    size_t total = 0;
    uint32_t li, s;

    for (li = 0; li < m->h.n_blocks; li++) {
        uint32_t layer = li + 1;
        if (layer >= m->n_layers) continue;
        uint32_t hidx = m->base_idx[layer];
        uint32_t nt = m->dir[layer].n_tensors;
        if (nt > nslot) nt = nslot;
        for (s = 0; s < nt; s++) {
            const LlfTensorMeta* mt = &m->metas[hidx + s];
            if (mt->dtype != DT_Q4K || mt->size == 0) continue;
            uint32_t o, i;
            if (tensor_out_in(mt, &o, &i) != 0 || (i % 256) != 0) continue;
            total += (size_t)o * ((size_t)(i / 256) * 144);
        }
    }
    if (total == 0 || total > ctx->wq_bytes) {
        ylog_warn("vulkan: Q4_K pack size %zu vs buf %zu", total, ctx->wq_bytes);
        return -1;
    }

    uint8_t* blob = (uint8_t*)malloc(total);
    if (!blob) return -1;
    size_t cursor = 0;
    for (li = 0; li < m->h.n_blocks; li++) {
        uint32_t layer = li + 1;
        if (layer >= m->n_layers) continue;
        const uint8_t* base =
            (const uint8_t*)e->ws.map.base + m->dir[layer].offset;
        uint32_t hidx = m->base_idx[layer];
        uint32_t nt = m->dir[layer].n_tensors;
        if (nt > nslot) nt = nslot;
        for (s = 0; s < nt; s++) {
            const LlfTensorMeta* mt = &m->metas[hidx + s];
            if (mt->dtype != DT_Q4K || mt->size == 0) continue;
            uint32_t o, i;
            if (tensor_out_in(mt, &o, &i) != 0 || (i % 256) != 0) continue;
            size_t nbytes = (size_t)o * ((size_t)(i / 256) * 144);
            if (nbytes != mt->size) {
                /* size 字段为准 */
                nbytes = (size_t)mt->size;
            }
            if (cursor + nbytes > total) {
                free(blob);
                return -1;
            }
            memcpy(blob + cursor, base + mt->offset, nbytes);
            ctx->wq_off[(size_t)layer * nslot + s] = cursor;
            cursor += nbytes;
        }
    }
    int rc = vulkan_wq_upload(ctx, blob, cursor);
    free(blob);
    return rc;
}

static int vk_load_weights(Engine* e, char* err, size_t errlen)
{
    VulkanCtx* ctx = e->dev && e->dev->handle ? (VulkanCtx*)e->dev->handle : NULL;
    if (!ctx) {
        if (err && errlen) snprintf(err, errlen, "null vulkan ctx");
        return -1;
    }
    e->w_dev = ctx;
    e->d_kv = e->kv;
    e->weights_ready = 1;
    e->device_mode = ctx->host_shim ? DEV_MODE_VULKAN_HOST : DEV_MODE_VULKAN;
    ctx->n_layers = e->ws.model.n_layers;
    ctx->hidden = e->ws.model.h.hidden;

    if (!ctx->host_shim) {
        uint32_t max_in = ctx->hidden, max_out = ctx->hidden;
        size_t total_wq = 0;
        scan_block_q4k(&e->ws.model, &max_in, &max_out, &total_wq);
        char cerr[256];
        if (vulkan_compute_setup(ctx, ctx->hidden, max_in, max_out, total_wq,
                                 e->ws.model.n_layers, BLOCK_TENSORS,
                                 cerr, sizeof(cerr)) != 0) {
            ylog_warn("vulkan: compute setup failed (%s); fwd stays CPU", cerr);
        } else {
            if (vulkan_selftest_rmsnorm(ctx) != 0) {
                ylog_warn("vulkan: rmsnorm selftest failed; GPU rmsnorm disabled");
                ctx->compute_ready = 0;
            }
            if (ctx->gemv_ready && vulkan_selftest_gemv_q4k(ctx) != 0) {
                ylog_warn("vulkan: gemv_q4k selftest failed; GPU gemv disabled");
                ctx->gemv_ready = 0;
            }
            if (ctx->gemv_ready && pack_upload_q4k(e, ctx) != 0) {
                ylog_warn("vulkan: Q4_K resident upload failed; gemv falls back host-W");
                ctx->wq_resident = 0;
            }
            if (ctx->fuse_ready) {
                char aerr[256];
                const LlfHeader* h = &e->ws.model.h;
                uint32_t kv_dim = h->n_kv_heads * h->head_dim;
                if (vulkan_attn_setup(ctx, h->n_blocks, e->max_seq, kv_dim,
                                      h->n_heads, h->n_kv_heads, h->head_dim,
                                      aerr, sizeof(aerr)) != 0) {
                    ylog_warn("vulkan: attn setup failed (%s)", aerr);
                }
            }
        }
    }

    vulkan_attach_fwd(e);
    ylog_info("vulkan: mode=%s gpu=%d layers=%u hidden=%u rms=%d gemv=%d resident=%d fuse=%d swi=%d attn=%d attn_o=%d",
              ctx->host_shim ? "host-shim" : "native",
              ctx->device_id, ctx->n_layers, ctx->hidden,
              ctx->compute_ready, ctx->gemv_ready, ctx->wq_resident,
              ctx->fuse_ready, ctx->swi_ready, ctx->attn_ready, ctx->attn_o_ready);
    return 0;
}

static void vk_free_dev(Engine* e)
{
    if (e->dev && e->dev->handle) {
        VulkanCtx* ctx = (VulkanCtx*)e->dev->handle;
        vulkan_shutdown(ctx);
        free(ctx);
        e->dev->handle = NULL;
    }
    e->w_dev = NULL;
    e->d_kv = NULL;
    e->weights_ready = 0;
    e->device_mode = DEV_MODE_CPU;
}

Device* device_create_vulkan(int device_id, char* err, size_t errlen)
{
#ifndef YLLM_VULKAN
    (void)device_id;
    if (err && errlen)
        snprintf(err, errlen, "Vulkan backend not built (YLLM_VULKAN=1)");
    return NULL;
#else
    Device* d = (Device*)calloc(1, sizeof(Device));
    VulkanCtx* ctx = (VulkanCtx*)calloc(1, sizeof(VulkanCtx));
    if (!d || !ctx) {
        free(d);
        free(ctx);
        if (err && errlen) snprintf(err, errlen, "oom");
        return NULL;
    }
    ctx->device_id = device_id;
    ctx->host_shim = 1;
#ifdef YLLM_VULKAN_HOST
    ylog_info("vulkan: forced host-shim (YLLM_VULKAN_HOST=1)");
#else
    {
        char verr[256];
        if (vulkan_try_init(ctx, device_id, verr, sizeof(verr)) != 0) {
            ylog_warn("vulkan: native init failed (%s), falling back to host-shim", verr);
            ctx->host_shim = 1;
            ctx->instance = NULL;
            ctx->phys = NULL;
            ctx->device = NULL;
            ctx->queue = NULL;
        } else {
            ylog_info("vulkan: native compute device ready");
        }
    }
#endif
    d->kind = DEV_VULKAN;
    d->id = ctx->device_id;
    d->handle = ctx;
    d->load_weights = vk_load_weights;
    d->free_dev = vk_free_dev;
    d->prefetch_layer = NULL;
    d->release_layer = NULL;
    return d;
#endif
}
