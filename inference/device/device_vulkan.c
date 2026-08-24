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
int vulkan_selftest_gemv_q6k(VulkanCtx* ctx);

static int tensor_out_in(const LlfTensorMeta* mt, uint32_t* out, uint32_t* in)
{
    if (mt->ndim < 2 || mt->shape[0] == 0 || mt->shape[1] == 0) return -1;
    uint32_t a = mt->shape[0], b = mt->shape[1];
    *out = a;
    *in = b;
    if (mt->dtype == DT_Q4K || mt->dtype == DT_Q6K) {
        size_t rowb = (mt->dtype == DT_Q4K) ? 144u : 210u;
        size_t e0 = (size_t)a * ((size_t)(b / 256) * rowb);
        size_t e1 = (size_t)b * ((size_t)(a / 256) * rowb);
        if (e0 == mt->size && e1 == mt->size) {
            /* 方阵歧义少见; 两侧都合法时取较大维为 out(vocab) */
            if (a >= b) { *out = a; *in = b; }
            else { *out = b; *in = a; }
        } else if (e0 != mt->size && e1 == mt->size) {
            *out = b; *in = a;
        } else if (e0 != mt->size && e1 != mt->size) {
            return -1;
        }
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
    /* lm_head 权体积计入 wq; 不抬 max_out(vocab 过大), gemv 分块写 logits */
    {
        uint32_t layer = m->h.n_blocks + 2;
        if (layer < m->n_layers) {
            const LlfTensorMeta* mt = &m->metas[m->base_idx[layer]];
            if (mt->size > 0) {
                uint32_t o, i;
                if (tensor_out_in(mt, &o, &i) == 0 && (i % 256) == 0) {
                    if (mt->dtype == DT_Q4K)
                        *total_wq += (size_t)o * ((size_t)(i / 256) * 144);
                    else if (mt->dtype == DT_Q6K)
                        *total_wq += (size_t)o * ((size_t)(i / 256) * 210);
                }
            }
        }
    }
}

static size_t max_block_q4k_bytes(const LlModel* m)
{
    size_t mx = 0;
    uint32_t li, s;
    for (li = 0; li < m->h.n_blocks; li++) {
        uint32_t layer = li + 1;
        if (layer >= m->n_layers) continue;
        size_t layer_sz = 0;
        uint32_t hidx = m->base_idx[layer];
        uint32_t nt = m->dir[layer].n_tensors;
        if (nt > BLOCK_TENSORS) nt = BLOCK_TENSORS;
        for (s = 0; s < nt; s++) {
            const LlfTensorMeta* mt = &m->metas[hidx + s];
            if (mt->dtype != DT_Q4K || mt->size == 0) continue;
            uint32_t o, i;
            if (tensor_out_in(mt, &o, &i) != 0 || (i % 256) != 0) continue;
            layer_sz += (size_t)o * ((size_t)(i / 256) * 144);
        }
        if (layer_sz > mx) mx = layer_sz;
    }
    return mx;
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
    {
        uint32_t layer = m->h.n_blocks + 2;
        if (layer < m->n_layers) {
            const LlfTensorMeta* mt = &m->metas[m->base_idx[layer]];
            if (mt->size > 0) {
                uint32_t o, i;
                if (tensor_out_in(mt, &o, &i) == 0 && (i % 256) == 0) {
                    if (mt->dtype == DT_Q4K)
                        total += (size_t)o * ((size_t)(i / 256) * 144);
                    else if (mt->dtype == DT_Q6K)
                        total += (size_t)o * ((size_t)(i / 256) * 210);
                }
            }
        }
    }
    if (total == 0) return -1;
    if (!ctx->wq_stream && total > ctx->wq_bytes) {
        ylog_warn("vulkan: Q4_K pack size %zu vs buf %zu", total, ctx->wq_bytes);
        return -1;
    }

    uint8_t* blob = (uint8_t*)malloc(total);
    if (!blob) return -1;
    size_t cursor = 0;
    ctx->lm_ready = 0;
    ctx->lm_off = (uint64_t)~0ull;
    ctx->lm_dtype = 0;
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
            if (mt->size != 0 && mt->size != nbytes) {
                ylog_warn("vulkan: Q4_K size mismatch slot=%u layer=%u calc=%zu meta=%llu; use calc",
                          s, layer, nbytes, (unsigned long long)mt->size);
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
    {
        uint32_t layer = m->h.n_blocks + 2;
        if (layer < m->n_layers) {
            const LlfTensorMeta* mt = &m->metas[m->base_idx[layer]];
            const uint8_t* base =
                (const uint8_t*)e->ws.map.base + m->dir[layer].offset;
            if (mt->dtype == DT_Q4K && mt->size > 0) {
                uint32_t o, i;
                if (tensor_out_in(mt, &o, &i) == 0 && (i % 256) == 0) {
                    size_t nbytes = (size_t)o * ((size_t)(i / 256) * 144);
                    if (nbytes != mt->size) nbytes = (size_t)mt->size;
                    if (cursor + nbytes <= total) {
                        memcpy(blob + cursor, base + mt->offset, nbytes);
                        ctx->lm_off = cursor;
                        ctx->lm_out = o;
                        ctx->lm_in = i;
                        ctx->lm_dtype = DT_Q4K;
                        ctx->wq_off[(size_t)layer * nslot + 0] = cursor;
                        cursor += nbytes;
                        ctx->lm_ready = 1;
                    } else {
                        ylog_warn("vulkan: lm_head pack overflow cursor=%zu need=%zu total=%zu",
                                  cursor, nbytes, total);
                    }
                } else {
                    ylog_warn("vulkan: lm_head skip shape dtype=%u size=%llu ndim=%u s0=%u s1=%u",
                              mt->dtype, (unsigned long long)mt->size, mt->ndim,
                              mt->shape[0], mt->shape[1]);
                }
            } else if (mt->dtype == DT_Q6K && mt->size > 0) {
                uint32_t o, i;
                if (tensor_out_in(mt, &o, &i) == 0 && (i % 256) == 0) {
                    size_t nbytes = (size_t)o * ((size_t)(i / 256) * 210);
                    if (nbytes != mt->size) nbytes = (size_t)mt->size;
                    if (cursor + nbytes <= total) {
                        memcpy(blob + cursor, base + mt->offset, nbytes);
                        ctx->lm_off = cursor;
                        ctx->lm_out = o;
                        ctx->lm_in = i;
                        ctx->lm_dtype = DT_Q6K;
                        ctx->wq_off[(size_t)layer * nslot + 0] = cursor;
                        cursor += nbytes;
                        ctx->lm_ready = 1;
                    } else {
                        ylog_warn("vulkan: lm_head Q6_K pack overflow cursor=%zu need=%zu total=%zu",
                                  cursor, nbytes, total);
                    }
                } else {
                    ylog_warn("vulkan: lm_head Q6_K skip shape dtype=%u size=%llu ndim=%u s0=%u s1=%u",
                              mt->dtype, (unsigned long long)mt->size, mt->ndim,
                              mt->shape[0], mt->shape[1]);
                }
            } else {
                ylog_info("vulkan: lm_head dtype=%u (CPU fallback)", mt->dtype);
            }
        }
    }
    if (ctx->wq_stream) {
        ctx->host_wq = blob;
        ctx->host_wq_bytes = cursor;
        ctx->wq_resident = 1;
        ctx->stream_layer = (uint32_t)~0u;
        if (ctx->lm_ready) {
            if (ctx->lm_dtype == DT_Q6K)
                ylog_info("vulkan: lm_head Q6_K stream-chunked out=%u in=%u", ctx->lm_out, ctx->lm_in);
            else
                ylog_info("vulkan: lm_head Q4_K stream-chunked out=%u in=%u", ctx->lm_out, ctx->lm_in);
        }
        ylog_info("vulkan: Q4_K stream host=%zuMB layer_gpu=%zuMB",
                  cursor / (1024 * 1024), ctx->wq_bytes / (1024 * 1024));
        return 0;
    }
    int rc = vulkan_wq_upload(ctx, blob, cursor);
    free(blob);
    if (rc == 0 && ctx->lm_ready) {
        const char* tag = (ctx->lm_dtype == DT_Q6K) ? "Q6_K" : "Q4_K";
        ylog_info("vulkan: lm_head %s resident out=%u in=%u", tag, ctx->lm_out, ctx->lm_in);
    }
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
    ctx->n_blocks = e->ws.model.h.n_blocks;
    memcpy(&ctx->norm_eps, &e->ws.model.h.norm_eps_bits, 4);

    if (!ctx->host_shim) {
        uint32_t max_in = ctx->hidden, max_out = ctx->hidden;
        size_t total_wq = 0;
        scan_block_q4k(&e->ws.model, &max_in, &max_out, &total_wq);
        ctx->layer_wq_max = 0;
        ctx->stream_base = 0;
        size_t layer_max = max_block_q4k_bytes(&e->ws.model);
        if (layer_max == 0) layer_max = total_wq;
        layer_max = (layer_max + 4095u) & ~(size_t)4095u;
        if (layer_max < 144 * 8) layer_max = 144 * 8;
        ctx->layer_wq_max = layer_max;

        /* iGPU / 超大 SSBO / YLLM_VK_STREAM=1 → 按层流式; dGPU 能装下则一次 resident */
        ctx->wq_stream = 0;
        size_t gpu_wq = total_wq;
        {
            const char* env = getenv("YLLM_VK_STREAM");
            int force_stream = (env && env[0] == '1') ? 1 : 0;
            int force_resident = (env && env[0] == '0') ? 1 : 0;
            int need_stream = 0;
            if (!force_resident) {
                if (ctx->integrated_gpu) need_stream = 1;
                else if (total_wq > ctx->max_ssbo_range) need_stream = 1;
                else if (force_stream) need_stream = 1;
            }
            if (need_stream) {
                ctx->wq_stream = 1;
                gpu_wq = layer_max;
                ylog_info("vulkan: weight stream on (total=%zuMB layer_gpu=%zuMB igpu=%d)",
                          total_wq / (1024 * 1024), layer_max / (1024 * 1024),
                          ctx->integrated_gpu);
            } else {
                ylog_info("vulkan: weight resident (total=%zuMB ssbo_max=%zuMB)",
                          total_wq / (1024 * 1024),
                          (size_t)(ctx->max_ssbo_range / (1024 * 1024)));
            }
        }
        char cerr[256];
        uint32_t lm_vocab = 0;
        {
            LlModel* m = &e->ws.model;
            uint32_t layer = m->h.n_blocks + 2;
            if (layer < m->n_layers) {
                const LlfTensorMeta* mt = &m->metas[m->base_idx[layer]];
                if ((mt->dtype == DT_Q4K || mt->dtype == DT_Q6K) && mt->size > 0) {
                    uint32_t o, i;
                    if (tensor_out_in(mt, &o, &i) == 0 && (i % 256) == 0)
                        lm_vocab = o;
                }
            }
        }
        if (vulkan_compute_setup(ctx, ctx->hidden, max_in, max_out, gpu_wq,
                                 e->ws.model.n_layers, BLOCK_TENSORS, lm_vocab,
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
            if (ctx->lm_ready && ctx->gemv_q6k_ready && !getenv("YLLM_VK_NOSELFTEST") &&
                vulkan_selftest_gemv_q6k(ctx) != 0) {
                ylog_warn("vulkan: gemv_q6k selftest failed; lm_head CPU fallback");
                ctx->lm_ready = 0;
            }
            if (ctx->lm_ready && ctx->gemv_ds_lm && !ctx->wq_stream &&
                ctx->lm_out > 0 && ctx->logits_bytes >= (size_t)ctx->lm_out * 4) {
                ctx->lm_one_submit = 1;
                ylog_info("vulkan: lm_head one-submit vocab=%u", ctx->lm_out);
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
    ylog_info("vulkan: mode=%s gpu=%d layers=%u hidden=%u rms=%d gemv=%d resident=%d stream=%d fuse=%d swi=%d attn=%d attn_o=%d rope=%d block=%d embed=%d gpu_rope=%d lm=%d lm1=%d",
              ctx->host_shim ? "host-shim" : "native",
              ctx->device_id, ctx->n_layers, ctx->hidden,
              ctx->compute_ready, ctx->gemv_ready, ctx->wq_resident, ctx->wq_stream,
              ctx->fuse_ready, ctx->swi_ready, ctx->attn_ready, ctx->attn_o_ready,
              ctx->rope_ready, ctx->block_ready, ctx->embed_ready, ctx->use_gpu_rope,
              ctx->lm_ready, ctx->lm_one_submit);
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
