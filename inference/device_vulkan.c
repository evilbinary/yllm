/* device_vulkan.c — Vulkan 设备后端(Android / iOS·MoltenVK / PC)
 *
 * 启动时动态加载 loader 并创建 compute 设备; 失败则 host-shim(CPU fwd)。
 * native: 加载 rmsnorm.spv, 块内 F32/F16 RMSNorm 走 GPU。
 */
#include "device.h"
#include "yllm.h"
#include "log.h"
#include "vulkan_ctx.h"
#include "vulkan_load.h"
#include "vulkan_compute.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void vulkan_attach_fwd(Engine* e);
int vulkan_selftest_rmsnorm(VulkanCtx* ctx);

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
        char cerr[256];
        if (vulkan_compute_setup(ctx, ctx->hidden, NULL, cerr, sizeof(cerr)) != 0) {
            ylog_warn("vulkan: compute setup failed (%s); fwd stays CPU", cerr);
        } else if (vulkan_selftest_rmsnorm(ctx) != 0) {
            ylog_warn("vulkan: rmsnorm selftest failed; GPU rmsnorm disabled");
            ctx->compute_ready = 0;
        }
    }

    vulkan_attach_fwd(e);
    ylog_info("vulkan: mode=%s gpu=%d layers=%u hidden=%u compute=%d",
              ctx->host_shim ? "host-shim" : "native",
              ctx->device_id, ctx->n_layers, ctx->hidden, ctx->compute_ready);
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
