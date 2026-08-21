/* device_vulkan.c — Vulkan 设备后端(Android / iOS·MoltenVK / PC)
 *
 * P0: host-shim — 与 CPU 相同算子路径, 验证 --device vulkan 绑定与打包。
 * 后续: VkBuffer 权/KV + compute shader(见 docs/design-mobile.md)。
 */
#include "device.h"
#include "yllm.h"
#include "log.h"
#include "vulkan_ctx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int vk_load_weights(Engine* e, char* err, size_t errlen)
{
    VulkanCtx* ctx = e->dev && e->dev->handle ? (VulkanCtx*)e->dev->handle : NULL;
    if (!ctx) {
        if (err && errlen) snprintf(err, errlen, "null vulkan ctx");
        return -1;
    }
    /* P0 host-shim: 权仍走 mmap; d_kv 别名 host KV */
    e->w_dev = ctx;
    e->d_kv = e->kv;
    e->weights_ready = 1;
    e->device_mode = ctx->host_shim ? DEV_MODE_VULKAN_HOST : DEV_MODE_VULKAN;
    engine_attach_cpu_fwd(e);
    ctx->n_layers = e->ws.model.n_layers;
    ctx->hidden = e->ws.model.h.hidden;
    ylog_info("vulkan: %s device_id=%d layers=%u (P0 host-shim CPU compute)",
              ctx->host_shim ? "host-shim" : "native",
              ctx->device_id, ctx->n_layers);
    return 0;
}

static void vk_free_dev(Engine* e)
{
    if (e->dev && e->dev->handle) {
        free(e->dev->handle);
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
    /* 真 VkInstance 创建尚未落地: 一律 host-shim, 保证三端可链可跑 */
    ctx->host_shim = 1;
    d->kind = DEV_VULKAN;
    d->id = device_id;
    d->handle = ctx;
    d->load_weights = vk_load_weights;
    d->free_dev = vk_free_dev;
    d->prefetch_layer = NULL;
    d->release_layer = NULL;
    ylog_info("vulkan: created host-shim backend (shaders TBD)");
    return d;
#endif
}
