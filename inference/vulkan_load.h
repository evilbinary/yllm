/* vulkan_load.h — 动态加载 Vulkan(免链 MSVC vulkan-1.lib; Android 也可 dlopen) */
#ifndef YLLM_VULKAN_LOAD_H
#define YLLM_VULKAN_LOAD_H

#include "vulkan_ctx.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 尝试创建 instance + 选 physical/logical device。
 * 成功: ctx->host_shim=0, 填 instance/phys/device/queue。
 * 失败: 返回 -1, 调用方保持 host-shim。 */
int vulkan_try_init(VulkanCtx* ctx, int device_id, char* err, size_t errlen);
void vulkan_shutdown(VulkanCtx* ctx);

#ifdef __cplusplus
}
#endif

#endif
