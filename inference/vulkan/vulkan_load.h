/* vulkan_load.h */
#ifndef YLLM_VULKAN_LOAD_H
#define YLLM_VULKAN_LOAD_H

#include "vulkan_ctx.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int vulkan_try_init(VulkanCtx* ctx, int device_id, char* err, size_t errlen);
void vulkan_shutdown(VulkanCtx* ctx);

#ifdef __cplusplus
}
#endif

#endif
