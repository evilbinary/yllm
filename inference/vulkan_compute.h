/* vulkan_compute.h — rmsnorm compute 管线 */
#ifndef YLLM_VULKAN_COMPUTE_H
#define YLLM_VULKAN_COMPUTE_H

#include "vulkan_ctx.h"

/* 创建 buffer/pipeline; spv_path 可为 NULL(搜默认路径) */
int vulkan_compute_setup(VulkanCtx* ctx, uint32_t hidden, const char* spv_path,
                         char* err, size_t errlen);
/* y[n] = rmsnorm(x, w); host 指针, 内部 H2D/D2H */
int vulkan_k_rmsnorm(VulkanCtx* ctx, float* y, const float* x, const float* w,
                     uint32_t n, float eps);

#endif
