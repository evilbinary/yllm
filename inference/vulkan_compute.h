/* vulkan_compute.h — rmsnorm + Q4_K gemv */
#ifndef YLLM_VULKAN_COMPUTE_H
#define YLLM_VULKAN_COMPUTE_H

#include "vulkan_ctx.h"
#include <stddef.h>
#include <stdint.h>

/* max_in/out: 层内矩阵维; max_wq_bytes: 单次 gemv 最大权字节 */
int vulkan_compute_setup(VulkanCtx* ctx, uint32_t hidden,
                         uint32_t max_in, uint32_t max_out, size_t max_wq_bytes,
                         char* err, size_t errlen);
int vulkan_k_rmsnorm(VulkanCtx* ctx, float* y, const float* x, const float* w,
                     uint32_t n, float eps);
int vulkan_k_gemv_q4k(VulkanCtx* ctx, float* y, const float* x, const uint8_t* w,
                      uint32_t out, uint32_t in);

#endif
