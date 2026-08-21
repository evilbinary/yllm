/* vulkan_compute.h — rmsnorm + Q4_K gemv(常驻权) */
#ifndef YLLM_VULKAN_COMPUTE_H
#define YLLM_VULKAN_COMPUTE_H

#include "vulkan_ctx.h"
#include <stddef.h>
#include <stdint.h>

int vulkan_compute_setup(VulkanCtx* ctx, uint32_t hidden,
                         uint32_t max_in, uint32_t max_out,
                         size_t total_wq_bytes, uint32_t n_layers, uint32_t nslot,
                         char* err, size_t errlen);

/* 将打包好的 Q4_K blob 写入常驻缓冲, 并标记 resident */
int vulkan_wq_upload(VulkanCtx* ctx, const void* blob, size_t bytes);

int vulkan_k_rmsnorm(VulkanCtx* ctx, float* y, const float* x, const float* w,
                     uint32_t n, float eps);

/* 常驻权: w_byte_off 指向 blob 内偏移 */
int vulkan_k_gemv_q4k(VulkanCtx* ctx, float* y, const float* x,
                      uint32_t out, uint32_t in, uint64_t w_byte_off);

/* 自检/临时: 把 host w 拷到 offset 0 再算 */
int vulkan_k_gemv_q4k_host(VulkanCtx* ctx, float* y, const float* x,
                           const uint8_t* w, uint32_t out, uint32_t in);

#endif
