/* vulkan_compute.h — rmsnorm + Q4_K gemv + fused 段 */
#ifndef YLLM_VULKAN_COMPUTE_H
#define YLLM_VULKAN_COMPUTE_H

#include "vulkan_ctx.h"
#include <stddef.h>
#include <stdint.h>

int vulkan_compute_setup(VulkanCtx* ctx, uint32_t hidden,
                         uint32_t max_in, uint32_t max_out,
                         size_t total_wq_bytes, uint32_t n_layers, uint32_t nslot,
                         char* err, size_t errlen);

int vulkan_wq_upload(VulkanCtx* ctx, const void* blob, size_t bytes);

int vulkan_k_rmsnorm(VulkanCtx* ctx, float* y, const float* x, const float* w,
                     uint32_t n, float eps);
int vulkan_k_gemv_q4k(VulkanCtx* ctx, float* y, const float* x,
                      uint32_t out, uint32_t in, uint64_t w_byte_off);
int vulkan_k_gemv_q4k_host(VulkanCtx* ctx, float* y, const float* x,
                           const uint8_t* w, uint32_t out, uint32_t in);

/* 一次 submit: rmsnorm(x)+Q/K/V gemv → host q,k,v */
int vulkan_fused_norm_qkv(VulkanCtx* ctx,
                          const float* x, const float* wn, uint32_t hidden, float eps,
                          float* q, float* k, float* v, uint32_t kv_dim,
                          uint64_t off_q, uint64_t off_k, uint64_t off_v);

/* 一次 submit: rmsnorm(x)+gate+up → host */
int vulkan_fused_norm_gate_up(VulkanCtx* ctx,
                              const float* x, const float* wn, uint32_t hidden, float eps,
                              float* gate, float* up, uint32_t inter,
                              uint64_t off_gate, uint64_t off_up);

#endif
