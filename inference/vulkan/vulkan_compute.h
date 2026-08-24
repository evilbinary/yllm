/* vulkan_compute.h — rmsnorm + Q4_K gemv + fused 段 */
#ifndef YLLM_VULKAN_COMPUTE_H
#define YLLM_VULKAN_COMPUTE_H

#include "vulkan_ctx.h"
#include <stddef.h>
#include <stdint.h>

int vulkan_compute_setup(VulkanCtx* ctx, uint32_t hidden,
                         uint32_t max_in, uint32_t max_out,
                         size_t total_wq_bytes, uint32_t n_layers, uint32_t nslot,
                         uint32_t lm_vocab, char* err, size_t errlen);

int vulkan_wq_upload(VulkanCtx* ctx, const void* blob, size_t bytes);
int vulkan_wq_upload_range(VulkanCtx* ctx, const void* src, size_t dst_off, size_t bytes);

/* 按层把 host_wq[base..) 上传到 GPU wq[0..); 更新 stream_base */
int vulkan_stream_layer(VulkanCtx* ctx, uint32_t layer);

int vulkan_k_rmsnorm(VulkanCtx* ctx, float* y, const float* x, const float* w,
                     uint32_t n, float eps);
int vulkan_k_rmsnorm_inplace(VulkanCtx* ctx, const float* w, uint32_t n, float eps);
int vulkan_k_lm_gemv(VulkanCtx* ctx, float* y, const float* x, int x_on_dev,
                     uint32_t vocab, uint32_t hidden, uint64_t w_byte_off, uint32_t dtype);
int vulkan_k_gemv_q4k(VulkanCtx* ctx, float* y, const float* x,
                      uint32_t out, uint32_t in, uint64_t w_byte_off);
int vulkan_k_gemv_q6k(VulkanCtx* ctx, float* y, const float* x,
                      uint32_t out, uint32_t in, uint64_t w_byte_off);
int vulkan_k_gemv_q4k_host(VulkanCtx* ctx, float* y, const float* x,
                           const uint8_t* w, uint32_t out, uint32_t in);

int vulkan_k_embed_q4k(VulkanCtx* ctx, float* host_y, const uint8_t* table,
                       uint32_t token, uint32_t hidden);

/* 一次 submit: rmsnorm(x)+Q/K/V gemv → host q,k,v */
int vulkan_fused_norm_qkv(VulkanCtx* ctx,
                          const float* x, const float* wn, uint32_t hidden, float eps,
                          float* q, float* k, float* v, uint32_t kv_dim,
                          uint64_t off_q, uint64_t off_k, uint64_t off_v);

/* 一次 submit: rmsnorm+gate+up → host (无 swiglu 时回退) */
int vulkan_fused_norm_gate_up(VulkanCtx* ctx,
                              const float* x, const float* wn, uint32_t hidden, float eps,
                              float* gate, float* up, uint32_t inter,
                              uint64_t off_gate, uint64_t off_up);

/* 一次 submit: rmsnorm+gate+up+swiglu+down → host out[hidden] */
int vulkan_fused_ffn(VulkanCtx* ctx,
                     const float* x, const float* wn, uint32_t hidden, float eps,
                     float* out, uint32_t inter,
                     uint64_t off_gate, uint64_t off_up, uint64_t off_down);

/* rope 后: 写 GPU KV + attn decode
 * off_o == ~(uint64_t)0: 只 attn → att_out
 * 否则同 submit 做 O gemv → att_out 为 O 输出 */
int vulkan_k_attn_decode(VulkanCtx* ctx,
                         const float* q, const float* k, const float* v, float* att_out,
                         uint32_t layer, uint32_t pos,
                         uint16_t* host_k_row, uint16_t* host_v_row,
                         uint64_t off_o);

int vulkan_attn_setup(VulkanCtx* ctx, uint32_t n_blocks, uint32_t max_seq,
                      uint32_t kv_dim, uint32_t n_heads, uint32_t n_kv_heads,
                      uint32_t head_dim, char* err, size_t errlen);

/* rmsnorm+QKV; 可选 bias/qk-norm; rope; attn(+O)。out 为 O(需 off_o) 或 attn */
int vulkan_fused_qkv_rope_attn(VulkanCtx* ctx,
                               const float* x, const float* wn, uint32_t hidden, float eps,
                               float* out, uint32_t kv_dim,
                               uint64_t off_q, uint64_t off_k, uint64_t off_v, uint64_t off_o,
                               uint32_t layer, uint32_t pos, uint32_t rope_mode, float theta,
                               const float* bq, const float* bk, const float* bv,
                               const uint8_t* qnorm, uint32_t qnorm_dtype,
                               const uint8_t* knorm, uint32_t knorm_dtype,
                               uint16_t* host_k_row, uint16_t* host_v_row);

/* 整层 fused(2 submit): 激活常驻 buf_x; 无 qk-norm */
int vulkan_fused_block(VulkanCtx* ctx,
                       float* host_x, int upload_x,
                       const float* wn1, const float* wn2,
                       uint32_t hidden, float eps, float theta, uint32_t rope_mode,
                       uint32_t kv_dim, uint32_t inter,
                       uint64_t off_q, uint64_t off_k, uint64_t off_v, uint64_t off_o,
                       uint64_t off_g, uint64_t off_u, uint64_t off_d,
                       uint32_t layer, uint32_t pos,
                       const float* bq, const float* bk, const float* bv,
                       uint16_t* host_k_row, uint16_t* host_v_row,
                       int sync_host);

void vulkan_mark_x_host(VulkanCtx* ctx);
int vulkan_upload_x(VulkanCtx* ctx, const float* x, uint32_t hidden);
int vulkan_sync_x_to_host(VulkanCtx* ctx, float* host_x, uint32_t hidden);

/* YLLM_VK_PROF=1 时打印阶段耗时 */
void vulkan_prof_dump(VulkanCtx* ctx);

#endif
