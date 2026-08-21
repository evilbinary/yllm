/* cuda_kernels.h — 真 CUDA decode 算子 C ABI (仅 YLLM_CUDA && !YLLM_CUDA_HOST) */
#ifndef YLLM_CUDA_KERNELS_H
#define YLLM_CUDA_KERNELS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct Engine;
struct CudaCtx;

int cuda_k_cublas_create(void** out_handle, char* err, size_t errlen);
void cuda_k_cublas_destroy(void* handle);

/* y[out] = W_rm[out,in](fp32) · x[in](fp32) */
int cuda_k_gemv_f32(void* cublas, float* y, const float* w,
                    const float* x, uint32_t out, uint32_t in, char* err, size_t errlen);

void cuda_k_rmsnorm(float* y, const float* x, const float* w, uint32_t n, float eps);
void cuda_k_add(float* y, const float* a, const float* b, uint32_t n);
void cuda_k_add_bias(float* y, const float* bias, uint32_t n);
void cuda_k_swiglu(float* y, const float* gate, const float* up, uint32_t n);
void cuda_k_rope_llama(float* v, uint32_t d, uint32_t pos, float theta);
void cuda_k_rope_qwen(float* v, uint32_t d, uint32_t pos, float theta);
void cuda_k_rope_llama_heads(float* v, uint32_t n_heads, uint32_t head_dim, uint32_t pos, float theta);
void cuda_k_rope_qwen_heads(float* v, uint32_t n_heads, uint32_t head_dim, uint32_t pos, float theta);

void cuda_k_store_kv(uint16_t* kcache, uint16_t* vcache, const float* k, const float* v,
                     uint32_t pos, uint32_t kv_dim);

void cuda_k_attn_decode(float* att_out, float* att_scores,
                        const float* q, const uint16_t* kcache, const uint16_t* vcache,
                        uint32_t pos, uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
                        uint32_t kv_dim, uint32_t max_seq);

/* embed: 从 FP32 行表取 token 行 */
void cuda_k_embed_f32(float* y, const float* table, uint32_t token, uint32_t hidden);

int cuda_k_memcpy_h2d(void* dst, const void* src, size_t n);
int cuda_k_memcpy_d2h(void* dst, const void* src, size_t n);
int cuda_k_memcpy_d2d(void* dst, const void* src, size_t n);
int cuda_k_memset_zero(void* dst, size_t n);
void cuda_k_sync(void);

#ifdef __cplusplus
}
#endif

#endif /* YLLM_CUDA_KERNELS_H */
