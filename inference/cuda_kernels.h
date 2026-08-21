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

/* y[out] = W_rm[out,in](fp16) · x[in](fp32); xf16 为长度 in 的设备暂存 */
int cuda_k_gemv_f16(void* cublas, float* y, const uint16_t* w,
                    const float* x, uint16_t* xf16,
                    uint32_t out, uint32_t in, char* err, size_t errlen);
/* Y[B,out] = X[B,in] · W_rm^T; xf16 长度 ≥ B*in */
int cuda_k_gemm_f16(void* cublas, float* y, const uint16_t* w,
                    const float* x, uint16_t* xf16,
                    uint32_t out, uint32_t in, uint32_t B, char* err, size_t errlen);

void cuda_k_rmsnorm(float* y, const float* x, const float* w, uint32_t n, float eps);
void cuda_k_rmsnorm_batch(float* y, const float* x, const float* w, uint32_t n, float eps, uint32_t B);
void cuda_k_add(float* y, const float* a, const float* b, uint32_t n);
void cuda_k_add_batch(float* y, const float* a, const float* b, uint32_t n, uint32_t B);
void cuda_k_add_bias(float* y, const float* bias, uint32_t n);
void cuda_k_add_bias_batch(float* y, const float* bias, uint32_t n, uint32_t B);
void cuda_k_swiglu(float* y, const float* gate, const float* up, uint32_t n);
void cuda_k_swiglu_batch(float* y, const float* gate, const float* up, uint32_t n, uint32_t B);
void cuda_k_rope_llama(float* v, uint32_t d, uint32_t pos, float theta);
void cuda_k_rope_qwen(float* v, uint32_t d, uint32_t pos, float theta);
void cuda_k_rope_llama_heads(float* v, uint32_t n_heads, uint32_t head_dim, uint32_t pos, float theta);
void cuda_k_rope_qwen_heads(float* v, uint32_t n_heads, uint32_t head_dim, uint32_t pos, float theta);
void cuda_k_rope_llama_heads_batch(float* v, uint32_t n_heads, uint32_t head_dim,
                                   uint32_t pos_start, uint32_t B, float theta);
void cuda_k_rope_qwen_heads_batch(float* v, uint32_t n_heads, uint32_t head_dim,
                                  uint32_t pos_start, uint32_t B, float theta);

void cuda_k_store_kv(uint16_t* kcache, uint16_t* vcache, const float* k, const float* v,
                     uint32_t pos, uint32_t kv_dim);
void cuda_k_store_kv_batch(uint16_t* kcache, uint16_t* vcache, const float* k, const float* v,
                           uint32_t pos_start, uint32_t B, uint32_t kv_dim);

void cuda_k_attn_decode(float* att_out, float* att_scores,
                        const float* q, const uint16_t* kcache, const uint16_t* vcache,
                        uint32_t pos, uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
                        uint32_t kv_dim, uint32_t max_seq);
void cuda_k_attn_prefill(float* att_out, float* att_scores,
                         const float* q, const uint16_t* kcache, const uint16_t* vcache,
                         uint32_t pos_start, uint32_t B, uint32_t n_heads, uint32_t n_kv_heads,
                         uint32_t head_dim, uint32_t kv_dim, uint32_t max_seq, uint32_t q_stride);

/* embed: 从 FP16 行表取 token 行 → FP32 */
void cuda_k_embed_f16(float* y, const uint16_t* table, uint32_t token, uint32_t hidden);
void cuda_k_embed_f16_batch(float* y, const uint16_t* table, const uint32_t* tokens_dev,
                            uint32_t B, uint32_t hidden);

int cuda_k_memcpy_h2d(void* dst, const void* src, size_t n);
int cuda_k_memcpy_d2h(void* dst, const void* src, size_t n);
int cuda_k_memcpy_d2d(void* dst, const void* src, size_t n);
int cuda_k_memset_zero(void* dst, size_t n);
void cuda_k_sync(void);

#ifdef __cplusplus
}
#endif

#endif /* YLLM_CUDA_KERNELS_H */
