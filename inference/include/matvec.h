#ifndef YLLM_MATVEC_H
#define YLLM_MATVEC_H

#include <stdint.h>

void matmul(float* y, const float* x, const uint8_t* w, uint32_t out, uint32_t in, uint32_t dtype);
void matmul_f32_t(float* y, const float* x, const uint8_t* w, uint32_t in, uint32_t out);
void matmul_f16_t(float* y, const float* x, const uint8_t* w, uint32_t in, uint32_t out);
void matmul_q4k(float* y, const float* x, const uint8_t* w, uint32_t out, uint32_t in);
void matmul_q6k(float* y, const float* x, const uint8_t* w, uint32_t out, uint32_t in);
void matmul_q5k(float* y, const float* x, const uint8_t* w, uint32_t out, uint32_t in);
void matmul_iq4xs(float* y, const float* x, const uint8_t* w, uint32_t out, uint32_t in);
void matmul_w4b64(float* y, const float* x, const uint8_t* w, uint32_t out, uint32_t in);

/* W4B64 Arm82: 整矩阵字节数; out/in 须分别整除 8/64 */
size_t w4b64_bytes(uint32_t out, uint32_t in);
/* 兼容旧 API: 等价 w4b64_bytes(1,in) 的行主序假象已废, 返回 0; 请用 w4b64_bytes */
size_t w4b64_row_bytes(uint32_t in);
/* 从 f32 行主序 [out×in] 打包为 Arm82 W4 */
int w4b64_pack_mat_f32(uint8_t* dst, const float* src, uint32_t out, uint32_t in);
/* 从 Q4_K 行主序打包为 Arm82 W4 */
int w4b64_pack_mat_q4k(uint8_t* dst, const uint8_t* src, uint32_t out, uint32_t in);
/* Arm82 → Arm86 (i8mm SRC_UNIT=8) 就地/分缓冲 permute */
int w4b64_permute_arm82_to_arm86(uint8_t* dst, const uint8_t* src, uint32_t out, uint32_t in);
/* 运行时优先 i8mm(若硬件支持); 1=开 0=关; 返回实际是否启用 */
int w4b64_set_prefer_i8mm(int on);
int w4b64_prefer_i8mm(void);
/* 硬件是否有 i8mm(batch GEMM 用); 与 prefer(GEMV) 分开 */
void w4b64_set_hw_i8mm(int on);
int w4b64_has_i8mm(void);
/* Q6_K matmul 前激活量化(与 matmul_q6k 一致) */
void matvec_q8k_quant(const float* x, float* xq, uint32_t n);

/* 行分块 matmul: 只算 [row_begin, row_begin+n_rows) 行。
 * 量化权重行主序; F32/F16 列主序(out = 总列数, 行区间在列上连续)。
 * 不支持行分块的 dtype 退化为整算(n_rows 全量)。 */
void matmul_rows(float* y, const float* x, const uint8_t* w,
                 uint32_t row_begin, uint32_t n_rows, uint32_t in, uint32_t out, uint32_t dtype);
/* 单行字节(行分块释放用); 不支持行分块的 dtype 返回 0 */
size_t matmul_row_bytes(uint32_t dtype, uint32_t in);

void embed_f32(float* y, const uint8_t* w, uint32_t row, uint32_t hidden);
void embed_f16(float* y, const uint8_t* w, uint32_t row, uint32_t hidden);
void embed_q4k(float* y, const uint8_t* w, uint32_t row, uint32_t hidden);
void embed_q6k(float* y, const uint8_t* w, uint32_t row, uint32_t hidden);
void embed_q5k(float* y, const uint8_t* w, uint32_t row, uint32_t hidden);
void embed_iq4xs(float* y, const uint8_t* w, uint32_t row, uint32_t hidden);

void rmsnorm(float* y, const float* x, const uint8_t* w, uint32_t n, float eps, uint32_t dtype);
void rope_inplace(float* v, uint32_t d, uint32_t pos, float theta);
void rope_inplace_qwen(float* v, uint32_t d, uint32_t pos, float theta);
/* NEOX RoPE, 可选 freq_factors[d/2]: ang = pos * theta^(-2j/d) / factor[j] (llama.cpp ggml_rope_ext) */
void rope_inplace_qwen_ff(float* v, uint32_t d, uint32_t pos, float theta, const float* freq_factors);
/* NEOX, 预计算 inv_freq[j] = theta^(-2j/d) [/ factor]; pos=0 为恒等 */
void rope_inplace_neox_if(float* v, uint32_t d, uint32_t pos, const float* inv_freq);
void rope_inplace_mrope(float* v, uint32_t head_dim, uint32_t n_dims, uint32_t pos, float theta);
void l2norm_inplace(float* v, uint32_t n, float eps);
void conv1d_update(float* state, const float* x, const uint8_t* w, uint32_t dim, uint32_t kwidth);
void gdn_state_update(float* S, float* out, const float* q, const float* k,
                      const float* v, const float* gate, const float* beta,
                      uint32_t num_k_heads, uint32_t hk, uint32_t n_vheads, uint32_t hv);
void rmsnorm_gated(float* y, const float* x, const uint8_t* w, const float* z,
                   uint32_t n, float eps, uint32_t dtype);
void softmax(float* v, uint32_t n);
void swiglu(float* y, const float* gate, const float* up, uint32_t n);
void geglu(float* y, const float* gate, const float* up, uint32_t n);
void gelu_inplace(float* y, uint32_t n);
void rmsnorm_unit(float* y, const float* x, uint32_t n, float eps);
void add_inplace(float* y, const float* x, uint32_t n);
float vec_dot_f32_f16(const float* a, const uint16_t* b, uint32_t n);
void vec_axpy_f16(float* y, const uint16_t* x, float a, uint32_t n);

/* 单块解量化(导出供测试/embed 使用) */
void q4k_block(float* y, const uint8_t* blk, uint32_t stride);
float q6k_val(const uint8_t* blk, uint32_t e);

/* 整矩阵解量化为 FP16 行主序 [out × in](CUDA load_weights 用) */
int dequant_mat_f16(uint16_t* dst, const uint8_t* w, uint32_t out, uint32_t in, uint32_t dtype);
/* 整矩阵解量化为 FP32 行主序 */
int dequant_mat_f32(float* dst, const uint8_t* w, uint32_t out, uint32_t in, uint32_t dtype);

#endif
