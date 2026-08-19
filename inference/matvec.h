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

/* 单块解量化(导出供测试/embed 使用) */
void q4k_block(float* y, const uint8_t* blk, uint32_t stride);
float q6k_val(const uint8_t* blk, uint32_t e);

#endif
