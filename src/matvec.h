#ifndef YLLM_MATVEC_H
#define YLLM_MATVEC_H

#include <stdint.h>

void matmul(float* y, const float* x, const uint8_t* w, uint32_t out, uint32_t in, uint32_t dtype);
void matmul_f32_t(float* y, const float* x, const uint8_t* w, uint32_t in, uint32_t out);
void matmul_f16_t(float* y, const float* x, const uint8_t* w, uint32_t in, uint32_t out);
void matmul_q4k(float* y, const float* x, const uint8_t* w, uint32_t out, uint32_t in);
void matmul_q6k(float* y, const float* x, const uint8_t* w, uint32_t out, uint32_t in);
void matmul_iq4xs(float* y, const float* x, const uint8_t* w, uint32_t out, uint32_t in);

void embed_f32(float* y, const uint8_t* w, uint32_t row, uint32_t hidden);
void embed_f16(float* y, const uint8_t* w, uint32_t row, uint32_t hidden);
void embed_q4k(float* y, const uint8_t* w, uint32_t row, uint32_t hidden);
void embed_q6k(float* y, const uint8_t* w, uint32_t row, uint32_t hidden);
void embed_iq4xs(float* y, const uint8_t* w, uint32_t row, uint32_t hidden);

void rmsnorm(float* y, const float* x, const uint8_t* w, uint32_t n, float eps, uint32_t dtype);
void rope_inplace(float* v, uint32_t d, uint32_t pos, float theta);
void softmax(float* v, uint32_t n);
void swiglu(float* y, const float* gate, const float* up, uint32_t n);

#endif
