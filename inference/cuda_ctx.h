/* cuda_ctx.h — CUDA 后端私有状态(device_cuda / cuda_fwd / kernels 共享) */
#ifndef YLLM_CUDA_CTX_H
#define YLLM_CUDA_CTX_H

#include <stdint.h>
#include <stddef.h>

#define CUDA_OFF_NONE ((uint64_t)~0ULL)

typedef struct {
    int device_id;
    int host_shim;              /* 1 = RAM 镜像 + CPU 算子 */
    int gpu_compute;            /* 1 = FP16 权 + cublas decode */

    /* host-shim / 过渡: raw LLF 层 blob */
    uint8_t* w_blob;
    uint64_t* layer_off;
    uint64_t w_bytes;

    /* 真 GPU: 线性权解量化为 FP32 上卡(对齐 CPU Q4 路径; 显存换数值) */
    float* w_f32w;              /* device 线性权 */
    float* w_f32;               /* device norm/bias */
    uint64_t* off_f16;          /* host: 线性权元素偏移(沿用表名) */
    uint64_t* off_f32;
    uint32_t* dim_out;
    uint32_t* dim_in;
    uint64_t n_f16;             /* 线性权元素数 */
    uint64_t n_f32;

    uint32_t n_layers;
    uint16_t* kv_blob;          /* device KV 或 shim 下 NULL */
    size_t kv_bytes;

    /* device 激活(仅 gpu_compute) */
    float* d_x;
    float* d_hb;
    float* d_hb2;
    float* d_ffn;
    float* d_att;
    float* d_logits;
    void* cublas;               /* cublasHandle_t */
    uint32_t hidden;
    uint32_t kv_dim;
    uint32_t max_seq;
    uint32_t inter;
    uint32_t vocab;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t n_blocks;
    uint32_t arch;
    float eps;
    float theta;
} CudaCtx;

#endif /* YLLM_CUDA_CTX_H */
