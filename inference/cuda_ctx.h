/* cuda_ctx.h — CUDA 后端私有状态(device_cuda / cuda_fwd / kernels 共享) */
#ifndef YLLM_CUDA_CTX_H
#define YLLM_CUDA_CTX_H

#include <stdint.h>
#include <stddef.h>

#define CUDA_OFF_NONE ((uint64_t)~0ULL)

typedef struct {
    int device_id;
    int host_shim;              /* 1 = RAM 镜像 + CPU 算子 */

    /* host-shim / 过渡: raw LLF 层 blob */
    uint8_t* w_blob;
    uint64_t* layer_off;
    uint64_t w_bytes;

    /* 真 GPU: 线性权解量化为 FP16 上卡; norm/bias 仍 F32 */
    uint16_t* w_f16;            /* device 线性权 (IEEE half) */
    float* w_f32;               /* device norm/bias */
    uint64_t* off_f16;          /* host: 线性权元素偏移 */
    uint64_t* off_f32;
    uint32_t* dim_out;
    uint32_t* dim_in;
    uint64_t n_f16;             /* 线性权元素数 */
    uint64_t n_f32;

    uint32_t n_layers;
    uint16_t* kv_blob;          /* device KV 或 shim 下 NULL */
    size_t kv_bytes;

    /* device 激活 */
    float* d_x;
    float* d_hb;
    float* d_hb2;
    float* d_ffn;
    float* d_att;
    float* d_logits;
    uint16_t* d_xf16;           /* gemv/gemm 激活 FP16 暂存 */
    /* 批量 prefill (≤ pb_cap) */
    float* d_pb;
    float* d_pb2;
    float* d_pbq;
    float* d_pbk;
    float* d_pbv;
    float* d_pbg;
    float* d_pbu;
    float* d_pba;
    uint32_t* d_tokens;         /* prefill token ids [pb_cap] */
    uint32_t pb_cap;
    void* cublas;               /* cublasHandle_t */
    int x_on_dev;               /* 1 = d_x 为权威激活, 跳过层间 H2D/D2H */
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
