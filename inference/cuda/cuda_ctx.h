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

    /* 真 GPU: Q4_K 原生上卡; 非 Q4 线性权解到 FP16; norm/bias F32 */
    uint8_t* w_q4;              /* device Q4_K 字节包 */
    uint16_t* w_f16;            /* device 线性权 (非 Q4 解量化) */
    float* w_f32;               /* device norm/bias */
    uint64_t* off_q4;           /* host: Q4_K 字节偏移 */
    uint64_t* off_f16;          /* host: FP16 元素偏移 */
    uint64_t* off_f32;
    uint32_t* dim_out;
    uint32_t* dim_in;
    uint64_t n_q4;              /* Q4_K 字节数 */
    uint64_t n_f16;             /* FP16 元素数 */
    uint64_t n_f32;

    /* 流式上权: host 打包常驻; 设备缓冲 = 单层峰值; off_* 为当前驻留层的设备偏移 */
    uint8_t* h_q4;
    uint16_t* h_f16;
    float* h_f32;
    uint64_t* host_off_q4;
    uint64_t* host_off_f16;
    uint64_t* host_off_f32;
    int stream_w;
    uint32_t stream_layer;      /* 当前在设备上的层; ~0u = 无 */
    uint64_t max_layer_q4;
    uint64_t max_layer_f16;
    uint64_t max_layer_f32;

    uint32_t n_layers;
    uint16_t* kv_blob;          /* device KV 或 shim 下 NULL */
    size_t kv_bytes;

    /* device 激活 */
    float* d_x;
    float* d_hb;
    float* d_hb2;
    float* d_ffn;
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
