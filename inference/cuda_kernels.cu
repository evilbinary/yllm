/* cuda_kernels.cu — P2 decode: cublas GEMV + 小算子 */
#include "cuda_kernels.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#define CK_OK(call, err, errlen) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        if (err && errlen) snprintf(err, errlen, "CUDA: %s", cudaGetErrorString(_e)); \
        return -1; \
    } \
} while (0)

extern "C" int cuda_k_cublas_create(void** out_handle, char* err, size_t errlen)
{
    cublasHandle_t h = NULL;
    cublasStatus_t st = cublasCreate(&h);
    if (st != CUBLAS_STATUS_SUCCESS) {
        if (err && errlen) snprintf(err, errlen, "cublasCreate failed (%d)", (int)st);
        return -1;
    }
    *out_handle = (void*)h;
    return 0;
}

extern "C" void cuda_k_cublas_destroy(void* handle)
{
    if (handle) cublasDestroy((cublasHandle_t)handle);
}

__global__ void k_gemv_f16(float* y, const uint16_t* w, const float* x, uint32_t out, uint32_t in)
{
    uint32_t o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= out) return;
    const uint16_t* row = w + (size_t)o * in;
    float acc = 0.0f;
    for (uint32_t i = 0; i < in; i++)
        acc += __half2float(__ushort_as_half(row[i])) * x[i];
    y[o] = acc;
}

__global__ void k_f32_to_f16(uint16_t* y, const float* x, uint32_t n)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __half_as_ushort(__float2half(x[i]));
}

extern "C" int cuda_k_gemv_f16(void* cublas, float* y, const uint16_t* w,
                               const float* x, uint16_t* xf16,
                               uint32_t out, uint32_t in, char* err, size_t errlen)
{
    if (!y || !w || !x || out == 0 || in == 0) {
        if (err && errlen) snprintf(err, errlen, "gemv bad args");
        return -1;
    }
    if (cublas && xf16) {
        const float alpha = 1.0f, beta = 0.0f;
        k_f32_to_f16<<<(in + 255) / 256, 256>>>(xf16, x, in);
        /* W_rm[out,in] 作列主 (in x out), OP_T → y = W · x; 两侧 FP16, 累加 FP32 */
        cublasStatus_t st = cublasGemmEx((cublasHandle_t)cublas,
                                         CUBLAS_OP_T, CUBLAS_OP_N,
                                         (int)out, 1, (int)in,
                                         &alpha,
                                         w, CUDA_R_16F, (int)in,
                                         xf16, CUDA_R_16F, (int)in,
                                         &beta,
                                         y, CUDA_R_32F, (int)out,
                                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (st != CUBLAS_STATUS_SUCCESS) {
            if (err && errlen) snprintf(err, errlen, "cublasGemmEx %d", (int)st);
            return -1;
        }
        return 0;
    }
    k_gemv_f16<<<(out + 255) / 256, 256>>>(y, w, x, out, in);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        if (err && errlen) snprintf(err, errlen, "gemv launch: %s", cudaGetErrorString(e));
        return -1;
    }
    return 0;
}

extern "C" int cuda_k_gemm_f16(void* cublas, float* y, const uint16_t* w,
                               const float* x, uint16_t* xf16,
                               uint32_t out, uint32_t in, uint32_t B, char* err, size_t errlen)
{
    if (!y || !w || !x || out == 0 || in == 0 || B == 0) {
        if (err && errlen) snprintf(err, errlen, "gemm bad args");
        return -1;
    }
    if (B == 1)
        return cuda_k_gemv_f16(cublas, y, w, x, xf16, out, in, err, errlen);
    if (cublas && xf16) {
        const float alpha = 1.0f, beta = 0.0f;
        uint32_t n = B * in;
        k_f32_to_f16<<<(n + 255) / 256, 256>>>(xf16, x, n);
        /* X_rm[B,in] / Y_rm[B,out] 作列主 (in×B)/(out×B); W OP_T */
        cublasStatus_t st = cublasGemmEx((cublasHandle_t)cublas,
                                         CUBLAS_OP_T, CUBLAS_OP_N,
                                         (int)out, (int)B, (int)in,
                                         &alpha,
                                         w, CUDA_R_16F, (int)in,
                                         xf16, CUDA_R_16F, (int)in,
                                         &beta,
                                         y, CUDA_R_32F, (int)out,
                                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (st != CUBLAS_STATUS_SUCCESS) {
            if (err && errlen) snprintf(err, errlen, "cublasGemmEx B=%u %d", B, (int)st);
            return -1;
        }
        return 0;
    }
    for (uint32_t b = 0; b < B; b++) {
        if (cuda_k_gemv_f16(NULL, y + (size_t)b * out, w, x + (size_t)b * in, NULL,
                            out, in, err, errlen) != 0)
            return -1;
    }
    return 0;
}

__global__ void k_rmsnorm(float* y, const float* x, const float* w, uint32_t n, float eps)
{
    float s = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) s += x[i] * x[i];
    __shared__ float sh[256];
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) sh[threadIdx.x] += sh[threadIdx.x + stride];
        __syncthreads();
    }
    float inv = rsqrtf(sh[0] / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x)
        y[i] = x[i] * w[i] * inv;
}

extern "C" void cuda_k_rmsnorm(float* y, const float* x, const float* w, uint32_t n, float eps)
{
    k_rmsnorm<<<1, 256>>>(y, x, w, n, eps);
}

__global__ void k_rmsnorm_batch(float* y, const float* x, const float* w, uint32_t n, float eps)
{
    uint32_t b = blockIdx.x;
    const float* xb = x + (size_t)b * n;
    float* yb = y + (size_t)b * n;
    float s = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) s += xb[i] * xb[i];
    __shared__ float sh[256];
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) sh[threadIdx.x] += sh[threadIdx.x + stride];
        __syncthreads();
    }
    float inv = rsqrtf(sh[0] / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x)
        yb[i] = xb[i] * w[i] * inv;
}

extern "C" void cuda_k_rmsnorm_batch(float* y, const float* x, const float* w, uint32_t n, float eps, uint32_t B)
{
    if (B == 0) return;
    k_rmsnorm_batch<<<B, 256>>>(y, x, w, n, eps);
}

__global__ void k_add(float* y, const float* a, const float* b, uint32_t n)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i];
}

extern "C" void cuda_k_add(float* y, const float* a, const float* b, uint32_t n)
{
    k_add<<<(n + 255) / 256, 256>>>(y, a, b, n);
}

extern "C" void cuda_k_add_batch(float* y, const float* a, const float* b, uint32_t n, uint32_t B)
{
    uint32_t tot = n * B;
    if (tot == 0) return;
    k_add<<<(tot + 255) / 256, 256>>>(y, a, b, tot);
}

__global__ void k_add_bias(float* y, const float* bias, uint32_t n)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += bias[i];
}

extern "C" void cuda_k_add_bias(float* y, const float* bias, uint32_t n)
{
    k_add_bias<<<(n + 255) / 256, 256>>>(y, bias, n);
}

__global__ void k_add_bias_batch(float* y, const float* bias, uint32_t n, uint32_t B)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t tot = n * B;
    if (i >= tot) return;
    y[i] += bias[i % n];
}

extern "C" void cuda_k_add_bias_batch(float* y, const float* bias, uint32_t n, uint32_t B)
{
    uint32_t tot = n * B;
    if (tot == 0) return;
    k_add_bias_batch<<<(tot + 255) / 256, 256>>>(y, bias, n, B);
}

__global__ void k_swiglu(float* y, const float* gate, const float* up, uint32_t n)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        float silu = g / (1.0f + expf(-g));
        y[i] = silu * up[i];
    }
}

extern "C" void cuda_k_swiglu(float* y, const float* gate, const float* up, uint32_t n)
{
    k_swiglu<<<(n + 255) / 256, 256>>>(y, gate, up, n);
}

extern "C" void cuda_k_swiglu_batch(float* y, const float* gate, const float* up, uint32_t n, uint32_t B)
{
    uint32_t tot = n * B;
    if (tot == 0) return;
    k_swiglu<<<(tot + 255) / 256, 256>>>(y, gate, up, tot);
}

__global__ void k_rope_llama(float* v, uint32_t d, uint32_t pos, float theta)
{
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t half = d / 2;
    if (j >= half) return;
    float freq = powf(theta, -2.0f * (float)j / (float)d);
    float ang = freq * (float)pos;
    float c = cosf(ang), s = sinf(ang);
    float a = v[2 * j], b = v[2 * j + 1];
    v[2 * j] = a * c - b * s;
    v[2 * j + 1] = a * s + b * c;
}

__global__ void k_rope_qwen(float* v, uint32_t d, uint32_t pos, float theta)
{
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t half = d / 2;
    if (j >= half) return;
    float freq = powf(theta, -2.0f * (float)j / (float)d);
    float ang = freq * (float)pos;
    float c = cosf(ang), s = sinf(ang);
    float a = v[j], b = v[j + half];
    v[j] = a * c - b * s;
    v[j + half] = a * s + b * c;
}

extern "C" void cuda_k_rope_llama(float* v, uint32_t d, uint32_t pos, float theta)
{
    k_rope_llama<<<(d / 2 + 31) / 32, 32>>>(v, d, pos, theta);
}

extern "C" void cuda_k_rope_qwen(float* v, uint32_t d, uint32_t pos, float theta)
{
    k_rope_qwen<<<(d / 2 + 31) / 32, 32>>>(v, d, pos, theta);
}

__global__ void k_rope_llama_heads(float* v, uint32_t n_heads, uint32_t head_dim, uint32_t pos, float theta)
{
    uint32_t hh = blockIdx.x;
    uint32_t j = threadIdx.x;
    uint32_t half = head_dim / 2;
    if (hh >= n_heads || j >= half) return;
    float* vh = v + (size_t)hh * head_dim;
    float freq = powf(theta, -2.0f * (float)j / (float)head_dim);
    float ang = freq * (float)pos;
    float c = cosf(ang), s = sinf(ang);
    float a = vh[2 * j], b = vh[2 * j + 1];
    vh[2 * j] = a * c - b * s;
    vh[2 * j + 1] = a * s + b * c;
}

__global__ void k_rope_qwen_heads(float* v, uint32_t n_heads, uint32_t head_dim, uint32_t pos, float theta)
{
    uint32_t hh = blockIdx.x;
    uint32_t j = threadIdx.x;
    uint32_t half = head_dim / 2;
    if (hh >= n_heads || j >= half) return;
    float* vh = v + (size_t)hh * head_dim;
    float freq = powf(theta, -2.0f * (float)j / (float)head_dim);
    float ang = freq * (float)pos;
    float c = cosf(ang), s = sinf(ang);
    float a = vh[j], b = vh[j + half];
    vh[j] = a * c - b * s;
    vh[j + half] = a * s + b * c;
}

extern "C" void cuda_k_rope_llama_heads(float* v, uint32_t n_heads, uint32_t head_dim, uint32_t pos, float theta)
{
    k_rope_llama_heads<<<n_heads, (head_dim / 2 + 31) / 32 * 32>>>(v, n_heads, head_dim, pos, theta);
}

extern "C" void cuda_k_rope_qwen_heads(float* v, uint32_t n_heads, uint32_t head_dim, uint32_t pos, float theta)
{
    uint32_t t = head_dim / 2;
    if (t < 32) t = 32;
    k_rope_qwen_heads<<<n_heads, t>>>(v, n_heads, head_dim, pos, theta);
}

__global__ void k_rope_llama_heads_batch(float* v, uint32_t n_heads, uint32_t head_dim,
                                         uint32_t pos_start, uint32_t B, float theta)
{
    uint32_t b = blockIdx.y;
    uint32_t hh = blockIdx.x;
    uint32_t j = threadIdx.x;
    uint32_t half = head_dim / 2;
    if (b >= B || hh >= n_heads || j >= half) return;
    float* vh = v + (size_t)b * n_heads * head_dim + (size_t)hh * head_dim;
    uint32_t pos = pos_start + b;
    float freq = powf(theta, -2.0f * (float)j / (float)head_dim);
    float ang = freq * (float)pos;
    float c = cosf(ang), s = sinf(ang);
    float a = vh[2 * j], bv = vh[2 * j + 1];
    vh[2 * j] = a * c - bv * s;
    vh[2 * j + 1] = a * s + bv * c;
}

__global__ void k_rope_qwen_heads_batch(float* v, uint32_t n_heads, uint32_t head_dim,
                                        uint32_t pos_start, uint32_t B, float theta)
{
    uint32_t b = blockIdx.y;
    uint32_t hh = blockIdx.x;
    uint32_t j = threadIdx.x;
    uint32_t half = head_dim / 2;
    if (b >= B || hh >= n_heads || j >= half) return;
    float* vh = v + (size_t)b * n_heads * head_dim + (size_t)hh * head_dim;
    uint32_t pos = pos_start + b;
    float freq = powf(theta, -2.0f * (float)j / (float)head_dim);
    float ang = freq * (float)pos;
    float c = cosf(ang), s = sinf(ang);
    float a = vh[j], bv = vh[j + half];
    vh[j] = a * c - bv * s;
    vh[j + half] = a * s + bv * c;
}

extern "C" void cuda_k_rope_llama_heads_batch(float* v, uint32_t n_heads, uint32_t head_dim,
                                              uint32_t pos_start, uint32_t B, float theta)
{
    dim3 grid(n_heads, B);
    k_rope_llama_heads_batch<<<grid, (head_dim / 2 + 31) / 32 * 32>>>(
        v, n_heads, head_dim, pos_start, B, theta);
}

extern "C" void cuda_k_rope_qwen_heads_batch(float* v, uint32_t n_heads, uint32_t head_dim,
                                             uint32_t pos_start, uint32_t B, float theta)
{
    uint32_t t = head_dim / 2;
    if (t < 32) t = 32;
    dim3 grid(n_heads, B);
    k_rope_qwen_heads_batch<<<grid, t>>>(v, n_heads, head_dim, pos_start, B, theta);
}

__global__ void k_store_kv(uint16_t* kcache, uint16_t* vcache, const float* k, const float* v,
                           uint32_t pos, uint32_t kv_dim)
{
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= kv_dim) return;
    uint64_t off = (uint64_t)pos * kv_dim + j;
    kcache[off] = __half_as_ushort(__float2half_rn(k[j]));
    vcache[off] = __half_as_ushort(__float2half_rn(v[j]));
}

extern "C" void cuda_k_store_kv(uint16_t* kcache, uint16_t* vcache, const float* k, const float* v,
                                uint32_t pos, uint32_t kv_dim)
{
    k_store_kv<<<(kv_dim + 255) / 256, 256>>>(kcache, vcache, k, v, pos, kv_dim);
}

__global__ void k_store_kv_batch(uint16_t* kcache, uint16_t* vcache, const float* k, const float* v,
                                 uint32_t pos_start, uint32_t B, uint32_t kv_dim)
{
    uint32_t b = blockIdx.y;
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || j >= kv_dim) return;
    uint64_t off = (uint64_t)(pos_start + b) * kv_dim + j;
    const float* kb = k + (size_t)b * kv_dim;
    const float* vb = v + (size_t)b * kv_dim;
    kcache[off] = __half_as_ushort(__float2half_rn(kb[j]));
    vcache[off] = __half_as_ushort(__float2half_rn(vb[j]));
}

extern "C" void cuda_k_store_kv_batch(uint16_t* kcache, uint16_t* vcache, const float* k, const float* v,
                                      uint32_t pos_start, uint32_t B, uint32_t kv_dim)
{
    dim3 grid((kv_dim + 255) / 256, B);
    k_store_kv_batch<<<grid, 256>>>(kcache, vcache, k, v, pos_start, B, kv_dim);
}

__global__ void k_attn_decode(float* att_out, float* att_scores,
                              const float* q, const uint16_t* kcache, const uint16_t* vcache,
                              uint32_t pos, uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
                              uint32_t kv_dim, uint32_t max_seq)
{
    uint32_t hh = blockIdx.x;
    if (hh >= n_heads) return;
    uint32_t kv_head = hh * n_kv_heads / n_heads;
    const float* qh = q + (size_t)hh * head_dim;
    float* att_h = att_scores + (size_t)hh * max_seq;
    float inv_d = rsqrtf((float)head_dim);

    for (uint32_t s = threadIdx.x; s <= pos; s += blockDim.x) {
        const uint16_t* kh = kcache + (size_t)s * kv_dim + (size_t)kv_head * head_dim;
        float acc = 0.0f;
        for (uint32_t jj = 0; jj < head_dim; jj++)
            acc += qh[jj] * __half2float(__ushort_as_half(kh[jj]));
        att_h[s] = acc * inv_d;
    }
    __syncthreads();

    /* softmax on att_h[0..pos] — serial in thread 0 for simplicity */
    if (threadIdx.x == 0) {
        float m = att_h[0];
        for (uint32_t s = 1; s <= pos; s++) if (att_h[s] > m) m = att_h[s];
        float sum = 0.0f;
        for (uint32_t s = 0; s <= pos; s++) {
            att_h[s] = expf(att_h[s] - m);
            sum += att_h[s];
        }
        float inv = 1.0f / sum;
        for (uint32_t s = 0; s <= pos; s++) att_h[s] *= inv;
    }
    __syncthreads();

    float* out = att_out + (size_t)hh * head_dim;
    for (uint32_t jj = threadIdx.x; jj < head_dim; jj += blockDim.x) {
        float o = 0.0f;
        for (uint32_t s = 0; s <= pos; s++) {
            const uint16_t* vh = vcache + (size_t)s * kv_dim + (size_t)kv_head * head_dim;
            o += att_h[s] * __half2float(__ushort_as_half(vh[jj]));
        }
        out[jj] = o;
    }
}

extern "C" void cuda_k_attn_decode(float* att_out, float* att_scores,
                                   const float* q, const uint16_t* kcache, const uint16_t* vcache,
                                   uint32_t pos, uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
                                   uint32_t kv_dim, uint32_t max_seq)
{
    k_attn_decode<<<n_heads, 128>>>(att_out, att_scores, q, kcache, vcache,
                                    pos, n_heads, n_kv_heads, head_dim, kv_dim, max_seq);
}

/* grid: (n_heads, B); q / att_out stride = q_stride (通常 = hidden = n_heads*head_dim)
 * att_scores: [B × n_heads × max_seq] */
__global__ void k_attn_prefill(float* att_out, float* att_scores,
                               const float* q, const uint16_t* kcache, const uint16_t* vcache,
                               uint32_t pos_start, uint32_t B, uint32_t n_heads, uint32_t n_kv_heads,
                               uint32_t head_dim, uint32_t kv_dim, uint32_t max_seq, uint32_t q_stride)
{
    uint32_t hh = blockIdx.x;
    uint32_t bb = blockIdx.y;
    if (hh >= n_heads || bb >= B) return;
    uint32_t pos = pos_start + bb;
    uint32_t kv_head = hh * n_kv_heads / n_heads;
    const float* qh = q + (size_t)bb * q_stride + (size_t)hh * head_dim;
    float* att_h = att_scores + ((size_t)bb * n_heads + hh) * max_seq;
    float inv_d = rsqrtf((float)head_dim);

    for (uint32_t s = threadIdx.x; s <= pos; s += blockDim.x) {
        const uint16_t* kh = kcache + (size_t)s * kv_dim + (size_t)kv_head * head_dim;
        float acc = 0.0f;
        for (uint32_t jj = 0; jj < head_dim; jj++)
            acc += qh[jj] * __half2float(__ushort_as_half(kh[jj]));
        att_h[s] = acc * inv_d;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float m = att_h[0];
        for (uint32_t s = 1; s <= pos; s++) if (att_h[s] > m) m = att_h[s];
        float sum = 0.0f;
        for (uint32_t s = 0; s <= pos; s++) {
            att_h[s] = expf(att_h[s] - m);
            sum += att_h[s];
        }
        float inv = 1.0f / sum;
        for (uint32_t s = 0; s <= pos; s++) att_h[s] *= inv;
    }
    __syncthreads();

    float* out = att_out + (size_t)bb * q_stride + (size_t)hh * head_dim;
    for (uint32_t jj = threadIdx.x; jj < head_dim; jj += blockDim.x) {
        float o = 0.0f;
        for (uint32_t s = 0; s <= pos; s++) {
            const uint16_t* vh = vcache + (size_t)s * kv_dim + (size_t)kv_head * head_dim;
            o += att_h[s] * __half2float(__ushort_as_half(vh[jj]));
        }
        out[jj] = o;
    }
}

extern "C" void cuda_k_attn_prefill(float* att_out, float* att_scores,
                                    const float* q, const uint16_t* kcache, const uint16_t* vcache,
                                    uint32_t pos_start, uint32_t B, uint32_t n_heads, uint32_t n_kv_heads,
                                    uint32_t head_dim, uint32_t kv_dim, uint32_t max_seq, uint32_t q_stride)
{
    dim3 grid(n_heads, B);
    k_attn_prefill<<<grid, 128>>>(att_out, att_scores, q, kcache, vcache,
                                  pos_start, B, n_heads, n_kv_heads, head_dim, kv_dim, max_seq, q_stride);
}

__global__ void k_embed_f16(float* y, const uint16_t* table, uint32_t token, uint32_t hidden)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden)
        y[i] = __half2float(__ushort_as_half(table[(size_t)token * hidden + i]));
}

extern "C" void cuda_k_embed_f16(float* y, const uint16_t* table, uint32_t token, uint32_t hidden)
{
    k_embed_f16<<<(hidden + 255) / 256, 256>>>(y, table, token, hidden);
}

__global__ void k_embed_f16_batch(float* y, const uint16_t* table, const uint32_t* tokens,
                                  uint32_t B, uint32_t hidden)
{
    uint32_t b = blockIdx.y;
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || i >= hidden) return;
    uint32_t token = tokens[b];
    y[(size_t)b * hidden + i] =
        __half2float(__ushort_as_half(table[(size_t)token * hidden + i]));
}

extern "C" void cuda_k_embed_f16_batch(float* y, const uint16_t* table, const uint32_t* tokens_dev,
                                       uint32_t B, uint32_t hidden)
{
    dim3 grid((hidden + 255) / 256, B);
    k_embed_f16_batch<<<grid, 256>>>(y, table, tokens_dev, B, hidden);
}

extern "C" int cuda_k_memcpy_h2d(void* dst, const void* src, size_t n)
{
    return cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice) == cudaSuccess ? 0 : -1;
}

extern "C" int cuda_k_memcpy_d2h(void* dst, const void* src, size_t n)
{
    return cudaMemcpy(dst, src, n, cudaMemcpyDeviceToHost) == cudaSuccess ? 0 : -1;
}

extern "C" int cuda_k_memcpy_d2d(void* dst, const void* src, size_t n)
{
    return cudaMemcpy(dst, src, n, cudaMemcpyDeviceToDevice) == cudaSuccess ? 0 : -1;
}

extern "C" int cuda_k_memset_zero(void* dst, size_t n)
{
    return cudaMemset(dst, 0, n) == cudaSuccess ? 0 : -1;
}

extern "C" void cuda_k_sync(void)
{
    cudaDeviceSynchronize();
}
