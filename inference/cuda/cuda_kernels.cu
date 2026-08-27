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

/* ---- Q4_K: 144B/block, 256 weights; d,min = f16; scales @+4; qs @+16 ---- */
__device__ __forceinline__ void q4k_scale_min(int g, const uint8_t* scp, uint8_t* sc, uint8_t* mn)
{
    if (g < 4) {
        *sc = scp[g] & 63;
        *mn = scp[g + 4] & 63;
    } else {
        *sc = (uint8_t)((scp[g + 4] & 0xF) | ((scp[g - 4] >> 6) << 4));
        *mn = (uint8_t)((scp[g + 4] >> 4) | ((scp[g] >> 6) << 4));
    }
}

__device__ float q4k_dot256(const uint8_t* blk, const float* x)
{
    float d = __half2float(__ushort_as_half(((const uint16_t*)blk)[0]));
    float dmin = __half2float(__ushort_as_half(((const uint16_t*)blk)[1]));
    const uint8_t* scp = blk + 4;
    const uint8_t* qs = blk + 16;
    float acc = 0.0f;
    for (int g = 0; g < 8; g++) {
        uint8_t sc, mn;
        q4k_scale_min(g, scp, &sc, &mn);
        float d1 = d * (float)sc;
        float m1 = dmin * (float)mn;
        const uint8_t* q = qs + (g >> 1) * 32;
        const float* xb = x + g * 32;
        for (int e = 0; e < 32; e++) {
            uint8_t nib = (g & 1) ? (q[e] >> 4) : (q[e] & 0xF);
            acc += xb[e] * (d1 * (float)nib - m1);
        }
    }
    return acc;
}

__global__ void k_gemv_q4k(float* y, const uint8_t* w, const float* x, uint32_t out, uint32_t in)
{
    uint32_t o = blockIdx.x;
    if (o >= out) return;
    uint32_t nb = in / 256;
    const uint8_t* row = w + (size_t)o * nb * 144;
    float acc = 0.0f;
    for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x)
        acc += q4k_dot256(row + (size_t)b * 144, x + (size_t)b * 256);
    __shared__ float sh[128];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int s = (int)blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[o] = sh[0];
}

extern "C" int cuda_k_gemv_q4k(float* y, const uint8_t* w, const float* x,
                               uint32_t out, uint32_t in, char* err, size_t errlen)
{
    if (!y || !w || !x || out == 0 || in == 0 || (in % 256) != 0) {
        if (err && errlen) snprintf(err, errlen, "gemv_q4k bad args");
        return -1;
    }
    uint32_t nb = in / 256;
    uint32_t t = nb < 128u ? nb : 128u;
    if (t < 1) t = 1;
    /* blockDim 须为 2 的幂以便归约 */
    uint32_t p = 1;
    while (p < t) p <<= 1;
    if (p > 128) p = 128;
    k_gemv_q4k<<<out, p>>>(y, w, x, out, in);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        if (err && errlen) snprintf(err, errlen, "gemv_q4k: %s", cudaGetErrorString(e));
        return -1;
    }
    return 0;
}

__global__ void k_gemm_q4k(float* y, const uint8_t* w, const float* x,
                           uint32_t out, uint32_t in, uint32_t B)
{
    uint32_t o = blockIdx.x;
    uint32_t btok = blockIdx.y;
    if (o >= out || btok >= B) return;
    uint32_t nb = in / 256;
    const uint8_t* row = w + (size_t)o * nb * 144;
    const float* xb = x + (size_t)btok * in;
    float acc = 0.0f;
    for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x)
        acc += q4k_dot256(row + (size_t)b * 144, xb + (size_t)b * 256);
    __shared__ float sh[128];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int s = (int)blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        y[(size_t)btok * out + o] = sh[0];
}

extern "C" int cuda_k_gemm_q4k(float* y, const uint8_t* w, const float* x,
                               uint32_t out, uint32_t in, uint32_t B, char* err, size_t errlen)
{
    if (!y || !w || !x || out == 0 || in == 0 || B == 0 || (in % 256) != 0) {
        if (err && errlen) snprintf(err, errlen, "gemm_q4k bad args");
        return -1;
    }
    if (B == 1)
        return cuda_k_gemv_q4k(y, w, x, out, in, err, errlen);
    uint32_t nb = in / 256;
    uint32_t t = nb < 128u ? nb : 128u;
    if (t < 1) t = 1;
    uint32_t p = 1;
    while (p < t) p <<= 1;
    if (p > 128) p = 128;
    dim3 grid(out, B);
    k_gemm_q4k<<<grid, p>>>(y, w, x, out, in, B);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        if (err && errlen) snprintf(err, errlen, "gemm_q4k: %s", cudaGetErrorString(e));
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

__device__ __forceinline__ float gelu_approx(float x)
{
    float c = 0.7978845608f;
    float inner = c * (x + 0.044715f * x * x * x);
    float t = tanhf(inner);
    return 0.5f * x * (1.0f + t);
}

__global__ void k_geglu(float* y, const float* gate, const float* up, uint32_t n)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = gelu_approx(gate[i]) * up[i];
}

extern "C" void cuda_k_geglu(float* y, const float* gate, const float* up, uint32_t n)
{
    if (n == 0) return;
    k_geglu<<<(n + 255) / 256, 256>>>(y, gate, up, n);
}

__global__ void k_gelu(float* y, uint32_t n)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = gelu_approx(y[i]);
}

extern "C" void cuda_k_gelu(float* y, uint32_t n)
{
    if (n == 0) return;
    k_gelu<<<(n + 255) / 256, 256>>>(y, n);
}

__global__ void k_mul(float* y, const float* a, const float* b, uint32_t n)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] * b[i];
}

extern "C" void cuda_k_mul(float* y, const float* a, const float* b, uint32_t n)
{
    if (n == 0) return;
    k_mul<<<(n + 255) / 256, 256>>>(y, a, b, n);
}

__global__ void k_scale(float* y, float s, uint32_t n)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] *= s;
}

extern "C" void cuda_k_scale(float* y, float s, uint32_t n)
{
    if (n == 0) return;
    k_scale<<<(n + 255) / 256, 256>>>(y, s, n);
}

__global__ void k_scale_dev(float* y, const float* s, uint32_t n)
{
    float sc = s[0];
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] *= sc;
}

extern "C" void cuda_k_scale_dev(float* y, const float* s, uint32_t n)
{
    if (n == 0 || !s) return;
    k_scale_dev<<<(n + 255) / 256, 256>>>(y, s, n);
}

__global__ void k_rmsnorm_unit(float* y, const float* x, uint32_t n, float eps)
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
        y[i] = x[i] * inv;
}

extern "C" void cuda_k_rmsnorm_unit(float* y, const float* x, uint32_t n, float eps)
{
    k_rmsnorm_unit<<<1, 256>>>(y, x, n, eps);
}

__global__ void k_rmsnorm_unit_batch(float* y, const float* x, uint32_t n, float eps)
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
        yb[i] = xb[i] * inv;
}

extern "C" void cuda_k_rmsnorm_unit_batch(float* y, const float* x, uint32_t n, float eps, uint32_t B)
{
    if (B == 0 || n == 0) return;
    k_rmsnorm_unit_batch<<<B, 256>>>(y, x, n, eps);
}

__global__ void k_rope_neox_if_heads(float* v, uint32_t n_heads, uint32_t head_dim,
                                    uint32_t pos, const float* inv_freq)
{
    uint32_t hh = blockIdx.x;
    uint32_t j = threadIdx.x;
    uint32_t half = head_dim / 2;
    if (hh >= n_heads || j >= half || pos == 0 || !inv_freq) return;
    float* vh = v + (size_t)hh * head_dim;
    float ang = inv_freq[j] * (float)pos;
    float c = cosf(ang), s = sinf(ang);
    float a = vh[j], b = vh[j + half];
    vh[j] = a * c - b * s;
    vh[j + half] = a * s + b * c;
}

extern "C" void cuda_k_rope_neox_if_heads(float* v, uint32_t n_heads, uint32_t head_dim,
                                         uint32_t pos, const float* inv_freq)
{
    if (pos == 0 || !inv_freq || n_heads == 0 || head_dim < 2) return;
    k_rope_neox_if_heads<<<n_heads, head_dim / 2>>>(v, n_heads, head_dim, pos, inv_freq);
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

__device__ __forceinline__ float attn_block_sum(float v)
{
    __shared__ float sh[1024];
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = (int)blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    return sh[0];
}

/* Flash-style decode: online softmax, Q 常驻 shared, 无 O(seq) score 缓冲 */
__global__ void k_attn_flash_decode(float* att_out, const float* q,
                                    const uint16_t* kcache, const uint16_t* vcache,
                                    uint32_t s0, uint32_t pos, uint32_t n_heads, uint32_t n_kv_heads,
                                    uint32_t head_dim, uint32_t kv_dim, float scale)
{
    uint32_t hh = blockIdx.x;
    if (hh >= n_heads) return;
    uint32_t tid = threadIdx.x;
    uint32_t kv_head = hh * n_kv_heads / n_heads;
    extern __shared__ float q_s[];
    if (tid < head_dim)
        q_s[tid] = q[(size_t)hh * head_dim + tid];
    __syncthreads();

    float m = -1.0e30f;
    float l = 0.0f;
    float o = 0.0f;
    if (s0 > pos) {
        if (tid < head_dim)
            att_out[(size_t)hh * head_dim + tid] = 0.0f;
        return;
    }

    for (uint32_t s = s0; s <= pos; s++) {
        const uint16_t* kh = kcache + (size_t)s * kv_dim + (size_t)kv_head * head_dim;
        float partial = 0.0f;
        if (tid < head_dim)
            partial = q_s[tid] * __half2float(__ushort_as_half(kh[tid]));
        float score = attn_block_sum(partial) * scale;

        float m_new = fmaxf(m, score);
        float alpha = expf(m - m_new);
        float p = expf(score - m_new);
        float l_new = alpha * l + p;
        if (tid < head_dim) {
            const uint16_t* vh = vcache + (size_t)s * kv_dim + (size_t)kv_head * head_dim;
            o = alpha * o + p * __half2float(__ushort_as_half(vh[tid]));
        }
        m = m_new;
        l = l_new;
    }
    if (tid < head_dim)
        att_out[(size_t)hh * head_dim + tid] = o / l;
}

static uint32_t attn_threads(uint32_t head_dim)
{
    uint32_t t = 32;
    while (t < head_dim && t < 1024u) t <<= 1;
    return t;
}

extern "C" void cuda_k_attn_decode(float* att_out,
                                   const float* q, const uint16_t* kcache, const uint16_t* vcache,
                                   uint32_t pos, uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
                                   uint32_t kv_dim)
{
    cuda_k_attn_decode_win(att_out, q, kcache, vcache, 0, pos, n_heads, n_kv_heads,
                           head_dim, kv_dim, 1.0f / sqrtf((float)head_dim));
}

extern "C" void cuda_k_attn_decode_win(float* att_out,
                                       const float* q, const uint16_t* kcache, const uint16_t* vcache,
                                       uint32_t s0, uint32_t pos, uint32_t n_heads, uint32_t n_kv_heads,
                                       uint32_t head_dim, uint32_t kv_dim, float scale)
{
    if (head_dim == 0 || head_dim > 1024 || n_heads == 0) return;
    uint32_t t = attn_threads(head_dim);
    k_attn_flash_decode<<<n_heads, t, head_dim * sizeof(float)>>>(
        att_out, q, kcache, vcache, s0, pos, n_heads, n_kv_heads, head_dim, kv_dim, scale);
}

__global__ void k_attn_flash_prefill(float* att_out, const float* q,
                                     const uint16_t* kcache, const uint16_t* vcache,
                                     uint32_t pos_start, uint32_t B, uint32_t n_heads, uint32_t n_kv_heads,
                                     uint32_t head_dim, uint32_t kv_dim, uint32_t q_stride)
{
    uint32_t hh = blockIdx.x;
    uint32_t bb = blockIdx.y;
    if (hh >= n_heads || bb >= B) return;
    uint32_t tid = threadIdx.x;
    uint32_t pos = pos_start + bb;
    uint32_t kv_head = hh * n_kv_heads / n_heads;
    extern __shared__ float q_s[];
    if (tid < head_dim)
        q_s[tid] = q[(size_t)bb * q_stride + (size_t)hh * head_dim + tid];
    __syncthreads();

    float inv_d = rsqrtf((float)head_dim);
    float m = -1.0e30f;
    float l = 0.0f;
    float o = 0.0f;

    for (uint32_t s = 0; s <= pos; s++) {
        const uint16_t* kh = kcache + (size_t)s * kv_dim + (size_t)kv_head * head_dim;
        float partial = 0.0f;
        if (tid < head_dim)
            partial = q_s[tid] * __half2float(__ushort_as_half(kh[tid]));
        float score = attn_block_sum(partial) * inv_d;

        float m_new = fmaxf(m, score);
        float alpha = expf(m - m_new);
        float p = expf(score - m_new);
        float l_new = alpha * l + p;
        if (tid < head_dim) {
            const uint16_t* vh = vcache + (size_t)s * kv_dim + (size_t)kv_head * head_dim;
            o = alpha * o + p * __half2float(__ushort_as_half(vh[tid]));
        }
        m = m_new;
        l = l_new;
    }
    if (tid < head_dim)
        att_out[(size_t)bb * q_stride + (size_t)hh * head_dim + tid] = o / l;
}

extern "C" void cuda_k_attn_prefill(float* att_out,
                                    const float* q, const uint16_t* kcache, const uint16_t* vcache,
                                    uint32_t pos_start, uint32_t B, uint32_t n_heads, uint32_t n_kv_heads,
                                    uint32_t head_dim, uint32_t kv_dim, uint32_t q_stride)
{
    if (head_dim == 0 || head_dim > 1024 || n_heads == 0 || B == 0) return;
    uint32_t t = attn_threads(head_dim);
    dim3 grid(n_heads, B);
    k_attn_flash_prefill<<<grid, t, head_dim * sizeof(float)>>>(
        att_out, q, kcache, vcache, pos_start, B, n_heads, n_kv_heads, head_dim, kv_dim, q_stride);
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

__device__ void q4k_dequant256(float* y, const uint8_t* blk)
{
    float d = __half2float(__ushort_as_half(((const uint16_t*)blk)[0]));
    float dmin = __half2float(__ushort_as_half(((const uint16_t*)blk)[1]));
    const uint8_t* scp = blk + 4;
    const uint8_t* qs = blk + 16;
    for (int g = 0; g < 8; g++) {
        uint8_t sc, mn;
        q4k_scale_min(g, scp, &sc, &mn);
        float d1 = d * (float)sc;
        float m1 = dmin * (float)mn;
        const uint8_t* q = qs + (g >> 1) * 32;
        for (int e = 0; e < 32; e++) {
            uint8_t nib = (g & 1) ? (q[e] >> 4) : (q[e] & 0xF);
            y[g * 32 + e] = d1 * (float)nib - m1;
        }
    }
}

__global__ void k_embed_q4k(float* y, const uint8_t* table, uint32_t token, uint32_t hidden)
{
    uint32_t nb = hidden / 256;
    const uint8_t* row = table + (size_t)token * nb * 144;
    for (uint32_t b = blockIdx.x; b < nb; b += gridDim.x)
        q4k_dequant256(y + (size_t)b * 256, row + (size_t)b * 144);
}

extern "C" void cuda_k_embed_q4k(float* y, const uint8_t* table, uint32_t token, uint32_t hidden)
{
    uint32_t nb = hidden / 256;
    if (nb == 0) return;
    k_embed_q4k<<<nb, 1>>>(y, table, token, hidden);
}

__global__ void k_embed_q4k_batch(float* y, const uint8_t* table, const uint32_t* tokens,
                                  uint32_t B, uint32_t hidden)
{
    uint32_t btok = blockIdx.y;
    uint32_t blk = blockIdx.x;
    uint32_t nb = hidden / 256;
    if (btok >= B || blk >= nb) return;
    uint32_t token = tokens[btok];
    const uint8_t* row = table + (size_t)token * nb * 144;
    q4k_dequant256(y + (size_t)btok * hidden + (size_t)blk * 256, row + (size_t)blk * 144);
}

extern "C" void cuda_k_embed_q4k_batch(float* y, const uint8_t* table, const uint32_t* tokens_dev,
                                       uint32_t B, uint32_t hidden)
{
    uint32_t nb = hidden / 256;
    if (nb == 0 || B == 0) return;
    dim3 grid(nb, B);
    k_embed_q4k_batch<<<grid, 1>>>(y, table, tokens_dev, B, hidden);
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
