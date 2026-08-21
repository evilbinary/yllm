/* device_cpu.c — CPU 设备后端: load_weights 为空操作(继续用 host mmap + 堆 KV) */
#include "device.h"
#include "yllm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int cpu_load_weights(Engine* e, char* err, size_t errlen)
{
    (void)err;
    (void)errlen;
    /* host 权重已由 wmap/llf 映射; 设备侧别名 host KV 便于统一读路径 */
    e->w_dev = NULL;
    e->d_kv = e->kv;
    e->weights_ready = 1;
    e->device_mode = DEV_MODE_CPU;
    engine_attach_cpu_fwd(e);
    return 0;
}

static void cpu_free_dev(Engine* e)
{
    e->w_dev = NULL;
    e->d_kv = NULL;
    e->weights_ready = 0;
    e->device_mode = DEV_MODE_CPU;
}

static Device* device_create_cpu(int device_id)
{
    Device* d = (Device*)calloc(1, sizeof(Device));
    if (!d) return NULL;
    d->kind = DEV_CPU;
    d->id = device_id;
    d->load_weights = cpu_load_weights;
    d->free_dev = cpu_free_dev;
    d->prefetch_layer = NULL;
    d->release_layer = NULL;
    return d;
}

int device_kind_parse(const char* s, DeviceKind* out)
{
    if (!s || !out) return -1;
    if (strcmp(s, "cpu") == 0 || strcmp(s, "CPU") == 0) {
        *out = DEV_CPU;
        return 0;
    }
    if (strcmp(s, "cuda") == 0 || strcmp(s, "CUDA") == 0 ||
        strcmp(s, "gpu") == 0 || strcmp(s, "GPU") == 0) {
        *out = DEV_CUDA;
        return 0;
    }
    return -1;
}

int cuda_weight_mode_parse(const char* s, CudaWeightMode* out)
{
    if (!s || !out) return -1;
    if (strcmp(s, "auto") == 0 || strcmp(s, "AUTO") == 0) {
        *out = CUDA_W_AUTO;
        return 0;
    }
    if (strcmp(s, "q4k") == 0 || strcmp(s, "Q4K") == 0 ||
        strcmp(s, "q4") == 0 || strcmp(s, "Q4") == 0) {
        *out = CUDA_W_Q4K;
        return 0;
    }
    if (strcmp(s, "fp16") == 0 || strcmp(s, "FP16") == 0 ||
        strcmp(s, "f16") == 0 || strcmp(s, "F16") == 0) {
        *out = CUDA_W_FP16;
        return 0;
    }
    return -1;
}

Device* device_create(DeviceKind kind, int device_id, char* err, size_t errlen)
{
    if (kind == DEV_CPU)
        return device_create_cpu(device_id);
    if (kind == DEV_CUDA) {
#ifdef YLLM_CUDA
        /* P1: device_cuda.c 提供 device_create_cuda */
        extern Device* device_create_cuda(int device_id, char* err, size_t errlen);
        return device_create_cuda(device_id, err, errlen);
#else
        if (err && errlen)
            snprintf(err, errlen, "CUDA backend not built (rebuild with YLLM_CUDA=1)");
        return NULL;
#endif
    }
    if (err && errlen)
        snprintf(err, errlen, "unknown device kind %d", (int)kind);
    return NULL;
}

void device_destroy(Device* d)
{
    free(d);
}

#ifndef YLLM_CUDA
/* 无 CUDA 构建时的桩, 避免 engine 链接失败 */
const uint8_t* cuda_layer_base(const Engine* e, uint32_t layer)
{
    (void)e;
    (void)layer;
    return NULL;
}
void cuda_after_prefill(Engine* e, uint32_t n_pos)
{
    (void)e;
    (void)n_pos;
}
void cuda_sync_x_to_host(Engine* e)
{
    (void)e;
}
int cuda_embed(Engine* e, uint32_t token)
{
    (void)e;
    (void)token;
    return -1;
}
int cuda_final_norm(Engine* e)
{
    (void)e;
    return -1;
}
int cuda_lm_head(Engine* e)
{
    (void)e;
    return -1;
}
int cuda_prefill(Engine* e, const uint32_t* tokens, int n, int start_pos)
{
    (void)e;
    (void)tokens;
    (void)n;
    (void)start_pos;
    return -1;
}
#endif
