/* device.h — 推理设备后端(CPU / CUDA / Vulkan)
 *
 * 权重上设备统一入口: Device.load_weights(Engine*)。
 * 详见 docs/design-gpu-inference.md / docs/design-mobile.md。
 */
#ifndef YLLM_DEVICE_H
#define YLLM_DEVICE_H

#include <stddef.h>
#include <stdint.h>

struct Engine;
typedef struct Engine Engine;

typedef enum {
    DEV_CPU = 0,
    DEV_CUDA = 1,
    DEV_VULKAN = 2
} DeviceKind;

/* 实际推理路径(比 DeviceKind 细) */
typedef enum {
    DEV_MODE_CPU = 0,          /* 主机 mmap + CPU 算子 */
    DEV_MODE_CUDA_HOST = 1,    /* --device cuda 但 host-shim */
    DEV_MODE_CUDA = 2,         /* 真 CUDA */
    DEV_MODE_VULKAN_HOST = 3,  /* --device vulkan 但 host-shim(无 loader / 未完成 shader) */
    DEV_MODE_VULKAN = 4        /* 真 Vulkan compute */
} DeviceMode;

/* CUDA 线性权上卡格式(bind 前设置 e->cuda_wmode) */
typedef enum {
    CUDA_W_AUTO = 0,         /* 默认: DT_Q4K 走原生, 其余解到 FP16 */
    CUDA_W_Q4K = 1,          /* 同 AUTO(显式偏好原生 Q4_K) */
    CUDA_W_FP16 = 2          /* 强制全部线性权解量化为 FP16(更快, 更费显存) */
} CudaWeightMode;

typedef struct Device {
    DeviceKind kind;
    int id;             /* CUDA device index; CPU 忽略 */
    void* handle;       /* 后端私有状态(可空) */

    /* 把本 Engine 要用的权准备到设备(常驻或建流式通道)。
     * 范围: [layer_begin, layer_end)。成功后 e->weights_ready = 1。 */
    int (*load_weights)(Engine* e, char* err, size_t errlen);
    /* 释放设备侧资源(权重/设备 KV/scratch); 不释放 Device 结构体本身 */
    void (*free_dev)(Engine* e);
    /* 大模型流式可选; CPU/小模型可为 NULL */
    int (*prefetch_layer)(Engine* e, uint32_t layer);
    void (*release_layer)(Engine* e, uint32_t layer);
} Device;

/* 解析 "cpu" / "cuda" / "vulkan"(大小写不敏感); 未知返回 -1 */
int device_kind_parse(const char* s, DeviceKind* out);
/* 解析 "auto" / "q4k" / "fp16"; 未知返回 -1 */
int cuda_weight_mode_parse(const char* s, CudaWeightMode* out);

/* 创建后端。CUDA 未编译进本二进制时返回 NULL 并写 err。 */
Device* device_create(DeviceKind kind, int device_id, char* err, size_t errlen);

/* 设备权 blob 内一层基址(仅 DEV_CUDA load_weights 后有效; 否则 NULL) */
const uint8_t* cuda_layer_base(const Engine* e, uint32_t layer);

/* prefill(CPU) 之后把 host KV/激活同步到设备 */
void cuda_after_prefill(Engine* e, uint32_t n_pos);
/* 若激活只在设备上, 拉回 e->x(MTP / x_out 等) */
void cuda_sync_x_to_host(Engine* e);
/* host 已写入 e->x / 激活缓冲: 标记 d_x 失效, 下次 fwd 再 H2D(PP 收包后必调) */
void cuda_mark_x_host(Engine* e);
int cuda_embed(Engine* e, uint32_t token);
int cuda_final_norm(Engine* e);
int cuda_lm_head(Engine* e);
int vulkan_final_norm(Engine* e);
int vulkan_lm_head(Engine* e);
void vulkan_after_embed(Engine* e);
void vulkan_sync_x(Engine* e);
/* GPU 批 prefill; 失败返回 -1(调用方回退) */
int cuda_prefill(Engine* e, const uint32_t* tokens, int n, int start_pos);
/* PP 中/末段: host 激活 xin[n×hidden] → 本段 GPU 层 → x_out 或 logits */
int cuda_forward_batch_x(Engine* e, const float* xin, int n, uint32_t pos,
                         float* x_out, float* logits_out);

/* 仅 free(Device*); 设备资源须先 free_dev */
void device_destroy(Device* d);

#endif /* YLLM_DEVICE_H */
