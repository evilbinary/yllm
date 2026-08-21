/* device.h — 推理设备后端(CPU / CUDA)
 *
 * 权重上设备统一入口: Device.load_weights(Engine*)。
 * 详见 docs/design-gpu-inference.md。
 */
#ifndef YLLM_DEVICE_H
#define YLLM_DEVICE_H

#include <stddef.h>
#include <stdint.h>

struct Engine;
typedef struct Engine Engine;

typedef enum {
    DEV_CPU = 0,
    DEV_CUDA = 1
} DeviceKind;

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

/* 解析 "cpu" / "cuda"(大小写不敏感); 未知返回 -1 */
int device_kind_parse(const char* s, DeviceKind* out);

/* 创建后端。CUDA 未编译进本二进制时返回 NULL 并写 err。 */
Device* device_create(DeviceKind kind, int device_id, char* err, size_t errlen);

/* 仅 free(Device*); 设备资源须先 free_dev */
void device_destroy(Device* d);

#endif /* YLLM_DEVICE_H */
