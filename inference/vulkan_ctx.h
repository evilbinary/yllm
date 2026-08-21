/* vulkan_ctx.h — Vulkan 后端私有状态(三端共用; 现多为 host-shim) */
#ifndef YLLM_VULKAN_CTX_H
#define YLLM_VULKAN_CTX_H

#include <stdint.h>
#include <stddef.h>

typedef struct {
    int device_id;
    int host_shim;          /* 1 = 无 Vulkan loader / 未完成 shader, CPU 算 */
    /* 真 Vulkan 字段预留(instance/device/queue/buffers) */
    void* instance;         /* VkInstance */
    void* phys;             /* VkPhysicalDevice */
    void* device;           /* VkDevice */
    void* queue;            /* VkQueue */
    uint32_t queue_family;
    uint32_t n_layers;
    uint32_t hidden;
} VulkanCtx;

#endif /* YLLM_VULKAN_CTX_H */
