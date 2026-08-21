/* vulkan_ctx.h — Vulkan 后端私有状态 */
#ifndef YLLM_VULKAN_CTX_H
#define YLLM_VULKAN_CTX_H

#include <stdint.h>
#include <stddef.h>

typedef struct {
    int device_id;
    int host_shim;
    void* instance;
    void* phys;
    void* device;
    void* queue;
    uint32_t queue_family;

    /* compute 资源(rmsnorm) */
    void* cmd_pool;
    void* cmd;
    void* fence;
    void* desc_pool;
    void* desc_layout;
    void* desc_set;
    void* pipe_layout;
    void* pipeline;
    void* shader;

    void* buf_x;
    void* buf_y;
    void* buf_w;
    void* mem_x;
    void* mem_y;
    void* mem_w;
    size_t buf_bytes;       /* hidden * 4 容量 */

    float* host_w;          /* F16→F32 权 scratch */
    int compute_ready;      /* 1 = rmsnorm pipeline 可用 */
    uint32_t n_layers;
    uint32_t hidden;
} VulkanCtx;

#endif
