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

    /* 共享 cmd */
    void* cmd_pool;
    void* cmd;
    void* fence;

    /* 激活缓冲(按 max_in / max_out) */
    void* buf_x;
    void* buf_y;
    void* mem_x;
    void* mem_y;
    size_t x_bytes;
    size_t y_bytes;

    /* RMSNorm */
    void* rms_desc_pool;
    void* rms_desc_layout;
    void* rms_desc_set;
    void* rms_pipe_layout;
    void* rms_pipeline;
    void* rms_shader;
    void* buf_wn;
    void* mem_wn;
    size_t wn_bytes;
    float* host_w;          /* F16→F32 权 scratch */

    /* Q4_K gemv */
    void* gemv_desc_pool;
    void* gemv_desc_layout;
    void* gemv_desc_set;
    void* gemv_pipe_layout;
    void* gemv_pipeline;
    void* gemv_shader;
    void* buf_wq;
    void* mem_wq;
    size_t wq_bytes;
    int gemv_ready;

    int compute_ready;      /* rmsnorm 可用 */
    uint32_t n_layers;
    uint32_t hidden;
    uint32_t max_in;
    uint32_t max_out;
} VulkanCtx;

#endif
