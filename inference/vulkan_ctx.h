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

    void* cmd_pool;
    void* cmd;
    void* fence;

    void* buf_x;
    void* buf_y;
    void* mem_x;
    void* mem_y;
    size_t x_bytes;
    size_t y_bytes;

    void* rms_desc_pool;
    void* rms_desc_layout;
    void* rms_desc_set;
    void* rms_pipe_layout;
    void* rms_pipeline;
    void* rms_shader;
    void* buf_wn;
    void* mem_wn;
    size_t wn_bytes;
    float* host_w;

    void* gemv_desc_pool;
    void* gemv_desc_layout;
    void* gemv_desc_set;
    void* gemv_pipe_layout;
    void* gemv_pipeline;
    void* gemv_shader;
    void* buf_wq;
    void* mem_wq;
    size_t wq_bytes;        /* 常驻 Q4_K 总字节 */
    uint64_t* wq_off;       /* [layer*nslot + slot], UINT64_MAX=无 */
    uint32_t wq_nslot;
    int wq_resident;        /* 1 = 权已上传, gemv 不再 H2D W */
    int gemv_ready;

    int compute_ready;
    uint32_t n_layers;
    uint32_t hidden;
    uint32_t max_in;
    uint32_t max_out;
} VulkanCtx;

#endif
