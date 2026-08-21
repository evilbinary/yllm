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

    /* 激活: x 输入, y = rmsnorm 输出兼 gemv 输入 */
    void* buf_x;
    void* buf_y;
    void* mem_x;
    void* mem_y;
    /* gemv 多路输出(Q/K/V 或 gate/up/tmp) */
    void* buf_o0;
    void* buf_o1;
    void* buf_o2;
    void* mem_o0;
    void* mem_o1;
    void* mem_o2;
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
    void* gemv_desc_set;    /* x=buf_x y=buf_y — 单次 gemv/自检 */
    void* gemv_ds0;         /* x=buf_y y=buf_o0 */
    void* gemv_ds1;         /* x=buf_y y=buf_o1 */
    void* gemv_ds2;         /* x=buf_y y=buf_o2 */
    void* gemv_pipe_layout;
    void* gemv_pipeline;
    void* gemv_shader;
    void* buf_wq;
    void* mem_wq;
    size_t wq_bytes;
    uint64_t* wq_off;
    uint32_t wq_nslot;
    int wq_resident;
    int gemv_ready;
    int fuse_ready;

    int compute_ready;
    uint32_t n_layers;
    uint32_t hidden;
    uint32_t max_in;
    uint32_t max_out;
} VulkanCtx;

#endif
