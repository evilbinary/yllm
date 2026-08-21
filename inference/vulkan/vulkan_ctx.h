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
    void* gemv_desc_set;
    void* gemv_ds0;
    void* gemv_ds1;
    void* gemv_ds2;
    void* gemv_ds_xo; /* x=buf_o2 → y=buf_y (attn 后 O) */
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

    void* swi_desc_pool;
    void* swi_desc_layout;
    void* swi_desc_set;
    void* swi_pipe_layout;
    void* swi_pipeline;
    void* swi_shader;
    int swi_ready;

    /* RoPE / Attn — rope 暂 CPU; attn 用 GPU f32 KV; 可选 fused O */
    void* attn_desc_pool;
    void* attn_desc_layout;
    void* attn_desc_set;
    void* attn_pipe_layout;
    void* attn_pipeline;
    void* attn_shader;
    void* buf_kv;
    void* mem_kv;
    size_t kv_bytes;
    uint32_t kv_slots;
    uint32_t max_seq;
    uint32_t kv_dim;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t n_blocks;
    int attn_ready;
    int attn_o_ready; /* gemv_ds_xo 可用 */

    int compute_ready;
    uint32_t n_layers;
    uint32_t hidden;
    uint32_t max_in;
    uint32_t max_out;
} VulkanCtx;

#endif
