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
    float* host_w2; /* 第二段 norm 权暂存 */

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

    void* rope_desc_pool;
    void* rope_desc_layout;
    void* rope_ds_q; /* o0 */
    void* rope_ds_k; /* o1 */
    void* rope_pipe_layout;
    void* rope_pipeline;
    void* rope_shader;
    int rope_ready;

    /* residual / bias / store_kv */
    void* add_desc_pool;
    void* add_desc_layout;
    void* add_ds_xy; /* buf_x += buf_y */
    void* add_ds_xo; /* buf_x += buf_o0 */
    void* add_pipe_layout;
    void* add_pipeline;
    void* add_shader;
    void* bias_desc_pool;
    void* bias_desc_layout;
    void* bias_ds_q;
    void* bias_ds_k;
    void* bias_ds_v;
    void* bias_pipe_layout;
    void* bias_pipeline;
    void* bias_shader;
    void* buf_bias;
    void* mem_bias;
    size_t bias_bytes;
    void* skv_desc_pool;
    void* skv_desc_layout;
    void* skv_ds_k;
    void* skv_ds_v;
    void* skv_pipe_layout;
    void* skv_pipeline;
    void* skv_shader;
    int add_ready;
    int bias_ready;
    int skv_ready;
    int block_ready; /* 整层 fused 可用 */

    int x_on_dev; /* buf_x 为权威激活 */

    /* lm_head: Q4_K 常驻偏移(Q6_K 等走 CPU) */
    uint64_t lm_off;
    uint32_t lm_out;
    uint32_t lm_in;
    uint32_t lm_dtype;
    int lm_ready;
    float norm_eps;

    /* 流式权: 大模型按层上传; 亦规避 maxStorageBufferRange */
    uint8_t* host_wq;
    size_t host_wq_bytes;
    size_t layer_wq_max;
    int wq_stream;
    uint32_t stream_layer; /* 当前已上传层, ~0=无 */
    uint64_t stream_base;  /* host_wq 中本层起点, GPU 上相对 off=abs-base */

    int compute_ready;
    uint32_t n_layers;
    uint32_t hidden;
    uint32_t max_in;
    uint32_t max_out;
} VulkanCtx;

#endif
