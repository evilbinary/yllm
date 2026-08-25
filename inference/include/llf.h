#ifndef YLLM_LLF_H
#define YLLM_LLF_H

#include <stdint.h>
#include <stddef.h>

#define YLLM_MAGIC "YLLMLLF1"
#define YLLM_VERSION 6   /* gemma4 PLE 进 llf; 旧 gemma4 文件必须重转 */
#define LLF_HEADER_SIZE 128
#define LLF_DIR_ENTRY_SIZE 32
#define LLF_TENSOR_META_SIZE 64
#define LLF_ALIGN 4096

#define DT_F16 0
#define DT_F32 1
#define DT_BF16 2
#define DT_Q4K 3
#define DT_Q6K 4
#define DT_IQ4XS 5
#define DT_Q5K 6
#define DT_W4B64 7   /* int4 block-64, 行主序; 见 matvec w4b64_* */

/* 编译期: 1=gguf/llf 转换默认把线性 Q4_K 打成 W4B64; make YLLM_W4=0 关闭 */
#ifndef YLLM_W4
#define YLLM_W4 1
#endif

#define W4B64_BLK 64
#define W4B64_BLK_BYTES 34  /* f16 scale + 32B nibbles */

#define ARCH_LLAMA 0
#define ARCH_QWEN 1
#define ARCH_QWEN35 2
#define ARCH_GEMMA4 3

#define SLOT_EMBED 0
#define SLOT_NORM1 0
#define SLOT_Q 1
#define SLOT_K 2
#define SLOT_V 3
#define SLOT_O 4
#define SLOT_NORM2 5
#define SLOT_GATE 6
#define SLOT_UP 7
#define SLOT_DOWN 8
#define SLOT_QBIAS 9
#define SLOT_KBIAS 10
#define SLOT_VBIAS 11
#define SLOT_QNORM 12
#define SLOT_KNORM 13
#define SLOT_QKV 14         /* GDN attn_qkv [in, 2*hidden] */
#define SLOT_GATE_ATTN 15   /* GDN attn_gate [in, hidden] */
#define SLOT_QGATE 16       /* attention 层 attn_q 的 gate 半部(未用, 保留语义) */
#define SLOT_SSM_CONV1D 17  /* GDN conv1d [conv_kernel, conv_channels] */
#define SLOT_SSM_A 18       /* GDN ssm_a [n_vheads] */
#define SLOT_SSM_DT 19      /* GDN ssm_dt.bias [n_vheads] */
#define SLOT_SSM_ALPHA 20   /* GDN ssm_alpha [in, n_vheads] */
#define SLOT_SSM_BETA 21    /* GDN ssm_beta [in, n_vheads] */
#define SLOT_SSM_NORM 22    /* GDN ssm_norm [head_v_dim] */
#define SLOT_SSM_OUT 23     /* GDN ssm_out [hidden, in] */
#define SLOT_NORM3 24       /* gemma4 post_attention_norm [hidden] */
#define SLOT_NORM4 25       /* gemma4 post_feedforward_norm [hidden] */
#define SLOT_LAYER_SCALE 26 /* gemma4 layer_output_scale scalar [1] F32 */
/* gemma4 per-layer embedding: 复用 GDN 空槽(与 qwen35 不共存) */
#define SLOT_PLE_GATE SLOT_QKV       /* blk.N.inp_gate [hidden, n_ple] */
#define SLOT_PLE_PROJ SLOT_GATE_ATTN /* blk.N.proj [n_ple, hidden] */
#define SLOT_PLE_POST SLOT_QGATE     /* blk.N.post_norm [hidden] */
#define SLOT_PLE_TOK  1              /* embed 层 per_layer_token_embd */
#define SLOT_PLE_MPROJ 2             /* embed 层 per_layer_model_proj */
#define SLOT_PLE_PNORM 3             /* embed 层 per_layer_proj_norm */
#define SLOT_ROPE_FREQS 4            /* embed 层: gemma4 rope_freqs [head_dim/2] (块层同号是 SLOT_O) */
#define BLOCK_TENSORS 27
/* MTP(Multi-Token Prediction)槽: 存 output(lm_head)层的高槽位 27..30,
 * 与主 transformer 块共用 BLOCK_TENSORS 上限之外; llf 层目录 n_tensors 需容纳。
 * 布局(见 convert.c): output 层 = blocks+2, 槽 27=eh_proj 28=enorm 29=hnorm 30=shared_head_norm */
#define SLOT_MTP_EH 27
#define SLOT_MTP_ENORM 28
#define SLOT_MTP_HNORM 29
#define SLOT_MTP_HEAD_NORM 30
#define BLOCK_TENSORS_MTP 31

#pragma pack(push, 1)
typedef struct {
    uint8_t magic[8];
    uint32_t version;
    uint32_t arch;
    uint64_t file_size;
    uint32_t n_blocks;
    uint32_t vocab;
    uint32_t hidden;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t max_seq;
    uint32_t dtype;
    uint32_t norm_eps_bits;
    uint32_t rope_theta_bits;
    /* 通用扩展入口: 非 common 的架构专用字段不要直接放在公共头里 */
    uint64_t ext_ptr;             /* 预留: 指向扩展区(当前版本保留为 0) */
    uint8_t reserved[44];         /* 预留扩展数据 */
} LlfHeader;

typedef struct {
    uint64_t offset;
    uint64_t size;
    uint32_t n_tensors;
    uint32_t reserved;
    uint64_t reserved2;
} LlfLayerDir;

typedef struct {
    char name[24];
    uint32_t dtype;
    uint32_t ndim;
    uint32_t shape[4];
    uint64_t offset;
    uint64_t size;
} LlfTensorMeta;
#pragma pack(pop)

typedef struct {
    LlfHeader h;
    uint32_t n_layers;
    LlfLayerDir* dir;
    LlfTensorMeta* metas;
    uint32_t* base_idx;
} LlModel;

typedef struct {
    LlfHeader h;
    LlfLayerDir* dir;
    LlfTensorMeta* metas;
    uint32_t* base_idx;
} LlfDraft;

/* 内存映射文件抽象(实现见 platform.c) */
typedef struct {
    void* base;
    uint64_t size;
    int fd;
    void* hfile;
    void* hmap;
} WMap;

int llf_read(const WMap* map, LlModel* out);
int llf_check(const char* path, char* err, size_t errlen);
const char* llf_dtype_name(uint32_t dtype);

#endif
