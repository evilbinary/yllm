#ifndef YLLM_LLF_H
#define YLLM_LLF_H

#include <stdint.h>
#include <stddef.h>

#define YLLM_MAGIC "YLLMLLF1"
#define YLLM_VERSION 1
#define LLF_HEADER_SIZE 128
#define LLF_DIR_ENTRY_SIZE 32
#define LLF_TENSOR_META_SIZE 64
#define LLF_ALIGN 4096

#define DT_F16 0
#define DT_F32 1
#define DT_BF16 2
#define DT_Q4K 3
#define DT_Q6K 4

#define ARCH_LLAMA 0
#define ARCH_QWEN 1

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
#define BLOCK_TENSORS 9

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
    uint8_t reserved[52];
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
