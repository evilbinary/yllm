#ifndef YLLM_H
#define YLLM_H

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
    void* base;
    uint64_t size;
    int fd;
    void* hfile;
    void* hmap;
} WMap;

typedef struct {
    WMap map;
    LlModel model;
    uint8_t* pstate;
    uint8_t* hot;
    uint64_t* layer_size;
    uint64_t budget;
    uint64_t resident;
    int depth;
    void* worker;
    void* worker_th;
} Ws;

typedef struct {
    char** pieces;
    int* order;
    int n;
    int unk;
    int bos;
    int eos;
} Vocab;

typedef struct {
    LlfHeader h;
    LlfLayerDir* dir;
    LlfTensorMeta* metas;
    uint32_t* base_idx;
} LlfDraft;

void* ymalloc(size_t n);
void* ycalloc(size_t n, size_t s);
char* ystrdup(const char* s);
uint64_t ynow_ms(void);
void ymsleep(uint32_t ms);
int ythread_create(void* t, void (*fn)(void*), void* arg);
void ythread_join(void* t);
void ymutex_create(void** m);
void ymutex_lock(void* m);
void ymutex_unlock(void* m);
void ymutex_destroy(void* m);
int yfile_size(const char* path, uint64_t* size);
int wmap_open(const char* path, WMap* m);
void wmap_close(WMap* m);
void ws_prefetch(const Ws* ws, uint32_t layer);
void ws_release(const Ws* ws, uint32_t layer);
const void* ws_layer_ptr(const Ws* ws, uint32_t layer);
int llf_read(const WMap* map, LlModel* out);
int llf_check(const char* path, char* err, size_t errlen);

float f16_to_f32(uint16_t h);
uint16_t f32_to_f16(float f);
uint16_t bf16_to_f16(uint16_t b);
void f32_to_f16_buf(const float* src, uint16_t* dst, size_t n);
void bf16_to_f16_buf(const uint16_t* src, uint16_t* dst, size_t n);

int convert_safetensors(const char* in_path, const char* out_path, uint32_t max_seq, char* err, size_t errlen);
int convert_dummy(const char* out_path, uint32_t blocks, uint32_t hidden, uint32_t heads,
                  uint32_t kv_heads, uint32_t vocab, uint32_t seq, uint32_t seed, char* err, size_t errlen);
int dummy_vocab(const char* out_path, uint32_t vocab, char* err, size_t errlen);

int vocab_load(const char* path, Vocab* v);
void vocab_free(Vocab* v);
int vocab_encode(Vocab* v, const char* text, uint32_t* ids, int max);
int vocab_decode(Vocab* v, const uint32_t* ids, int n, char* out, int max);
int vocab_id(Vocab* v, const char* piece);

typedef struct {
    Ws ws;
    uint16_t* kv;
    uint32_t kv_dim;
    uint32_t max_seq;
    float* x;
    float* hb;
    float* hb2;
    float* att;
    float* logits;
    uint64_t stat_reads;
    uint64_t stat_releases;
    uint64_t stat_faults;
} Engine;

int engine_init(Engine* e, const char* model_path, uint64_t budget, int depth, char* err, size_t errlen);
void engine_free(Engine* e);
int engine_forward(Engine* e, uint32_t token, uint32_t pos);
int engine_sample(Engine* e, uint32_t vocab, float temp, float top_p, uint64_t* rng, uint32_t* out);
int engine_generate(Engine* e, const uint32_t* prompt, int nprompt, int ntokens,
                    float temp, float top_p, uint64_t seed,
                    void (*on_token)(uint32_t id, void* ctx), void* ctx, char* err, size_t errlen);
uint64_t engine_resident(const Engine* e);

uint64_t ysrand(uint64_t seed);
uint64_t yrng(uint64_t* s);

#endif
