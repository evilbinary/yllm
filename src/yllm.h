#ifndef YLLM_H
#define YLLM_H

#include "llf.h"

#include <stdint.h>
#include <stddef.h>

typedef struct {
    WMap map;
    LlModel model;
    uint8_t* pstate;
    uint8_t* hot;
    uint8_t* res;           /* mincore 真实驻留位图: 1=该层页缓存已驻留 */
    uint64_t* layer_size;
    uint64_t budget;        /* 字节预算(0=不限) */
    uint64_t resident;      /* 当前估算驻留字节 */
    uint32_t budget_layers; /* 自适应层数预算(内存受限模式) */
    long last_majflt;       /* 上次 getrusage majflt(缺页反馈) */
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
    int* sorted;      /* indices into pieces, dictionary order (for BPE lookup) */
    float* scores;    /* tokenizer.ggml.scores, per-piece BPE priority */
    uint32_t n_scores;
    uint32_t* ml;     /* merges: left piece ids (sorted by (ml,mr) for bisect) */
    uint32_t* mr;     /* merges: right piece ids */
    uint32_t* mid;    /* merges: merged piece id (l+r) */
    uint32_t* mrank;  /* merges: original rank (0 = highest priority) */
    uint32_t n_merges;
    char** mls;       /* merges 原始串(解析期临时, resolve 后释放) */
    char** mrs;
    int byte_level;   /* 1 = tiktoken/GPT-2 byte-level BPE (qwen2), 0 = sentencepiece */
    int32_t byte_ids[256]; /* tiktoken byte -> token id (byte_level 时预计算, -1=无) */
    char* chat_template; /* jinja2 chat template from gguf */
    int add_bos;      /* tokenizer.ggml.add_bos_token */
} Vocab;

/* ---- 平台层 ---- */
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
int wmap_resident(const WMap* m, uint64_t off, uint64_t sz);
const void* ws_layer_ptr(const Ws* ws, uint32_t layer);

float f16_to_f32(uint16_t h);
uint16_t f32_to_f16(float f);
uint16_t bf16_to_f16(uint16_t b);
void f32_to_f16_buf(const float* src, uint16_t* dst, size_t n);
void bf16_to_f16_buf(const uint16_t* src, uint16_t* dst, size_t n);

/* ---- 转换层 ---- */
int convert_safetensors(const char* in_path, const char* out_path, uint32_t max_seq, char* err, size_t errlen);
int convert_gguf(const char* in_path, const char* out_path, const char* vocab_out,
                 uint32_t max_seq, char* err, size_t errlen);
int convert_model(const char* fmt, const char* in, const char* out, const char* vocab_out,
                  uint32_t max_seq, char* err, size_t errlen);
int convert_dummy(const char* out_path, uint32_t blocks, uint32_t hidden, uint32_t heads,
                  uint32_t kv_heads, uint32_t vocab, uint32_t seq, uint32_t seed, char* err, size_t errlen);
int dummy_vocab(const char* out_path, uint32_t vocab, char* err, size_t errlen);

/* ---- tokenizer ---- */
int vocab_load(const char* path, Vocab* v);
void vocab_free(Vocab* v);
int vocab_encode(Vocab* v, const char* text, uint32_t* ids, int max);
int vocab_decode(Vocab* v, const uint32_t* ids, int n, char* out, int max);
int vocab_id(Vocab* v, const char* piece);
/* chat template: render a single-turn user message, returns number of ids */
int vocab_chat_ids(Vocab* v, const char* user_msg, uint32_t* ids, int max, int add_bos);
int vocab_has_template(Vocab* v);
/* ---- 引擎 ---- */
typedef struct {
    Ws ws;
    uint16_t* kv;
    uint32_t kv_dim;
    uint32_t max_seq;
    uint32_t inter;   /* FFN 中间维度(gate/up 输出宽) */
    float* x;
    float* hb;
    float* hb2;
    float* ffn;
    float* att;
    float* logits;
    uint64_t stat_reads;
    uint64_t stat_releases;
    uint64_t stat_faults;
} Engine;

typedef struct {
    uint32_t n_prefill;
    uint32_t n_decode;
    uint64_t prefill_ms;
    uint64_t decode_ms;
} EngineTimings;

int engine_init(Engine* e, const char* model_path, uint64_t budget, int depth, char* err, size_t errlen);
void engine_free(Engine* e);
int engine_forward(Engine* e, uint32_t token, uint32_t pos);
int engine_sample(Engine* e, uint32_t vocab, float temp, float top_p, uint64_t* rng, uint32_t* out);
int engine_generate(Engine* e, const uint32_t* prompt, int nprompt, int ntokens,
                    float temp, float top_p, uint64_t seed, int eos_stop,
                    void (*on_token)(uint32_t id, void* ctx), void* ctx,
                    EngineTimings* timings, char* err, size_t errlen);
uint64_t engine_resident(const Engine* e);

uint64_t ysrand(uint64_t seed);
uint64_t yrng(uint64_t* s);

#endif