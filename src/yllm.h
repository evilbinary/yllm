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
    int* sorted;      /* indices into pieces, dictionary order (for BPE lookup) */
    float* scores;    /* tokenizer.ggml.scores, per-piece BPE priority */
    uint32_t n_scores;
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

/* ---- 引擎 ---- */
typedef struct {
    Ws ws;
    uint16_t* kv;
    uint32_t kv_dim;
    uint32_t max_seq;
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

int engine_init(Engine* e, const char* model_path, uint64_t budget, int depth, char* err, size_t errlen);
void engine_free(Engine* e);
int engine_forward(Engine* e, uint32_t token, uint32_t pos);
int engine_sample(Engine* e, uint32_t vocab, float temp, float top_p, uint64_t* rng, uint32_t* out);
int engine_generate(Engine* e, const uint32_t* prompt, int nprompt, int ntokens,
                    float temp, float top_p, uint64_t seed,
                    void (*on_token)(uint32_t id, void* ctx), void* ctx, char* err, size_t errlen);
uint32_t engine_argmax(const float* logits, uint32_t n);
uint64_t engine_resident(const Engine* e);

uint64_t ysrand(uint64_t seed);
uint64_t yrng(uint64_t* s);

#endif