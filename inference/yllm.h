#ifndef YLLM_H
#define YLLM_H

#include "llf.h"
#include "device.h"

#include <stdint.h>
#include <stddef.h>

/* 张量级流式: 受限模式下按张量粒度释放(当前层不再整层驻留),
 * lm_head 按词汇行分块计算。默认开启, 编译期 -DYLLM_TENSOR_STREAM=0 关闭。 */
#ifndef YLLM_TENSOR_STREAM
#define YLLM_TENSOR_STREAM 1
#endif

typedef struct {
    WMap map;
    LlModel model;
    uint8_t* pstate;
    uint8_t* res;           /* mincore 真实驻留位图: 1=该层页缓存已驻留 */
    uint64_t* layer_size;
    uint64_t budget;        /* 字节预算(0=不限) */
    uint64_t resident;      /* 当前估算驻留字节 */
    uint32_t budget_layers; /* 层数预算(字节预算按平均层大小折算, 自适应微调) */
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
uint64_t ynow_ns(void);
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
#if YLLM_TENSOR_STREAM
void ws_prefetch_range(const Ws* ws, uint64_t off, uint64_t sz);
void ws_release_aligned(const Ws* ws, uint64_t off, uint64_t sz);
#endif
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
int vocab_chat_ids_multi(Vocab* v, const char* const* roles, const char* const* contents,
                         int n_msgs, uint32_t* ids, int max, int add_bos);
int vocab_has_template(Vocab* v);
/* ---- 引擎 ---- */
typedef struct Engine {
    Ws ws;
    uint16_t* kv;
    uint32_t kv_dim;
    uint32_t max_seq;
    uint32_t inter;   /* FFN 中间维度(gate/up 输出宽) */
    uint32_t arch;    /* 架构(ARCH_LLAMA/ARCH_QWEN/ARCH_QWEN35), engine_init 从 header 读入 */
    /* 层前向分派(engine_init 填一次): 不同架构挂不同实现 */
    int (*fwd_block)(struct Engine* e, uint32_t layer, uint32_t pos);
    int (*fwd_block_batch)(struct Engine* e, uint32_t layer, uint32_t pos_start, uint32_t B);
    uint32_t layer_begin; /* 分布式分片: 本进程层区间 [begin, end) */
    uint32_t layer_end;
    float* x; /* 单 token 路径的主激活缓冲 */
    float* hb;
    float* hb2;
    float* ffn;
    float* att;
    float* logits;
    /* qwen35(混合架构)专用: 每 GDN 层固定大小 O(1) 状态 */
    float* ssm_state;   /* [n_gdn × n_vheads × head_v_dim²] 递归状态 S */
    float* ssm_conv;    /* [n_gdn × (conv_kernel-1) × conv_channels] conv1d 延迟线 */
    float* scratch;     /* GDN/attention 层临时工作区 */
    /* 批量 prefill 工作区(每批 ≤ PB_MAX token) */
    float* pb;      /* [PB_MAX × hidden]  输入/残差 */
    float* pb2;     /* [PB_MAX × hidden]  norm/o 输出 */
    float* pbq;     /* [PB_MAX × hidden]  query */
    float* pbk;     /* [PB_MAX × kv_dim]  key */
    float* pbv;     /* [PB_MAX × kv_dim]  value */
    float* pbg;     /* [PB_MAX × inter]   gate */
    float* pbu;     /* [PB_MAX × inter]   up */
    float* pba;     /* [PB_MAX × n_heads × max_seq] 注意力分数 */
    uint32_t pb_cap;    /* 当前分配的批容量 */
    uint64_t stat_reads;
    uint64_t stat_releases;
    uint64_t stat_faults;
    /* MTP(Multi-Token Prediction)权重槽(在 output 层)。0 = 模型无 MTP。 */
    uint32_t mtp_eh_slot;    /* eh_proj [in→out] */
    uint32_t mtp_enorm_slot; /* embed norm */
    uint32_t mtp_hnorm_slot; /* hidden norm */
    uint32_t mtp_headnorm_slot; /* shared head norm */
    uint32_t mtp_layer;      /* MTP 块所在 llf 层(n_blocks, 0 = 无); 主干到 mtp_layer-1 */
    float* mtp_h;            /* [hidden] 主干最后一层 norm 前 hidden(MTP 输入) */
    float* mtp_logits;       /* [vocab] MTP 预测 logits */
    int mtp_enable;          /* 运行期开关: 1 = speculative decoding */
    int mtp_h_ready;         /* mtp_h 已由最近一次主干前向刷新 */
    /* 设备后端(docs/design-gpu-inference.md): 默认 CPU; CUDA 经 engine_bind_device */
    Device* dev;
    void* d_kv;              /* 设备 KV; CPU 下别名 kv */
    void* w_dev;             /* 设备权重根/层表; CPU 为 NULL */
    int weights_ready;       /* load_weights 成功 */
    DeviceMode device_mode;  /* 实际前向路径; load_weights / bind 时置位 */
    CudaWeightMode cuda_wmode; /* CUDA 线性权: auto|q4k|fp16; bind 前设置 */
} Engine;

typedef struct {
    uint32_t n_prefill;
    uint32_t n_decode;
    uint64_t prefill_ms;
    uint64_t decode_ms;
} EngineTimings;

int engine_init(Engine* e, const char* model_path, uint64_t budget, int depth, char* err, size_t errlen);
void engine_free(Engine* e);
/* 绑定/切换设备并调用 load_weights。engine_init 末尾已绑 DEV_CPU。
 * dist_split_layers 后应再 load_weights(或再次 bind)以匹配本 rank 层段。 */
int engine_bind_device(Engine* e, DeviceKind kind, int device_id, char* err, size_t errlen);
/* 仅重新 load_weights(切层后刷新本段); 无 dev 时返回 -1 */
int engine_load_weights(Engine* e, char* err, size_t errlen);
/* 用显式 layer_base / kv 跑默认块前向(CPU 算子)。
 * CUDA host-shim 的 load_weights 把权拷到 w_dev 后复用此函数校验/过渡。 */
int engine_fwd_block_at(Engine* e, uint32_t layer, uint32_t pos,
                        const uint8_t* layer_base, uint16_t* kv);
/* 恢复 CPU fwd_block 指针(按 arch) */
void engine_attach_cpu_fwd(Engine* e);
int engine_forward(Engine* e, uint32_t token, uint32_t pos);
int engine_forward_range(Engine* e, uint32_t token, int need_embed, uint32_t pos,
                         float* x_out, float* logits_out);
void engine_set_layers(Engine* e, uint32_t begin, uint32_t end);
int engine_sample(Engine* e, uint32_t vocab, float temp, float top_p, uint64_t* rng, uint32_t* out);
/* 批量 matmul(批量 prefill): y[B×out] = x[B×in] · W^T */
void matmul_batch(float* y, const float* x, const uint8_t* w, uint32_t out, uint32_t in,
                  uint32_t dtype, uint32_t B);

/* 批量 prefill: 一次处理 n 个 prompt token(start_pos 起), 结果 logits 为最后 token */
int engine_forward_prefill(Engine* e, const uint32_t* tokens, int n, int start_pos);
/* 分布式批量前向: tokens → 本 rank 层段 → 每 token 激活(仅 PP 首段, 内部 embed) */
int engine_forward_batch_tokens(Engine* e, const uint32_t* tokens, int n, uint32_t pos,
                                float* x_out);
/* 分布式批量前向: 输入激活 → 本 rank 层段 → 输出激活(中段)或最后 token logits(末段) */
int engine_forward_batch_x(Engine* e, const float* xin, int n, uint32_t pos,
                           float* x_out, float* logits_out);

int engine_generate(Engine* e, const uint32_t* prompt, int nprompt, int ntokens,
                    float temp, float top_p, uint64_t seed, int eos_stop,
                    int (*on_token)(uint32_t id, void* ctx), void* ctx,
                    EngineTimings* timings, char* err, size_t errlen);
uint64_t engine_resident(const Engine* e);

uint64_t ysrand(uint64_t seed);
uint64_t yrng(uint64_t* s);

#endif