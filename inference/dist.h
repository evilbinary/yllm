#ifndef YLLM_DIST_H
#define YLLM_DIST_H

#include "yllm.h"
#include <stdint.h>

/* 生成 token 输出回调(解码并打印到 stdout) */
typedef int (*dist_token_cb)(uint32_t id, void* ctx);

/* 层流水线分布式拓扑 */
typedef struct {
    int rank;
    int ranks;
    int up_fd;     /* 从 rank-1 收激活 */
    int down_fd;   /* 发激活给 rank+1 */
    int log_fd;    /* 末 rank→rank0 的 top-k logits 通道 */
    uint16_t* x16; /* fp16 激活中转缓冲 */
    uint32_t cap_x;
    const volatile int* quit;   /* 非空时 worker 空闲等待周期检查(进程退出信号) */
    float lse;     /* 全量 log-sum-exp(logits 归一化常数) */
    uint32_t* heap; /* top-k 选择堆 */
    uint32_t cap_k;
    uint64_t n_x_sent;    /* 发送 X 帧次数 */
    uint64_t n_x_recv;    /* 接收 X 帧次数 */
    uint64_t n_log_sent;  /* 发送 logits 帧次数 */
    uint64_t n_log_recv;  /* 接收 logits 帧次数 */
    uint64_t bytes_sent;  /* 发送合计字节(payload) */
    uint64_t bytes_recv;  /* 接收合计字节(payload) */
    uint64_t nanos_wait_send;  /* 网络阻塞(send) 纳秒 */
    uint64_t nanos_wait_recv;  /* 网络阻塞(recv) 纳秒 */
    double elapsed_ms;        /* 运行总时长(ms), 供平均带宽计算 */
} Dist;

void dist_print_stats(Dist* d, const char* tag);

int dist_init(Dist* d, int rank, int ranks, uint16_t port_base, const char* const* addrs,
              const volatile int* quit);
int dist_send_x(Dist* d, uint32_t pos, const float* x, uint32_t hidden, int fp16);
int dist_recv_x(Dist* d, uint32_t* pos, float* x, uint32_t hidden, int fp16);
int dist_send_xb(Dist* d, uint32_t pos, const float* x, uint32_t count,
                 uint32_t hidden, int fp16);
int dist_recv_xb(Dist* d, uint32_t* pos, float* x, uint32_t cap_count,
                 uint32_t hidden, int fp16);
int dist_send_logits(Dist* d, const float* logits, uint32_t vocab, uint32_t topk);
int dist_recv_logits(Dist* d, uint32_t* ids, float* logits, uint32_t topk, float* lse_out);
int dist_send_done(Dist* d);
void dist_close(Dist* d);

/* 跨机分布式: 传 addrs(逗号分隔节点 IP, 长度=ranks, 第 i 个是 rank i 的地址)可为空,
 * 为空时用 127.0.0.1 退化为单机多进程测试。 */
/* 会话模式参数(PP 各段共享):
 *   master: key/pos = 会话续接点, 结束时 pos = 结束位置
 *   worker: my_pos = 本段已有 kv 位置(校验用), 结束时 my_pos = 本段 kv 已推进位置
 *   cache_dir = kv 落盘目录(空 = 纯内存) */
typedef struct {
    char key[64];
    uint32_t pos;
    uint32_t my_pos;
    const char* cache_dir;
    char last_key[64];          /* worker: 上次处理的会话 key(变化时重置本段 kv 并恢复) */
    const volatile int* quit;   /* 非空时 worker 空闲等待周期检查(进程退出信号) */
} DistSess;

int dist_gen(Engine* e, Vocab* v, const uint32_t* ids, int nprompt,
             int ntokens, float temp, float top_p, uint64_t seed,
             int rank, int ranks, int port_base, const char* addrs, int dist_fp16,
             uint64_t t0, dist_token_cb emit, void* ctx,
             DistSess* sess);   /* NULL = 非会话模式 */
int dist_send_sess(Dist* d, const char* key, uint32_t pos);
int dist_recv_sess(Dist* d, char* key, uint32_t* pos);

/* 按字节均衡切层(末 rank 含 norm+head, 故少分块), 并设置本 rank 的层范围。
 * 返回 0 成功, -1 失败(ranks > blocks)。 */
int dist_split_layers(Engine* e, int rank, int ranks);

#endif
