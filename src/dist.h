#ifndef YLLM_DIST_H
#define YLLM_DIST_H

#include <stdint.h>

/* 层流水线分布式拓扑 */
typedef struct {
    int rank;
    int ranks;
    int up_fd;     /* 从 rank-1 收激活 */
    int down_fd;   /* 发激活给 rank+1 */
    int log_fd;    /* 末 rank→rank0 的 top-k logits 通道 */
    uint16_t* x16; /* fp16 激活中转缓冲 */
    uint32_t cap_x;
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
} Dist;

void dist_print_stats(Dist* d, const char* tag);

int dist_init(Dist* d, int rank, int ranks, uint16_t port_base);
int dist_send_x(Dist* d, uint32_t pos, const float* x, uint32_t hidden, int fp16);
int dist_recv_x(Dist* d, uint32_t* pos, float* x, uint32_t hidden, int fp16);
int dist_send_logits(Dist* d, const float* logits, uint32_t vocab, uint32_t topk);
int dist_recv_logits(Dist* d, uint32_t* ids, float* logits, uint32_t topk, float* lse_out);
int dist_send_done(Dist* d);
void dist_close(Dist* d);

#endif
