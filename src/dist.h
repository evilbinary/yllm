#ifndef YLLM_DIST_H
#define YLLM_DIST_H

#include <stdint.h>

/* 层流水线分布式拓扑 */
typedef struct {
    int rank;
    int ranks;
    int up_fd;    /* 从 rank-1 收激活 */
    int down_fd;  /* 发激活给 rank+1 */
    int log_fd;   /* 末 rank→rank0 的 logits 通道(两端各持一端) */
} Dist;

int dist_init(Dist* d, int rank, int ranks, uint16_t port_base);
int dist_send_x(Dist* d, uint32_t pos, const float* x, uint32_t hidden);
int dist_recv(Dist* d, uint32_t* pos, float* x, uint32_t hidden, float* logits, uint32_t vocab);
int dist_send_logits(Dist* d, const float* logits, uint32_t vocab);
int dist_send_done(Dist* d);
void dist_close(Dist* d);

#endif
