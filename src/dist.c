/* dist.c — 层流水线分布式传输(TCP)
 *
 * 拓扑: rank i 把激活发给 rank i+1; 末 rank 把 top-k logits 发回 rank 0。
 * 帧格式: [4B type][4B len][payload]
 *   type 1 = X(激活: pos(4B) + hidden 个 fp16)
 *   type 2 = LOGITS_K(top-k: k(4B) + k × [id(4B) + logit fp16(2B)])
 *   type 3 = DONE(终止)
 */
#include "dist.h"
#include "yllm.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#ifndef _WIN32
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <time.h>
#include <math.h>
#endif

#define DTYPE_X 1
#define DTYPE_LOGITS_K 2
#define DTYPE_DONE 3

static int sock_listen(uint16_t port)
{
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(port);
    if (bind(fd, (struct sockaddr*)&addr, sizeof(addr)) != 0) { close(fd); return -1; }
    if (listen(fd, 8) != 0) { close(fd); return -1; }
    return fd;
}

static int sock_connect(uint16_t port)
{
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(port);
    int attempt;
    for (attempt = 0; attempt < 50; attempt++) {
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) return -1;
        if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
            int one = 1;
            setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
            return fd;
        }
        close(fd);
        /* 对端可能尚未 listen(启动竞态), 每 200ms 重试 */
        struct timespec ts;
        ts.tv_sec = 0;
        ts.tv_nsec = 200 * 1000 * 1000;
        nanosleep(&ts, NULL);
    }
    return -1;
}

static int xio(int fd, const void* buf, size_t n)
{
    const char* p = (const char*)buf;
    while (n > 0) {
        ssize_t r = (ssize_t)send(fd, p, n, 0);
        if (r <= 0) return -1;
        p += r;
        n -= (size_t)r;
    }
    return 0;
}
static int xrecv(int fd, void* buf, size_t n)
{
    char* p = (char*)buf;
    while (n > 0) {
        ssize_t r = (ssize_t)recv(fd, p, n, 0);
        if (r <= 0) return -1;
        p += r;
        n -= (size_t)r;
    }
    return 0;
}

/* 建立连接: 所有进程先 bind+listen, 再按拓扑 connect */
int dist_init(Dist* d, int rank, int ranks, uint16_t port_base)
{
    memset(d, 0, sizeof(*d));
    d->rank = rank;
    d->ranks = ranks;
    d->up_fd = -1;
    d->down_fd = -1;
    d->log_fd = -1;
    int listen_fd = sock_listen((uint16_t)(port_base + (uint16_t)rank));
    if (listen_fd < 0) { fprintf(stderr, "dist: rank %d cannot listen on port %u\n", rank, port_base + rank); return -1; }

    if (rank < ranks - 1) {
        d->down_fd = sock_connect((uint16_t)(port_base + (uint16_t)(rank + 1)));
        if (d->down_fd < 0) { fprintf(stderr, "dist: rank %d cannot connect to %u\n", rank, port_base + rank + 1); return -1; }
    }
    if (rank == 0) {
        d->log_fd = (int)accept(listen_fd, NULL, NULL);
    } else {
        d->up_fd = (int)accept(listen_fd, NULL, NULL);
        if (rank == ranks - 1) {
            d->log_fd = sock_connect(port_base);
            if (d->log_fd < 0) { fprintf(stderr, "dist: rank %d cannot connect back to rank 0\n", rank); return -1; }
        }
    }
    close(listen_fd);
    return 0;
}

/* 激活传输: fp16 减半带宽, 但引入量化误差(可能翻转 argmax); fp32 与单机逐位一致 */
int dist_send_x(Dist* d, uint32_t pos, const float* x, uint32_t hidden, int fp16)
{
    uint8_t hdr[8];
    uint32_t len = 4 + hidden * (fp16 ? 2 : 4);
    uint32_t type = DTYPE_X;
    memcpy(hdr, &type, 4);
    memcpy(hdr + 4, &len, 4);
    if (xio(d->down_fd, hdr, 8) != 0) return -1;
    if (xio(d->down_fd, &pos, 4) != 0) return -1;
    if (fp16) {
        if (d->x16 == NULL || d->cap_x < hidden) {
            free(d->x16);
            d->x16 = (uint16_t*)ymalloc((size_t)hidden * 2);
            d->cap_x = hidden;
        }
        uint32_t i;
        for (i = 0; i < hidden; i++) d->x16[i] = f32_to_f16(x[i]);
        return xio(d->down_fd, d->x16, (size_t)hidden * 2);
    }
    return xio(d->down_fd, x, (size_t)hidden * 4);
}

/* 返回帧类型: 1=X(pos/x 已填充), 3=DONE, -1=错误 */
int dist_recv_x(Dist* d, uint32_t* pos, float* x, uint32_t hidden, int fp16)
{
    int fd = d->rank == 0 ? d->log_fd : d->up_fd;
    uint8_t hdr[8];
    if (xrecv(fd, hdr, 8) != 0) return -1;
    uint32_t type, len;
    memcpy(&type, hdr, 4);
    memcpy(&len, hdr + 4, 4);
    if (type == DTYPE_DONE) return DTYPE_DONE;
    if (type == DTYPE_X) {
        if (len < 4) return -1;
        if (xrecv(fd, pos, 4) != 0) return -1;
        if (fp16) {
            if (d->x16 == NULL || d->cap_x < hidden) {
                free(d->x16);
                d->x16 = (uint16_t*)ymalloc((size_t)hidden * 2);
                d->cap_x = hidden;
            }
            if (xrecv(fd, d->x16, (size_t)hidden * 2) != 0) return -1;
            uint32_t i;
            for (i = 0; i < hidden; i++) x[i] = f16_to_f32(d->x16[i]);
        } else {
            if (xrecv(fd, x, (size_t)hidden * 4) != 0) return -1;
        }
        return DTYPE_X;
    }
    return -1;
}

/* logits 全量传输(保证采样与单机逐位一致) */
int dist_send_logits(Dist* d, const float* logits, uint32_t vocab, uint32_t topk)
{
    (void)topk;
    uint8_t hdr[8];
    uint32_t len = vocab * 4;
    uint32_t type = DTYPE_LOGITS_K;
    memcpy(hdr, &type, 4);
    memcpy(hdr + 4, &len, 4);
    if (xio(d->log_fd, hdr, 8) != 0) return -1;
    return xio(d->log_fd, logits, (size_t)vocab * 4);
}

/* 接收全量 logits */
int dist_recv_logits(Dist* d, uint32_t* ids, float* logits, uint32_t topk, float* lse_out)
{
    int fd = d->log_fd;
    uint8_t hdr[8];
    if (xrecv(fd, hdr, 8) != 0) return -1;
    uint32_t type, len;
    memcpy(&type, hdr, 4);
    memcpy(&len, hdr + 4, 4);
    if (type != DTYPE_LOGITS_K) return -1;
    uint32_t n = len / 4;
    if (n > topk) n = topk;
    if (xrecv(fd, logits, (size_t)n * 4) != 0) return -1;
    if (lse_out) *lse_out = 0.0f;
    (void)ids;
    return (int)n;
}

int dist_send_done(Dist* d)
{
    uint8_t hdr[8];
    uint32_t type = DTYPE_DONE;
    uint32_t len = 0;
    memcpy(hdr, &type, 4);
    memcpy(hdr + 4, &len, 4);
    if (d->rank < d->ranks - 1) {
        if (xio(d->down_fd, hdr, 8) != 0) return -1;
    }
    return 0;
}

void dist_close(Dist* d)
{
    if (d->up_fd >= 0) close(d->up_fd);
    if (d->down_fd >= 0) close(d->down_fd);
    if (d->log_fd >= 0) close(d->log_fd);
    free(d->x16);
    free(d->heap);
    memset(d, 0, sizeof(*d));
}
