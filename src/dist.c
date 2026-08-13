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
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <time.h>
#define close(fd) closesocket(fd)
#define ssize_t int
#else
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
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, (const char*)&one, sizeof(one));
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
            setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (const char*)&one, sizeof(one));
            return fd;
        }
        close(fd);
        /* 对端可能尚未 listen(启动竞态), 每 200ms 重试 */
#ifdef _WIN32
        Sleep(200);
#else
        struct timespec ts;
        ts.tv_sec = 0;
        ts.tv_nsec = 200 * 1000 * 1000;
        nanosleep(&ts, NULL);
#endif
    }
    return -1;
}

static int xio(Dist* d, int fd, const void* buf, size_t n, int is_send)
{
    uint64_t t0 = ynow_ns();
    const char* p = (const char*)buf;
    while (n > 0) {
#ifdef _WIN32
        WSABUF wb;
        wb.buf = (char*)p;
        wb.len = (ULONG)(n > 0x7fffffff ? 0x7fffffff : n);
        DWORD sent = 0;
        DWORD flags = 0;
        int wsa = is_send ? WSASend((SOCKET)fd, &wb, 1, &sent, flags, NULL, NULL)
                          : WSARecv((SOCKET)fd, &wb, 1, &sent, &flags, NULL, NULL);
        if (wsa != 0) goto err_fail;
        if (sent == 0) goto err_fail;
        p += sent;
        n -= sent;
#else
        ssize_t r = (ssize_t)send(fd, p, n, 0);
        if (r <= 0) goto err_fail;
        p += r;
        n -= (size_t)r;
#endif
    }
    if (is_send) { d->bytes_sent += (uint64_t)(p - (const char*)buf); d->nanos_wait_send += ynow_ns() - t0; }
    else         { d->bytes_recv += (uint64_t)(p - (const char*)buf); d->nanos_wait_recv += ynow_ns() - t0; }
    return 0;
err_fail:
    if (getenv("YLLM_DISTDBG")) {
#ifdef _WIN32
        fprintf(stderr, "[distdbg] rank %d xio %s FAIL fd=%d n=%u WSAGetLastError=%d\n", d->rank, is_send ? "send" : "recv", fd, (unsigned)n, WSAGetLastError());
#else
        fprintf(stderr, "[distdbg] rank %d xio %s FAIL fd=%d n=%u errno=%d\n", d->rank, is_send ? "send" : "recv", fd, (unsigned)n, (int)errno);
#endif
    }
    return -1;
}
static int xrecv(Dist* d, int fd, void* buf, size_t n)
{
    return xio(d, fd, buf, n, 0);
}

/* 建立连接: 所有进程先 bind+listen, 再按拓扑 connect */
int dist_init(Dist* d, int rank, int ranks, uint16_t port_base)
{
#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
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
    if (xio(d, d->down_fd, hdr, 8, 1) != 0) return -1;
    if (xio(d, d->down_fd, &pos, 4, 1) != 0) return -1;
    if (fp16) {
        if (d->x16 == NULL || d->cap_x < hidden) {
            free(d->x16);
            d->x16 = (uint16_t*)ymalloc((size_t)hidden * 2);
            d->cap_x = hidden;
        }
        uint32_t i;
        for (i = 0; i < hidden; i++) d->x16[i] = f32_to_f16(x[i]);
        if (xio(d, d->down_fd, d->x16, (size_t)hidden * 2, 1) != 0) return -1;
    } else {
        if (xio(d, d->down_fd, x, (size_t)hidden * 4, 1) != 0) return -1;
    }
    d->n_x_sent++;
    return 0;
}

/* 返回帧类型: 1=X(pos/x 已填充), 3=DONE, -1=错误 */
int dist_recv_x(Dist* d, uint32_t* pos, float* x, uint32_t hidden, int fp16)
{
    int fd = d->rank == 0 ? d->log_fd : d->up_fd;
    uint8_t hdr[8];
    if (xrecv(d, fd, hdr, 8) != 0) return -1;
    uint32_t type, len;
    memcpy(&type, hdr, 4);
    memcpy(&len, hdr + 4, 4);
    if (type == DTYPE_DONE) return DTYPE_DONE;
    if (type == DTYPE_X) {
        if (len < 4) return -1;
        if (xrecv(d, fd, pos, 4) != 0) return -1;
        if (fp16) {
            if (d->x16 == NULL || d->cap_x < hidden) {
                free(d->x16);
                d->x16 = (uint16_t*)ymalloc((size_t)hidden * 2);
                d->cap_x = hidden;
            }
            if (xrecv(d, fd, d->x16, (size_t)hidden * 2) != 0) return -1;
            uint32_t i;
            for (i = 0; i < hidden; i++) x[i] = f16_to_f32(d->x16[i]);
        } else {
            if (xrecv(d, fd, x, (size_t)hidden * 4) != 0) return -1;
        }
        d->n_x_recv++;
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
    if (xio(d, d->log_fd, hdr, 8, 1) != 0) return -1;
    if (xio(d, d->log_fd, logits, (size_t)vocab * 4, 1) != 0) return -1;
    d->n_log_sent++;
    return 0;
}

/* 接收全量 logits */
int dist_recv_logits(Dist* d, uint32_t* ids, float* logits, uint32_t topk, float* lse_out)
{
    int fd = d->log_fd;
    uint8_t hdr[8];
    if (xrecv(d, fd, hdr, 8) != 0) return -1;
    uint32_t type, len;
    memcpy(&type, hdr, 4);
    memcpy(&len, hdr + 4, 4);
    if (type != DTYPE_LOGITS_K) return -1;
    uint32_t n = len / 4;
    if (n > topk) n = topk;
    if (xrecv(d, fd, logits, (size_t)n * 4) != 0) return -1;
    if (lse_out) *lse_out = 0.0f;
    (void)ids;
    d->n_log_recv++;
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
        if (xio(d, d->down_fd, hdr, 8, 1) != 0) return -1;
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

/* 打印本 rank 的传输带宽汇总采样 */
void dist_print_stats(Dist* d, const char* tag)
{
    double mb_sent = (double)d->bytes_sent / (1024.0 * 1024.0);
    double mb_recv = (double)d->bytes_recv / (1024.0 * 1024.0);
    double sec_send = (double)d->nanos_wait_send / 1e9;
    double sec_recv = (double)d->nanos_wait_recv / 1e9;
    fprintf(stderr, "[%s] rank %d/%d  X frames: sent=%llu recv=%llu | logits frames: sent=%llu recv=%llu\n",
            tag, d->rank, d->ranks,
            (unsigned long long)d->n_x_sent, (unsigned long long)d->n_x_recv,
            (unsigned long long)d->n_log_sent, (unsigned long long)d->n_log_recv);
    fprintf(stderr, "[%s] rank %d          bytes: sent=%.2f MB recv=%.2f MB | block: send %.1f ms recv %.1f ms\n",
            tag, d->rank, mb_sent, mb_recv, sec_send * 1000.0, sec_recv * 1000.0);
    fprintf(stderr, "[%s] rank %d          bandwidth: send %.1f MB/s recv %.1f MB/s\n",
            tag, d->rank,
            sec_send > 0 ? mb_sent / sec_send : 0.0,
            sec_recv > 0 ? mb_recv / sec_recv : 0.0);
    if (d->elapsed_ms > 0) {
        double sec_el = d->elapsed_ms / 1000.0;
        fprintf(stderr, "[%s] rank %d          avg throughput: send %.2f MB/s recv %.2f MB/s (total %.2f s)\n",
                tag, d->rank, mb_sent / sec_el, mb_recv / sec_el, sec_el);
    }
}
