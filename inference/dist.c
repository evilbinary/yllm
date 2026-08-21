/* dist.c — 层流水线分布式传输(TCP)
 *
 * 拓扑: rank i 把激活发给 rank i+1; 末 rank 把 top-k logits 发回 rank 0。
 * 帧格式: [4B type][4B len][payload]
 *   type 1 = X(激活: pos(4B) + hidden 个 fp16)
 *   type 2 = LOGITS_K(top-k: k(4B) + k × [id(4B) + logit fp16(2B)])
 *   type 3 = DONE(终止)
 */
#include "dist.h"
#include "cache.h"
#include "yllm.h"
#include "log.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <time.h>
#define ssize_t int
#else
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <time.h>
#include <math.h>
#include <poll.h>
#endif

static int sock_close_fd(int fd)
{
#ifdef _WIN32
    return closesocket((SOCKET)fd);
#else
    return close(fd);
#endif
}

/* 等待 fd 可读(毫秒超时): 可读返回 1, 超时返回 0, 错误返回 -1 */
static int dist_wait_readable(int fd, int ms)
{
#ifdef _WIN32
    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(fd, &rfds);
    struct timeval tv;
    tv.tv_sec = ms / 1000;
    tv.tv_usec = (ms % 1000) * 1000;
    int sel = select(fd + 1, &rfds, NULL, NULL, &tv);
    if (sel < 0) return -1;
    return sel > 0 && FD_ISSET(fd, &rfds) ? 1 : 0;
#else
    struct pollfd pfd;
    pfd.fd = fd;
    pfd.events = POLLIN;
    int sel = poll(&pfd, 1, ms);
    if (sel < 0) return -1;
    return sel > 0 && (pfd.revents & POLLIN) ? 1 : 0;
#endif
}

#define DTYPE_X 1
#define DTYPE_LOGITS_K 2
#define DTYPE_DONE 3
#define DTYPE_SESS 4
#define DTYPE_XB 5

/* PP prefill 批量化(默认开): 批量 XB 帧消除逐 token logits 回执软同步。
 * 编译时 -DYLLM_BATCH_PREFILL=0 关闭, 回到逐 token 路径。 */
#ifndef YLLM_BATCH_PREFILL
#define YLLM_BATCH_PREFILL 1
#endif

/* PP 批量 prefill 每批 token 数 */
#define PP_XB 16

static int sock_listen(uint16_t port)
{
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, (const char*)&one, sizeof(one));
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(port);
    if (bind(fd, (struct sockaddr*)&addr, sizeof(addr)) != 0) { sock_close_fd(fd); return -1; }
    if (listen(fd, 8) != 0) { sock_close_fd(fd); return -1; }
    return fd;
}

static int sock_connect(uint16_t port, const char* ip)
{
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    if (ip && ip[0]) {
#ifdef _WIN32
        if (inet_pton(AF_INET, ip, &addr.sin_addr) != 1)
            addr.sin_addr.S_un.S_addr = inet_addr(ip);
#else
        if (inet_pton(AF_INET, ip, &addr.sin_addr) != 1)
            addr.sin_addr.s_addr = inet_addr(ip);
#endif
    } else {
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    }
    addr.sin_port = htons(port);
    int attempt;
    /* 跨机场景对端可能仍在加载模型/尚未 listen, 重试 500 次 ×200ms = 100s */
    for (attempt = 0; attempt < 500; attempt++) {
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) return -1;
        if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
            int one = 1;
            setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (const char*)&one, sizeof(one));
            return fd;
        }
        sock_close_fd(fd);
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
        ssize_t r = is_send ? (ssize_t)send(fd, p, n, 0)
                            : (ssize_t)recv(fd, p, n, 0);
        if (r <= 0) {
            int e0 = errno;
            if (getenv("YLLM_DISTDBG"))
                ylog_info("xio %s %s FAIL fd=%d n=%u errno=%d", is_send ? "send" : "recv",
                          r == 0 ? "EOF" : "ERR", fd, (unsigned)n, e0);
            goto err_fail;
        }
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

/* 建立连接: 所有进程先 bind+listen(INADDR_ANY), 再按拓扑 connect。
 * addrs: 每 rank 节点 IP 数组(长度 ranks); NULL 时用 loopback。 */
/* 等待 accept(可中断): 返回 fd; -2 = quit 信号; -1 = 错误 */
static int dist_accept_wait(int listen_fd, const volatile int* quit)
{
    if (!quit) return (int)accept(listen_fd, NULL, NULL);
    for (;;) {
        if (*quit) { ylog_info("dist: accept interrupted by quit"); return -2; }
        int sel = dist_wait_readable(listen_fd, 200);
        if (sel < 0) return -1;
        if (sel > 0) return (int)accept(listen_fd, NULL, NULL);
    }
}

int dist_init(Dist* d, int rank, int ranks, uint16_t port_base, const char* const* addrs,
              const volatile int* quit)
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
    d->quit = quit;
    int listen_fd = sock_listen((uint16_t)(port_base + (uint16_t)rank));
    if (listen_fd < 0) { fprintf(stderr, "dist: rank %d cannot listen on port %u\n", rank, port_base + rank); return -1; }

    if (rank < ranks - 1) {
        const char* ip = addrs ? addrs[rank + 1] : NULL;
        d->down_fd = sock_connect((uint16_t)(port_base + (uint16_t)(rank + 1)), ip);
        if (d->down_fd < 0) { fprintf(stderr, "dist: rank %d cannot connect to %u\n", rank, port_base + rank + 1); sock_close_fd(listen_fd); return -1; }
    }
    if (rank == 0) {
        d->log_fd = dist_accept_wait(listen_fd, quit);
        if (d->log_fd < 0) { int rc = (d->log_fd == -2) ? -2 : -1; sock_close_fd(listen_fd); return rc; }
    } else {
        d->up_fd = dist_accept_wait(listen_fd, quit);
        if (d->up_fd < 0) { int rc = (d->up_fd == -2) ? -2 : -1; sock_close_fd(listen_fd); return rc; }
        if (rank == ranks - 1) {
            const char* ip = addrs ? addrs[0] : NULL;
            d->log_fd = sock_connect(port_base, ip);
            if (d->log_fd < 0) { fprintf(stderr, "dist: rank %d cannot connect back to rank 0\n", rank); sock_close_fd(listen_fd); return -1; }
        }
    }
    sock_close_fd(listen_fd);
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
    if (d->quit) {
        /* worker 空闲等激活: 周期超时检查退出信号(DRAIN 后及时退出落盘) */
        for (;;) {
            if (*d->quit) return -2;
            int sel = dist_wait_readable(fd, 500);
            if (sel > 0) break;
            if (sel < 0) return -1;
        }
    }
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

/* 批量激活传输(prefill): [type=5][len][pos 4B][count 4B][x: count*hidden*(2|4)B] */
int dist_send_xb(Dist* d, uint32_t pos, const float* x, uint32_t count,
                 uint32_t hidden, int fp16)
{
    uint8_t hdr[8];
    uint32_t esz = fp16 ? 2u : 4u;
    uint32_t len = 4 + 4 + count * hidden * esz;
    uint32_t type = DTYPE_XB;
    memcpy(hdr, &type, 4);
    memcpy(hdr + 4, &len, 4);
    if (xio(d, d->down_fd, hdr, 8, 1) != 0) return -1;
    if (xio(d, d->down_fd, &pos, 4, 1) != 0) return -1;
    if (xio(d, d->down_fd, &count, 4, 1) != 0) return -1;
    if (getenv("YLLM_DISTDBG"))
        ylog_info("send_xb: count=%u fp16=%d pos=%u len=%u", count, fp16, pos, len);
    if (fp16) {
        uint16_t* x16 = (uint16_t*)ymalloc((size_t)count * hidden * 2);
        uint32_t i;
        for (i = 0; i < count * hidden; i++) x16[i] = f32_to_f16(x[i]);
        int rc = xio(d, d->down_fd, x16, (size_t)count * hidden * 2, 1);
        free(x16);
        if (rc != 0) return -1;
    } else {
        if (xio(d, d->down_fd, x, (size_t)count * hidden * 4, 1) != 0) return -1;
    }
    d->n_x_sent++;
    return 0;
}

/* 接收批量激活; 返回 count(>0), -1=错误 */
int dist_recv_xb(Dist* d, uint32_t* pos, float* x, uint32_t cap_count,
                 uint32_t hidden, int fp16)
{
    int fd = d->rank == 0 ? d->log_fd : d->up_fd;
    if (d->quit) {
        for (;;) {
            if (*d->quit) return -2;
            int sel = dist_wait_readable(fd, 500);
            if (sel > 0) break;
            if (sel < 0) return -1;
        }
    }
    uint8_t hdr[8];
    if (d->has_peek) {
        /* 会话握手探测暂存的帧头: 直接消费, 不重复 recv */
        memcpy(hdr, d->peek_hdr, 8);
        d->has_peek = 0;
    } else {
        if (xrecv(d, fd, hdr, 8) != 0) return -1;
    }
    uint32_t type, len;
    memcpy(&type, hdr, 4);
    memcpy(&len, hdr + 4, 4);
    if (getenv("YLLM_DISTDBG"))
        ylog_info("recv_xb: type=%u len=%u", type, len);
    if (type == DTYPE_DONE) return -3;
    if (type == DTYPE_X) {
        /* decode 单 token 帧(与批量 prefill 混流): 按 count=1 处理 */
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
        return 1;
    }
    if (type != DTYPE_XB) return -1;
    uint32_t esz = fp16 ? 2u : 4u;
    uint32_t count = (len - 8) / (hidden * esz);
    if (count < 1 || count > cap_count) return -1;
    if (xrecv(d, fd, pos, 4) != 0) return -1;
    if (xrecv(d, fd, &count, 4) != 0) return -1;
    if (fp16) {
        if (d->x16 == NULL || d->cap_x < count * hidden) {
            free(d->x16);
            d->x16 = (uint16_t*)ymalloc((size_t)count * hidden * 2);
            d->cap_x = count * hidden;
        }
        if (xrecv(d, fd, d->x16, (size_t)count * hidden * 2) != 0) return -1;
        uint32_t i;
        for (i = 0; i < count * hidden; i++) x[i] = f16_to_f32(d->x16[i]);
    } else {
        if (xrecv(d, fd, x, (size_t)count * hidden * 4) != 0) return -1;
    }
    d->n_x_recv++;
    return (int)count;
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
    if (xrecv(d, fd, hdr, 8) != 0) {
        if (getenv("YLLM_DISTDBG"))
            ylog_info("recv_logits: xrecv hdr FAIL errno=%d", errno);
        return -1;
    }
    uint32_t type, len;
    memcpy(&type, hdr, 4);
    memcpy(&len, hdr + 4, 4);
    if (type != DTYPE_LOGITS_K) {
        if (getenv("YLLM_DISTDBG"))
            ylog_info("recv_logits: unexpected type=%u len=%u", type, len);
        return -1;
    }
    uint32_t n = len / 4;
    if (n > topk) n = topk;
    if (xrecv(d, fd, logits, (size_t)n * 4) != 0) return -1;
    if (lse_out) *lse_out = 0.0f;
    (void)ids;
    d->n_log_recv++;
    return (int)n;
}

/* 会话握手帧: [type=4][len=68][key 64B][pos 4B], 由 master 在首帧前发给下一段 */
int dist_send_sess(Dist* d, const char* key, uint32_t pos)
{
    uint8_t hdr[8];
    uint32_t type = DTYPE_SESS;
    uint32_t len = 64 + 4;
    memcpy(hdr, &type, 4);
    memcpy(hdr + 4, &len, 4);
    if (xio(d, d->down_fd, hdr, 8, 1) != 0) return -1;
    char kbuf[64];
    memset(kbuf, 0, sizeof(kbuf));
    snprintf(kbuf, sizeof(kbuf), "%s", key);
    if (xio(d, d->down_fd, kbuf, 64, 1) != 0) return -1;
    if (xio(d, d->down_fd, &pos, 4, 1) != 0) return -1;
    return 0;
}

int dist_recv_sess(Dist* d, char* key, uint32_t* pos)
{
    int fd = d->rank == 0 ? d->log_fd : d->up_fd;
    uint8_t hdr[8];
    if (xrecv(d, fd, hdr, 8) != 0) return -1;
    uint32_t type, len;
    memcpy(&type, hdr, 4);
    memcpy(&len, hdr + 4, 4);
    if (type != DTYPE_SESS || len < 68) {
        /* 非会话模式(master 未发握手): 暂存帧头供后续 recv_xb 读取, 返回 -2 表示无会话 */
        memcpy(d->peek_hdr, hdr, 8);
        d->has_peek = 1;
        return -2;
    }
    char kbuf[64];
    if (xrecv(d, fd, kbuf, 64) != 0) return -1;
    if (xrecv(d, fd, pos, 4) != 0) return -1;
    kbuf[63] = 0;
    snprintf(key, 64, "%s", kbuf);
    return 0;
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
    if (d->up_fd >= 0) sock_close_fd(d->up_fd);
    if (d->down_fd >= 0) sock_close_fd(d->down_fd);
    if (d->log_fd >= 0) sock_close_fd(d->log_fd);
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
    ylog_info("[%s] rank %d/%d X frames: sent=%llu recv=%llu | logits frames: sent=%llu recv=%llu | bytes: sent=%.2f MB recv=%.2f MB | block: send %.1f ms recv %.1f ms | bw: send %.1f MB/s recv %.1f MB/s",
            tag, d->rank, d->ranks,
            (unsigned long long)d->n_x_sent, (unsigned long long)d->n_x_recv,
            (unsigned long long)d->n_log_sent, (unsigned long long)d->n_log_recv,
            mb_sent, mb_recv, sec_send * 1000.0, sec_recv * 1000.0,
            sec_send > 0 ? mb_sent / sec_send : 0.0,
            sec_recv > 0 ? mb_recv / sec_recv : 0.0);
    if (d->elapsed_ms > 0) {
        double sec_el = d->elapsed_ms / 1000.0;
        ylog_info("[%s] rank %d avg throughput: send %.2f MB/s recv %.2f MB/s (total %.2f s)",
                tag, d->rank, mb_sent / sec_el, mb_recv / sec_el, sec_el);
    }
}

/* 分布式层流水线推理: 各 rank 均执行, 按 rank 分 master / middle / last。
 * addrs: 每 rank 节点 IP(逗号分隔, 长度=ranks); NULL 退化为 loopback。
 * emit 为生成 token 的输出回调(跑在 rank0)。 */
int dist_gen(Engine* e, Vocab* v, const uint32_t* ids, int nprompt,
             int ntokens, float temp, float top_p, uint64_t seed,
             int rank, int ranks, int port_base, const char* addrs, int dist_fp16,
             uint64_t t0, dist_token_cb emit, void* ctx,
             DistSess* sess)
{
    Dist dist;
    char err[256];
    const char* addr_list[64];
    const char* const* addr_arr = NULL;
    if (addrs && addrs[0] && ranks <= 64) {
        char tmp[1024];
        snprintf(tmp, sizeof(tmp), "%s", addrs);
        char* tok = strtok(tmp, ",");
        int i = 0;
        static char nodes[64][128];
        while (tok && i < ranks) {
            snprintf(nodes[i], sizeof(nodes[i]), "%s", tok);
            addr_list[i] = nodes[i];
            tok = strtok(NULL, ",");
            i++;
        }
        if (i == ranks) addr_arr = addr_list;
    }
    int init_rc = dist_init(&dist, rank, ranks, (uint16_t)port_base, addr_arr,
                            sess ? sess->quit : NULL);
    if (init_rc == -2) return 0;   /* 被 quit 中断: 正常退出, 不报错 */
    if (init_rc != 0) {
        ylog_error("dist init failed");
        return 1;
    }
    dist.quit = sess ? sess->quit : NULL;
    uint32_t hidden = e->ws.model.h.hidden;
    uint32_t vocab_sz = e->ws.model.h.vocab;
    enum { TOPK = 1024 };
    float* xbuf = (float*)ymalloc((size_t)hidden * 4 * PP_XB);
    uint32_t* k_ids = (uint32_t*)ymalloc((size_t)TOPK * 4);

    uint64_t rng = ysrand(seed);
    int ngen = 0;
    uint64_t t_dec0 = 0;
    int dist_stats = getenv("YLLM_DIST_STATS") != NULL;
    const int STATS_EVERY = 8;
    int rc = 0;

    if (rank == 0) {
        /* master: embed + 自己块段, 采样由收到的 top-k logits 决定 */
        uint32_t pos = sess ? sess->pos : 0;
        int i;
        int pend = 0;
        if (sess) {
            if (dist_send_sess(&dist, sess->key, sess->pos) != 0) {
                rc = -1; snprintf(err, sizeof(err), "send sess failed");
            }
        }
#if YLLM_BATCH_PREFILL
        for (i = 0; i < nprompt && rc == 0; ) {
            int nb = nprompt - i;
            if (nb > PP_XB) nb = PP_XB;
            float* xb = (float*)ymalloc((size_t)hidden * 4 * PP_XB);
            if (engine_forward_batch_tokens(e, ids + i, nb, pos, xb) != 0) {
                free(xb); rc = -1; snprintf(err, sizeof(err), "batch prefill failed"); break;
            }
            if (dist_send_xb(&dist, pos, xb, (uint32_t)nb, hidden, dist_fp16) != 0) {
                free(xb); rc = -1; snprintf(err, sizeof(err), "send_xb prompt failed"); break;
            }
            if (getenv("YLLM_DISTDBG"))
                ylog_info("R0 send_xb pos=%u xb[0..3]=%g %g %g %g", pos,
                          (double)xb[0], (double)xb[1], (double)xb[2], (double)xb[3]);
            free(xb);
            pos += (uint32_t)nb;
            i += nb;
        }
#else
        for (i = 0; i < nprompt && rc == 0; i++) {
            engine_forward_range(e, ids[i], 1, pos, xbuf, NULL);
            if (dist_send_x(&dist, pos, xbuf, hidden, dist_fp16) != 0) {
                rc = -1; snprintf(err, sizeof(err), "send_x prompt failed"); break;
            }
            pos++;
        }
#endif
        float lse = 0.0f;
        /* 丢弃 prompt 阶段多余的 logits(末 rank 每批回 1 个) */
        {
#if YLLM_BATCH_PREFILL
            int nbatch = (nprompt + PP_XB - 1) / PP_XB;
            for (i = 0; i < nbatch - 1; i++) {
#else
            for (i = 0; i < nprompt - 1; i++) {
#endif
                if (dist_recv_logits(&dist, k_ids, e->logits, vocab_sz, &lse) <= 0) { rc = -1; snprintf(err, sizeof(err), "recv logits prompt failed"); break; }
            }
        }
        int dist_timing = getenv("YLLM_DISTTIMING") != NULL;
        uint64_t t_wait = 0, t_samp = 0, t_fwd = 0, t_snd = 0;
        for (i = 0; i < ntokens && rc == 0; i++) {
            if (pos >= e->max_seq) break;
            if (t_dec0 == 0) t_dec0 = ynow_ms();
            uint64_t t0_tok = ynow_ms();
            int k = dist_recv_logits(&dist, k_ids, e->logits, vocab_sz, &lse);
            uint64_t t1_tok = ynow_ms();
            if (k <= 0) { rc = -1; snprintf(err, sizeof(err), "dist recv logits failed"); break; }
            pend = 0;
            uint32_t nxt;
            if (engine_sample(e, vocab_sz, temp, top_p, &rng, &nxt) != 0) { rc = -1; break; }
            uint64_t t2_tok = ynow_ms();
            if (getenv("YLLM_DISTDBG")) {
                fprintf(stderr, "R0 sampled k=%d id=%u [%s] logit=%.2f\n", k, nxt,
                        v->pieces[nxt], e->logits[nxt]);
            }
            if (nxt == (uint32_t)v->eos) break;
            engine_forward_range(e, nxt, 1, pos, xbuf, NULL);
            uint64_t t3_tok = ynow_ms();
            if (dist_send_x(&dist, pos, xbuf, hidden, dist_fp16) != 0) { rc = -1; break; }
            if (getenv("YLLM_DISTDBG"))
                ylog_info("R0 send_x pos=%u xbuf[0..3]=%g %g %g %g", pos,
                          (double)xbuf[0], (double)xbuf[1], (double)xbuf[2], (double)xbuf[3]);
            uint64_t t4_tok = ynow_ms();
            /* emit 放到 send_x 之后: 先让 worker 开工, 再输出 token。
             * 避免 emit 的同步 send/日志阻塞推迟 worker 启动(stream/慢客户端下反压)。 */
            if (emit) emit(nxt, ctx);
            if (dist_timing) {
                t_wait += t1_tok - t0_tok;
                t_samp += t2_tok - t1_tok;
                t_fwd += t3_tok - t2_tok;
                t_snd += t4_tok - t3_tok;
                ylog_info("tok%d: wait=%u sample=%u fwd=%u send=%u total=%u",
                          i + 1,
                          (unsigned)(t1_tok - t0_tok), (unsigned)(t2_tok - t1_tok),
                          (unsigned)(t3_tok - t2_tok), (unsigned)(t4_tok - t3_tok),
                          (unsigned)(t4_tok - t0_tok));
            }
            pos++;
            ngen++;
            pend = 1;
            if (dist_stats && (ngen % STATS_EVERY) == 0) {
                char tag[48];
                snprintf(tag, sizeof(tag), "dist@tok%d", ngen);
                dist_print_stats(&dist, tag);
            }
        }
        if (dist_timing) {
            ylog_info("tok sum: wait=%llu sample=%llu fwd=%llu send=%llu (16 tok)",
                      (unsigned long long)t_wait, (unsigned long long)t_samp,
                      (unsigned long long)t_fwd, (unsigned long long)t_snd);
        }
        /* 若最后一个 X 帧刚发出、其 logits 响应尚未读走则冲刷掉,
         * 避免 last rank 的 send 打到已关闭 socket。eos/错误退出时 pend=0 则不占。 */
        if (pend) {
            float* flush = (float*)ymalloc((size_t)vocab_sz * 4);
            (void)dist_recv_logits(&dist, NULL, flush, vocab_sz, NULL);
            free(flush);
        }
        /* 会话模式: 生成结束(eos/上限)后各段补写 eos 的 kv, 与单机一致 */
        if (sess && rc == 0 && pos < e->max_seq) {
            engine_forward_range(e, (uint32_t)v->eos, 1, pos, xbuf, NULL);
            if (dist_send_x(&dist, pos, xbuf, hidden, dist_fp16) == 0) {
                pos++;
                float* flush = (float*)ymalloc((size_t)vocab_sz * 4);
                (void)dist_recv_logits(&dist, NULL, flush, vocab_sz, NULL);
                free(flush);
            }
        }
        if (sess) sess->pos = pos;
        dist_send_done(&dist);
        uint64_t t_end = ynow_ms();
        uint64_t t_dec = t_dec0 ? t_dec0 : t_end;
        if (ngen > 0) ylog_raw_log("\n");   /* 生成的最后一个 token 后换行 */
        ylog_info("decode:  %d tokens in %.2f s (%.2f tok/s)", ngen,
               (double)(t_end - t_dec) / 1000.0,
               (double)ngen * 1000.0 / (double)(t_end - t_dec > 0 ? t_end - t_dec : 1));
    } else {
        /* 中段/末段 rank: 收激活 → 算自己块段 → 转发/出 top-k logits */
        uint32_t pos;
        int t;
        int nf = 0;
        /* 会话握手: 校验本段 kv 状态与期望一致
         * (-2 = master 未握手, 非会话模式, 跳过会话初始化直接走 X/XB 流) */
        if (sess) {
            char skey[64] = "";
            uint32_t spos = 0;
            int hs = dist_recv_sess(&dist, skey, &spos);
            if (hs == -1) {
                rc = -1; snprintf(err, sizeof(err), "sess handshake failed");
            } else if (hs == 0) {
                snprintf(sess->key, sizeof(sess->key), "%s", skey);
                sess->pos = spos;
                /* 中段: 把手握转发给下一段(3+ rank 拓扑) */
                if (rank < ranks - 1) {
                    if (dist_send_sess(&dist, skey, spos) != 0) {
                        rc = -1; snprintf(err, sizeof(err), "sess fwd failed");
                    }
                }
                /* 新会话(与上次不同 key): 本段 kv 重置, 按会话 key 从磁盘恢复
                 * (仅 my_pos==0 会漏掉多会话交替场景: 上轮结束的 pos 非 0) */
                if (sess->cache_dir && strcmp(sess->last_key, skey) != 0) {
                    snprintf(sess->last_key, sizeof(sess->last_key), "%s", skey);
                    sess->my_pos = 0;
                    char path[512];
                    char ext[32];
                    snprintf(ext, sizeof(ext), ".r%d.kv", rank);
                    cache_path(path, sizeof(path), sess->cache_dir, skey, ext);
                    uint32_t loaded = 0;
                    if (sess_kv_load(e, path, &loaded) == 0) {
                        sess->my_pos = loaded;
                        ylog_info("dist worker: kv restored (%u tokens) from %s", loaded, path);
                    }
                }
                if (spos != sess->my_pos) {
                    if (spos == 0) {
                        /* 全量重发: 本段 kv 由 X 流从 0 覆盖 */
                        ylog_warn("dist worker: full resend (pos %u -> 0)", sess->my_pos);
                        sess->my_pos = 0;
                    } else {
                        ylog_warn("dist worker: sess pos mismatch (%u vs %u), need full resend",
                                  spos, sess->my_pos);
                        rc = -1;
                    }
                }
            }
        }
        int w_timing = getenv("YLLM_DISTTIMING") != NULL;
        for (;;) {
            uint64_t w0 = ynow_ms();
            t = dist_recv_xb(&dist, &pos, xbuf, PP_XB, hidden, dist_fp16);
            uint64_t w1 = ynow_ms();
            if (t == -3) { /* DONE: 向后转发并退出 */
                dist_send_done(&dist);
                break;
            }
            if (t < 1 || rc) {
                if (t >= 0 && rc == 0 && getenv("YLLM_DISTDBG"))
                    ylog_info("worker recv_xb exit: t=%d rc=%d", t, rc);
                break;
            }
            nf += t;
            uint64_t f0 = ynow_ms();
            if (t == 1) {
                /* 单 token(decode): 走单发路径 —— 批量路径在 B=1 时反量化开销
                 * 无 batch 分摊, 比单发优化版 matmul 慢 ~7 倍 */
                memcpy(e->x, xbuf, (size_t)hidden * 4);
                cuda_mark_x_host(e);
                if (rank == ranks - 1) {
                    engine_forward_range(e, 0, 0, pos, NULL, e->logits);
                    uint64_t f1 = ynow_ms();
                    if (getenv("YLLM_DISTDBG"))
                        ylog_info("wtok last x[0..3]=%g %g %g %g logits[0..3]=%g %g %g %g",
                                  (double)e->x[0], (double)e->x[1], (double)e->x[2], (double)e->x[3],
                                  (double)e->logits[0], (double)e->logits[1], (double)e->logits[2], (double)e->logits[3]);
                    if (w_timing)
                        ylog_info("wtok %d: wait=%u fwd=%u", nf,
                                  (unsigned)(w1 - w0), (unsigned)(f1 - f0));
                    if (dist_send_logits(&dist, e->logits, vocab_sz, TOPK) != 0) { rc = -1; break; }
                } else {
                    engine_forward_range(e, 0, 0, pos, xbuf, NULL);
                    if (dist_send_xb(&dist, pos, xbuf, 1, hidden, dist_fp16) != 0) { rc = -1; break; }
                }
            } else if (rank == ranks - 1) {
                if (getenv("YLLM_DISTDBG"))
                    ylog_info("wbatch: t=%d pos=%u", t, pos);
                if (engine_forward_batch_x(e, xbuf, t, pos, NULL, e->logits) != 0) {
                    ylog_info("worker batch-x failed: t=%d pos=%u rc=%d", t, pos, rc);
                    rc = -1; break;
                }
                if (dist_send_logits(&dist, e->logits, vocab_sz, TOPK) != 0) {
                    ylog_info("worker send_logits failed: t=%d pos=%u", t, pos);
                    rc = -1; break;
                }
            } else {
                if (engine_forward_batch_x(e, xbuf, t, pos, xbuf, NULL) != 0) {
                    ylog_info("worker batch-x failed: t=%d pos=%u rc=%d", t, pos, rc);
                    rc = -1; break;
                }
                if (dist_send_xb(&dist, pos, xbuf, (uint32_t)t, hidden, dist_fp16) != 0) {
                    ylog_info("worker send_xb failed: t=%d pos=%u", t, pos);
                    rc = -1; break;
                }
            }
            if (sess) sess->my_pos = pos + (uint32_t)t;   /* 本段 kv 已推进 */
            if (getenv("YLLM_DISTDBG"))
                ylog_info("frame %d: %u tokens in %llu ms", nf, (uint32_t)t,
                          (unsigned long long)(ynow_ms() - f0));
            if (dist_stats && (nf % STATS_EVERY) == 0) {
                char tag[48];
                snprintf(tag, sizeof(tag), "dist@X%d", nf);
                dist_print_stats(&dist, tag);
            }
        }
        if (getenv("YLLM_DISTDBG"))
            ylog_info("worker loop exit: t=%d rc=%d nf=%u", t, rc, (unsigned)nf);
    }
    free(xbuf);
    free(k_ids);

    dist.elapsed_ms = (double)(ynow_ms() - t0);
    if (dist_stats || getenv("YLLM_DISTDBG")) dist_print_stats(&dist, "dist");
    dist_close(&dist);
    if (rc != 0) ylog_error("dist generate failed: %s", err);
    return rc == 0 ? 0 : 1;
}

/* 按字节均衡切层: 各 rank 领一个块区间, 末 rank 略少分块(norm+head)。
 * 写入 engine 的层范围供 dist_gen 各段使用。 */
int dist_split_layers(Engine* e, int rank, int ranks)
{
    uint32_t blocks = e->ws.model.h.n_blocks;
    if ((uint32_t)ranks > blocks) {
        ylog_error("ranks %d > blocks %u", ranks, blocks);
        return -1;
    }
    uint32_t b;
    uint64_t total = 0;
    for (b = 1; b <= blocks; b++) total += e->ws.model.dir[b].size;
    /* 可选按算力加权切层: YLLM_DIST_WEIGHTS="w0,w1,.." 为各 rank 相对权重
     * (总字节占比), 用于异构机器(如本机 6 核 + 远程 28 核)均衡负载。
     * 缺省或参数不合法时退化为按字节均分。 */
    float wcum[64];
    int i, have_w = 0;
    const char* ws = getenv("YLLM_DIST_WEIGHTS");
    if (ws && *ws && ranks <= 64) {
        float w[64];
        int cnt = 0;
        const char* p = ws;
        while (*p && cnt < ranks) {
            w[cnt++] = (float)atof(p);
            while (*p && *p != ',') p++;
            if (*p == ',') p++;
        }
        if (cnt == ranks) {
            float sum = 0;
            for (i = 0; i < ranks; i++) sum += w[i];
            if (sum > 0) {
                wcum[0] = w[0] / sum;
                for (i = 1; i < ranks; i++) wcum[i] = wcum[i - 1] + w[i] / sum;
                wcum[ranks - 1] = 1.0f;
                have_w = 1;
            }
        }
    }
    if (!have_w)
        for (i = 0; i < ranks; i++) wcum[i] = (float)(i + 1) / (float)ranks;
    uint32_t bs[64]; /* 每个 rank 的起始 block */
    bs[0] = 1;
    uint64_t acc = 0;
    int r = 1;
    for (b = 1; b <= blocks && r < ranks; b++) {
        acc += e->ws.model.dir[b].size;
        if (acc >= (uint64_t)((double)total * wcum[r - 1])) { bs[r] = b + 1; r++; }
    }
    while (r < ranks) { bs[r] = blocks + 1; r++; }
    if (bs[ranks - 1] > blocks + 1) bs[ranks - 1] = blocks + 1;
    uint32_t begin = bs[rank];
    uint32_t end = rank + 1 < ranks ? bs[rank + 1] : e->ws.model.n_layers;
    if (rank == 0) begin = 0; /* rank0 含 embed */
    engine_set_layers(e, begin, end);
    ylog_info("dist_split_layers: rank %d/%d layers [%u,%u) of %u (blocks %u)",
              rank, ranks, begin, end, e->ws.model.n_layers, blocks);
    return 0;
}
