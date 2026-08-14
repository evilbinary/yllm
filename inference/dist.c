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
#endif

static int sock_close_fd(int fd)
{
#ifdef _WIN32
    return closesocket((SOCKET)fd);
#else
    return sock_close_fd(fd);
#endif
}

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

/* 建立连接: 所有进程先 bind+listen(INADDR_ANY), 再按拓扑 connect。
 * addrs: 每 rank 节点 IP 数组(长度 ranks); NULL 时用 loopback。 */
int dist_init(Dist* d, int rank, int ranks, uint16_t port_base, const char* const* addrs)
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
    fprintf(stderr, "[distdbg] rank %d/%d: listen %u ok\n", rank, ranks, port_base + rank);

    if (rank < ranks - 1) {
        const char* ip = addrs ? addrs[rank + 1] : NULL;
        d->down_fd = sock_connect((uint16_t)(port_base + (uint16_t)(rank + 1)), ip);
        fprintf(stderr, "[distdbg] rank %d: connect down %s:%u -> %d\n", rank, ip ? ip : "(null)", port_base + rank + 1, d->down_fd);
        if (d->down_fd < 0) { fprintf(stderr, "dist: rank %d cannot connect to %u\n", rank, port_base + rank + 1); return -1; }
    }
    if (rank == 0) {
        d->log_fd = (int)accept(listen_fd, NULL, NULL);
        fprintf(stderr, "[distdbg] rank %d: accept log -> %d\n", rank, d->log_fd);
    } else {
        d->up_fd = (int)accept(listen_fd, NULL, NULL);
        fprintf(stderr, "[distdbg] rank %d: accept up -> %d\n", rank, d->up_fd);
        if (rank == ranks - 1) {
            const char* ip = addrs ? addrs[0] : NULL;
            d->log_fd = sock_connect(port_base, ip);
            fprintf(stderr, "[distdbg] rank %d: connect back %s:%u -> %d\n", rank, ip ? ip : "(null)", port_base, d->log_fd);
            if (d->log_fd < 0) { fprintf(stderr, "dist: rank %d cannot connect back to rank 0\n", rank); return -1; }
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
             uint64_t t0, dist_token_cb emit, void* ctx)
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
    if (dist_init(&dist, rank, ranks, (uint16_t)port_base, addr_arr) != 0) {
        ylog_error("dist init failed");
        return 1;
    }
    uint32_t hidden = e->ws.model.h.hidden;
    uint32_t vocab_sz = e->ws.model.h.vocab;
    enum { TOPK = 1024 };
    float* xbuf = (float*)ymalloc((size_t)hidden * 4);
    uint32_t* k_ids = (uint32_t*)ymalloc((size_t)TOPK * 4);

    uint64_t rng = ysrand(seed);
    int ngen = 0;
    int dist_stats = getenv("YLLM_DIST_STATS") != NULL;
    const int STATS_EVERY = 8;
    int rc = 0;

    if (rank == 0) {
        /* master: embed + 自己块段, 采样由收到的 top-k logits 决定 */
        uint32_t pos = 0;
        int i;
        int pend = 0;
        ylog_info("[dist] master start nprompt=%d ntokens=%d hidden=%u", nprompt, ntokens, hidden);
        for (i = 0; i < nprompt; i++) {
            engine_forward_range(e, ids[i], 1, pos, xbuf, NULL);
            if (dist_send_x(&dist, pos, xbuf, hidden, dist_fp16) != 0) { rc = -1; snprintf(err, sizeof(err), "send_x prompt failed"); break; }
            pos++;
        }
        ylog_info("[dist] master prefill done, sent %d X, recv logits...", nprompt);
        float lse = 0.0f;
        /* 丢弃 prompt 阶段多余的 top-k(末 rank 对每个 X 帧都回 logits) */
        for (i = 0; i < nprompt - 1; i++) {
            if (dist_recv_logits(&dist, k_ids, e->logits, vocab_sz, &lse) <= 0) { rc = -1; snprintf(err, sizeof(err), "recv logits prompt failed"); break; }
        }
        ylog_info("[dist] master prompt logits drained (rc=%d), gen loop...", rc);
        for (i = 0; i < ntokens && rc == 0; i++) {
            if (pos >= e->max_seq) break;
            int k = dist_recv_logits(&dist, k_ids, e->logits, vocab_sz, &lse);
            if (k <= 0) { rc = -1; snprintf(err, sizeof(err), "dist recv logits failed"); break; }
            pend = 0;
            uint32_t nxt;
            if (engine_sample(e, vocab_sz, temp, top_p, &rng, &nxt) != 0) { rc = -1; break; }
            if (getenv("YLLM_DISTDBG")) {
                fprintf(stderr, "R0 sampled k=%d id=%u [%s] logit=%.2f\n", k, nxt,
                        v->pieces[nxt], e->logits[nxt]);
            }
            if (emit) emit(nxt, ctx);
            if (nxt == (uint32_t)v->eos) break;
            engine_forward_range(e, nxt, 1, pos, xbuf, NULL);
            if (dist_send_x(&dist, pos, xbuf, hidden, dist_fp16) != 0) { rc = -1; break; }
            pos++;
            ngen++;
            pend = 1;
            if (dist_stats && (ngen % STATS_EVERY) == 0) {
                char tag[48];
                snprintf(tag, sizeof(tag), "dist@tok%d", ngen);
                dist_print_stats(&dist, tag);
            }
        }
        ylog_info("[dist] master gen loop done rc=%d ngen=%d", rc, ngen);
        /* 若最后一个 X 帧刚发出、其 logits 响应尚未读走则冲刷掉,
         * 避免 last rank 的 send 打到已关闭 socket。eos/错误退出时 pend=0 则不占。 */
        if (pend) {
            float* flush = (float*)ymalloc((size_t)vocab_sz * 4);
            (void)dist_recv_logits(&dist, NULL, flush, vocab_sz, NULL);
            free(flush);
        }
        dist_send_done(&dist);
        ylog_info("decode:  %d tokens in %.2f s (%.1f tok/s)", ngen,
               (double)(ynow_ms() - t0) / 1000.0,
               (double)ngen * 1000.0 / (double)(ynow_ms() - t0 > 0 ? ynow_ms() - t0 : 1));
    } else {
        /* 中段/末段 rank: 收激活 → 算自己块段 → 转发/出 top-k logits */
        uint32_t pos;
        int t;
        int nf = 0;
        ylog_info("[dist] worker %d/%d: waiting X frames...", rank, ranks);
        while ((t = dist_recv_x(&dist, &pos, xbuf, hidden, dist_fp16)) >= 0) {
            if (t == 3) { /* DONE: 向后转发并退出 */
                dist_send_done(&dist);
                break;
            }
            if (t != 1) break;
            nf++;
            memcpy(e->x, xbuf, (size_t)hidden * 4); /* 输入激活入引擎缓冲 */
            if (rank == ranks - 1) {
                engine_forward_range(e, 0, 0, pos, NULL, e->logits);
                if (dist_send_logits(&dist, e->logits, vocab_sz, TOPK) != 0) { rc = -1; break; }
            } else {
                engine_forward_range(e, 0, 0, pos, xbuf, NULL);
                if (dist_send_x(&dist, pos, xbuf, hidden, dist_fp16) != 0) { rc = -1; break; }
            }
            if ((nf & 7) == 0) ylog_info("[dist] worker processed %d X frames", nf);
            if (dist_stats && (nf % STATS_EVERY) == 0) {
                char tag[48];
                snprintf(tag, sizeof(tag), "dist@X%d", nf);
                dist_print_stats(&dist, tag);
            }
        }
        ylog_info("[dist] worker done rc=%d nf=%d", rc, nf);
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
    uint64_t per = total / (uint32_t)ranks;
    uint32_t bs[64]; /* 每个 rank 的起始 block */
    bs[0] = 1;
    uint64_t acc = 0;
    int r = 1;
    for (b = 1; b <= blocks && r < ranks; b++) {
        acc += e->ws.model.dir[b].size;
        if (acc >= per * (uint64_t)r) { bs[r] = b + 1; r++; }
    }
    while (r < ranks) { bs[r] = blocks + 1; r++; }
    if (bs[ranks - 1] > blocks + 1) bs[ranks - 1] = blocks + 1;
    uint32_t begin = bs[rank];
    uint32_t end = rank + 1 < ranks ? bs[rank + 1] : e->ws.model.n_layers;
    if (rank == 0) begin = 0; /* rank0 含 embed */
    engine_set_layers(e, begin, end);
    return 0;
}
