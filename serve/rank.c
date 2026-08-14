/* rank.c — yllm rank: 常驻推理单元
 *
 * 公用 rank 池的最小计算单元: 加载模型层段权重常驻,
 * 监听推理端口, 循环处理服务层帧(PING/STAT/INFER/DRAIN/QUIT)。
 *
 * 用法:
 *   yllm rank --model <file.llf> --vocab <file> [--port N]
 *             [--budget-mb N] [--depth N] [--temp F] [--top-p F] [--seed N]
 *             [--log <file>] [--log-level lvl] [--no-console]
 *
 * P1 阶段: 单 rank(ranks=1, 完整模型本地推理)。
 * 多 rank 流水线(dist_split_layers + dist 收发)在 P2 接入。
 */
#include "protocol.h"
#include "frame.h"
#include "node.h"
#include "sock.h"
#include "../inference/yllm.h"
#include "../inference/log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <pthread.h>
#include <stdint.h>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#define ssize_t int
#else
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#endif

#define RANK_MAX_LINE 8192

typedef struct {
    Engine engine;
    Vocab vocab;
    float temp;
    float top_p;
    uint64_t seed;
    uint64_t uptime_s;
    /* 统一节点身份(心跳发 supervisor, 生命周期归 supervisor) */
    Node node;
    pthread_mutex_t engine_lock;   /* 引擎单实例: INFER 互斥, PING/STAT 不阻塞 */
    volatile int quit;   /* QUIT 后主循环退出, 心跳线程一并退出 */
} Rank;

/* ---- socket 辅助(与 dist_worker 同款) ---- */

static void ws_init_rank(void)
{
#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
}

static int xrecv_rank(int fd, void* buf, size_t n)
{
    char* p = (char*)buf;
    while (n > 0) {
#ifdef _WIN32
        int r = recv(fd, p, (int)(n > 0x7fffffff ? 0x7fffffff : n), 0);
#else
        ssize_t r = recv(fd, p, n, 0);
#endif
        if (r <= 0) return -1;
        p += (size_t)r;
        n -= (size_t)r;
    }
    return 0;
}

/* 发送: 返回 -1 表示对端已断开(调用方应中止) */
static int xsend_rank(int fd, const void* buf, size_t n)
{
    const char* p = (const char*)buf;
    while (n > 0) {
#ifdef _WIN32
        int r = send(fd, p, (int)(n > 0x7fffffff ? 0x7fffffff : n), 0);
#else
        ssize_t r = send(fd, p, n, 0);
#endif
        if (r <= 0) return -1;
        p += (size_t)r;
        n -= (size_t)r;
    }
    return 0;
}

/* 读一行到 max(不含换行), 返回长度或 -1 */
static int recv_line_rank(int fd, char* buf, size_t max)
{
    size_t n = 0;
    for (;;) {
        char c;
#ifdef _WIN32
        int r = recv(fd, &c, 1, 0);
#else
        ssize_t r = recv(fd, &c, 1, 0);
#endif
        if (r <= 0) return -1;
        if (c == '\n') break;
        if (c == '\r') continue;
        if (n < max - 1) buf[n++] = c;
    }
    buf[n] = '\0';
    return (int)n;
}

/* 发送一行文本; 返回 -1 表示对端断开 */
static int send_line(int fd, const char* fmt, ...)
{
    char buf[RANK_MAX_LINE];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    /* 帧级版本: 响应 "CMD v=<ver> <args>"(T 流帧不走这里, 保持裸流) */
    char cmd[64];
    int n = 0;
    if (sscanf(buf, "%63s%n", cmd, &n) == 1) {
        const char* p = buf + n;
        while (*p == ' ') p++;
        if (xsend_rank(fd, cmd, strlen(cmd)) != 0) return -1;
        if (xsend_rank(fd, " v=", 3) != 0) return -1;
        if (xsend_rank(fd, PROTO_VERSION_STR, strlen(PROTO_VERSION_STR)) != 0) return -1;
        if (*p) {
            if (xsend_rank(fd, " ", 1) != 0) return -1;
            if (xsend_rank(fd, p, strlen(p)) != 0) return -1;
        }
        return xsend_rank(fd, "\n", 1);
    }
    if (xsend_rank(fd, buf, strlen(buf)) != 0) return -1;
    return xsend_rank(fd, "\n", 1);
}

/* ---- 帧处理 ---- */

static int handle_ping(int fd, Rank* r)
{
    /* P1 单 rank: 加载完成即 READY */
    send_line(fd, "OK READY");
    return 0;
}

static int handle_stat(int fd, Rank* r)
{
    Engine* e = &r->engine;
    uint64_t uptime = r->uptime_s ? (uint64_t)(time(NULL) - (time_t)r->uptime_s) : 0;
    double kv_mb = (double)engine_resident(e) / 1048576.0;
    send_line(fd, "OK inflight=0 kv_mb=%.1f prefix_hits=0 uptime_s=%llu", kv_mb,
              (unsigned long long)uptime);
    return 0;
}

/* 生成回调: 逐 token 流式回 T 帧 */
typedef struct {
    int fd;
    Vocab* vocab;
} TokenCtx;

/* 生成回调: 逐 token 流式回 T 帧; 返回非 0 中止生成(对端断开) */
static int on_token_rank(uint32_t id, void* ctx)
{
    TokenCtx* tc = (TokenCtx*)ctx;
    char tmp[65536];
    vocab_decode(tc->vocab, &id, 1, tmp, sizeof(tmp));
    size_t len = strlen(tmp);
    char hdr[64];
    int hn = snprintf(hdr, sizeof(hdr), "T %zu\n", len);
    if (xsend_rank(tc->fd, hdr, (size_t)hn) != 0) return -1;
    if (xsend_rank(tc->fd, tmp, len) != 0) return -1;
    ylog_raw_log("%s", tmp);
    return 0;
}

static int handle_infer(int fd, Rank* r, char* args)
{
    int max_tokens = 0;
    long nbytes = 0;
    if (sscanf(args, "%d %ld", &max_tokens, &nbytes) != 2 || max_tokens <= 0 || nbytes < 0) {
        send_line(fd, "ERR bad INFER args");
        return 0;
    }
    /* 每请求独立缓冲(线程化后不可共享 r->ids) */
    char* pb = (char*)ymalloc((size_t)nbytes + 8192);
    if (!pb) { send_line(fd, "ERR oom"); return 0; }
    if (xrecv_rank(fd, pb, (size_t)nbytes) != 0) { free(pb); ylog_warn("rank: recv payload FAILED"); return -1; }
    pb[nbytes] = '\0';
    ylog_info("rank: payload=[%s]", pb);
    uint32_t* ids = (uint32_t*)ymalloc((size_t)nbytes + 8192 + 4096);
    if (!ids) { free(pb); send_line(fd, "ERR oom"); return 0; }
    int nprompt;
    if (vocab_has_template(&r->vocab))
        nprompt = vocab_chat_ids(&r->vocab, pb, ids, (int)nbytes + 8192, r->vocab.add_bos);
    else
        nprompt = vocab_encode(&r->vocab, pb, ids, (int)nbytes + 8192);
    free(pb);
    if (nprompt < 0) { free(ids); send_line(fd, "ERR encode failed"); return 0; }

    EngineTimings tim;
    memset(&tim, 0, sizeof(tim));
    char err[512];
    uint64_t t0 = ynow_ms();

    TokenCtx tc;
    tc.fd = fd;
    tc.vocab = &r->vocab;
    /* 引擎单实例: 串行执行推理; 并发 INFER 在此排队, PING/STAT 无需等锁 */
    pthread_mutex_lock(&r->engine_lock);
    int rc = engine_generate(&r->engine, ids, nprompt, max_tokens,
                             r->temp, r->top_p, r->seed, r->vocab.eos,
                             on_token_rank, &tc, &tim, err, sizeof(err));
    ylog_info("rank: generate rc=%d (%d tokens, %llu ms)",
              rc, tim.n_decode, (unsigned long long)(ynow_ms() - t0));
    pthread_mutex_unlock(&r->engine_lock);
    free(ids);
    uint64_t ms = ynow_ms() - t0;
    if (rc != 0) {
        send_line(fd, "ERR generate: %s", err);
        return 0;
    }
    send_line(fd, "DONE %u %d %llu", tim.n_decode, 0, (unsigned long long)ms);
    return 0;
}

static int handle_frame(int fd, Rank* r, const Frame* f)
{
    if (strcmp(f->cmd, PROTO_PING) == 0) return handle_ping(fd, r);
    if (strcmp(f->cmd, PROTO_STAT) == 0) return handle_stat(fd, r);
    if (strcmp(f->cmd, PROTO_INFER) == 0) return handle_infer(fd, r, (char*)f->args);
    if (strcmp(f->cmd, PROTO_DRAIN) == 0) { send_line(fd, "OK"); r->quit = 1; return 2; }
    if (strcmp(f->cmd, PROTO_QUIT) == 0) { send_line(fd, "OK"); r->quit = 1; return 2; }
    send_line(fd, "ERR unknown cmd");
    return 0;
}

/* ---- 主服务循环 ---- */

static Rank* rank_conn_rank;

/* 每连接一线程的处理入口 */
static void* rank_conn(void* arg)
{
    int fd = (int)(intptr_t)arg;
    Rank* r = rank_conn_rank;
    Frame f;
    ylog_info("rank: conn accepted fd=%d", fd);
    int frc = frame_recv(fd, &f);
    if (frc >= 0) {
        handle_frame(fd, r, &f);
    }
    ylog_info("rank: closing conn fd=%d", fd);
    sock_close(fd);
    return NULL;
}

/* 独立心跳线程: 生成长达数分钟也不被 supervisor 误判 DEAD */
static void rank_hb_thread(void* arg)
{
    Rank* r = (Rank*)arg;
    while (!r->quit) {
        sock_sleep_ms(2000);
        if (r->node.sv_enabled) {
            r->node.kv_mb = (double)engine_resident(&r->engine) / 1048576.0;
            node_heartbeat(&r->node);
        }
    }
}

static int run_rank(int port, Rank* r)
{
    ws_init_rank();
    int srv = sock_listen((uint16_t)port, 8);
    if (srv < 0) return 1;
    ylog_info("rank: ready on port %u (model loaded, 常驻等待请求)", port);

    void* hb = NULL;
    if (r->node.sv_enabled) ythread_create(&hb, rank_hb_thread, r);
    for (;;) {
        int fd = sock_accept_with_timeout(srv, 500);
        if (fd >= 0) {
            ylog_info("rank: accept fd=%d", fd);
            /* 每连接一线程: 长生成期间 PING/STAT 仍可即时响应,
             * 并发 INFER 在 engine_lock 排队 */
            pthread_t t;
            if (pthread_create(&t, NULL, rank_conn, (void*)(intptr_t)fd) != 0) {
                ylog_error("rank: pthread_create FAILED, closing fd=%d", fd);
                sock_close(fd);
            } else {
                pthread_detach(t);
            }
        }
        if (r->quit) break;
    }
    r->quit = 1;
    sock_close(srv);
    return 0;
}

/* ---- 参数解析: 统一走 config(main.c 解析) ---- */

#include "config.h"

int cmd_rank(ServeConfig* cfg)
{
    if (!cfg->model[0]) {
        fprintf(stderr, "usage: yllm rank --model <file.llf> [--vocab <file>] [--port N] "
                        "[--supervisor <ip:port>] [--id <name>] "
                        "[--budget-mb N] [--depth N] [--temp F] [--top-p F] [--seed N] "
                        "[--config <yaml>]\n");
        return 1;
    }

    Rank r;
    memset(&r, 0, sizeof(r));
    r.temp = cfg->temp;
    r.top_p = cfg->top_p;
    r.seed = cfg->seed;
    r.uptime_s = (uint64_t)time(NULL);
    if (cfg->node_id[0] && strcmp(cfg->node_id, "node-0") != 0)
        snprintf(r.node.node_id, sizeof(r.node.node_id), "%s", cfg->node_id);
    else
        snprintf(r.node.node_id, sizeof(r.node.node_id), "rank-%s", cfg->model);
    snprintf(r.node.type, sizeof(r.node.type), "rank");
    snprintf(r.node.model, sizeof(r.node.model), "%s", cfg->model);
    r.node.state = NODE_STATE_READY;
    /* addr 上报: 本机真实 IP + 监听端口(心跳携带, supervisor 直接存, server 据此连接) */
    {
        char lip[64] = "127.0.0.1";
        sock_local_ip(lip, sizeof(lip));
        snprintf(r.node.addr, sizeof(r.node.addr), "%s:%u", lip, cfg->rank_port_base);
    }
    if (cfg->sv_host[0]) {
        snprintf(r.node.sv_host, sizeof(r.node.sv_host), "%s", cfg->sv_host);
        r.node.sv_port = (uint16_t)cfg->sv_port;
        r.node.sv_enabled = 1;
    }

    if (vocab_load(cfg->vocab, &r.vocab) != 0) {
        ylog_error("rank: cannot load vocab: %s", cfg->vocab);
        return 1;
    }
    char err[1024];
    uint64_t budget = (uint64_t)cfg->budget_mb * 1024 * 1024;
    if (engine_init(&r.engine, cfg->model, budget, cfg->depth, err, sizeof(err)) != 0) {
        ylog_error("rank: engine init failed: %s", err);
        vocab_free(&r.vocab);
        return 1;
    }
    pthread_mutex_init(&r.engine_lock, NULL);
    rank_conn_rank = &r;

    int rc = run_rank(cfg->rank_port_base, &r);

    engine_free(&r.engine);
    vocab_free(&r.vocab);
    return rc;
}
