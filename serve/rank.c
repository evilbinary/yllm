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

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#define close(fd) closesocket(fd)
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
    uint32_t* ids;
    uint32_t ids_cap;
    float temp;
    float top_p;
    uint64_t seed;
    uint64_t uptime_s;
    /* 统一节点身份(心跳发 supervisor, 生命周期归 supervisor) */
    Node node;
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

static void xsend_rank(int fd, const void* buf, size_t n)
{
    const char* p = (const char*)buf;
    while (n > 0) {
#ifdef _WIN32
        int r = send(fd, p, (int)(n > 0x7fffffff ? 0x7fffffff : n), 0);
#else
        ssize_t r = send(fd, p, n, 0);
#endif
        if (r <= 0) return;
        p += (size_t)r;
        n -= (size_t)r;
    }
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

/* 发送一行文本 */
static void send_line(int fd, const char* fmt, ...)
{
    char buf[RANK_MAX_LINE];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    xsend_rank(fd, buf, strlen(buf));
    xsend_rank(fd, "\n", 1);
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

static void on_token_rank(uint32_t id, void* ctx)
{
    TokenCtx* tc = (TokenCtx*)ctx;
    char tmp[65536];
    vocab_decode(tc->vocab, &id, 1, tmp, sizeof(tmp));
    size_t len = strlen(tmp);
    char hdr[64];
    int hn = snprintf(hdr, sizeof(hdr), "T %zu\n", len);
    xsend_rank(tc->fd, hdr, (size_t)hn);
    xsend_rank(tc->fd, tmp, len);
    ylog_raw("%s", tmp);
}

static int handle_infer(int fd, Rank* r, char* args)
{
    int max_tokens = 0;
    long nbytes = 0;
    if (sscanf(args, "%d %ld", &max_tokens, &nbytes) != 2 || max_tokens <= 0 || nbytes < 0) {
        send_line(fd, "ERR bad INFER args");
        return 0;
    }
    if ((size_t)nbytes >= (size_t)r->ids_cap - 4096) {
        free(r->ids);
        r->ids = (uint32_t*)ymalloc((size_t)nbytes + 8192);
        r->ids_cap = (uint32_t)nbytes + 8192;
    }
    if (xrecv_rank(fd, r->ids, (size_t)nbytes) != 0) return -1;
    r->ids[nbytes] = '\0';
    char* prompt = (char*)r->ids;

    int nprompt = vocab_encode(&r->vocab, prompt, r->ids, (int)r->ids_cap - 4096);
    if (nprompt < 0) { send_line(fd, "ERR encode failed"); return 0; }

    EngineTimings tim;
    memset(&tim, 0, sizeof(tim));
    char err[512];
    uint64_t t0 = ynow_ms();

    TokenCtx tc;
    tc.fd = fd;
    tc.vocab = &r->vocab;
    int rc = engine_generate(&r->engine, r->ids, nprompt, max_tokens,
                             r->temp, r->top_p, r->seed, 1,
                             on_token_rank, &tc, &tim, err, sizeof(err));
    uint64_t ms = ynow_ms() - t0;
    if (rc != 0) {
        send_line(fd, "ERR generate: %s", err);
        return 0;
    }
    send_line(fd, "DONE %u %d %llu", tim.n_decode, 0, (unsigned long long)ms);
    return 0;
}

static int handle_frame(int fd, Rank* r, char* line)
{
    if (strcmp(line, PROTO_PING) == 0) return handle_ping(fd, r);
    if (strcmp(line, PROTO_STAT) == 0) return handle_stat(fd, r);
    if (strncmp(line, PROTO_INFER " ", 6) == 0) return handle_infer(fd, r, line + 6);
    if (strcmp(line, PROTO_DRAIN) == 0) { send_line(fd, "OK"); return 2; }
    if (strcmp(line, PROTO_QUIT) == 0) { send_line(fd, "OK"); return 2; }
    send_line(fd, "ERR unknown cmd");
    return 0;
}

/* ---- 主服务循环 ---- */

static int run_rank(int port, Rank* r)
{
    ws_init_rank();
    int srv = sock_listen((uint16_t)port, 8);
    if (srv < 0) return 1;
    ylog_info("rank: ready on port %u (model loaded, 常驻等待请求)", port);

    uint64_t last_hb = 0;
    for (;;) {
        int fd = sock_accept_with_timeout(srv, 500);
        if (fd >= 0) {
            char line[RANK_MAX_LINE];
            int n = recv_line_rank(fd, line, sizeof(line));
            if (n >= 0) {
                int rc = handle_frame(fd, r, line);
                if (rc == 2) { close(fd); break; }
            }
            close(fd);
        }
        /* 周期心跳 → supervisor(数据面: 活着就报, 判死/重拉归 supervisor) */
        uint64_t now = (uint64_t)time(NULL);
        if (r->node.sv_enabled && now - last_hb >= 2) {
            last_hb = now;
            r->node.kv_mb = (double)engine_resident(&r->engine) / 1048576.0;
            node_heartbeat(&r->node);
        }
    }
    close(srv);
    return 0;
}

/* ---- 参数解析(与 main.c 同款) ---- */

typedef struct { const char* key; const char* val; } ArgR;

static const char* opt_r(ArgR* args, int n, const char* key, const char* def)
{
    int i;
    for (i = 0; i < n; i++)
        if (strcmp(args[i].key, key) == 0) return args[i].val;
    return def;
}

int cmd_rank(int argc, char** argv)
{
    ArgR a[24];
    int n = 0;
    int i;
    for (i = 2; i + 1 < argc && n < 24; i += 2) {
        if (argv[i][0] != '-') break;
        a[n].key = argv[i];
        while (*a[n].key == '-') a[n].key++;
        a[n].val = argv[i + 1];
        n++;
    }
    const char* model = opt_r(a, n, "model", NULL);
    const char* vocab_path = opt_r(a, n, "vocab", "vocab.txt");
    const char* node_id = opt_r(a, n, "id", NULL);
    const char* sv_addr = opt_r(a, n, "supervisor", NULL);
    int port = atoi(opt_r(a, n, "port", "9410"));
    int budget_mb = atoi(opt_r(a, n, "budget-mb", "0"));
    int depth = atoi(opt_r(a, n, "depth", "2"));
    float temp = (float)atof(opt_r(a, n, "temp", "1.0"));
    float top_p = (float)atof(opt_r(a, n, "top-p", "0.9"));
    uint64_t seed = (uint64_t)strtoull(opt_r(a, n, "seed", "42"), NULL, 10);

    if (!model) {
        fprintf(stderr, "usage: yllm rank --model <file.llf> [--vocab <file>] [--port N] "
                        "[--supervisor <ip:port>] [--id <name>] "
                        "[--budget-mb N] [--depth N] [--temp F] [--top-p F] [--seed N]\n");
        return 1;
    }

    Rank r;
    memset(&r, 0, sizeof(r));
    r.temp = temp;
    r.top_p = top_p;
    r.seed = seed;
    r.uptime_s = (uint64_t)time(NULL);
    if (node_id) snprintf(r.node.node_id, sizeof(r.node.node_id), "%s", node_id);
    else snprintf(r.node.node_id, sizeof(r.node.node_id), "rank-%s", model);
    snprintf(r.node.type, sizeof(r.node.type), "rank");
    snprintf(r.node.model, sizeof(r.node.model), "%s", model);
    r.node.state = NODE_STATE_READY;
    if (sv_addr) {
        const char* colon = strchr(sv_addr, ':');
        if (colon) {
            size_t hlen = (size_t)(colon - sv_addr);
            if (hlen >= sizeof(r.node.sv_host)) hlen = sizeof(r.node.sv_host) - 1;
            memcpy(r.node.sv_host, sv_addr, hlen);
            r.node.sv_host[hlen] = '\0';
            r.node.sv_port = (uint16_t)atoi(colon + 1);
            r.node.sv_enabled = 1;
        }
    }

    if (vocab_load(vocab_path, &r.vocab) != 0) {
        ylog_error("rank: cannot load vocab: %s", vocab_path);
        return 1;
    }
    char err[1024];
    uint64_t budget = (uint64_t)budget_mb * 1024 * 1024;
    if (engine_init(&r.engine, model, budget, depth, err, sizeof(err)) != 0) {
        ylog_error("rank: engine init failed: %s", err);
        vocab_free(&r.vocab);
        return 1;
    }
    r.ids = (uint32_t*)ymalloc(4096);
    r.ids_cap = 4096;

    int rc = run_rank(port, &r);

    free(r.ids);
    engine_free(&r.engine);
    vocab_free(&r.vocab);
    return rc;
}
