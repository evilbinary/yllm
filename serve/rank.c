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
    if ((size_t)nbytes >= (size_t)r->ids_cap - 4096) {
        free(r->ids);
        r->ids = (uint32_t*)ymalloc((size_t)nbytes + 8192);
        r->ids_cap = (uint32_t)nbytes + 8192;
    }
    if (xrecv_rank(fd, r->ids, (size_t)nbytes) != 0) return -1;
    ((char*)r->ids)[nbytes] = '\0';
    char* prompt = (char*)r->ids;

    int nprompt;
    if (vocab_has_template(&r->vocab))
        nprompt = vocab_chat_ids(&r->vocab, prompt, r->ids, (int)r->ids_cap - 4096, r->vocab.add_bos);
    else
        nprompt = vocab_encode(&r->vocab, prompt, r->ids, (int)r->ids_cap - 4096);
    if (nprompt < 0) { send_line(fd, "ERR encode failed"); return 0; }

    EngineTimings tim;
    memset(&tim, 0, sizeof(tim));
    char err[512];
    uint64_t t0 = ynow_ms();

    TokenCtx tc;
    tc.fd = fd;
    tc.vocab = &r->vocab;
    int rc = engine_generate(&r->engine, r->ids, nprompt, max_tokens,
                             r->temp, r->top_p, r->seed, r->vocab.eos,
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
    snprintf(r.node.addr, sizeof(r.node.addr), "%s:%u", cfg->sv_host, cfg->rank_port_base);
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
    r.ids = (uint32_t*)ymalloc(4096);
    r.ids_cap = 4096;

    int rc = run_rank(cfg->rank_port_base, &r);

    free(r.ids);
    engine_free(&r.engine);
    vocab_free(&r.vocab);
    return rc;
}
