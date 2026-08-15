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
#include "../inference/dist.h"
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
    /* 组内协作(多段流水线) */
    int dist_rank;       /* 段号 0..ranks-1 */
    int dist_ranks;      /* 总段数(1 = 单机, 不走协作) */
    uint16_t pipe_base;  /* 协作口基址 = rank_port_base + RANK_PIPE_OFFSET */
    char addrs_csv[1024];/* 组内各段节点 IP(逗号分隔, 段号顺序) */
    int dist_fp16;
    int serve_port;      /* 本段 serve 口(worker 段也监听, 供 PING/STAT/DRAIN 管理) */
    /* 会话模式最小记账: 引擎 KV 归属的会话 + 已推进位置 */
    char sess_key[64];
    uint32_t sess_pos;
} Rank;

#define RANK_PIPE_OFFSET 100 /* 协作口基址偏移(避开 serve/server 端口区段) */

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
    int n_tokens;
} TokenCtx;

/* 生成回调: 逐 token 流式回 T 帧; 返回非 0 中止生成(对端断开) */
static int on_token_rank(uint32_t id, void* ctx)
{
    TokenCtx* tc = (TokenCtx*)ctx;
    tc->n_tokens++;
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

/* 会话模式推理: payload = 增量 token 二进制(server 已渲染并算好续接点)。
 * rank 只做最小记账: sess_key/sess_pos; token 列表/历史在 server 侧管理。 */
static int handle_infer_session(int fd, Rank* r, const char* key, uint32_t max_tokens,
                                uint32_t resume, const uint32_t* delta, uint32_t ndelta)
{
    EngineTimings tim;
    memset(&tim, 0, sizeof(tim));
    uint64_t t0 = ynow_ms();
    TokenCtx tc;
    tc.fd = fd;
    tc.vocab = &r->vocab;
    tc.n_tokens = 0;

    /* 换会话: 引擎 KV 状态未知, 必须从 0 全量(server 应发全量历史) */
    if (strcmp(key, r->sess_key) != 0) {
        if (resume != 0) {
            send_line(fd, "ERR session not cached on this rank, resume must be 0");
            return 0;
        }
        snprintf(r->sess_key, sizeof(r->sess_key), "%s", key);
        r->sess_pos = 0;
    }
    if (resume != r->sess_pos) {
        send_line(fd, "ERR resume mismatch: rank has %u, got %u", r->sess_pos, resume);
        return 0;
    }
    if ((uint64_t)r->sess_pos + ndelta > r->engine.max_seq) {
        send_line(fd, "ERR context full");
        return 0;
    }

    r->node.state = NODE_STATE_BUSY;
    r->node.inflight++;
    node_heartbeat(&r->node);
    pthread_mutex_lock(&r->engine_lock);

    /* 增量 prefill: 旧 token 的 KV 已在本引擎缓存, 只算新 token */
    if (ndelta > 0)
        engine_forward_prefill(&r->engine, delta, (int)ndelta, r->sess_pos);
    r->sess_pos += ndelta;
    tim.n_prefill = ndelta;

    /* decode 循环: 采样 → 流式回 → 前向推进(与 engine_generate 同构) */
    uint64_t rng = ysrand(r->seed);
    uint32_t ngen = 0;
    uint32_t pos = r->sess_pos;
    for (uint32_t i = 0; i < max_tokens && pos < r->engine.max_seq; i++) {
        uint32_t nxt;
        if (engine_sample(&r->engine, r->engine.ws.model.h.vocab, r->temp, r->top_p, &rng, &nxt) != 0) break;
        if ((int)nxt == r->vocab.eos) break;
        if (on_token_rank(nxt, &tc) != 0) break;
        engine_forward(&r->engine, nxt, pos);
        pos++;
        ngen++;
    }
    r->sess_pos = pos;
    tim.n_decode = ngen;
    tim.decode_ms = ynow_ms() - t0;

    ylog_info("rank: session=%s generate ok (%u delta + %u gen tokens, %llu ms)",
              key, ndelta, ngen, (unsigned long long)(ynow_ms() - t0));
    pthread_mutex_unlock(&r->engine_lock);
    r->node.state = NODE_STATE_READY;
    r->node.inflight--;
    node_heartbeat(&r->node);

    send_line(fd, "DONE %u %u %llu", tc.n_tokens, 0, (unsigned long long)(ynow_ms() - t0));
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
    /* 组内 rank 信息随请求携带(server 从 sv 租用获得): 按数据办事 */
    {
        const char* seg = proto_get(args, "seg");
        const char* segs = proto_get(args, "segs");
        const char* peers = proto_get(args, "peers");
        if (segs && atoi(segs) > 1) {
            r->dist_ranks = atoi(segs);
            if (seg) r->dist_rank = atoi(seg);
            if (peers) snprintf(r->addrs_csv, sizeof(r->addrs_csv), "%s", peers);
        }
    }
    /* 每请求独立缓冲(线程化后不可共享 r->ids) */
    char* pb = (char*)ymalloc((size_t)nbytes + 8192);
    if (!pb) { send_line(fd, "ERR oom"); return 0; }
    if (xrecv_rank(fd, pb, (size_t)nbytes) != 0) { free(pb); ylog_warn("rank: recv payload FAILED"); return -1; }
    pb[nbytes] = '\0';
    /* 会话模式: 带 key= 字段时 payload 为增量 token 二进制(server 已渲染, 不 tokenize) */
    {
        const char* key = proto_get(args, "key");
        if (key) {
            const char* rs = proto_get(args, "resume");
            uint32_t resume = rs ? (uint32_t)strtoul(rs, NULL, 10) : 0;
            if (nbytes % 4 != 0) { free(pb); send_line(fd, "ERR session payload must be token bytes"); return 0; }
            uint32_t ndelta = (uint32_t)(nbytes / 4);
            int rc = handle_infer_session(fd, r, key, (uint32_t)max_tokens, resume,
                                          (const uint32_t*)pb, ndelta);
            free(pb);
            return rc;
        }
    }
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
    tc.n_tokens = 0;
    /* 通知 supervisor: 本 rank 进入 BUSY(推理中), 供调度/状态展示 */
    r->node.state = NODE_STATE_BUSY;
    r->node.inflight++;
    node_heartbeat(&r->node);
    /* 引擎单实例: 串行执行推理; 并发 INFER 在此排队, PING/STAT 无需等锁 */
    pthread_mutex_lock(&r->engine_lock);
    int rc;
    if (r->dist_ranks > 1) {
        /* 多段流水线: rank0 为 master, 与组内兄弟协作(兄弟段在各自进程等激活) */
        rc = dist_gen(&r->engine, &r->vocab, ids, nprompt, max_tokens,
                      r->temp, r->top_p, r->seed,
                      r->dist_rank, r->dist_ranks, r->pipe_base, r->addrs_csv, r->dist_fp16,
                      t0, on_token_rank, &tc);
    } else {
        rc = engine_generate(&r->engine, ids, nprompt, max_tokens,
                             r->temp, r->top_p, r->seed, r->vocab.eos,
                             on_token_rank, &tc, &tim, err, sizeof(err));
    }
    ylog_info("rank: generate rc=%d (%d tokens, %llu ms)",
              rc, tc.n_tokens, (unsigned long long)(ynow_ms() - t0));
    pthread_mutex_unlock(&r->engine_lock);
    /* 推理结束: 恢复 READY 并通知 supervisor */
    r->node.state = NODE_STATE_READY;
    r->node.inflight--;
    node_heartbeat(&r->node);
    free(ids);
    uint64_t ms = ynow_ms() - t0;
    if (rc != 0) {
        if (r->dist_ranks > 1)
            send_line(fd, "ERR generate: dist pipeline failed");
        else
            send_line(fd, "ERR generate: %s", err);
        return 0;
    }
    send_line(fd, "DONE %u %d %llu", tc.n_tokens, 0, (unsigned long long)ms);
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

/* ---- 主循环 ---- */

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

/* worker 段心跳线程: 管理口(PING/STAT/DRAIN/QUIT)优先 accept + 每 2s 心跳。
 * worker 主线程阻塞在 dist 等激活, 管理口由本线程承载, 进程内仍仅 2 线程。 */
static void worker_hb_thread(void* arg)
{
    Rank* r = (Rank*)arg;
    int srv = r->serve_port > 0 ? sock_listen((uint16_t)r->serve_port, 8) : -1;
    if (srv >= 0)
        ylog_info("rank: worker serve port %u up (管理口)", r->serve_port);
    uint64_t last_hb = 0;
    while (!r->quit) {
        /* 管理口优先(短超时 accept, 保证 PING/STAT 及时响应) */
        if (srv >= 0) {
            int i;
            for (i = 0; i < 4 && !r->quit; i++) {
                int fd = sock_accept_with_timeout(srv, 200);
                if (fd < 0) break;
                Frame f;
                if (frame_recv(fd, &f) >= 0)
                    handle_frame(fd, r, &f);
                sock_close(fd);
            }
        }
        /* 心跳节律(2s, 用时间判断, 不被管理口阻塞) */
        uint64_t now = (uint64_t)time(NULL);
        if (r->node.sv_enabled && now - last_hb >= 2) {
            last_hb = now;
            r->node.kv_mb = (double)engine_resident(&r->engine) / 1048576.0;
            node_heartbeat(&r->node);
        }
        if (!r->quit) sock_sleep_ms(50);
    }
    if (srv >= 0) sock_close(srv);
}

/* 主循环: worker 段(rank>0) = dist 等激活循环(serve 管理口由心跳线程承载);
 * rank0 = 监听 serve 口, 串行处理连接(推理是一个过程, 不并发)。
 * 进程内仅主线程 + 心跳线程。 */
static int run_rank(int port, Rank* r)
{
    ws_init_rank();
    void* hb = NULL;
    if (r->node.sv_enabled) ythread_create(&hb, rank_hb_thread, r);

    if (r->dist_rank > 0 && r->dist_ranks > 1) {
        if (r->node.sv_enabled) ythread_create(&hb, worker_hb_thread, r);
        while (!r->quit) {
            int rc = dist_gen(&r->engine, &r->vocab, NULL, 0, 0,
                              r->temp, r->top_p, r->seed,
                              r->dist_rank, r->dist_ranks, r->pipe_base, r->addrs_csv, r->dist_fp16,
                              ynow_ms(), NULL, NULL);
            if (r->quit) break;
            if (rc != 0) {
                ylog_warn("rank: pipeline round failed (rc=%d), retry", rc);
                sock_sleep_ms(1000);
            }
        }
    } else {
        int srv = sock_listen((uint16_t)port, 8);
        if (srv < 0) { r->quit = 1; if (hb) ythread_join(&hb); return 1; }
        ylog_info("rank: ready on port %u (model loaded, 常驻等待请求)", port);
        while (!r->quit) {
            int fd = sock_accept_with_timeout(srv, 500);
            if (fd >= 0) {
                ylog_info("rank: accept fd=%d", fd);
                Frame f;
                if (frame_recv(fd, &f) >= 0)
                    handle_frame(fd, r, &f);
                ylog_info("rank: closing conn fd=%d", fd);
                sock_close(fd);
            }
        }
        sock_close(srv);
    }
    r->quit = 1;
    if (hb) ythread_join(&hb);
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

    sock_init(); /* sock_local_ip(取本机 IP)依赖 winsock, 必须先初始化 */

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
    /* model 显示注册名(router/server 用), 缺省取路径 basename */
    if (cfg->model_name[0] && strcmp(cfg->model_name, "default") != 0) {
        snprintf(r.node.model, sizeof(r.node.model), "%s", cfg->model_name);
    } else {
        const char* base = strrchr(cfg->model, '/');
        base = base ? base + 1 : cfg->model;
        const char* dot = strrchr(base, '.');
        if (dot && dot != base) {
            size_t blen = (size_t)(dot - base);
            if (blen >= sizeof(r.node.model)) blen = sizeof(r.node.model) - 1;
            memcpy(r.node.model, base, blen);
            r.node.model[blen] = '\0';
        } else {
            snprintf(r.node.model, sizeof(r.node.model), "%s", base);
        }
    }
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

    /* 多段协作: 协作口基址 = serve 基准(自己的 --port - 段号) + 偏移, 全组统一;
     * 成员地址: worker 段来自命令(--peers, sv 自动下发); rank0 以 INFER 数据为准(启动时的兜底) */
    r.dist_rank = cfg->rank_idx;
    r.dist_ranks = cfg->ranks > 1 ? cfg->ranks : 1;
    r.pipe_base = (uint16_t)(cfg->rank_port_base - r.dist_rank + RANK_PIPE_OFFSET);
    r.serve_port = cfg->rank_port_base;
    if (cfg->peers[0])
        snprintf(r.addrs_csv, sizeof(r.addrs_csv), "%s", cfg->peers);
    if (r.dist_ranks > 1) {
        if (dist_split_layers(&r.engine, r.dist_rank, r.dist_ranks) != 0) {
            ylog_error("rank: dist_split_layers failed (seg %d/%d)", r.dist_rank, r.dist_ranks);
            engine_free(&r.engine);
            vocab_free(&r.vocab);
            return 1;
        }
    }

    pthread_mutex_init(&r.engine_lock, NULL);

    int rc = run_rank(cfg->rank_port_base, &r);

    engine_free(&r.engine);
    vocab_free(&r.vocab);
    return rc;
}
