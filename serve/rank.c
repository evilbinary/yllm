/* rank.c — yllm rank: 常驻推理单元
 *
 * 公用 rank 池的最小计算单元: 加载模型层段权重常驻,
 * 监听推理端口, 循环处理服务层帧(PING/STAT/INFER/DRAIN/QUIT)。
 *
 * 用法:
 *   yllm rank --model <file.llf> --vocab <file> [--port N]
 *             [--budget auto|NMB|NG] [--depth N] [--temp F] [--top-p F] [--seed N]
 *             [--log <file>] [--log-level lvl] [--no-console]
 *
 * P1 阶段: 单 rank(ranks=1, 完整模型本地推理)。
 * 多 rank 流水线(dist_split_layers + dist 收发)在 P2 接入。
 */
#include "protocol.h"
#include "frame.h"
#include "node.h"
#include "sock.h"
#include "../inference/include/yllm.h"
#include "../inference/include/cache.h"
#include "../inference/include/dist.h"
#include "../inference/include/log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <errno.h>
#include <pthread.h>
#include <stdint.h>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <sys/stat.h>   /* rank_auto_budget 用 stat()/struct stat */
#define ssize_t int
#else
#include <unistd.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <netinet/in.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <netinet/tcp.h>
#include <arpa/inet.h>
#endif

#define RANK_MAX_LINE 8192
#define RANK_JOB_CAP 8
#define RANK_WORK_MAX 8

typedef struct {
    int fd;
    char args[256];
    char* payload;
    long nbytes;
} RankJob;

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
    volatile int quit;   /* QUIT/DRAIN 后主循环退出, 心跳线程一并退出 */
    /* 组内协作(多段流水线) */
    int dist_rank;       /* 段号 0..ranks-1 */
    int dist_ranks;      /* 总段数(1 = 单机, 不走协作) */
    uint16_t pipe_base;  /* 协作口基址 = rank_port_base + RANK_PIPE_OFFSET */
    char addrs_csv[1024];/* 组内各段节点 IP(逗号分隔, 段号顺序) */
    int dist_fp16;
    int serve_port;      /* 本段 serve 口(IO 线程独占 listen) */
    /* 会话模式最小记账: 引擎 KV 归属的会话 + 已推进位置 */
    char cache_key[64];
    uint32_t cache_pos;
    char cache_dir[256];   /* KV 落盘目录(空 = 纯内存) */
    /* IO 分发 + work 队列 */
    int work_mode;         /* 0=serial 1=parallel */
    int work_threads;      /* serial→1; parallel→N */
    pthread_mutex_t q_mu;
    pthread_cond_t q_cv;
    RankJob jobs[RANK_JOB_CAP];
    int q_head, q_tail, q_len;
    int in_flight;
    int draining;
    int drain_fd;          /* DRAIN 连接: 排空后再回 OK; -1=无 */
    volatile int io_stop;  /* IO 线程退出(排空完成或 listen 失败) */
    double kv_mb_cache;    /* hb 刷新, STAT 免抢 engine */
    uint32_t job_need;     /* 本轮 prompt/delta token 数; 0=无进行中作业 */
    DistLive job;          /* pos / prefill|decode / tok/s */
} Rank;

static void rank_job_begin(Rank* r, uint32_t start, uint32_t need, uint64_t t0)
{
    r->job_need = need;
    r->job.start = start;
    r->job.pos = start;
    r->job.phase = 1;
    r->job.t0 = t0;
    r->job.dec_t0 = 0;
    r->job.dec_start = start;
    r->job.pf_tps = 0;
    r->job.dec_tps = 0;
}

static void rank_job_end(Rank* r)
{
    DistLive* L = &r->job;
    uint64_t now = ynow_ms();
    if (L->phase == 1 && L->t0) {
        uint32_t n = L->pos >= L->start ? L->pos - L->start : 0;
        uint64_t ms = now > L->t0 ? now - L->t0 : 1;
        if (n) L->pf_tps = (float)((double)n * 1000.0 / (double)ms);
    } else if (L->phase == 2 && L->dec_t0) {
        uint32_t n = L->pos >= L->dec_start ? L->pos - L->dec_start : 0;
        uint64_t ms = now > L->dec_t0 ? now - L->dec_t0 : 1;
        if (n) L->dec_tps = (float)((double)n * 1000.0 / (double)ms);
    }
    L->phase = 0;
    r->job_need = 0;
}

#define RANK_PIPE_OFFSET 100 /* 协作口基址偏移(避开 serve/server 端口区段) */

static int rank_ensure_peers(Rank* r);
static int rank_query_peers(Rank* r, char* out, size_t outsz);

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
    int omp_n = 1;
#ifdef _OPENMP
    omp_n = omp_get_max_threads();
    if (omp_n < 1) omp_n = 1;
#endif
    int inflight, queued, mode, nwork;
    pthread_mutex_lock(&r->q_mu);
    inflight = r->in_flight;
    queued = r->q_len;
    mode = r->work_mode;
    nwork = r->work_threads;
    pthread_mutex_unlock(&r->q_mu);
    double kv_mb = r->kv_mb_cache;
    if (kv_mb <= 0.0)
        kv_mb = (double)engine_resident(e) / 1048576.0;
    uint32_t jneed = r->job_need;
    DistLive L = r->job;
    uint64_t now = ynow_ms();
    uint64_t jms = (L.phase && L.t0) ? (now - L.t0) : 0;
    float pf = L.pf_tps, dc = L.dec_tps;
    const char* phase = "-";
    if (L.phase == 1) {
        phase = "prefill";
        {
            uint32_t n = L.pos >= L.start ? L.pos - L.start : 0;
            uint64_t ms = now > L.t0 && L.t0 ? now - L.t0 : 1;
            if (n) pf = (float)((double)n * 1000.0 / (double)ms);
        }
        dc = 0;
    } else if (L.phase == 2) {
        phase = "decode";
        if (L.dec_t0) {
            uint32_t n = L.pos >= L.dec_start ? L.pos - L.dec_start : 0;
            uint64_t ms = now > L.dec_t0 ? now - L.dec_t0 : 1;
            if (n) dc = (float)((double)n * 1000.0 / (double)ms);
        }
    }
    send_line(fd,
              "OK inflight=%d queued=%d work=%s threads=%d kv_mb=%.1f prefix_hits=0 "
              "uptime_s=%llu layers[%u,%u) omp=%d job_pos=%u job_need=%u job_ms=%llu "
              "job_phase=%s job_pf_tps=%.2f job_dec_tps=%.2f",
              inflight, queued, mode ? "parallel" : "serial", nwork, kv_mb,
              (unsigned long long)uptime, e->layer_begin, e->layer_end, omp_n,
              L.pos, jneed, (unsigned long long)jms, phase, pf, dc);
    return 0;
}

/* 生成回调: 逐 token 流式回 T 帧 */
typedef struct {
    int fd;
    Vocab* vocab;
    int n_tokens;
    int cache_frame;   /* 1 = 会话模式: 发 TS 帧(带 token id, 供 router 记账) */
} TokenCtx;

/* 生成回调: 逐 token 流式回 T 帧; 返回非 0 中止生成(对端断开) */
static int on_token_rank(uint32_t id, void* ctx)
{
    TokenCtx* tc = (TokenCtx*)ctx;
    tc->n_tokens++;
    char tmp[256];
    vocab_decode(tc->vocab, &id, 1, tmp, sizeof(tmp));
    size_t len = strlen(tmp);
    if (len + 32 > sizeof(tmp)) len = sizeof(tmp) - 32;
    char hdr[64];
    int hn = tc->cache_frame
        ? snprintf(hdr, sizeof(hdr), "TS %zu %u\n", len, id)
        : snprintf(hdr, sizeof(hdr), "T %zu\n", len);
    /* 合并头+payload 为一次 send(减少系统调用) */
    char one[320];
    memcpy(one, hdr, (size_t)hn);
    memcpy(one + hn, tmp, len);
    if (xsend_rank(tc->fd, one, (size_t)hn + len) != 0) return -1;
    ylog_raw_log("%s", tmp);
    return 0;
}

/* 会话模式推理: payload = 增量 token 二进制(server 已渲染并算好续接点)。
 * rank 只做最小记账: cache_key/cache_pos; token 列表/历史在 server 侧管理。 */
/* 缓存文件名安全化(与 server 一致): 替换 ':' 等文件系统非法字符 */
static void cache_file_name(char* out, size_t outsz, const char* key, const char* ext)
{
    size_t o = 0;
    const char* p;
    for (p = key; *p && o + 16 < outsz; p++) {
        char c = *p;
        if (c == ':' || c == '/' || c == '\\' || c == '?' || c == '*' ||
            c == '<' || c == '>' || c == '|' || c == '"')
            c = '_';
        out[o++] = c;
    }
    if (o + strlen(ext) + 1 < outsz) memcpy(out + o, ext, strlen(ext) + 1);
}

/* 本 rank 的会话 kv 路径: <dir>/<安全化key>.r<rank>.kv(PP 各段不互相覆盖) */
static void cache_kv_path(char* out, size_t outsz, const char* dir, const char* key, int rank)
{
    char base[512];
    char ext[32];
    snprintf(ext, sizeof(ext), ".r%d.kv", rank);
    cache_file_name(base, sizeof(base), key, ext);
    if (dir && dir[0])
        snprintf(out, outsz, "%s/%s", dir, base);
    else
        snprintf(out, outsz, "%s", base);
}

/* 每轮推理成功后落盘: 重启后续接依赖与 .sess 对齐的 .rN.kv */
static void rank_save_sess_kv(Rank* r)
{
    if (!r->cache_dir[0] || !r->cache_key[0] || r->cache_pos == 0) return;
    char path[512];
    cache_kv_path(path, sizeof(path), r->cache_dir, r->cache_key, r->dist_rank);
    if (sess_kv_save(&r->engine, r->cache_pos, path) == 0)
        ylog_info("rank: session %s kv saved (%u tokens) -> %s", r->cache_key, r->cache_pos, path);
}

static void rank_unlink_sess_kv(Rank* r, const char* key)
{
    if (!r->cache_dir[0] || !key || !key[0]) return;
    char path[512];
    cache_kv_path(path, sizeof(path), r->cache_dir, key, r->dist_rank);
    if (remove(path) == 0)
        ylog_info("rank: stale kv removed %s", path);
}

static int handle_infer_cache(int fd, Rank* r, const char* key, uint32_t max_tokens,
                                uint32_t resume, const uint32_t* delta, uint32_t ndelta)
{
    EngineTimings tim;
    memset(&tim, 0, sizeof(tim));
    uint64_t t0 = ynow_ms();
    TokenCtx tc;
    tc.fd = fd;
    tc.vocab = &r->vocab;
    tc.n_tokens = 0;
    tc.cache_frame = 1;

    /* 换会话: 先把旧会话 KV 落盘, 再尝试从磁盘恢复新会话 KV */
#if YLLM_SESS_DEBUG
    ylog_info("rank: sess key=[%s] cur=[%s] resume=%u cur_pos=%u ndelta=%u",
              key, r->cache_key, resume, r->cache_pos, ndelta);
#endif
    if (strcmp(key, r->cache_key) != 0) {
        if (r->cache_dir[0] && r->cache_key[0]) {
            char path[512];
            cache_kv_path(path, sizeof(path), r->cache_dir, r->cache_key, r->dist_rank);
            if (sess_kv_save(&r->engine, r->cache_pos, path) == 0)
                ylog_info("rank: session %s kv saved (%u tokens) -> %s", r->cache_key, r->cache_pos, path);
        }
        snprintf(r->cache_key, sizeof(r->cache_key), "%s", key);
        r->cache_pos = 0;
        /* 磁盘恢复: 载入 <dir>/<key>.kv, resume 需与载入的 pos 一致 */
        if (r->cache_dir[0] && resume != 0) {
            char path[512];
            cache_kv_path(path, sizeof(path), r->cache_dir, key, r->dist_rank);
            uint32_t loaded = 0;
            if (sess_kv_load(&r->engine, path, &loaded) == 0 && loaded == resume) {
                r->cache_pos = loaded;
                ylog_info("rank: session %s kv restored (%u tokens) from %s", key, loaded, path);
            } else {
                /* 坏/过期 kv: 删掉, 让 server 走 resume=0 全量重建(同会话, 无需用户新开) */
                ylog_warn("rank: session %s kv load failed/pos mismatch (%u vs %u), drop kv",
                          key, loaded, resume);
                rank_unlink_sess_kv(r, key);
                r->cache_pos = 0;
                send_line(fd, PROTO_ERROR " session kv not cached on this rank, resume must be 0");
                return 0;
            }
        } else if (resume != 0) {
            ylog_warn("rank: sess key mismatch, resume must be 0");
            send_line(fd, PROTO_ERROR " session not cached on this rank, resume must be 0");
            return 0;
        }
    }
    if (resume != r->cache_pos) {
        if (resume == 0) {
            /* 全量重发: 以 0 为基, KV 由 X 流从 pos=0 覆盖 */
            ylog_warn("rank: full resend (pos %u -> 0)", r->cache_pos);
            r->cache_pos = 0;
        } else {
            {
                uint32_t have = r->cache_pos;
                ylog_warn("rank: resume mismatch: rank has %u, got %u; drop kv", have, resume);
                rank_unlink_sess_kv(r, key);
                r->cache_pos = 0;
                send_line(fd, PROTO_ERROR " resume mismatch: rank has %u, got %u", have, resume);
                return 0;
            }
        }
    }
    if ((uint64_t)r->cache_pos + ndelta > r->engine.max_seq) {
        send_line(fd, PROTO_ERROR " context full");
        return 0;
    }

    pthread_mutex_lock(&r->engine_lock);

    tim.n_prefill = ndelta;

    if (r->dist_ranks > 1) {
        /* PP 流水线: 各段各自处理自己的层段; master 走 dist_gen(带会话握手) */
        DistSess sess;
        memset(&sess, 0, sizeof(sess));
        snprintf(sess.key, sizeof(sess.key), "%s", key);
        sess.pos = resume;                 /* master 从续接点开始 */
        sess.my_pos = r->cache_pos;
        sess.cache_dir = r->cache_dir[0] ? r->cache_dir : NULL;
        rank_job_begin(r, resume, ndelta, t0);
        sess.live = &r->job;
        int rc2 = dist_gen(&r->engine, &r->vocab, delta, (int)ndelta, (int)max_tokens,
                           r->temp, r->top_p, r->seed,
                           r->dist_rank, r->dist_ranks, r->pipe_base, r->addrs_csv, r->dist_fp16,
                           t0, on_token_rank, &tc, &sess);
        r->cache_pos = sess.pos;
        r->job.pos = sess.pos;
        rank_job_end(r);
        tim.n_decode = (uint32_t)tc.n_tokens;
        tim.decode_ms = ynow_ms() - t0;
        ylog_info("rank: sess %s pp done rc=%d (resume=%u end=%u, %d tokens, %llu ms)",
                  key, rc2, resume, sess.pos, tc.n_tokens,
                  (unsigned long long)(ynow_ms() - t0));
        pthread_mutex_unlock(&r->engine_lock);
        if (rc2 != 0) {
            /* 失败不得伪装成功: server 收到 ERR 后走全量重发/退避重试 */
            send_line(fd, PROTO_ERROR " dist generate failed");
            return 0;
        }
        rank_save_sess_kv(r);
        send_line(fd, PROTO_DONE " %u %u %llu", tc.n_tokens, 0, (unsigned long long)(ynow_ms() - t0));
        return 0;
    }

    /* 增量 prefill: 旧 token 的 KV 已在本引擎缓存, 只算新 token */
    uint64_t t_prefill = ynow_ms();
    rank_job_begin(r, r->cache_pos, ndelta, t0);
    if (ndelta > 0)
        engine_forward_prefill(&r->engine, delta, (int)ndelta, r->cache_pos);
    r->cache_pos += ndelta;
    r->job.pos = r->cache_pos;
    if (ndelta > 0)
        ylog_info("prefill: %u tokens in %.2f s (%.2f tok/s)", ndelta,
               (double)(ynow_ms() - t_prefill) / 1000.0,
               (double)ndelta * 1000.0 / (double)(ynow_ms() - t_prefill > 0 ? ynow_ms() - t_prefill : 1));
    {
        uint64_t now = ynow_ms();
        uint64_t pms = now > t_prefill ? now - t_prefill : 1;
        if (ndelta)
            r->job.pf_tps = (float)((double)ndelta * 1000.0 / (double)pms);
        r->job.dec_t0 = now;
        r->job.dec_start = r->cache_pos;
        r->job.phase = 2;
    }

    /* decode 循环: 采样 → 流式回 → 前向推进(与 engine_generate 同构) */
    uint64_t t_dec0 = ynow_ms();
    uint64_t rng = ysrand(r->seed);
    uint32_t ngen = 0;
    uint32_t pos = r->cache_pos;
    for (uint32_t i = 0; i < max_tokens && pos < r->engine.max_seq; i++) {
        uint32_t nxt;
        if (engine_sample(&r->engine, r->engine.ws.model.h.vocab, r->temp, r->top_p, &rng, &nxt) != 0) break;
        if ((int)nxt == r->vocab.eos) break;
        if (on_token_rank(nxt, &tc) != 0) break;
        engine_forward(&r->engine, nxt, pos);
        pos++;
        ngen++;
        r->job.pos = pos;
    }
    /* 结束(命中 eos 或达上限)后补 eos 到 kv, 使 rank pos 与 router 缓存(回复+eos)一致 */
    if (pos < r->engine.max_seq) {
        engine_forward(&r->engine, (uint32_t)r->vocab.eos, pos);
        pos++;
    }
    r->cache_pos = pos;
    r->job.pos = pos;
    rank_job_end(r);
    tim.n_decode = ngen;
    tim.decode_ms = ynow_ms() - t_dec0;

    if (ngen > 0) ylog_raw_log("\n");   /* 生成的最后一个 token 后换行, 避免与统计日志挤在同一行 */
    ylog_info("decode:  %u tokens in %.2f s (%.2f tok/s)", ngen,
           (double)(ynow_ms() - t_dec0) / 1000.0,
           (double)ngen * 1000.0 / (double)(ynow_ms() - t_dec0 > 0 ? ynow_ms() - t_dec0 : 1));
    ylog_info("rank: session=%s generate ok (%u delta + %u gen tokens, %llu ms)",
              key, ndelta, ngen, (unsigned long long)(ynow_ms() - t0));
    rank_save_sess_kv(r);
    pthread_mutex_unlock(&r->engine_lock);

    send_line(fd, PROTO_DONE " %u %u %llu", tc.n_tokens, 0, (unsigned long long)(ynow_ms() - t0));
    return 0;
}

static int handle_infer(int fd, Rank* r, char* args, char* pb, long nbytes)
{
    int max_tokens = 0;
    long nbytes_arg = 0;
    if (sscanf(args, "%d %ld", &max_tokens, &nbytes_arg) != 2 || max_tokens <= 0 || nbytes_arg < 0) {
        send_line(fd, PROTO_ERROR " bad INFER args");
        free(pb);
        return 0;
    }
    if (nbytes != nbytes_arg) {
        send_line(fd, PROTO_ERROR " payload size mismatch");
        free(pb);
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
        /* 采样参数随请求覆盖(HTTP temperature/top_p 透传) */
        const char* tp = proto_get(args, "temp");
        const char* pp = proto_get(args, "top_p");
        if (tp) r->temp = (float)atof(tp);
        if (pp) r->top_p = (float)atof(pp);
    }
    if (!pb) { send_line(fd, PROTO_ERROR " oom"); return 0; }
    /* 会话模式: 带 key= 字段时 payload 为增量 token 二进制(server 已渲染, 不 tokenize) */
    {
        const char* key = proto_get(args, "key");
        if (key) {
            /* proto_get 返回行内剩余部分, 截到空格 */
            char keybuf[64];
            size_t kl = strcspn(key, " ");
            if (kl >= sizeof(keybuf)) kl = sizeof(keybuf) - 1;
            memcpy(keybuf, key, kl);
            keybuf[kl] = '\0';
            key = keybuf;
            const char* rs = proto_get(args, "resume");
            uint32_t resume = rs ? (uint32_t)strtoul(rs, NULL, 10) : 0;
#if YLLM_SESS_DEBUG
            ylog_info("rank: sess INFER key=[%s] resume=%u nbytes=%ld", key, resume, nbytes);
#endif
            if (nbytes % 4 != 0) { free(pb); send_line(fd, PROTO_ERROR " session payload must be token bytes"); return 0; }
            uint32_t ndelta = (uint32_t)(nbytes / 4);
            int rc = handle_infer_cache(fd, r, key, (uint32_t)max_tokens, resume,
                                          (const uint32_t*)pb, ndelta);
            free(pb);
            return rc;
        }
    }
    ylog_info("rank: payload=[%s]", pb);
    uint32_t* ids = (uint32_t*)ymalloc((size_t)nbytes + 8192 + 4096);
    if (!ids) { free(pb); send_line(fd, PROTO_ERROR " oom"); return 0; }
    int nprompt;
    if (vocab_has_template(&r->vocab))
        nprompt = vocab_chat_ids(&r->vocab, pb, ids, (int)nbytes + 8192, r->vocab.add_bos);
    else
        nprompt = vocab_encode(&r->vocab, pb, ids, (int)nbytes + 8192);
    free(pb);
    if (nprompt < 0) { free(ids); send_line(fd, PROTO_ERROR " encode failed"); return 0; }

    EngineTimings tim;
    memset(&tim, 0, sizeof(tim));
    char err[512];
    uint64_t t0 = ynow_ms();

    TokenCtx tc;
    tc.fd = fd;
    tc.vocab = &r->vocab;
    tc.n_tokens = 0;
    tc.cache_frame = 0;   /* 普通 INFER: 发 T 帧(server/route 只认 "T " 前缀) */
    /* 引擎单实例: 串行执行推理; 并发 INFER 在此排队, PING/STAT 无需等锁 */
    pthread_mutex_lock(&r->engine_lock);
    int rc;
    if (r->dist_ranks > 1) {
        /* 多段流水线: rank0 为 master, 与组内兄弟协作(兄弟段在各自进程等激活) */
        rc = dist_gen(&r->engine, &r->vocab, ids, nprompt, max_tokens,
                      r->temp, r->top_p, r->seed,
                      r->dist_rank, r->dist_ranks, r->pipe_base, r->addrs_csv, r->dist_fp16,
                      t0, on_token_rank, &tc, NULL);
    } else {
        rc = engine_generate(&r->engine, ids, nprompt, max_tokens,
                             r->temp, r->top_p, r->seed, r->vocab.eos,
                             on_token_rank, &tc, &tim, err, sizeof(err));
    }
    ylog_info("rank: generate rc=%d (%d tokens, %llu ms)",
              rc, tc.n_tokens, (unsigned long long)(ynow_ms() - t0));
    pthread_mutex_unlock(&r->engine_lock);
    free(ids);
    uint64_t ms = ynow_ms() - t0;
    if (rc != 0) {
        if (r->dist_ranks > 1)
            send_line(fd, PROTO_ERROR " generate: dist pipeline failed");
        else
            send_line(fd, PROTO_ERROR " generate: %s", err);
        return 0;
    }
    send_line(fd, PROTO_DONE " %u %d %llu", tc.n_tokens, 0, (unsigned long long)ms);
    return 0;
}

/* 刷新 supervisor 可见负载(须持 q_mu) */
static void rank_refresh_load_locked(Rank* r)
{
    r->node.inflight = r->in_flight + r->q_len;
    r->node.state = (r->node.inflight > 0) ? NODE_STATE_BUSY : NODE_STATE_READY;
}

static void rank_try_finish_drain(Rank* r)
{
    int dfd;
    pthread_mutex_lock(&r->q_mu);
    if (!r->draining || r->q_len > 0 || r->in_flight > 0 || r->drain_fd < 0) {
        pthread_mutex_unlock(&r->q_mu);
        return;
    }
    dfd = r->drain_fd;
    r->drain_fd = -1;
    pthread_mutex_unlock(&r->q_mu);

    pthread_mutex_lock(&r->engine_lock);
    rank_save_sess_kv(r);
    pthread_mutex_unlock(&r->engine_lock);
    send_line(dfd, "OK");
    ylog_info("rank: DRAIN/QUIT ok (kv saved, exiting)");
    sock_close(dfd);
    r->quit = 1;
    r->io_stop = 1;
}

/* IO 线程: 独占 listen; PING/STAT 同步; INFER 入队; DRAIN 挂起至排空 */
static void rank_io_thread(void* arg)
{
    Rank* r = (Rank*)arg;
    int port = r->serve_port > 0 ? r->serve_port : 0;
    int srv = sock_listen((uint16_t)port, 8);
    if (srv < 0) {
        ylog_error("rank: listen %d failed", port);
        r->quit = 1;
        r->io_stop = 1;
        pthread_cond_broadcast(&r->q_cv);
        return;
    }
    ylog_info("rank: io listen on %u (dispatch; work=%s threads=%d)",
              port, r->work_mode ? "parallel" : "serial", r->work_threads);

    while (!r->io_stop) {
        rank_try_finish_drain(r);
        if (r->io_stop) break;

        int fd = sock_accept_with_timeout(srv, 200);
        if (fd < 0) continue;

        Frame f;
        if (frame_recv(fd, &f) < 0) {
            sock_close(fd);
            continue;
        }

        if (strcmp(f.cmd, PROTO_PING) == 0) {
            handle_ping(fd, r);
            sock_close(fd);
            continue;
        }
        if (strcmp(f.cmd, PROTO_STAT) == 0) {
            handle_stat(fd, r);
            sock_close(fd);
            continue;
        }
        if (strcmp(f.cmd, PROTO_DRAIN) == 0 || strcmp(f.cmd, PROTO_QUIT) == 0) {
            pthread_mutex_lock(&r->q_mu);
            if (r->drain_fd >= 0) {
                pthread_mutex_unlock(&r->q_mu);
                send_line(fd, PROTO_ERROR " already draining");
                sock_close(fd);
                continue;
            }
            r->draining = 1;
            r->quit = 1; /* 打断 pipe dist_gen; infer work 排空后退出 */
            r->drain_fd = fd;
            pthread_cond_broadcast(&r->q_cv);
            pthread_mutex_unlock(&r->q_mu);
            ylog_info("rank: %s accepted, waiting queue drain", f.cmd);
            rank_try_finish_drain(r);
            continue;
        }
        if (strcmp(f.cmd, PROTO_INFER) != 0) {
            send_line(fd, PROTO_ERROR " unknown cmd");
            sock_close(fd);
            continue;
        }

        /* worker 段不接 INFER(数据面走 pipe) */
        if (r->dist_rank > 0 && r->dist_ranks > 1) {
            send_line(fd, PROTO_ERROR " worker rank does not accept INFER");
            sock_close(fd);
            continue;
        }

        int max_tokens = 0;
        long nbytes = 0;
        if (sscanf(f.args, "%d %ld", &max_tokens, &nbytes) != 2 || max_tokens <= 0 || nbytes < 0) {
            send_line(fd, PROTO_ERROR " bad INFER args");
            sock_close(fd);
            continue;
        }

        pthread_mutex_lock(&r->q_mu);
        if (r->draining) {
            pthread_mutex_unlock(&r->q_mu);
            send_line(fd, PROTO_ERROR " draining");
            sock_close(fd);
            continue;
        }
        if (r->q_len >= RANK_JOB_CAP) {
            pthread_mutex_unlock(&r->q_mu);
            send_line(fd, PROTO_ERROR " queue full");
            sock_close(fd);
            continue;
        }
        pthread_mutex_unlock(&r->q_mu);

        char* pb = (char*)ymalloc((size_t)nbytes + 1);
        if (!pb) {
            send_line(fd, PROTO_ERROR " oom");
            sock_close(fd);
            continue;
        }
        if (xrecv_rank(fd, pb, (size_t)nbytes) != 0) {
            free(pb);
            ylog_warn("rank: recv payload FAILED");
            sock_close(fd);
            continue;
        }
        pb[nbytes] = '\0';

        pthread_mutex_lock(&r->q_mu);
        if (r->draining || r->q_len >= RANK_JOB_CAP) {
            pthread_mutex_unlock(&r->q_mu);
            free(pb);
            send_line(fd, PROTO_ERROR " queue full");
            sock_close(fd);
            continue;
        }
        RankJob* job = &r->jobs[r->q_tail];
        job->fd = fd;
        snprintf(job->args, sizeof(job->args), "%s", f.args);
        job->payload = pb;
        job->nbytes = nbytes;
        r->q_tail = (r->q_tail + 1) % RANK_JOB_CAP;
        r->q_len++;
        rank_refresh_load_locked(r);
        pthread_cond_signal(&r->q_cv);
        pthread_mutex_unlock(&r->q_mu);
        node_heartbeat(&r->node);
        /* fd 所有权交给 work, IO 不再 close */
    }

    sock_close(srv);
    pthread_cond_broadcast(&r->q_cv);
}

/* rank0 work: 从队列取 INFER, 在本线程回包(不打断 IO) */
static void rank_infer_work_thread(void* arg)
{
    Rank* r = (Rank*)arg;
    for (;;) {
        RankJob job;
        pthread_mutex_lock(&r->q_mu);
        while (r->q_len == 0 && !r->quit)
            pthread_cond_wait(&r->q_cv, &r->q_mu);
        if (r->q_len == 0) {
            pthread_mutex_unlock(&r->q_mu);
            break;
        }
        job = r->jobs[r->q_head];
        memset(&r->jobs[r->q_head], 0, sizeof(RankJob));
        r->q_head = (r->q_head + 1) % RANK_JOB_CAP;
        r->q_len--;
        r->in_flight++;
        rank_refresh_load_locked(r);
        pthread_mutex_unlock(&r->q_mu);
        node_heartbeat(&r->node);

        handle_infer(job.fd, r, job.args, job.payload, job.nbytes);
        sock_close(job.fd);

        pthread_mutex_lock(&r->q_mu);
        r->in_flight--;
        rank_refresh_load_locked(r);
        pthread_cond_broadcast(&r->q_cv);
        pthread_mutex_unlock(&r->q_mu);
        node_heartbeat(&r->node);
        rank_try_finish_drain(r);
    }
}

/* worker 段: pipe 上 dist_gen 循环(不吃 INFER 队列) */
static void rank_pipe_work_thread(void* arg)
{
    Rank* r = (Rank*)arg;
    while (!r->quit) {
        if (rank_ensure_peers(r) != 0) break;
        DistSess wsess;
        memset(&wsess, 0, sizeof(wsess));
        wsess.my_pos = r->cache_pos;
        wsess.cache_dir = r->cache_dir[0] ? r->cache_dir : NULL;
        wsess.quit = &r->quit;
        r->job.start = r->cache_pos;
        r->job.pos = r->cache_pos;
        r->job.t0 = ynow_ms();
        r->job.phase = 0;
        r->job.dec_t0 = 0;
        wsess.live = &r->job;
        pthread_mutex_lock(&r->engine_lock);
        int rc = dist_gen(&r->engine, &r->vocab, NULL, 0, 0,
                          r->temp, r->top_p, r->seed,
                          r->dist_rank, r->dist_ranks, r->pipe_base, r->addrs_csv, r->dist_fp16,
                          ynow_ms(), NULL, NULL, &wsess);
        if (wsess.key[0])
            snprintf(r->cache_key, sizeof(r->cache_key), "%s", wsess.key);
        r->cache_pos = wsess.my_pos;
        r->job.pos = wsess.my_pos;
        rank_job_end(r);
        rank_save_sess_kv(r);
        pthread_mutex_unlock(&r->engine_lock);
        if (r->quit) break;
        if (rc != 0) {
            ylog_warn("rank: pipeline round failed (rc=%d), retry", rc);
            sock_sleep_ms(1000);
        }
    }
    pthread_mutex_lock(&r->engine_lock);
    rank_save_sess_kv(r);
    pthread_mutex_unlock(&r->engine_lock);
    rank_try_finish_drain(r);
}

/* 独立心跳: 不碰 serve listen */
static void rank_hb_thread(void* arg)
{
    Rank* r = (Rank*)arg;
    while (!r->quit) {
        sock_sleep_ms(2000);
        if (r->node.sv_enabled) {
            r->kv_mb_cache = (double)engine_resident(&r->engine) / 1048576.0;
            r->node.kv_mb = r->kv_mb_cache;
            node_heartbeat(&r->node);
        }
    }
}

/* 主循环: io 独占 listen + hb + work(串行1/并行N 或 worker pipe) */
static int run_rank(int port, Rank* r)
{
    ws_init_rank();
    r->serve_port = port;
    r->drain_fd = -1;
    pthread_mutex_init(&r->q_mu, NULL);
    pthread_cond_init(&r->q_cv, NULL);

    void* hb = NULL;
    void* io = NULL;
    void* works[RANK_WORK_MAX];
    int nw = 0;
    int i;
    memset(works, 0, sizeof(works));

    if (r->node.sv_enabled) ythread_create(&hb, rank_hb_thread, r);
    ythread_create(&io, rank_io_thread, r);

    if (r->dist_rank > 0 && r->dist_ranks > 1) {
        ythread_create(&works[0], rank_pipe_work_thread, r);
        nw = 1;
    } else {
        nw = r->work_threads;
        if (nw < 1) nw = 1;
        if (nw > RANK_WORK_MAX) nw = RANK_WORK_MAX;
        for (i = 0; i < nw; i++)
            ythread_create(&works[i], rank_infer_work_thread, r);
    }

    if (io) ythread_join(&io);
    r->quit = 1;
    pthread_cond_broadcast(&r->q_cv);
    for (i = 0; i < nw; i++)
        if (works[i]) ythread_join(&works[i]);
    if (hb) ythread_join(&hb);

    pthread_mutex_lock(&r->engine_lock);
    rank_save_sess_kv(r);
    pthread_mutex_unlock(&r->engine_lock);
    pthread_cond_destroy(&r->q_cv);
    pthread_mutex_destroy(&r->q_mu);
    return 0;
}

/* ---- 参数解析: 统一走 config(main.c 解析) ---- */

#include "config.h"

/* peers 是否全部在本机(同机 PP 部署): 任一地址非本机则视为跨机, 返回 0 */
static int peers_all_local(const char* peers)
{
    if (!peers || !peers[0]) return 1;
    char lip[64] = "127.0.0.1";
    sock_local_ip(lip, sizeof(lip));
    char buf[512];
    snprintf(buf, sizeof(buf), "%s", peers);
    char* tok = strtok(buf, ",");
    while (tok) {
        if (strcmp(tok, "127.0.0.1") != 0 && strcmp(tok, "localhost") != 0 &&
            strcmp(tok, lip) != 0)
            return 0;
        tok = strtok(NULL, ",");
    }
    return 1;
}

/* 向 supervisor QUERY_RANKS, 按管理口端口升序拼 peers CSV(与 LEASE 一致)。
 * 返回 peer 数; 失败 -1。out 仅 IP(无端口)。 */
static int rank_query_peers(Rank* r, char* out, size_t outsz)
{
    if (!r->node.sv_enabled || outsz == 0) return -1;
    out[0] = '\0';
    int fd = sock_connect(r->node.sv_host, r->node.sv_port, 3);
    if (fd < 0) return -1;
    sock_set_timeout(fd, 5);
    char qargs[256];
    snprintf(qargs, sizeof(qargs), "model=%s", r->node.model);
    frame_send(fd, PROTO_QUERY_RANKS, qargs);
    char addrs[64][128];
    int ports[64];
    int n = 0;
    for (;;) {
        Frame f;
        if (frame_recv(fd, &f) < 0) break;
        if (strcmp(f.cmd, PROTO_RANK_INFO) == 0) {
            const char* a = proto_get(f.args, "addr");
            if (!a || !a[0] || n >= 64) continue;
            snprintf(addrs[n], sizeof(addrs[n]), "%s", a);
            {
                const char* colon = strrchr(addrs[n], ':');
                ports[n] = colon ? atoi(colon + 1) : 0;
            }
            n++;
        } else if (strcmp(f.cmd, PROTO_QUERY_DONE) == 0) {
            break;
        }
    }
    sock_close(fd);
    {
        int i, j;
        for (i = 0; i < n; i++)
            for (j = i + 1; j < n; j++)
                if (ports[j] < ports[i]) {
                    int tp = ports[i]; ports[i] = ports[j]; ports[j] = tp;
                    char ta[128];
                    snprintf(ta, sizeof(ta), "%s", addrs[i]);
                    snprintf(addrs[i], sizeof(addrs[i]), "%s", addrs[j]);
                    snprintf(addrs[j], sizeof(addrs[j]), "%s", ta);
                }
        for (i = 0; i < n; i++) {
            char* colon = strchr(addrs[i], ':');
            if (colon) *colon = '\0';
            if (i) strncat(out, ",", outsz - strlen(out) - 1);
            strncat(out, addrs[i], outsz - strlen(out) - 1);
        }
    }
    return n;
}

/* worker/缺 peers 时: 轮询 supervisor 直到凑齐 dist_ranks 段 */
static int rank_ensure_peers(Rank* r)
{
    if (r->addrs_csv[0] || r->dist_ranks <= 1) return 0;
    if (!r->node.sv_enabled) {
        ylog_error("rank: multi-rank needs --peers or --supervisor (auto-discover)");
        return -1;
    }
    while (!r->quit) {
        char peers[1024];
        int n = rank_query_peers(r, peers, sizeof(peers));
        if (n >= r->dist_ranks && peers[0]) {
            snprintf(r->addrs_csv, sizeof(r->addrs_csv), "%s", peers);
            ylog_info("rank: peers auto-discovered %s (%d ranks)", r->addrs_csv, n);
            return 0;
        }
        ylog_info("rank: waiting peers via supervisor (%d/%d ready)",
                  n < 0 ? 0 : n, r->dist_ranks);
        sock_sleep_ms(1000);
    }
    return -1;
}

/* budget auto: 预算 = min(模型大小, 可用内存 - 余量) */
static uint64_t rank_auto_budget(ServeConfig* cfg)
{
    uint64_t avail = 0;
#ifndef _WIN32
    FILE* f = fopen("/proc/meminfo", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "MemAvailable:", 13) == 0) {
                uint64_t kb = 0;
                sscanf(line + 13, "%llu", (unsigned long long*)&kb);
                avail = kb * 1024;
                break;
            }
        }
        fclose(f);
    }
    if (avail == 0) {
        long avph = sysconf(_SC_AVPHYS_PAGES);
        long pgsz = sysconf(_SC_PAGESIZE);
        avail = (avph > 0 && pgsz > 0) ? (uint64_t)avph * (uint64_t)pgsz : 0;
    }
#endif
    struct stat st;
    uint64_t model_bytes = stat(cfg->model, &st) == 0 ? (uint64_t)st.st_size : 0;
    if (avail == 0) return model_bytes;
    uint64_t reserve = avail / 8;
    if (avail < (uint64_t)1024 * 1024 * 1024) reserve = avail / 4;
    uint64_t cap = avail > reserve ? avail - reserve : avail / 2;
    return model_bytes < cap ? model_bytes : cap;
}

int cmd_rank(ServeConfig* cfg)
{
    if (!cfg->model[0]) {
        fprintf(stderr, "usage: yllm rank --model <file.llf> [--vocab <file>] [--port N] "                        "[--supervisor <ip:port>] [--id <name>] "
                        "[--budget auto|NMB|NG] [--depth N] [--temp F] [--top-p F] [--seed N] "
                        "[--work-mode serial|parallel] [--work-threads N] "
                        "[--device cpu|cuda|vulkan] [--gpu N] [--gpu-weights auto|q4k|fp16] "
                        "[--gpu-layers N] [--gpu-stream 0|1] [--config <yaml>]\n");
        return 1;
    }

    sock_init(); /* sock_local_ip(取本机 IP)依赖 winsock, 必须先初始化 */

    /* 多段 PP 且全部段在本机: 限制 OpenMP 线程数为 核数/段数, 避免多 rank
     * 进程线程超额争抢核(实测不减半 2 段 32 线程 vs 16 核, 速度慢 ~2.7 倍)。
     * 跨机部署(存在远端 peer)时各机只有本机段, 不限制, 跑满本机核。
     * 必须在首次 parallel 区域(engine_init)之前设置; 环境变量优先。 */
#ifdef _OPENMP
    if (getenv("OMP_NUM_THREADS") == NULL) {
        int nr = cfg->ranks > 1 ? cfg->ranks : 1;
        /* 有 peers 且跨机 / 无 peers 的远端 worker: 本机通常只跑一段, 跑满核 */
        if (nr > 1 && cfg->peers[0] && !peers_all_local(cfg->peers))
            nr = 1;
        else if (nr > 1 && !cfg->peers[0] && cfg->rank_idx > 0)
            nr = 1;
        int n = omp_get_num_procs();
        if (n < 1) n = 1;
        int t = n / nr;
        if (t < 1) t = 1;
        omp_set_num_threads(t);
    }
    ylog_info("rank: OpenMP threads=%d (OMP_NUM_THREADS=%s)",
              omp_get_max_threads(),
              getenv("OMP_NUM_THREADS") ? getenv("OMP_NUM_THREADS") : "(auto)");
#endif

    Rank r;
    memset(&r, 0, sizeof(r));
    r.temp = cfg->temp;
    r.top_p = cfg->top_p;
    r.seed = cfg->seed;
    r.uptime_s = (uint64_t)time(NULL);
    if (cfg->cache_dir[0])
        snprintf(r.cache_dir, sizeof(r.cache_dir), "%s", cfg->cache_dir);
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
    int64_t budget_mb = cfg->budget;   /* -1 = 自动 */
    uint64_t budget = (uint64_t)(budget_mb < 0 ? (int64_t)(rank_auto_budget(cfg) / (1024 * 1024)) : budget_mb) * 1024 * 1024;
    if (engine_init(&r.engine, cfg->model, budget, cfg->depth, err, sizeof(err)) != 0) {
        ylog_error("rank: engine init failed: %s", err);
        vocab_free(&r.vocab);
        return 1;
    }

    /* 设备绑定: 默认 cpu; --device cuda 需 YLLM_CUDA=1 构建 */
    {
        DeviceKind dk = DEV_CPU;
        if (cfg->device[0] && device_kind_parse(cfg->device, &dk) != 0) {
            ylog_error("rank: bad --device %s (want cpu|cuda|vulkan)", cfg->device);
            engine_free(&r.engine);
            vocab_free(&r.vocab);
            return 1;
        }
        if (cfg->gpu_weights[0] &&
            cuda_weight_mode_parse(cfg->gpu_weights, &r.engine.cuda_wmode) != 0) {
            /* CPU 路径不依赖该字段; 避免错误配置/脏串直接把 rank 打死 */
            if (dk != DEV_CPU) {
                ylog_error("rank: bad --gpu-weights %s (want auto|q4k|fp16)", cfg->gpu_weights);
                engine_free(&r.engine);
                vocab_free(&r.vocab);
                return 1;
            }
            ylog_warn("rank: ignore bad gpu-weights [%s], use auto (device=cpu)", cfg->gpu_weights);
            r.engine.cuda_wmode = CUDA_W_AUTO;
        }
        if (cfg->gpu_layers >= 0)
            engine_set_gpu_layers(&r.engine, cfg->gpu_layers);
        r.engine.cuda_stream_w = cfg->gpu_stream ? 1 : 0;
        if (dk != DEV_CPU) {
            if (engine_bind_device(&r.engine, dk, cfg->gpu, err, sizeof(err)) != 0) {
                ylog_error("rank: bind device %s gpu=%d failed: %s", cfg->device, cfg->gpu, err);
                engine_free(&r.engine);
                vocab_free(&r.vocab);
                return 1;
            }
            ylog_info("rank: device=%s gpu=%d gpu-weights=%s gpu-layers=%d gpu-stream=%d weights_ready=%d",
                      cfg->device[0] ? cfg->device : "cpu", cfg->gpu,
                      cfg->gpu_weights[0] ? cfg->gpu_weights : "auto",
                      cfg->gpu_layers, cfg->gpu_stream,
                      r.engine.weights_ready);
        }
    }

    /* 多段协作: 协作口基址 = serve 基准(自己的 --port - 段号) + 偏移, 全组统一;
     * 成员地址: --peers 可选覆盖; 否则 worker 经 QUERY_RANKS 发现, rank0 以 INFER/LEASE 为准 */
    r.dist_rank = cfg->rank_idx;
    r.dist_ranks = cfg->ranks > 1 ? cfg->ranks : 1;
    r.dist_fp16 = cfg->dist_fp16;
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
        /* 层段确定后重新 load_weights(CUDA 只装本段; CPU 为空操作) */
        if (engine_load_weights(&r.engine, err, sizeof(err)) != 0) {
            ylog_error("rank: load_weights after split failed: %s", err);
            engine_free(&r.engine);
            vocab_free(&r.vocab);
            return 1;
        }
    }

    pthread_mutex_init(&r.engine_lock, NULL);

    /* work 消费模式: serial=1 线程; parallel=N 线程(引擎仍 engine_lock 互斥) */
    r.work_mode = 0;
    r.work_threads = 1;
    if (cfg->work_mode[0] && strcmp(cfg->work_mode, "parallel") == 0) {
        r.work_mode = 1;
        r.work_threads = cfg->work_threads > 0 ? cfg->work_threads : 2;
        if (r.work_threads > RANK_WORK_MAX) r.work_threads = RANK_WORK_MAX;
    } else if (cfg->work_threads > 1) {
        /* 未写 parallel 但给了 threads>1 → 视为 parallel */
        r.work_mode = 1;
        r.work_threads = cfg->work_threads > RANK_WORK_MAX ? RANK_WORK_MAX : cfg->work_threads;
    }
    ylog_info("rank: work-mode=%s work-threads=%d",
              r.work_mode ? "parallel" : "serial", r.work_threads);

    int rc = run_rank(cfg->rank_port_base, &r);

    engine_free(&r.engine);
    vocab_free(&r.vocab);
    return rc;
}
