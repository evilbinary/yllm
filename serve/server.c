/* server.c — yllm server: 业务逻辑组
 *
 * 从公用 rank 池租用一组 rank(leader = rank0), 接收 router 转发的请求,
 * 派发给 leader, 流式透传回 router。
 * 心跳只发 supervisor(数据面, 生命周期归 supervisor);
 * 注册表增删由 supervisor 通知 router(server 不直接 REGISTER)。
 *
 * 用法:
 *   yllm server --id server-a --model <name> --leader <ip:port>
 *               --supervisor <ip:port> [--port N]
 *               [--log <file>] [--log-level lvl] [--no-console]
 *
 * P2: 单 rank 组(leader 直连); 多组租用/SCALE 在 P3 完善。
 */
#include "protocol.h"
#include "frame.h"
#include "node.h"
#include "sock.h"
#include "server.h"
#include "../inference/yllm.h"
#include "../inference/log.h"
#include <time.h>
#include <pthread.h>
#include <stdint.h>

#define SRV_MAX_LINE 8192

/* INFER 转发: 连 leader rank → 发 INFER → 逐帧透传回客户端 */
static void forward_infer(int client_fd, Server* s, const char* args)
{
    int fd = sock_connect(s->leader_host, s->leader_port, 5);
    if (fd < 0) {
        sock_send_line(client_fd, "ERR server: cannot connect leader %s:%u",
                       s->leader_host, s->leader_port);
        return;
    }
    /* leader 无响应(如 rank 被 STOP/僵死) 60s 后报错, 不无限挂起 */
    sock_set_timeout(fd, 60);
    Frame f;
    snprintf(f.cmd, sizeof(f.cmd), "%s", PROTO_INFER);
    /* 组内 rank 信息随请求捎给 rank0(它按数据组织协作) */
    snprintf(f.args, sizeof(f.args), "%s seg=0 segs=%d peers=%s",
             args, s->lease_ranks > 0 ? s->lease_ranks : 1,
             s->lease_peers[0] ? s->lease_peers : "127.0.0.1");
    frame_send(fd, f.cmd, f.args);

    /* 读 prompt 长度, 透传 prompt bytes */
    long nbytes = frame_payload_len(&f);
    if (nbytes > 0) {
        char* pb = (char*)malloc((size_t)nbytes);
        if (!pb) { sock_close(fd); sock_send_line(client_fd, "ERR server: oom"); return; }
        if (sock_recv_n(client_fd, pb, (size_t)nbytes) != 0) {
            free(pb); sock_close(fd); return;
        }
        sock_send_n(fd, pb, (size_t)nbytes);
        free(pb);
    }

    /* 透传响应(T 帧 + DONE): select 等待可读再 recv
     * (Windows 上 SO_RCVTIMEO 阻塞 recv 在数据未到时可能立即返回 EOF) */
    char out[SRV_MAX_LINE];
    int done = 0;
    while (!done) {
        int sel = sock_wait_readable(fd, 60000);
        if (sel <= 0) {
            ylog_warn("server: leader %s:%u no response (rc=%d)", s->leader_host, s->leader_port, sel);
            sock_send_line(client_fd, "ERR server: leader timeout"); break;
        }
        int n = sock_recv_line(fd, out, sizeof(out));
        if (n < 0) {
            ylog_warn("server: recv from leader %s:%u disconnected", s->leader_host, s->leader_port);
            sock_send_line(client_fd, "ERR server: leader disconnected"); break;
        }
        sock_send_line(client_fd, "%s", out);
        if (strncmp(out, PROTO_DONE, 4) == 0) done = 1;
        else if (strncmp(out, "T ", 2) == 0) {
            size_t tlen = 0;
            char* payload = frame_t_payload(fd, out, sizeof(out), &tlen);
            if (!payload) break;
            sock_send_n(client_fd, payload, tlen);
            if (payload != out) free(payload);
        }
    }
    sock_close(fd);
}


/* 缓存文件名安全化: 替换文件系统非法字符(NTFS 等不支持 ':') */
static void sess_file_name(char* out, size_t outsz, const char* key, const char* ext)
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
    if (o + strlen(ext) + 1 < outsz) {
        memcpy(out + o, ext, strlen(ext) + 1);
    }
}

/* 会话 INFER: INFER_SESS <max_tokens> <nbytes> key=<key>\n<新消息文本>
 * server 侧: 词表渲染新消息 → 会话缓存累积 → 只发增量 token 给 rank(key/resume)。
 * rank 状态未知(重启)时自动全量重发。 */
static void forward_infer_sess(int client_fd, Server* s, const char* args)
{
    int max_tokens = 0;
    long nbytes = 0;
    char key[64] = "";
    if (sscanf(args, "%d %ld", &max_tokens, &nbytes) != 2 || max_tokens <= 0 || nbytes < 0) {
        sock_send_line(client_fd, "ERR bad INFER_SESS args");
        return;
    }
    const char* kv = strstr(args, "key=");
    if (kv) {
        kv += 4;
        size_t kl = strcspn(kv, " ");
        if (kl >= sizeof(key)) kl = sizeof(key) - 1;
        memcpy(key, kv, kl);
        key[kl] = '\0';
    }
    if (!s->sess_vocab_ok) { sock_send_line(client_fd, "ERR server: session vocab not loaded"); return; }

    char* msg = (char*)malloc((size_t)nbytes + 1);
    if (!msg) { sock_send_line(client_fd, "ERR server: oom"); return; }
    if (sock_recv_n(client_fd, msg, (size_t)nbytes) != 0) { free(msg); return; }
    msg[nbytes] = '\0';

    pthread_mutex_lock(&s->sess_lock);
    SessVal* sv = sess_get(&s->sess, key);
    uint32_t resume = 0;
    int first = 0;
    if (!sv) {
        sv = sess_put(&s->sess, key);
        first = 1;
        /* 磁盘恢复: 有 <dir>/<key>.sess 则载入历史 */
        if (s->cache_dir[0]) {
            char path[512];
            sess_file_name(path + snprintf(path, sizeof(path), "%s/", s->cache_dir), sizeof(path), key, ".sess");
            if (sess_load(sv, path) == 0) {
                first = 0;
                resume = sv->n;
                ylog_info("server: session %s restored from %s (%u tokens)", key, path, sv->n);
            }
        }
    }
    else resume = sv->n;   /* 缓存含尾部 eos, 与 rank kv 逐 token 对齐 */
    uint32_t* tokens = (uint32_t*)ymalloc(65536 * 4);
    int n = vocab_chat_ids(&s->sess_vocab, msg, tokens, 65536, first ? 1 : 0);
    free(msg);
    if (n <= 0) {
        free(tokens);
        pthread_mutex_unlock(&s->sess_lock);
        sock_send_line(client_fd, "ERR server: render failed");
        return;
    }
    sess_commit(sv, tokens, (uint32_t)n);
    pthread_mutex_unlock(&s->sess_lock);

    int sent = 0;
    int retry = 0;
    for (;;) {
        int fd = sock_connect(s->leader_host, s->leader_port, 5);
        if (fd < 0) { free(tokens); sock_send_line(client_fd, "ERR server: cannot connect leader"); return; }
        sock_set_timeout(fd, 60);
        uint32_t sr = sent ? 0 : resume;
        const uint32_t* st = tokens;
        uint32_t sn = (uint32_t)n;
        if (sent) {   /* 全量重发: 用缓存 tokens */
            pthread_mutex_lock(&s->sess_lock);
            SessVal* rv = sess_get(&s->sess, key);
            if (!rv) { pthread_mutex_unlock(&s->sess_lock); sock_close(fd); free(tokens); return; }
            st = rv->tokens;
            sn = rv->n;
            pthread_mutex_unlock(&s->sess_lock);
        }
        char fargs[256];
        snprintf(fargs, sizeof(fargs), "%d %u key=%s resume=%u seg=0 segs=%d peers=%s",
                 max_tokens, sn * 4, key, sr,
                 s->lease_ranks > 0 ? s->lease_ranks : 1,
                 s->lease_peers[0] ? s->lease_peers : "127.0.0.1");
        frame_send(fd, PROTO_INFER, fargs);
        sock_send_n(fd, st, (size_t)sn * 4);
        ylog_info("server: sess key=%s resume=%u +%u tokens -> rank", key, sr, sn);
#if YLLM_SESS_DEBUG
        { /* 抓包: 完整帧行 + 全部增量 token(id + 文本) */
            char ids[1024], txt[1024];
            size_t io = 0, to = 0;
            for (uint32_t di = 0; di < sn && io < sizeof(ids) - 20 && to < sizeof(txt) - 20; di++) {
                io += snprintf(ids + io, sizeof(ids) - io, "%u%s", st[di], di + 1 < sn ? "," : "");
                char tmp[64];
                vocab_decode(&s->sess_vocab, &st[di], 1, tmp, sizeof(tmp));
                size_t tl = strlen(tmp);
                if (to + tl + 2 < sizeof(txt)) {
                    memcpy(txt + to, tmp, tl);
                    to += tl;
                    txt[to++] = '|';
                }
            }
            ids[io] = 0;
            txt[to] = 0;
            ylog_info("SERVER->RANK 帧行: [INFER %d %u key=%s resume=%u]",
                      max_tokens, sn * 4, key, sr);
            ylog_info("SERVER->RANK payload: %u 个 token, ids=[%s]", sn, ids);
            ylog_info("SERVER->RANK payload: 文本=[%s]", txt);
        }
#endif

        char out[SRV_MAX_LINE];
        int done = 0, rc = 0;
        while (!done) {
            int sel = sock_wait_readable(fd, 60000);
            if (sel <= 0) { sock_send_line(client_fd, "ERR server: leader timeout"); rc = -1; break; }
            int nn = sock_recv_line(fd, out, sizeof(out));
            if (nn < 0) { rc = -1; break; }
            if (strncmp(out, "ERR", 3) == 0) {
                if (!sent) { sock_close(fd); sent = 1; rc = -2; break; }  /* 全量重试 */
                if (retry < 5) {
                    /* 生成前失败(未产出 token): 退避重试, 等 worker 段就绪 */
                    sock_close(fd);
                    retry++;
                    rc = -2;
                    ylog_warn("server: sess %s backend reject (%s), retry %d/5", key, out + 4, retry);
                    sock_sleep_ms(1000 * retry);
                    break;
                }
                sock_send_line(client_fd, "%s", out);
                rc = -1;
                break;
            }
            sock_send_line(client_fd, "%s", out);
            if (strncmp(out, PROTO_DONE, 4) == 0) {
                done = 1;
                pthread_mutex_lock(&s->sess_lock);
                SessVal* ev = sess_get(&s->sess, key);
                if (ev && s->sess_vocab.eos >= 0) sess_append(ev, (uint32_t)s->sess_vocab.eos);
                /* 落盘: token 列表(会话重启恢复用) */
                if (ev && s->cache_dir[0]) {
                    char path[512];
                    sess_file_name(path + snprintf(path, sizeof(path), "%s/", s->cache_dir), sizeof(path), key, ".sess");
                    int rc2 = sess_save(ev, path);
                    ylog_info("server: sess %s saved (%u tokens) rc=%d -> %s", key, ev->n, rc2, path);
                }
                pthread_mutex_unlock(&s->sess_lock);
            } else if (strncmp(out, "TS ", 3) == 0) {
                uint32_t tok = 0;
                size_t tlen = 0;
                sscanf(out + 3, "%zu %u", &tlen, &tok);
                if (tlen <= 0) { rc = -1; break; }
                char* payload = (char*)malloc(tlen + 1);
                if (!payload) { rc = -1; break; }
                if (sock_recv_n(fd, payload, tlen) != 0) { free(payload); rc = -1; break; }
                payload[tlen] = '\0';
                pthread_mutex_lock(&s->sess_lock);
                SessVal* gv = sess_get(&s->sess, key);
                if (gv) sess_append(gv, tok);
                pthread_mutex_unlock(&s->sess_lock);
                sock_send_n(client_fd, payload, tlen);
                free(payload);
            }
        }
        sock_close(fd);
        if (rc == -2) continue;
        free(tokens);
        return;
    }
}

/* 向 sv 租用该模型一组 rank(标记忙), 成功填 leader_host/port */
static int server_lease(Server* s)
{
    char args[512];
    snprintf(args, sizeof(args), "%s model=%s", s->node.node_id, s->node.model);
    if (strcmp(s->lease_strategy, "permanent") == 0) {
        strncat(args, " permanent", sizeof(args) - strlen(args) - 1);
    } else if (strcmp(s->lease_strategy, "timed") == 0 && s->lease_duration > 0) {
        char d[64];
        snprintf(d, sizeof(d), " duration=%d", s->lease_duration);
        strncat(args, d, sizeof(args) - strlen(args) - 1);
    }
    int fd = sock_connect(s->node.sv_host, s->node.sv_port, 3);
    if (fd < 0) return -1;
    sock_set_timeout(fd, 5);
    frame_send(fd, PROTO_LEASE, args);
    Frame f;
    int rc = -1;
    if (frame_recv(fd, &f) >= 0 && strcmp(f.cmd, "OK") == 0) {
        const char* leader = proto_get(f.args, "leader");
        if (leader) {
            const char* colon = strchr(leader, ':');
            if (colon) {
                size_t hlen = (size_t)(colon - leader);
                if (hlen >= sizeof(s->leader_host)) hlen = sizeof(s->leader_host) - 1;
                memcpy(s->leader_host, leader, hlen);
                s->leader_host[hlen] = '\0';
                s->leader_port = (uint16_t)atoi(colon + 1);
                rc = 0;
            }
        }
        s->lease_ranks = atoi(proto_get(f.args, "ranks") ? proto_get(f.args, "ranks") : "1");
        const char* peers = proto_get(f.args, "peers");
        if (peers) snprintf(s->lease_peers, sizeof(s->lease_peers), "%s", peers);
        else s->lease_peers[0] = '\0';
        const char* rids = proto_get(f.args, "rank_ids");
        if (rids) snprintf(s->lease_rank_ids, sizeof(s->lease_rank_ids), "%s", rids);
        else s->lease_rank_ids[0] = '\0';
    }
    sock_close(fd);
    return rc;
}

/* 释放租用(rank 回池) */
static void server_release(Server* s)
{
    int fd = sock_connect(s->node.sv_host, s->node.sv_port, 3);
    if (fd < 0) return;
    sock_set_timeout(fd, 5);
    frame_send(fd, PROTO_RELEASE, s->node.node_id);
    Frame f;
    (void)frame_recv(fd, &f);
    sock_close(fd);
}

static void handle_frame(int fd, Server* s, const char* cmd, const char* args)
{
    if (strcmp(cmd, PROTO_INFER_SESS) == 0) {
        s->node.inflight++;
        int request_lease = strcmp(s->lease_strategy, "request") == 0;
        int ok = 1;
        if (request_lease && server_lease(s) != 0) {
            ylog_warn("server: sess lease failed (no free rank), reject");
            sock_send_line(fd, "ERR server: no rank leased");
            ok = 0;
        }
        if (ok) {
            forward_infer_sess(fd, s, args);
            if (request_lease) server_release(s);
        }
        s->node.inflight--;
        return;
    }
    if (strcmp(cmd, PROTO_INFER) == 0) {
        s->node.inflight++;
        int request_lease = strcmp(s->lease_strategy, "request") == 0;
        int ok = 1;
        if (request_lease && server_lease(s) != 0) {
            ylog_warn("server: lease failed (no free rank), reject");
            sock_send_line(fd, "ERR server: no rank leased");
            ok = 0;
        }
        if (ok) forward_infer(fd, s, args);
        if (request_lease) server_release(s);
        s->node.inflight--;
    } else if (strcmp(cmd, PROTO_PING) == 0) {
        frame_send(fd, "OK", "READY");
    } else if (strcmp(cmd, PROTO_STAT) == 0) {
        uint64_t uptime = s->start_s ? (uint64_t)(time(NULL) - (time_t)s->start_s) : 0;
        char st[640];
        snprintf(st, sizeof(st),
                 "inflight=%d kv_mb=0.0 prefix_hits=0 uptime_s=%llu "
                 "lease_ranks=%d lease_peers=%s lease_rank_ids=%s",
                 s->node.inflight, (unsigned long long)uptime,
                 s->lease_ranks > 0 ? s->lease_ranks : 1,
                 s->lease_peers[0] ? s->lease_peers : "127.0.0.1",
                 s->lease_rank_ids[0] ? s->lease_rank_ids : "?");
        frame_send(fd, "OK", st);
    } else if (strcmp(cmd, PROTO_DRAIN) == 0 || strcmp(cmd, PROTO_QUIT) == 0) {
        frame_send(fd, "OK", NULL);
        s->quit = 1;
    } else {
        frame_send(fd, "ERR", "unknown cmd");
    }
}

/* 单连接处理(线程化: 长生成不阻塞该 server 的其他连接)。
 * s 随线程参数传入, 不共享全局(多 server 并存时避免串台)。 */
typedef struct { Server* s; int fd; } SrvConnArg;
static void* srv_conn(void* arg)
{
    SrvConnArg* a = (SrvConnArg*)arg;
    int fd = a->fd;
    Server* s = a->s;
    free(a);
    Frame f;
    if (frame_recv(fd, &f) >= 0)
        handle_frame(fd, s, f.cmd, f.args);
    sock_close(fd);
    return NULL;
}

/* 独立心跳线程: 长请求转发期间 accept 循环被占用, 心跳不能停 */
static void srv_hb_thread(void* arg)
{
    Server* s = (Server*)arg;
    while (!s->quit) {
        sock_sleep_ms(2000);
        s->node.kv_mb = 0; /* P2: 无组内汇总, 占位 */
        node_heartbeat(&s->node);
    }
}

static void server_sess_init(Server* s)
{
    if (s->vocab_path[0]) {
        if (vocab_load(s->vocab_path, &s->sess_vocab) == 0) {
            s->sess_vocab_ok = 1;
            sess_init(&s->sess, 64);
            pthread_mutex_init(&s->sess_lock, NULL);
            ylog_info("server: session vocab=%s loaded (%d pieces)", s->vocab_path, s->sess_vocab.n);
        } else {
            ylog_warn("server: session vocab load failed: %s", s->vocab_path);
        }
    }
}

int server_run(Server* s)
{
    server_sess_init(s);
    sock_init();

    /* leader 未指定: 按租用策略 —— timed/permanent 启动时租一次;
     * request 策略不预租(每次推理时 LEASE/RELEASE) */
    if (!s->leader_host[0] && strcmp(s->lease_strategy, "request") != 0) {
        int tries;
        for (tries = 0; tries < 90; tries++) {
            if (server_lease(s) == 0) break;
            sock_sleep_ms(1000);
        }
        if (!s->leader_host[0]) {
            ylog_error("server: lease failed (no ready rank for %s)", s->node.model);
            return 1;
        }
    }

    int srv = sock_listen(s->port, 8);
    if (srv < 0) return 1;
    ylog_info("server %s: listening on port %u (leader=%s:%u, heartbeat → %s:%u)",
              s->node.node_id, s->port, s->leader_host, s->leader_port,
              s->node.sv_host, s->node.sv_port);

    void* hb = NULL;
    if (s->node.sv_enabled) ythread_create(&hb, srv_hb_thread, s);
    for (;;) {
        int fd = sock_accept_with_timeout(srv, 500);
        if (fd >= 0) {
            SrvConnArg* a = (SrvConnArg*)malloc(sizeof(*a));
            if (a) {
                a->s = s;
                a->fd = fd;
                pthread_t t;
                if (pthread_create(&t, NULL, srv_conn, a) != 0) {
                    free(a);
                    sock_close(fd);
                } else {
                    pthread_detach(t);
                }
            } else {
                sock_close(fd);
            }
        }
        if (s->quit) break;
    }
    sock_close(srv);
    return 0;
}

/* 自动发现 leader: 向 supervisor 查询该模型的 ready rank, 取第一个填 leader_host/port。
 * 返回 0 成功, -1 失败(无 rank / 查询失败)。 */
int server_resolve_leader(Server* s, const char* sv_host, uint16_t sv_port, const char* model)
{
    int fd = sock_connect(sv_host, sv_port, 3);
    if (fd < 0) return -1;
    sock_set_timeout(fd, 5);
    char qargs[512];
    snprintf(qargs, sizeof(qargs), "model=%s", model ? model : "");
    frame_send(fd, PROTO_QUERY_RANKS, qargs);
    for (;;) {
        Frame f;
        if (frame_recv(fd, &f) < 0) break;
        if (strcmp(f.cmd, PROTO_RANK_INFO) == 0) {
            const char* a = proto_get(f.args, "addr");
            if (a) {
                const char* colon = strchr(a, ':');
                if (colon) {
                    size_t hlen = (size_t)(colon - a);
                    if (hlen >= sizeof(s->leader_host)) hlen = sizeof(s->leader_host) - 1;
                    memcpy(s->leader_host, a, hlen);
                    s->leader_host[hlen] = '\0';
                    s->leader_port = (uint16_t)atoi(colon + 1);
                    sock_close(fd);
                    ylog_info("server: leader auto-discovered %s:%u (model=%s)",
                              s->leader_host, s->leader_port, model);
                    return 0;
                }
            }
        } else if (strcmp(f.cmd, PROTO_QUERY_DONE) == 0) {
            break;
        }
    }
    sock_close(fd);
    return -1;
}

int cmd_server(ServeConfig* cfg)
{
    if (!cfg->model_name[0]) {
        fprintf(stderr, "usage: yllm server --id <name> --model-name <name> [--leader <ip:port>] "
                        "--supervisor <ip:port> [--port N]\n");
        return 1;
    }

    Server s;
    memset(&s, 0, sizeof(s));
    if (cfg->node_id[0] && strcmp(cfg->node_id, "node-0") != 0)
        snprintf(s.node.node_id, sizeof(s.node.node_id), "%s", cfg->node_id);
    else
        snprintf(s.node.node_id, sizeof(s.node.node_id), "server-0");
    snprintf(s.node.type, sizeof(s.node.type), "server");
    snprintf(s.node.model, sizeof(s.node.model), "%s", cfg->model_name);
    s.node.state = NODE_STATE_READY;
    s.port = (uint16_t)cfg->server_port;
    s.start_s = (uint64_t)time(NULL);
    if (cfg->model[0])
        snprintf(s.resolve_model, sizeof(s.resolve_model), "%s", cfg->model);
    if (cfg->vocab[0])
        snprintf(s.vocab_path, sizeof(s.vocab_path), "%s", cfg->vocab);
    if (cfg->cache_dir[0])
        snprintf(s.cache_dir, sizeof(s.cache_dir), "%s", cfg->cache_dir);

    /* 租用策略(默认 request: 每次推理 LEASE/RELEASE) */
    snprintf(s.lease_strategy, sizeof(s.lease_strategy), "%s",
             cfg->lease_strategy[0] ? cfg->lease_strategy : "request");
    s.lease_duration = cfg->lease_duration;

    /* 解析 leader ip:port; 未配置 → 走租用(LEASE), request 策略每次推理时租 */
    if (cfg->leader[0]) {
        const char* colon = strchr(cfg->leader, ':');
        if (!colon) { fprintf(stderr, "bad leader addr: %s\n", cfg->leader); return 1; }
        size_t hlen = (size_t)(colon - cfg->leader);
        if (hlen >= sizeof(s.leader_host)) hlen = sizeof(s.leader_host) - 1;
        memcpy(s.leader_host, cfg->leader, hlen);
        s.leader_host[hlen] = '\0';
        s.leader_port = (uint16_t)atoi(colon + 1);
    }

    /* 自身 addr(上报 supervisor 用): ip:port */
    snprintf(s.node.addr, sizeof(s.node.addr), "%s:%u", cfg->sv_host, s.port);

    /* supervisor 地址 */
    snprintf(s.node.sv_host, sizeof(s.node.sv_host), "%s", cfg->sv_host);
    s.node.sv_port = (uint16_t)cfg->sv_port;
    s.node.sv_enabled = 1;

    return server_run(&s);
}
