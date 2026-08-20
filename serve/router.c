/* router.c — yllm router: 调度层
 *
 * 只做路由决策: server 候选表由 supervisor 通知(SERVER_ADD/DEL/UPDATE)维护,
 * 也可主动 QUERY_SERVERS 拉取快照。把客户端请求转发到选中 server 并透传结果。
 *
 * 用法:
 *   yllm router --port <N> [--supervisor <ip:port>] [--strategy least|round-robin]
 *               [--log <file>] [--log-level lvl] [--no-console]
 * 客户端:
 *   yllm router --send "<model> <max_tokens> <prompt>"
 */
#include "protocol.h"
#include "frame.h"
#include "node.h"
#include "sock.h"
#include "router.h"
#include "router_http.h"
#include "../inference/log.h"
#include <time.h>
#include <stdint.h>

#define RT_MAX_LINE 8192

static RtServer* find_server(Router* r, const char* id)
{
    int i;
    for (i = 0; i < r->n_servers; i++)
        if (strcmp(r->servers[i].id, id) == 0) return &r->servers[i];
    return NULL;
}

/* 加锁的 find(调用方必须已持有 lock) */
static RtServer* find_server_locked(Router* r, const char* id)
{
    return find_server(r, id);
}

/* supervisor 通知: SERVER_ADD */
static void handle_server_add(Router* r, const char* args)
{
    Frame f;
    snprintf(f.cmd, sizeof(f.cmd), "X");
    snprintf(f.args, sizeof(f.args), "%s", args);
    char id[128];
    const char* sp = strchr(args, ' ');
    if (!sp) { if (sscanf(args, "%127s", id) == 1) sp = args + strlen(args); else return; }
    size_t idlen = (size_t)(sp - args);
    if (idlen >= sizeof(id)) idlen = sizeof(id) - 1;
    memcpy(id, args, idlen);
    id[idlen] = '\0';

    pthread_mutex_lock(&r->lock);
    RtServer* s = find_server_locked(r, id);
    if (!s) {
        if (r->n_servers >= RT_MAX_SERVERS) return;
        s = &r->servers[r->n_servers++];
        memset(s, 0, sizeof(*s));
        snprintf(s->id, sizeof(s->id), "%s", id);
    }
    char vb[256];
    if (frame_get(&f, "model", vb, sizeof(vb)) == 0) snprintf(s->model, sizeof(s->model), "%s", vb);
    if (frame_get(&f, "leader", vb, sizeof(vb)) == 0) {
        const char* colon = strchr(vb, ':');
        if (colon) {
            size_t hlen = (size_t)(colon - vb);
            if (hlen >= sizeof(s->leader_host)) hlen = sizeof(s->leader_host) - 1;
            memcpy(s->leader_host, vb, hlen);
            s->leader_host[hlen] = '\0';
            s->leader_port = (uint16_t)atoi(colon + 1);
        }
    }
    s->state = NODE_STATE_READY;
    s->last_update = (uint64_t)time(NULL);
    pthread_mutex_unlock(&r->lock);
    ylog_info("router: SERVER_ADD %s model=%s leader=%s:%u", s->id, s->model,
              s->leader_host, s->leader_port);
}

/* supervisor 通知: SERVER_DEL */
static void handle_server_del(Router* r, const char* args)
{
    char id[128];
    if (sscanf(args, "%127s", id) != 1) return;
    pthread_mutex_lock(&r->lock);
    RtServer* s = find_server_locked(r, id);
    if (s) {
        s->state = NODE_STATE_DEAD;
        ylog_warn("router: SERVER_DEL %s", id);
    }
    pthread_mutex_unlock(&r->lock);
}

/* supervisor 通知: SERVER_UPDATE */
static void handle_server_update(Router* r, const char* args)
{
    char id[128];
    int inflight = 0;
    double kv = 0;
    if (sscanf(args, "%127s inflight=%d kv_mb=%lf", id, &inflight, &kv) >= 1) {
        pthread_mutex_lock(&r->lock);
        RtServer* s = find_server_locked(r, id);
        if (s) {
            s->inflight = inflight;
            s->kv_mb = kv;
            s->last_update = (uint64_t)time(NULL);
        }
        pthread_mutex_unlock(&r->lock);
    }
}

/* 主动查询 supervisor: 拉取 server 快照(方式B) */
static void query_servers(Router* r, const char* sv_host, uint16_t sv_port)
{
    int fd = sock_connect(sv_host, sv_port, 3);
    if (fd < 0) { ylog_warn("router: cannot reach supervisor %s:%u", sv_host, sv_port); return; }
    frame_send(fd, PROTO_QUERY_SERVERS, NULL);
    Frame f;
    while (frame_recv(fd, &f) >= 0) {
        if (strcmp(f.cmd, PROTO_QUERY_DONE) == 0) break;
        if (strcmp(f.cmd, PROTO_SERVER_INFO) == 0) {
            /* 只登记 server 节点(节点表含 rank/router, 需按 type 过滤) */
            Frame ff0;
            snprintf(ff0.cmd, sizeof(ff0.cmd), "X");
            snprintf(ff0.args, sizeof(ff0.args), "%s", f.args);
            char vb0[64];
            if (frame_get(&ff0, "type", vb0, sizeof(vb0)) == 0 && strcmp(vb0, "server") != 0)
                continue;
            char id[128];
            if (sscanf(f.args, "%127s", id) == 1) {
                RtServer* s = find_server(r, id);
                if (!s) {
                    /* 暂存: 简单处理为 ADD */
                    char buf[1024];
                    snprintf(buf, sizeof(buf), "%s %s", id, f.args + strlen(id) + 1);
                    handle_server_add(r, buf);
                } else {
                    Frame ff;
                    snprintf(ff.cmd, sizeof(ff.cmd), "X");
                    snprintf(ff.args, sizeof(ff.args), "%s", f.args);
                    char vb[256];
                    pthread_mutex_lock(&r->lock);
                    if (frame_get(&ff, "state", vb, sizeof(vb)) == 0) {
                        if (strcmp(vb, "ready") == 0) s->state = NODE_STATE_READY;
                        else if (strcmp(vb, "dead") == 0) s->state = NODE_STATE_DEAD;
                    }
                    if (frame_get(&ff, "leader", vb, sizeof(vb)) == 0) {
                        const char* colon = strchr(vb, ':');
                        if (colon) {
                            size_t hlen = (size_t)(colon - vb);
                            if (hlen >= sizeof(s->leader_host)) hlen = sizeof(s->leader_host) - 1;
                            memcpy(s->leader_host, vb, hlen);
                            s->leader_host[hlen] = '\0';
                            s->leader_port = (uint16_t)atoi(colon + 1);
                        }
                    }
                    if (frame_get(&ff, "model", vb, sizeof(vb)) == 0)
                        snprintf(s->model, sizeof(s->model), "%s", vb);
                    pthread_mutex_unlock(&r->lock);
                }
            }
        }
    }
    sock_close(fd);
}

/* 路由决策: 选一个 READY server */
/* 模型是否已注册(存在 server 条目, 无论是否就绪) */
static int router_model_known(Router* r, const char* model)
{
    int i, known = 0;
    pthread_mutex_lock(&r->lock);
    for (i = 0; i < r->n_servers; i++)
        if (strcmp(r->servers[i].model, model) == 0) { known = 1; break; }
    pthread_mutex_unlock(&r->lock);
    return known;
}

static RtServer* pick_server(Router* r, const char* model)
{
    int i;
    int n = 0;
    pthread_mutex_lock(&r->lock);
    for (i = 0; i < r->n_servers; i++)
        if (r->servers[i].state == NODE_STATE_READY && strcmp(r->servers[i].model, model) == 0) n++;
    if (n == 0) { pthread_mutex_unlock(&r->lock); return NULL; }

    if (r->strategy && strcmp(r->strategy, "round-robin") == 0) {
        r->rr_counter = (r->rr_counter + 1) % n;
        int k = 0;
        for (i = 0; i < r->n_servers; i++) {
            if (r->servers[i].state == NODE_STATE_READY && strcmp(r->servers[i].model, model) == 0) {
                if (k == r->rr_counter) return &r->servers[i];
                k++;
            }
        }
    }
    /* 默认 least-inflight */
    RtServer* best = NULL;
    for (i = 0; i < r->n_servers; i++) {
        if (r->servers[i].state == NODE_STATE_READY && strcmp(r->servers[i].model, model) == 0) {
            if (!best || r->servers[i].inflight < best->inflight)
                best = &r->servers[i];
        }
    }
    pthread_mutex_unlock(&r->lock);
    return best;
}

/* 客户端 INFER: 解析 <model> <max_tokens> <n_bytes>\n + prompt */
/* 通用 INFER 转发: 路由到 server → 转发给 leader rank → 逐 token 回调。
 * on_token(token utf8, ctx) 每生成一个 token 回调一次; 返回 0 成功, -1 失败 */
int router_infer(Router* r, const char* model, int max_tokens,
                 const char* prompt, size_t plen,
                 void (*on_token)(const char* utf8, size_t len, void* ctx), void* ctx,
                 float temp, float top_p)
{
    RtServer* s = pick_server(r, model);
    if (!s) {
        if (!router_model_known(r, model)) {
            ylog_warn("router_infer: model %s not registered", model);
            return -2;   /* 模型不存在, 立即返回, 不轮询 */
        }
        int i;
        for (i = 0; i < 60 && !s; i++) {
            sock_sleep_ms(500);
            s = pick_server(r, model);
        }
        if (!s) { ylog_warn("router_infer: no ready server for %s", model); return -1; }
    }
    int sfd = sock_connect(s->leader_host, s->leader_port, 5);
    if (sfd < 0) { ylog_warn("router_infer: connect %s:%u fail", s->leader_host, s->leader_port); return -1; }
    char args[192];
    snprintf(args, sizeof(args), "%d %zu temp=%.6g top_p=%.6g", max_tokens, plen, temp, top_p);
    frame_send_payload(sfd, PROTO_INFER, args, prompt, plen);

    pthread_mutex_lock(&r->lock);
    s->inflight++;
    pthread_mutex_unlock(&r->lock);
    char out[RT_MAX_LINE];
    int rc = 0;
    int done = 0;
    while (!done) {
        int n = sock_recv_line(sfd, out, sizeof(out));
        if (n < 0) { rc = -1; break; }
        if (strncmp(out, PROTO_DONE, 4) == 0) done = 1;
        else if (strncmp(out, "T ", 2) == 0) {
            size_t tlen = 0;
            char* payload = frame_t_payload(sfd, out, sizeof(out), &tlen);
            if (!payload) { rc = -1; break; }
            if (on_token) on_token(payload, tlen, ctx);
            if (payload != out) free(payload);
        }
    }
    pthread_mutex_lock(&r->lock);
    s->inflight--;
    pthread_mutex_unlock(&r->lock);
    sock_close(sfd);
    return rc;
}

/* 会话模式推理(转发): 带会话 key + 新消息文本发给 server(会话管理在 server 侧)。
 * server 渲染/缓存后只把增量 token 发给 rank; 这里只透传 TS 帧文本。 */
int router_infer_sess(Router* r, const char* model, int max_tokens,
                      const char* sess_key, const char* new_msg, size_t msg_len,
                      void (*on_token)(const char* utf8, size_t len, void* ctx), void* ctx,
                      float temp, float top_p)
{
    RtServer* s = pick_server(r, model);
    if (!s) {
        if (!router_model_known(r, model)) {
            ylog_warn("router_infer_sess: model %s not registered", model);
            return -2;   /* 模型不存在, 立即返回, 不轮询 */
        }
        int i;
        for (i = 0; i < 60 && !s; i++) {
            sock_sleep_ms(500);
            s = pick_server(r, model);
        }
        if (!s) { ylog_warn("router_infer_sess: no ready server for %s", model); return -1; }
    }
    int sfd = sock_connect(s->leader_host, s->leader_port, 5);
    if (sfd < 0) { ylog_warn("router_infer_sess: connect %s:%u fail", s->leader_host, s->leader_port); return -1; }
    char args[192];
    snprintf(args, sizeof(args), "%d %zu key=%s temp=%.6g top_p=%.6g",
             max_tokens, msg_len, sess_key, temp, top_p);
#if YLLM_SESS_DEBUG
    ylog_info("ROUTER->SERVER: INFER_SESS args=[%s] msg=[%.60s]", args, new_msg);
#endif
    frame_send_payload(sfd, PROTO_INFER_SESS, args, new_msg, msg_len);

    pthread_mutex_lock(&r->lock);
    s->inflight++;
    pthread_mutex_unlock(&r->lock);
    char out[RT_MAX_LINE];
    int rc = 0;
    int done = 0;
    while (!done) {
        int n2 = sock_recv_line(sfd, out, sizeof(out));
        if (n2 < 0) { rc = -1; break; }
        if (strncmp(out, PROTO_DONE, 4) == 0) done = 1;
        else if (strncmp(out, "TS ", 3) == 0) {
            size_t tlen = 0;
            sscanf(out + 3, "%zu", &tlen);
            if (tlen <= 0) { rc = -1; break; }
            char* payload = (char*)malloc(tlen + 1);
            if (!payload) { rc = -1; break; }
            if (sock_recv_n(sfd, payload, tlen) != 0) { free(payload); rc = -1; break; }
            payload[tlen] = '\0';
            if (on_token) on_token(payload, tlen, ctx);
            free(payload);
        }
    }
    pthread_mutex_lock(&r->lock);
    s->inflight--;
    pthread_mutex_unlock(&r->lock);
    sock_close(sfd);
    return rc;
}

static void handle_client_infer(int fd, Router* r, const char *args)
{
    char model[128];
    int max_tokens = 0;
    long nbytes = 0;
    if (sscanf(args, "%127s %d %ld", model, &max_tokens, &nbytes) != 3) {
        sock_send_line(fd, "ERR bad INFER: usage <model> <max_tokens> <n_bytes>");
        return;
    }
    RtServer* s = pick_server(r, model);
    if (!s) {
        sock_send_line(fd, "ERR no ready server for model %s", model);
        return;
    }
    int sfd = sock_connect(s->leader_host, s->leader_port, 5);
    if (sfd < 0) {
        sock_send_line(fd, "ERR cannot connect server %s", s->id);
        return;
    }
    /* 转发 INFER 头 + prompt(args 只含 max_tokens, nbytes 由 payload 函数追加) */
    char infer_args[64];
    snprintf(infer_args, sizeof(infer_args), "%d", max_tokens);
    if (nbytes > 0) {
        char* pb = (char*)malloc((size_t)nbytes);
        if (!pb) { sock_close(sfd); sock_send_line(fd, "ERR oom"); return; }
        if (sock_recv_n(fd, pb, (size_t)nbytes) != 0) { free(pb); sock_close(sfd); return; }
        frame_send_payload(sfd, PROTO_INFER, infer_args, pb, (size_t)nbytes);
        free(pb);
    } else {
        frame_send_payload(sfd, PROTO_INFER, infer_args, NULL, 0);
    }
    pthread_mutex_lock(&r->lock);
    s->inflight++;
    pthread_mutex_unlock(&r->lock);
    /* 透传 T/DONE */
    char out[RT_MAX_LINE];
    int done = 0;
    while (!done) {
        int n = sock_recv_line(sfd, out, sizeof(out));
        if (n < 0) { sock_send_line(fd, "ERR server disconnected"); break; }
        sock_send_line(fd, "%s", out);
        if (strncmp(out, PROTO_DONE, 4) == 0) done = 1;
        else if (strncmp(out, "T ", 2) == 0) {
            size_t tlen = 0;
            char* payload = frame_t_payload(sfd, out, sizeof(out), &tlen);
            if (!payload) break;
            sock_send_n(fd, payload, tlen);
            if (payload != out) free(payload);
        }
    }
    pthread_mutex_lock(&r->lock);
    s->inflight--;
    pthread_mutex_unlock(&r->lock);
    sock_close(sfd);
}

/* 客户端模式: --send "<model> <max_tokens> <prompt>" */
static int run_client(Router* r, const char* send)
{
    sock_init();
    char model[128];
    int max_tokens = 0;
    const char* prompt = send;
    if (sscanf(send, "%127s %d", model, &max_tokens) != 2) {
        fprintf(stderr, "usage: yllm router --send \"<model> <max_tokens> <prompt>\"\n");
        return 1;
    }
    /* 跳过 model 和 max_tokens 两个 token, 剩余为 prompt */
    prompt = send;
    int skip = 0;
    while (*prompt) {
        while (*prompt == ' ') prompt++;
        if (!*prompt) break;
        while (*prompt && *prompt != ' ') prompt++;
        skip++;
        if (skip >= 2) break;
    }
    while (*prompt == ' ') prompt++;

    int fd = (int)socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return 1;
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(r->port);
    if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
        fprintf(stderr, "cannot connect router on port %u\n", r->port);
        sock_close(fd);
        return 1;
    }
    size_t plen = strlen(prompt);
    char args[160];
    snprintf(args, sizeof(args), "%s %d", model, max_tokens);
    frame_send_payload(fd, PROTO_INFER, args, prompt, plen);
    /* 读响应: T 帧打印 token 内容 */
    char out[RT_MAX_LINE];
    int done = 0;
    while (!done) {
        int n = sock_recv_line(fd, out, sizeof(out));
        if (n < 0) break;
        if (strncmp(out, "T ", 2) == 0) {
            long tlen = atol(out + 2);
            if (tlen > 0 && tlen < (long)sizeof(out)) {
                if (sock_recv_n(fd, out, (size_t)tlen) != 0) break;
                fwrite(out, 1, (size_t)tlen, stdout);
                fflush(stdout);
            }
        } else {
            printf("%s\n", out);
            if (strncmp(out, PROTO_DONE, 4) == 0) done = 1;
        }
    }
    sock_close(fd);
    return 0;
}

/* 单连接处理(线程化: 长生成不阻塞其他连接) */
static Router* router_conn_router;  /* 简化线程参数传递 */

static void* router_conn(void* arg)
{
    int fd = (int)(intptr_t)arg;
    Router* r = router_conn_router;
    Frame f;
    if (frame_recv(fd, &f) >= 0) {
        if (strcmp(f.cmd, PROTO_INFER) == 0)
            handle_client_infer(fd, r, f.args);
        else if (strcmp(f.cmd, PROTO_SERVER_ADD) == 0)
            handle_server_add(r, f.args);
        else if (strcmp(f.cmd, PROTO_SERVER_DEL) == 0)
            handle_server_del(r, f.args);
        else if (strcmp(f.cmd, PROTO_SERVER_UPDATE) == 0)
            handle_server_update(r, f.args);
        else if (strcmp(f.cmd, PROTO_QUIT) == 0) {
            frame_send(fd, "OK", NULL);
            r->quit = 1;
        } else if (strcmp(f.cmd, PROTO_PING) == 0) {
            frame_send(fd, "OK", "READY");
        } else if (strcmp(f.cmd, PROTO_STAT) == 0) {
            int total = 0, i2;
            pthread_mutex_lock(&r->lock);
            for (i2 = 0; i2 < r->n_servers; i2++) total += r->servers[i2].inflight;
            pthread_mutex_unlock(&r->lock);
            char st[256];
            snprintf(st, sizeof(st), "servers=%d inflight=%d kv_mb=0.0 prefix_hits=0 uptime_s=0",
                     r->n_servers, total);
            frame_send(fd, "OK", st);
        } else
            frame_send(fd, "ERR", "unknown cmd");
    }
    sock_close(fd);
    return NULL;
}

int router_run(Router* r, const char* sv_host, uint16_t sv_port)
{
    sock_init();
    int srv = sock_listen(r->port, 16);
    if (srv < 0) return 1;
    ylog_info("router: listening on port %u", r->port);
    router_conn_router = r;

    uint64_t last_hb = 0;
    uint64_t last_query = 0;
    for (;;) {
        int fd = sock_accept_with_timeout(srv, 500);
        if (fd >= 0) {
            pthread_t t;
            pthread_create(&t, NULL, router_conn, (void*)(intptr_t)fd);
            pthread_detach(t);
        }
        if (r->quit) break;
        uint64_t now = (uint64_t)time(NULL);
        if (now - last_hb >= 2) {
            last_hb = now;
            node_heartbeat(&r->node);
        }
        /* 主动查询 supervisor(方式B), 每 5s 拉一次快照 */
        if (sv_host[0] && now - last_query >= 5) {
            last_query = now;
            query_servers(r, sv_host, sv_port);
        }
    }
    sock_close(srv);
    return 0;
}

int cmd_router(ServeConfig* cfg)
{
    Router r;
    memset(&r, 0, sizeof(r));
    r.port = (uint16_t)cfg->router_port;
    pthread_mutex_init(&r.lock, NULL);
    r.strategy = cfg->strategy;
    snprintf(r.node.node_id, sizeof(r.node.node_id), "%s", cfg->node_id);
    snprintf(r.node.type, sizeof(r.node.type), "router");
    r.node.state = NODE_STATE_READY;

    char sv_host[128] = "";
    uint16_t sv_port = 0;
    if (cfg->sv_host[0]) {
        snprintf(sv_host, sizeof(sv_host), "%s", cfg->sv_host);
        sv_port = (uint16_t)cfg->sv_port;
        snprintf(r.node.sv_host, sizeof(r.node.sv_host), "%s", cfg->sv_host);
        r.node.sv_port = sv_port;
        r.node.sv_enabled = 1;
    }

    if (cfg->send[0]) return run_client(&r, cfg->send);
    /* OpenAI 兼容 HTTP(可选) */
    if (cfg->http_port > 0) {
        router_http_set_api_log(cfg->api_log);
        router_http_start(&r, (uint16_t)cfg->http_port);
    }
    return router_run(&r, sv_host, sv_port);
}
