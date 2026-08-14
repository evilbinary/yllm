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

int server_run(Server* s)
{
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
