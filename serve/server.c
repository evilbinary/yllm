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
#include "../inference/log.h"
#include <time.h>

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
    Frame f;
    snprintf(f.cmd, sizeof(f.cmd), "%s", PROTO_INFER);
    snprintf(f.args, sizeof(f.args), "%s", args);
    frame_send(fd, f.cmd, f.args);

    /* 读 prompt 长度, 透传 prompt bytes */
    long nbytes = frame_payload_len(&f);
    if (nbytes > 0) {
        char* pb = (char*)malloc((size_t)nbytes);
        if (!pb) { close(fd); sock_send_line(client_fd, "ERR server: oom"); return; }
        if (sock_recv_n(client_fd, pb, (size_t)nbytes) != 0) { free(pb); close(fd); return; }
        sock_send_n(fd, pb, (size_t)nbytes);
        free(pb);
    }

    /* 透传响应(T 帧 + DONE) */
    char out[SRV_MAX_LINE];
    int done = 0;
    while (!done) {
        int n = sock_recv_line(fd, out, sizeof(out));
        if (n < 0) { sock_send_line(client_fd, "ERR server: leader disconnected"); break; }
        sock_send_line(client_fd, "%s", out);
        if (strncmp(out, PROTO_DONE, 4) == 0) done = 1;
        if (strncmp(out, "T ", 2) == 0) {
            long tlen = atol(out + 2);
            if (tlen > 0 && tlen < (long)sizeof(out)) {
                if (sock_recv_n(fd, out, (size_t)tlen) != 0) break;
                sock_send_n(client_fd, out, (size_t)tlen);
            }
        }
    }
    close(fd);
}

static void handle_frame(int fd, Server* s, const char* cmd, const char* args)
{
    if (strcmp(cmd, PROTO_INFER) == 0) {
        s->node.inflight++;
        forward_infer(fd, s, args);
        s->node.inflight--;
    } else if (strcmp(cmd, PROTO_DRAIN) == 0) {
        frame_send(fd, "OK", NULL);
        /* P2: 立即退出 */
    } else {
        frame_send(fd, "ERR", "unknown cmd");
    }
}

int server_run(Server* s)
{
    sock_init();

    int srv = sock_listen(s->port, 8);
    if (srv < 0) return 1;
    ylog_info("server %s: listening on port %u (leader=%s:%u, heartbeat → %s:%u)",
              s->node.node_id, s->port, s->leader_host, s->leader_port,
              s->node.sv_host, s->node.sv_port);

    uint64_t last_hb = 0;
    for (;;) {
        int fd = sock_accept_with_timeout(srv, 500);
        if (fd >= 0) {
            Frame f;
            if (frame_recv(fd, &f) >= 0)
                handle_frame(fd, s, f.cmd, f.args);
            close(fd);
        }
        uint64_t now = (uint64_t)time(NULL);
        if (now - last_hb >= 2) {
            last_hb = now;
            s->node.kv_mb = 0; /* P2: 无组内汇总, 占位 */
            node_heartbeat(&s->node);
        }
    }
    close(srv);
    return 0;
}

int cmd_server(ServeConfig* cfg)
{
    if (!cfg->model_name[0] || !cfg->leader[0]) {
        fprintf(stderr, "usage: yllm server --id <name> --model-name <name> --leader <ip:port> "
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

    /* 解析 leader ip:port */
    const char* colon = strchr(cfg->leader, ':');
    if (!colon) { fprintf(stderr, "bad leader addr: %s\n", cfg->leader); return 1; }
    size_t hlen = (size_t)(colon - cfg->leader);
    if (hlen >= sizeof(s.leader_host)) hlen = sizeof(s.leader_host) - 1;
    memcpy(s.leader_host, cfg->leader, hlen);
    s.leader_host[hlen] = '\0';
    s.leader_port = (uint16_t)atoi(colon + 1);

    /* 自身 addr(上报 supervisor 用): ip:port */
    snprintf(s.node.addr, sizeof(s.node.addr), "%s:%u", cfg->sv_host, s.port);

    /* supervisor 地址 */
    snprintf(s.node.sv_host, sizeof(s.node.sv_host), "%s", cfg->sv_host);
    s.node.sv_port = (uint16_t)cfg->sv_port;
    s.node.sv_enabled = 1;

    return server_run(&s);
}
