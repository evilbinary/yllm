/* supervisor.c — yllm supervisor: 管理节点(生命周期层, 最小骨架)
 *
 * P2 骨架职责:
 *   ① 收所有节点(rank/server/router)的 HEARTBEAT, 维护统一汇总表;
 *   ② 驱动 router: server 注册(ADD)/注销(DEL)/状态推送(UPDATE);
 *   ③ 支持 router 主动 QUERY_SERVERS;
 *   ④ 判死: 心跳超时 → 标记 DEAD → 通知 router DEL(P3 补重拉)。
 *
 * 用法:
 *   yllm supervisor --port <N> [--router <ip:port[,ip:port...]>]
 *                   [--log <file>] [--log-level lvl] [--no-console]
 */
#include "protocol.h"
#include "frame.h"
#include "node.h"
#include "sock.h"
#include "supervisor.h"
#include "../inference/log.h"
#include <time.h>

static SvNode* find_node(Supervisor* s, const char* id)
{
    int i;
    for (i = 0; i < s->n_nodes; i++)
        if (strcmp(s->nodes[i].node.node_id, id) == 0) return &s->nodes[i];
    return NULL;
}

/* 向所有 router 发一帧(router 地址从统一 Node.addr 解析) */
static void notify_routers(Supervisor* s, const char* cmd, const char* args)
{
    int i;
    for (i = 0; i < s->n_routers; i++) {
        Node* rn = &s->routers[i];
        const char* colon = strchr(rn->addr, ':');
        if (!colon) continue;
        int fd = sock_connect_host(rn->addr, (size_t)(colon - rn->addr), (uint16_t)atoi(colon + 1), 3);
        if (fd < 0) continue;
        frame_send(fd, cmd, args);
        close(fd);
    }
}

/* server 状态变化时通知 router */
static void sync_server_to_router(Supervisor* s, SvNode* n)
{
    Node* nd = &n->node;
    char args[512];
    if (nd->state == NODE_STATE_READY && !n->router_notified) {
        snprintf(args, sizeof(args), "%s model=%s leader=%s", nd->node_id, nd->model, nd->addr);
        notify_routers(s, PROTO_SERVER_ADD, args);
        n->router_notified = 1;
        ylog_info("supervisor: SERVER_ADD %s model=%s leader=%s", nd->node_id, nd->model, nd->addr);
    } else if (nd->state == NODE_STATE_DEAD && n->router_notified) {
        notify_routers(s, PROTO_SERVER_DEL, nd->node_id);
        n->router_notified = 0;
        ylog_warn("supervisor: SERVER_DEL %s (dead)", nd->node_id);
    } else if (n->router_notified) {
        snprintf(args, sizeof(args), "%s inflight=%d kv_mb=%.1f", nd->node_id, nd->inflight, nd->kv_mb);
        notify_routers(s, PROTO_SERVER_UPDATE, args);
    }
}

static void handle_heartbeat(Supervisor* s, int fd, const char* args)
{
    Node n;
    memset(&n, 0, sizeof(n));
    node_parse_hb(args, &n);

    SvNode* sv = find_node(s, n.node_id);
    if (!sv) {
        if (s->n_nodes >= SV_MAX_NODES) { frame_send(fd, "ERR", "too many nodes"); return; }
        sv = &s->nodes[s->n_nodes++];
        memset(sv, 0, sizeof(*sv));
        sv->node = n;
        ylog_info("supervisor: node %s type=%s joined", n.node_id, n.type);
    } else {
        sv->node.state = n.state;
        sv->node.inflight = n.inflight;
        sv->node.kv_mb = n.kv_mb;
        if (n.model[0]) snprintf(sv->node.model, sizeof(sv->node.model), "%s", n.model);
        if (n.addr[0])  snprintf(sv->node.addr, sizeof(sv->node.addr), "%s", n.addr);
        sv->node.last_hb = n.last_hb;
    }
    /* 只把 server 同步给 router; rank/router 只记录 */
    if (strcmp(sv->node.type, "server") == 0)
        sync_server_to_router(s, sv);
    frame_send(fd, "OK", NULL);
}

/* router 主动查询: 返回所有 server 快照 */
static void handle_query_servers(Supervisor* s, int fd)
{
    int i;
    for (i = 0; i < s->n_nodes; i++) {
        Node* n = &s->nodes[i].node;
        if (strcmp(n->type, "server") != 0) continue;
        char args[512];
        snprintf(args, sizeof(args), "%s model=%s leader=%s state=%s inflight=%d kv_mb=%.1f",
                 n->node_id, n->model, n->addr,
                 n->state == NODE_STATE_READY ? "ready" :
                 n->state == NODE_STATE_DEAD ? "dead" : "loading",
                 n->inflight, n->kv_mb);
        frame_send(fd, "SERVER_INFO", args);
    }
    frame_send(fd, "QUERY_DONE", NULL);
}

static void handle_frame(Supervisor* s, int fd, const char* cmd, const char* args)
{
    if (strcmp(cmd, "HEARTBEAT") == 0) handle_heartbeat(s, fd, args);
    else if (strcmp(cmd, "QUERY_SERVERS") == 0) handle_query_servers(s, fd);
    else frame_send(fd, "ERR", "unknown cmd");
}

int supervisor_run(Supervisor* s)
{
    sock_init();
    int srv = sock_listen(s->port, 16);
    if (srv < 0) return 1;
    ylog_info("supervisor: listening on port %u", s->port);

    uint64_t last_check = 0;
    for (;;) {
        int fd = sock_accept_with_timeout(srv, 500);
        if (fd >= 0) {
            Frame f;
            if (frame_recv(fd, &f) >= 0)
                handle_frame(s, fd, f.cmd, f.args);
            close(fd);
        }
        /* 判死: server 超时 → DEL 通知 router(P3 补重拉) */
        uint64_t now = (uint64_t)time(NULL);
        if (now - last_check >= 2) {
            last_check = now;
            int i;
            for (i = 0; i < s->n_nodes; i++) {
                SvNode* n = &s->nodes[i];
                if (n->node.state != NODE_STATE_DEAD &&
                    strcmp(n->node.type, "server") == 0 &&
                    node_is_dead(&n->node, SV_HB_TIMEOUT)) {
                    n->node.state = NODE_STATE_DEAD;
                    ylog_warn("supervisor: server %s DEAD", n->node.node_id);
                    sync_server_to_router(s, n);
                }
            }
        }
    }
    close(srv);
    return 0;
}

typedef struct { const char* key; const char* val; } ArgSV;

static const char* opt_sv(ArgSV* args, int n, const char* key, const char* def)
{
    int i;
    for (i = 0; i < n; i++)
        if (strcmp(args[i].key, key) == 0) return args[i].val;
    return def;
}

int cmd_supervisor(int argc, char** argv)
{
    ArgSV a[24];
    int n = 0;
    int i;
    for (i = 2; i + 1 < argc && n < 24; i += 2) {
        if (argv[i][0] != '-') break;
        a[n].key = argv[i];
        while (*a[n].key == '-') a[n].key++;
        a[n].val = argv[i + 1];
        n++;
    }
    const char* routers = opt_sv(a, n, "router", NULL);
    int port = atoi(opt_sv(a, n, "port", "9500"));

    Supervisor s;
    memset(&s, 0, sizeof(s));
    s.port = (uint16_t)port;

    if (routers) {
        char tmp[2048];
        snprintf(tmp, sizeof(tmp), "%s", routers);
        char* tok = strtok(tmp, ",");
        while (tok != NULL && s.n_routers < SV_MAX_ROUTERS) {
            Node* rn = &s.routers[s.n_routers++];
            memset(rn, 0, sizeof(*rn));
            snprintf(rn->node_id, sizeof(rn->node_id), "router-%d", s.n_routers - 1);
            snprintf(rn->type, sizeof(rn->type), "router");
            snprintf(rn->addr, sizeof(rn->addr), "%s", tok);
            tok = strtok(NULL, ",");
        }
    }

    return supervisor_run(&s);
}
