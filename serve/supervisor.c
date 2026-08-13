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

static void sync_server_to_router(Supervisor* s, SvNode* n);   /* 前向声明 */

/* ---- 拉起进程(生命周期面, 唯一 spawn 的地方) ---- */

static int spawn_proc(const char* cmdline)
{
#ifdef _WIN32
    STARTUPINFOA si; PROCESS_INFORMATION pi;
    memset(&si, 0, sizeof(si)); memset(&pi, 0, sizeof(pi));
    si.cb = sizeof(si);
    if (!CreateProcessA(NULL, (char*)cmdline, NULL, NULL, FALSE, CREATE_NO_WINDOW,
                        NULL, NULL, &si, &pi)) {
        ylog_error("supervisor: CreateProcess fail: %s", cmdline);
        return -1;
    }
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return (int)pi.dwProcessId;
#else
    pid_t pid = fork();
    if (pid < 0) return -1;
    if (pid == 0) {
        setsid();
        char* cmd = (char*)malloc(strlen(cmdline) + 1);
        if (!cmd) _exit(127);
        strcpy(cmd, cmdline);
        execl("/bin/sh", "sh", "-c", cmd, (char*)NULL);
        _exit(127);
    }
    return (int)pid;
#endif
}

/* 拉起一个 rank(段 r, 模型 m) */
static int supervisor_spawn_rank(Supervisor* s, int r)
{
    char cmd[4096];
    uint16_t rport = (uint16_t)(s->rank_port_base + r);
    snprintf(cmd, sizeof(cmd),
             "\"%s\" rank --model \"%s\" --vocab \"%s\" --port %u "
             "--supervisor %s:%u --id rank-%d --log logs/rank-%d.log",
             s->bin, s->model, s->vocab, rport, s->sv_host, s->port, r, r);
    ylog_info("supervisor: spawn rank %d on port %u", r, rport);
    return spawn_proc(cmd);
}

/* 拉起一个 server(业务组, leader 指向 rank 组) */
static int supervisor_spawn_server(Supervisor* s, int idx, const char* model_name)
{
    char cmd[4096];
    uint16_t sport = (uint16_t)(s->server_port_base + idx);
    uint16_t lport = (uint16_t)(s->rank_port_base);   /* leader = rank0 */
    snprintf(cmd, sizeof(cmd),
             "\"%s\" server --id server-%d --model-name \"%s\" --leader %s:%u "
             "--supervisor %s:%u --port %u --log logs/server-%d.log",
             s->bin, idx, model_name, s->sv_host, lport, s->sv_host, s->port,
             sport, idx);
    ylog_info("supervisor: spawn server %d on port %u", idx, sport);
    return spawn_proc(cmd);
}

/* 自愈: 心跳超时的 server/rank 自动重拉 */
static void heal_dead(Supervisor* s)
{
    int i;
    for (i = 0; i < s->n_nodes; i++) {
        SvNode* n = &s->nodes[i];
        if (n->node.state == NODE_STATE_DEAD) continue;
        if (!node_is_dead(&n->node, SV_HB_TIMEOUT)) continue;
        if (strcmp(n->node.type, "server") == 0) {
            n->node.state = NODE_STATE_DEAD;
            ylog_warn("supervisor: server %s DEAD, respawn", n->node.node_id);
            sync_server_to_router(s, n);
            if (s->auto_heal) {
                int idx = atoi(n->node.node_id + strlen("server-"));
                supervisor_spawn_server(s, idx, n->node.model[0] ? n->node.model : "default");
            }
        } else if (strcmp(n->node.type, "rank") == 0) {
            n->node.state = NODE_STATE_DEAD;
            ylog_warn("supervisor: rank %s DEAD, respawn", n->node.node_id);
            if (s->auto_heal) {
                int r = atoi(n->node.node_id + strlen("rank-"));
                supervisor_spawn_rank(s, r);
            }
        }
    }
}

/* SCALE: server 请求扩容(拉起新的 rank 段组) */
static void handle_scale(Supervisor* s, const char* args)
{
    char id[128];
    int need = 0;
    if (sscanf(args, "%127s need_groups=%d", id, &need) != 2) return;
    ylog_info("supervisor: SCALE %s need_groups=%d", id, need);
    if (s->auto_heal) {
        /* P3 简化: 拉起一组 rank(rank0..ranks-1) */
        int r;
        for (r = 0; r < s->ranks; r++)
            supervisor_spawn_rank(s, r);
    }
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
    else if (strcmp(cmd, PROTO_SCALE) == 0) { handle_scale(s, args); frame_send(fd, "OK", NULL); }
    else frame_send(fd, "ERR", "unknown cmd");
}

/* 启动引导: 按配置自动拉起 ranks 个 rank + 1 个 server(无需手动启动) */
static void supervisor_bootstrap(Supervisor* s)
{
    if (!s->model[0] || !s->bin[0]) {
        ylog_info("supervisor: no model/bin configured, skip bootstrap (manual spawn only)");
        return;
    }
    int r;
    for (r = 0; r < s->ranks; r++)
        supervisor_spawn_rank(s, r);
    supervisor_spawn_server(s, 0, s->model_name[0] ? s->model_name : "default");
    ylog_info("supervisor: bootstrap done (%d rank(s) + 1 server)", s->ranks);
}

int supervisor_run(Supervisor* s)
{
    sock_init();
    int srv = sock_listen(s->port, 16);
    if (srv < 0) return 1;
    ylog_info("supervisor: listening on port %u", s->port);

    supervisor_bootstrap(s);

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
            heal_dead(s);
        }
    }
    close(srv);
    return 0;
}

int cmd_supervisor(ServeConfig* cfg)
{
    Supervisor s;
    memset(&s, 0, sizeof(s));
    s.port = (uint16_t)cfg->sv_port;
    s.ranks = cfg->ranks > 0 ? cfg->ranks : 1;
    s.rank_port_base = (uint16_t)cfg->rank_port_base;
    s.server_port_base = (uint16_t)cfg->server_port;
    s.auto_heal = cfg->auto_heal;
    snprintf(s.sv_host, sizeof(s.sv_host), "%s", cfg->sv_host);
    snprintf(s.bin, sizeof(s.bin), "%s", cfg->bin);
    if (cfg->model[0]) snprintf(s.model, sizeof(s.model), "%s", cfg->model);
    if (cfg->vocab[0]) snprintf(s.vocab, sizeof(s.vocab), "%s", cfg->vocab);
    if (cfg->model_name[0]) snprintf(s.model_name, sizeof(s.model_name), "%s", cfg->model_name);

    if (cfg->router_addrs[0]) {
        char tmp[2048];
        snprintf(tmp, sizeof(tmp), "%s", cfg->router_addrs);
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
