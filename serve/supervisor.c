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
#include <errno.h>
#include <time.h>
#include <signal.h>

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
/* 动态步长: 所有模型里最大的 ranks; rank-N 端口 = base + N(连续, 无魔数上限) */
static int model_stride(const Supervisor* s)
{
    int stride = 1, mi;
    for (mi = 0; mi < s->n_models && mi < CFG_MAX_MODELS; mi++)
        if (s->models[mi].ranks > stride) stride = s->models[mi].ranks;
    return stride;
}
static uint16_t model_rank_base(const Supervisor* s, int mi)
{
    return (uint16_t)(s->rank_port_base + mi * model_stride(s));
}
static uint16_t model_server_port(const Supervisor* s, int mi)
{
    return (uint16_t)(s->server_port_base + mi * model_stride(s));
}

/* 记录已拉起进程(node_id -> pid) */
static void spawn_record(Supervisor* s, const char* id, int pid)
{
    int i;
    for (i = 0; i < s->n_spawns; i++) {
        if (strcmp(s->spawn_ids[i], id) == 0) {
            s->spawn_pids[i] = pid;
            return;
        }
    }
    if (s->n_spawns < SV_MAX_NODES) {
        snprintf(s->spawn_ids[s->n_spawns], sizeof(s->spawn_ids[0]), "%s", id);
        s->spawn_pids[s->n_spawns] = pid;
        s->n_spawns++;
    }
}

/* 杀掉已拉起的进程组(sh -c + rank 都在同一会话, kill(-pid) 全杀) */
static void spawn_kill(Supervisor* s, const char* id)
{
    int i;
    for (i = 0; i < s->n_spawns; i++) {
        if (strcmp(s->spawn_ids[i], id) == 0) {
            if (s->spawn_pids[i] > 0) {
                ylog_warn("supervisor: kill stale process of %s (pid %d)", id, s->spawn_pids[i]);
#ifndef _WIN32
                kill(-s->spawn_pids[i], SIGKILL);
                kill(s->spawn_pids[i], SIGKILL);
#endif
            }
            s->spawn_pids[i] = 0;
            return;
        }
    }
}

static int supervisor_spawn_rank(Supervisor* s, int mi, int r)
{
    if (mi >= s->n_models) return -1;
    ModelCfg* mc = &s->models[mi];
    char cmd[4096];
    uint16_t rport = (uint16_t)(model_rank_base(s, mi) + r);
    int ranks = mc->ranks > 0 ? mc->ranks : 1;
    /* peers 自动生成(本机 IP × 总段数): 同机部署零配置; rank0 以 INFER 数据为准, 远程段手工起时自带 */
    char peers[512] = "";
    {
        char lip[64] = "127.0.0.1";
        sock_local_ip(lip, sizeof(lip));
        int i;
        for (i = 0; i < ranks; i++) {
            if (i) strncat(peers, ",", sizeof(peers) - strlen(peers) - 1);
            strncat(peers, lip, sizeof(peers) - strlen(peers) - 1);
        }
    }
    snprintf(cmd, sizeof(cmd),
             "\"%s\" rank --model \"%s\" --vocab \"%s\" --model-name \"%s\" --port %u "
             "--supervisor %s:%u --id rank-%d --rank %d --ranks %d --peers %s "
             "--dist-fp16 %d --budget-mb %d --budget-auto %d "
             "--cache-dir \"%s\" --log logs/%s-rank-%d.log",
             s->bin, mc->model, mc->vocab, mc->name, rport, s->sv_host, s->port,
             mi * model_stride(s) + r, r, ranks, peers, mc->dist_fp16,
             s->budget_mb, s->budget_auto,
             s->cache_dir[0] ? s->cache_dir : ".", mc->name, r);
    ylog_info("supervisor: spawn rank %d (model %s) on port %u", r, mc->name, rport);
    int pid = spawn_proc(cmd);
    if (pid > 0) {
        char id[128];
        snprintf(id, sizeof(id), "rank-%d", mi * model_stride(s) + r);
        spawn_record(s, id, pid);
    }
    return pid;
}

/* 拉起一个 server(业务组, leader 指向该模型 rank0) */
static int supervisor_spawn_server(Supervisor* s, int mi)
{
    if (mi >= s->n_models) return -1;
    ModelCfg* mc = &s->models[mi];
    char cmd[4096];
    uint16_t sport = model_server_port(s, mi);
    uint16_t lport = model_rank_base(s, mi);   /* leader = rank0 */
    snprintf(cmd, sizeof(cmd),
             "\"%s\" server --id server-%d --model-name \"%s\" --leader %s:%u "
             "--supervisor %s:%u --port %u --cache-dir \"%s\" "
             "--log logs/%s-server-%d.log",
             s->bin, mi, mc->name, s->sv_host, lport, s->sv_host, s->port,
             sport, s->cache_dir[0] ? s->cache_dir : ".", mc->name, mi);
    ylog_info("supervisor: spawn server %d (model %s) on port %u", mi, mc->name, sport);
    int pid = spawn_proc(cmd);
    if (pid > 0) {
        char id[128];
        snprintf(id, sizeof(id), "server-%d", mi);
        spawn_record(s, id, pid);
    }
    return pid;
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
            /* server 死了 → 回收其租约(rank 回池) */
            {
                int j;
                for (j = 0; j < s->n_nodes; j++) {
                    SvNode* rk = &s->nodes[j];
                    if (rk->leased_by[0] && strcmp(rk->leased_by, n->node.node_id) == 0) {
                        rk->leased_by[0] = '\0';
                        rk->lease_expire = 0;
                    }
                }
            }
            if (s->auto_heal) {
                spawn_kill(s, n->node.node_id);
                int idx = atoi(n->node.node_id + strlen("server-"));
                supervisor_spawn_server(s, idx);
            }
        } else if (strcmp(n->node.type, "rank") == 0) {
            n->node.state = NODE_STATE_DEAD;
            ylog_warn("supervisor: rank %s DEAD, respawn", n->node.node_id);
            /* rank 死了 → 释放租约, 组内其他 rank 一并回池(流水线断) */
            if (n->leased_by[0]) {
                int j;
                for (j = 0; j < s->n_nodes; j++) {
                    SvNode* rk = &s->nodes[j];
                    if (rk->leased_by[0] && strcmp(rk->leased_by, n->leased_by) == 0) {
                        rk->leased_by[0] = '\0';
                        rk->lease_expire = 0;
                    }
                }
            }
            if (s->auto_heal) {
                spawn_kill(s, n->node.node_id);
                int idx = atoi(n->node.node_id + strlen("rank-"));
                int st = model_stride(s);
                supervisor_spawn_rank(s, idx / st, idx % st);
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
            supervisor_spawn_rank(s, 0, r);
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
        sock_close(fd);
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
        char args[512];
        snprintf(args, sizeof(args), "%s type=%s model=%s leader=%s state=%s inflight=%d kv_mb=%.1f",
                 n->node_id, n->type, n->model, n->addr,
                 n->state == NODE_STATE_READY ? "ready" :
                 n->state == NODE_STATE_BUSY ? "busy" :
                 n->state == NODE_STATE_DEAD ? "dead" : "loading",
                 n->inflight, n->kv_mb);
        frame_send(fd, "SERVER_INFO", args);
    }
    /* router 节点(注册表, 无心跳, 固定 READY) */
    for (i = 0; i < s->n_routers; i++) {
        Node* rn = &s->routers[i];
        char args[256];
        snprintf(args, sizeof(args), "%s type=%s state=ready addr=%s",
                 rn->node_id, rn->type, rn->addr);
        frame_send(fd, "SERVER_INFO", args);
    }
    frame_send(fd, "QUERY_DONE", NULL);
}

/* server 主动查询: 返回该模型所有 ready rank 的真实地址(自动发现) */
static void handle_query_ranks(Supervisor* s, int fd, const char* args)
{
    const char* want = proto_get(args, "model");
    int i;
    for (i = 0; i < s->n_nodes; i++) {
        Node* n = &s->nodes[i].node;
        if (strcmp(n->type, "rank") != 0) continue;
        if (want && want[0] && n->model[0] && strcmp(n->model, want) != 0) continue;
        if (n->state != NODE_STATE_READY) continue;
        char a[512];
        snprintf(a, sizeof(a), "%s type=rank model=%s addr=%s state=ready",
                 n->node_id, n->model, n->addr);
        frame_send(fd, PROTO_RANK_INFO, a);
    }
    frame_send(fd, PROTO_QUERY_DONE, NULL);
}

/* server 租用该模型一组空闲 rank(标记忙)。KISS: 组 = 该模型全部未租 rank */
static void handle_lease(Supervisor* s, int fd, const char* args)
{
    /* <server-id> model=<name>: 第一个 token 是 server-id */
    char srv_id[128] = "";
    const char* model = proto_get(args, "model");
    {
        const char* p = args;
        while (*p == ' ') p++;
        const char* sp = p;
        while (*p && *p != ' ') p++;
        size_t sl = (size_t)(p - sp);
        if (sl >= sizeof(srv_id)) sl = sizeof(srv_id) - 1;
        memcpy(srv_id, sp, sl);
        srv_id[sl] = '\0';
    }
    if (!srv_id[0] || !model) { frame_send(fd, "ERR", "bad LEASE args"); return; }

    /* 策略: duration=<s> | permanent | 缺省 = request(请求级) */
    uint64_t expire = 0;
    const char* dur = proto_get(args, "duration");
    if (dur && strcmp(dur, "permanent") == 0) {
        expire = 0; /* permanent: 0 且 leased_by 非空即永久 */
    } else if (dur && atoi(dur) > 0) {
        expire = (uint64_t)time(NULL) + (uint64_t)atoi(dur);
    } else {
        /* request: 由 server 推理后 RELEASE, 这里照常租出 */
    }

    /* 找该模型全部 ready + 未租的 rank; leader = 端口最小(段0 = rank0) */
    int i, n = 0;
    const char* leader = NULL;
    int leader_port = 0;
    for (i = 0; i < s->n_nodes; i++) {
        SvNode* sv = &s->nodes[i];
        Node* nd = &sv->node;
        if (strcmp(nd->type, "rank") != 0) continue;
        if (nd->state != NODE_STATE_READY) continue;
        if (model[0] && nd->model[0] && strcmp(nd->model, model) != 0) continue;
        if (sv->leased_by[0] && strcmp(sv->leased_by, srv_id) != 0) continue; /* 别人租的跳过 */
        n++;
        int p = 0;
        const char* colon = strchr(nd->addr, ':');
        if (colon) p = atoi(colon + 1);
        if (!leader || p < leader_port) { leader = nd->addr; leader_port = p; }
    }
    if (n == 0 || !leader) { frame_send(fd, "ERR", "no-rank"); return; }
    /* 收集组内成员 IP(端口升序 = 段号序; 暂存 ip:port, 排序后去端口) */
    char peers[1024] = "";
    char rank_ids[512] = "";
    {
        char ps[64][128];
        int pn = 0, j;
        for (j = 0; j < s->n_nodes; j++) {
            SvNode* sv = &s->nodes[j];
            Node* nd = &sv->node;
            if (strcmp(nd->type, "rank") != 0) continue;
            if (nd->state != NODE_STATE_READY) continue;
            if (model[0] && nd->model[0] && strcmp(nd->model, model) != 0) continue;
            if (sv->leased_by[0] && strcmp(sv->leased_by, srv_id) != 0) continue;
            if (pn < 64) {
                snprintf(ps[pn], sizeof(ps[pn]), "%s", nd->addr);
                pn++;
                if (rank_ids[0]) strncat(rank_ids, ",", sizeof(rank_ids) - strlen(rank_ids) - 1);
                strncat(rank_ids, nd->node_id, sizeof(rank_ids) - strlen(rank_ids) - 1);
            }
        }
        int a, b;
        for (a = 0; a < pn; a++)
            for (b = a + 1; b < pn; b++) {
                int pa = atoi(strrchr(ps[a], ':') + 1);
                int pb = atoi(strrchr(ps[b], ':') + 1);
                if (pb < pa) {
                    char t[128]; snprintf(t, sizeof(t), "%s", ps[a]);
                    snprintf(ps[a], sizeof(ps[a]), "%s", ps[b]);
                    snprintf(ps[b], sizeof(ps[b]), "%s", t);
                }
            }
        for (a = 0; a < pn; a++) {
            char* colon = strchr(ps[a], ':');
            if (colon) *colon = '\0';
            if (a) strncat(peers, ",", sizeof(peers) - strlen(peers) - 1);
            strncat(peers, ps[a], sizeof(peers) - strlen(peers) - 1);
        }
    }
    for (i = 0; i < s->n_nodes; i++) {
        SvNode* sv = &s->nodes[i];
        Node* nd = &sv->node;
        if (strcmp(nd->type, "rank") != 0) continue;
        if (nd->state != NODE_STATE_READY) continue;
        if (model[0] && nd->model[0] && strcmp(nd->model, model) != 0) continue;
        if (sv->leased_by[0] && strcmp(sv->leased_by, srv_id) != 0) continue;
        snprintf(sv->leased_by, sizeof(sv->leased_by), "%s", srv_id);
        sv->lease_expire = expire;
    }
    ylog_info("supervisor: LEASE %s model=%s ranks=%d leader=%s peers=%s",
              srv_id, model, n, leader, peers);
    char a[640];
    snprintf(a, sizeof(a), "LEASED ranks=%d leader=%s peers=%s rank_ids=%s", n, leader, peers, rank_ids);
    frame_send(fd, "OK", a);
    ylog_info("supervisor: LEASE reply sent to %s (n=%d)", srv_id, n);
}

/* server 释放租用: 该 server 的 rank 全部回池 */
static void handle_release(Supervisor* s, int fd, const char* args)
{
    char srv_id[128] = "";
    const char* p = args;
    while (*p == ' ') p++;
    const char* sp = p;
    while (*p && *p != ' ') p++;
    size_t sl = (size_t)(p - sp);
    if (sl >= sizeof(srv_id)) sl = sizeof(srv_id) - 1;
    memcpy(srv_id, sp, sl);
    srv_id[sl] = '\0';
    int i, n = 0;
    for (i = 0; i < s->n_nodes; i++) {
        SvNode* sv = &s->nodes[i];
        if (sv->leased_by[0] && strcmp(sv->leased_by, srv_id) == 0) {
            sv->leased_by[0] = '\0';
            sv->lease_expire = 0;
            n++;
        }
    }
    ylog_info("supervisor: RELEASE %s (%d rank freed)", srv_id, n);
    frame_send(fd, "OK", NULL);
}

/* timed 租约到期 → 自动回池 */
static void lease_expire_check(Supervisor* s)
{
    uint64_t now = (uint64_t)time(NULL);
    int i;
    for (i = 0; i < s->n_nodes; i++) {
        SvNode* sv = &s->nodes[i];
        if (sv->lease_expire > 0 && sv->lease_expire <= now) {
            ylog_info("supervisor: lease of %s (rank %s) expired, freed",
                      sv->leased_by, sv->node.node_id);
            sv->leased_by[0] = '\0';
            sv->lease_expire = 0;
        }
    }
}

static void handle_frame(Supervisor* s, int fd, const char* cmd, const char* args)
{
    if (strcmp(cmd, "HEARTBEAT") == 0) handle_heartbeat(s, fd, args);
    else if (strcmp(cmd, "QUERY_SERVERS") == 0) handle_query_servers(s, fd);
    else if (strcmp(cmd, PROTO_QUERY_RANKS) == 0) handle_query_ranks(s, fd, args);
    else if (strcmp(cmd, PROTO_LEASE) == 0) handle_lease(s, fd, args);
    else if (strcmp(cmd, PROTO_RELEASE) == 0) handle_release(s, fd, args);
    else if (strcmp(cmd, PROTO_SCALE) == 0) { handle_scale(s, args); frame_send(fd, "OK", NULL); }
    else if (strcmp(cmd, PROTO_QUIT) == 0 || strcmp(cmd, PROTO_DRAIN) == 0) {
        frame_send(fd, "OK", NULL);
        s->quit = 1;
    }
    else frame_send(fd, "ERR", "unknown cmd");
}

/* 启动引导: 按配置自动拉起 ranks 个 rank + 1 个 server(无需手动启动) */
static void supervisor_bootstrap(Supervisor* s)
{
    if (!s->model[0] || !s->bin[0]) {
        ylog_info("supervisor: no model/bin configured, skip bootstrap (manual spawn only)");
        return;
    }
    int mi, r;
    for (mi = 0; mi < s->n_models; mi++) {
        ModelCfg* mc = &s->models[mi];
        int ranks = mc->ranks;
        if (ranks <= 0) {
            /* ranks=0: 该模型 rank 全部外部部署(分布式), 不 spawn */
            ylog_info("supervisor: model %s ranks=0, skip spawn (external rank)", mc->name);
            continue;
        }
        /* local: 本机拉起的段数(-1 未指定 = 全部; 0 = 全部外部; N = 前 N 段本地) */
        int local = mc->local < 0 ? ranks : (mc->local < ranks ? mc->local : ranks);
        for (r = 0; r < local; r++)
            supervisor_spawn_rank(s, mi, r);
        if (local < ranks)
            ylog_info("supervisor: model %s ranks=%d local=%d (段 %d..%d 外部部署)",
                      mc->name, ranks, local, local, ranks - 1);
        if (!s->no_spawn_server)
            supervisor_spawn_server(s, mi);
    }
    ylog_info("supervisor: bootstrap done (%d model(s)%s)", s->n_models,
              s->no_spawn_server ? ", server in-process" : "");
}

int supervisor_run(Supervisor* s)
{
    sock_init();
    int srv = sock_listen(s->port, 16);
    if (srv < 0) {
#ifdef _WIN32
        ylog_error("supervisor: sock_listen fail port=%u WSAErr=%d", s->port, WSAGetLastError());
#else
        ylog_error("supervisor: sock_listen fail port=%u errno=%d", s->port, errno);
#endif
        return 1;
    }
    ylog_info("supervisor: listening on port %u", s->port);

    supervisor_bootstrap(s);

    uint64_t last_check = 0;
    for (;;) {
        int fd = sock_accept_with_timeout(srv, 500);
        if (fd >= 0) {
            Frame f;
            if (frame_recv(fd, &f) >= 0)
                handle_frame(s, fd, f.cmd, f.args);
            sock_close(fd);
        }
        /* 判死: server 超时 → DEL 通知 router(P3 补重拉) */
        uint64_t now = (uint64_t)time(NULL);
        if (now - last_check >= 2) {
            last_check = now;
            heal_dead(s);
            lease_expire_check(s);
        }
        if (s->quit) break;
    }
    sock_close(srv);
    return 0;
}

int cmd_supervisor(ServeConfig* cfg)
{
    /* --only-model / --model <名字> 按模型名筛选(多模型 serve.yaml 只拉起指定模型) */
    if (cfg->only_model[0]) {
        config_filter_models(cfg, cfg->only_model);
    } else if (cfg->model[0] && cfg->n_models > 1) {
        int mi, hit = 0;
        for (mi = 0; mi < cfg->n_models; mi++)
            if (strcmp(cfg->models[mi].name, cfg->model) == 0) { hit = 1; break; }
        if (hit) config_filter_models(cfg, cfg->model);
    }

    Supervisor s;
    memset(&s, 0, sizeof(s));
    s.port = (uint16_t)cfg->sv_port;
    s.ranks = cfg->ranks > 0 ? cfg->ranks : 1;
    s.rank_port_base = (uint16_t)cfg->rank_port_base;
    s.server_port_base = (uint16_t)cfg->server_port;
    s.auto_heal = cfg->auto_heal;
    s.budget_mb = cfg->budget_mb;
    s.budget_auto = cfg->budget_auto;
    snprintf(s.sv_host, sizeof(s.sv_host), "%s", cfg->sv_host);
    snprintf(s.bin, sizeof(s.bin), "%s", cfg->bin);
    if (cfg->model[0]) snprintf(s.model, sizeof(s.model), "%s", cfg->model);
    if (cfg->vocab[0]) snprintf(s.vocab, sizeof(s.vocab), "%s", cfg->vocab);
    if (cfg->model_name[0]) snprintf(s.model_name, sizeof(s.model_name), "%s", cfg->model_name);
    s.n_models = cfg->n_models;
    {
        int mi;
        for (mi = 0; mi < cfg->n_models && mi < CFG_MAX_MODELS; mi++) s.models[mi] = cfg->models[mi];
    }

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
